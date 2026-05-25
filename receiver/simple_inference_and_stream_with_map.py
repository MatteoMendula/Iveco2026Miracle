#!/usr/bin/env python3
"""
Folder Inference Streamer  (with optional PLY point-cloud side panel)
=====================================================================
Reads images from a folder, runs TRT depth + segmentation on each, composes
a visualisation and pushes it as an RTSP stream to MediaMTX.

Layout — WITHOUT --ply-file  (original behaviour)
--------------------------------------------------
  [ DEPTH  |  SEGMENTATION ]
               [ original ]   ← optional, bottom-right PiP

Layout — WITH --ply-file
-------------------------
  ┌─────────────────┬─────────────────────────┐
  │   DEPTH (top)   │                         │
  │─────────────────│   PLY point cloud       │
  │  SEGMENTATION   │   (auto-rotating on Y)  │
  │    (bottom)     │                         │
  │  [original PiP] │                         │
  └─────────────────┴─────────────────────────┘

  Left half  = depth stacked above segmentation  (+ optional original PiP)
  Right half = Playwright/Three.js WebGL render of the PLY, spinning on Y-axis

Usage
-----
  # without PLY (original mode):
  python folder_inference_streamer.py --input-dir ./frames

  # with PLY side-panel:
  python folder_inference_streamer.py --input-dir ./frames \\
        --ply-file ./cloud.ply [--show-original] [--loop]

MediaMTX (run once before this script):
  docker run --rm -it --network=host bluenviron/mediamtx:latest

Consume with:
  ffplay rtsp://localhost:8554/aurelius
  vlc   rtsp://localhost:8554/aurelius
"""

import argparse
import queue
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import tensorrt as trt

from datasets_multitask import UNIFIED_CLASSES, UNIFIED_NAME_TO_ID

# ── Configuration ─────────────────────────────────────────────────────────────

DEPTH_TRT = "./models/depth_teacher.trt"
SEG_TRT   = "./models/seg_teacher.trt"

IMG_SIZE  = 512
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RTSP_FPS       = 15
STREAM_W       = 1280   # wider to accommodate the 3-D panel
STREAM_H       = 720
ORIGINAL_SCALE = 0.25

# Y-axis rotation speed for the point cloud (degrees per frame)
PLY_ROT_DEG_PER_FRAME = 0.8

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

CUSTOM_COLOR_MAP_RGB = {
    "background":       (0,   0,   0),
    "car":              (0,   255, 0),
    "truck":            (50,  205, 50),
    "bus":              (34,  139, 34),
    "van":              (0,   255, 127),
    "bridge":           (139, 69,  19),
    "swimming-pool":    (30,  144, 255),
    "road":             (255, 255, 150),
    "roof":             (0,   0,   255),
    "roof-transparent": (0,   100, 255),
    "field-green":      (112, 194, 33),
    "field-wild":       (34,  139, 34),
    "building":         (0,   165, 255),
    "person":           (0,   0,   255),
}


# ── Palette ───────────────────────────────────────────────────────────────────

def _build_seg_palette() -> np.ndarray:
    num_classes = len(UNIFIED_CLASSES)
    rng = np.random.default_rng(1234)
    colors = rng.integers(0, 255, size=(num_classes, 3), dtype=np.uint8)
    for name, color in CUSTOM_COLOR_MAP_RGB.items():
        if name in UNIFIED_NAME_TO_ID:
            colors[UNIFIED_NAME_TO_ID[name]] = np.array(color, dtype=np.uint8)
    return colors

SEG_COLORS = _build_seg_palette()


# ── Placeholder frame ─────────────────────────────────────────────────────────

def make_placeholder(width: int = STREAM_W, height: int = STREAM_H,
                     message: str = "Waiting...") -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (18, 18, 18)
    cv2.line(frame, (40, height // 2 - 36), (width - 40, height // 2 - 36), (60, 60, 60), 1)
    lines = [
        ("AURELIUS", 1.4, (200, 200, 200), 2),
        (message,    0.65, (120, 120, 120), 1),
    ]
    for i, (text, scale, color, thickness) in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        x = (width  - tw) // 2
        y = height // 2 + i * int(th * 2.6)
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
    cv2.line(frame, (40, height // 2 + 52), (width - 40, height // 2 + 52), (60, 60, 60), 1)
    return frame

PLACEHOLDER_WAITING  = make_placeholder(message="Loading images...")
PLACEHOLDER_DONE     = make_placeholder(message="All images processed. Stream ended.")


# ── TRT Wrapper ───────────────────────────────────────────────────────────────

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTModel:
    def __init__(self, engine_path: str):
        with open(engine_path, "rb") as f:
            engine_bytes = f.read()
        runtime = trt.Runtime(TRT_LOGGER)
        self.engine  = runtime.deserialize_cuda_engine(engine_bytes)
        self.context = self.engine.create_execution_context()
        self.input_name  = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        out_shape = tuple(self.engine.get_tensor_shape(self.output_name))
        self._out_shape = out_shape
        print(f"   TRT '{Path(engine_path).name}' — out_shape={out_shape}")

    def __call__(self, x: torch.Tensor):
        x = x.contiguous()
        output = torch.empty(self._out_shape, dtype=torch.float32, device=DEVICE)
        stream = torch.cuda.current_stream().cuda_stream
        self.context.set_tensor_address(self.input_name,  x.data_ptr())
        self.context.set_tensor_address(self.output_name, output.data_ptr())
        self.context.execute_async_v3(stream_handle=stream)
        return output


# ── Inference helpers ─────────────────────────────────────────────────────────

def _preprocess(frame_bgr: np.ndarray):
    orig_h, orig_w = frame_bgr.shape[:2]
    scale   = min(IMG_SIZE / orig_w, IMG_SIZE / orig_h)
    new_w   = int(orig_w * scale)
    new_h   = int(orig_h * scale)
    resized = cv2.resize(frame_bgr, (new_w, new_h))
    padded  = np.full((IMG_SIZE, IMG_SIZE, 3), 128, dtype=np.uint8)
    dw, dh  = (IMG_SIZE - new_w) // 2, (IMG_SIZE - new_h) // 2
    padded[dh:dh + new_h, dw:dw + new_w] = resized
    rgb     = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    x       = torch.from_numpy(rgb.astype(np.float32) / 255.0)
    x       = x.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    x       = x.to(memory_format=torch.channels_last)
    x_norm  = (x - 0.5) / 0.5
    return x_norm, dw, dh, new_w, new_h


def _depth_to_rgb(depth_tensor: torch.Tensor) -> np.ndarray:
    d = depth_tensor.detach().cpu().float()
    d_min, d_max = float(d.min()), float(d.max())
    if d_max <= d_min + 1e-6:
        gray = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    else:
        gray = ((d - d_min) / (d_max - d_min + 1e-6) * 255).numpy().astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_MAGMA)


def _seg_to_color(seg_tensor: torch.Tensor) -> np.ndarray:
    seg_np = seg_tensor.detach().cpu().numpy().astype(np.int64)
    seg_np = np.clip(seg_np, 0, len(UNIFIED_CLASSES) - 1)
    rgb    = SEG_COLORS[seg_np]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def run_inference(frame_bgr, depth_model, seg_model):
    """Returns (depth_bgr, seg_bgr, original_bgr) all at the input frame resolution."""
    orig_h, orig_w = frame_bgr.shape[:2]
    x_norm, dw, dh, new_w, new_h = _preprocess(frame_bgr)

    with torch.no_grad():
        depth_out = depth_model(x_norm)
        seg_out   = seg_model(x_norm)
    torch.cuda.synchronize()

    depth_map        = (depth_out[0, 0] if depth_out.ndim == 4 else depth_out[0]) * 255.0
    depth_bgr_padded = _depth_to_rgb(depth_map)
    seg_mask         = seg_out.argmax(dim=1)[0]
    seg_bgr_padded   = _seg_to_color(seg_mask)

    def _unpad_resize(img):
        cropped = img[dh:dh + new_h, dw:dw + new_w]
        return cv2.resize(cropped, (orig_w, orig_h))

    return _unpad_resize(depth_bgr_padded), _unpad_resize(seg_bgr_padded), frame_bgr.copy()


# ── PLY Point-Cloud Renderer (Playwright + Three.js headless) ────────────────

# Inline HTML page fed to the headless browser.
# The PLY file is injected as a base64 data-URL so no HTTP server is needed.
# Auto-rotation runs inside the page; Python only calls screenshot() each frame.
_PLY_VIEWER_HTML = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  * {{ margin: 0; padding: 0; }}
  body {{ background: #111317; overflow: hidden; width: {W}px; height: {H}px; }}
  canvas {{ display: block; }}
  #label {{
    position: absolute; top: 10px; left: 10px;
    font: 700 14px/1 'Courier New', monospace;
    color: #fff; letter-spacing: 2px; opacity: 0.85;
    text-shadow: 0 1px 4px #000;
  }}
</style>
<script type="importmap">
{{
  "imports": {{
    "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
  }}
}}
</script>
</head>
<body>
<div id="label">POINT CLOUD</div>
<script type="module">
import * as THREE from 'three';
import {{ PLYLoader }} from 'three/addons/loaders/PLYLoader.js';

const W = {W}, H = {H};
const ROT_SPEED = {ROT_SPEED};   // radians per rAF frame (~60 fps)

// ── Renderer ────────────────────────────────────────────────────────────────
// preserveDrawingBuffer MUST be true so canvas.toDataURL() can read
// pixels after the frame is presented (WebGL default clears the buffer
// immediately after composition, which is why capture was black).
const renderer = new THREE.WebGLRenderer({{ antialias: true, preserveDrawingBuffer: true }});
renderer.setPixelRatio(1);
renderer.setSize(W, H);
renderer.toneMapping    = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;
document.body.appendChild(renderer.domElement);

const scene  = new THREE.Scene();
scene.background = new THREE.Color(0x111317);

const camera = new THREE.PerspectiveCamera(45, W / H, 0.001, 1000);
// Elevated bird's-eye view so the flat terrain fills the centre of the panel.
// Y=1.6 puts the camera well above the cloud; Z=1.2 keeps it in frame.
camera.position.set(0, 1.6, 1.2);
camera.lookAt(0, 0, 0);

// ── Load PLY from base64 data-URL ────────────────────────────────────────────
const loader  = new PLYLoader();
const dataUrl = "{DATA_URL}";

loader.load(dataUrl, (geometry) => {{
  geometry.computeBoundingBox();
  geometry.center();
  geometry.computeBoundingSphere();

  const r = geometry.boundingSphere.radius;
  geometry.scale(1/r, 1/r, 1/r);   // normalise to unit sphere

  // ── PCA: find the "up" axis (smallest-variance principal component) ────
  const pos = geometry.attributes.position;
  const N   = pos.count;
  let mx=0,my=0,mz=0;
  for(let i=0;i<N;i++){{mx+=pos.getX(i);my+=pos.getY(i);mz+=pos.getZ(i);}}
  mx/=N; my/=N; mz/=N;

  let cxx=0,cxy=0,cxz=0,cyy=0,cyz=0,czz=0;
  for(let i=0;i<N;i++){{
    const dx=pos.getX(i)-mx, dy=pos.getY(i)-my, dz=pos.getZ(i)-mz;
    cxx+=dx*dx; cxy+=dx*dy; cxz+=dx*dz;
    cyy+=dy*dy; cyz+=dy*dz; czz+=dz*dz;
  }}
  // Power-iteration for smallest eigenvector of 3×3 symmetric matrix.
  // We want the eigenvector of (covMax·I - cov), i.e. the one with LEAST variance.
  // Simple approach: run power-iter on the adjugate to find the smallest.
  // For 3×3 we use the analytical characteristic trick instead:
  // deflate by the two largest and what remains is the smallest.
  // Practical shortcut: just compare variances along X/Y/Z and use the flattest axis.
  const varX = cxx/N, varY = cyy/N, varZ = czz/N;
  let upAxis;  // 0=X,1=Y,2=Z
  if(varX <= varY && varX <= varZ) upAxis = 0;
  else if(varY <= varX && varY <= varZ) upAxis = 1;
  else upAxis = 2;

  // Rotate geometry so that upAxis → world-Y
  if(upAxis === 0) {{
    geometry.applyMatrix4(new THREE.Matrix4().makeRotationZ(Math.PI/2));
  }} else if(upAxis === 2) {{
    geometry.applyMatrix4(new THREE.Matrix4().makeRotationX(-Math.PI/2));
  }}
  // upAxis===1 already aligned

  // ── Material ─────────────────────────────────────────────────────────────
  const hasColors = !!geometry.attributes.color;
  const mat = new THREE.PointsMaterial({{
    size:            0.018,
    vertexColors:    hasColors,
    color:           hasColors ? 0xffffff : 0xaaccff,
    sizeAttenuation: true,
    transparent:     true,
    opacity:         0.92,
  }});

  const cloud = new THREE.Points(geometry, mat);
  scene.add(cloud);

  // ── Animate ───────────────────────────────────────────────────────────────
  (function animate() {{
    requestAnimationFrame(animate);
    cloud.rotation.y += ROT_SPEED;
    renderer.render(scene, camera);
  }})();
}});
</script>
</body>
</html>
"""


# JS snippet injected into the page: grabs the WebGL canvas pixels directly
# into a Uint8Array and returns them as a base64 JPEG — 10-20× faster than
# Playwright's built-in page.screenshot() which does a full lossless PNG encode.
_JS_GRAB_FRAME = """
() => {
    const canvas = document.querySelector('canvas');
    if (!canvas) return null;
    // JPEG at quality 0.82 gives good colour fidelity at ~1/10 the cost of PNG
    return canvas.toDataURL('image/jpeg', 0.82).split(',')[1];
}
"""


class PlyRenderer:
    """
    Renders a PLY point cloud via a headless Chromium browser (Three.js /
    WebGL).  The browser spins its own rAF loop; Python grabs frames by
    evaluating a JS snippet that reads the canvas directly into a JPEG
    data-URL — much faster than page.screenshot().

    Architecture
    ------------
    A dedicated "browser thread" owns the asyncio event-loop and the
    Playwright page.  A second "capture thread" runs at TARGET_FPS and
    continuously grabs fresh frames into a 1-slot shared buffer.
    render() just reads the latest buffer frame — zero blocking on the
    inference side.

    Requires
    --------
      pip install playwright
      playwright install chromium
    """

    TARGET_FPS = 30   # how fast the capture loop tries to grab frames

    def __init__(self, ply_path: str, width: int, height: int, rot_speed: float = 0.025):
        import base64
        import asyncio

        try:
            from playwright.async_api import async_playwright  # noqa – import check
        except ImportError:
            raise RuntimeError(
                "playwright is required.  Install it with:\n"
                "  pip install playwright && playwright install chromium"
            )

        print(f"[PLY] Loading '{ply_path}' as base64 data-URL …")
        ply_bytes = Path(ply_path).read_bytes()
        b64       = base64.b64encode(ply_bytes).decode()
        data_url  = f"data:application/octet-stream;base64,{b64}"
        print(f"[PLY] PLY size: {len(ply_bytes)/1e6:.1f} MB  "
              f"→ data-URL {len(data_url)//1024} KB (b64)")

        html = _PLY_VIEWER_HTML.format(
            W        = width,
            H        = height,
            ROT_SPEED= rot_speed,
            DATA_URL = data_url,
        )

        # ── shared state ──────────────────────────────────────────────────
        self._ready      = threading.Event()
        self._stop       = threading.Event()
        self._exc        = None
        self._loop       = None          # set by browser thread before _ready
        self._page       = None          # set by browser thread before _ready

        # 1-slot frame buffer: inference thread reads, capture thread writes.
        # Protected by a simple lock; reading always gets the latest frame.
        self._frame_lock = threading.Lock()
        self._frame: np.ndarray | None = None

        # ── browser thread: owns asyncio loop + Playwright ────────────────
        self._browser_thread = threading.Thread(
            target = self._run_browser,
            args   = (html, width, height),
            name   = "ply_browser",
            daemon = True,
        )
        self._browser_thread.start()

        self._ready.wait(timeout=60)
        if self._exc:
            raise self._exc
        if self._page is None:
            raise RuntimeError("[PLY] Browser thread failed to start.")

        # ── capture thread: grabs frames at TARGET_FPS ────────────────────
        self._capture_thread = threading.Thread(
            target = self._run_capture,
            name   = "ply_capture",
            daemon = True,
        )
        self._capture_thread.start()
        print("[PLY] Browser ready — capture loop running.")

    # ── browser thread ────────────────────────────────────────────────────

    def _run_browser(self, html: str, width: int, height: int):
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        loop.run_until_complete(self._browser_main(html, width, height))

    async def _browser_main(self, html: str, width: int, height: int):
        from playwright.async_api import async_playwright
        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(
                    headless = True,
                    args     = [
                        "--disable-gpu-sandbox",
                        "--use-gl=egl",
                        "--enable-webgl",
                        "--ignore-gpu-blocklist",
                        f"--window-size={width},{height}",
                    ],
                )
                page = await browser.new_page(
                    viewport={"width": width, "height": height}
                )
                print("[PLY] Launching headless Chromium …")
                await page.set_content(html, wait_until="domcontentloaded")

                print("[PLY] Waiting for WebGL canvas …")
                await page.wait_for_function(
                    """() => {
                        const c = document.querySelector('canvas');
                        if (!c) return false;
                        return !!(c.getContext('webgl2') || c.getContext('webgl'));
                    }""",
                    timeout=30_000,
                )
                await page.wait_for_timeout(2000)
                self._page = page          # expose to capture thread
                self._ready.set()          # unblock __init__

                # Keep the loop alive until stop is requested
                import asyncio as _aio
                while not self._stop.is_set():
                    await _aio.sleep(0.1)

                await browser.close()
        except Exception as e:
            self._exc = e
            self._ready.set()

    # ── capture thread ────────────────────────────────────────────────────

    def _run_capture(self):
        """
        Continuously grabs canvas frames via JS evaluation and writes them
        into the 1-slot buffer.  Runs at up to TARGET_FPS.
        Uses run_coroutine_threadsafe to talk to the asyncio page safely.
        """
        import asyncio, base64, time

        interval = 1.0 / self.TARGET_FPS

        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                # Evaluate the JS grab snippet in the browser's event loop
                fut = asyncio.run_coroutine_threadsafe(
                    self._page.evaluate(_JS_GRAB_FRAME),
                    self._loop,
                )
                b64_jpeg = fut.result(timeout=5)   # base64 JPEG string
                if b64_jpeg:
                    jpeg_bytes = base64.b64decode(b64_jpeg)
                    buf        = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                    frame      = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    if frame is not None:
                        with self._frame_lock:
                            self._frame = frame
            except Exception as e:
                print(f"[PLY capture] {e}")

            elapsed = time.perf_counter() - t0
            wait    = interval - elapsed
            if wait > 0:
                time.sleep(wait)

    # ── public API ────────────────────────────────────────────────────────

    def render(self) -> np.ndarray | None:
        """
        Returns the most recently captured frame (BGR uint8).
        Non-blocking: returns None if no frame is ready yet.
        """
        with self._frame_lock:
            return self._frame

    def close(self):
        self._stop.set()
        self._capture_thread.join(timeout=3)
        self._browser_thread.join(timeout=5)

    def __del__(self):
        self.close()


# ── Frame composer ────────────────────────────────────────────────────────────

def compose_frame_with_ply(depth_bgr, seg_bgr, original_bgr, show_original,
                            ply_frame: np.ndarray,
                            out_w=STREAM_W, out_h=STREAM_H):
    """
    Left half : depth (top) + segmentation (bottom), with optional original PiP.
    Right half: rotating PLY render passed in as ply_frame.
    """
    left_w  = out_w // 2
    right_w = out_w - left_w
    half_h  = out_h // 2

    # ── left column ──────────────────────────────────────────────────────────
    depth_resized = cv2.resize(depth_bgr, (left_w, half_h))
    seg_resized   = cv2.resize(seg_bgr,   (left_w, half_h))

    for img, label in ((depth_resized, "DEPTH"), (seg_resized, "SEGMENTATION")):
        cv2.putText(img, label, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    left_col = np.concatenate([depth_resized, seg_resized], axis=0)  # (out_h, left_w, 3)

    # Optional PiP of the original in the bottom-left corner of the left column
    if show_original:
        pip_w = int(left_w  * ORIGINAL_SCALE)
        pip_h = int(out_h   * ORIGINAL_SCALE)
        pip   = cv2.resize(original_bgr, (pip_w, pip_h))
        border = 2
        x1 = left_w - pip_w - border
        y1 = out_h  - pip_h - border
        left_col[y1 - border:y1 + pip_h + border,
                 x1 - border:x1 + pip_w + border] = 255
        left_col[y1:y1 + pip_h, x1:x1 + pip_w] = pip

    # ── right column ─────────────────────────────────────────────────────────
    right_col = cv2.resize(ply_frame, (right_w, out_h))

    # Thin separator line
    canvas = np.concatenate([left_col, right_col], axis=1)
    canvas[:, left_w - 1:left_w + 1] = (80, 80, 80)

    return canvas


def compose_frame(depth_bgr, seg_bgr, original_bgr, show_original,
                  out_w=STREAM_W, out_h=STREAM_H):
    """
    Original two-panel layout (no PLY).
    depth | segmentation  side by side, full height.
    """
    half_w = out_w // 2
    depth_resized = cv2.resize(depth_bgr, (half_w, out_h))
    seg_resized   = cv2.resize(seg_bgr,   (half_w, out_h))

    for img, label in ((depth_resized, "DEPTH"), (seg_resized, "SEGMENTATION")):
        cv2.putText(img, label, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    canvas = np.concatenate([depth_resized, seg_resized], axis=1)

    if show_original:
        pip_w  = int(out_w * ORIGINAL_SCALE)
        pip_h  = int(out_h * ORIGINAL_SCALE)
        pip    = cv2.resize(original_bgr, (pip_w, pip_h))
        border = 2
        x1 = out_w - pip_w - border
        y1 = out_h - pip_h - border
        canvas[y1 - border:y1 + pip_h + border,
               x1 - border:x1 + pip_w + border] = 255
        canvas[y1:y1 + pip_h, x1:x1 + pip_w] = pip

    return canvas


# ── FFmpeg RTSP pusher ────────────────────────────────────────────────────────

def build_ffmpeg_cmd(rtsp_url, fps, width, height):
    return [
        "ffmpeg",
        "-loglevel", "warning",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-x264-params", "nal-hrd=cbr",
        "-b:v", "2M",
        "-minrate", "2M",
        "-maxrate", "2M",
        "-bufsize", "4M",
        "-pix_fmt", "yuv420p",
        "-g", str(fps * 2),
        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        rtsp_url,
    ]


# ── Image loader ──────────────────────────────────────────────────────────────

def collect_images(input_dir: Path) -> list[Path]:
    paths = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )
    if not paths:
        raise FileNotFoundError(
            f"No supported images found in '{input_dir}'. "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTS))}"
        )
    return paths


# ── Threads ───────────────────────────────────────────────────────────────────

def inference_thread(image_paths, result_queue, depth_model, seg_model,
                     show_original, loop, stop_event,
                     ply_renderer=None):
    """
    Iterates over image_paths, runs inference on each, composes frames
    (optionally including the spinning PLY panel), and puts them in
    result_queue.
    """
    total = len(image_paths)
    print(f"[Inference] {total} image(s) to process. Loop={loop}  PLY={'yes' if ply_renderer else 'no'}")

    while not stop_event.is_set():
        for idx, path in enumerate(image_paths, 1):
            if stop_event.is_set():
                break

            frame_bgr = cv2.imread(str(path))
            if frame_bgr is None:
                print(f"\n[Inference] Could not read '{path.name}' — skipping.")
                continue

            t0 = time.perf_counter()
            try:
                depth_bgr, seg_bgr, orig_bgr = run_inference(
                    frame_bgr, depth_model, seg_model
                )

                if ply_renderer is not None:
                    # Non-blocking: capture thread keeps the buffer fresh.
                    # Fall back to placeholder if buffer not filled yet.
                    ply_frame = ply_renderer.render()
                    if ply_frame is None:
                        right_w = STREAM_W - STREAM_W // 2
                        ply_frame = np.full((STREAM_H, right_w, 3), 30, dtype=np.uint8)
                        cv2.putText(ply_frame, 'PLY loading...', (20, STREAM_H // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
                    composed  = compose_frame_with_ply(
                        depth_bgr, seg_bgr, orig_bgr, show_original, ply_frame
                    )
                else:
                    composed = compose_frame(depth_bgr, seg_bgr, orig_bgr, show_original)

            except Exception as e:
                print(f"\n[Inference] Error on '{path.name}': {e}")
                composed = PLACEHOLDER_WAITING

            elapsed_ms = (time.perf_counter() - t0) * 1000
            print(f"[Inference] [{idx:4d}/{total}] {path.name:<40s}  {elapsed_ms:6.1f} ms")

            while not stop_event.is_set():
                try:
                    result_queue.put(composed, timeout=0.1)
                    break
                except queue.Full:
                    pass

        if loop:
            print("[Inference] End of folder — looping.")
        else:
            print("[Inference] All images processed.")
            result_queue.put(None)
            break

    print("[Inference] Stopped.")


def rtsp_pusher_thread(result_queue, rtsp_url, stop_event):
    """
    Pulls composed frames from result_queue and writes them to FFmpeg at a
    steady RTSP_FPS rate.
    """
    cmd = build_ffmpeg_cmd(rtsp_url, RTSP_FPS, STREAM_W, STREAM_H)
    print(f"[RTSP] Starting FFmpeg → {rtsp_url}")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    frame_interval = 1.0 / RTSP_FPS
    last_frame     = PLACEHOLDER_WAITING
    pushed         = 0
    done           = False

    def write_frame(f):
        nonlocal pushed
        # Ensure the frame matches the expected stream dimensions
        if f.shape[1] != STREAM_W or f.shape[0] != STREAM_H:
            f = cv2.resize(f, (STREAM_W, STREAM_H))
        try:
            proc.stdin.write(f.tobytes())
            proc.stdin.flush()
            pushed += 1
        except BrokenPipeError:
            return False
        return True

    try:
        while not stop_event.is_set():
            t0 = time.perf_counter()

            if done:
                if not write_frame(PLACEHOLDER_DONE):
                    break
            else:
                try:
                    frame = result_queue.get(timeout=0.05)
                    if frame is None:
                        done  = True
                        frame = PLACEHOLDER_DONE
                    last_frame = frame
                except queue.Empty:
                    frame = last_frame

                if not write_frame(frame):
                    print("\n[RTSP] FFmpeg pipe broken — restarting...")
                    proc.wait()
                    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

            elapsed = time.perf_counter() - t0
            sleep_t = frame_interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    except Exception as e:
        print(f"\n[RTSP] Error: {e}")
        stop_event.set()
    finally:
        try:
            proc.stdin.close()
            proc.wait(timeout=3)
        except Exception:
            proc.kill()
        print(f"\n[RTSP] Closed. Pushed {pushed} frames total.")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Folder inference streamer: TRT depth+seg on images → RTSP."
    )
    parser.add_argument(
        "--input-dir", required=True, type=Path,
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--ply-file", default=None, type=Path,
        help="Optional path to a COLMAP/point-cloud .ply file.  When provided, "
             "the stream layout changes: depth+seg on the left, spinning 3-D "
             "point cloud on the right.",
    )
    parser.add_argument(
        "--show-original", action="store_true", default=False,
        help="Overlay the original frame as a PiP in the bottom corner of the "
             "depth/seg panel.",
    )
    parser.add_argument(
        "--loop", action="store_true", default=False,
        help="Loop over the image folder indefinitely instead of stopping.",
    )
    parser.add_argument(
        "--rtsp-url", default="rtsp://localhost:8554/aurelius",
        help="RTSP publish URL (default: rtsp://localhost:8554/aurelius).",
    )
    parser.add_argument(
        "--ply-rot-speed", type=float, default=0.025,
        help="Point-cloud rotation speed in radians per browser animation frame "
             "(default: 0.008 ≈ one full revolution every ~13 s).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Folder Inference Streamer")
    print("=" * 60)
    print(f"  Input dir     : {args.input_dir}")
    print(f"  PLY file      : {args.ply_file or 'none (classic layout)'}")
    print(f"  RTSP URL      : {args.rtsp_url}")
    print(f"  Show original : {args.show_original}")
    print(f"  Loop          : {args.loop}")
    print(f"  Output size   : {STREAM_W}×{STREAM_H} @ {RTSP_FPS} FPS")
    print(f"  Device        : {DEVICE}")
    print("=" * 60 + "\n")

    # Validate PLY path early
    ply_renderer = None
    if args.ply_file is not None:
        if not args.ply_file.is_file():
            raise FileNotFoundError(f"PLY file not found: '{args.ply_file}'")
        if args.ply_file.suffix.lower() != ".ply":
            raise ValueError(f"Expected a .ply file, got: '{args.ply_file}'")

        # Right panel = half stream width, full height
        ply_w = STREAM_W - STREAM_W // 2
        ply_h = STREAM_H
        rot_speed    = args.ply_rot_speed
        ply_renderer = PlyRenderer(str(args.ply_file), width=ply_w, height=ply_h,
                                   rot_speed=rot_speed)

    image_paths = collect_images(args.input_dir)
    print(f"Found {len(image_paths)} image(s) in '{args.input_dir}'.\n")

    print("Loading TRT engines...")
    depth_model = TRTModel(DEPTH_TRT)
    seg_model   = TRTModel(SEG_TRT)
    print("Engines loaded.\n")

    stop_event   = threading.Event()
    result_queue = queue.Queue(maxsize=RTSP_FPS)

    threads = [
        threading.Thread(
            target=inference_thread,
            args=(image_paths, result_queue, depth_model, seg_model,
                  args.show_original, args.loop, stop_event, ply_renderer),
            name="inference", daemon=True,
        ),
        threading.Thread(
            target=rtsp_pusher_thread,
            args=(result_queue, args.rtsp_url, stop_event),
            name="rtsp_pusher", daemon=True,
        ),
    ]

    for t in threads:
        t.start()
        print(f"[Main] Thread '{t.name}' started.")

    print("\n[Main] Running — press Ctrl+C to stop.\n")

    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[Main] Ctrl+C received — shutting down...")
        stop_event.set()

    for t in threads:
        t.join(timeout=5)

    if ply_renderer is not None:
        ply_renderer.close()
        print("[Main] PLY renderer closed.")

    print("[Main] All threads stopped. Bye.")


if __name__ == "__main__":
    main()