#!/usr/bin/env python3
"""
Folder Inference Streamer
=========================
Reads images from a folder, runs TRT depth + segmentation on each, composes
a side-by-side visualisation and pushes it as an RTSP stream to MediaMTX.

Layout
------
  [ DEPTH  |  SEGMENTATION ]
               [ original ]   ← optional, controlled by --show-original

Usage
-----
  python folder_inference_streamer.py --input-dir ./frames [--show-original] [--loop]

MediaMTX (run once before this script):
  docker run --rm -it --network=host bluenviron/mediamtx:latest

Consume the stream with e.g.:
  ffplay rtsp://localhost:8554/aurelius
  vlc rtsp://localhost:8554/aurelius
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

RTSP_FPS    = 15
STREAM_W    = 1024
STREAM_H    = 512
ORIGINAL_SCALE = 0.25

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


# ── Frame composer ────────────────────────────────────────────────────────────

def compose_frame(depth_bgr, seg_bgr, original_bgr, show_original,
                  out_w=STREAM_W, out_h=STREAM_H):
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
    """Return a sorted list of supported image paths from the given directory."""
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
                     show_original, loop, stop_event):
    """
    Iterates over image_paths, runs inference on each, and puts composed
    frames into result_queue.  Loops forever if --loop is set, otherwise
    pushes a 'done' sentinel after the last image.
    """
    total = len(image_paths)
    print(f"[Inference] {total} image(s) to process. Loop={loop}")

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
                composed = compose_frame(depth_bgr, seg_bgr, orig_bgr, show_original)
            except Exception as e:
                print(f"\n[Inference] Error on '{path.name}': {e}")
                composed = PLACEHOLDER_WAITING

            elapsed_ms = (time.perf_counter() - t0) * 1000
            print(f"[Inference] [{idx:4d}/{total}] {path.name:<40s}  {elapsed_ms:6.1f} ms")

            # Block until there's space — keeps inference in sync with RTSP rate
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
            # Push sentinel: None signals the RTSP pusher to show the 'done' frame
            result_queue.put(None)
            break

    print("[Inference] Stopped.")


def rtsp_pusher_thread(result_queue, rtsp_url, stop_event):
    """
    Pulls composed frames from result_queue and writes them to FFmpeg at a
    steady RTSP_FPS rate.  Repeats the last frame when the queue is momentarily
    empty.  Stops (and holds the 'done' placeholder) when it receives None.
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
                # Stream the 'done' placeholder indefinitely until Ctrl+C
                if not write_frame(PLACEHOLDER_DONE):
                    break
            else:
                try:
                    frame = result_queue.get(timeout=0.05)
                    if frame is None:
                        # Sentinel — switch to 'done' placeholder
                        done = True
                        frame = PLACEHOLDER_DONE
                    last_frame = frame
                except queue.Empty:
                    frame = last_frame   # repeat last real frame

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
        "--show-original", action="store_true", default=False,
        help="Overlay the original frame as a PiP in the bottom-right corner.",
    )
    parser.add_argument(
        "--loop", action="store_true", default=False,
        help="Loop over the image folder indefinitely instead of stopping.",
    )
    parser.add_argument(
        "--rtsp-url", default="rtsp://localhost:8554/aurelius",
        help="RTSP publish URL (default: rtsp://localhost:8554/aurelius).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Folder Inference Streamer")
    print("=" * 60)
    print(f"  Input dir     : {args.input_dir}")
    print(f"  RTSP URL      : {args.rtsp_url}")
    print(f"  Show original : {args.show_original}")
    print(f"  Loop          : {args.loop}")
    print(f"  Output size   : {STREAM_W}×{STREAM_H} @ {RTSP_FPS} FPS")
    print(f"  Device        : {DEVICE}")
    print("=" * 60 + "\n")

    image_paths = collect_images(args.input_dir)
    print(f"Found {len(image_paths)} image(s) in '{args.input_dir}'.\n")

    print("Loading TRT engines...")
    depth_model = TRTModel(DEPTH_TRT)
    seg_model   = TRTModel(SEG_TRT)
    print("Engines loaded.\n")

    stop_event   = threading.Event()
    # Queue sized to buffer ~1 second of output — prevents inference from
    # running too far ahead of the RTSP pusher
    result_queue = queue.Queue(maxsize=RTSP_FPS)

    threads = [
        threading.Thread(
            target=inference_thread,
            args=(image_paths, result_queue, depth_model, seg_model,
                  args.show_original, args.loop, stop_event),
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

    print("[Main] All threads stopped. Bye.")


if __name__ == "__main__":
    main()