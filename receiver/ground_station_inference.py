#!/usr/bin/env python3
"""
Ground Station — Inference + RTSP Streamer
==========================================
Receives WebP frames from the drone via core_transport, runs TRT depth +
segmentation on the freshest frame, composes a side-by-side visualisation
and pushes it as an RTSP stream to a local MediaMTX container.

Layout
------
  [ DEPTH  |  SEGMENTATION ]
                 [ original ]   ← optional, controlled by --show-original

Resilience
----------
  - The RTSP stream is opened immediately at startup and kept alive forever,
    even before the first drone frame arrives or after the drone disconnects.
  - While no real frames are available a placeholder is pushed so viewers
    always see something (dark frame + "Aurelius — waiting for drone…").
  - The receiver loops forever: it re-binds and waits for a new connection
    whenever the drone drops without stopping inference or the RTSP pusher.

Usage
-----
  python ground_station_inference.py [--show-original] [--rtsp-url URL]

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

from core_transport import ObjectReceiver
from datasets_multitask import UNIFIED_CLASSES, UNIFIED_NAME_TO_ID

# ── Configuration ────────────────────────────────────────────────────────────

BIND_IP   = "0.0.0.0"
PORT      = 50010

DEPTH_TRT = "./models/depth_teacher.trt"
SEG_TRT   = "./models/seg_teacher.trt"

IMG_SIZE  = 512
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

QUEUE_MAX = 10          # keep only the last N frames
RTSP_FPS  = 15          # target FPS for the RTSP stream
STREAM_W  = 1024        # output frame width  (2 × 512 side-by-side)
STREAM_H  = 512         # output frame height
ORIGINAL_SCALE = 0.25   # size of the PiP original relative to the full output

# How long to wait before re-binding the receiver after a disconnect (seconds)
RECEIVER_RETRY_DELAY = 2.0

SEGMENTATION_INTEREST_CLASSES = [
    "building", "roof", "roof-transparent", "truck", "bus", "car",
    "person", "van", "tricycle", "small-vehicle", "large-vehicle", "plane",
]
INTEREST_IDS = [
    UNIFIED_NAME_TO_ID[n]
    for n in SEGMENTATION_INTEREST_CLASSES
    if n in UNIFIED_NAME_TO_ID
]
MIN_SEG_PIXEL_AREA = 100

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


# ── Palette ──────────────────────────────────────────────────────────────────

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

def make_placeholder(width: int = STREAM_W, height: int = STREAM_H) -> np.ndarray:
    """
    Dark frame with a centred status message, used while no drone data arrives.
    Matches the exact pixel format expected by the FFmpeg pipe (BGR, uint8).
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (18, 18, 18)   # very dark grey — not pure black

    # Thin horizontal rule
    cv2.line(frame, (40, height // 2 - 36), (width - 40, height // 2 - 36),
             (60, 60, 60), 1)

    lines = [
        ("AURELIUS", 1.4, (200, 200, 200), 2),
        ("Waiting for drone stream...", 0.65, (120, 120, 120), 1),
    ]
    for i, (text, scale, color, thickness) in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        x = (width  - tw) // 2
        y = height // 2 + i * int(th * 2.6)
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    # Thin horizontal rule below
    cv2.line(frame, (40, height // 2 + 52), (width - 40, height // 2 + 52),
             (60, 60, 60), 1)

    return frame

PLACEHOLDER_FRAME = make_placeholder()


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
    """Letterbox → normalise → NCHW tensor.  Returns (tensor, dw, dh, scale)."""
    orig_h, orig_w = frame_bgr.shape[:2]
    scale  = min(IMG_SIZE / orig_w, IMG_SIZE / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized = cv2.resize(frame_bgr, (new_w, new_h))
    padded  = np.full((IMG_SIZE, IMG_SIZE, 3), 128, dtype=np.uint8)
    dw, dh  = (IMG_SIZE - new_w) // 2, (IMG_SIZE - new_h) // 2
    padded[dh:dh + new_h, dw:dw + new_w] = resized
    rgb   = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    x     = torch.from_numpy(rgb.astype(np.float32) / 255.0)
    x     = x.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    x     = x.to(memory_format=torch.channels_last)
    x_norm = (x - 0.5) / 0.5
    return x_norm, rgb.astype(np.uint8), dw, dh, new_w, new_h


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
    """Returns (depth_bgr, seg_bgr, original_bgr) all at original frame resolution."""
    orig_h, orig_w = frame_bgr.shape[:2]
    x_norm, img_rgb, dw, dh, new_w, new_h = _preprocess(frame_bgr)

    with torch.no_grad():
        depth_out = depth_model(x_norm)
        seg_out   = seg_model(x_norm)
    torch.cuda.synchronize()

    depth_map = (depth_out[0, 0] if depth_out.ndim == 4 else depth_out[0]) * 255.0
    depth_bgr_padded = _depth_to_rgb(depth_map)

    seg_mask = seg_out.argmax(dim=1)[0]
    seg_bgr_padded = _seg_to_color(seg_mask)

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


# ── Thread-safe queues ────────────────────────────────────────────────────────

class FrameQueue:
    """Thread-safe queue that keeps only the last `maxsize` frames."""

    def __init__(self, maxsize=QUEUE_MAX):
        self._q    = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()

    def put(self, frame):
        with self._lock:
            if self._q.full():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
            self._q.put_nowait(frame)

    def get_latest(self, timeout=1.0):
        """Drain the queue and return only the most recent frame."""
        latest = None
        try:
            while True:
                latest = self._q.get_nowait()
        except queue.Empty:
            pass
        if latest is not None:
            return latest
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None


# ── Threads ───────────────────────────────────────────────────────────────────

def receiver_thread(frame_queue: FrameQueue, stop_event: threading.Event):
    """
    Receives frames from the drone indefinitely.
    When the drone disconnects the receiver is rebuilt and we wait for the
    next connection — the stop_event is NEVER set here, so the rest of the
    pipeline keeps running (and pushing the placeholder to RTSP).
    """
    attempt = 0
    while not stop_event.is_set():
        attempt += 1
        print(f"\n[Receiver] Binding on {BIND_IP}:{PORT}  (attempt {attempt})...")
        try:
            receiver = ObjectReceiver(BIND_IP, PORT, name="ground_station", send_ack=False)
            receiver.start()
        except Exception as e:
            print(f"[Receiver] Could not bind: {e}  — retrying in {RECEIVER_RETRY_DELAY}s...")
            time.sleep(RECEIVER_RETRY_DELAY)
            continue

        print("[Receiver] Ready — waiting for drone frames.")

        frames_this_sec = 0
        last_fps_time   = time.perf_counter()
        current_fps     = 0.0

        try:
            while not stop_event.is_set():
                obj = receiver.receive_object()
                if obj is None:
                    # Drone disconnected — break inner loop, rebuild receiver
                    print("\n[Receiver] Drone disconnected — will re-listen.")
                    break

                if obj.get("msg_type") == "camera_frame":
                    byte_data = obj["image_bytes"]
                    np_arr    = np.frombuffer(byte_data, np.uint8)
                    frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue

                    frame_queue.put(frame)

                    frames_this_sec += 1
                    now = time.perf_counter()
                    if now - last_fps_time >= 1.0:
                        current_fps     = frames_this_sec / (now - last_fps_time)
                        frames_this_sec = 0
                        last_fps_time   = now

                    seq = obj.get("seq", 0)
                    kb  = len(byte_data) / 1024.0
                    print(f"\r[Receiver] Seq: {seq:05d} | {kb:5.1f} KB | "
                          f"FPS in: {current_fps:4.1f}",
                          end="", flush=True)

        except Exception as e:
            print(f"\n[Receiver] Error: {e}")
        finally:
            try:
                receiver.close()
            except Exception:
                pass

        if not stop_event.is_set():
            print(f"[Receiver] Retrying in {RECEIVER_RETRY_DELAY}s...")
            time.sleep(RECEIVER_RETRY_DELAY)

    print("\n[Receiver] Stopped.")


def inference_thread(frame_queue, result_queue, depth_model, seg_model,
                     show_original, stop_event):
    """
    Pulls frames from frame_queue and runs TRT inference.
    When no frame is available the placeholder is forwarded directly so the
    RTSP pusher always has something to send.
    """
    print("[Inference] Ready.")
    inf_times   = []
    idle_warned = False

    while not stop_event.is_set():
        frame = frame_queue.get_latest(timeout=0.5)

        if frame is None:
            # No drone frame — forward the placeholder unchanged
            if not idle_warned:
                print("\n[Inference] No frames — streaming placeholder.")
                idle_warned = True
            _push_to_result(result_queue, PLACEHOLDER_FRAME)
            continue

        idle_warned = False
        t0 = time.perf_counter()
        try:
            depth_bgr, seg_bgr, orig_bgr = run_inference(frame, depth_model, seg_model)
            composed = compose_frame(depth_bgr, seg_bgr, orig_bgr, show_original)
        except Exception as e:
            print(f"\n[Inference] Error: {e}")
            _push_to_result(result_queue, PLACEHOLDER_FRAME)
            continue

        elapsed_ms = (time.perf_counter() - t0) * 1000
        inf_times.append(elapsed_ms)
        _push_to_result(result_queue, composed)

        avg_ms = sum(inf_times[-30:]) / min(len(inf_times), 30)
        print(f"[Inference] {elapsed_ms:5.1f} ms  (avg {avg_ms:5.1f} ms)",
              end="", flush=True)

    print("\n[Inference] Stopped.")


def _push_to_result(result_queue, frame):
    """Non-blocking push to the RTSP result queue — evicts oldest if full."""
    try:
        result_queue.put_nowait(frame)
    except queue.Full:
        try:
            result_queue.get_nowait()
        except queue.Empty:
            pass
        result_queue.put_nowait(frame)


def rtsp_pusher_thread(result_queue, rtsp_url, stop_event):
    """
    Pushes frames to FFmpeg via stdin at a steady RTSP_FPS rate.
    Falls back to the placeholder when the inference queue is empty.
    Restarts FFmpeg automatically if the pipe breaks.
    """
    cmd = build_ffmpeg_cmd(rtsp_url, RTSP_FPS, STREAM_W, STREAM_H)
    print(f"[RTSP] Starting FFmpeg → {rtsp_url}")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    frame_interval = 1.0 / RTSP_FPS
    last_frame     = PLACEHOLDER_FRAME   # ← start with placeholder, not None
    pushed         = 0

    try:
        while not stop_event.is_set():
            t0 = time.perf_counter()

            try:
                frame = result_queue.get(timeout=0.05)
                last_frame = frame
            except queue.Empty:
                frame = last_frame   # repeat last frame (real or placeholder)

            try:
                proc.stdin.write(frame.tobytes())
                proc.stdin.flush()
                pushed += 1
            except BrokenPipeError:
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
        print(f"\n[RTSP] Closed. Pushed {pushed} frames.")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Ground station: TRT inference + RTSP stream from drone feed."
    )
    parser.add_argument(
        "--show-original", action="store_true", default=False,
        help="Overlay the original frame as a PiP in the bottom-right corner.",
    )
    parser.add_argument(
        "--rtsp-url", default="rtsp://localhost:8554/aurelius",
        help="RTSP publish URL (default: rtsp://localhost:8554/aurelius).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  Ground Station — TRT Inference + RTSP Stream")
    print("=" * 60)
    print(f"  RTSP URL      : {args.rtsp_url}")
    print(f"  Show original : {args.show_original}")
    print(f"  Output size   : {STREAM_W}×{STREAM_H} @ {RTSP_FPS} FPS")
    print(f"  Device        : {DEVICE}")
    print("=" * 60 + "\n")

    print("Loading TRT engines...")
    depth_model = TRTModel(DEPTH_TRT)
    seg_model   = TRTModel(SEG_TRT)
    print("Engines loaded.\n")

    stop_event   = threading.Event()
    frame_queue  = FrameQueue(maxsize=QUEUE_MAX)
    result_queue = queue.Queue(maxsize=4)

    threads = [
        threading.Thread(
            target=receiver_thread,
            args=(frame_queue, stop_event),
            name="receiver", daemon=True,
        ),
        threading.Thread(
            target=inference_thread,
            args=(frame_queue, result_queue, depth_model, seg_model,
                  args.show_original, stop_event),
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