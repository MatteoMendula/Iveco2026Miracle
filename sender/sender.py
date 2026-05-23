#!/usr/bin/env python3
"""
Drone Camera Sender
===================
Captures frames from a V4L2 USB camera, optionally saves raw frames to disk,
encodes them as WebP, and streams them to the ground station via core_transport.

Resilience
----------
  If the connection drops (network loss, ground station restart, etc.) the sender
  automatically retries the connection indefinitely — no human intervention needed.

Usage
-----
  python drone_camera_sender.py [--save-dir PATH]

  --save-dir PATH   If given, every raw (pre-resize/pre-WebP) frame is saved as
                    a timestamped PNG inside PATH.  Directory is created if it
                    doesn't exist.  Saving runs in a background thread so it
                    never blocks the capture / send loop.
"""

import argparse
import queue
import threading
import time
from pathlib import Path

import cv2

from core_transport import ObjectSender

# ── Configuration ─────────────────────────────────────────────────────────────

HOST       = "100.111.65.121"   # Ground station IP
PORT       = 50010

TARGET_W   = 512
TARGET_H   = 288
WEBP_PARAMS = [int(cv2.IMWRITE_WEBP_QUALITY), 95]

# How long to wait between reconnection attempts (seconds)
RECONNECT_DELAY = 3.0

# Max frames buffered for the disk-save worker before dropping
SAVE_QUEUE_MAX = 30


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drone camera → WebP → ground station streamer."
    )
    parser.add_argument(
        "--save-dir",
        metavar="PATH",
        default=None,
        help=(
            "Directory to save raw (pre-compression) frames as PNG files. "
            "Created automatically if it does not exist. "
            "Omit this flag to disable saving."
        ),
    )
    return parser.parse_args()


# ── Background disk-save worker ───────────────────────────────────────────────

class FrameSaver:
    """
    Consumes (seq, raw_frame) tuples from an internal queue and writes them
    to disk in a dedicated daemon thread, so saving never stalls the sender.
    """

    def __init__(self, save_dir: str):
        self._dir = Path(save_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._q = queue.Queue(maxsize=SAVE_QUEUE_MAX)
        self._thread = threading.Thread(target=self._worker, name="frame_saver", daemon=True)
        self._thread.start()
        print(f"[Saver] Saving raw frames to: {self._dir.resolve()}")

    def submit(self, seq: int, frame: "cv2.Mat") -> None:
        """Non-blocking enqueue — drops the frame if the queue is full."""
        try:
            self._q.put_nowait((seq, frame.copy()))
        except queue.Full:
            pass  # disk is too slow; silently drop rather than block capture

    def _worker(self) -> None:
        while True:
            seq, frame = self._q.get()
            try:
                ts   = time.strftime("%Y%m%d_%H%M%S")
                path = self._dir / f"frame_{ts}_{seq:06d}.png"
                cv2.imwrite(str(path), frame)
            except Exception as e:
                print(f"\n[Saver] Write error: {e}")


# ── Camera helpers ─────────────────────────────────────────────────────────────

def open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError("Could not open /dev/video0 via V4L2.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    return cap


def crop_and_encode(frame):
    """
    Centre-crop to 16:9, resize, WebP-encode.
    Returns (encoded_bytes, scaled_frame) or (None, None) on failure.
    """
    h, w      = frame.shape[:2]
    target_ar = TARGET_W / TARGET_H
    new_h     = int(w / target_ar)
    if new_h < h:
        start_y = (h - new_h) // 2
        frame   = frame[start_y:start_y + new_h, :]

    scaled = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".webp", scaled, WEBP_PARAMS)
    if not ok:
        return None, None
    return buf.tobytes(), scaled


# ── Connection management ─────────────────────────────────────────────────────

def make_sender() -> ObjectSender:
    return ObjectSender(HOST, PORT, name="drone_camera", compress=False, require_ack=False)


def connect_with_retry(sender: ObjectSender) -> None:
    """Block until a connection is established, retrying indefinitely."""
    attempt = 0
    while True:
        attempt += 1
        try:
            print(f"[Sender] Connecting to {HOST}:{PORT}  (attempt {attempt})...")
            sender.connect()
            print("[Sender] Connected.\n")
            return
        except Exception as e:
            print(f"[Sender] Connection failed: {e}  — retrying in {RECONNECT_DELAY}s...")
            time.sleep(RECONNECT_DELAY)


# ── Main stream loop ──────────────────────────────────────────────────────────

def stream_camera(args: argparse.Namespace) -> None:
    saver = FrameSaver(args.save_dir) if args.save_dir else None

    print("=" * 56)
    print("  Drone Camera Sender")
    print("=" * 56)
    print(f"  Ground station : {HOST}:{PORT}")
    print(f"  Output size    : {TARGET_W}×{TARGET_H}")
    print(f"  Save raw frames: {args.save_dir or 'disabled'}")
    print("=" * 56 + "\n")

    cap = open_camera()

    seq = 0

    # Outer loop — reconnects whenever the connection is lost
    while True:
        sender = make_sender()
        connect_with_retry(sender)

        # Inner loop — capture & send until something breaks
        try:
            while True:
                t_start = time.perf_counter()

                ret, raw_frame = cap.read()
                if not ret:
                    print("\n[Sender] Camera read failed — reconnecting camera...")
                    cap.release()
                    time.sleep(1.0)
                    cap = open_camera()
                    continue

                t_cap = time.perf_counter()

                # Optionally save the raw frame (before any processing)
                if saver is not None:
                    saver.submit(seq, raw_frame)

                encoded_bytes, _ = crop_and_encode(raw_frame)
                if encoded_bytes is None:
                    print("\n[Sender] Encode failed — skipping frame.")
                    continue

                t_proc = time.perf_counter()

                payload = {
                    "msg_type":    "camera_frame",
                    "seq":         seq,
                    "timestamp":   time.time(),
                    "image_bytes": encoded_bytes,
                }

                sender.send_object(payload)

                t_send = time.perf_counter()

                proc_ms  = (t_proc - t_cap)   * 1000
                send_ms  = (t_send - t_proc)  * 1000
                total_ms = (t_send - t_start) * 1000
                kb       = len(encoded_bytes) / 1024.0

                print(
                    f"\r[Sender] Seq: {seq:05d} | {kb:5.1f} KB | "
                    f"Proc: {proc_ms:4.1f} ms | Send: {send_ms:4.1f} ms | "
                    f"Total: {total_ms:4.1f} ms",
                    end="", flush=True,
                )
                seq += 1

        except KeyboardInterrupt:
            print("\n\n[Sender] Stopped by user.")
            cap.release()
            sender.close()
            return

        except Exception as e:
            # Network drop, broken pipe, ground station restarted, etc.
            print(f"\n[Sender] Connection lost: {e}")
            print(f"[Sender] Retrying in {RECONNECT_DELAY}s...\n")
            try:
                sender.close()
            except Exception:
                pass
            time.sleep(RECONNECT_DELAY)
            # Immediately go back to the outer loop and reconnect


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    stream_camera(parse_args())