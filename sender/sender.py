#!/usr/bin/env python3
"""
Drone Camera + MTI Sender
=========================
Captures frames from a V4L2 USB camera, optionally saves raw frames to disk,
encodes them as WebP, and streams them to the ground station via core_transport.

MTI telemetry is read from the ``state_latest.json`` file written by
``collect_payload_supervisor.py`` and streamed to the ground station as
``mti_sample`` messages.

If ``--mti-save-dir`` is given, every new MTI sample is also appended to
``<mti-save-dir>/mti_samples.jsonl`` on the drone.

Resilience
----------
  If the connection drops the sender retries indefinitely — no human
  intervention needed.

Usage
-----
  python sender.py [options]

  --mti-state-json PATH   Path to state_latest.json written by the supervisor.
                          Defaults to mti_readings/state_latest.json next to
                          this script.
  --mti-save-dir PATH     Also append MTI samples to <PATH>/mti_samples.jsonl.
  --save-dir PATH         Save raw (pre-compression) camera frames as PNG.
"""

import argparse
import json
import os
import queue
import threading
import time
from pathlib import Path

import cv2

from core_transport import ObjectSender

# ── Configuration ──────────────────────────────────────────────────────────────

DEFAULT_HOST     = "100.111.65.121"   # Ground station IP
PORT             = 50010

TARGET_W         = 512
TARGET_H         = 288
WEBP_PARAMS      = [int(cv2.IMWRITE_WEBP_QUALITY), 95]

RECONNECT_DELAY  = 3.0   # seconds between reconnection attempts
MAX_FPS          = 10    # upper bound on capture rate
SAVE_QUEUE_MAX   = 30    # max frames buffered for disk-save worker

MTI_POLL_HZ      = 50    # how often to poll state_latest.json (Hz)
MTI_QUEUE_MAX    = 200   # samples buffered between reader and sender threads

# Default path for state_latest.json if --mti-state-json is not provided
_SCRIPT_DIR          = Path(__file__).resolve().parent
DEFAULT_MTI_STATE    = _SCRIPT_DIR / "mti_readings" / "state_latest.json"


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drone camera + MTI → ground station streamer."
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Ground station IP address (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--max-fps-capture",
        type=float,
        default=MAX_FPS,
        help=f"Maximum frames per second to capture (default: {MAX_FPS})",
    )
    parser.add_argument(
        "--save-dir",
        metavar="PATH",
        default=None,
        help=(
            "Directory to save raw (pre-compression) frames as PNG files. "
            "Created automatically if it does not exist. Omit to disable."
        ),
    )
    parser.add_argument(
        "--mti-state-json",
        metavar="PATH",
        default=str(DEFAULT_MTI_STATE),
        help=(
            "Path to state_latest.json produced by collect_payload_supervisor.py. "
            f"Default: {DEFAULT_MTI_STATE}"
        ),
    )
    parser.add_argument(
        "--mti-save-dir",
        metavar="PATH",
        default=None,
        help=(
            "If set, every MTI sample is also appended to "
            "<PATH>/mti_samples.jsonl on the drone. "
            "Directory is created automatically. Omit to disable."
        ),
    )
    return parser.parse_args()


# ── Background disk-save worker (camera frames) ────────────────────────────────

class FrameSaver:
    """
    Consumes (seq, raw_frame) tuples from an internal queue and writes them
    to disk in a dedicated daemon thread so saving never stalls the sender.
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
            pass

    def _worker(self) -> None:
        while True:
            seq, frame = self._q.get()
            try:
                ts   = time.strftime("%Y%m%d_%H%M%S")
                path = self._dir / f"frame_{ts}_{seq:06d}.png"
                cv2.imwrite(str(path), frame)
            except Exception as e:
                print(f"\n[Saver] Write error: {e}")


# ── MTI reader thread ──────────────────────────────────────────────────────────

class MtiReader(threading.Thread):
    """
    Polls ``state_json_path`` at ``MTI_POLL_HZ`` and enqueues every new sample
    (deduped by packet_counter + t_unix_ns) into a shared queue for the send loop.

    If ``save_path`` is provided, each new sample is also appended to that
    JSONL file (line-buffered, best-effort — no fsync).
    """

    def __init__(
        self,
        state_json_path: Path,
        out_queue: "queue.Queue[dict]",
        save_path: "Path | None" = None,
    ):
        super().__init__(name="mti_reader", daemon=True)
        self._state_path = state_json_path
        self._queue      = out_queue
        self._stop       = threading.Event()
        self._save_file  = None
        self._save_lock  = threading.Lock()

        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_file = open(save_path, "a", buffering=1, encoding="utf-8")
            print(f"[MtiReader] Appending MTI samples to: {save_path.resolve()}")

        self.read_count  = 0
        self.error_count = 0

    def _read_state(self):
        try:
            with open(self._state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _append_to_file(self, sample: dict) -> None:
        if self._save_file is None:
            return
        line = json.dumps(sample, separators=(",", ":"), ensure_ascii=False)
        with self._save_lock:
            try:
                self._save_file.write(line + "\n")
            except Exception as e:
                print(f"\n[MtiReader] Save error: {e}")

    def run(self) -> None:
        interval = 1.0 / MTI_POLL_HZ
        last_key = None

        while not self._stop.is_set():
            t0     = time.perf_counter()
            sample = self._read_state()

            if sample is not None:
                key = (sample.get("packet_counter"), sample.get("t_unix_ns"))
                if key != last_key:
                    last_key = key
                    self.read_count += 1
                    self._append_to_file(sample)
                    try:
                        self._queue.put_nowait(sample)
                    except queue.Full:
                        pass  # send loop can't keep up; drop rather than block
            else:
                self.error_count += 1

            wait = interval - (time.perf_counter() - t0)
            if wait > 0:
                time.sleep(wait)

    def stop(self) -> None:
        self._stop.set()
        self.join(timeout=3)
        if self._save_file is not None:
            with self._save_lock:
                try:
                    self._save_file.flush()
                    os.fsync(self._save_file.fileno())
                    self._save_file.close()
                except Exception:
                    pass


# ── Camera helpers ─────────────────────────────────────────────────────────────

def open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError("Could not open /dev/video0 via V4L2.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    return cap


def crop_and_encode(frame):
    """Centre-crop to 16:9, resize, WebP-encode.
    Returns (encoded_bytes, scaled_frame) or (None, None) on failure."""
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


# ── Connection management ──────────────────────────────────────────────────────

def make_sender(host: str) -> ObjectSender:
    return ObjectSender(host, PORT, name="drone_camera", compress=False, require_ack=False)


def connect_with_retry(host: str, sender: ObjectSender) -> None:
    attempt = 0
    while True:
        attempt += 1
        try:
            print(f"[Sender] Connecting to {host}:{PORT}  (attempt {attempt})...")
            sender.connect()
            print("[Sender] Connected.\n")
            return
        except Exception as e:
            print(f"[Sender] Connection failed: {e}  — retrying in {RECONNECT_DELAY}s...")
            time.sleep(RECONNECT_DELAY)


# ── MTI drain helper ───────────────────────────────────────────────────────────

def drain_mti_queue(mti_queue: "queue.Queue[dict]", sender: ObjectSender) -> int:
    """Flush all queued MTI samples to the ground station. Returns count sent."""
    sent = 0
    while True:
        try:
            sample = mti_queue.get_nowait()
        except queue.Empty:
            break
        sender.send_object({"msg_type": "mti_sample", **sample})
        sent += 1
    return sent


# ── Main stream loop ───────────────────────────────────────────────────────────

def stream_camera(args: argparse.Namespace) -> None:
    saver = FrameSaver(args.save_dir) if args.save_dir else None

    state_path = Path(args.mti_state_json)
    save_path  = Path(args.mti_save_dir) / "mti_samples.jsonl" if args.mti_save_dir else None
    mti_queue  = queue.Queue(maxsize=MTI_QUEUE_MAX)
    mti_reader = MtiReader(state_path, mti_queue, save_path)
    mti_reader.start()

    print("=" * 58)
    print("  Drone Camera + MTI Sender")
    print("=" * 58)
    print(f"  Ground station  : {args.host}:{PORT}")
    print(f"  Output size     : {TARGET_W}×{TARGET_H}")
    print(f"  Max capture FPS : {args.max_fps_capture}")
    print(f"  Save raw frames : {args.save_dir or 'disabled'}")
    print(f"  MTI state file  : {state_path}")
    print(f"  MTI sample log  : {save_path or 'disabled'}")
    print("=" * 58 + "\n")

    frame_interval = 1.0 / args.max_fps_capture
    cap = open_camera()
    seq = 0

    try:
        # Outer loop — reconnects whenever the connection is lost
        while True:
            sender = make_sender(args.host)
            connect_with_retry(args.host, sender)

            try:
                while True:
                    t_start = time.perf_counter()

                    # Flush any MTI samples that arrived since the last frame
                    mti_sent = drain_mti_queue(mti_queue, sender)

                    ret, raw_frame = cap.read()
                    if not ret:
                        print("\n[Sender] Camera read failed — reconnecting camera...")
                        cap.release()
                        time.sleep(1.0)
                        cap = open_camera()
                        continue

                    t_cap = time.perf_counter()

                    if saver is not None:
                        saver.submit(seq, raw_frame)

                    encoded_bytes, _ = crop_and_encode(raw_frame)
                    if encoded_bytes is None:
                        print("\n[Sender] Encode failed — skipping frame.")
                        continue

                    t_proc = time.perf_counter()

                    sender.send_object({
                        "msg_type":    "camera_frame",
                        "seq":         seq,
                        "timestamp":   time.time(),
                        "image_bytes": encoded_bytes,
                    })

                    t_send = time.perf_counter()

                    print(
                        f"\r[Sender] Seq: {seq:05d} | "
                        f"{len(encoded_bytes)/1024:5.1f} KB | "
                        f"Proc: {(t_proc-t_cap)*1000:4.1f} ms | "
                        f"Send: {(t_send-t_proc)*1000:4.1f} ms | "
                        f"Total: {(t_send-t_start)*1000:4.1f} ms | "
                        f"MTI: {mti_sent:3d}",
                        end="", flush=True,
                    )
                    seq += 1

                    sleep_time = frame_interval - (time.perf_counter() - t_start)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            except KeyboardInterrupt:
                raise

            except Exception as e:
                print(f"\n[Sender] Connection lost: {e}")
                print(f"[Sender] Retrying in {RECONNECT_DELAY}s...\n")
                try:
                    sender.close()
                except Exception:
                    pass
                time.sleep(RECONNECT_DELAY)

    except KeyboardInterrupt:
        print("\n\n[Sender] Stopped by user.")

    finally:
        cap.release()
        try:
            sender.close()
        except Exception:
            pass
        mti_reader.stop()
        print(
            f"[MtiReader] Samples read: {mti_reader.read_count} | "
            f"Errors: {mti_reader.error_count}"
        )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    stream_camera(parse_args())