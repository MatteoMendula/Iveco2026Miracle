#!/usr/bin/env python3
import argparse
import csv
import json
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import cv2
except Exception:
    cv2 = None

from mti_tool import compact_dict, iter_decoded_packets


def unix_time_ns() -> int:
    try:
        return time.time_ns()
    except AttributeError:
        return int(time.time() * 1e9)


def monotonic_time_ns() -> int:
    try:
        return time.monotonic_ns()
    except AttributeError:
        return int(time.time() * 1e9)


@dataclass
class MtiSample:
    t_unix_ns: int
    t_monotonic_ns: int
    source: str
    packet_counter: Optional[int] = None
    sample_time_fine: Optional[int] = None
    lat_deg: Optional[float] = None
    lon_deg: Optional[float] = None
    alt_ellipsoid_m: Optional[float] = None
    alt_msl_m: Optional[float] = None
    vel_x_mps: Optional[float] = None
    vel_y_mps: Optional[float] = None
    vel_z_mps: Optional[float] = None
    roll_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    yaw_deg: Optional[float] = None
    quat_w: Optional[float] = None
    quat_x: Optional[float] = None
    quat_y: Optional[float] = None
    quat_z: Optional[float] = None
    accel_x_mps2: Optional[float] = None
    accel_y_mps2: Optional[float] = None
    accel_z_mps2: Optional[float] = None
    gyro_x_rps: Optional[float] = None
    gyro_y_rps: Optional[float] = None
    gyro_z_rps: Optional[float] = None
    mag_x: Optional[float] = None
    mag_y: Optional[float] = None
    mag_z: Optional[float] = None
    status_word: Optional[int] = None
    status_word_hex: Optional[str] = None
    self_test_ok: Optional[bool] = None
    filter_valid: Optional[bool] = None
    gnss_fix: Optional[bool] = None
    filter_mode_bits: Optional[int] = None
    filter_mode_name: Optional[str] = None
    note: Optional[str] = None
    frame_hex: Optional[str] = None


class JsonlWriter:
    def __init__(self, path: Path, flush_every: int = 1):
        self.path = path
        self.flush_every = flush_every
        self.count = 0
        self.f = open(path, "a", buffering=1, encoding="utf-8")
        self.lock = threading.Lock()

    def write(self, obj: Dict[str, Any]) -> None:
        line = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
        with self.lock:
            self.f.write(line + "\n")
            self.count += 1
            if self.count % self.flush_every == 0:
                self.f.flush()
                os.fsync(self.f.fileno())

    def close(self) -> None:
        with self.lock:
            self.f.flush()
            os.fsync(self.f.fileno())
            self.f.close()


class CsvWriter:
    def __init__(self, path: Path, header: List[str]):
        self.f = open(path, "a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.f, fieldnames=header)
        if self.f.tell() == 0:
            self.writer.writeheader()
            self.f.flush()
            os.fsync(self.f.fileno())
        self.lock = threading.Lock()

    def write(self, row: Dict[str, Any]) -> None:
        with self.lock:
            self.writer.writerow(row)
            self.f.flush()
            os.fsync(self.f.fileno())

    def close(self) -> None:
        with self.lock:
            self.f.flush()
            os.fsync(self.f.fileno())
            self.f.close()


class ManifestWriter:
    def __init__(self, run_dir: Path):
        self.path = run_dir / "manifest.json"

    def write(self, data: Dict[str, Any]) -> None:
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)


class TelemetryLogger(threading.Thread):
    def __init__(
        self,
        mti_port: str,
        baud: int,
        timeout_s: float,
        out_jsonl: Path,
        state_json: Path,
        raw_jsonl: Path,
    ):
        super().__init__(daemon=True)
        self.mti_port = mti_port
        self.baud = baud
        self.timeout_s = timeout_s
        self.out = JsonlWriter(out_jsonl, flush_every=1)
        self.raw = JsonlWriter(raw_jsonl, flush_every=1)
        self.state_json = state_json
        self.stop_event = threading.Event()
        self.latest_sample: Optional[MtiSample] = None
        self.packet_count = 0
        self.good_count = 0
        self.error_count = 0

    def update_state_file(self, sample: MtiSample) -> None:
        tmp = self.state_json.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(asdict(sample), f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.state_json)

    @staticmethod
    def _sample_from_packet(pkt) -> MtiSample:
        d = pkt.decoded
        return MtiSample(
            t_unix_ns=unix_time_ns(),
            t_monotonic_ns=monotonic_time_ns(),
            source="xbus",
            packet_counter=d.get("packet_counter"),
            sample_time_fine=d.get("sample_time_fine"),
            lat_deg=d.get("lat_deg"),
            lon_deg=d.get("lon_deg"),
            alt_ellipsoid_m=d.get("alt_ellipsoid_m"),
            alt_msl_m=d.get("alt_msl_m"),
            vel_x_mps=d.get("vel_x_mps"),
            vel_y_mps=d.get("vel_y_mps"),
            vel_z_mps=d.get("vel_z_mps"),
            roll_deg=d.get("roll_deg"),
            pitch_deg=d.get("pitch_deg"),
            yaw_deg=d.get("yaw_deg"),
            quat_w=d.get("quat_w"),
            quat_x=d.get("quat_x"),
            quat_y=d.get("quat_y"),
            quat_z=d.get("quat_z"),
            accel_x_mps2=d.get("accel_x_mps2"),
            accel_y_mps2=d.get("accel_y_mps2"),
            accel_z_mps2=d.get("accel_z_mps2"),
            gyro_x_rps=d.get("gyro_x_rps"),
            gyro_y_rps=d.get("gyro_y_rps"),
            gyro_z_rps=d.get("gyro_z_rps"),
            mag_x=d.get("mag_x"),
            mag_y=d.get("mag_y"),
            mag_z=d.get("mag_z"),
            status_word=d.get("status_word"),
            status_word_hex=d.get("status_word_hex"),
            self_test_ok=d.get("self_test_ok"),
            filter_valid=d.get("filter_valid"),
            gnss_fix=d.get("gnss_fix"),
            filter_mode_bits=d.get("filter_mode_bits"),
            filter_mode_name=d.get("filter_mode_name"),
            note=pkt.mid_name,
            frame_hex=pkt.frame_hex,
        )

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                for pkt in iter_decoded_packets(self.mti_port, self.baud, self.timeout_s):
                    if self.stop_event.is_set():
                        break
                    self.packet_count += 1
                    self.raw.write(
                        {
                            "t_unix_ns": unix_time_ns(),
                            "t_monotonic_ns": monotonic_time_ns(),
                            "bus_id": pkt.bus_id,
                            "mid": pkt.mid,
                            "mid_name": pkt.mid_name,
                            "payload_len": pkt.payload_len,
                            "frame_hex": pkt.frame_hex,
                            "decoded": compact_dict(pkt.decoded),
                        }
                    )
                    sample = self._sample_from_packet(pkt)
                    self.good_count += 1
                    self.latest_sample = sample
                    self.out.write(asdict(sample))
                    self.update_state_file(sample)
            except Exception as e:
                self.error_count += 1
                self.out.write(
                    {
                        "t_unix_ns": unix_time_ns(),
                        "t_monotonic_ns": monotonic_time_ns(),
                        "event": "mti_error",
                        "error": repr(e),
                    }
                )
                time.sleep(1.0)

    def stop(self) -> None:
        self.stop_event.set()
        self.join(timeout=5)
        self.out.close()
        self.raw.close()


class CameraLogger(threading.Thread):
    def __init__(self, device: str, frames_dir: Path, index_csv: Path, width: int, height: int, fps: float, jpeg_quality: int = 92):
        super().__init__(daemon=True)
        self.device = device
        self.frames_dir = frames_dir
        self.csv = CsvWriter(index_csv, ["frame_idx", "t_unix_ns", "t_monotonic_ns", "path", "width", "height"])
        self.width = width
        self.height = height
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self.stop_event = threading.Event()
        self.frame_idx = 0
        self.last_ok = False

    def run(self) -> None:
        if cv2 is None:
            raise RuntimeError("opencv-python is not installed. Install with: pip install opencv-python")

        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        while not self.stop_event.is_set():
            if not cap.isOpened():
                time.sleep(1.0)
                cap.release()
                cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                continue

            ok, frame = cap.read()
            if not ok or frame is None:
                self.last_ok = False
                time.sleep(0.1)
                continue

            self.last_ok = True
            t_unix_ns = unix_time_ns()
            t_mono_ns = monotonic_time_ns()
            rel = f"{self.frame_idx:08d}_{t_unix_ns}.jpg"
            out_path = self.frames_dir / rel
            tmp_path = out_path.with_name(out_path.name + ".tmp.jpg")
            cv2.imwrite(str(tmp_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            os.replace(tmp_path, out_path)
            self.csv.write(
                {
                    "frame_idx": self.frame_idx,
                    "t_unix_ns": t_unix_ns,
                    "t_monotonic_ns": t_mono_ns,
                    "path": str(Path("camera") / "frames" / rel),
                    "width": frame.shape[1],
                    "height": frame.shape[0],
                }
            )
            self.frame_idx += 1

        cap.release()

    def stop(self) -> None:
        self.stop_event.set()
        self.join(timeout=5)
        self.csv.close()


class NoopCameraLogger(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop_event = threading.Event()
        self.last_ok = True

    def run(self) -> None:
        while not self.stop_event.is_set():
            time.sleep(0.5)

    def stop(self) -> None:
        self.stop_event.set()
        self.join(timeout=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="Root payload data directory.")
    p.add_argument("--run-name", default=None)
    p.add_argument("--mti-port", default="/dev/ttyUSB0")
    p.add_argument("--mti-baud", type=int, default=115200)
    p.add_argument("--mti-timeout", type=float, default=0.2)
    p.add_argument("--camera-device", default="/dev/video0")
    p.add_argument("--camera-width", type=int, default=1280)
    p.add_argument("--camera-height", type=int, default=720)
    p.add_argument("--camera-fps", type=float, default=10.0)
    p.add_argument("--jpeg-quality", type=int, default=92)
    p.add_argument("--disable-camera", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_name = args.run_name or time.strftime("payload_%Y%m%d_%H%M%S")
    root = Path(args.root)
    run_dir = root / run_name
    telemetry_dir = run_dir / "telemetry"
    camera_dir = run_dir / "camera"
    frames_dir = camera_dir / "frames"
    logs_dir = run_dir / "logs"
    for d in [telemetry_dir, frames_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    manifest = ManifestWriter(run_dir)
    manifest.write(
        {
            "run_name": run_name,
            "created_unix_ns": unix_time_ns(),
            "created_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "paths": {
                "telemetry_jsonl": str(Path("telemetry") / "telemetry.jsonl"),
                "telemetry_state_json": str(Path("telemetry") / "state_latest.json"),
                "telemetry_raw_jsonl": str(Path("telemetry") / "raw_packets.jsonl"),
                "camera_index_csv": str(Path("camera") / "frames.csv"),
                "camera_frames_dir": str(Path("camera") / "frames"),
                "supervisor_log": str(Path("logs") / "supervisor.log"),
            },
            "config": vars(args),
        }
    )

    supervisor_log = JsonlWriter(logs_dir / "supervisor.log", flush_every=1)
    telemetry = TelemetryLogger(
        mti_port=args.mti_port,
        baud=args.mti_baud,
        timeout_s=args.mti_timeout,
        out_jsonl=telemetry_dir / "telemetry.jsonl",
        state_json=telemetry_dir / "state_latest.json",
        raw_jsonl=telemetry_dir / "raw_packets.jsonl",
    )

    if args.disable_camera:
        camera = NoopCameraLogger()
    else:
        camera = CameraLogger(
            device=args.camera_device,
            frames_dir=frames_dir,
            index_csv=camera_dir / "frames.csv",
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            jpeg_quality=args.jpeg_quality,
        )

    stop_flag = threading.Event()

    def handle_stop(signum, _frame):
        supervisor_log.write({"t_unix_ns": unix_time_ns(), "event": "signal", "signum": signum})
        stop_flag.set()

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    supervisor_log.write({"t_unix_ns": unix_time_ns(), "event": "startup"})
    telemetry.start()
    camera.start()

    try:
        while not stop_flag.is_set():
            if not telemetry.is_alive():
                supervisor_log.write({"t_unix_ns": unix_time_ns(), "event": "telemetry_thread_exit"})
                stop_flag.set()
                break
            if not camera.is_alive():
                supervisor_log.write({"t_unix_ns": unix_time_ns(), "event": "camera_thread_exit"})
                stop_flag.set()
                break
            time.sleep(0.5)
    finally:
        supervisor_log.write({"t_unix_ns": unix_time_ns(), "event": "shutdown_begin"})
        camera.stop()
        telemetry.stop()
        supervisor_log.write({"t_unix_ns": unix_time_ns(), "event": "shutdown_end"})
        supervisor_log.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
