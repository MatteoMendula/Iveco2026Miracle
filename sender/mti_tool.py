#!/usr/bin/env python3
import argparse
import json
import math
import struct
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Generator, List, Tuple

import serial

PREAMBLE = 0xFA
MID_MT_DATA2 = 0x36


class XbusError(Exception):
    pass


class XbusFrameParser:
    def __init__(self) -> None:
        self.buf = bytearray()

    @staticmethod
    def checksum_ok(frame: bytes) -> bool:
        return (sum(frame[1:]) & 0xFF) == 0

    def feed(self, chunk: bytes) -> List[bytes]:
        self.buf.extend(chunk)
        out: List[bytes] = []
        while True:
            if len(self.buf) < 5:
                break
            try:
                start = self.buf.index(PREAMBLE)
            except ValueError:
                self.buf.clear()
                break
            if start > 0:
                del self.buf[:start]
            if len(self.buf) < 5:
                break

            if self.buf[3] != 0xFF:
                payload_len = self.buf[3]
                total_len = 5 + payload_len
            else:
                if len(self.buf) < 7:
                    break
                payload_len = (self.buf[4] << 8) | self.buf[5]
                total_len = 7 + payload_len

            if len(self.buf) < total_len:
                break

            frame = bytes(self.buf[:total_len])
            del self.buf[:total_len]
            if self.checksum_ok(frame):
                out.append(frame)
            else:
                if self.buf:
                    continue
        return out


@dataclass
class DecodedPacket:
    t_wall: str
    bus_id: int
    mid: int
    mid_name: str
    payload_len: int
    frame_hex: str
    decoded: Dict[str, Any]


def _read_be_u16(b: bytes) -> int:
    return int.from_bytes(b, "big", signed=False)


def _read_be_u32(b: bytes) -> int:
    return int.from_bytes(b, "big", signed=False)


def _read_be_f32x3(b: bytes) -> Tuple[float, float, float]:
    return struct.unpack(">3f", b)


def _read_be_f32x4(b: bytes) -> Tuple[float, float, float, float]:
    return struct.unpack(">4f", b)


def _read_xsens_fp1632_6(b: bytes) -> float:
    """Decode Xsens 16.32 fixed-point transmitted in 6 bytes.

    Xsens does not send these 6 bytes as a normal big-endian signed 48-bit integer.
    The transmitted byte order is the 32-bit fractional part first, followed by the
    low 16 bits of the integer part, i.e. [b3 b2 b1 b0 b5 b4] from the underlying
    64-bit fixed-point number.
    """
    if len(b) != 6:
        raise ValueError(f"expected 6 bytes for fp16.32, got {len(b)}")
    frac_u32 = _read_be_u32(b[0:4])
    int_low_u16 = _read_be_u16(b[4:6])
    raw48 = (int_low_u16 << 32) | frac_u32
    if raw48 & (1 << 47):
        raw48 -= 1 << 48
    return raw48 / float(1 << 32)


def quaternion_to_euler_deg(w: float, x: float, y: float, z: float) -> Tuple[float, float, float]:
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.degrees(math.copysign(math.pi / 2.0, sinp))
    else:
        pitch = math.degrees(math.asin(sinp))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))
    return roll, pitch, yaw


def decode_status_word(sw: int) -> Dict[str, Any]:
    mode_bits = (sw >> 23) & 0x7
    filter_mode_name = {
        0: "No filter",
        1: "Orientation",
        2: "Inertial",
        3: "With GNSS",
    }.get(mode_bits, f"Unknown({mode_bits})")
    return {
        "status_word": sw,
        "status_word_hex": f"0x{sw:08X}",
        "self_test_ok": bool(sw & (1 << 0)),
        "filter_valid": bool(sw & (1 << 1)),
        "gnss_fix": bool(sw & (1 << 2)),
        "no_rotation_update": bool(sw & (1 << 8)),
        "clip_acc": bool(sw & (1 << 19)),
        "clip_gyr": bool(sw & (1 << 20)),
        "clip_mag": bool(sw & (1 << 21)),
        "filter_mode_bits": mode_bits,
        "filter_mode_name": filter_mode_name,
    }


def decode_mtdata2_item(data_id: int, payload: bytes) -> Dict[str, Any]:
    if data_id == 0x1020 and len(payload) == 2:
        return {"packet_counter": _read_be_u16(payload)}

    if data_id == 0x1060 and len(payload) == 4:
        return {"sample_time_fine": _read_be_u32(payload)}

    if data_id == 0x2010 and len(payload) == 16:
        q_w, q_x, q_y, q_z = _read_be_f32x4(payload)
        roll_deg, pitch_deg, yaw_deg = quaternion_to_euler_deg(q_w, q_x, q_y, q_z)
        return {
            "quat_w": q_w,
            "quat_x": q_x,
            "quat_y": q_y,
            "quat_z": q_z,
            "roll_deg": roll_deg,
            "pitch_deg": pitch_deg,
            "yaw_deg": yaw_deg,
        }

    if data_id == 0x4020 and len(payload) == 12:
        ax, ay, az = _read_be_f32x3(payload)
        return {"accel_x_mps2": ax, "accel_y_mps2": ay, "accel_z_mps2": az}

    if data_id == 0x8020 and len(payload) == 12:
        gx, gy, gz = _read_be_f32x3(payload)
        return {"gyro_x_rps": gx, "gyro_y_rps": gy, "gyro_z_rps": gz}

    if data_id == 0xC020 and len(payload) == 12:
        mx, my, mz = _read_be_f32x3(payload)
        return {"mag_x": mx, "mag_y": my, "mag_z": mz}

    if data_id == 0xE020 and len(payload) == 4:
        return decode_status_word(_read_be_u32(payload))

    if data_id == 0x5042 and len(payload) == 12:
        lat = _read_xsens_fp1632_6(payload[0:6])
        lon = _read_xsens_fp1632_6(payload[6:12])
        return {"lat_deg": lat, "lon_deg": lon}

    if data_id == 0x5022 and len(payload) == 6:
        return {"alt_ellipsoid_m": _read_xsens_fp1632_6(payload[0:6])}

    if data_id == 0xD012 and len(payload) == 18:
        vx = _read_xsens_fp1632_6(payload[0:6])
        vy = _read_xsens_fp1632_6(payload[6:12])
        vz = _read_xsens_fp1632_6(payload[12:18])
        return {"vel_x_mps": vx, "vel_y_mps": vy, "vel_z_mps": vz}

    return {f"raw_0x{data_id:04X}": payload.hex()}


def parse_mtdata2_payload(payload: bytes) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    items: List[Dict[str, Any]] = []
    i = 0
    while i < len(payload):
        if i + 3 > len(payload):
            out["parse_warning"] = "truncated_item_header"
            break
        data_id = _read_be_u16(payload[i : i + 2])
        size = payload[i + 2]
        start = i + 3
        end = start + size
        if end > len(payload):
            out["parse_warning"] = "truncated_item_payload"
            break
        decoded = decode_mtdata2_item(data_id, payload[start:end])
        items.append({"data_id": f"0x{data_id:04X}", "size": size, "decoded": decoded})
        out.update(decoded)
        i = end
    out["items"] = items
    return out


def frame_payload(frame: bytes) -> bytes:
    if len(frame) < 5 or frame[0] != PREAMBLE:
        raise XbusError("not an Xbus frame")
    if frame[3] != 0xFF:
        return frame[4:-1]
    return frame[6:-1]


def frame_name(mid: int) -> str:
    if mid == MID_MT_DATA2:
        return "MTData2"
    return f"MID_0x{mid:02X}"


def decode_frame(frame: bytes) -> DecodedPacket:
    bus_id = frame[1]
    mid = frame[2]
    payload = frame_payload(frame)
    decoded: Dict[str, Any] = {}
    if mid == MID_MT_DATA2:
        decoded = parse_mtdata2_payload(payload)
    return DecodedPacket(
        t_wall=time.strftime("%Y-%m-%d %H:%M:%S"),
        bus_id=bus_id,
        mid=mid,
        mid_name=frame_name(mid),
        payload_len=len(payload),
        frame_hex=frame.hex(),
        decoded=decoded,
    )


def open_serial(port: str, baud: int, timeout: float) -> serial.Serial:
    return serial.Serial(port, baud, timeout=timeout)


def iter_decoded_packets(port: str, baud: int, timeout: float) -> Generator[DecodedPacket, None, None]:
    parser = XbusFrameParser()
    with open_serial(port, baud, timeout) as ser:
        while True:
            chunk = ser.read(4096)
            if not chunk:
                continue
            for frame in parser.feed(chunk):
                yield decode_frame(frame)


def compact_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if k != "items"}


def _cmd_stream(args: argparse.Namespace) -> int:
    count = 0
    for pkt in iter_decoded_packets(args.port, args.baud, args.timeout):
        info = asdict(pkt)
        if args.json:
            print(json.dumps(info, separators=(",", ":"), ensure_ascii=False))
        else:
            dec = compact_dict(pkt.decoded)
            print(
                f"[{pkt.t_wall}] bus=0x{pkt.bus_id:02X} mid=0x{pkt.mid:02X} "
                f"name={pkt.mid_name} payload_len={pkt.payload_len} decoded={dec}"
            )
        count += 1
        if args.limit and count >= args.limit:
            return 0
    return 0


def _cmd_raw(args: argparse.Namespace) -> int:
    count = 0
    parser = XbusFrameParser()
    with open_serial(args.port, args.baud, args.timeout) as ser:
        while True:
            chunk = ser.read(4096)
            if not chunk:
                continue
            for frame in parser.feed(chunk):
                pkt = decode_frame(frame)
                print(
                    f"[{pkt.t_wall}] bus=0x{pkt.bus_id:02X} mid=0x{pkt.mid:02X} "
                    f"name={pkt.mid_name} payload_len={pkt.payload_len} frame={pkt.frame_hex}"
                )
                count += 1
                if args.limit and count >= args.limit:
                    return 0
    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MTi serial/Xbus utility")
    sub = p.add_subparsers(dest="cmd", required=False)

    for name in ("stream", "raw"):
        sp = sub.add_parser(name)
        sp.add_argument("--port", default="/dev/ttyUSB0")
        sp.add_argument("--baud", type=int, default=115200)
        sp.add_argument("--timeout", type=float, default=0.2)
        sp.add_argument("--limit", type=int, default=0)
        if name == "stream":
            sp.add_argument("--json", action="store_true")

    p.set_defaults(cmd="stream")
    return p


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()
    if args.cmd == "raw":
        return _cmd_raw(args)
    return _cmd_stream(args)


if __name__ == "__main__":
    raise SystemExit(main())
