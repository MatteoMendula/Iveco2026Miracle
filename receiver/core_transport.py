#!/usr/bin/env python3
"""
core_transport.py

Simple reusable TCP transport layer for sending Python objects between machines.

Supports:
  - dicts
  - lists
  - strings
  - ints/floats/bools/None
  - bytes
  - NumPy arrays
  - PyTorch tensors, if torch is installed
  - organized MTi-7 packets as dictionaries/lists/bytes

Protocol:
  1. TCP connect
  2. HELLO handshake
  3. length-prefixed framed messages
  4. pickle serialization
  5. optional zlib compression
  6. object returned to higher-level code

Security note:
  This uses pickle. Only use this on trusted links/devices.
"""

import argparse
import json
import pickle
import socket
import struct
import time
import zlib
from dataclasses import dataclass
from typing import Any, Optional, Tuple


MAGIC = b"PTX1"
VERSION = 1

# 4-byte unsigned header length
HEADER_LEN_STRUCT = struct.Struct("!I")


@dataclass
class TransportStats:
    messages: int = 0
    raw_payload_bytes: int = 0
    wire_payload_bytes: int = 0
    start_time: float = 0.0
    end_time: float = 0.0

    def reset(self) -> None:
        self.messages = 0
        self.raw_payload_bytes = 0
        self.wire_payload_bytes = 0
        self.start_time = time.perf_counter()
        self.end_time = self.start_time

    def update(self, raw_bytes: int, wire_bytes: int) -> None:
        if self.messages == 0:
            self.start_time = time.perf_counter()
        self.messages += 1
        self.raw_payload_bytes += raw_bytes
        self.wire_payload_bytes += wire_bytes
        self.end_time = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return max(self.end_time - self.start_time, 1e-9)

    @property
    def raw_mbps(self) -> float:
        return (self.raw_payload_bytes * 8.0) / self.elapsed / 1e6

    @property
    def wire_mbps(self) -> float:
        return (self.wire_payload_bytes * 8.0) / self.elapsed / 1e6


def _recvall(sock: socket.socket, n: int) -> bytes:
    """
    Receive exactly n bytes or raise ConnectionError.
    """
    chunks = []
    remaining = n

    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Socket closed while receiving data.")
        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)


def _send_json_frame(sock: socket.socket, obj: dict) -> None:
    data = json.dumps(obj).encode("utf-8")
    sock.sendall(HEADER_LEN_STRUCT.pack(len(data)))
    sock.sendall(data)


def _recv_json_frame(sock: socket.socket) -> dict:
    size_bytes = _recvall(sock, HEADER_LEN_STRUCT.size)
    size = HEADER_LEN_STRUCT.unpack(size_bytes)[0]
    data = _recvall(sock, size)
    return json.loads(data.decode("utf-8"))


def _serialize_object(obj: Any, compress: bool) -> Tuple[dict, bytes, int]:
    """
    Serialize object and return:
      header dict, wire payload, raw serialized payload size
    """
    raw_payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    raw_size = len(raw_payload)

    if compress:
        wire_payload = zlib.compress(raw_payload, level=1)
        compression = "zlib"
    else:
        wire_payload = raw_payload
        compression = "none"

    header = {
        "magic": MAGIC.decode("ascii"),
        "version": VERSION,
        "kind": "object",
        "serializer": "pickle",
        "compression": compression,
        "raw_payload_bytes": raw_size,
        "wire_payload_bytes": len(wire_payload),
        "timestamp": time.time(),
    }

    return header, wire_payload, raw_size


def _deserialize_object(header: dict, wire_payload: bytes) -> Any:
    if header.get("magic") != MAGIC.decode("ascii"):
        raise ValueError(f"Bad magic value: {header.get('magic')}")

    if header.get("version") != VERSION:
        raise ValueError(f"Unsupported protocol version: {header.get('version')}")

    compression = header.get("compression", "none")

    if compression == "zlib":
        raw_payload = zlib.decompress(wire_payload)
    elif compression == "none":
        raw_payload = wire_payload
    else:
        raise ValueError(f"Unsupported compression type: {compression}")

    return pickle.loads(raw_payload)


class ObjectSender:
    """
    Generic object sender.

    Example:
        sender = ObjectSender("192.168.168.19", 50010)
        sender.connect()
        sender.send_object({"hello": "world"})
        sender.close()
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        name: str = "sender",
        timeout: float = 10.0,
        compress: bool = False,
        require_ack: bool = True,
    ):
        self.host = host
        self.port = port
        self.name = name
        self.timeout = timeout
        self.compress = compress
        self.require_ack = require_ack
        self.sock: Optional[socket.socket] = None
        self.stats = TransportStats()

    def connect(self) -> None:
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self.sock.settimeout(self.timeout)

        hello = {
            "magic": MAGIC.decode("ascii"),
            "version": VERSION,
            "kind": "hello",
            "role": "sender",
            "name": self.name,
            "time": time.time(),
        }

        _send_json_frame(self.sock, hello)
        reply = _recv_json_frame(self.sock)

        if reply.get("kind") != "hello_ack":
            raise ConnectionError(f"Handshake failed. Got reply: {reply}")

        if reply.get("magic") != MAGIC.decode("ascii"):
            raise ConnectionError(f"Handshake failed. Bad magic: {reply}")

    def send_object(self, obj: Any) -> Optional[dict]:
        if self.sock is None:
            raise RuntimeError("Sender is not connected. Call connect() first.")

        header, wire_payload, raw_size = _serialize_object(obj, self.compress)

        _send_json_frame(self.sock, header)
        self.sock.sendall(wire_payload)

        self.stats.update(raw_bytes=raw_size, wire_bytes=len(wire_payload))

        if self.require_ack:
            ack = _recv_json_frame(self.sock)
            if ack.get("kind") != "ack":
                raise ConnectionError(f"Expected ACK, got: {ack}")
            return ack

        return None

    def send_close(self) -> None:
        if self.sock is None:
            return

        close_msg = {
            "magic": MAGIC.decode("ascii"),
            "version": VERSION,
            "kind": "close",
            "time": time.time(),
        }
        _send_json_frame(self.sock, close_msg)

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.send_close()
            except Exception:
                pass

            try:
                self.sock.close()
            except Exception:
                pass

            self.sock = None


class ObjectReceiver:
    """
    Generic object receiver.

    Example:
        receiver = ObjectReceiver("0.0.0.0", 50010)
        receiver.start()
        obj = receiver.receive_object()
    """

    def __init__(
        self,
        bind_ip: str,
        port: int,
        *,
        name: str = "receiver",
        timeout: Optional[float] = None,
        backlog: int = 1,
        send_ack: bool = True,
    ):
        self.bind_ip = bind_ip
        self.port = port
        self.name = name
        self.timeout = timeout
        self.backlog = backlog
        self.send_ack = send_ack

        self.server_sock: Optional[socket.socket] = None
        self.client_sock: Optional[socket.socket] = None
        self.client_addr = None
        self.stats = TransportStats()

    def start(self) -> None:
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((self.bind_ip, self.port))
        self.server_sock.listen(self.backlog)

        print(f"[receiver] Listening on {self.bind_ip}:{self.port}")

        self.client_sock, self.client_addr = self.server_sock.accept()
        self.client_sock.settimeout(self.timeout)

        print(f"[receiver] Connection from {self.client_addr}")

        hello = _recv_json_frame(self.client_sock)

        if hello.get("kind") != "hello":
            raise ConnectionError(f"Expected HELLO, got: {hello}")

        if hello.get("magic") != MAGIC.decode("ascii"):
            raise ConnectionError(f"Bad magic in HELLO: {hello}")

        reply = {
            "magic": MAGIC.decode("ascii"),
            "version": VERSION,
            "kind": "hello_ack",
            "role": "receiver",
            "name": self.name,
            "time": time.time(),
        }

        _send_json_frame(self.client_sock, reply)
        print("[receiver] Handshake complete")

    def receive_object(self) -> Optional[Any]:
        """
        Receive one object.

        Returns:
          - Python object for normal messages
          - None when sender sends close
        """
        if self.client_sock is None:
            raise RuntimeError("Receiver is not connected. Call start() first.")

        header = _recv_json_frame(self.client_sock)
        kind = header.get("kind")

        if kind == "close":
            return None

        if kind != "object":
            raise ConnectionError(f"Expected object frame, got: {header}")

        wire_size = int(header["wire_payload_bytes"])
        raw_size = int(header["raw_payload_bytes"])

        wire_payload = _recvall(self.client_sock, wire_size)
        obj = _deserialize_object(header, wire_payload)

        self.stats.update(raw_bytes=raw_size, wire_bytes=wire_size)

        if self.send_ack:
            ack = {
                "magic": MAGIC.decode("ascii"),
                "version": VERSION,
                "kind": "ack",
                "received_wire_payload_bytes": wire_size,
                "received_raw_payload_bytes": raw_size,
                "receiver_time": time.time(),
            }
            _send_json_frame(self.client_sock, ack)

        return obj

    def close(self) -> None:
        for s in [self.client_sock, self.server_sock]:
            if s is not None:
                try:
                    s.close()
                except Exception:
                    pass

        self.client_sock = None
        self.server_sock = None


def demo_receiver(bind_ip: str, port: int) -> None:
    receiver = ObjectReceiver(bind_ip, port)
    receiver.start()

    try:
        while True:
            obj = receiver.receive_object()
            if obj is None:
                print("[receiver] Sender closed connection.")
                break

            print("[receiver] Got object:")
            print(obj)

    finally:
        receiver.close()


def demo_sender(host: str, port: int) -> None:
    sender = ObjectSender(host, port, compress=False)
    sender.connect()

    try:
        test_packet = {
            "type": "mti7_packet",
            "sample_time_fine": 123456789,
            "filter_valid": True,
            "gnss_fix": True,
            "quat": {
                "w": 0.999,
                "x": 0.001,
                "y": 0.002,
                "z": 0.003,
            },
            "euler_deg": {
                "roll": 0.4,
                "pitch": 0.86,
                "yaw": 43.4,
            },
            "position": {
                "lat": 33.0,
                "lon": -117.0,
                "alt": 100.0,
            },
            "raw_bytes": b"\x01\x02\x03\x04",
        }

        ack = sender.send_object(test_packet)
        print("[sender] ACK:", ack)

    finally:
        sender.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    rx = sub.add_parser("recv")
    rx.add_argument("--bind-ip", default="0.0.0.0")
    rx.add_argument("--port", type=int, default=50010)

    tx = sub.add_parser("send")
    tx.add_argument("--host", required=True)
    tx.add_argument("--port", type=int, default=50010)

    args = parser.parse_args()

    if args.mode == "recv":
        demo_receiver(args.bind_ip, args.port)
    elif args.mode == "send":
        demo_sender(args.host, args.port)


if __name__ == "__main__":
    main()