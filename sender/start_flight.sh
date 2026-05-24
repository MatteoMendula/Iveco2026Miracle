#!/usr/bin/env bash
# ── start_flight.sh ────────────────────────────────────────────────────────────
# Launches collect_payload_supervisor.py and sender.py as background processes
# (immune to SSH disconnection via nohup).  Both share the same run directory.
#
# Usage:
#   ./start_flight.sh [RUN_NAME]
#
#   RUN_NAME  Optional. Defaults to flight_YYYYMMDD_HHMMSS.
#
# Logs:
#   <DATA_ROOT>/<RUN_NAME>/logs/supervisor.log   (supervisor internal JSONL)
#   <DATA_ROOT>/<RUN_NAME>/logs/supervisor.out   (stdout/stderr)
#   <DATA_ROOT>/<RUN_NAME>/logs/sender.out       (stdout/stderr)
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Edit these to match your setup ────────────────────────────────────────────
DATA_ROOT="/data"
GROUND_STATION_HOST="100.100.100.100"
MTI_PORT="/dev/ttyUSB0"
MTI_BAUD=115200
CAMERA_DEVICE="/dev/video0"
CAMERA_FPS=10
MAX_FPS_CAPTURE=5
JPEG_QUALITY=92
# ──────────────────────────────────────────────────────────────────────────────

RUN_NAME="${1:-flight_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${DATA_ROOT}/${RUN_NAME}"
LOGS_DIR="${RUN_DIR}/logs"

mkdir -p "${LOGS_DIR}"

echo "========================================"
echo "  Flight session: ${RUN_NAME}"
echo "  Run directory : ${RUN_DIR}"
echo "  Ground station: ${GROUND_STATION_HOST}"
echo "========================================"

# ── 1. Supervisor (MTI + camera raw capture) ───────────────────────────────────
nohup python -u collect_payload_supervisor.py \
    --root        "${DATA_ROOT}"           \
    --run-name    "${RUN_NAME}"            \
    --mti-port    "${MTI_PORT}"            \
    --mti-baud    "${MTI_BAUD}"            \
    --camera-device "${CAMERA_DEVICE}"     \
    --camera-fps  "${CAMERA_FPS}"          \
    --jpeg-quality "${JPEG_QUALITY}"       \
    > "${LOGS_DIR}/supervisor.out" 2>&1 &

SUPERVISOR_PID=$!
echo "[start_flight] supervisor PID: ${SUPERVISOR_PID}"

# Give the supervisor a moment to create state_latest.json before sender starts
sleep 2

# ── 2. Sender (camera stream + MTI relay to ground station) ───────────────────
nohup python -u sender.py \
    --host              "${GROUND_STATION_HOST}"                          \
    --max-fps-capture   "${MAX_FPS_CAPTURE}"                             \
    --save-dir          "${RUN_DIR}/camera/frames_sender"                \
    --mti-state-json    "${RUN_DIR}/telemetry/state_latest.json"         \
    --mti-save-dir      "${RUN_DIR}/telemetry"                           \
    > "${LOGS_DIR}/sender.out" 2>&1 &

SENDER_PID=$!
echo "[start_flight] sender     PID: ${SENDER_PID}"

# ── Save PIDs for easy shutdown ────────────────────────────────────────────────
echo "${SUPERVISOR_PID}" > "${LOGS_DIR}/supervisor.pid"
echo "${SENDER_PID}"     > "${LOGS_DIR}/sender.pid"

echo ""
echo "Both processes running in background."
echo "To stop:  ./stop_flight.sh ${RUN_NAME}"
echo "Logs:     tail -f ${LOGS_DIR}/sender.out"
echo "          tail -f ${LOGS_DIR}/supervisor.out"