#!/usr/bin/env bash
# ── stop_flight.sh ─────────────────────────────────────────────────────────────
# Gracefully stops supervisor and sender launched by start_flight.sh.
#
# Usage:
#   ./stop_flight.sh [RUN_NAME]
#
#   RUN_NAME  Must match the name used with start_flight.sh.
#             If omitted, stops the most recently started session.
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

DATA_ROOT="/data"

if [[ $# -ge 1 ]]; then
    RUN_NAME="$1"
else
    # Find the most recently modified .pid file under DATA_ROOT
    LATEST=$(find "${DATA_ROOT}" -name "supervisor.pid" -printf "%T@ %p\n" 2>/dev/null \
             | sort -n | tail -1 | awk '{print $2}')
    if [[ -z "${LATEST}" ]]; then
        echo "No running flight session found under ${DATA_ROOT}."
        exit 1
    fi
    RUN_NAME=$(echo "${LATEST}" | sed "s|${DATA_ROOT}/||;s|/logs/supervisor.pid||")
fi

LOGS_DIR="${DATA_ROOT}/${RUN_NAME}/logs"

stop_pid() {
    local name="$1"
    local pidfile="${LOGS_DIR}/${name}.pid"
    if [[ ! -f "${pidfile}" ]]; then
        echo "[stop_flight] No PID file for ${name} — already stopped?"
        return
    fi
    local pid
    pid=$(<"${pidfile}")
    if kill -0 "${pid}" 2>/dev/null; then
        echo "[stop_flight] Stopping ${name} (PID ${pid})..."
        kill -SIGTERM "${pid}"
        # Wait up to 5 s for clean exit, then SIGKILL
        for _ in $(seq 1 10); do
            sleep 0.5
            kill -0 "${pid}" 2>/dev/null || break
        done
        if kill -0 "${pid}" 2>/dev/null; then
            echo "[stop_flight] ${name} did not exit — sending SIGKILL"
            kill -SIGKILL "${pid}"
        else
            echo "[stop_flight] ${name} stopped cleanly."
        fi
    else
        echo "[stop_flight] ${name} (PID ${pid}) is not running."
    fi
    rm -f "${pidfile}"
}

echo "Stopping flight session: ${RUN_NAME}"
stop_pid sender
stop_pid supervisor
echo "Done."