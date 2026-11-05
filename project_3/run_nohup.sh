#!/usr/bin/env bash
# Launch a full Project 3 batch in the background (nohup).
# NICE runs on the CPU. CUDA_VISIBLE_DEVICES only controls GPU visibility;
# it does not enable GPU computation or bind the job to a CPU or GPU node.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -x "$SCRIPT_DIR/run_experiments.sh" ]]; then
  echo "[ERROR] run_experiments.sh is not executable; run: chmod +x run_experiments.sh" >&2
  exit 1
fi
if [[ ! -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
  echo "[ERROR] missing .venv; run: ./install_env.sh" >&2
  exit 1
fi

mkdir -p "$SCRIPT_DIR/logs"
STAMP="$(date '+%Y%m%d_%H%M%S')"
LOG="$SCRIPT_DIR/logs/nohup_full_${STAMP}.log"
PID_FILE="$SCRIPT_DIR/logs/nohup_full.pid"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "[ERROR] previous nohup still running (pid=$old_pid). Stop it first." >&2
    exit 1
  fi
fi

# Defaults for a full submission-quality batch (override via env).
export FE_LIST="${FE_LIST-20000 50000 100000}"
export SEED="${SEED-1001}"
export FUNCS="${FUNCS-1-30}"
export WORKERS="${WORKERS-16}"
export PYTHON_BIN="${PYTHON_BIN-$SCRIPT_DIR/.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"

CLEAN_FLAG=()
if [[ "${1:-}" == "--clean" ]]; then
  CLEAN_FLAG=(--clean)
fi

nohup env \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  FE_LIST="$FE_LIST" \
  SEED="$SEED" \
  FUNCS="$FUNCS" \
  WORKERS="$WORKERS" \
  PYTHON_BIN="$PYTHON_BIN" \
  /bin/bash "$SCRIPT_DIR/run_experiments.sh" "${CLEAN_FLAG[@]}" \
  >"$LOG" 2>&1 &

echo $! >"$PID_FILE"
echo "[INFO] started pid=$(cat "$PID_FILE")"
echo "[INFO] log: $LOG"
echo "[INFO] FE_LIST=$FE_LIST WORKERS=$WORKERS CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[INFO] follow: tail -f $LOG"
