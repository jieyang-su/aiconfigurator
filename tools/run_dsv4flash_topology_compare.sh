#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-start}"

SYSTEMS_PATH="${SYSTEMS_PATH:-$PWD/src/aiconfigurator/systems}"
MODEL="${MODEL:-sgl-project/DeepSeek-V4-Flash-FP8}"
BACKEND="${BACKEND:-sglang}"
BACKEND_VERSION="${BACKEND_VERSION:-0.5.10.post2}"
TOTAL_GPUS="${TOTAL_GPUS:-32}"
ISL="${ISL:-2048}"
OSL="${OSL:-500}"
GEMM_QUANT="${GEMM_QUANT:-fp8_block}"
DATABASE_MODE="${DATABASE_MODE:-HYBRID}"
TTFT="${TTFT:-10000}"
TPOT="${TPOT:-5000}"
OUT_DIR="${OUT_DIR:-results/dsv4flash_topology_compare}"
mkdir -p "$OUT_DIR"

CMD_LOG="$OUT_DIR/commands.sh"
PID_FILE="$OUT_DIR/pids.env"

run_case() {
  local system="$1"
  local log_file="$2"
  local pid_var="$3"
  cat <<CMD | tee -a "$CMD_LOG"
nohup aiconfigurator cli default \\
  --systems-paths "$SYSTEMS_PATH" \\
  --model "$MODEL" \\
  --total-gpus "$TOTAL_GPUS" \\
  --system "$system" \\
  --backend "$BACKEND" \\
  --backend-version "$BACKEND_VERSION" \\
  --isl "$ISL" \\
  --osl "$OSL" \\
  --gemm-quant-mode "$GEMM_QUANT" \\
  --database-mode "$DATABASE_MODE" \\
  --ttft "$TTFT" \\
  --tpot "$TPOT" \\
  > "$log_file" 2>&1 &
CMD
  nohup aiconfigurator cli default \
    --systems-paths "$SYSTEMS_PATH" \
    --model "$MODEL" \
    --total-gpus "$TOTAL_GPUS" \
    --system "$system" \
    --backend "$BACKEND" \
    --backend-version "$BACKEND_VERSION" \
    --isl "$ISL" \
    --osl "$OSL" \
    --gemm-quant-mode "$GEMM_QUANT" \
    --database-mode "$DATABASE_MODE" \
    --ttft "$TTFT" \
    --tpot "$TPOT" \
    > "$log_file" 2>&1 &
  local pid=$!
  echo "$pid_var=$pid" >> "$PID_FILE"
}

start_runs() {
  : > "$CMD_LOG"
  : > "$PID_FILE"
  run_case "rtx_pro_6000_scaleup_32" "$OUT_DIR/output_scaleup_32.log" "SCALEUP_PID"
  run_case "rtx_pro_6000_scaleout_2x16" "$OUT_DIR/output_scaleout_2x16.log" "SCALEOUT_PID"
  echo "Started both jobs. PID file: $PID_FILE"
}

check_runs() {
  if [[ ! -f "$PID_FILE" ]]; then
    echo "No pid file found: $PID_FILE"; exit 1
  fi
  # shellcheck disable=SC1090
  source "$PID_FILE"
  for pair in SCALEUP_PID SCALEOUT_PID; do
    pid="${!pair:-}"
    if [[ -z "$pid" ]]; then
      echo "$pair missing"
      continue
    fi
    if kill -0 "$pid" 2>/dev/null; then
      echo "$pair=$pid RUNNING"
    else
      echo "$pair=$pid FINISHED"
    fi
  done
}

wait_runs() {
  if [[ ! -f "$PID_FILE" ]]; then
    echo "No pid file found: $PID_FILE"; exit 1
  fi
  # shellcheck disable=SC1090
  source "$PID_FILE"
  local rc=0
  for pair in SCALEUP_PID SCALEOUT_PID; do
    pid="${!pair:-}"
    if [[ -z "$pid" ]]; then
      echo "$pair missing"
      rc=1
      continue
    fi
    if kill -0 "$pid" 2>/dev/null; then
      echo "Waiting $pair=$pid ..."
      wait "$pid" || rc=$?
    fi
  done
  return "$rc"
}

case "$ACTION" in
  start) start_runs ;;
  check) check_runs ;;
  wait) wait_runs ;;
  *) echo "Usage: $0 [start|check|wait]"; exit 1 ;;
esac
