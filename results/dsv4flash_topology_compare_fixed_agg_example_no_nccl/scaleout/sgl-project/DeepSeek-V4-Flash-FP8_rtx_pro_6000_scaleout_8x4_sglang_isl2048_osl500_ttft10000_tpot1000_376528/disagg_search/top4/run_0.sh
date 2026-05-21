#!/bin/bash
set -e
trap 'echo "Cleaning up..."; kill 0 2>/dev/null || true' EXIT INT TERM

export MODEL_PATH=${MODEL_PATH:-"sgl-project/DeepSeek-V4-Flash-FP8"}
export HF_TOKEN=${HF_TOKEN:-"None"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"sgl-project/DeepSeek-V4-Flash-FP8"}
export HEAD_NODE_IP=${HEAD_NODE_IP:-"0.0.0.0"}
export ETCD_ENDPOINTS="${HEAD_NODE_IP}:2379"
export NATS_SERVER="nats://${HEAD_NODE_IP}:4222"

FRONTEND_SYSTEM_PORT=${FRONTEND_SYSTEM_PORT:-8080}
AGG_SYSTEM_PORT=${AGG_SYSTEM_PORT:-8081}
PREFILL_WORKERS=1
DECODE_WORKERS=0
PREFILL_SYSTEM_PORT_BASE=${PREFILL_SYSTEM_PORT_BASE:-8082}
DECODE_SYSTEM_PORT_BASE=${DECODE_SYSTEM_PORT_BASE:-$((PREFILL_SYSTEM_PORT_BASE + PREFILL_WORKERS))}

OTEL_SERVICE_NAME=dynamo-frontend \
python3 -m dynamo.frontend --http-port "8000" 2>&1 | sed "s/^/[Frontend] /" &

PREFILL_GPU=16
for ((w=0; w<PREFILL_WORKERS; w++)); do
  BASE=$(( w * PREFILL_GPU ))
  GPU_LIST=$(seq -s, $BASE $((BASE+PREFILL_GPU-1)))
  WORKER_IDX=$(( w + 1 ))
  SYSTEM_PORT=$(( PREFILL_SYSTEM_PORT_BASE + w ))
  WORKER_NAME="dynamo-worker-prefill"
  if (( PREFILL_WORKERS > 1 )); then
    WORKER_NAME="${WORKER_NAME}-${WORKER_IDX}"
  fi
  ( CUDA_VISIBLE_DEVICES=$GPU_LIST \
  OTEL_SERVICE_NAME="${WORKER_NAME}" \
  DYN_SYSTEM_PORT="${SYSTEM_PORT}" \
    python3 -m dynamo.sglang \
      --model-path "$MODEL_PATH" \
      --served-model-name "$SERVED_MODEL_NAME" \
      --tensor-parallel-size "16" --pipeline-parallel-size "1" --data-parallel-size "1" --kv-cache-dtype "fp8_e4m3" --max-prefill-tokens "3548" --max-running-requests "1" --expert-parallel-size "16" --speculative-algorithm "NEXTN" --speculative-num-steps "1" --disaggregation-transfer-backend "nixl" --disaggregation-mode prefill \
      --host "0.0.0.0" \
      --enable-metrics 2>&1 | sed "s/^/[Prefill $w] /" ) &
done

wait
