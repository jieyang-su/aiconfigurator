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
PREFILL_WORKERS=0
DECODE_WORKERS=0
PREFILL_SYSTEM_PORT_BASE=${PREFILL_SYSTEM_PORT_BASE:-8082}
DECODE_SYSTEM_PORT_BASE=${DECODE_SYSTEM_PORT_BASE:-$((PREFILL_SYSTEM_PORT_BASE + PREFILL_WORKERS))}

OTEL_SERVICE_NAME=dynamo-frontend \
python3 -m dynamo.frontend --http-port "8000" 2>&1 | sed "s/^/[Frontend] /" &

AGG_GPU=256
AGG_WORKERS=1
AGG_GPU_OFFSET=0
for ((w=0; w<AGG_WORKERS; w++)); do
  BASE=$(( AGG_GPU_OFFSET + w * AGG_GPU ))
  GPU_LIST=$(seq -s, $BASE $((BASE+AGG_GPU-1)))
  WORKER_IDX=$(( w + 1 ))
  SYSTEM_PORT=$(( AGG_SYSTEM_PORT + w ))
  WORKER_NAME="dynamo-worker"
  if (( AGG_WORKERS > 1 )); then
    WORKER_NAME="${WORKER_NAME}-${WORKER_IDX}"
  fi
  ( CUDA_VISIBLE_DEVICES=$GPU_LIST \
  OTEL_SERVICE_NAME="${WORKER_NAME}" \
  DYN_SYSTEM_PORT="${SYSTEM_PORT}" \
  python3 -m dynamo.sglang \
    --model-path "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --tensor-parallel-size "16" --pipeline-parallel-size "1" --data-parallel-size "16" --kv-cache-dtype "fp8_e4m3" --max-prefill-tokens "3548" --enable-mixed-chunk --max-running-requests "512" --enable-dp-attention --expert-parallel-size "16" --cuda-graph-bs 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 16 20 24 28 32 32 40 48 56 64 64 80 96 112 128 128 160 192 224 256 256 320 384 448 512 1 2 3 4 5 6 7 8 9 10 11 12 13 --speculative-algorithm "NEXTN" --speculative-num-steps "1" \
    --host "0.0.0.0" \
    --enable-metrics 2>&1 | sed "s/^/[Worker $w] /" ) &
done
wait
