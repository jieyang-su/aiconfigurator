#!/usr/bin/env bash
set -Eeuo pipefail

AIC_ROOT="${AIC_ROOT:-/workspace/aiconfigurator}"
MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-V4-Flash}"
RUN_ROOT="${RUN_ROOT:-$AIC_ROOT/collector/experiments/deepseek_v4_flash_h20_sglang_0.5.18/rerun_full_$(date +%Y%m%d_%H%M%S)}"
COLLECTOR="$AIC_ROOT/collector/collect.py"
COMMON=(--backend sglang --model-path "$MODEL_PATH" --gpu h20_sxm --sm 90)
DIST="$RUN_ROOT/moe_token_distribution_perf.txt"

mkdir -p "$RUN_ROOT"/{plans,checkpoints}
if [[ ! -s "$DIST" ]]; then
  (
    cd "$RUN_ROOT"
    COLLECTOR_MOE_DISTRIBUTION_MODEL_PATH="$MODEL_PATH" \
    COLLECTOR_MOE_DISTRIBUTION_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    COLLECTOR_MOE_DISTRIBUTION_SINGLE_CARD_EP_SIM=true \
    COLLECTOR_MOE_DISTRIBUTION_EP_SIZES="${EP_SIZES:-1,2,4,8}" \
    COLLECTOR_MOE_DISTRIBUTION_PHASES="context generation" \
      python3 "$COLLECTOR" "${COMMON[@]}" \
        --ops moe_token_distribution \
        --keep-csv \
        --checkpoint-dir "$RUN_ROOT/checkpoints/distribution"
  )
fi
if [[ ! -s "$DIST" ]]; then echo "missing distribution: $DIST" >&2; exit 2; fi
cd "$RUN_ROOT"
export COLLECTOR_MOE_RECORDED_DISTRIBUTION_FILE="$DIST"
export COLLECTOR_MOE_RECORDED_DISTRIBUTION=recorded
python3 "$COLLECTOR" "${COMMON[@]}" --ops dsv4_csa_context_module dsv4_csa_generation_module dsv4_csa_topk_calib dsv4_hca_context_module dsv4_hca_generation_module gemm mhc_module moe --checkpoint-dir "$RUN_ROOT/checkpoints/operators"

echo
echo "采集完成，结果目录：$RUN_ROOT"
find "$RUN_ROOT" -maxdepth 2 -type f \( -name '*_perf.txt' -o -name '*_perf.parquet' -o -name 'collection_meta.yaml' \) -printf '%P\n' | sort
