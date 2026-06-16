#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Production-oriented DeepSeek-V3 WideEP MoE sweep for SGLang.
#
# This wrapper keeps the collector style: it drives the existing
# collect_deepep_moe.py entrypoint through environment variables and writes the
# normal collector perf files under one output directory.  It is intended for
# H20 DeepSeek-V3/V3.1 production calibration where EP=1,2,4,8 should be
# collected as first-class silicon rows instead of derived rows.

set -euo pipefail

THIS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${THIS_DIR}/../../.." && pwd)

MODEL_PATH=${COLLECTOR_MODEL_PATH:-${MOE_MODEL_PATH:-${DEEPSEEK_MODEL_PATH:-/model/DeepSeek-V3}}}
RUN_ROOT=${COLLECTOR_WIDEEP_MOE_RUN_ROOT:-"/cold/tair-kvcache/moe_calibration_runs/h20_dsv3_moe_prod_ep_sweep_$(date +%Y%m%d_%H%M%S)"}
OUTPUT_DIR=${COLLECTOR_WIDEEP_MOE_OUTPUT_DIR:-"${RUN_ROOT}/sglang_wideep_moe"}
LOG_DIR="${RUN_ROOT}/logs"
LOG_FILE="${LOG_DIR}/collect_deepseek_v3_moe_ep_sweep.log"
MANIFEST="${RUN_ROOT}/collector_manifest.txt"

EP_SIZES=${COLLECTOR_WIDEEP_MOE_EP_SIZES:-"1 2 4 8"}
PREFILL_TOKENS=${COLLECTOR_WIDEEP_MOE_PREFILL_TOKENS:-"128 512 2048 4096 8192"}
DECODE_TOKENS=${COLLECTOR_WIDEEP_MOE_DECODE_TOKENS:-"1 2 4 8 16 32 64 128"}
DISTRIBUTIONS=${COLLECTOR_WIDEEP_MOE_DISTRIBUTIONS:-"uniform power_law"}
EPLB_MODE=${COLLECTOR_WIDEEP_MOE_EPLB_MODE:-true}
MEM_FRACTION_STATIC=${COLLECTOR_WIDEEP_MOE_MEM_FRACTION_STATIC:-0.3}
SKIP_DECODE=${COLLECTOR_WIDEEP_MOE_SKIP_DECODE:-0}

case "${EPLB_MODE}" in
  true)
    # AIC DeepSeek-V3 WideEP uses alpha=0.6 for context MoE when enable_eplb=true.
    POWER_LAW_ALPHAS=${COLLECTOR_WIDEEP_MOE_POWER_LAW_ALPHAS:-"0.6"}
    ;;
  false)
    # AIC DeepSeek-V3 WideEP uses alpha=1.01 for context MoE when enable_eplb=false.
    POWER_LAW_ALPHAS=${COLLECTOR_WIDEEP_MOE_POWER_LAW_ALPHAS:-"1.01"}
    ;;
  both)
    POWER_LAW_ALPHAS=${COLLECTOR_WIDEEP_MOE_POWER_LAW_ALPHAS:-"0.6 1.01"}
    ;;
  *)
    echo "COLLECTOR_WIDEEP_MOE_EPLB_MODE must be one of: true, false, both; got ${EPLB_MODE}" >&2
    exit 2
    ;;
esac

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[wideep-moe-ep-sweep] run_root=${RUN_ROOT}"
echo "[wideep-moe-ep-sweep] output_dir=${OUTPUT_DIR}"
echo "[wideep-moe-ep-sweep] model_path=${MODEL_PATH}"
echo "[wideep-moe-ep-sweep] ep_sizes=${EP_SIZES}"
echo "[wideep-moe-ep-sweep] eplb_mode=${EPLB_MODE}"
echo "[wideep-moe-ep-sweep] distributions=${DISTRIBUTIONS}"
echo "[wideep-moe-ep-sweep] power_law_alphas=${POWER_LAW_ALPHAS}"
echo "[wideep-moe-ep-sweep] prefill_tokens=${PREFILL_TOKENS}"
echo "[wideep-moe-ep-sweep] decode_tokens=${DECODE_TOKENS}"
echo "[wideep-moe-ep-sweep] skip_decode=${SKIP_DECODE}"

cat > "${MANIFEST}" <<EOF
DeepSeek-V3 SGLang WideEP MoE EP sweep
generated_at=$(date -Iseconds)
repo_root=${REPO_ROOT}
collector_script=${THIS_DIR}/collect_deepseek_v3_moe_ep_sweep.sh
collector_module=${THIS_DIR}/collect_deepep_moe.py
model_path=${MODEL_PATH}
run_root=${RUN_ROOT}
output_dir=${OUTPUT_DIR}
ep_sizes=${EP_SIZES}
eplb_mode=${EPLB_MODE}
distributions=${DISTRIBUTIONS}
power_law_alphas=${POWER_LAW_ALPHAS}
prefill_tokens=${PREFILL_TOKENS}
decode_tokens=${DECODE_TOKENS}
mem_fraction_static=${MEM_FRACTION_STATIC}
skip_decode=${SKIP_DECODE}
log_file=${LOG_FILE}
EOF

export COLLECTOR_MODEL_PATH="${MODEL_PATH}"
export COLLECTOR_WIDEEP_MOE_EP_SIZES="${EP_SIZES}"
export COLLECTOR_WIDEEP_MOE_PREFILL_TOKENS="${PREFILL_TOKENS}"
export COLLECTOR_WIDEEP_MOE_DECODE_TOKENS="${DECODE_TOKENS}"
export COLLECTOR_WIDEEP_MOE_DISTRIBUTIONS="${DISTRIBUTIONS}"
export COLLECTOR_WIDEEP_MOE_POWER_LAW_ALPHAS="${POWER_LAW_ALPHAS}"
export COLLECTOR_WIDEEP_MOE_MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC}"
export COLLECTOR_WIDEEP_MOE_SKIP_DECODE="${SKIP_DECODE}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/collector:${PYTHONPATH:-}"

python3 "${THIS_DIR}/collect_deepep_moe.py" --output-path "${OUTPUT_DIR}"

python3 - "${OUTPUT_DIR}" <<'PY'
import sys
from pathlib import Path

import pandas as pd

output_dir = Path(sys.argv[1])
for name in ("wideep_context_moe_perf", "wideep_generation_moe_perf"):
    txt = output_dir / f"{name}.txt"
    parquet = output_dir / f"{name}.parquet"
    if not txt.exists():
        print(f"[wideep-moe-ep-sweep] skip parquet conversion, missing {txt}")
        continue
    df = pd.read_csv(txt)
    df.to_parquet(parquet, index=False)
    print(f"[wideep-moe-ep-sweep] wrote {parquet} rows={len(df)}")
PY

echo "[wideep-moe-ep-sweep] completed successfully"
echo "[wideep-moe-ep-sweep] manifest=${MANIFEST}"
