#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Materialize a model MoE token-distribution pre-collection into AIC systems
# data, then replace the synthetic uniform WideEP MoE rows with the recorded
# distribution rows.  This is intended to run before the normal operator
# collection/query flow for a model+hardware+backend-version tuple.

set -euo pipefail

REPO_ROOT=${AIC_REPO_ROOT:-/cold/tair-kvcache/aiconfigurator}
MODEL=${AIC_MOE_MODEL:-deepseek-v3}
SYSTEM=${AIC_MOE_SYSTEM:-h20_sxm}
BACKEND=${AIC_MOE_BACKEND:-sglang}
BACKEND_VERSION=${AIC_MOE_BACKEND_VERSION:-0.0.0.dev1+g959d8a09d-aicop-20260613}
DEVICE=${AIC_MOE_DEVICE:-H20}
PHASE=${AIC_MOE_PHASE:-context}
RECORDED_DISTRIBUTION=${AIC_MOE_RECORDED_DISTRIBUTION:-recorded_rankprobe}
OUTPUT_DISTRIBUTION=${AIC_MOE_OUTPUT_DISTRIBUTION:-uniform}
ARCHIVE_DISTRIBUTION=${AIC_MOE_ARCHIVE_DISTRIBUTION:-synthetic_uniform}

SYSTEMS_ROOT=${AIC_SYSTEMS_ROOT:?Set AIC_SYSTEMS_ROOT to the target systems overlay/root}
RANK_STAGE_CSV=${AIC_MOE_RANK_STAGE_CSV:?Set AIC_MOE_RANK_STAGE_CSV to rank0/rank1 stage comparison CSV}
SUMMARY_CSV=${AIC_MOE_DISTRIBUTION_SUMMARY_CSV:-}
RECORDER_DIR=${AIC_MOE_DISTRIBUTION_RECORDER_DIR:-}

NUM_TOKENS=${AIC_MOE_NUM_TOKENS:-"128 2048"}
TOPK=${AIC_MOE_TOPK:-8}
NUM_EXPERTS=${AIC_MOE_NUM_EXPERTS:-256}
HIDDEN_SIZE=${AIC_MOE_HIDDEN_SIZE:-7168}
INTER_SIZE=${AIC_MOE_INTER_SIZE:-2048}
MOE_TP_SIZE=${AIC_MOE_TP_SIZE:-1}
MOE_EP_SIZE=${AIC_MOE_EP_SIZE:-2}
MOE_DTYPE=${AIC_MOE_DTYPE:-fp8_block}
PROFILED_MOE_LAYERS=${AIC_MOE_PROFILED_MOE_LAYERS:-3}

DATA_DIR="${SYSTEMS_ROOT}/data/${SYSTEM}/${BACKEND}/${BACKEND_VERSION}"
mkdir -p "${DATA_DIR}"

if [[ -z "${SUMMARY_CSV}" ]]; then
  if [[ -z "${RECORDER_DIR}" ]]; then
    echo "Set either AIC_MOE_DISTRIBUTION_SUMMARY_CSV or AIC_MOE_DISTRIBUTION_RECORDER_DIR" >&2
    exit 2
  fi
  SUMMARY_CSV="${DATA_DIR}/moe_token_distribution_summary.csv"
  PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}" python3 \
    "${REPO_ROOT}/tools/moe_calibration/aic_moe_calibrate.py" summarize-expert-distribution \
    --input-dir "${RECORDER_DIR}" \
    --output "${SUMMARY_CSV}"
fi

PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}" python3 \
  "${REPO_ROOT}/tools/moe_calibration/moe_distribution_strategy.py" build-distribution-table \
  --summary-csv "${SUMMARY_CSV}" \
  --output-txt "${DATA_DIR}/moe_token_distribution_perf.txt" \
  --output-parquet "${DATA_DIR}/moe_token_distribution_perf.parquet" \
  --version "${BACKEND_VERSION}" \
  --device "${DEVICE}" \
  --model "${MODEL}" \
  --phase "${PHASE}" \
  --distribution "${RECORDED_DISTRIBUTION}" \
  --topk "${TOPK}" \
  --num-experts "${NUM_EXPERTS}" \
  --num-tokens ${NUM_TOKENS}

PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}" python3 \
  "${REPO_ROOT}/tools/moe_calibration/moe_distribution_strategy.py" materialize-wideep-from-distribution \
  --base-table "${DATA_DIR}/wideep_context_moe_perf.parquet" \
  --distribution-table "${DATA_DIR}/moe_token_distribution_perf.parquet" \
  --rank-stage-csv "${RANK_STAGE_CSV}" \
  --output-table "${DATA_DIR}/wideep_context_moe_perf.parquet" \
  --output-compare-csv "${DATA_DIR}/moe_token_distribution_materialize_summary.csv" \
  --distribution "${RECORDED_DISTRIBUTION}" \
  --output-distribution "${OUTPUT_DISTRIBUTION}" \
  --archive-existing-output-distribution-as "${ARCHIVE_DISTRIBUTION}" \
  --num-profiled-moe-layers "${PROFILED_MOE_LAYERS}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --inter-size "${INTER_SIZE}" \
  --topk "${TOPK}" \
  --num-experts "${NUM_EXPERTS}" \
  --moe-tp-size "${MOE_TP_SIZE}" \
  --moe-ep-size "${MOE_EP_SIZE}" \
  --moe-dtype "${MOE_DTYPE}" \
  --num-tokens ${NUM_TOKENS}

echo "Wrote ${DATA_DIR}/moe_token_distribution_perf.txt"
echo "Updated ${DATA_DIR}/wideep_context_moe_perf.parquet with ${OUTPUT_DISTRIBUTION} from ${RECORDED_DISTRIBUTION}"
