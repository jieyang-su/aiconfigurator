#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Non-invasive DeepSeek-V3 MoE calibration loop for AIC.
#
# This script keeps AIC's normal model/simulation code unchanged.  It prepares
# a systems overlay, optionally materializes recorded MoE distribution rows into
# the WideEP compute table, optionally installs a DeepEP dispatch/combine table,
# writes a DeepSeek-V3 experiment YAML, then can run AIC against that overlay.

set -euo pipefail

REPO_ROOT=${AIC_REPO_ROOT:-/cold/tair-kvcache/aiconfigurator}
DEFAULT_SYSTEMS_ROOT="${REPO_ROOT}/src/aiconfigurator/systems"

MODEL_PATH=${AIC_MOE_MODEL_PATH:-deepseek-ai/DeepSeek-V3}
MODEL_SHORT=${AIC_MOE_MODEL_SHORT:-deepseek-v3}
SYSTEM=${AIC_MOE_SYSTEM:-h20_sxm}
BACKEND=${AIC_MOE_BACKEND:-sglang}
BACKEND_VERSION=${AIC_MOE_BACKEND_VERSION:-0.0.0.dev1+g959d8a09d-aicop-20260613}
DEVICE=${AIC_MOE_DEVICE:-H20}
DATABASE_MODE=${AIC_MOE_DATABASE_MODE:-SILICON}
ENABLE_EPLB=${AIC_MOE_ENABLE_EPLB:-true}

RUN_ROOT=${AIC_MOE_RUN_ROOT:-"${REPO_ROOT}/../moe_calibration_runs/deepseek_v3_moe_closed_loop_$(date +%Y%m%d_%H%M%S)"}
SOURCE_SYSTEMS_ROOT=${AIC_MOE_SOURCE_SYSTEMS_ROOT:-"${DEFAULT_SYSTEMS_ROOT}"}
CALIBRATED_SYSTEMS_ROOT=${AIC_MOE_CALIBRATED_SYSTEMS_ROOT:-"${RUN_ROOT}/aic_systems_overlay_calibrated"}
FORCE_OVERLAY=${AIC_MOE_FORCE_OVERLAY:-0}

RUN_PRECOLLECT=${AIC_MOE_RUN_PRECOLLECT:-auto}
RUN_QUERY_SMOKE=${AIC_MOE_RUN_QUERY_SMOKE:-1}
RUN_SIMULATION=${AIC_MOE_RUN_SIMULATION:-0}
SIMULATION_REQUIRED=${AIC_MOE_SIMULATION_REQUIRED:-0}
DERIVE_MISSING_EP=${AIC_MOE_DERIVE_MISSING_EP:-0}
DERIVE_SOURCE_EP=${AIC_MOE_DERIVE_SOURCE_EP:-2}
DERIVE_LATENCY_SCALE=${AIC_MOE_DERIVE_LATENCY_SCALE:-1.0}
DERIVE_OVERWRITE_EP=${AIC_MOE_DERIVE_OVERWRITE_EP:-0}
DERIVE_DEEPEP_LL=${AIC_MOE_DERIVE_DEEPEP_LL:-0}

WORKLOAD_DISTRIBUTION=${AIC_MOE_WORKLOAD_DISTRIBUTION:-uniform}
RECORDED_DISTRIBUTION=${AIC_MOE_RECORDED_DISTRIBUTION:-recorded_rankprobe}
OUTPUT_DISTRIBUTION=${AIC_MOE_OUTPUT_DISTRIBUTION:-uniform}
ARCHIVE_DISTRIBUTION=${AIC_MOE_ARCHIVE_DISTRIBUTION:-synthetic_uniform}

NUM_TOKENS=${AIC_MOE_NUM_TOKENS:-"128 2048 4096 8192"}
QUERY_NUM_TOKENS=${AIC_MOE_QUERY_NUM_TOKENS:-"128 512 2048 4096 8192"}
TOPK=${AIC_MOE_TOPK:-8}
NUM_EXPERTS=${AIC_MOE_NUM_EXPERTS:-256}
HIDDEN_SIZE=${AIC_MOE_HIDDEN_SIZE:-7168}
INTER_SIZE=${AIC_MOE_INTER_SIZE:-2048}
MOE_TP_SIZE=${AIC_MOE_TP_SIZE:-1}
MOE_EP_SIZE=${AIC_MOE_EP_SIZE:-2}
MOE_DTYPE=${AIC_MOE_DTYPE:-fp8_block}
GEMM_DTYPE=${AIC_GEMM_DTYPE:-fp8_block}
KVCACHE_DTYPE=${AIC_KVCACHE_DTYPE:-bfloat16}
FMHA_DTYPE=${AIC_FMHA_DTYPE:-fp8_block}
COMM_DTYPE=${AIC_COMM_DTYPE:-half}
PROFILED_MOE_LAYERS=${AIC_MOE_PROFILED_MOE_LAYERS:-3}

TOTAL_GPUS=${AIC_MOE_TOTAL_GPUS:-8}
MODEL_TP_SIZE=${AIC_MODEL_TP_SIZE:-8}
MODEL_PP_SIZE=${AIC_MODEL_PP_SIZE:-1}
MODEL_DP_SIZE=${AIC_MODEL_DP_SIZE:-1}
ISL=${AIC_MOE_ISL:-4000}
OSL=${AIC_MOE_OSL:-1000}
TTFT=${AIC_MOE_TTFT:-2000.0}
TPOT=${AIC_MOE_TPOT:-30.0}
TOP_N=${AIC_MOE_TOP_N:-5}

DEEPEP_NORMAL_PERF_TXT=${AIC_MOE_DEEPEP_NORMAL_PERF_TXT:-}
DEEPEP_NORMAL_PERF_PARQUET=${AIC_MOE_DEEPEP_NORMAL_PERF_PARQUET:-}
NCCL_PERF_TXT=${AIC_NCCL_PERF_TXT:-}
NCCL_PERF_PARQUET=${AIC_NCCL_PERF_PARQUET:-}
NCCL_VERSION=${AIC_NCCL_VERSION:-2.28.9}

DATA_DIR="${CALIBRATED_SYSTEMS_ROOT}/data/${SYSTEM}/${BACKEND}/${BACKEND_VERSION}"
ARTIFACT_DIR="${RUN_ROOT}/aic_closed_loop"
YAML_PATH="${ARTIFACT_DIR}/deepseek_v3_moe_calibrated.yaml"
SIM_RESULT_DIR="${ARTIFACT_DIR}/aic_simulation_results"
SIM_LOG="${ARTIFACT_DIR}/aic_simulation.log"
SIM_STATUS_FILE="${ARTIFACT_DIR}/aic_simulation_status.txt"
QUERY_COMPUTE_OUTPUT="${ARTIFACT_DIR}/aic_query_moe_compute_${DATABASE_MODE}.txt"
QUERY_COMM_OUTPUT="${ARTIFACT_DIR}/aic_query_deepep_normal_${DATABASE_MODE}.txt"
MANIFEST_PATH="${ARTIFACT_DIR}/closed_loop_manifest.txt"

PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPATH

mkdir -p "${ARTIFACT_DIR}"

case "${ENABLE_EPLB}" in
  true|false) ;;
  1) ENABLE_EPLB=true ;;
  0) ENABLE_EPLB=false ;;
  *)
    echo "AIC_MOE_ENABLE_EPLB must be true/false/1/0, got ${ENABLE_EPLB}" >&2
    exit 2
    ;;
esac

log() {
  printf '[moe-closed-loop] %s\n' "$*"
}

need_file() {
  local path=$1
  local label=$2
  if [[ ! -f "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 2
  fi
}

copy_or_create_overlay() {
  if [[ -d "${CALIBRATED_SYSTEMS_ROOT}" ]]; then
    if [[ "${FORCE_OVERLAY}" == "1" ]]; then
      log "Recreating systems overlay: ${CALIBRATED_SYSTEMS_ROOT}"
      python3 "${REPO_ROOT}/tools/moe_calibration/moe_distribution_strategy.py" make-overlay \
        --source-overlay "${SOURCE_SYSTEMS_ROOT}" \
        --target-overlay "${CALIBRATED_SYSTEMS_ROOT}" \
        --force
    else
      log "Using existing systems overlay: ${CALIBRATED_SYSTEMS_ROOT}"
    fi
  else
    log "Creating systems overlay: ${CALIBRATED_SYSTEMS_ROOT}"
    python3 "${REPO_ROOT}/tools/moe_calibration/moe_distribution_strategy.py" make-overlay \
      --source-overlay "${SOURCE_SYSTEMS_ROOT}" \
      --target-overlay "${CALIBRATED_SYSTEMS_ROOT}"
  fi
  mkdir -p "${DATA_DIR}"
}

should_run_precollect() {
  if [[ "${RUN_PRECOLLECT}" == "1" ]]; then
    return 0
  fi
  if [[ "${RUN_PRECOLLECT}" == "0" ]]; then
    return 1
  fi
  if [[ -n "${AIC_MOE_RANK_STAGE_CSV:-}" ]] && \
     { [[ -n "${AIC_MOE_DISTRIBUTION_SUMMARY_CSV:-}" ]] || [[ -n "${AIC_MOE_DISTRIBUTION_RECORDER_DIR:-}" ]]; }; then
    return 0
  fi
  return 1
}

run_precollect_if_requested() {
  if should_run_precollect; then
    log "Running MoE distribution precollect materialization"
    AIC_REPO_ROOT="${REPO_ROOT}" \
    AIC_SYSTEMS_ROOT="${CALIBRATED_SYSTEMS_ROOT}" \
    AIC_MOE_MODEL="${MODEL_SHORT}" \
    AIC_MOE_SYSTEM="${SYSTEM}" \
    AIC_MOE_BACKEND="${BACKEND}" \
    AIC_MOE_BACKEND_VERSION="${BACKEND_VERSION}" \
    AIC_MOE_DEVICE="${DEVICE}" \
    AIC_MOE_RECORDED_DISTRIBUTION="${RECORDED_DISTRIBUTION}" \
    AIC_MOE_OUTPUT_DISTRIBUTION="${OUTPUT_DISTRIBUTION}" \
    AIC_MOE_ARCHIVE_DISTRIBUTION="${ARCHIVE_DISTRIBUTION}" \
    AIC_MOE_NUM_TOKENS="${NUM_TOKENS}" \
    AIC_MOE_TOPK="${TOPK}" \
    AIC_MOE_NUM_EXPERTS="${NUM_EXPERTS}" \
    AIC_MOE_HIDDEN_SIZE="${HIDDEN_SIZE}" \
    AIC_MOE_INTER_SIZE="${INTER_SIZE}" \
    AIC_MOE_TP_SIZE="${MOE_TP_SIZE}" \
    AIC_MOE_EP_SIZE="${MOE_EP_SIZE}" \
    AIC_MOE_DTYPE="${MOE_DTYPE}" \
    AIC_MOE_PROFILED_MOE_LAYERS="${PROFILED_MOE_LAYERS}" \
      "${REPO_ROOT}/tools/moe_calibration/run_moe_distribution_precollect.sh"
  else
    log "Skipping precollect materialization; validating existing calibrated compute table"
    need_file "${DATA_DIR}/wideep_context_moe_perf.txt" "WideEP context MoE txt"
    need_file "${DATA_DIR}/wideep_context_moe_perf.parquet" "WideEP context MoE parquet"
    need_file "${DATA_DIR}/moe_token_distribution_perf.txt" "MoE token distribution txt"
  fi
}

derive_missing_ep_if_requested() {
  if [[ "${DERIVE_MISSING_EP}" != "1" ]]; then
    return
  fi
  log "Deriving WideEP MoE compute rows: source_ep=${DERIVE_SOURCE_EP}, target_ep=${MOE_EP_SIZE}, scale=${DERIVE_LATENCY_SCALE}, overwrite=${DERIVE_OVERWRITE_EP}"
  python3 - "${DATA_DIR}/wideep_context_moe_perf.parquet" "${DATA_DIR}/wideep_context_moe_perf.txt" \
    "${DERIVE_SOURCE_EP}" "${MOE_EP_SIZE}" "${DERIVE_LATENCY_SCALE}" "${DERIVE_OVERWRITE_EP}" <<'PY'
import sys
from pathlib import Path

import pandas as pd

parquet_path = Path(sys.argv[1])
txt_path = Path(sys.argv[2])
source_ep = int(sys.argv[3])
target_ep = int(sys.argv[4])
scale = float(sys.argv[5])
overwrite = sys.argv[6] == "1"

df = pd.read_parquet(parquet_path)
if (df["moe_ep_size"] == target_ep).any() and not overwrite:
    print(f"Target moe_ep_size={target_ep} already exists in {parquet_path}; no rows derived")
    df.to_csv(txt_path, index=False)
    raise SystemExit(0)

source = df[df["moe_ep_size"] == source_ep].copy()
if source.empty:
    raise SystemExit(f"No source rows with moe_ep_size={source_ep} in {parquet_path}")

derived = source.copy()
derived["moe_ep_size"] = target_ep
derived["latency"] = derived["latency"].astype(float) * scale
if "kernel_source" in derived.columns:
    derived["kernel_source"] = (
        derived["kernel_source"].astype(str)
        + f"_derived_ep{source_ep}_to_ep{target_ep}_scale{scale:g}"
    )

if overwrite:
    df = df[df["moe_ep_size"] != target_ep].copy()
merged = pd.concat([df, derived], ignore_index=True)
merged.to_parquet(parquet_path, index=False)
merged.to_csv(txt_path, index=False)
print(f"Derived {len(derived)} rows for moe_ep_size={target_ep} from source_ep={source_ep}")
PY
}

install_deepep_table_if_requested() {
  if [[ -z "${DEEPEP_NORMAL_PERF_TXT}" && -z "${DEEPEP_NORMAL_PERF_PARQUET}" ]]; then
    log "No DeepEP table override requested; validating existing communication table"
    need_file "${DATA_DIR}/wideep_deepep_normal_perf.txt" "DeepEP normal txt"
    need_file "${DATA_DIR}/wideep_deepep_normal_perf.parquet" "DeepEP normal parquet"
    return
  fi

  log "Installing DeepEP dispatch/combine table into calibrated overlay"
  if [[ -n "${DEEPEP_NORMAL_PERF_TXT}" ]]; then
    need_file "${DEEPEP_NORMAL_PERF_TXT}" "DeepEP normal txt source"
    cp "${DEEPEP_NORMAL_PERF_TXT}" "${DATA_DIR}/wideep_deepep_normal_perf.txt"
  fi
  if [[ -n "${DEEPEP_NORMAL_PERF_PARQUET}" ]]; then
    need_file "${DEEPEP_NORMAL_PERF_PARQUET}" "DeepEP normal parquet source"
    cp "${DEEPEP_NORMAL_PERF_PARQUET}" "${DATA_DIR}/wideep_deepep_normal_perf.parquet"
  elif [[ -n "${DEEPEP_NORMAL_PERF_TXT}" ]]; then
    python3 - "${DATA_DIR}/wideep_deepep_normal_perf.txt" "${DATA_DIR}/wideep_deepep_normal_perf.parquet" <<'PY'
import sys
import pandas as pd

txt, parquet = sys.argv[1:3]
pd.read_csv(txt).to_parquet(parquet, index=False)
PY
  fi
}

derive_deepep_ll_if_requested() {
  if [[ "${DERIVE_DEEPEP_LL}" != "1" ]]; then
    return
  fi
  log "Deriving DeepEP LL table from normal dispatch/combine table"
  python3 - \
    "${DATA_DIR}/wideep_deepep_normal_perf.parquet" \
    "${DATA_DIR}/wideep_deepep_ll_perf.parquet" \
    "${DATA_DIR}/wideep_deepep_ll_perf.txt" <<'PY'
import sys
from pathlib import Path

import pandas as pd

normal_path = Path(sys.argv[1])
ll_parquet = Path(sys.argv[2])
ll_txt = Path(sys.argv[3])

normal = pd.read_parquet(normal_path)
ll = pd.DataFrame(
    {
        "framework": normal.get("framework", "sglang"),
        "version": normal.get("version", ""),
        "device": normal.get("device", ""),
        "op_name": "ll_derived_from_normal",
        "node_num": normal["node_num"],
        "kernel_source": normal.get("kernel_source", "deepep").astype(str) + "_derived_ll_from_normal",
        "hidden_size": normal["hidden_size"],
        "num_token": normal["num_token"],
        "num_topk": normal["num_topk"],
        "num_experts": normal["num_experts"],
        "dispatch_avg_t_us": normal["dispatch_transmit_us"].astype(float)
        + normal["dispatch_notify_us"].astype(float),
        "combine_avg_t_us": normal["combine_transmit_us"].astype(float)
        + normal["combine_notify_us"].astype(float),
    }
)
if "power" in normal.columns:
    ll["power"] = normal["power"]
ll.to_parquet(ll_parquet, index=False)
ll.to_csv(ll_txt, index=False)
print(f"Derived {len(ll)} DeepEP LL rows from {normal_path}")
PY
}

install_nccl_table_if_requested() {
  if [[ -z "${NCCL_PERF_TXT}" && -z "${NCCL_PERF_PARQUET}" ]]; then
    return
  fi
  local nccl_dir="${CALIBRATED_SYSTEMS_ROOT}/data/${SYSTEM}/nccl/${NCCL_VERSION}"
  mkdir -p "${nccl_dir}"
  log "Installing NCCL table into calibrated overlay: ${nccl_dir}"
  if [[ -n "${NCCL_PERF_PARQUET}" ]]; then
    need_file "${NCCL_PERF_PARQUET}" "NCCL parquet source"
    python3 - "${NCCL_PERF_PARQUET}" "${nccl_dir}/nccl_perf.parquet" "${nccl_dir}/nccl_perf.txt" <<'PY'
import sys
import pandas as pd

source, parquet, txt = sys.argv[1:4]
df = pd.read_parquet(source)
df.to_parquet(parquet, index=False)
df.to_csv(txt, index=False)
PY
  else
    need_file "${NCCL_PERF_TXT}" "NCCL txt source"
    cp "${NCCL_PERF_TXT}" "${nccl_dir}/nccl_perf.txt"
    python3 - "${nccl_dir}/nccl_perf.txt" "${nccl_dir}/nccl_perf.parquet" <<'PY'
import sys
import pandas as pd

txt, parquet = sys.argv[1:3]
pd.read_csv(txt).to_parquet(parquet, index=False)
PY
  fi
}

write_experiment_yaml() {
  log "Writing DeepSeek-V3 calibrated AIC experiment YAML: ${YAML_PATH}"
  cat > "${YAML_PATH}" <<YAML
exps:
  - deepseek_v3_h20_sglang_moe_calibrated_agg

deepseek_v3_h20_sglang_moe_calibrated_agg:
  mode: "patch"
  serving_mode: "agg"
  model_path: "${MODEL_PATH}"
  total_gpus: ${TOTAL_GPUS}
  system_name: "${SYSTEM}"
  backend_name: "${BACKEND}"
  backend_version: "${BACKEND_VERSION}"
  database_mode: "${DATABASE_MODE}"
  isl: ${ISL}
  osl: ${OSL}
  ttft: ${TTFT}
  tpot: ${TPOT}
  enable_wideep: true
  moe_backend: "deepep_moe"
  enable_eplb: ${ENABLE_EPLB}
  workload_distribution: "${WORKLOAD_DISTRIBUTION}"
  config:
    nextn: 1
    nextn_accept_rates: [0.85, 0, 0, 0, 0]
    worker_config:
      gemm_quant_mode: "${GEMM_DTYPE}"
      moe_quant_mode: "${MOE_DTYPE}"
      kvcache_quant_mode: "${KVCACHE_DTYPE}"
      fmha_quant_mode: "${FMHA_DTYPE}"
      comm_quant_mode: "${COMM_DTYPE}"
      workload_distribution: "${WORKLOAD_DISTRIBUTION}"
      num_gpu_per_worker: [${TOTAL_GPUS}]
      tp_list: [${MODEL_TP_SIZE}]
      pp_list: [${MODEL_PP_SIZE}]
      dp_list: [${MODEL_DP_SIZE}]
      moe_tp_list: [${MOE_TP_SIZE}]
      moe_ep_list: [${MOE_EP_SIZE}]
YAML
}

run_query_smoke_if_requested() {
  if [[ "${RUN_QUERY_SMOKE}" != "1" ]]; then
    return
  fi
  log "Running AIC operator smoke queries against calibrated overlay"
  local eplb_args=()
  if [[ "${ENABLE_EPLB}" == "true" ]]; then
    eplb_args+=(--enable-eplb)
  fi

  python3 "${REPO_ROOT}/tools/moe_calibration/aic_moe_calibrate.py" query-aic \
    --systems-root "${CALIBRATED_SYSTEMS_ROOT}" \
    --system "${SYSTEM}" \
    --backend "${BACKEND}" \
    --backend-version "${BACKEND_VERSION}" \
    --database-mode "${DATABASE_MODE}" \
    --num-tokens ${QUERY_NUM_TOKENS} \
    --output "${QUERY_COMPUTE_OUTPUT}" \
    --hidden-size "${HIDDEN_SIZE}" \
    --inter-size "${INTER_SIZE}" \
    --topk "${TOPK}" \
    --num-experts "${NUM_EXPERTS}" \
    --moe-tp-size "${MOE_TP_SIZE}" \
    --moe-ep-size "${MOE_EP_SIZE}" \
    --quant-mode "${MOE_DTYPE}" \
    --distribution "${WORKLOAD_DISTRIBUTION}" \
    --moe-backend deepep_moe \
    "${eplb_args[@]}"

  python3 "${REPO_ROOT}/tools/moe_calibration/aic_moe_calibrate.py" query-deepep-dispatch \
    --systems-root "${CALIBRATED_SYSTEMS_ROOT}" \
    --system "${SYSTEM}" \
    --backend "${BACKEND}" \
    --backend-version "${BACKEND_VERSION}" \
    --database-mode "${DATABASE_MODE}" \
    --num-tokens ${QUERY_NUM_TOKENS} \
    --output "${QUERY_COMM_OUTPUT}" \
    --hidden-size "${HIDDEN_SIZE}" \
    --topk "${TOPK}" \
    --num-experts "${NUM_EXPERTS}" \
    --node-num 1 \
    --sms 20 \
    --deepep-mode normal \
    --phase context \
    --distribution balanced
}

run_simulation_if_requested() {
  if [[ "${RUN_SIMULATION}" != "1" ]]; then
    log "Skipping full AIC simulation. Set AIC_MOE_RUN_SIMULATION=1 to execute it."
    return
  fi
  log "Running AIC DeepSeek-V3 simulation with calibrated systems overlay"
  set +e
  python3 -m aiconfigurator.main cli exp \
    --yaml-path "${YAML_PATH}" \
    --systems-paths "${CALIBRATED_SYSTEMS_ROOT},default" \
    --save-dir "${SIM_RESULT_DIR}" \
    --top-n "${TOP_N}" 2>&1 | tee "${SIM_LOG}"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ "${status}" -eq 0 ]]; then
    echo "status=success" > "${SIM_STATUS_FILE}"
    return
  fi
  echo "status=failed exit_code=${status}" > "${SIM_STATUS_FILE}"
  log "AIC simulation failed with exit code ${status}; log: ${SIM_LOG}"
  if [[ "${SIMULATION_REQUIRED}" == "1" ]]; then
    exit "${status}"
  fi
}

write_manifest() {
  cat > "${MANIFEST_PATH}" <<EOF
DeepSeek-V3 MoE AIC closed loop
generated_at=$(date -Iseconds)
repo_root=${REPO_ROOT}
model_path=${MODEL_PATH}
system=${SYSTEM}
backend=${BACKEND}
backend_version=${BACKEND_VERSION}
database_mode=${DATABASE_MODE}
calibrated_systems_root=${CALIBRATED_SYSTEMS_ROOT}
data_dir=${DATA_DIR}
workload_distribution=${WORKLOAD_DISTRIBUTION}
recorded_distribution=${RECORDED_DISTRIBUTION}
output_distribution=${OUTPUT_DISTRIBUTION}
enable_eplb=${ENABLE_EPLB}
model_tp_size=${MODEL_TP_SIZE}
model_pp_size=${MODEL_PP_SIZE}
model_dp_size=${MODEL_DP_SIZE}
moe_tp_size=${MOE_TP_SIZE}
moe_ep_size=${MOE_EP_SIZE}
gemm_dtype=${GEMM_DTYPE}
moe_dtype=${MOE_DTYPE}
kvcache_dtype=${KVCACHE_DTYPE}
fmha_dtype=${FMHA_DTYPE}
comm_dtype=${COMM_DTYPE}
derive_missing_ep=${DERIVE_MISSING_EP}
derive_source_ep=${DERIVE_SOURCE_EP}
derive_latency_scale=${DERIVE_LATENCY_SCALE}
derive_overwrite_ep=${DERIVE_OVERWRITE_EP}
derive_deepep_ll=${DERIVE_DEEPEP_LL}
num_tokens=${NUM_TOKENS}
query_num_tokens=${QUERY_NUM_TOKENS}
yaml_path=${YAML_PATH}
compute_query=${QUERY_COMPUTE_OUTPUT}
comm_query=${QUERY_COMM_OUTPUT}
nccl_perf_txt=${NCCL_PERF_TXT}
nccl_perf_parquet=${NCCL_PERF_PARQUET}
nccl_version=${NCCL_VERSION}
simulation_result_dir=${SIM_RESULT_DIR}
run_precollect=${RUN_PRECOLLECT}
run_query_smoke=${RUN_QUERY_SMOKE}
run_simulation=${RUN_SIMULATION}
simulation_required=${SIMULATION_REQUIRED}
simulation_log=${SIM_LOG}
simulation_status_file=${SIM_STATUS_FILE}
EOF
  log "Wrote manifest: ${MANIFEST_PATH}"
}

copy_or_create_overlay
run_precollect_if_requested
derive_missing_ep_if_requested
install_deepep_table_if_requested
derive_deepep_ll_if_requested
install_nccl_table_if_requested
write_experiment_yaml
run_query_smoke_if_requested
run_simulation_if_requested
write_manifest

log "Done. Use this systems overlay in AIC with: --systems-paths ${CALIBRATED_SYSTEMS_ROOT},default"
