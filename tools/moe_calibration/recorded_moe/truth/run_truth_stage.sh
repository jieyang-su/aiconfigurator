#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Stage-based recorded MoE real-machine truth runner.
# Run inside the target platform container.

set -euo pipefail

if [[ -d /cold/tair-kvcache/aiconfigurator ]]; then
  DEFAULT_AIC_SRC=/cold/tair-kvcache/aiconfigurator
else
  DEFAULT_AIC_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
fi

AIC_SRC="${AIC_SRC:-${DEFAULT_AIC_SRC}}"
PROFILE_SCRIPT="${AIC_SRC}/tools/moe_calibration/profile_dsv3_moe_dense_refresh.sh"
SGLANG_SRC="${SGLANG_SRC:-/sgl-workspace/sglang}"
MODEL_PATH="${MODEL_PATH:-/model/DeepSeek-V3}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"

PLATFORM="${PLATFORM:?set PLATFORM, for example h20, h100, or a new platform label}"
FAMILY="${FAMILY:?set FAMILY to ordinary_context, ordinary_generation, wideep_context, or wideep_generation}"
DATASET="${DATASET:-sharegpt}"
if [[ -z "${EPS:-}" ]]; then
  case "${FAMILY}" in
    ordinary_context|ordinary_generation) EPS="${DEFAULT_EPS:-1 2 4 8}" ;;
    wideep_context|wideep_generation) EPS="${DEFAULT_EPS:-2 4 8}" ;;
    *) echo "unknown family: ${FAMILY}" >&2; exit 2 ;;
  esac
fi
SESSIONS="${SESSIONS:-3}"

RESULTS_ROOT="${RESULTS_ROOT:-${AIC_SRC}/results/${PLATFORM}_recorded_moe_truth}"
case "${RESULTS_ROOT}" in
  /*) ;;
  *) RESULTS_ROOT="${AIC_SRC}/${RESULTS_ROOT}" ;;
esac

SHAREGPT_DATASET="${SHAREGPT_DATASET:-/model/ShareGPT_V3_unfiltered_cleaned_split.json}"
LONGBENCH_DATASET="${LONGBENCH_DATASET:-/model/data/longbench_sharegpt_format_for_moe_profile.json}"

case "${FAMILY}" in
  wideep_context)
    DEFAULT_CONTEXT_TOKENS="${DEFAULT_CONTEXT_TOKENS:-64 128 512 640 1536 2048 2560 4096 5120 8192 10240 12288 14336 16384 18888}"
    DEFAULT_GENERATION_TOKENS="${DEFAULT_GENERATION_TOKENS:-}"
    ;;
  wideep_generation)
    DEFAULT_CONTEXT_TOKENS="${DEFAULT_CONTEXT_TOKENS:-}"
    DEFAULT_GENERATION_TOKENS="${DEFAULT_GENERATION_TOKENS:-8 32 40 64 128 288 512 896 1024 1280}"
    ;;
  ordinary_context)
    DEFAULT_CONTEXT_TOKENS="${DEFAULT_CONTEXT_TOKENS:-4 8 32 64 128 512 640 896 1536 2048 4096 5120 8192 12288 16384}"
    DEFAULT_GENERATION_TOKENS="${DEFAULT_GENERATION_TOKENS:-}"
    ;;
  ordinary_generation)
    DEFAULT_CONTEXT_TOKENS="${DEFAULT_CONTEXT_TOKENS:-}"
    DEFAULT_GENERATION_TOKENS="${DEFAULT_GENERATION_TOKENS:-4 8 32 40 64 128 288 512 896 1024 1280 4096 8192 16384}"
    ;;
esac
CONTEXT_TOKENS="${CONTEXT_TOKENS:-${DEFAULT_CONTEXT_TOKENS}}"
GENERATION_TOKENS="${GENERATION_TOKENS:-${DEFAULT_GENERATION_TOKENS}}"
WARMUP_REPETITIONS="${WARMUP_REPETITIONS:-3}"

PROFILE_STOP_TIMEOUT="${PROFILE_STOP_TIMEOUT:-2400}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-1800}"
WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT:-3600}"
PARSE_WORKERS="${PARSE_WORKERS:-16}"
LAYER_AGGREGATION="${LAYER_AGGREGATION:-median}"
PROFILE_LAYERS="${PROFILE_LAYERS:-3 4 5}"
RUN_EPLB_OFF="${RUN_EPLB_OFF:-1}"
RUN_EPLB_ON="${RUN_EPLB_ON:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
CLEAN_PROFILES_AFTER_PARSE="${CLEAN_PROFILES_AFTER_PARSE:-1}"
DRY_RUN="${DRY_RUN:-0}"

dataset_env() {
  case "$1" in
    sharegpt)
      echo "PROMPT_SOURCE=sharegpt PROFILE_DATASET=${SHAREGPT_DATASET}"
      ;;
    longbench)
      echo "PROMPT_SOURCE=sharegpt PROFILE_DATASET=${LONGBENCH_DATASET}"
      ;;
    *)
      echo "unknown dataset: $1" >&2
      exit 2
      ;;
  esac
}

visible_gpu_count() {
  python3 - <<'PY'
import os
value = os.environ.get("CUDA_VISIBLE_DEVICES")
if value:
    print(len([item for item in value.split(",") if item.strip()]))
else:
    try:
        import subprocess
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        print(len([line for line in out.splitlines() if line.strip()]))
    except Exception:
        print(0)
PY
}

assert_ep_visible() {
  local ep="$1"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi
  local count
  count="$(visible_gpu_count)"
  if (( count < ep )); then
    echo "ERROR: EP${ep} requested but only ${count} GPU(s) visible." >&2
    exit 2
  fi
}

devices_for_ep() {
  local ep="$1"
  assert_ep_visible "${ep}"
  local devices=()
  local i
  for ((i = 0; i < ep; i++)); do
    devices+=("${i}")
  done
  local IFS=,
  echo "${devices[*]}"
}

base_port_for() {
  local family="$1"
  local ep="$2"
  local session="$3"
  local base
  case "${family}" in
    ordinary_context) base=37100 ;;
    ordinary_generation) base=37200 ;;
    wideep_context) base=37300 ;;
    wideep_generation) base=37400 ;;
    *) base=37500 ;;
  esac
  echo $((base + ep * 20 + session * 4))
}

deep_ep_env_args() {
  if [[ "${PLATFORM}" == "h20" ]]; then
    printf '%s\n' \
      "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-1024}" \
      "NVSHMEM_IBGDA_NIC_HANDLER=${NVSHMEM_IBGDA_NIC_HANDLER:-cpu}" \
      "NVSHMEM_IB_GID_INDEX=${NVSHMEM_IB_GID_INDEX:-3}" \
      "NVSHMEM_ENABLE_NIC_PE_MAPPING=${NVSHMEM_ENABLE_NIC_PE_MAPPING:-1}" \
      "NVSHMEM_HCA_PE_MAPPING=${NVSHMEM_HCA_PE_MAPPING:-mlx5_bond_0:1:2,mlx5_bond_1:1:2}" \
      "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-bond0}" \
      "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}" \
      "NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_bond_0,mlx5_bond_1}" \
      "NVSHMEM_BOOTSTRAP=${NVSHMEM_BOOTSTRAP:-UID}" \
      "NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=${NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME:-bond0}" \
      "NVSHMEM_SOCKET_IFNAME=${NVSHMEM_SOCKET_IFNAME:-bond0}"
  else
    printf '%s\n' \
      "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-1024}"
  fi
}

phase_for_family() {
  case "$1" in
    ordinary_context|wideep_context) echo context ;;
    ordinary_generation|wideep_generation) echo generation ;;
    *) echo "unknown family: $1" >&2; exit 2 ;;
  esac
}

family_dir() {
  case "$1" in
    wideep_generation) echo wideep_generation_low_latency ;;
    *) echo "$1" ;;
  esac
}

complete_phase_session() {
  local phase="$1"
  local out="$2"
  local require_eplb_on="$3"
  [[ -s "${out}/parsed/${phase}_dense_refresh_eplb_off_rank_aggregate.csv" ]] || return 1
  [[ "${require_eplb_on}" != "1" || -s "${out}/parsed/${phase}_dense_refresh_eplb_on_rank_aggregate.csv" ]] || return 1
}

write_manifest_header() {
  local dir
  dir="$(family_dir "${FAMILY}")"
  mkdir -p "${RESULTS_ROOT}/${DATASET}/${dir}"
cat >"${RESULTS_ROOT}/${DATASET}/${dir}/MANIFEST.md" <<EOF
# ${PLATFORM} ${DATASET} ${FAMILY} recorded MoE truth

- platform: ${PLATFORM}
- family: ${FAMILY}
- dataset: ${DATASET}
- eps: ${EPS}
- server TP_SIZE: EP_SIZE for each run
- AIC compare moe_tp_size: 1
- AIC compare moe_ep_size: EP_SIZE
- ordinary EP1 eplb_on: skipped
- sessions: ${SESSIONS}
- context tokens: ${CONTEXT_TOKENS}
- generation tokens: ${GENERATION_TOKENS}
- warmup repetitions: ${WARMUP_REPETITIONS}
- profile stop timeout: ${PROFILE_STOP_TIMEOUT}
- parse workers: ${PARSE_WORKERS}
- layer aggregation: ${LAYER_AGGREGATION}
- enable profile cuda graph: 0
- clean profiles after parse: ${CLEAN_PROFILES_AFTER_PARSE}
- skip completed: ${SKIP_COMPLETED}
- AIC_SRC: \`${AIC_SRC}\`
- SGLANG_SRC: \`${SGLANG_SRC}\`
- MODEL_PATH: \`${MODEL_PATH}\`
- TOKENIZER_PATH: \`${TOKENIZER_PATH}\`

Runs:

EOF
}

append_manifest_run() {
  local ep="$1"
  local session="$2"
  local output_dir="$3"
  local dir
  dir="$(family_dir "${FAMILY}")"
  cat >>"${RESULTS_ROOT}/${DATASET}/${dir}/MANIFEST.md" <<EOF
- ep=${ep}, session=${session}: \`${output_dir}\`
EOF
}

run_profile() {
  local output_dir="$1"
  shift
  mkdir -p "$(dirname "${output_dir}")"
  local -a cmd=(env "$@" "${PROFILE_SCRIPT}")
  echo
  echo "### ${output_dir}"
  printf '%q ' "${cmd[@]}"
  echo
  if [[ "${DRY_RUN}" != "1" ]]; then
    "${cmd[@]}" 2>&1 | tee "${output_dir}.run.log"
    if [[ "${CLEAN_PROFILES_AFTER_PARSE}" == "1" ]]; then
      rm -rf -- "${output_dir}/profiles_eplb_off" "${output_dir}/profiles_eplb_on"
    fi
  fi
}

run_one() {
  local ep="$1"
  local session="$2"
  local phase data_env out run_eplb_on dir
  phase="$(phase_for_family "${FAMILY}")"
  dir="$(family_dir "${FAMILY}")"
  data_env="$(dataset_env "${DATASET}")"
  out="${RESULTS_ROOT}/${DATASET}/${dir}/ep${ep}/session${session}"
  run_eplb_on="${RUN_EPLB_ON}"
  if [[ "${FAMILY}" == ordinary_* && "${ep}" == "1" ]]; then
    run_eplb_on=0
  fi

  append_manifest_run "${ep}" "${session}" "${out}"
  if [[ "${SKIP_COMPLETED}" == "1" ]] && complete_phase_session "${phase}" "${out}" "${run_eplb_on}"; then
    echo "Skipping completed ${FAMILY}: ${out}"
    return
  fi

  case "${FAMILY}" in
    ordinary_context)
      # shellcheck disable=SC2086
      run_profile "${out}" \
        AIC_SRC="${AIC_SRC}" SGLANG_SRC="${SGLANG_SRC}" MODEL_PATH="${MODEL_PATH}" TOKENIZER_PATH="${TOKENIZER_PATH}" \
        OUTPUT_DIR="${out}" CUDA_VISIBLE_DEVICES="$(devices_for_ep "${ep}")" \
        BASE_PORT="$(base_port_for "${FAMILY}" "${ep}" "${session}")" EP_SIZE="${ep}" TP_SIZE="${ep}" \
        MOE_A2A_BACKEND=none DEEPEP_MODE=none DISABLE_CUDA_GRAPH=1 ENABLE_PROFILE_CUDA_GRAPH=0 \
        RUN_CONTEXT=1 RUN_GENERATION=0 RUN_EPLB_OFF="${RUN_EPLB_OFF}" RUN_EPLB_ON="${run_eplb_on}" \
        CONTEXT_TOKENS="${CONTEXT_TOKENS}" SAMPLES=1 CONTEXT_WARMUP_REPETITIONS="${WARMUP_REPETITIONS}" \
        PROFILE_STAGE=collector/moe CONTEXT_TIMING_SOURCE=kernel_external_id \
        LAYER_AGGREGATION="${LAYER_AGGREGATION}" PROFILE_STOP_TIMEOUT="${PROFILE_STOP_TIMEOUT}" \
        SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT}" WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT}" \
        PARSE_WORKERS="${PARSE_WORKERS}" PROFILE_LAYERS="${PROFILE_LAYERS}" \
        ${data_env}
      ;;
    ordinary_generation)
      # shellcheck disable=SC2086
      run_profile "${out}" \
        AIC_SRC="${AIC_SRC}" SGLANG_SRC="${SGLANG_SRC}" MODEL_PATH="${MODEL_PATH}" TOKENIZER_PATH="${TOKENIZER_PATH}" \
        OUTPUT_DIR="${out}" CUDA_VISIBLE_DEVICES="$(devices_for_ep "${ep}")" \
        BASE_PORT="$(base_port_for "${FAMILY}" "${ep}" "${session}")" EP_SIZE="${ep}" TP_SIZE="${ep}" \
        MOE_A2A_BACKEND=none DEEPEP_MODE=none DISABLE_CUDA_GRAPH=0 ENABLE_PROFILE_CUDA_GRAPH=0 \
        RUN_CONTEXT=0 RUN_GENERATION=1 RUN_EPLB_OFF="${RUN_EPLB_OFF}" RUN_EPLB_ON="${run_eplb_on}" \
        GENERATION_TOKEN_SCOPE="${GENERATION_TOKEN_SCOPE:-local}" GENERATION_TOKENS="${GENERATION_TOKENS}" \
        PROFILE_STAGE=collector/moe GENERATION_TIMING_SOURCE=kernel_external_id GENERATION_OCCURRENCE_AGGREGATION=max_chunk \
        SAMPLES=1 GENERATION_REPETITIONS=1 GENERATION_WARMUP_REPETITIONS="${WARMUP_REPETITIONS}" \
        LAYER_AGGREGATION="${LAYER_AGGREGATION}" PROFILE_STOP_TIMEOUT="${PROFILE_STOP_TIMEOUT}" \
        SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT}" WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT}" \
        PARSE_WORKERS="${PARSE_WORKERS}" PROFILE_LAYERS="${PROFILE_LAYERS}" \
        ${data_env}
      ;;
    wideep_context)
      # shellcheck disable=SC2086
      run_profile "${out}" \
        AIC_SRC="${AIC_SRC}" SGLANG_SRC="${SGLANG_SRC}" MODEL_PATH="${MODEL_PATH}" TOKENIZER_PATH="${TOKENIZER_PATH}" \
        $(deep_ep_env_args) \
        OUTPUT_DIR="${out}" CUDA_VISIBLE_DEVICES="$(devices_for_ep "${ep}")" \
        BASE_PORT="$(base_port_for "${FAMILY}" "${ep}" "${session}")" EP_SIZE="${ep}" TP_SIZE="${ep}" \
        MOE_A2A_BACKEND=deepep DEEPEP_MODE=normal DISABLE_CUDA_GRAPH=1 ENABLE_PROFILE_CUDA_GRAPH=0 \
        RUN_CONTEXT=1 RUN_GENERATION=0 RUN_EPLB_OFF="${RUN_EPLB_OFF}" RUN_EPLB_ON="${run_eplb_on}" \
        CONTEXT_TOKENS="${CONTEXT_TOKENS}" SAMPLES=1 CONTEXT_WARMUP_REPETITIONS="${WARMUP_REPETITIONS}" \
        PROFILE_STAGE=routed/compute CONTEXT_TIMING_SOURCE=kernel_external_id \
        LAYER_AGGREGATION="${LAYER_AGGREGATION}" PROFILE_STOP_TIMEOUT="${PROFILE_STOP_TIMEOUT}" \
        SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT}" WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT}" \
        PARSE_WORKERS="${PARSE_WORKERS}" PROFILE_LAYERS="${PROFILE_LAYERS}" \
        ${data_env}
      ;;
    wideep_generation)
      # shellcheck disable=SC2086
      run_profile "${out}" \
        AIC_SRC="${AIC_SRC}" SGLANG_SRC="${SGLANG_SRC}" MODEL_PATH="${MODEL_PATH}" TOKENIZER_PATH="${TOKENIZER_PATH}" \
        $(deep_ep_env_args) \
        OUTPUT_DIR="${out}" CUDA_VISIBLE_DEVICES="$(devices_for_ep "${ep}")" \
        BASE_PORT="$(base_port_for "${FAMILY}" "${ep}" "${session}")" EP_SIZE="${ep}" TP_SIZE="${ep}" \
        MOE_A2A_BACKEND=deepep DEEPEP_MODE=low_latency DISABLE_CUDA_GRAPH=0 ENABLE_PROFILE_CUDA_GRAPH=0 \
        RUN_CONTEXT=0 RUN_GENERATION=1 RUN_EPLB_OFF="${RUN_EPLB_OFF}" RUN_EPLB_ON="${run_eplb_on}" \
        GENERATION_TOKEN_SCOPE=global GENERATION_TOKENS="${GENERATION_TOKENS}" \
        PROFILE_STAGE=routed/compute GENERATION_TIMING_SOURCE=auto GENERATION_OCCURRENCE_AGGREGATION=max_chunk \
        SAMPLES=1 GENERATION_REPETITIONS=1 GENERATION_WARMUP_REPETITIONS="${WARMUP_REPETITIONS}" \
        LAYER_AGGREGATION="${LAYER_AGGREGATION}" PROFILE_STOP_TIMEOUT="${PROFILE_STOP_TIMEOUT}" \
        SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT}" WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT}" \
        PARSE_WORKERS="${PARSE_WORKERS}" PROFILE_LAYERS="${PROFILE_LAYERS}" \
        ${data_env}
      ;;
    *)
      echo "unknown family: ${FAMILY}" >&2
      exit 2
      ;;
  esac
}

write_manifest_header

for ep in ${EPS}; do
  for ((session = 1; session <= SESSIONS; session++)); do
    run_one "${ep}" "${session}"
  done
done

echo
echo "Manifest: ${RESULTS_ROOT}/${DATASET}/$(family_dir "${FAMILY}")/MANIFEST.md"
