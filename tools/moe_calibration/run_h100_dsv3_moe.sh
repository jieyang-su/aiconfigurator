#!/usr/bin/env bash
set -euo pipefail

# Template runner for DeepSeekV3 MoE calibration on a remote H100 host.
# Edit the environment variables below, then run selected functions manually
# or run the whole script as-is for the single-GPU smoke + sweep path.

: "${SGLANG_SRC:=/workspace/sglang}"
: "${AIC_SRC:=/workspace/aiconfigurator-jieyang}"
: "${MODEL_PATH:=deepseek-ai/DeepSeek-V3}"
: "${AIC_MODEL_PATH:=${MODEL_PATH}}"
: "${CUDA_VISIBLE_DEVICES:=7}"
: "${GPU_LIST_DOCKER:=device=7}"
: "${SYSTEM:=h100_sxm}"
: "${BACKEND_VERSION:=0.5.9}"
: "${DEEPEP_DISPATCH_BACKEND_VERSION:=${BACKEND_VERSION}}"
: "${AIC_SYSTEMS_ROOT:=${AIC_SRC}/src/aiconfigurator/systems}"
: "${AIC_DATABASE_MODE:=SILICON}"
: "${AIC_TRACE_DURATION_COLUMN:=duration_us}"
: "${NUM_HIDDEN_LAYERS:=6}"
: "${FIRST_K_DENSE_REPLACE:=3}"
: "${EXPECTED_MOE_LAYERS:=$(( NUM_HIDDEN_LAYERS > FIRST_K_DENSE_REPLACE ? NUM_HIDDEN_LAYERS - FIRST_K_DENSE_REPLACE : 0 ))}"
: "${TP_SIZE:=1}"
: "${EP_SIZE:=1}"
: "${MOE_TP_SIZE:=1}"
: "${MOE_EP_SIZE:=1}"
: "${HIDDEN_SIZE:=7168}"
: "${INTER_SIZE:=2048}"
: "${TOPK:=8}"
: "${NUM_EXPERTS:=256}"
: "${QUANT_MODE:=bfloat16}"
: "${DISTRIBUTION:=balanced}"
: "${WIDEEP_DISTRIBUTION:=uniform}"
: "${DEEPEP_NODE_NUM:=1}"
: "${DEEPEP_SMS:=20}"
: "${AIC_COLLECTOR_PLAN_LIMIT:=16}"
: "${AIC_COLLECTOR_LIMIT:=4}"
: "${TOKEN_SWEEP:=128 512 2048 4096}"
: "${INTERP_TOKEN_SWEEP:=96 160 384 768 1536 3072 6144}"
: "${OUT_DIR:=/workspace/moe_calibration_runs/h100_dsv3_moe_$(date +%Y%m%d_%H%M%S)}"

init_output() {
  mkdir -p "${OUT_DIR}"/{profile,logs,aic,parsed,configs}
  exec 3> "${OUT_DIR}/logs/commands.xtrace.log"
  export BASH_XTRACEFD=3
  set -x
}

sglang_common_args=(
  --model-path "${MODEL_PATH}"
  --load-format dummy
  --tp-size "${TP_SIZE}"
  --ep-size "${EP_SIZE}"
  --json-model-override-args "{\"num_hidden_layers\":${NUM_HIDDEN_LAYERS},\"first_k_dense_replace\":${FIRST_K_DENSE_REPLACE}}"
  --disable-cuda-graph
  --disable-shared-experts-fusion
)

record_env() {
  cat > "${OUT_DIR}/configs/run_manifest.env" <<EOF
SGLANG_SRC=${SGLANG_SRC}
AIC_SRC=${AIC_SRC}
MODEL_PATH=${MODEL_PATH}
AIC_MODEL_PATH=${AIC_MODEL_PATH}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
SYSTEM=${SYSTEM}
BACKEND_VERSION=${BACKEND_VERSION}
DEEPEP_DISPATCH_BACKEND_VERSION=${DEEPEP_DISPATCH_BACKEND_VERSION}
AIC_DATABASE_MODE=${AIC_DATABASE_MODE}
AIC_TRACE_DURATION_COLUMN=${AIC_TRACE_DURATION_COLUMN}
NUM_HIDDEN_LAYERS=${NUM_HIDDEN_LAYERS}
FIRST_K_DENSE_REPLACE=${FIRST_K_DENSE_REPLACE}
EXPECTED_MOE_LAYERS=${EXPECTED_MOE_LAYERS}
TP_SIZE=${TP_SIZE}
EP_SIZE=${EP_SIZE}
MOE_TP_SIZE=${MOE_TP_SIZE}
MOE_EP_SIZE=${MOE_EP_SIZE}
HIDDEN_SIZE=${HIDDEN_SIZE}
INTER_SIZE=${INTER_SIZE}
TOPK=${TOPK}
NUM_EXPERTS=${NUM_EXPERTS}
QUANT_MODE=${QUANT_MODE}
DISTRIBUTION=${DISTRIBUTION}
WIDEEP_DISTRIBUTION=${WIDEEP_DISTRIBUTION}
DEEPEP_NODE_NUM=${DEEPEP_NODE_NUM}
DEEPEP_SMS=${DEEPEP_SMS}
AIC_COLLECTOR_PLAN_LIMIT=${AIC_COLLECTOR_PLAN_LIMIT}
AIC_COLLECTOR_LIMIT=${AIC_COLLECTOR_LIMIT}
TOKEN_SWEEP=${TOKEN_SWEEP}
INTERP_TOKEN_SWEEP=${INTERP_TOKEN_SWEEP}
OUT_DIR=${OUT_DIR}
AIC_SYSTEMS_ROOT=${AIC_SYSTEMS_ROOT}
DISABLE_SHARED_EXPERTS_FUSION=1
EOF
  cd "${SGLANG_SRC}"
  git rev-parse HEAD | tee "${OUT_DIR}/configs/sglang_commit.txt"
  cd "${AIC_SRC}"
  git rev-parse HEAD | tee "${OUT_DIR}/configs/aic_commit.txt"
  nvidia-smi --query-gpu=index,name,uuid,compute_cap,driver_version,memory.total --format=csv \
    | tee "${OUT_DIR}/configs/gpu_info.csv"
  python - <<'PY' | tee "${OUT_DIR}/configs/python_packages.txt"
import importlib.metadata as m
for pkg in ["torch", "sglang", "transformers", "flashinfer-python"]:
    try:
        print(pkg, m.version(pkg))
    except Exception as exc:
        print(pkg, "NA", exc)
PY
}

write_stage_alignment_map() {
  cat > "${OUT_DIR}/configs/stage_alignment_map.csv" <<EOF
sglang_stage,aic_target,primary_compare_csv,notes
module,DeepSeek MoE forward closure,sglang_aic_moe_module_breakdown_summary.csv,Full non-TBO forward closure check; includes router shared experts routed path output postprocess and optional all-reduce. Do not compare this directly with a single AIC MoE table.
topk+routed/compute,PerfDatabase.query_moe moe_backend empty,aic_vs_sglang_topk_compute.csv,Strict ordinary AIC collector boundary: select_experts plus fused_moe kernel compute; excludes DeepSeek router shared experts output postprocess and SGLang dispatcher/combine wrapper overhead.
collector/moe,SGLang routed wrapper sanity check,aic_vs_sglang_collector_moe.csv,Full SGLang routed path wrapper: topk plus self.experts; may include dispatcher/combine wrapper overhead around fused_moe.
topk,select_experts breakdown,sglang_aic_moe_summary.csv,Used with routed/compute to match ordinary AIC moe_perf; not a standalone AIC table in this workflow.
routed_experts,FusedMoE end-to-end routed path,sglang_aic_moe_summary.csv,Contains dispatch compute combine and possible all-reduce depending on backend.
routed/compute,PerfDatabase.query_moe moe_backend deepep_moe,aic_vs_sglang_wideep_compute.csv,Expert core compute boundary; combines with topk for strict ordinary AIC moe_perf and aligns alone with WideEP run_moe_core data.
routed/dispatch+routed/combine,PerfDatabase.query_wideep_deepep_normal or ll,aic_vs_sglang_deepep_dispatch_combine.csv,Non-TBO DeepEP communication boundary.
routed/dispatch_a+routed/dispatch_b+routed/combine_a+routed/combine_b,PerfDatabase.query_wideep_deepep_normal or ll,aic_vs_sglang_deepep_tbo_dispatch_combine.csv,TBO segmented DeepEP communication boundary.
shared_experts,DeepSeek shared expert breakdown,sglang_aic_moe_module_breakdown_summary.csv,Not part of ordinary moe_perf; use for module residual attribution.
router,DeepSeek gate breakdown,sglang_aic_moe_module_breakdown_summary.csv,Not part of ordinary moe_perf; use for module residual attribution.
output_postprocess,MoE output elementwise tail,sglang_aic_moe_module_breakdown_summary.csv,Routed scaling and shared-output add; no direct ordinary moe_perf target.
output_all_reduce,Tensor parallel all-reduce tail,sglang_aic_moe_module_breakdown_summary.csv,Only present when TP path executes an unfused all-reduce.
EOF
}

check_code() {
  if (( EXPECTED_MOE_LAYERS <= 0 )); then
    echo "EXPECTED_MOE_LAYERS=${EXPECTED_MOE_LAYERS}; increase NUM_HIDDEN_LAYERS above FIRST_K_DENSE_REPLACE=${FIRST_K_DENSE_REPLACE} to include DeepSeekV3 MoE layers." >&2
    exit 1
  fi

  cd "${SGLANG_SRC}"
  grep -R "SGLANG_AIC_MOE_PROFILE" -n \
    python/sglang/srt/models/deepseek_v2.py \
    python/sglang/srt/layers/moe/fused_moe_triton/layer.py
  python -m py_compile \
    python/sglang/srt/models/deepseek_v2.py \
    python/sglang/srt/layers/moe/fused_moe_triton/layer.py

  cd "${AIC_SRC}"
  python -m py_compile \
    collector/collect.py \
    collector/sglang/collect_moe.py \
    collector/wideep/sglang/collect_deepep_moe.py \
    tools/moe_calibration/aic_moe_calibrate.py
  python tools/moe_calibration/aic_moe_calibrate.py self-test

}

run_bench_one_batch_prefill() {
  local isl="$1"
  local tag="$2"
  cd "${SGLANG_SRC}/python"
  export PYTHONPATH="${SGLANG_SRC}/python:${PYTHONPATH:-}"
  export CUDA_VISIBLE_DEVICES
  export SGLANG_AIC_MOE_PROFILE=1
  export SGLANG_TORCH_PROFILER_DIR="${OUT_DIR}/profile/${tag}_isl_${isl}"
  mkdir -p "${SGLANG_TORCH_PROFILER_DIR}"

  python -m sglang.bench_one_batch \
    "${sglang_common_args[@]}" \
    --batch-size 1 \
    --input-len "${isl}" \
    --output-len 1 \
    --profile \
    --profile-stage prefill \
    --profile-filename-prefix "dsv3_moe_${tag}_isl_${isl}" \
    2>&1 | tee "${OUT_DIR}/logs/sglang_${tag}_isl_${isl}.log"
}

run_bench_one_batch_decode() {
  local isl="$1"
  local osl="$2"
  cd "${SGLANG_SRC}/python"
  export PYTHONPATH="${SGLANG_SRC}/python:${PYTHONPATH:-}"
  export CUDA_VISIBLE_DEVICES
  export SGLANG_AIC_MOE_PROFILE=1
  export SGLANG_TORCH_PROFILER_DIR="${OUT_DIR}/profile/decode_ntok_1_isl_${isl}_osl_${osl}"
  mkdir -p "${SGLANG_TORCH_PROFILER_DIR}"

  python -m sglang.bench_one_batch \
    "${sglang_common_args[@]}" \
    --batch-size 1 \
    --input-len "${isl}" \
    --output-len "${osl}" \
    --profile \
    --profile-stage decode \
    --profile-filename-prefix "dsv3_moe_decode_ntok_1_isl_${isl}_osl_${osl}" \
    2>&1 | tee "${OUT_DIR}/logs/sglang_decode_ntok_1_isl_${isl}_osl_${osl}.log"
}

run_engine_offline_prefill() {
  local isl="$1"
  cd "${SGLANG_SRC}/python"
  export PYTHONPATH="${SGLANG_SRC}/python:${PYTHONPATH:-}"
  export CUDA_VISIBLE_DEVICES
  export SGLANG_AIC_MOE_PROFILE=1
  export SGLANG_TORCH_PROFILER_DIR="${OUT_DIR}/profile/engine_prefill_isl_${isl}"
  mkdir -p "${SGLANG_TORCH_PROFILER_DIR}"

  python -m sglang.bench_offline_throughput \
    "${sglang_common_args[@]}" \
    --backend engine \
    --dataset-name random \
    --random-input-len "${isl}" \
    --random-output-len 1 \
    --num-prompts 1 \
    --profile \
    --skip-warmup \
    --result-filename "${OUT_DIR}/logs/engine_prefill_isl_${isl}.jsonl" \
    2>&1 | tee "${OUT_DIR}/logs/sglang_engine_prefill_isl_${isl}.log"
}

print_server_commands() {
  cat <<EOF
# Start server:
cd "${SGLANG_SRC}/python"
export PYTHONPATH="${SGLANG_SRC}/python:\${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
export SGLANG_AIC_MOE_PROFILE=1
export SGLANG_TORCH_PROFILER_DIR="${OUT_DIR}/profile/server"
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR="${OUT_DIR}/expert_distribution"
mkdir -p "\${SGLANG_TORCH_PROFILER_DIR}" "\${SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR}"
python -m sglang.launch_server \\
  --model-path "${MODEL_PATH}" \\
  --load-format dummy \\
  --tp-size "${TP_SIZE}" \\
  --ep-size "${EP_SIZE}" \\
  --json-model-override-args '{"num_hidden_layers":${NUM_HIDDEN_LAYERS},"first_k_dense_replace":${FIRST_K_DENSE_REPLACE}}' \\
  --disable-cuda-graph \\
  --disable-shared-experts-fusion \\
  --expert-distribution-recorder-mode stat \\
  --expert-distribution-recorder-buffer-size -1 \\
  --host 0.0.0.0 \\
  --port 30000

# Start profile:
curl -X POST http://127.0.0.1:30000/start_profile \\
  -H 'Content-Type: application/json' \\
  -d '{"profile_by_stage": false, "profile_prefix": "dsv3_moe_server"}'

# Start expert distribution recording:
curl -X POST http://127.0.0.1:30000/start_expert_distribution_record

# Send one request:
curl -X POST http://127.0.0.1:30000/generate \\
  -H 'Content-Type: application/json' \\
  -d '{"text": "hello", "sampling_params": {"temperature": 0, "max_new_tokens": 1}}'

# Stop and dump expert distribution:
curl -X POST http://127.0.0.1:30000/stop_expert_distribution_record
curl -X POST http://127.0.0.1:30000/dump_expert_distribution_record

# Stop profile:
curl -X POST http://127.0.0.1:30000/stop_profile

# Convert expert distribution .pt records to CSV summary:
cd "${AIC_SRC}"
export PYTHONPATH="${AIC_SRC}/src:${AIC_SRC}:\${PYTHONPATH:-}"
python tools/moe_calibration/aic_moe_calibrate.py summarize-expert-distribution \\
  --input-dir "${OUT_DIR}/expert_distribution" \\
  --first-moe-layer-id "${FIRST_K_DENSE_REPLACE}" \\
  --output "${OUT_DIR}/parsed/sglang_expert_distribution_summary.csv"
EOF
}

parse_traces() {
  cd "${AIC_SRC}"
  export PYTHONPATH="${AIC_SRC}/src:${AIC_SRC}:${PYTHONPATH:-}"
  python tools/moe_calibration/aic_moe_calibrate.py parse-trace \
    --trace-root "${OUT_DIR}/profile" \
    --output "${OUT_DIR}/parsed/sglang_aic_moe_events.csv"
  python tools/moe_calibration/aic_moe_calibrate.py validate-trace \
    --trace-csv "${OUT_DIR}/parsed/sglang_aic_moe_events.csv" \
    --expected-layers "${EXPECTED_MOE_LAYERS}" \
    --output "${OUT_DIR}/parsed/sglang_aic_moe_trace_validation.csv"
  python tools/moe_calibration/aic_moe_calibrate.py summarize-trace \
    --trace-csv "${OUT_DIR}/parsed/sglang_aic_moe_events.csv" \
    --stage module router collector/moe topk shared_experts routed_experts routed/dispatch routed/compute routed/combine routed/dispatch_a routed/dispatch_b routed/combine_a routed/combine_b routed/all_reduce output_postprocess output_all_reduce \
    --duration-column "${AIC_TRACE_DURATION_COLUMN}" \
    --output "${OUT_DIR}/parsed/sglang_aic_moe_summary.csv"
  python tools/moe_calibration/aic_moe_calibrate.py breakdown-trace \
    --trace-csv "${OUT_DIR}/parsed/sglang_aic_moe_events.csv" \
    --component router collector/moe shared_experts output_postprocess output_all_reduce \
    --output "${OUT_DIR}/parsed/sglang_aic_moe_module_breakdown.csv" \
    --summary-output "${OUT_DIR}/parsed/sglang_aic_moe_module_breakdown_summary.csv"
}

query_aic() {
  cd "${AIC_SRC}"
  export PYTHONPATH="${AIC_SRC}/src:${AIC_SRC}:${PYTHONPATH:-}"
  python tools/moe_calibration/aic_moe_calibrate.py query-aic \
    --systems-root "${AIC_SYSTEMS_ROOT}" \
    --system "${SYSTEM}" \
    --backend sglang \
    --backend-version "${BACKEND_VERSION}" \
    --database-mode "${AIC_DATABASE_MODE}" \
    --phase context \
    --num-tokens ${TOKEN_SWEEP} ${INTERP_TOKEN_SWEEP} \
    --hidden-size "${HIDDEN_SIZE}" \
    --inter-size "${INTER_SIZE}" \
    --topk "${TOPK}" \
    --num-experts "${NUM_EXPERTS}" \
    --quant-mode "${QUANT_MODE}" \
    --distribution "${DISTRIBUTION}" \
    --moe-tp-size "${MOE_TP_SIZE}" \
    --moe-ep-size "${MOE_EP_SIZE}" \
    --output "${OUT_DIR}/parsed/aic_moe_predictions.csv"
}

query_wideep_compute_aic() {
  cd "${AIC_SRC}"
  export PYTHONPATH="${AIC_SRC}/src:${AIC_SRC}:${PYTHONPATH:-}"
  python tools/moe_calibration/aic_moe_calibrate.py query-aic \
    --systems-root "${AIC_SYSTEMS_ROOT}" \
    --system "${SYSTEM}" \
    --backend sglang \
    --backend-version "${BACKEND_VERSION}" \
    --database-mode "${AIC_DATABASE_MODE}" \
    --phase context \
    --num-tokens ${TOKEN_SWEEP} ${INTERP_TOKEN_SWEEP} \
    --hidden-size "${HIDDEN_SIZE}" \
    --inter-size "${INTER_SIZE}" \
    --topk "${TOPK}" \
    --num-experts "${NUM_EXPERTS}" \
    --quant-mode "${QUANT_MODE}" \
    --distribution "${WIDEEP_DISTRIBUTION}" \
    --moe-tp-size "${MOE_TP_SIZE}" \
    --moe-ep-size "${MOE_EP_SIZE}" \
    --moe-backend deepep_moe \
    --output "${OUT_DIR}/parsed/aic_wideep_compute_predictions.csv"
}

compare_real_vs_aic() {
  cd "${AIC_SRC}"
  export PYTHONPATH="${AIC_SRC}/src:${AIC_SRC}:${PYTHONPATH:-}"
  python tools/moe_calibration/aic_moe_calibrate.py compare \
    --trace-csv "${OUT_DIR}/parsed/sglang_aic_moe_events.csv" \
    --aic-csv "${OUT_DIR}/parsed/aic_moe_predictions.csv" \
    --real-stage topk+routed/compute \
    --distribution "${DISTRIBUTION}" \
    --phase context \
    --duration-column "${AIC_TRACE_DURATION_COLUMN}" \
    --output "${OUT_DIR}/parsed/aic_vs_sglang_topk_compute.csv"

  python "${AIC_SRC}/tools/moe_calibration/aic_moe_calibrate.py" compare \
    --trace-csv "${OUT_DIR}/parsed/sglang_aic_moe_events.csv" \
    --aic-csv "${OUT_DIR}/parsed/aic_moe_predictions.csv" \
    --real-stage collector/moe \
    --distribution "${DISTRIBUTION}" \
    --phase context \
    --duration-column "${AIC_TRACE_DURATION_COLUMN}" \
    --output "${OUT_DIR}/parsed/aic_vs_sglang_collector_moe.csv"
}

compare_wideep_compute_real_vs_aic() {
  cd "${AIC_SRC}"
  export PYTHONPATH="${AIC_SRC}/src:${AIC_SRC}:${PYTHONPATH:-}"
  python tools/moe_calibration/aic_moe_calibrate.py compare \
    --trace-csv "${OUT_DIR}/parsed/sglang_aic_moe_events.csv" \
    --aic-csv "${OUT_DIR}/parsed/aic_wideep_compute_predictions.csv" \
    --real-stage routed/compute \
    --distribution "${WIDEEP_DISTRIBUTION}" \
    --phase context \
    --duration-column "${AIC_TRACE_DURATION_COLUMN}" \
    --output "${OUT_DIR}/parsed/aic_vs_sglang_wideep_compute.csv"
}

make_calibration_report() {
  cd "${AIC_SRC}"
  export PYTHONPATH="${AIC_SRC}/src:${AIC_SRC}:${PYTHONPATH:-}"
  python tools/moe_calibration/aic_moe_calibrate.py make-report \
    --compare-csv \
      "${OUT_DIR}/parsed/aic_vs_sglang_topk_compute.csv" \
      "${OUT_DIR}/parsed/aic_vs_sglang_collector_moe.csv" \
      "${OUT_DIR}/parsed/aic_vs_sglang_generation_topk_compute.csv" \
      "${OUT_DIR}/parsed/aic_vs_sglang_generation_collector_moe.csv" \
      "${OUT_DIR}/parsed/aic_vs_sglang_wideep_compute.csv" \
      "${OUT_DIR}/parsed/aic_vs_sglang_deepep_dispatch_combine.csv" \
      "${OUT_DIR}/parsed/aic_vs_sglang_deepep_tbo_dispatch_combine.csv" \
    --breakdown-summary-csv "${OUT_DIR}/parsed/sglang_aic_moe_module_breakdown_summary.csv" \
    --expert-distribution-csv "${OUT_DIR}/parsed/sglang_expert_distribution_summary.csv" \
    --stage-map-csv "${OUT_DIR}/configs/stage_alignment_map.csv" \
    --output "${OUT_DIR}/parsed/deepseekv3_moe_calibration_report.md"
}

query_deepep_dispatch_aic() {
  cd "${AIC_SRC}"
  export PYTHONPATH="${AIC_SRC}/src:${AIC_SRC}:${PYTHONPATH:-}"
  python tools/moe_calibration/aic_moe_calibrate.py query-deepep-dispatch \
    --systems-root "${AIC_SYSTEMS_ROOT}" \
    --system "${SYSTEM}" \
    --backend sglang \
    --backend-version "${DEEPEP_DISPATCH_BACKEND_VERSION}" \
    --database-mode "${AIC_DATABASE_MODE}" \
    --phase context \
    --num-tokens ${TOKEN_SWEEP} ${INTERP_TOKEN_SWEEP} \
    --hidden-size "${HIDDEN_SIZE}" \
    --topk "${TOPK}" \
    --num-experts "${NUM_EXPERTS}" \
    --node-num "${DEEPEP_NODE_NUM}" \
    --sms "${DEEPEP_SMS}" \
    --deepep-mode normal \
    --distribution "${WIDEEP_DISTRIBUTION}" \
    --output "${OUT_DIR}/parsed/aic_deepep_dispatch_predictions.csv"
}

compare_deepep_dispatch_real_vs_aic() {
  cd "${AIC_SRC}"
  export PYTHONPATH="${AIC_SRC}/src:${AIC_SRC}:${PYTHONPATH:-}"
  python tools/moe_calibration/aic_moe_calibrate.py compare \
    --trace-csv "${OUT_DIR}/parsed/sglang_aic_moe_events.csv" \
    --aic-csv "${OUT_DIR}/parsed/aic_deepep_dispatch_predictions.csv" \
    --real-stage routed/dispatch+routed/combine \
    --distribution "${WIDEEP_DISTRIBUTION}" \
    --phase context \
    --duration-column "${AIC_TRACE_DURATION_COLUMN}" \
    --output "${OUT_DIR}/parsed/aic_vs_sglang_deepep_dispatch_combine.csv"
}

compare_deepep_tbo_real_vs_aic() {
  cd "${AIC_SRC}"
  export PYTHONPATH="${AIC_SRC}/src:${AIC_SRC}:${PYTHONPATH:-}"
  python tools/moe_calibration/aic_moe_calibrate.py compare \
    --trace-csv "${OUT_DIR}/parsed/sglang_aic_moe_events.csv" \
    --aic-csv "${OUT_DIR}/parsed/aic_deepep_dispatch_predictions.csv" \
    --real-stage routed/dispatch_a+routed/dispatch_b+routed/combine_a+routed/combine_b \
    --distribution "${WIDEEP_DISTRIBUTION}" \
    --phase context \
    --duration-column "${AIC_TRACE_DURATION_COLUMN}" \
    --output "${OUT_DIR}/parsed/aic_vs_sglang_deepep_tbo_dispatch_combine.csv"
}

summarize_expert_distribution() {
  cd "${AIC_SRC}"
  export PYTHONPATH="${AIC_SRC}/src:${AIC_SRC}:${PYTHONPATH:-}"
  python tools/moe_calibration/aic_moe_calibrate.py summarize-expert-distribution \
    --input-dir "${OUT_DIR}/expert_distribution" \
    --first-moe-layer-id "${FIRST_K_DENSE_REPLACE}" \
    --output "${OUT_DIR}/parsed/sglang_expert_distribution_summary.csv"
}

build_aic_systems_overlay() {
  local collector_dir="${1:-${OUT_DIR}/aic/collector_moe_smoke}"
  local overlay_root="${OUT_DIR}/aic_systems_overlay"
  local target_dir="${overlay_root}/data/${SYSTEM}/sglang/${BACKEND_VERSION}"
  local dispatch_target_dir="${overlay_root}/data/${SYSTEM}/sglang/${DEEPEP_DISPATCH_BACKEND_VERSION}"

  case "${overlay_root}" in
    "${OUT_DIR}"/*) ;;
    *) echo "Refusing to remove overlay outside OUT_DIR: ${overlay_root}" >&2; exit 1 ;;
  esac
  rm -rf "${overlay_root}"
  mkdir -p "${overlay_root}" "${target_dir}" "${dispatch_target_dir}"
  cp -a "${AIC_SRC}/src/aiconfigurator/systems/." "${overlay_root}/"

  # The collector writes *_perf.txt staging files first. Convert them to
  # parquet while keeping the text files for debugging, then copy the parquet
  # files into the PerfDatabase layout used by query_moe.
  PYTHONPATH="${AIC_SRC}/src:${AIC_SRC}:${PYTHONPATH:-}" python - <<PY
from collector.helper import finalize_perf_outputs
finalize_perf_outputs("${collector_dir}", recursive=True, delete_source=False)
PY

  mapfile -t moe_files < <(find "${collector_dir}" -name "moe_perf.parquet" -print)
  if (( ${#moe_files[@]} == 0 )); then
    echo "No moe_perf.parquet found under ${collector_dir}; collector output cannot drive AIC query." >&2
    exit 1
  fi
  cp "${moe_files[0]}" "${target_dir}/moe_perf.parquet"

  mapfile -t wideep_context_files < <(find "${collector_dir}" -name "wideep_context_moe_perf.parquet" -print)
  if (( ${#wideep_context_files[@]} > 0 )); then
    cp "${wideep_context_files[0]}" "${target_dir}/wideep_context_moe_perf.parquet"
  fi
  mapfile -t wideep_generation_files < <(find "${collector_dir}" -name "wideep_generation_moe_perf.parquet" -print)
  if (( ${#wideep_generation_files[@]} > 0 )); then
    cp "${wideep_generation_files[0]}" "${target_dir}/wideep_generation_moe_perf.parquet"
  fi
  mapfile -t deepep_normal_files < <(find "${collector_dir}" -name "wideep_deepep_normal_perf.parquet" -print)
  if (( ${#deepep_normal_files[@]} > 0 )); then
    cp "${deepep_normal_files[0]}" "${dispatch_target_dir}/wideep_deepep_normal_perf.parquet"
  fi
  mapfile -t deepep_ll_files < <(find "${collector_dir}" -name "wideep_deepep_ll_perf.parquet" -print)
  if (( ${#deepep_ll_files[@]} > 0 )); then
    cp "${deepep_ll_files[0]}" "${dispatch_target_dir}/wideep_deepep_ll_perf.parquet"
  fi

  export AIC_SYSTEMS_ROOT="${overlay_root}"
  echo "${AIC_SYSTEMS_ROOT}" | tee "${OUT_DIR}/configs/aic_systems_root.txt"
  find "${target_dir}" "${dispatch_target_dir}" -maxdepth 1 -type f -name "*moe*perf.parquet" -print \
    | sort -u | tee "${OUT_DIR}/configs/aic_overlay_files.txt"
}

run_aic_collector_smoke() {
  local collector_root="${OUT_DIR}/aic/collector_moe_smoke"
  mkdir -p "${collector_root}"
  cd "${collector_root}"
  export PYTHONPATH="${AIC_SRC}/src:${AIC_SRC}:${PYTHONPATH:-}"
  export CUDA_VISIBLE_DEVICES
  python "${AIC_SRC}/collector/collect.py" \
    --backend sglang \
    --model-path "${AIC_MODEL_PATH}" \
    --ops moe \
    --sm 90 \
    --limit "${AIC_COLLECTOR_LIMIT}" \
    --keep-csv \
    2>&1 | tee "${OUT_DIR}/logs/aic_collect_moe_smoke.log"
}

run_aic_collector_plan() {
  local collector_root="${OUT_DIR}/aic/collector_moe_plan"
  mkdir -p "${collector_root}"
  cd "${collector_root}"
  export PYTHONPATH="${AIC_SRC}/src:${AIC_SRC}:${PYTHONPATH:-}"
  python "${AIC_SRC}/collector/collect.py" \
    --backend sglang \
    --model-path "${AIC_MODEL_PATH}" \
    --ops moe \
    --sm 90 \
    --limit "${AIC_COLLECTOR_PLAN_LIMIT}" \
    --plan-only \
    2>&1 | tee "${OUT_DIR}/logs/aic_collect_moe_plan.log"
}

main() {
  if [[ "${1:-}" == "print-server-commands" ]]; then
    print_server_commands
    return 0
  fi
  init_output
  record_env
  write_stage_alignment_map
  check_code
  if [[ "${1:-}" == "check-only" ]]; then
    echo "Check-only passed: ${OUT_DIR}"
    return 0
  fi
  run_bench_one_batch_prefill 128 smoke_prefill
  for isl in ${TOKEN_SWEEP}; do
    if [[ "${isl}" == "128" ]]; then
      continue
    fi
    run_bench_one_batch_prefill "${isl}" direct_prefill
  done
  for isl in ${INTERP_TOKEN_SWEEP}; do
    run_bench_one_batch_prefill "${isl}" interp_prefill
  done
  parse_traces
  run_aic_collector_plan
  run_aic_collector_smoke
  build_aic_systems_overlay "${OUT_DIR}/aic/collector_moe_smoke"
  query_aic
  compare_real_vs_aic
  make_calibration_report
  tar -czf "${OUT_DIR}.tar.gz" -C "$(dirname "${OUT_DIR}")" "$(basename "${OUT_DIR}")"
  echo "Done: ${OUT_DIR}"
  echo "Archive: ${OUT_DIR}.tar.gz"
}

main "$@"
