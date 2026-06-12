# DeepSeekV3 MoE H100 Smoke Checklist

This checklist verifies that one H100 smoke run produced calibration-ready MoE data.

## 1. Syntax Gate

Run inside the H100 container:

```bash
cd "${SGLANG_SRC}"
python -m py_compile \
  python/sglang/srt/models/deepseek_v2.py \
  python/sglang/srt/layers/moe/fused_moe_triton/layer.py

cd "${AIC_SRC}"
python -m py_compile tools/moe_calibration/aic_moe_calibrate.py
python tools/moe_calibration/aic_moe_calibrate.py self-test
bash -n tools/moe_calibration/run_h100_dsv3_moe.sh

# Optional: run the runner built-in checks without launching SGLang/collector.
bash tools/moe_calibration/run_h100_dsv3_moe.sh check-only
```

## 2. Minimal Smoke

```bash
cd "${AIC_SRC}"
SGLANG_SRC="${SGLANG_SRC}" \
AIC_SRC="${AIC_SRC}" \
MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-V3}" \
CUDA_VISIBLE_DEVICES=7 \
NUM_HIDDEN_LAYERS=6 \
FIRST_K_DENSE_REPLACE=3 \
EXPECTED_MOE_LAYERS=3 \
TOKEN_SWEEP="128" \
INTERP_TOKEN_SWEEP="" \
OUT_DIR="${OUT_DIR}" \
bash tools/moe_calibration/run_h100_dsv3_moe.sh
```

The runner passes `--disable-shared-experts-fusion` by default. Keep this enabled for ordinary `moe_perf` calibration so `collector/moe` excludes DeepSeek shared experts.

## 3. Required Files

```bash
test -s "${OUT_DIR}/parsed/sglang_aic_moe_events.csv"
test -s "${OUT_DIR}/parsed/sglang_aic_moe_trace_validation.csv"
test -s "${OUT_DIR}/parsed/sglang_aic_moe_summary.csv"
test -s "${OUT_DIR}/parsed/aic_vs_sglang_topk_compute.csv"
test -s "${OUT_DIR}/parsed/aic_vs_sglang_collector_moe.csv"
test -s "${OUT_DIR}/configs/stage_alignment_map.csv"
grep -q "^DISABLE_SHARED_EXPERTS_FUSION=1$" "${OUT_DIR}/configs/run_manifest.env"
```

## 4. Trace Coverage

```bash
python tools/moe_calibration/aic_moe_calibrate.py validate-trace \
  --trace-csv "${OUT_DIR}/parsed/sglang_aic_moe_events.csv" \
  --required-stage module router collector/moe topk shared_experts routed_experts routed/compute output_postprocess \
  --expected-layers "${EXPECTED_MOE_LAYERS}" \
  --output "${OUT_DIR}/parsed/sglang_aic_moe_trace_validation_strict.csv" \
  --fail-on-missing
```

Pass condition: no `missing` or `incomplete_layers` for required stages. If `collector/moe` or `shared_experts` is missing in ordinary single-GPU smoke, first check that `--disable-shared-experts-fusion` was present in the SGLang command.

## 5. Layer Id Sanity

For the default `NUM_HIDDEN_LAYERS=6` and `FIRST_K_DENSE_REPLACE=3`, the MoE layer ids must be `3 4 5`, not `0 1 2`:

```bash
python - <<'PY'
import csv
import os

path = os.path.join(os.environ["OUT_DIR"], "parsed", "sglang_aic_moe_trace_validation_strict.csv")
expected = " ".join(str(i) for i in range(
    int(os.environ["FIRST_K_DENSE_REPLACE"]),
    int(os.environ["NUM_HIDDEN_LAYERS"]),
))
required = {"module", "router", "collector/moe", "topk", "shared_experts", "routed_experts", "routed/compute", "output_postprocess"}

with open(path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

bad = []
for row in rows:
    if row["phase"] == "context" and row["num_tokens"] == "128" and row["stage"] in required:
        if row["layer_ids"] != expected or row["status"] != "ok":
            bad.append(row)

if bad:
    raise SystemExit(f"Unexpected MoE layer coverage; expected {expected}, got {bad[:3]}")
print(f"MoE layer ids OK: {expected}")
PY
```

If expert distribution was recorded, verify its `layer_id` also uses the real DeepSeekV3 layer id:

```bash
if test -s "${OUT_DIR}/parsed/sglang_expert_distribution_summary.csv"; then
  column -s, -t < "${OUT_DIR}/parsed/sglang_expert_distribution_summary.csv" | head -20
fi
```

Pass condition: `layer_id` starts at `FIRST_K_DENSE_REPLACE` and `logical_layer_index` starts at `0`.

## 6. Compare Sanity

```bash
column -s, -t < "${OUT_DIR}/parsed/aic_vs_sglang_topk_compute.csv"
column -s, -t < "${OUT_DIR}/parsed/aic_vs_sglang_collector_moe.csv"
```

Pass condition: primary rows have `phase=context`, `real_stage=topk+routed/compute`, non-empty `real_samples`, and an `aic_source`. The secondary `collector/moe` table may be slightly slower because it includes SGLang routed-wrapper overhead around the fused kernel.

For DeepEP/TBO runs, do not use this strict ordinary-smoke checklist unchanged: `collector/moe` can be absent, and TBO may not emit a `module` event because it dispatches `op_gate/op_select_experts/op_dispatch_*/op_experts/op_combine_*/op_output` directly. Use the runbook DeepEP/TBO validation commands instead.
