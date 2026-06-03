# Perf Compare Automation

This directory keeps the automation scripts, compare-case configs, and notes
used for perf database comparison work.  The goal is to keep the PR-facing
artifacts in one place instead of scattering scripts under `tools/` and large
JSON configs under `docs/perf_database/`.

## Layout

- `scripts/dsv4flash_topology_automation.py`
  Runs multi-case topology comparisons from a JSON config.  It supports per-case
  backend/backend_version, quant modes, NCCL perf files, custom parallel search
  spaces, Pareto plots, and cost-per-million-token plots.
- `scripts/dsv4flash_single_system_mode_automation.py`
  Runs agg/disagg mode comparisons for a single system.
- `scripts/run_dsv4flash_topology_compare.sh`
  Legacy shell helper for starting/checking/waiting two topology jobs.
- `configs/*.json`
  Ready-to-run experiment configs for DSV4, DSV3.1, Kimi K2.5, and MiniMax.
- `docs/*.md`
  Supporting notes, calibration references, and legacy command references.

The Python scripts locate the repository root by walking upward until
`src/aiconfigurator` is found, so they can be invoked from the repo root or from
inside this directory.

## Common Commands

Run a multi-case topology comparison:

```bash
python3 tools/perf_compare_automation/scripts/dsv4flash_topology_automation.py \
  --config tools/perf_compare_automation/configs/DSV3.1-1024.json
```

Run a DSV4 Pro comparison:

```bash
python3 tools/perf_compare_automation/scripts/dsv4flash_topology_automation.py \
  --config tools/perf_compare_automation/configs/DSV4-PRO-1024.json
```

Run a single-system agg/disagg comparison:

```bash
python3 tools/perf_compare_automation/scripts/dsv4flash_single_system_mode_automation.py \
  --config tools/perf_compare_automation/configs/dsv4flash_single_system_agg_disagg_search_example.json
```

Use the legacy two-job shell helper:

```bash
bash tools/perf_compare_automation/scripts/run_dsv4flash_topology_compare.sh start
bash tools/perf_compare_automation/scripts/run_dsv4flash_topology_compare.sh check
bash tools/perf_compare_automation/scripts/run_dsv4flash_topology_compare.sh wait
```

## Config Notes

Top-level fields provide defaults:

- `systems_path`
- `model`
- `backend`
- `backend_version`
- `gemm_quant_mode`
- `kvcache_quant_mode`
- `fmha_quant_mode`
- `moe_quant_mode`
- `comm_quant_mode`
- `database_mode`
- runtime fields such as `isl`, `osl`, `ttft`, `tpot`, `total_gpus`

Each entry in `compare_cases` can override these fields.  This is useful when a
single plot compares heterogeneous cases, for example:

- H20 + SGLang + `deepseek-ai/DeepSeek-V3.1`
- PRO6000 + TRT-LLM + `nvidia/DeepSeek-V3.1-NVFP4`

Quant mode keys are intentionally explicit even when they match model defaults,
so generated commands and logs are easy to audit.

## Parallel Search Overrides

Configs may provide `search_parallel` blocks for `agg` and `disagg`.  The
automation scripts translate these blocks into YAML patches for AIC CLI runs.
This keeps experiment-specific search spaces out of the SDK defaults while
still making the exact search reproducible from the JSON file.

## Cost Plot

Configs may include `gpu_hourly_cost_usd`, for example:

```json
{
  "gpu_hourly_cost_usd": {
    "h20": 1.0,
    "PRO6000": 0.75
  }
}
```

The topology automation uses this to derive:

```text
$/Mtokens = ($/GPU/hour) / (tokens/s/GPU * 3600) * 1e6
```

This is intentionally a hardware-cost abstraction.  Update the config values
when final cost assumptions are available.

## Outputs

Each run writes under the config's `out_dir`, typically:

- per-case `output_<label>.log`
- generated YAML patches
- raw AIC result folders
- compare CSV files
- Pareto plots
- cost-per-million-token plots

The scripts print an `[automation-context]` block before each case so logs show
the resolved model/backend/version/quant/cost/NCCL inputs.
