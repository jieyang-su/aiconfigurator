# Upstream Merge Review Notes

This document records the PR-by-PR merge of upstream
`ai-dynamo/aiconfigurator` into this fork.  The intended final direction is to
align with upstream where possible, while migrating useful fork-only
functionality into upstream-compatible generic paths.

## Current State

- Working branch: `upstream-pr-by-pr-20260603`
- Base fork merge point before upstream replay: `f3cf48c`
- Preserved old-main branch: `archive/main-before-perf-compare-20260603`
- Current upstream replay is stopped at:
  - `2e197491 feat: vllm deepseek v4 (#1002)`
- Current state is an active cherry-pick conflict.  Do not run another
  cherry-pick until the conflict is resolved, skipped, or aborted.

## Merge Principles

1. Prefer upstream naming and structure for long-term compatibility.
2. Do not keep fork-only parallel names unless they are needed for existing
   perf data compatibility.
3. For DeepSeek-V4, prefer generic `dsv4_*` operator/file names as the public
   interface.
4. Migrate useful fork-specific behavior for individual models, such as
   V4-Flash or V4-Pro, into generic DSV4 code paths through model metadata:
   `architecture`, native head count, model path, and perf filename prefix
   where appropriate.
5. Keep fork-only optimizations when they are compatible with upstream:
   DSV4 operator calibration, SM120/PRO6000 handling, collector subprocess GPU
   resolution, and robust sparse-kernel lookup.
6. During conflict resolution, stop for a decision before making changes that
   would delete or rename existing fork functionality.

## Successfully Replayed Commits

These commits were cherry-picked or resolved and committed on the replay
branch.

| Upstream PR | Commit | Status | Notes |
| --- | --- | --- | --- |
| #1077 | `d8a5d078` | merged | XPU collector compatibility for vllm-xpu >= 0.20. |
| #1072 | `b3a1968e` | merged | Support matrix search query sync. |
| #1086 | `dd3f9378` | merged | Cursor Cloud instructions in `AGENTS.md`. |
| #1082 | `9d6fd117` | merged | Bumped `aiconfigurator-core` to 0.9.0 and enabled `cargo-deny`. |
| #1075 | `70b7f490` | merged | Support matrix memory/cache improvements.  A later duplicate replay attempt was skipped after detecting the same subject was already merged. |
| #1079 | `2890fc8b` | merged | Generator/validator NVBug fixes. |
| #1069 | `64779241` | merged | Database tests force in-parent loading. |
| #1084 | `50739aca` | merged | Missing silicon data typed errors. |
| #1073 | `82e217ac` | merged | Nemotron 3 Ultra hybrid mode. |
| #1060 | `59d76d7e` | merged | Support matrix `HW_INCOMPATIBLE` result. |
| #1074 | `3b536cdd` | merged | Infer `moe_tp` and `moe_ep` when only one is provided. |
| #1078 | `6163ff46` | merged | Generator bench script prefix cache support. |
| #1097 | `aada7931` | merged | Webapp parameter name fix. |
| #1095 | `f3540df8` | merged | Support matrix update. |
| #1104 | `26f5b7a0` | resolved and merged | DSA missing module data is now reported as `PerfDataNotAvailableError`, while preserving this fork's `_query_silicon_or_hybrid` flow and cubic-to-linear fallback. |
| #1091 | `22cc3c6c` | merged | Split support matrix by system. |
| #1076 | `ed084c16` | resolved and merged | Preserved `_query_silicon_or_hybrid`; integrated upstream DSA interpolation-miss formatting into `get_silicon()`. |
| #1080 | `12553652` | resolved and merged | Added static estimate mode/detail report.  Preserved fork estimate-context logging, `mock_moe_policy`, comm debug logging, and DSV4 operator calibration while adopting upstream `PerformanceResult.source` semantics. |

## Resolved Conflict Decisions

### #1104: DSA Missing Data

Conflict files:

- `src/aiconfigurator/sdk/perf_database.py`
- `tests/unit/sdk/database/test_dsa_module.py`

Conflict cause:

- Upstream added typed unavailable-data behavior for DSA module queries.
- This fork already had `_query_silicon_or_hybrid` and cubic-to-linear
  interpolation fallback around the same code.

Resolution:

- Kept `_query_silicon_or_hybrid`.
- Added upstream `PerfDataNotAvailableError` wrapping for missing DSA keys and
  interpolation misses.
- Kept fork cubic-to-linear fallback.
- Fixed context DSA lookup to prefer the raw piecewise interpolation result and
  only fall back to cubic/linear when needed.

### #1076: AIConfigurator 0.9.0 RC0 NVBugs

Conflict files:

- `src/aiconfigurator/sdk/perf_database.py`

Conflict cause:

- Upstream further improved DSA interpolation-miss messages.
- This overlapped with the just-resolved #1104 DSA fallback code.

Resolution:

- Kept the fork's unified `_query_silicon_or_hybrid` structure.
- Integrated upstream `_is_dsa_interpolation_miss` and
  `_format_dsa_unavailable_message`.
- Converted recognized interpolation misses inside `get_silicon()` into
  `PerfDataNotAvailableError`, allowing HYBRID fallback to remain centralized.

### #1080: Static Estimate Mode and Breakdown Report

Conflict files:

- `src/aiconfigurator/cli/main.py`
- `src/aiconfigurator/sdk/operations.py`
- `src/aiconfigurator/sdk/perf_database.py`

Conflict cause:

- Upstream added static estimate mode, detailed breakdown reporting, and
  `PerformanceResult.source` tracking.
- This fork had estimate-context logging, custom communication debug tracing,
  `mock_moe_policy`, and DSV4 operator calibration.

Resolution:

- `main.py`:
  - Kept upstream `--detail` and safe `nextn_accept_rates` parsing.
  - Preserved fork `[estimate-context]` resolved systems/perf source logging.
- `operations.py`:
  - Kept upstream `PerformanceResult` aggregation.
  - Migrated fork `mock_moe_policy` to operate on `PerformanceResult` group
    totals without losing `source` or `energy`.
- `perf_database.py`:
  - Kept upstream `_interp_pr(..., source="silicon")`.
  - Kept fork communication debug logging.
  - Kept fork `AIC_PREFER_NCCL_FOR_CUSTOM_ALLREDUCE` behavior.
  - Applied DSV4 calibration after `_interp_pr` where needed so calibrated
    results preserve source metadata.

## Current Conflict: #1002 vLLM DeepSeek-V4

Upstream commit:

- `2e197491 feat: vllm deepseek v4 (#1002)`

Current conflict files:

- `collector/collect.py`
- `collector/common_test_cases.py`
- `collector/registry_types.py`
- `collector/sglang/collect_dsv4_attn.py`
- `collector/sglang/collect_mhc_module.py`
- `collector/sglang/deepseekv4_sparse_modules.py`
- `src/aiconfigurator/sdk/common.py`
- `src/aiconfigurator/sdk/models/deepseek_v4.py`
- `src/aiconfigurator/sdk/operations.py`
- `src/aiconfigurator/sdk/perf_database.py`
- `tests/unit/sdk/database/test_dsv4_sparse.py`

Upstream additions without major fork overlap:

- `collector/vllm/collect_dsv4_attn.py`
- `collector/vllm/collect_mhc_module.py`
- vLLM registry and utility support for DSV4.

Recommended decision:

- Use upstream generic `dsv4_*` as the public naming and long-term data shape.
- Do not preserve separate public `dsv4_flash_*` or `dsv4_pro_*` aliases.
- Migrate useful fork-only behavior from `dsv4_flash` / `dsv4_pro` into the
  generic implementation.
- Treat SM120 as an explicit runtime branch, not as a generic fallback.  The
  torch path is crash-only fallback.
- Select SGLang attention backend by explicit runtime version:
  - `v0.5.10` -> `attention_backend=compressed`
  - `v0.5.12` -> `attention_backend=dsv4`

### Proposed #1002 Merge Strategy

#### 1. Public perf data names

Prefer upstream:

- `dsv4_csa_context_module_perf.txt`
- `dsv4_hca_context_module_perf.txt`
- `dsv4_csa_generation_module_perf.txt`
- `dsv4_hca_generation_module_perf.txt`
- `dsv4_paged_mqa_logits_module_perf.txt`
- `dsv4_hca_attn_module_perf.txt`
- `mhc_module_perf.txt`

Migration from fork:

- Treat former `dsv4_flash_*` and `dsv4_pro_*` as model-specific cases under
  generic `dsv4_*`, distinguished by:
  - `model`
  - `architecture`
  - `num_heads` or native head count
  - `compress_ratio`
  - `gemm_type`
  - `tp_size`

Final decision:

- Do not keep public compatibility aliases for old `dsv4_flash_*` /
  `dsv4_pro_*` perf names.  Regenerated data should use generic `dsv4_*`
  files and distinguish Flash/Pro with row metadata.

#### 2. Collector registry

Prefer upstream generic op names for new runs:

- `dsv4_csa_context_module`
- `dsv4_hca_context_module`
- `dsv4_csa_generation_module`
- `dsv4_hca_generation_module`
- `dsv4_paged_mqa_logits_module`
- `dsv4_hca_attn_module`

Migrate fork behavior:

- Keep model filtering and path resolution for:
  - `sgl-project/DeepSeek-V4-Flash-FP8`
  - `sgl-project/DeepSeek-V4-Pro-FP8`
  - upstream/default DeepSeek-V4 names.
- Preserve subprocess GPU resolution via `resolve_subprocess_visible_device`.
- Preserve model-specific `perf_filename_prefix` only if needed during a
  transitional data migration.

Final decision:

- Old collector ops such as `dsv4_flash_*` / `dsv4_pro_*` should not remain
  callable.  Registry exposes only generic `dsv4_*`.

#### 3. `collector/common_test_cases.py`

Prefer upstream structure:

- Generic DSV4 model list and test-case generation.

Migrate fork behavior:

- Preserve model-path driven specialization:
  - Flash vs Pro model selection.
  - Sparse kernel model filters.
  - smoke/full cases for SGLang sparse kernels.
- Preserve fork's larger/SM120-oriented sparse shapes only if they are still
  needed for collector coverage.

Final decision:

- Use generic DSV4 cases with model-filtered parameters.  Pro-specific TP
  sweep width is chosen from the model path, not from a public op name.

#### 4. `collector/sglang/collect_dsv4_attn.py`

Prefer upstream filename and generic collector:

- Keep `collector/sglang/collect_dsv4_attn.py`.
- Do not restore `collect_dsv4_flash_attn.py` as a separate file.

Migrate fork behavior:

- Keep model-specific native head handling.
- Keep robust port retry/subprocess behavior.
- Keep dummy local config handling and quantization overrides that were needed
  for SGLang collection.
- Keep `resolve_subprocess_visible_device`.

Final decision:

- Output generic `dsv4_*` files only.  Rows carry `model`, `architecture`,
  native `num_heads`, `compress_ratio`, `gemm_type`, and `tp_size`.
- `--sglang-version-branch` is limited to `auto`, `v0.5.10`, and `v0.5.12`.
  `v0.5.10` selects `attention_backend=compressed`; `v0.5.12` selects
  `attention_backend=dsv4`.

#### 5. `collector/sglang/deepseekv4_sparse_modules.py`

Prefer upstream generic kernel names:

- `paged_mqa_logits`
- `hca_attn`

Migrate fork behavior:

- Keep support for SM120/PRO6000 fallback paths if still present in the fork.
- Keep model-family detection, but route it through generic DSV4 output names.
- Keep better subprocess GPU handling.
- Keep architecture/native head metadata in logged rows.

Final decision:

- No public separate filenames.  Use generic filenames plus row metadata.
- SM120 uses the SGLang SM120 FlashMLA entrypoint when available; torch is only
  used after an illegal-memory-access crash and is marked as crash fallback in
  the logged kernel source.

#### 6. SDK model and operations

Prefer upstream generic operation classes and query names.

Migrate fork behavior:

- Keep `architecture` as an extra generic metadata axis.
- Align native-head naming with upstream: `native_heads` and `tp_size`.
- Pass `architecture` from `DeepSeekV4Model` to operation query classes.

#### 7. `perf_database.py`

Prefer upstream generic loaders and query functions.

Migrate fork behavior:

- Loader should tolerate both row shapes during transition:
  - upstream generic keyed by native heads
  - fork rows that include `architecture`
- Query should accept optional `architecture` and native-head metadata.
- Sparse-kernel lookup should support:
  - exact match
  - native-head fallback
  - architecture fallback where rows include it
  - robust interpolation fallback from the fork.
- Preserve DSV4 operator calibration and source-aware `PerformanceResult`.

Final decision:

- Generic files only.  Existing old checked-in data was removed by upstream
  `#1002`; regenerate current data into generic files.

#### 8. Tests

Prefer upstream `test_dsv4_sparse.py` as the canonical file.

Migrate fork behavior:

- Add parametrized coverage for Flash and Pro model paths using generic DSV4
  cases.
- Preserve tests for architecture/native-head lookup if those are retained.

Final decision:

- Tests assert generic names and metadata-driven model differentiation.

## #1002 Resolution Summary

Resolution in progress:

- Started from upstream `#1002` generic DeepSeek-V4 implementation.
- Removed old SGLang registry public entries for `dsv4_pro_*`.
- Kept generic DSV4 op/file names.
- Migrated fork model-specific behavior into metadata-aware paths:
  `architecture`, native `num_heads`, `tp_size`, and model-path-driven TP
  sweeps.
- Added `architecture` as an extra loader/query axis for DSV4 module and sparse
  kernel data.
- Kept SM120 as an active HCA path when `flash_mla_sm120` is available; torch
  fallback is crash-only.
- Corrected SGLang branch mapping:
  `v0.5.10 -> compressed`, `v0.5.12 -> dsv4`.

## Current Conflict: #1122 Generic Collector Model Resolver

Upstream commit:

- `a4827ce2 refactor(collector): replace DeepSeek-specific helpers with _resolve_local_model_path() (#708) (#1122)`

Conflict files:

- `collector/helper.py`
- `collector/sglang/collect_wideep_deepep_moe.py`

Conflict cause:

- Upstream replaced DeepSeek-specific helper behavior with a generic
  `_resolve_local_model_path(model_id)` that resolves local config directories,
  AIC cached model configs, or HuggingFace config downloads.
- This fork already had MoE collector model-path priority across
  `COLLECTOR_LOCAL_MODEL_PATH`, `COLLECTOR_MODEL_PATH`, `MOE_MODEL_PATH`, and
  legacy `DEEPSEEK_MODEL_PATH`, plus SGLang-specific config rewriting in
  `collect_wideep_deepep_moe.py`.

Resolution:

- Keep upstream `_resolve_local_model_path(model_id)` as the shared resolver.
- Add `_get_moe_model_path()` as a thin compatibility wrapper preserving the
  existing MoE environment-variable priority, but delegate actual resolution to
  `_resolve_local_model_path`.
- Keep `collect_wideep_deepep_moe.py` model-specific SGLang config rewrite and
  subprocess GPU mapping via `resolve_subprocess_visible_device`.
- Make wideep MoE config loading call `_resolve_local_model_path()` first so AIC
  cached configs and HF side-car quant configs work consistently.

## Current Conflict File-by-File Analysis

This section records the active `#1002` conflict at the file level.  No
conflict has been resolved yet.

### `collector/collect.py`

Conflict cause:

- This fork added `--sglang-version-branch` and writes
  `COLLECTOR_SGLANG_VERSION_BRANCH`.
- This fork auto-expanded `--model-path` for DeepSeek-V4 Flash/Pro into
  `dsv4_flash_*` / `dsv4_pro_*` op names.
- Upstream removed the model-specific auto-expand because it now exposes
  generic `dsv4_*` ops.

Recommended resolution:

- Keep `--sglang-version-branch`; it is orthogonal to upstream `#1002`.
- Drop model-specific auto-expand to `dsv4_flash_*` / `dsv4_pro_*`.
- If auto-expand is still useful, retarget it to generic `dsv4_*` ops and let
  model filtering decide Flash vs Pro.
- Keep the short logging scope idea, but rename it to generic `dsv4` if used.

Decision needed:

- Whether current automation depends on calling collect with only
  `--model-path sgl-project/DeepSeek-V4-*-FP8` and no `--ops`.  If yes, keep a
  generic `dsv4` auto-expand; otherwise accept upstream behavior.

### `collector/common_test_cases.py`

Conflict cause:

- Upstream replaced V4-Flash-specific helpers with generic DSV4 model
  selection and generic test-case functions.
- This fork added:
  - DeepSeek-V3.1, MiniMax-M2.7, and NVIDIA NVFP4 aliases.
  - separate Flash/Pro model aliases and output helpers.
  - Pro TP sizes up to 16.
  - pre-Blackwell FP4 expert guard that skips `fp8_block` for DSV4 Flash.
  - separate Flash/Pro sparse test builders.

Recommended resolution:

- Keep upstream generic `_selected_dsv4_models()` and generic
  `get_dsv4_*_test_cases()` names.
- Migrate this fork's extra supported model names into the generic supported
  model lists.
- Keep model-specific TP sizing as a helper used inside generic case generation:
  Flash uses `[1, 2, 4, 8]`, Pro may use `[1, 2, 4, 8, 16]` if SGLang layout
  supports it.
- Keep the pre-Blackwell FP4 expert guard only if it still reflects the current
  SGLang runtime.  Otherwise remove it and rely on upstream generic sweep.
- Remove public `get_dsv4_flash_*` / `get_dsv4_pro_*` helpers unless needed as
  temporary aliases.

Decision needed:

- Whether to keep Pro TP=16 coverage in generic DSV4 module sweeps.
- Whether the pre-Blackwell `fp8_block` skip is still desired for collector
  output correctness.

### `collector/registry_types.py`

Conflict cause:

- Upstream defines generic `PerfFile.DSV4_*` enum values.
- This fork defines separate `DSV4_FLASH_*` and `DSV4_PRO_*` enum values.

Recommended resolution:

- Prefer upstream generic `DSV4_*` enum values.
- Do not keep Flash/Pro public enum names as first-class values.
- If old automation still references old names, add compatibility aliases only
  in the collector registry layer, not as canonical perf files.

Decision needed:

- Whether external scripts still refer to old enum/op names.  If not, remove
  the old names.

### `collector/sglang/collect_dsv4_attn.py`

Conflict cause:

- Upstream renamed `collect_dsv4_flash_attn.py` to generic
  `collect_dsv4_attn.py`.
- This fork's deleted file had substantial behavior:
  - SGLang version branch flag.
  - `resolve_subprocess_visible_device`.
  - more detailed child process error tail.
  - `perf_filename_prefix` for Flash/Pro-specific outputs.
  - Blackwell/Hopper MoE runner backend adjustment for `fp8_block`.
  - `attention_backend="dsv4"` instead of upstream `"compressed"`.
  - model-specific native head handling.

Recommended resolution:

- Keep upstream generic file and entrypoint names.
- Migrate fork behavior that is runtime robustness, not naming:
  - `--sglang-version-branch`.
  - `resolve_subprocess_visible_device`.
  - detailed subprocess output tail.
  - MoE runner backend adjustment if still required for current SGLang.
  - native head metadata in logging/query rows.
- Prefer upstream generic output prefix `dsv4`.
- Re-check `attention_backend`: upstream uses `"compressed"`, fork uses
  `"dsv4"`.  This is a runtime compatibility question, not just naming.

Decision needed:

- Which SGLang branch/runtime is the target for this PR: upstream current
  `"compressed"` backend, fork/adapted `"dsv4"` backend, or both via
  `--sglang-version-branch`.

### `collector/sglang/collect_mhc_module.py`

Conflict cause:

- This fork filters by `COLLECTOR_MODEL_PATH` and runs one task per `pre/post`.
- Upstream generates one task per unique `(phase, hidden_size, hc_mult, model)`.

Recommended resolution:

- Prefer upstream per-model/per-shape task generation.
- Preserve this fork's model-path filter by applying it before building
  upstream-style cases.

Decision needed:

- No major decision if model filtering is retained.

### `collector/sglang/deepseekv4_sparse_modules.py`

Conflict cause:

- Upstream genericized sparse module collection and reads model config for
  `native_heads`, `index_n_heads`, and `index_head_dim`.
- This fork added many runtime/fallback features:
  - SM120 SGLang fallback detection.
  - legacy runtime torch fallback for HCA.
  - architecture/family-based output filenames.
  - `COLLECTOR_FORCE_DSV4_FLASH_SPARSE`.
  - token-level causal HCA input builder.
  - subprocess fallback after illegal memory access.

Recommended resolution:

- Prefer upstream generic function names and generic output files.
- Keep upstream model-config-driven `native_heads/index_*` discovery.
- Migrate fork's token-level causal HCA builder if it is more faithful to
  serving semantics than upstream's shared-batch synthetic layout.
- Keep `resolve_subprocess_visible_device` and support-status logging.
- Do not keep Flash/Pro-specific output filenames as canonical.
- Be careful with old SM120 torch fallback.  Since the recent optimization
  branch was rolled back and testing showed poor benefit, only keep fallback if
  it prevents crashes; do not make it the default performance path.

Decision needed:

- Whether to keep the torch fallback path for SM120 HCA as a crash-only
  fallback, or remove it entirely and require the upstream/native path.

### `src/aiconfigurator/sdk/common.py`

Conflict cause:

- Upstream generic `PerfDataFilename.dsv4_*`.
- This fork Flash/Pro-specific `PerfDataFilename.dsv4_flash_*` and
  `dsv4_pro_*`.

Recommended resolution:

- Prefer upstream generic `PerfDataFilename.dsv4_*`.
- Do not expose Flash/Pro-specific enum values unless needed for temporary
  migration.
- Distinguish Flash/Pro through row metadata and model config, not filename.

Decision needed:

- Same as registry: whether old checked-in/generated data must remain readable
  without regeneration.

### `src/aiconfigurator/sdk/models/deepseek_v4.py`

Conflict cause:

- This fork passes `architecture` and `native_num_heads` into attention module
  operations.
- Upstream did not include these parameters because its generic data shape keys
  by native heads directly.

Recommended resolution:

- Keep the functionality but align naming with upstream:
  - pass `native_heads` or `native_num_heads` consistently.
  - keep `architecture` if loaders support architecture-keyed rows.
- Defaults must keep upstream generic behavior working when metadata is absent.

Decision needed:

- Prefer naming `native_heads` to match upstream and avoid both spellings where
  possible.

### `src/aiconfigurator/sdk/operations.py`

Conflict cause:

- This fork added optional `architecture` and `native_num_heads` to DSV4
  attention operation constructors and query calls.
- Upstream generic operation signatures do not carry those values.

Recommended resolution:

- Keep optional metadata parameters, but name them consistently with upstream
  (`native_heads` preferred).
- Pass metadata through to `PerfDatabase` only for silicon lookup; SOL math
  should continue using existing local-head math.

Decision needed:

- No major decision if `perf_database.py` keeps metadata-aware lookup.

### `src/aiconfigurator/sdk/perf_database.py`

Conflict cause:

- Upstream:
  - generic DSV4 loaders.
  - generic sparse kernel data keyed by `native_heads -> tp -> past_kv -> isl
    -> bs`.
  - robust generic DSV4 lookup.
- This fork:
  - separate Flash/Pro filenames.
  - architecture/native-head axes.
  - DSV4 attention calibration source system.
  - kernel-delta correction using sparse kernel rows.
  - robust lookup and raw piecewise interpolation.

Recommended resolution:

- Use upstream generic loader/query function names.
- Keep metadata-aware data shape as a superset:
  - support upstream rows keyed by native heads.
  - support fork rows with `architecture` where present.
- Keep DSV4 attention calibration and operator roofline scale from this fork.
- Keep robust lookup and sparse kernel delta correction.
- Prefer generic filename loading.  Old Flash/Pro filenames can be optional
  compatibility fallbacks only if required.

Decision needed:

- Whether to retain old `dsv4_flash_*` / `dsv4_pro_*` file fallback loading
  for a transition period.

### `tests/unit/sdk/database/test_dsv4_sparse.py`

Conflict cause:

- Upstream canonicalized tests under generic `test_dsv4_sparse.py`.
- This fork had `test_dsv4_flash_sparse.py` with model-specific helper names
  and alias tests.

Recommended resolution:

- Prefer upstream test file and generic helper names.
- Migrate useful fork tests by parametrizing over:
  - `sgl-project/DeepSeek-V4-Flash-FP8`
  - `sgl-project/DeepSeek-V4-Pro-FP8`
- Tests should assert generic DSV4 op/file names plus model metadata, not old
  public helper names.

Decision needed:

- No major decision if old public names are dropped.

## `1d12d321` / `#1113 refactor: aic collector v2`

Status: in progress during upstream replay.

High-level conflict cause:

- Upstream moved collector case ownership from Python constants in
  `collector/common_test_cases.py` into YAML plus `collector/case_generator.py`.
- This fork had model additions and collector behavior in the old Python layer:
  DeepSeek-V3.1, MiniMax-M2.7, Kimi/MiniMax WideEP MoE wrapper handling,
  DeepSeek-V4 Pro TP=16, and SGLang version-branch selection.

Resolution applied:

- Kept upstream collector v2 as the canonical architecture.
- Deleted `collector/common_test_cases.py`; migrated retained behavior into
  YAML and `case_generator.py`.
- Added `deepseek-ai/DeepSeek-V3.1` to
  `collector/cases/models/DeepseekV3ForCausalLM_cases.yaml` for MoE, MLA, and
  MLA module collection. Existing upstream YAML already covered
  `MiniMaxAI/MiniMax-M2.7` / `nvidia/MiniMax-M2.7-NVFP4`.
- Added DSV4 model-specific `module_tp_sizes_by_model` in
  `DeepseekV4ForCausalLM_cases.yaml`; `DeepSeek-V4-Pro` and
  `sgl-project/DeepSeek-V4-Pro-FP8` get TP=16 under the generic `dsv4_*`
  path.
- Added `_dsv4_module_tp_sizes(model_path)` to `case_generator.py` and used it
  in DSV4 module case generation.
- Removed public `dsv4_flash_*` test-case aliases from `case_generator.py`,
  `collect_dsv4_attn.py`, and `deepseekv4_sparse_modules.py`.
- Kept `collect.py --sglang-version-branch` with only `auto`, `v0.5.10`,
  `0.5.10`, `v0.5.12`, `0.5.12`. The flag sets
  `COLLECTOR_SGLANG_VERSION_BRANCH` and does not interfere with collector v2
  planning.
- In `collect_dsv4_attn.py`, `v0.5.10` maps to
  `attention_backend=compressed`, while `v0.5.12` maps to
  `attention_backend=dsv4`.
- In `collect_mla_module.py`, retained v0.5.10/v0.5.12 ForwardBatch and
  forward-context compatibility via `version_compat.py`, while using upstream
  YAML-backed model specs.
- In the moved WideEP MoE collector
  `collector/wideep/sglang/collect_deepep_moe.py`, migrated fork support for:
  nested `text_config` lookup, `num_local_experts`, stripping checkpoint
  quantization metadata for dummy DeepEP collection, reduced local config
  generation, subprocess visible-device mapping, progress timeout logging, and
  writing split WideEP perf files beside the caller-provided `perf_filename`.

Remaining review points:

- Verify no external users still depend on public `dsv4_flash_*` collector
  helper names. The current decision is to drop them.
- Confirm whether `version_compat.py` should continue accepting internal
  aliases like `legacy/current` from the environment. CLI choices no longer
  expose those names.

## `12dca98b` / `#1155 refactor: complete lazy-load Pattern + A cleanup`

Status: resolved during upstream replay.

High-level conflict cause:

- Upstream completed the operations refactor: `_legacy.py` is deleted, CSV
  loaders move out of `perf_database.py` into owning `operations/*.py` modules,
  and tests patch loaders at the owning module rather than at
  `perf_database` re-export sites.
- This fork still carried DeepSeek-V4 architecture-aware loader/query behavior
  around `perf_database.py` and DSV4 sparse tests.

Resolution applied:

- Accepted upstream deletion of `src/aiconfigurator/sdk/operations/_legacy.py`.
  No DSV4 or other op classes are kept in the legacy module.
- Kept upstream's slim `perf_database.py` structure with loader re-exports.
  Added only the small `DEFAULT_DSV4_ARCHITECTURE` re-export and DSV4 query
  wrapper `architecture` passthrough so model code can keep passing
  `architecture=self.architecture`.
- Moved/kept fork DSV4 behavior in `src/aiconfigurator/sdk/operations/dsv4.py`:
  architecture bucket selection, architecture-aware context/generation/sparse
  loaders, sparse-kernel lookup by architecture/native_heads/tp_size, and
  context/generation query selection by architecture.
- Accepted upstream test fixture ownership changes in
  `tests/unit/sdk/database/conftest.py`.
- Accepted upstream interpolation tests against the new
  `aiconfigurator.sdk.interpolation` module.
- Updated DSV4 sparse tests to import DSV4 constants/helpers from
  `aiconfigurator.sdk.operations.dsv4` while keeping `LoadedOpData` and loader
  re-exports through `perf_database` where upstream still exposes them.
- Resolved an adjacent `report_and_save.py` conflict between the fork's
  `all_results.csv` export and upstream's `--inclusive-tpot` display dataframe:
  both behaviors were retained. `all_results.csv` is still written after
  dropping `_per_ops_source`; best-config output now uses upstream's
  `display_best_configs` copy.

Rationale:

- The requested end state is upstream-aligned generic DSV4 public API, with
  Flash/Pro differences represented as model/architecture/native-head
  metadata. Keeping `perf_database.py` slim and placing the fork-specific DSV4
  metadata behavior in `operations/dsv4.py` matches both upstream's new
  ownership model and the fork's compatibility needs.

## `299aaea9` / `#1131 feat: collect SGLang DeepSeek-V4 attention as full modules`

Status: resolved during upstream replay.

High-level conflict cause:

- Upstream replaced the older sparse-only DeepSeek-V4 attention collection with
  full-module CSA/HCA context and generation collectors, runtime-limit helpers,
  default DSV4 op expansion in `collect.py`, and newer SGLang module replay
  paths.
- This fork had already added generic DSV4 model selection, per-model TP
  ranges, v0.5.10/v0.5.12 SGLang backend selection, visible-device remapping,
  and forward-batch/context compatibility for pinned SGLang builds.

Resolution applied:

- Accepted upstream full-module DSV4 collector behavior and default
  `collect.py --model-path <DeepSeek-V4>` auto-expansion to generic
  `dsv4_*` module ops.
- Preserved the fork's `--sglang-version-branch` flag and environment
  propagation. The public CLI choices remain `auto`, `v0.5.10`, `0.5.10`,
  `v0.5.12`, and `0.5.12`; v0.5.10 selects `attention_backend=compressed`,
  while v0.5.12 selects `attention_backend=dsv4`.
- Preserved per-model DSV4 TP-size handling via
  `_dsv4_module_tp_sizes(model_path)` and combined it with upstream's native
  FP4 precision filtering for `deepseek-ai/*` checkpoints.
- Dropped upstream's temporary public `dsv4_flash_*` compatibility aliases from
  `case_generator.py`, matching the decision that public APIs should use only
  generic `dsv4_*` names.
- Kept upstream runtime-limit, chunked-allocation, piecewise replay, and module
  CUDA-graph scaffolding in `collect_mla_module.py`, but routed ForwardBatch
  creation and forward context setup through the fork's compatibility wrappers
  so both old and new SGLang APIs remain usable.
- Kept upstream MoE collector import layout preference for the new
  `moe_runner.triton_utils` package while preserving fallback imports for the
  older `fused_moe_triton` package.

Rationale:

- This keeps the fork aligned with upstream's generic DSV4 collector model,
  while retaining the practical compatibility switches needed for existing
  SGLang 0.5.10/0.5.12 collection environments and model-specific DSV4 Pro TP
  sweeps.

## Auto-applied upstream commits after `#1131`

Status: applied without manual conflict resolution.

Commits:

- `1ce6ff60` / `#1092 feat: add non-causal encoder attention perf support`
- `b2add2fb` / `#1153 feat: add tests&harness&execution plan for sdk core rust migration`
- `bbeb6549` / `#1152 feat: add FPM forward pass perf model with online tuning`
- `3bf63c7c` / `#1154 fix(ci): resolve PR number for fork PRs in accuracy regression comment`
- `a05aacbb` / `#1160 fix: update sglang dsa module collection cases`
- `5a4caa8e` / `#1156 fix: update b60 support matrix`
- `69b3cff3` / `#1170 chore: add root cargo workspace`
- `a54593e2` / `#1174 Order support matrix systems by priority`

Resolution applied:

- No manual conflicts occurred. These commits were accepted as upstream state.
- For `#1160`, Git auto-merged the DSA module collection updates into the
  already compatibility-adjusted MLA collector.

## `f93be4ae` / `#1047 perf: replace perf CSV assets with parquet`

Status: resolved during upstream replay.

High-level conflict cause:

- Upstream added `collector/collect.py --keep-csv` while converting perf data
  assets from text/CSV staging files to parquet final outputs.
- This fork had added `--sglang-version-branch` in the same parser location
  for SGLang DSV4 collector compatibility.

Resolution applied:

- Kept both CLI flags. `--sglang-version-branch` still sets
  `COLLECTOR_SGLANG_VERSION_BRANCH`; `--keep-csv` still controls upstream's
  parquet finalization behavior.
- Accepted upstream parquet conversion and finalization changes elsewhere.

Rationale:

- The flags control independent concerns: SGLang API compatibility versus perf
  asset storage format. Keeping both preserves fork collector compatibility
  while adopting upstream parquet perf assets.

## `f93be4ae` / `#1047 perf: replace perf CSV assets with parquet`

Status: resolved during upstream replay.

High-level conflict cause:

- Upstream added collector parquet finalization and the `--keep-csv` CLI flag,
  while this fork had added `--sglang-version-branch` at the same parser
  location.
- The commit also converts many checked-in perf-data assets from `.txt` CSV
  files to `.parquet`.

Resolution applied:

- Kept both collector CLI flags. `--sglang-version-branch` still controls
  SGLang v0.5.10/v0.5.12 attention backend compatibility, and upstream's
  `--keep-csv` remains available for preserving staging CSVs.
- Accepted upstream parquet perf-data assets and deletions of the corresponding
  `.txt` files as the canonical data format for this replay.

Rationale:

- The two CLI flags are independent. Accepting upstream parquet data keeps the
  fork aligned with current upstream perf-database storage while preserving the
  fork's SGLang collector compatibility switch.
