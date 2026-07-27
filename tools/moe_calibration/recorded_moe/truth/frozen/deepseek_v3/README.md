# DeepSeek-V3 Recorded MoE Frozen Truth

This directory archives the compact frozen real-machine truth used to validate
DeepSeek-V3 recorded MoE AIC calibration.

## Scope

- Model family: DeepSeek-V3
- Backend: SGLang
- Platforms: H20 SXM and H100 SXM
- Families:
  - `ordinary_context`
  - `ordinary_generation`
  - `wideep_context`
  - `wideep_generation`

## Files

- `four_modes_frozen_truth.csv`: combined H20 + H100 frozen truth.
- `h20_four_modes_frozen_truth.csv`: H20-only frozen truth.
- `h100_four_modes_frozen_truth.csv`: H100-only frozen truth.
- `four_modes_frozen_truth_coverage.csv`: compact coverage by platform/family/EP/EPLB.
- `four_modes_frozen_truth_source_summary.csv`: provenance summary by source group and truth method.

## Source

The frozen truth is assembled from:

- `results/four_modes_latest_truth_20260726_h100_ep4_context_supplement/four_modes_latest_truth_summary.csv`

That source was based on the previous four-mode latest truth plus H100 EP4
context supplements:

- Base: `results/four_modes_latest_truth_20260725_aic_covered_context_supplement/four_modes_latest_truth_summary.csv`
- Patch: `results/h100_truth_ep4_context_supplement_20260726_summary/h100_ep4_context_supplement_truth_summary.csv`

## Truth Semantics

The truth target is real-machine routed/compute MoE latency, not AIC-predicted
latency and not a value inferred from AIC error.

- Ordinary context/generation and WideEP context use the existing
  torch-profiler/kernel-external-id based real-machine parser path.
- WideEP generation uses the nsys CUDA graph semantic routed/compute parser,
  because torch profiler can capture the wrong replay/window under CUDA graph.

The `truth_method`, `source_group`, `sessions`, `values_us`, `spread_pct`, and
`stability` columns should be kept with the truth rows. They are part of the
audit trail and help explain why a row is accepted.

## Known Gaps

- H100 is a 4-GPU environment in this calibration set, so H100 EP8 truth is not
  available here.
- Raw nsys reports, profiler traces, and per-session run directories are not
  archived in this directory. Keep only compact frozen truth and provenance
  summaries in git.

## Usage

Use this directory as the fixed validation truth for DeepSeek-V3 recorded MoE
AIC gates. The collector/materializer must not read this truth at runtime to
produce latency; truth is only for validation, regression checks, and migration
audits.
