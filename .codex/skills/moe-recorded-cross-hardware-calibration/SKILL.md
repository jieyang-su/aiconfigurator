---
name: moe-recorded-cross-hardware-calibration
description: Use when validating or improving AIC recorded MoE accuracy across hardware for one fixed model and already-validated SGLang/backend runtime semantic, reusing frozen old-hardware truth when semantics match, comparing recorded against truth and balanced/power_law/uniform baselines, and designing platform-independent calibration changes.
---

# MoE Recorded Cross-Hardware Calibration

## Goal

Use this skill to prove whether recorded AIC MoE latency generalizes across
hardware for one fixed model and already-validated SGLang/backend runtime
semantic, and to converge errors without platform-name special cases.  Runtime
version migration is out of scope; if the runtime semantic is not fixed, use
`moe-recorded-aic-porting` first.

## Workflow

1. Freeze inputs:
   - Reuse already-confirmed old-hardware AIC compact data and frozen truth.
   - Do not rerun H20/H100 just because a new hardware appears.
   - Confirm the model/runtime semantic is already validated by
     `moe-recorded-aic-porting`.
   - Rerun or re-smoke old hardware only if case semantics or truth coverage is
     known to be wrong.

2. Confirm model family scope:
   - Use `moe-recorded-aic-porting` first if the model case matrix is unclear.
   - Use `moe-recorded-aic-porting` first if the SGLang/backend runtime path has
     changed or collector/source compatibility is unclear.
   - Only generate WideEP truth/gate work for models that actually support
     WideEP/DeepEP.

3. Run new-hardware AIC:
   - Use the model's default recorded collector command.
   - Default no-keep output should not leave source/debug files.
   - Keep-source mode is allowed only for audit and must not alter final
     latency.

4. Design or reuse truth:
   - Read [truth-strategy.md](references/truth-strategy.md).
   - For WideEP generation with cuda graph, read
     [nsys-wideep-generation-truth.md](references/nsys-wideep-generation-truth.md).
   - Before running SGLang truth, read
     [sglang-instrumentation.md](references/sglang-instrumentation.md).
   - If the SGLang/backend runtime is not already validated, stop and use
     `moe-recorded-aic-porting` first.  This skill assumes the runtime version
     is already a fixed, validated input.

5. Run final-output-only gates:
   - AIC compact `*.txt` vs frozen truth.
   - Recorded vs historical baselines such as balanced, power_law, and uniform.
   - Read [gate-and-error-attribution.md](references/gate-and-error-attribution.md).

6. Attribute errors:
   - Check token/EP/EPLB/family semantics first.
   - Then inspect source stability and source/materializer contract.
   - Then question truth instrumentation and parser selection.
   - Do not patch one bad point into a thick policy.

7. Change strategy only after replay:
   - Use old-hardware and new-hardware source together.
   - Prove the candidate is platform-independent.
   - Read [cross-hardware-generalization.md](references/cross-hardware-generalization.md).

8. Publish only after convergence:
   - Merge formal collector/materializer changes.
   - Rerun AIC default command.
   - Gate final compact latency.
   - Publish compact data and the method artifacts listed in
     [data-publishing.md](references/data-publishing.md).

## Invariants

- Truth validates recorded AIC; it does not feed collector runtime.
- Final compact latency is the gate target, not raw source or offline
  materializer guesses.
- No platform-name special cases.
- Runtime migration is out of scope; it must be resolved before this skill runs.
- If a point lacks truth coverage, mark it as unverified instead of claiming
  convergence.
- If nsys markers are missing or ambiguous, do not select a WideEP generation
  truth value.

## Common Files

- Recorded tools archive: `tools/moe_calibration/recorded_moe/`
- Truth runners: `tools/moe_calibration/recorded_moe/truth/`
- SGLang marker reference: `tools/moe_calibration/recorded_moe/truth/sglang_instrumentation/`
- Gate scripts: `tools/moe_calibration/recorded_moe/gate/`
- Offline replay/audit scripts: `tools/moe_calibration/recorded_moe/replay/`
- Recorded materializer: `collector/moe_recorded_materializer.py`
