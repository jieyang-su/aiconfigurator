---
name: moe-recorded-aic-porting
description: Use when migrating AIC recorded MoE measurement to a MoE model or SGLang/backend version, deciding ordinary/WideEP case matrix, runtime path compatibility, collector source contracts, token/EP/EPLB semantics, materializer integration, clean-latency outputs, and default collector commands without using real-machine truth as runtime input.
---

# MoE Recorded AIC Porting

## Goal

Use this skill to make a MoE model and backend/runtime version produce correct
recorded AIC data.  The output of the work is a model-specific and
runtime-version-specific recorded case matrix, collector source coverage,
materializer wiring, clean compact latency tables, a default collector command,
and bring-up-hardware truth validation.

Real-machine truth is part of the porting validation loop, but collector runtime
and materializer logic must not depend on truth.

## Workflow

1. Identify the model MoE capability before editing code:
   - Read [model-capability-matrix.md](references/model-capability-matrix.md).
   - Decide whether the model needs ordinary MoE only or both ordinary and
     WideEP/DeepEP MoE.
   - Treat ordinary context/generation as a paired capability by default.
   - Treat WideEP context/generation as a paired capability by default when the
     backend supports WideEP/DeepEP.
   - Record the backend package version, SGLang commit/image, and MoE runtime
     path because these can change source semantics.

2. Check backend/runtime version compatibility:
   - Read [runtime-version-contract.md](references/runtime-version-contract.md).
   - Confirm SGLang MoE classes, WideEP/DeepEP flags, cuda graph behavior, and
     profile APIs still match the recorded collector assumptions.

3. Define the case matrix:
   - Family, phase, EP list, optional EPLB list, token list, backend, model
     config, and single-card EP simulation setting.
   - Do not copy DeepSeek-V3 token/EP lists blindly.
   - Read [token-ep-eplb-semantics.md](references/token-ep-eplb-semantics.md).

4. Audit collector source:
   - `moe_token_distribution` creates routed workload shape and replay inputs.
   - `moe` produces ordinary MoE source/final rows.
   - `wideep_moe` is needed only for models with WideEP/DeepEP coverage.
   - Read [collector-source-contract.md](references/collector-source-contract.md).

5. Wire the materializer and clean-latency path:
   - Source rows are input material, not final latency.
   - Final compact `latency` is the only value compared by later gates.
   - Keep no-keep and keep-source behavior identical for final latency.
   - Read [materializer-clean-latency.md](references/materializer-clean-latency.md).

6. Validate default collector behavior:
   - Default command should not require debug/source retention.
   - Debug/source retention may be controlled by env vars, but must not change
     final compact latency.
   - Check `Total errors`, expected compact files, and missing family files.

7. Validate against real-machine truth on bring-up hardware:
   - Read [porting-truth-gate.md](references/porting-truth-gate.md).
   - Gate final compact latency against truth.
   - If it does not converge, attribute model/runtime/source/materializer/truth
     causes before changing code.

8. If a new runtime version fails validation on a chosen bring-up hardware:
   - Stay in this skill; do not hand off to cross-hardware calibration yet.
   - Read [runtime-version-contract.md](references/runtime-version-contract.md).
   - Validate the new version's AIC source/materializer/clean-latency path.
   - Run or reuse real-machine truth only to validate semantics, not as
     materializer input.
   - Fix runtime/source/materializer/truth compatibility first.

## Invariants

- AIC recorded is the primary method; truth is not used at runtime.
- Use single-card EP simulation as the default AIC path unless the user asks for
  a different experimental path.
- Do not write platform-name fixes such as H20/H100 special cases.
- Do not install server/profile truth into collector source, materializer
  parameters, or compact latency tables.
- Do not generate WideEP files for a model that does not support WideEP/DeepEP.
- EPLB is optional and usually backend/WideEP-related; do not force EPLB for
  ordinary-only models.
- Treat SGLang/backend version changes as semantic changes until code-path
  compatibility has been checked.
- Model-specific parameters may be adapted only when they follow model
  structure or runtime source semantics; the recorded AIC method must remain
  the same.

## Common Files

- Collector entry: `collector/collect.py`
- Ordinary MoE collector: `collector/sglang/collect_moe.py`
- MoE distribution collector: `collector/sglang/collect_moe_distribution.py`
- WideEP collector: `collector/wideep/sglang/collect_deepep_moe.py`
- Recorded materializer: `collector/moe_recorded_materializer.py`
- Archived recorded tools: `tools/moe_calibration/recorded_moe/`

## Handoff

After porting, hand off to `moe-recorded-cross-hardware-calibration` only when
the model/runtime semantic is stable, bring-up truth gate is understood, and the
user asks whether recorded AIC is accurate across hardware platforms.  If the
runtime version itself is changing or its recorded AIC semantics are not
validated yet, remain in this skill.
