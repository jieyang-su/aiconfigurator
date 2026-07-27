# Cross-Hardware Generalization

The target is a platform-independent recorded MoE method within one fixed model
and SGLang/backend runtime semantic.  New hardware should be able to run AIC
recorded collector without first fitting a hardware-specific truth scale.

A SGLang/backend version is an input to this workflow, not a calibration axis.
If the version changes, leave this workflow and use `moe-recorded-aic-porting`
to validate the new version's recorded AIC semantics first.  After that, run a
separate cross-hardware calibration for the validated version if needed.

## Reuse Old Hardware Baselines

If old hardware data is already frozen and trusted:

- reuse old AIC compact data;
- reuse old frozen truth;
- do not rerun old hardware by default;
- only rerun when semantics or coverage changed.

For a new hardware:

1. run recorded AIC;
2. run matching real-machine truth;
3. gate new hardware and old hardware together;
4. compare recorded against old modes;
5. analyze errors.

## What Counts As Generalized

Good signs:

- no platform-name branches;
- materializer uses current hardware source and model semantics;
- source/materializer contracts are stable within the runtime version being
  calibrated;
- source health guard only controls measurement stability;
- fit-all across available hardware is good;
- leave-one-platform failures are explainable by missing coverage or known
  extrapolation gaps.

Bad signs:

- `if platform == "h100"` style logic;
- truth-derived scale required before AIC can run;
- runtime source/truth semantics are not fixed before starting hardware gate;
- a small set of points dominates max error with no source/truth explanation;
- WideEP generation selected by a parser with missing marker evidence.

## Offline Replay Before Mainline

Use archived replay tools:

```text
tools/moe_calibration/recorded_moe/replay/analyze_source_contract.py
tools/moe_calibration/recorded_moe/replay/analyze_generalization_flow.py
```

Replay candidate rules:

- include old and new hardware source for the same runtime semantic;
- include all frozen truth;
- include recorded and baseline candidate sets;
- report MAPE, p90, max error, and worst point;
- mark missing truth instead of extrapolating silently.

Only merge collector/materializer changes after replay converges.
