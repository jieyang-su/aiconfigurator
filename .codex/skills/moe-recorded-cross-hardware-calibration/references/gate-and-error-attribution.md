# Gate And Error Attribution

Gate must compare final compact AIC latency against frozen truth.  It should not
recompute latency through a separate offline materializer unless the task is
explicitly an offline replay.

## Final-Output Gate

Archived gate script:

```text
tools/moe_calibration/recorded_moe/gate/evaluate_clean_latency.py
```

Inputs:

```text
--truth-points <frozen truth csv>
--clean <platform>=<collector compact dir>
--output-dir <gate output dir>
```

Outputs:

- point-level errors;
- summary by platform/family/phase;
- over-threshold points;
- missing truth points;
- extra AIC points;
- duplicate candidate rows.

## Baseline Comparison

Recorded should be compared with historical modes when available:

- balanced;
- power_law;
- uniform;
- existing platform default.

The goal is not necessarily perfect zero error on a new platform, but recorded
should be better than old modes and should not have unexplained huge max error.

## Attribution Order

For a bad point, check in this order:

1. Is the AIC point keyed to the same family/phase/token/EP/EPLB as truth?
2. Does the model actually support this case?
3. Is token global/local/replay-local semantic correct?
4. Is EP single-card simulation vs real EP documented?
5. Did SGLang/backend version change the runtime path or source schema?
6. Is final compact latency the value being gated?
7. Are no-keep and keep-source final latency identical?
8. Is source stable for short-token cases?
9. Did source columns reach materializer intact?
10. Did materializer use a secondary/debug output incorrectly?
11. Is truth stable and correctly instrumented for this runtime version?
12. Is the point an uncovered EP/hardware extrapolation within this runtime
    semantic?

Avoid:

- platform-name patches;
- per-point truth fitting;
- changing truth to match AIC;
- using raw source as final latency without materializer semantics.
