# Porting Truth Gate

Porting is not complete when the collector merely runs.  A migrated model or
runtime version must also pass a bring-up-hardware truth validation.

Truth is used only for validation and attribution.  It must not feed collector
runtime, source row selection, or materializer inputs.

## Gate Inputs

Use:

- final compact AIC latency, not raw source;
- a model/family-appropriate real-machine truth method;
- the same public key semantics:
  - family;
  - phase;
  - token;
  - EP;
  - EPLB;
  - model/runtime version.

For WideEP generation with cuda graph, use nsys semantic routed/compute truth
or another proven method with equivalent marker/parser evidence.

## Expected Output

The gate should produce:

- point-level AIC vs truth error;
- MAPE / p90 / max error;
- missing truth points;
- extra AIC points;
- worst-point source/materializer/truth diagnostics.

## If It Does Not Converge

Treat non-convergence as a porting issue until proven otherwise.

Check in this order:

1. Case mismatch:
   - model config;
   - family/phase;
   - token semantic;
   - EP/EPLB;
   - quant/backend flags;
   - cuda graph mode.

2. Source mismatch:
   - source schema missing or renamed;
   - replay shape no longer matches case matrix;
   - source health guard changed the selected source row;
   - no-keep and keep-source final latency differ.

3. Materializer mismatch:
   - required model parameters wrong;
   - expert/top-k/shared-expert assumptions wrong;
   - model-specific constants need updating;
   - runtime source fields changed;
   - debug/secondary outputs are used as primary latency.

4. Truth mismatch:
   - marker not in server worker;
   - marker wraps dispatch/combine instead of routed compute;
   - torch profiler window includes neighboring work;
   - nsys parser evidence belongs to a different runtime path;
   - selected kernels include communication/attention/topk/shared work.

5. Runtime/kernel change:
   - operator implementation changed in the backend version;
   - cuda graph capture/replay changed;
   - kernel names or graph-node metadata changed;
   - real latency changed and truth confirms it.

## Allowed Fixes

Allowed:

- update model-specific parameters derived from model config;
- update source schema adapters for a backend/runtime version;
- update materializer inputs when source semantics changed;
- update truth instrumentation/parser for the runtime version;
- add thin model/runtime-driven compatibility guards.

Not allowed:

- platform-name fixes;
- version-name fixes that only chase one point;
- truth-derived scales as runtime input;
- changing the recorded AIC method into a thick policy;
- forcing truth to match AIC.

If the recorded method cannot generalize to a model/runtime because the runtime
operator semantics genuinely changed, document the cause and add a thin
compatibility adapter for that model/runtime semantic.  Then re-run the default
collector and the bring-up truth gate.

