# Runtime Version Contract

Recorded AIC porting is not only model-dependent.  It is also backend/runtime
version-dependent.  A SGLang image or package upgrade can change MoE execution
paths, cuda graph behavior, profile APIs, and available collector inputs.

## Record The Runtime

For every porting run, record:

```yaml
backend: sglang
backend_package_version:
backend_git_commit:
container_image:
python_path:
model_path:
model_config_revision:
cuda_version:
torch_version:
deepep_version_or_source:
```

## Check SGLang MoE Path

Before assuming old logic applies, locate the actual code path in the target
SGLang version:

```text
FusedMoE.forward_impl
run_moe_core
dispatcher.dispatch
dispatcher.combine
moe_forward_piecewise_cuda_graph_impl
DeepEP / WideEP dispatcher classes
```

Questions to answer:

- Did the fused MoE implementation file move?
- Did ordinary MoE and WideEP/DeepEP share the same class as before?
- Did generation switch into a piecewise cuda graph path?
- Are EP/EPLB flags and CLI names unchanged?
- Did source columns or CSV names change?
- Did `/start_profile` request schema change?
- Did kernel names or graph-node metadata change enough to break parsers?

## Collector Compatibility

If runtime path changed:

- update collector routing/version guards;
- update source parsing and materializer inputs;
- add compatibility notes to the case matrix;
- run a small default collector smoke before full collection;
- do not claim old recorded source semantics still hold without evidence.

## Truth Compatibility Handoff

If the version change can affect profile/truth marker placement, rerun
instrumentation smoke before trusting truth comparisons for that version.
Frozen truth from an old version may be useful as historical context, but it
does not prove the new runtime path is correctly instrumented.

## New Runtime Version Validation Failure

If a new runtime version fails recorded AIC validation on the chosen bring-up
hardware, treat it as a porting/compatibility problem first, not as
cross-hardware calibration.  Most version upgrades should not need a new
recorded method because operator semantics usually stay close; prove the cause
before adding compatibility logic.

Recommended order:

1. Confirm the compared points are identical:
   - same model config and layer truncation;
   - same family/phase/token/EP/EPLB;
   - same public token semantic;
   - same quantization and backend flags;
   - same cuda graph mode.

2. Diff runtime code paths:
   - ordinary MoE class/function path;
   - WideEP/DeepEP dispatcher path;
   - `run_moe_core` call site;
   - piecewise cuda graph path;
   - EPLB flag handling.

3. Diff collector source schema:
   - compact rows and distributions;
   - raw/source columns;
   - replay manifest shape;
   - source health guard decisions;
   - no-keep vs keep-source final latency.

4. Diff materializer inputs:
   - required columns present;
   - secondary/debug outputs not used as primary source;
   - source-to-final-latency conversion unchanged or intentionally adapted;
   - no truth-derived values entered source or final latency.

5. Re-validate truth instrumentation:
   - marker exists in server workers;
   - marker wraps routed compute only;
   - parser evidence exists for the current version;
   - selected kernels exclude dispatch/combine/communication/attention/shared/topk.

6. Interpret the result:
   - If AIC source/final changed but runtime kernels did not, fix collector or
     materializer compatibility.
   - If truth changed but source did not, audit instrumentation/parser.
   - If runtime kernels really changed and truth confirms it, update the
     version's recorded source/materializer semantics with a thin,
     model/runtime-driven rule.
   - If the new version's recorded AIC semantics are validated on bring-up
     hardware, that version can become the fixed input for cross-hardware
     calibration.

Do not:

- write version-name special cases to chase one point;
- use truth as collector runtime input.
