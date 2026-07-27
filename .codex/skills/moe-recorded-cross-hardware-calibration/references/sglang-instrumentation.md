# SGLang Instrumentation

Default SGLang images may not contain AIC MoE truth markers.  When changing
SGLang version, package, git commit, image, or model runtime path, reapply and
validate instrumentation before running truth.

## Reference Files

Use the archived reference files first:

```text
tools/moe_calibration/recorded_moe/truth/sglang_instrumentation/sglang_moe_profile_markers.md
tools/moe_calibration/recorded_moe/truth/sglang_instrumentation/fused_moe_triton_layer_reference.patch
```

## Required Marker Contract

Logical profiler ranges:

```text
aic_moe/layer_{layer_id}/routed/dispatch
aic_moe/layer_{layer_id}/routed/compute
aic_moe/layer_{layer_id}/routed/combine
aic_moe/layer_{layer_id}/routed/all_reduce
```

WideEP generation nsys marker:

```text
aic_nsys/layer_{layer_id}/routed/compute
```

The nsys marker must wrap the same operation as `aic_moe/.../routed/compute`.

## Where To Patch

For fused MoE Triton SGLang paths, start with:

```text
python/sglang/srt/layers/moe/fused_moe_triton/layer.py
```

Search terms when the file moved:

```text
FusedMoE.forward_impl
dispatcher.dispatch
run_moe_core
dispatcher.combine
```

Version changes may also move the target into model-specific files such as
`deepseek_v2.py`, `qwen3_moe.py`, or a piecewise cuda graph implementation.
When that happens, keep the same semantic contract but update the patch target.

Correct placement:

```python
with _aic_moe_record_function(f"aic_moe/layer_{self.layer_id}/routed/compute"):
    with _aic_nsys_range(f"aic_nsys/layer_{self.layer_id}/routed/compute"):
        combine_input = self.run_moe_core(dispatch_output=dispatch_output)
```

Do not wrap:

- the whole MoE layer;
- dispatch/combine;
- router/topk;
- attention;
- shared expert;
- all-reduce;
- the HTTP client request.

## Runtime Flags

Server workers must receive:

```text
SGLANG_AIC_MOE_PROFILE=1
SGLANG_AIC_NSYS_SEMANTIC_MARKERS=1
SGLANG_AIC_NSYS_CUDA_EVENTS=1
```

If these are set only in the client process, the trace is invalid.

## Verification

After a smoke capture, query the exported nsys SQLite:

```sql
select count(*) from NVTX_EVENTS
where text like 'aic_nsys/layer_%/routed/compute';
```

Also verify:

- `cudaProfilerStart_v4000` exists;
- `CUDA_GRAPH_NODE_EVENTS` has rows;
- replay kernels have `graphNodeId`;
- expected MoE layers appear.

## Version Migration Checklist

For each SGLang/backend version change:

```text
record old/new package version and commit
record container image and mounted source path
locate current MoE dispatch/compute/combine code
verify env flags reach server workers
run one ordinary MoE profile smoke if ordinary truth is needed
run one WideEP generation nsys smoke if WideEP generation truth is needed
check NVTX marker count and layer coverage
check graph node metadata and parser selected kernels
only then run full truth or reuse old truth for comparison
```

Do not assume a reference patch applies cleanly.  It is a semantic template,
not a guaranteed patch for every SGLang version.
