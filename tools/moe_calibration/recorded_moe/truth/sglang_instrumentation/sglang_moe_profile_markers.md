# SGLang MoE Profile Marker Contract

This file records the SGLang-side instrumentation required by recorded MoE
truth workflows.  It is a reference for re-applying markers when the target
container uses a fresh SGLang checkout.

## Target File

For SGLang versions using the fused MoE Triton implementation, start from:

```text
python/sglang/srt/layers/moe/fused_moe_triton/layer.py
```

Other model-specific MoE paths can exist, for example `deepseek_v2.py` or
model-specific piecewise implementations.  The same contract applies, but the
exact file and function must be re-located by searching for:

```text
run_moe_core
dispatcher.dispatch
dispatcher.combine
FusedMoE.forward_impl
```

## Required Marker Semantics

The real-machine truth target is routed expert compute, not dispatch/combine
communication and not attention/topk/shared expert work.

Required logical ranges:

```text
aic_moe/layer_{layer_id}/routed/dispatch
aic_moe/layer_{layer_id}/routed/compute
aic_moe/layer_{layer_id}/routed/combine
aic_moe/layer_{layer_id}/routed/all_reduce
```

Only `routed/compute` is the MoE routed compute truth target.  The other ranges
are useful for audit and for proving they are excluded.

For WideEP generation nsys truth, an additional NVTX marker is required:

```text
aic_nsys/layer_{layer_id}/routed/compute
```

This NVTX range must wrap the exact same compute call as
`aic_moe/layer_{layer_id}/routed/compute`.

## Correct Placement

For normal fused MoE:

```python
with _aic_moe_record_function(f"aic_moe/layer_{self.layer_id}/routed/dispatch"):
    dispatch_output = self.dispatcher.dispatch(...)

with _aic_moe_record_function(f"aic_moe/layer_{self.layer_id}/routed/compute"):
    with _aic_nsys_range(f"aic_nsys/layer_{self.layer_id}/routed/compute"):
        combine_input = self.run_moe_core(dispatch_output=dispatch_output)

with _aic_moe_record_function(f"aic_moe/layer_{self.layer_id}/routed/combine"):
    final_hidden_states = self.dispatcher.combine(...)
```

The nsys parser learns CUDA graph node identity from kernels created inside
`aic_nsys/layer_N/routed/compute` during graph capture, then maps those nodes to
replay kernels after `cudaProfilerStart`.

Do not put the nsys marker around:

- the whole MoE layer;
- `dispatcher.dispatch`;
- `dispatcher.combine`;
- router/topk;
- attention;
- shared expert;
- all-reduce;
- the client-side HTTP request.

## Runtime Flags

The client enables these flags before launching SGLang:

```text
SGLANG_AIC_MOE_PROFILE=1
SGLANG_AIC_NSYS_SEMANTIC_MARKERS=1
SGLANG_AIC_NSYS_CUDA_EVENTS=1
```

Expected behavior:

- `SGLANG_AIC_MOE_PROFILE=1` enables torch profiler `record_function` ranges.
- `SGLANG_AIC_NSYS_SEMANTIC_MARKERS=1` enables server-side NVTX ranges.
- `SGLANG_AIC_NSYS_CUDA_EVENTS=1` is reserved for implementations that also add
  CUDA event begin/end markers.  The current graph-node parser requires the
  compute NVTX range; CUDA event markers are an audit aid, not the selected
  truth source.

## Validation Checklist

Before trusting a WideEP generation nsys truth run, verify the exported nsys
SQLite contains:

```text
NVTX_EVENTS text like aic_nsys/layer_%/routed/compute
CUPTI_ACTIVITY_KIND_RUNTIME row for cudaProfilerStart_v4000
CUDA_GRAPH_NODE_EVENTS rows
CUPTI_ACTIVITY_KIND_KERNEL rows with graphNodeId
```

Parser-side expectations:

- `parse_wideep_generation_event_nodes.py` must find capture-time compute
  markers before `cudaProfilerStart`.
- It must map marker-owned `CUDA_GRAPH_NODE_EVENTS.graphNodeId` through
  `originalGraphNodeId` to replay kernel `graphNodeId`.
- Excluded kernels include DeepEP/NCCL/NVSHMEM/attention/shared/topk-related
  names.
- Selected latency is the union of routed compute replay kernels, normally
  aggregated by layer/session outside the parser.

Failure signatures:

- No `aic_nsys/layer_%/routed/compute`: SGLang marker patch is missing or env
  flag is not passed to server workers.
- NVTX exists but no graph nodes: marker was outside CUDA graph capture, or nsys
  did not collect graph metadata.
- Graph nodes exist but selected kernels are empty: marker wrapped the wrong
  code region or graph-node mapping changed in the SGLang/CUDA version.
- Selected value is much smaller than old truth and has no layer coverage:
  likely client-side or wrong-window marker.
- Selected value includes DeepEP/NCCL/NVSHMEM kernels: parser include/exclude
  rules or marker placement is wrong.

