# nsys WideEP Generation Truth

Use this reference when WideEP generation runs with cuda graph and torch
profiler cannot reliably isolate routed compute.

## Target

The truth target is WideEP generation routed expert compute:

```text
include: routed expert compute replay kernels
exclude: dispatch, combine, DeepEP/NCCL/NVSHMEM communication, attention,
         shared expert, router/topk, unrelated decode kernels
```

## Required Tools

Archived project tools:

```text
tools/moe_calibration/recorded_moe/truth/nsys/run_wideep_generation_client.py
tools/moe_calibration/recorded_moe/truth/nsys/parse_wideep_generation_event_nodes.py
tools/moe_calibration/recorded_moe/truth/nsys/select_wideep_generation_candidate.py
tools/moe_calibration/recorded_moe/truth/sglang_instrumentation/
```

## Capture Logic

1. Patch SGLang with server-side compute NVTX markers:

   ```text
   aic_nsys/layer_{layer_id}/routed/compute
   ```

2. Start SGLang under nsys and issue a request through the archived client.
3. The client uses `/start_profile` with `activities=["CUDA_PROFILER"]` and
   captures one prefill plus decode steps.
4. The parser:
   - finds capture-time compute markers before `cudaProfilerStart`;
   - collects CUDA graph nodes created inside those markers;
   - maps original graph nodes to replay graph nodes;
   - reads replay kernels after `cudaProfilerStart`;
   - filters non-compute kernels;
   - reports union kernel time by layer/session.
5. The selector chooses stable semantic candidates and marks unstable points.

If SGLang version changed, repeat this as a smoke before any full run.  The
parser depends on both marker placement and CUDA graph metadata; either can
change across runtime versions.

## Validation Before Trusting Values

The nsys SQLite must contain:

```sql
NVTX_EVENTS where text like 'aic_nsys/layer_%/routed/compute'
CUPTI_ACTIVITY_KIND_RUNTIME row for cudaProfilerStart_v4000
CUDA_GRAPH_NODE_EVENTS
CUPTI_ACTIVITY_KIND_KERNEL graphNodeId
```

Reject the run if:

- no `aic_nsys` compute markers exist;
- markers exist only on the client side;
- graph node metadata is missing;
- selected kernels contain DeepEP/NCCL/NVSHMEM/attention/shared/topk names;
- selected layer coverage is missing for expected MoE layers;
- values are stable but obviously from a different window.
- the parser evidence was validated on an older SGLang version but not on the
  current one.

## Common Failure Meaning

```text
missing marker:
  SGLang patch/env not active in server workers

marker outside graph capture:
  wrong placement or cuda graph capture path changed

empty selected kernels:
  graph-node mapping changed or marker wrapped wrong code

very small truth:
  client/window marker, not routed compute

very large truth:
  dispatch/combine or long-tail communication leaked into compute selection
```
