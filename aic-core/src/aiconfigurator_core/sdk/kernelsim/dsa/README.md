# DSA Index KernelSim Models

This package contains the analytical models for the sparse-attention Index
MQA score kernel and fused TopK/index-transform kernel used by the granular DSA
paths of DeepSeek V3.2 and GLM 5.2.

## Calibration Boundary

The parameters were fitted from one H100 SXM with SGLang 0.5.12. The collector
used AIC `benchmark_with_power` and mandatory CUDA Graph replay, excluding
eager-wrapper allocation, per-call CUDA event construction and host-side
synchronization from the modeled kernel boundary.

The measured scope is deliberately narrow:

- Index MQA: DeepGEMM-style FP8 E4M3 paged/ragged score generation;
- TopK: SGLang fused transform over FP32 scores with int32 index output;
- index heads 32/64, head dimension 128, page/KV block 64;
- query tile 16 and ragged score chunk capacity 8,000,000 slots;
- Index MQA: batch 1--32, ragged query 1/16/64/256 and context 2048--65536;
- TopK natural: batch 1--128, query rows through 8192 and context through 524288;
- TopK threshold 2048; paged and ragged layouts.

Every estimate emits `DsaIndexModelWarning` and records the limitation in the
result. This is intentional: the model is a semi-mature no-table fallback, not
a claim of silicon-equivalent accuracy.

## Index MQA v2

The selected candidate is layout-specific task service rather than roofline:

```text
paged:
  N_service = batch * ceil(context / 64)
  latency = floor_paged + N_service * cycles_paged / clock

ragged:
  split logical scores at 8M slots
  build 16-row query tiles and 64-token KV blocks
  cap cross-request service span by max(1, 128 / index_heads)
  latency = chunk_count * floor_ragged
          + N_service_ragged * cycles_ragged / clock
```

The H100 standard parameters are:

```text
floor_paged_us    = 3.6963528879
floor_ragged_us   = 4.9474693849
cycles_paged      = 3.7524554887
cycles_ragged     = 414.7553222539
```

Five-fold physical-shape OOF MAPE was 8.89%. FLOPs and bytes remain in the
result as inspectable lower-bound diagnostics but do not determine v2 latency.
The service cycles are fitted scheduler coefficients, not portable instruction
cycles. In particular, the 16/64 tiles, 8M chunk rule and `128 / heads` cap may
change with DeepGEMM, SGLang or GPU architecture.

## TopK Natural v3

Production serving uses the natural FP32 score emitted by Index MQA.  The v3
model was refitted on 348 deduplicated physical shapes, including context up to
524K and 8192 query rows:

```text
context <= K:
  tile = 256
  waves = ceil(query_rows / 256)

context > K:
  tile = 128
  waves = ceil(query_rows / 128)
  tail_waves = max(0, waves - 4)

latency = branch_floor
        + executed_score_bytes / HBM * inverse_efficiency
        + ragged_tail_score_bytes / HBM * tail_inverse_efficiency
```

Five-fold physical-shape OOF MAPE is 11.91%, median predicted/measured ratio
0.989 and P90 APE 27.03%.  This replaces the v2 natural branch, which had
27.63% MAPE after the large-row data exposed systematic underestimation.

The wave sizes, four-wave tail, floors and efficiencies are H100/SGLang kernel
recipe parameters, not GPU architectural constants.  Only the body scales with
the supplied HBM bandwidth.  SM count/clock, L2, shared memory, register
occupancy and launch latency are absorbed rather than independently modeled;
cross-GPU extrapolation is therefore weak.  H100-to-H200 with the same kernel
is a plausible first-order use, while A100, Blackwell and non-NVIDIA targets
remain low-confidence fallbacks.

## TopK Diagnostic v2

Synthetic flat/top-last distributions retain the v2 kernel regime:

```text
context <= K:
  latency = short_floor(layout)

context > K:
  latency = short_floor(layout)
          + long_floor(distribution)
          + a(distribution) * log2(context / K)
          + b(distribution) * max(0, (context - 32768) / 32768)
          + r(distribution) * max(0, rows / 166.113 - 1)
```

`flat` and `top_last` remain diagnostic alternatives. Production serving uses
the separate natural v3 wave model above.

These distributions are collector diagnostics and do not determine the
ANALYTICAL serving estimate. The top-last recipe remains high risk even inside
the original H100 calibration range.

## Parameter Levels

`standard` uses the H100 v2 Index MQA/diagnostic TopK parameters and v3 natural
TopK refit. `high` and `low` are conservative and
optimistic engineering scenarios:

- Index MQA: `1.20x` and `0.80x` around the standard task-service estimate;
- TopK: `1.35x` and `0.75x` around the standard kernel-regime estimate.

They are planning envelopes, not statistical confidence intervals and not
independent cross-hardware calibrations.

## TopK and Transfer Limits

The formula accepts any positive K and moves the short/long boundary and
`log2(context / K)` coordinate accordingly. The numerical coefficients remain
strongly coupled to K=2048. A different K can alter the selection algorithm,
register/shared-memory use, CTA shape, output traffic and backend dispatch.
Consequently:

- K=2048 on the calibrated recipe is the only directly validated case;
- another K with the same kernel family is a low-confidence trend estimate and
  emits a warning;
- a changed kernel symbol, tile, backend, score layout or transform boundary is
  a new recipe and requires new modeling;
- changing output bytes alone is not a valid adjustment: the bytes-only
  candidate had 53.45% OOF MAPE.

The model has not been validated on H200, Blackwell, consumer GPUs or non-NVIDIA
accelerators. Cross-hardware use should retain the formula only as a hypothesis;
the H100 floors, task cycles, row saturation and tail threshold must not be
treated as universal hardware properties.

## AIC Integration

`sdk.operations.dsa.DSAIndexScore` and `DSATopKSelect` call these models through
`sdk.kernelsim.analytical`. DeepSeek V3.2 and GLM 5.2 both build those granular
operations in `sdk.models.deepseek_v32`; GLM's shared-index layer fraction is
applied by the operation scale factor. In ANALYTICAL mode the module primary is
not queried, so these estimates remain independent of silicon tables.
