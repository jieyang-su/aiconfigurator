# DeepSeek V4 Operator Calibration

This note describes the DeepSeek V4 operator calibration path used to estimate
performance for a target system from a measured source system. The common use
case is estimating PRO6000 behavior from H20 silicon data while keeping the
target system topology and communication data.

## Example

```bash
AIC_ALLOW_UNSUPPORTED_DSV4_TP=1 \
AIC_DSV4_ATTENTION_CALIBRATE_FROM=h20_sxm \
AIC_DSV4_ATTENTION_CALIBRATE_SYSTEM_PATTERN=PRO6000 \
AIC_DSV4_ATTENTION_CALIBRATE_MODE=roofline \
AIC_DISABLE_HYBRID_SHARED_LAYER=1 \
AIC_PREFER_NCCL_FOR_CUSTOM_ALLREDUCE=1 \
python3 tools/perf_compare_automation/scripts/dsv4flash_topology_automation.py --config tools/perf_compare_automation/configs/DSV4-PRO-1024.json
```

## Data Flow

1. The requested target system is still loaded normally. For example, a
   `PRO6000` system uses the PRO6000 YAML, GPU spec, node topology, P2P
   bandwidth, NCCL version, and target communication files.

2. If `AIC_DSV4_ATTENTION_CALIBRATE_FROM` is set and the target system name
   contains `AIC_DSV4_ATTENTION_CALIBRATE_SYSTEM_PATTERN`, operator perf files
   are loaded from the source system's data directory instead. For example,
   `h20_sxm` measured operator rows can be used as the silicon baseline for a
   PRO6000 estimate.

3. NCCL and oneCCL data stay on the target system. Communication is topology
   and network dependent, so the calibration path does not borrow source-system
   NCCL tables. When `AIC_PREFER_NCCL_FOR_CUSTOM_ALLREDUCE=1` is set, custom
   allreduce queries prefer the target NCCL path instead of a borrowed custom
   allreduce table.

4. The source silicon latency is scaled to the target by the selected
   calibration mode.

## Roofline Mode

`AIC_DSV4_ATTENTION_CALIBRATE_MODE=roofline` uses a per-operator speed-of-light
ratio:

```text
estimated_target_latency = measured_source_latency * target_SOL / source_SOL
```

For the same operator shape, the source and target SOL values are computed from
that system's GPU spec:

```text
sol_math = ops / gpu_compute
sol_mem  = bytes / gpu_mem_bw
SOL      = max(sol_math, sol_mem)
```

This is different from a global scaling factor. A compute-bound operator can
speed up on a target with higher tensor-core throughput, while a memory-bound
operator can slow down on a target with lower memory bandwidth.

The current per-operator roofline path is wired into the main DeepSeek V4
operator surfaces:

- GEMM
- compute scale and scale matrix
- MoE
- WideEP MoE compute
- DeepSeek V4 mHC
- DeepSeek V4 context attention module
- DeepSeek V4 generation attention module

Operators without a dedicated per-op SOL callback fall back to the global
calibration ratio for the selected mode.

## Other Modes

`AIC_DSV4_ATTENTION_CALIBRATE_MODE` also accepts these values:

- `roofline`: per-operator `target_SOL / source_SOL` when available.
- `memory`: global `source_mem_bw / target_mem_bw`.
- `compute`: global `source_compute / target_compute`.
- `min`: global `min(memory_scale, compute_scale)`.
- A numeric value: fixed latency multiplier.

`roofline` is the preferred mode for source-to-target extrapolation because it
uses each queried shape's compute and memory intensity.

## Caveats

This calibration estimates target latency from source silicon measurements. It
does not replace target silicon data. Kernel implementation differences,
scheduling effects, launch overheads, cache behavior, and quantization-specific
hardware details can still make the real target diverge from the estimate.

Communication remains target-specific by design. If a topology comparison looks
unexpected, first check whether the dominant difference is coming from operator
calibration, NCCL/custom allreduce selection, or inter-node bandwidth.
