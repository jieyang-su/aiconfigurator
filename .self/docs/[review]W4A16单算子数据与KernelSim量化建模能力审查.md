# W4A16 单算子数据与 KernelSim 量化建模能力审查

审查日期：2026-08-14。

## 当前结论

AIC 必须区分量化 identity、Silicon 实采、Empirical 跨精度迁移、SOL roofline 和
KernelSim Analytical。当前代码状态如下：

1. Hopper 没有原生 FP4 Tensor Core 仍可执行 W4A16。权重在加载路径融合解量化，
   activation 和矩阵计算属于 BF16 family，不能使用 `fp4_tc_flops`。
2. 普通 W4A16 GEMM 的直接数据很少，主要是 A100/L40S/H200 的旧 TRT-LLM
   `PLUGIN_V2_WeightOnlyQuantMatmul`；H100 SGLang/vLLM 没有通用 W4A16 GEMM 表。
3. H100/H200 MoE 的 W4A16 数据较多，包含 Marlin、Triton MXFP4、vLLM 和 TRT-LLM，
   但 SGLang K3 EP>1 存在 collector/backend 异常，不能无筛选纳入拟合。
4. 本轮已经为 `w4a16_mxfp4` 和 `w4a16_mxfp4_cutlass` 增加
   `w4a16_mxfp4_bf16_transfer`。它让 Hopper Kimi K3 和同 identity 的模型可完成
   Analytical 执行，但没有重新拟合，只是低可信 transfer proxy。
5. 通用 `GEMMQuantMode.int4_wo` 和 `MoEQuantMode.int4_wo` 仍没有生产 KernelSim
   recipe；Kimi K2.5 或全层 INT4 模型仍可能明确失败。
6. W8A16 已增加 `int8_wo` GEMM/MoE transfer proxy，并保持 Python/Rust enum parity。
   它同样不是 W8A16 Silicon 校准。

“Hopper K3 Analytical 因 W4A16 完全无法运行”已不是当前状态。正确表述是：路径已闭环，
MoE 结果仍受 BF16 参数迁移和 backend 外推限制。

## 精度语义

| AIC identity | 权重/激活 | compute peak | 当前 Analytical |
|---|---|---|---|
| `GEMMQuantMode.int8_wo` | INT8/BF16 | BF16 | W8A16 transfer proxy |
| `GEMMQuantMode.int4_wo` | INT4/BF16 | BF16 | 未支持 |
| `MoEQuantMode.int8_wo` | INT8/BF16 | BF16 | W8A16 transfer proxy |
| `MoEQuantMode.int4_wo` | INT4/BF16 | BF16 | 未支持 |
| `MoEQuantMode.w4a16_mxfp4` | MXFP4/BF16 | BF16 | W4A16 transfer proxy |
| `MoEQuantMode.w4a16_mxfp4_cutlass` | MXFP4/BF16 | BF16 | 同一 W4A16 transfer proxy |
| `w4a8_mxfp4_mxfp8*` | MXFP4/MXFP8 | FP8 | 暂借 NVFP4 proxy |
| `nvfp4` | NVFP4/NVFP4 | FP4 | Blackwell CuTeDSL recipe |

W4A16 的收益主要来自权重带宽和 cache footprint，不等价于得到 4 倍 BF16 峰值。
真实表现还依赖 fused dequant、scale layout、group size、tile、small-M 分派和 launch。

## Silicon 数据覆盖

### 普通 GEMM

直接标记为 `int4_wo` 的 GEMM 数据主要包括：

| system | backend/version | 规模 | kernel source |
|---|---|---:|---|
| A100 SXM | TRT-LLM 1.0.0 | 6,048 行 | `PLUGIN_V2_WeightOnlyQuantMatmul` |
| L40S | TRT-LLM 1.0.0 | 6,048 行 | 同上 |
| H200 SXM | TRT-LLM 1.2.0rc5 | 9,240 行 | 同上 |

这些数据不足以建立跨 SGLang/vLLM/Marlin 的通用 W4A16 GEMM。当前生产 collector
也没有形成完整的 H100 weight-only GEMM sweep。

### MoE

H100/H200 的 SGLang、vLLM 和 TRT-LLM 数据包含 `int4_wo` 或
`w4a16_mxfp4`，通常覆盖 token、TP、EP 和多个模型 shape。需要保留三个限制：

- Kimi K3 SGLang Marlin 的 EP>1 行存在大量 IMA crash/缺测，不能视为随机缺失；
- vLLM 同类数据更完整，说明异常不是 W4A16 硬件的普遍限制；
- Marlin、Triton、CUTLASS 和 TRT-LLM 的 collector 边界不同，不能共用一个高精度参数集。

`w4a16_mxfp4_cutlass` 是 loader 根据 kernel source 重标的 identity。当前专用 Silicon
覆盖仍不足，不能把 K3 Marlin 或 GPT-OSS Triton 行冒充 DeepSeek V4 CUTLASS 真值。

## 当前 KernelSim 实现

### W4A16 MoE transfer

模型复用 `bf16_triton` 的 Sum-3P 参数：

```text
latency = launch_bf16
        + flops_bf16 / (BF16_peak * eta_compute_bf16)
        + logical_bytes_w4a16 / (HBM_BW * eta_mem_bf16)
```

只改变以下工作量：

- 权重值按 MXFP4 packed 0.5 byte/value；
- 每 32 个权重加入一个逻辑 E8M0 scale byte；
- activation、intermediate 和 output 保持 BF16；
- compute FLOPs 和 BF16 peak 不变。

没有显式计入：unpack/dequant 指令、backend tile、workspace、small-M 分支和重新拟合的
launch floor。模型首次使用会发出 `W4A16TransferModelWarning`，scope 标记
`provisional BF16 parameter transfer`；EP>1 额外标注理想均匀外推。

### W8A16 GEMM/MoE transfer

W8A16 同样复用 BF16 参数，只替换 INT8 weight 和 FP32 per-output-channel scale 流量。
它用于 BF16-only 硬件的粗粒度架构比较，不代表已有成熟 W8A16 backend 性能模型。

### 仍未覆盖

- 通用 INT4 W4A16 GEMM；
- `int4_wo` MoE；
- Marlin/Triton/CUTLASS 分开的 W4A16 参数；
- W4A8 MXFP4/MXFP8 的独立模型；
- 非 NVIDIA 硬件的 tile、scheduler 和 dequant 代价。

## Database mode 行为

| mode | 有原生表 | 无原生表 |
|---|---|---|
| SILICON | 精确命中或 Silicon 插值/外推 | typed error |
| HYBRID | 优先 Silicon | 按 policy 做 empirical transfer |
| EMPIRICAL | 本 quant utilization 或 sibling transfer | 无候选则失败 |
| SOL | BF16 compute roof 加低比特权重流量 | 理论下界/粗估 |
| ANALYTICAL | 不查表 | 仅上述显式 KernelSim recipe/proxy |

`source="empirical"` 和 `source="analytical"` 都不等于实测。报告必须同时保留 recipe、
transfer kind、warning 和 operation provenance。

## 对模型的实际影响

### Kimi K3

K3 checkpoint 通常只量化 routed experts，普通 attention/dense/shared GEMM 保持 BF16。
因此 Hopper 的直接阻塞原本是 `w4a16_mxfp4` MoE；本轮 transfer proxy 已打通端到端，
但 LatentMoE 又是主要时延项，结果只能作为低可信排序和瓶颈参考。

### Kimi K2.5 与全层 INT4

若 routed experts 使用 `int4_wo`，或 artifact 没有 ignore list 导致普通 GEMM 也为
`int4_wo`，当前 Analytical 仍会在对应未支持 recipe 处失败。

### DeepSeek V4 Pro Hopper

`w4a16_mxfp4_cutlass` 当前可通过同一 BF16 transfer proxy 执行，但它没有 CUTLASS
专用 Silicon 校准。可运行不代表该路径已成熟。

## 后续优先级

1. 按 Marlin、Triton、CUTLASS 分开清洗 H100/H200 数据并拟合 W4A16 MoE 参数；排除
   collector 的 EP>1 错误行。
2. 补 H100 纯 kernel W4A16 GEMM，覆盖 M=1..128、常见 N/K、group size 和 backend。
3. 为 `int4_wo` MoE 建立独立 recipe，不与 MXFP4 identity 混用。
4. 为 W4A8 MXFP4/MXFP8 建立独立模型，停止长期借用 NVFP4 CuTeDSL。
5. 增加 H100/H200 K3、K2.5、V4 Pro 的 no-GPU smoke 和跨 mode provenance 测试。

## 使用边界

- W4A16/W8A16 transfer proxy 只适合无实采时的方案筛选和敏感性分析。
- 不能把 warning 隐藏后把结果标为 Silicon，也不能用其证明 backend 可执行性。
- decode 小 token 和大 EP 是最高风险外推区。
- 正式容量结论应优先使用同 system/backend/kernel/shape 的 Silicon 数据或专项采集。
