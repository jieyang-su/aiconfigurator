# KernelSim 组件与新增算子模型说明

## 组件职责

KernelSim 位于 SDK 的 kernelsim 组件中，为 Analytical 提供基于 shape、硬件能力和工程参数的算子级时延估算。模型不负责模型结构拆解、数据库查表或服务速率匹配；这些由 model、operation、PerfDatabase 和 Task 层完成。

每个模型通常提供 shape schema、估算函数、standard/low/high 参数和适用性警告。Analytical 入口统一传播 `AnalyticalConfig`，并将结果标记为 `source="analytical"`。

## 已覆盖模型

| 模型 | 主要用途 | 关键输入 |
|---|---|---|
| GEMM | BF16/FP8 block、SGLang GEMM、DeepGEMM recipe | M/N/K、dtype、recipe、硬件计算峰值 |
| FA/MLA | FA2/FA3 和 MLA attention | Q/KV shape、head 数、head dim、causal、硬件带宽/计算能力 |
| BMM | 通用 batch matrix multiplication | batch、矩阵形状、dtype、访存和计算量 |
| MoE | 普通 MoE 单卡计算部分 | token、hidden/intermediate、EP/TP、dtype、效率参数 |
| KDA | Kimi K3 Delta Attention core | phase、backend、batch/sequence、head geometry、conv/scan/recurrence route |
| DSA Index MQA | DSA/V3.2/GLM index score | ragged/paged、batch、query/context、index heads、head dim |
| DSA TopK | DSA index transform/TopK | layout、context、query、K、score recipe |
| MSA Index | MiniMax M3 BF16 Triton 风格 index | batch、Q、context、4-head index shape |
| DSV4 TopK v1/v2 | V4 CSA prefill/decode TopK | variant、batch、fresh/prefix、压缩比、K |

## 参数与硬件字段

`standard` 是当前拟合或推荐参数；`low` 和 `high` 是工程估计包络。不同模型的具体比例由模型 profile 定义，不能跨模型直接比较。它们用于表达硬件效率、启动开销和建模不确定性，不代表置信区间。

FA/MLA、DSA、MSA 和 DSV4 模型可能需要以下 system YAML 字段：

```text
sm_count
clock_hz
shared_memory_per_sm_bytes
l2_capacity_bytes
l2_bandwidth_bytes_s
vector_peak_flops
mem_bw
各 dtype tensor-core peak
```

缺少硬件字段时，模型应明确报错或发出适用性警告，而不是静默使用不匹配硬件参数。

## DSA 与 MSA 专用模型

DSA Index MQA 按 ragged/paged 语义分别处理，并包含启动项、计算和访存项；DSA TopK 使用自然 score recipe。它们主要用于 DeepSeek V3.2 和 GLM5.2，不应直接当成所有稀疏 attention 的通用精确 kernel。

MSA Index 是 M3 的 BF16、低头数 Triton 风格路径，当前为 H=4 的阶段性模型。虽然其数学任务类似 DSA index score，但 kernel launch、tile、规约和固定开销可能不同，必须保留独立模型。

## DSV4 TopK v1/v2

V4 TopK 按执行阶段分为：

- `v1`：prefill/context radix TopK，使用 causal row lengths、active rows、CTA waves 和 scan 特征。
- `v2`：decode，区分 register one-pass、register two-pass 和长上下文 cluster/persistent regime。

Flash K=512 是主要多硬件校准范围。Pro K=1024 的 v1 使用 H100 定向补采 profile；v2 的 K=1024 与 K=512 在配对数据中接近，因此复用 v2 结构，但仍发出范围警告。

这两个模型对其他 K、压缩比、score distribution、backend 和 SGLang 版本只提供有限外推，不应将参数简单按 K 线性缩放。

## 计算语义

各模型至少应保留：

- 启动/固定开销，避免小算子被 roofline 公式严重高估。
- FLOPs 与访存量，用于跨 shape 趋势和硬件差异分析。
- tile、wave、context regime 或 cache/压缩结构等已被数据证明的重要特征。
- dtype 对应的计算峰值、元素字节数和实际通信量。

对 norm、rope、store 等未单独校准的小算子，可以使用访存和启动项的快速估算；但其 source 应保持可审计，不能在报告中误称为 silicon 查表。

## 混合精度 transfer proxy

Analytical 1.1 增加三类不重新拟合的工作量迁移：

| proxy | 复用参数 | 只改变的主要工作量 | 状态 |
|---|---|---|---|
| W8A16 GEMM | BF16 GEMM Sum-3P | INT8 weight + FP32 output scale bytes | 低可信 |
| W8A16 MoE | BF16 Triton MoE | INT8 expert weight + scale bytes | 低可信 |
| W4A16 MXFP4 MoE | BF16 Triton MoE | packed MXFP4 weight + E8M0 block scale bytes | 低可信 |

这些 proxy 保持 BF16 FLOPs、compute peak、activation/intermediate traffic、launch 和效率参数。
它们不包含 backend-specific dequant、tile、workspace 和 small-M 代价，首次使用会发出 warning，
不得当作 Silicon 或重新拟合模型。通用 `int4_wo` GEMM/MoE 仍未支持。

FA v4 允许计算 dtype 与 KV cache 物理字节宽度分离。`FP8 KV + BF16 FMHA` 只减少
KV HBM/L2 流量，反量化假定融合且代价忽略；V4 sparse attention 还使用其 584-byte
packed KV contract。MLA 的 FP8 KV 只影响全局 cache 容量，算子时延暂按 BF16 compute
proxy 估算。

DSA Index MQA 在硬件无 FP8 peak 时可使用 BF16 resource-scaled proxy。它保留 H100
FP8 launch/task-service 结构，只按 BF16/FP8 理论 resource ratio 缩放任务项，不代表
存在已采 BF16 DeepGEMM kernel。

## KDA 模型边界

KDA KernelSim 覆盖 SGLang/vLLM 的 conv、prefill scan、decode recurrence 和部分 fused
decode/verify estimator。其 prefill v4 saturation 依赖固定 64-token chunk、已核实 SM 数
和 K3 geometry。当前 Analytical adapter 尚未完整共享 Silicon 的 fused route resolver，
因此 fused conv/onorm ownership、DSPARK verify 和 backend 边界仍是主要风险。

## 使用建议与风险

- 无实测数据时优先使用 Analytical `standard`，并用 `low/high` 做敏感性区间。
- FP8 GEMM 默认使用 SGLang recipe；DeepGEMM 必须显式选择 Hopper/Blackwell profile。
- MLA/attention 的 dtype 能力边界必须在前端和 operation 层校验。
- DSV4、DSA、MSA 模型强绑定 SGLang kernel 语义；跨 backend 或版本迁移需要重新验证。
- 单硬件拟合模型只能用于阶段性排序、瓶颈分析和无卡估算，不能替代多硬件实采校准。
- 国产 GPU YAML 的 SM/共享内存字段只是 KernelSim 调度代理；显式 architecture capability
  决定 FP8/FP4 路由，结果不代表 NVIDIA ISA 或 backend 可执行性。
