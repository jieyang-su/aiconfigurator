# 新模型 Analytical 模式适配说明

## 定位

Analytical 是 AIC 在缺少实机性能数据时的理论评估路径。它以模型配置和算子语义推导实际 shape，拆解模型模块，调用 KernelSim 或小算子公式估算时延，并通过通信理论模型完成端到端和 PD 分离仿真。

它不等同于 silicon 查表，也不承诺还原某个 SGLang kernel 的全部调度细节。当前实现优先覆盖 SGLang、NVIDIA GPU 和已完成校准的模型/算子组合。

## 模型覆盖

### DeepSeek V3.2 与 GLM5.2

两者使用 DSA granular attention 表达：

```text
投影 GEMM
index MQA
TopK/index transform
稀疏 attention core
输出投影及必要的小算子
```

Analytical 强制走 granular 路径，不依赖 DSA module 实测表。Index MQA 和 TopK 使用专用 KernelSim；GEMM、FA 类主体和小算子复用现有模型。DSA 的 FP8 index 数据目前主要对应 SGLang 的 FP8 DeepGEMM 风格路径，跨 backend 使用时需谨慎。

### MiniMax M3

M3 的模型图区分前三个 dense layer 和后续 MSA/MoE layer。dense layer 使用普通 QKV、attention、projection 和 dense FFN；稀疏层使用 MSA index、block selection、selected-block attention、共享专家及相关通信。

MSA index 是 BF16 Triton 风格的阶段性 KernelSim，当前主要依据 H100 数据；它不是 DSA index 的完全等价替代。MSA 的 TopK/block-max 仍以轻量或代理模型估算，结果适合趋势评估，不宜视为跨硬件精确预测。

### DeepSeek V4

V4 attention 按压缩比例拆为：

- SWA：短窗口 attention。当前部分 silicon module 路径沿用 HCA 近似，Analytical 使用真实 ratio-0 granular 语义。
- CSA：c4 压缩缓存，包含 index MQA、TopK、稀疏 attention core 和压缩 KV 流程。
- HCA：c128 压缩缓存，使用 HCA attention 和压缩 KV 流程，不包含 CSA index/TopK。
- mHC：作为 attention block 外的独立 module/算子序列建模。

V4 granular 复用 GEMM、FA、DSA index、DSV4 TopK、ElementWise 和通信组件。Context CP 场景还包括 index/key、compressed KV 的 all-gather。V4 module 仍可作为 SILICON/HYBRID 的 primary；ANALYTICAL 走 granular，以保证无实测数据时能够运行。

### Kimi K3

K3 官方适配已经是 granular operation 图，而不是单一 KDA module。69 个 KDA hybrid
layer 与 24 个 MLA layer分别组合投影、KDA conv/scan/recurrence、MLA、AttnRes、
LatentMoE、shared experts、通信和 DSPARK draft。Analytical 只为 KDA core 增加专用
KernelSim，其余继续复用 GEMM、MLA/BMM、MoE、ElementWise 和通信模型。

需要区分两个责任层：官方 K3 图仍存在 fused KDA onorm ownership、LatentMoE norm
顺序、AttnRes 粗化、双池内存和 PP stage imbalance 风险；Analytical 跟随层仍存在
fused decode/verify route 未完全共享、W4A16/W4A8 MoE proxy 和 backend fidelity 风险。

H100/H200 的 `w4a16_mxfp4` 已可通过 BF16 transfer proxy 跑通，不再属于完全失败；
但该 proxy 没有重新拟合。Blackwell 的 W4A8 仍借 NVFP4 recipe，也不能视为原生模型。

## dtype 与 backend

当前推荐配置为：

| 部分 | 推荐精度 |
|---|---|
| 权重 GEMM | FP8 block |
| MoE 计算 | FP8 block |
| V4 attention math | BF16 |
| V4 KV cache | FP8 |
| DSA/MLA 主体 | 依模型和已校准表，通常 BF16 math |
| MSA index | BF16 |
| Hopper K3 routed MoE | W4A16 MXFP4/BF16 transfer proxy |
| Blackwell K3 routed MoE | W4A8 经 NVFP4 proxy |

V4 generation module 的数据键主要由 KV cache dtype 决定，因此 V4 常用 `BF16 attention math + FP8 KV cache`，不能简单把 FMHA 和 KV cache 都设置成 BF16。

Analytical 默认面向 SGLang。非 SGLang backend 可以进入通用接口，但当前模型的采集边界、融合方式、dtype 和通信语义来自 SGLang，跨 backend 结果必须视为迁移性估算。

W8A16 GEMM/MoE、W4A16 MXFP4 MoE、FP8 KV + BF16 FA、MLA FP8 KV 和 DSA Index BF16
都属于显式 proxy。它们解决无卡执行与基础负载迁移，不代表已新增同 dtype/backend 的
Silicon 数据或完成重新拟合。

## module 与 granular

在 SILICON/HYBRID 中，module 通常优先查询实测 module 数据；缺失时按 operation 实现决定是否进入 granular。Analytical 不应依赖 module 表，而是直接组合子算子。

审计中应区分：

| 标记 | 含义 |
|---|---|
| `module_silicon` | P/D module 都由 silicon 数据解析 |
| `module_estimated` | 仍执行 module 对象，但 CP/组合路径返回 estimated 理论结果 |
| `granular_silicon` | module miss 后，非零 granular 子算子全部命中 silicon |
| `granular_mixed` | granular 中混合 silicon、empirical 和 estimated/KernelSim |

顶层 `SILICON` 不必然代表整条路径纯实测；小算子可能本身使用经验公式，CP module 也可能返回 estimated。比较 silicon 和 analytical 时必须保留这些来源信息。

## 适用边界

- DSV4 Flash TopK K=512 有多硬件表支撑；Pro K=1024 的 v1 profile 主要来自 H100 定向补采，跨硬件可信度有限。
- MSA index、DSA index/TopK 仍属于阶段性模型，存在单硬件拟合和 backend 绑定风险。
- 未覆盖 dtype、不同 SGLang 版本、不同 kernel recipe、非 NVIDIA GPU 和未知 shape 不应直接解释为精确性能。
- `standard` 用于常规排序和趋势；`low/high` 是工程包络，不是统计置信区间。
- 国产系统仅支持 Analytical estimate，架构能力由 YAML override 决定，SM 字段仅作
  KernelSim 微架构代理；当前通信限制在一个 supernode 内。
- `tp_first` 是逻辑 rank placement 的理论估算模式；Rust engine-step 尚未传播 placement，
  因而自动回退 Python。`independent` 保持原有行为。

推荐将 Analytical 用于无实测数据时的硬件/并行配置筛选、瓶颈定位和敏感性分析；正式容量承诺仍应结合 silicon 或专项实机校准。
