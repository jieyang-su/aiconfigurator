# DSA 细粒度模型迁移实现说明

## 范围

DeepSeek V3.2 和 GLM-5.2 现在通过统一的 `DEEPSEEKV32` 模型族构造 DSA attention。模型层不再直接把 `ContextDSAModule` 或 `GenerationDSAModule` 放进算子序列，而是使用 `FallbackOp` 包装：

```text
SILICON/HYBRID: module primary -> granular fallback
ANALYTICAL/SOL: granular fallback
```

因此 ANALYTICAL 不依赖 DSA module 实测表；它只使用现有 GEMM、ElementWise、NCCL 和三个 DSA KernelSim 原语。

## Granular recipe

每层的顺序为：

```text
q_a_proj, kv_a_proj, q/kv norm, q_b_proj
index K/Q/gate projections, index norm/RoPE/quant
index-K all-gather (CP)
index MQA score, TopK/index transform
latent KV all-gather (CP)
sparse attention, o_proj
```

`DSAIndexScore` 在 context 使用 ragged、generation 使用 paged；`DSATopKSelect` 使用对应布局；`DSASparseAttention` 按 selected pair 数计算稀疏 MLA 主体。context 的 `full_context <= index_topk` 采用 SGLang select-all 近似，score 和 TopK 为零。

V3.2 每层计算 indexer。GLM-5.2 按 SGLang 的共享 index 规则使用精确 `21/78` producer fraction，而不是简单的 `1/4`。

## 迁移注意点

新版 operation 仍被 SDK 测试和 breakdown 代码按 `context_attention`、`generation_attention` 名称识别，并读取 `_gemm_quant_mode`。因此 wrapper 保留旧名称和元数据，同时由 `FallbackOp.get_weights()` 委托 primary/fallback 的权重计算，避免破坏已有量化配置逻辑。

CP 通信现在放在对应消费者之前：index-K all-gather 位于 score 前，latent-KV all-gather 位于 sparse attention 前。其 latency 总和不变，但 breakdown 顺序与实际依赖关系一致。

## 适用性与限制

index MQA/TopK KernelSim 仍是单 H100、SGLang 0.5.12 临时采集基础上的阶段性模型；其警告会保留。稀疏 attention 目前使用 FA profile 的工程效率和解析工作量，尚未有跨硬件实测校准。GLM 共享 index 和 context select-all 是明确的 SGLang 语义假设，其他 backend 不应自动继承。

完整模型 CLI sweep 仍受 SGLang CP 并行配置约束：`cp_size > 1` 时不能同时使用 attention TP/DP。固定合法并行配置的 operation/backend 路径可执行；候选生成器若产生非法组合，应由 sweep 层过滤，而不是由 DSA 模型放宽约束。

## 验证

新增 `tests/unit/sdk/test_dsa_granular_analytical.py` 覆盖模型结构、V3.2/GLM 维度、共享 index 比例、select-all 和 ANALYTICAL 禁止 module 查询。定向 KernelSim、模型、CP、DSA 数据库测试共 `315 passed`。
