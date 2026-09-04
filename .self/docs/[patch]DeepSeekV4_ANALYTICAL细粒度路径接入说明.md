# DeepSeek V4 ANALYTICAL 细粒度路径接入说明

## 实现结果

DeepSeek V4 attention 已从仅有 module operation 的结构扩展为：

```text
FallbackOp(
    primary = 原有 V4 CSA/HCA module,
    fallback = ratio-aware granular operations,
)
```

模式选择为：

- `ANALYTICAL`：始终执行 granular，不查询 V4 attention module 表。
- `SILICON/HYBRID`：保持原 module 查询和既有 fallback 行为。
- `SOL/EMPIRICAL`：保持原 V4 module 理论/数据驱动路径，避免迁移改变旧结果。

为支持这一点，`FallbackOp` 增加了通用的 `primary_excluded_modes`，不改变原有 `silicon_primary_only` 调用者。

## Granular 结构

公共部分复用已有 GEMM 和 ElementWise：

- `q_a`、`wkv`、`q_b` 投影。
- q/kv norm。
- output inverse RoPE。
- grouped `wo_a` 的等价 GEMM 工作量。
- `wo_b` 投影。

各压缩比的差异为：

- ratio 0：真实 SWA core，不再在 ANALYTICAL 中冒充 HCA。
- ratio 4：主 compressor、独立 index compressor、index q/weight projection、DSA Index MQA、DSA TopK、CSA sparse attention。
- ratio 128：主 compressor 和 HCA sparse attention，无 index/TopK。

ratio 0 的 SILICON primary 仍保留历史 ratio 128 module 近似，避免改变现有表驱动结果；wrapper metadata 保留真实 ratio 0，ANALYTICAL fallback 使用真实 SWA。

## 复用和扩展

DSA Index/TopK 增加默认值为 1 的 `context_stride`。V4 CSA 传入 4，使 score 和 TopK 面向 `full_context/4` 的 compressed index cache；原 DSV3.2/GLM 调用不受影响。

当 V4 prefill 的 fresh Q 大于 compressed K 时，Index MQA 将 Q 切成满足旧校准 shape 契约的逻辑块并累加。该处理是对现有 H100 DSA 模型的代理复用，不表示 V4 kernel 已重新拟合。

新增 `DeepSeekV4SparseAttention` 作为轻量 adapter：先精确计算 SWA、CSA/HCA compressed causal pair 数，再转换为等价矩形 MQA workload，交给现有 FA KernelSim。KV 是单份 compressed stream，`kv_heads=1`，不会按 query heads 重复计流量。

FA `AttentionShape` 现在允许非 causal 等价 workload 出现 `Q>K`；普通 causal attention 仍保持 `KV>=Q` 校验。

## Context Parallelism

ANALYTICAL granular 显式恢复了原先 module 内部承担的 CP 通信：

- CSA：index K all-gather + c4 compressed KV all-gather。
- HCA：window-capped KV all-gather + c128 compressed KV all-gather。
- ratio 0：window-capped KV all-gather。

`DeepSeekV4KVAllGather` 只负责 V4 消息量推导，实际通信时延仍委托已有 NCCL 查询和 `AnalyticalConfig.communication_mode`，没有建立新的通信性能模型。

## mHC

`DeepSeekV4MHCModule` 在 `ANALYTICAL` 下直接使用完整无表 SOL 工作量，并返回 `source="analytical"`。公式已有 `sites=2`，一次覆盖 attention 和 FFN 两个 mHC 站点，不再额外拆分或重复计算。

## 当前限制

- 首期以 SGLang FP8 checkpoint 和普通 MoE 为主要可运行配置。
- 官方 FP4 expert/MegaMoE 没有被 FP8 或普通 MoE 静默替代；显式选择未支持组合时保留现有明确失败。
- CSA Index/TopK 沿用有限 H100 数据拟合的 DSA 模型，属于代理模型。
- compressor fusion、online c128、multi-stream overlap 和 fused cache write 未做精确重叠还原。
- 非 SGLang backend 可使用理论路径，但仍受既有 backend compatibility warning 约束。

## 验证

- V4 granular、DSA 回归、V4 module/sparse table、CP、model config、FallbackOp 和 FA KernelSim 共 305 项测试通过。
- 单独的 V4 FP8 SGLang static context/decode 测试确认可在 H100 `ANALYTICAL` 下无 module 查询执行。
- context/generation breakdown 均未出现 `silicon` source。
- `ruff check` 和 `git diff --check` 通过。
