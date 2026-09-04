# DeepSeek V4 ANALYTICAL 细粒度建模可行性与实施计划

## 1. 目标与结论

本文是 DeepSeek V4 接入 AIC `ANALYTICAL`/granular 路径的前期设计，不是本轮生产代码变更。目标是在没有 V4 silicon 表时，依据模型结构、SGLang 0.5.14 的真实执行路径和已有 KernelSim 基元，给出可解释的算子序列与理论时延。

初步结论：V4 的主体可以复用现有 GEMM、FA/MLA、DSA index MQA、TopK、ElementWise、MoE 和通信模型，但不能把 V4 attention 直接当成一个已有 MLA module，也不能把 CSA/HCA 的稀疏工作统一为普通 FA。需要新增一个 V4 granular attention 组合器，以及少量 V4-specific 的边界和工作量规则；是否新增拟合 kernel 模型，应由纯 GPU kernel 采集结果决定。

首期建议支持：SGLang backend、BF16 activation/cache、FP8 GEMM（SGLang recipe）、普通 MoE 和显式通信 dtype。FP4 expert、MegaMoE/DeepEP、跨后端迁移和精确 stream overlap 不纳入首期承诺。V4 目前有 module 表，但 module 表只能作为 silicon 路径和校验基准，不能替代无数据 analytical 的 granular 语义。

## 2. 依据与版本口径

本分析使用以下可复核来源：

- DeepSeek 官方 Hugging Face：`deepseek-ai/DeepSeek-V4-Pro`，commit `b5968e9190ef611bbf34a7229255be88a0e937c1`，包括 `config.json`、`inference/model.py` 和技术报告链接。
- DeepSeek V4 Technical Report：[arXiv:2606.19348](https://arxiv.org/abs/2606.19348)。报告描述 CSA、HCA、mHC 以及百万 token 场景的压缩目标。
- 本地 SGLang 0.5.14 source audit：`.self/analytical-mode/dsa-model-pareto/moe_analysis/source_audit/sglang-0.5.14/`。
- AIC 当前实现：`aic-core/src/aiconfigurator_core/sdk/models/deepseek_v4.py`、`sdk/operations/dsv4.py`，以及 `collector/sglang/collect_dsv4_attn.py`、`deepseekv4_sparse_modules.py`、`collect_mhc_module.py`。

版本边界必须写入结果 provenance。V4 kernel 名称、压缩 page layout、FP4/FP8 量化和多 stream 策略随 SGLang/sgl-kernel 版本变化，不能仅以模型名称推断。

## 3. 模型和层级结构

### 3.1 代表配置

V4 Pro 当前配置的关键维度是：61 layers、hidden size 7168、128 query heads、1 KV head、head dim 512、q LoRA rank 1536、o LoRA rank 1024、o groups 16、index heads 64、index head dim 128、index top-k 1024、window 128、`hc_mult=4`、Sinkhorn 20 次、384 routed experts、top-6、expert intermediate 3072。

V4 Flash 为较小变体，约 43 layers、hidden 4096、64 query heads、q/o rank 1024、8 output groups、index top-k 512、256 experts、expert intermediate 2048。实际值必须始终从 model config 读取，不能把 Pro 参数写成全局常量。

### 3.2 单层逻辑序列

实际模型是两个 mHC 站点包围的 attention 和 MoE，而不是简单的 residual + attention + FFN：

```text
mHC pre(attention)
 -> attention norm
 -> V4 attention (SWA / CSA / HCA)
 -> mHC post(attention)
 -> mHC pre(FFN)
 -> FFN norm
 -> shared FFN / router / dispatch
 -> routed MoE
 -> combine
 -> mHC post(FFN)
```

AIC 当前模型还显式建了 shared gate-up、shared activation、shared FFN2、router、MoEDispatch、MoE、post-MoEDispatch。值得注意的是，AIC operation 列表表面上只有一组 `mhc_pre/mhc_post`，但其查询公式使用 `sites=2`，一次聚合 attention 与 FFN 两个站点；因此 granular 迁移时应选择“保留聚合 operation”或“拆成两个站点”之一，不能二者叠加。MegaMoE 是另一个 module 边界，要求 SGLang、`moe_tp=1`、`moe_ep>1`，不能在首期自动混入普通 MoE 结果。

压缩比是层属性，不是运行时任意参数：

- `0`：纯 sliding-window attention（当前 AIC 为复用 HCA module 的临时近似，granular 不应沿用）。
- `4`：CSA，压缩 KV、indexer score、top-k 选择，再对选中的 compressed KV 做稀疏 attention。
- `128`：HCA，重压缩/窗口化 KV attention，没有 CSA indexer/top-k。

## 4. 精确执行边界和算子拆解

### 4.1 公共 attention 投影

按 SGLang `DeepseekV4Attention` 的实际张量流，attention 内部至少包括：

```text
q_a = Wq_a(x)                         [tokens, hidden] -> [tokens, q_lora]
q_a = RMSNorm(q_a)
q = Wq_b(q_a)                         -> [tokens, local_heads, head_dim]
kv = Wkv(x)                           -> [tokens, head_dim]
kv = RMSNorm(kv)
wo_a / output absorption              -> [tokens, local_groups, o_lora_rank]
wo_b                                -> [tokens, hidden]
```

其中 q_b、wkv、wo_a/wo_b 的 TP 切分和 dtype 需从模型 config 与 AIC parallel config 推导。qkv 投影、compressor 和 cache store 的融合不能假设为多个可独立重叠的 kernel；granular 结果应保留逻辑项，并通过 `source`/备注标明它们可能由 fused kernel 承担。

### 4.2 CSA (`compress_ratio=4`)

建议 analytical 序列如下：

```text
q_a GEMM + q_norm
q_b GEMM
wkv GEMM + kv_norm
CSA compressor / rope / compressed-KV cache update
indexer q_b GEMM
indexer weight projection GEMM
indexer compressed-KV score (DSA index-MQA 代理)
index score transform / ReLU-weight reduction
top-k transform (DSA TopK 代理，topk=min(config.index_topk, compressed_len))
CSA sparse attention core (FA/MLA family, selected compressed KV)
output rope / absorption
wo_a GEMM
wo_b GEMM
```

官方参考代码的 index score 语义是：`q` 与压缩 KV cache 做多头点积，经过 ReLU 后按 learned weight 加权求和，必要时做 all-reduce，再 mask 和 top-k。它与 DSA index-MQA 的核心计算同类，但 V4 的 `index_n_heads/index_head_dim/index_topk`、压缩 cache、FP8/FP4 路径和 page layout 不应未经验证直接等同。

CSA 的 attention pair 数不能按 dense full KV 计算。对 context 应使用 causal 的 fresh/prefix 有效 pair；compressed pair 近似为 `min(index_topk, floor(kv_len/4))`，decode 为每个请求的对应 compressed cache 数和 top-k 上限。KV traffic 是 compressed stream 单份读取，不能乘 query head 数。

### 4.3 HCA (`compress_ratio=128`)

```text
q_a/q_norm/q_b
wkv/kv_norm
c128 compressor / rope / cache update
HCA compressed attention core (FA/MLA family)
output rope / absorption
wo_a/wo_b
```

HCA 没有 CSA 的 index-MQA 和 TopK。其 attention 是窗口/重压缩语义，attention pairs 由 `compressed_len` 和窗口上限决定。SGLang 还提供 online compression 和 overlap compression；`c4` 有 overlap，`c128` 可配置 online path。这些是 collector/kernel schedule 属性，不能在基础 FLOPs 公式中假定一定存在。

### 4.4 ratio 0 / SWA

ratio 0 是真实的 sliding-window attention，不应使用 HCA 的 module 表伪装。首期可用普通 FA/MLA 的 window-limited pairs 和 MQA KV bytes 表达：

```text
普通投影 + RMSNorm/RoPE/cache transform
FA/MLA attention(window_size=128, full KV head=1)
输出投影
```

如果 silicon 需要与现有表对齐，可以保留当前 ratio 0 -> HCA 的兼容行为，但必须标记为 `silicon_approximation`，禁止让 analytical 静默采用该替代。

## 5. mHC 语义与建模

SGLang 的 mHC pre 不是一个普通 elementwise：先对 `[tokens, hc_mult, hidden]` 做 norm-aware mixing GEMM，产生 `(2+hc_mult)*hc_mult` 个 mixing 值，再执行 sigmoid/scale/base、Sinkhorn 行列归一化（默认 20 iterations），输出 pre/post/comb。随后 post 对当前输出和 residual 做 `post * x + comb * residual` 的混合。

可复用边界：

- mixing projection：GEMM（但 `K=hc_mult*hidden`、`N=(2+hc_mult)*hc_mult`，属于小 N、低算术强度）。
- norm、sigmoid、scale、Sinkhorn：ElementWise/小 kernel recipe，建议统一按 token、`hc_mult`、iteration 计 memory + launch，不创造 Sinkhorn 专用拟合模型。
- post residual mix：ElementWise/BMM 风格的小算子，先用 launch-aware memory 模型。

当前 `DeepSeekV4MHCModule` 的 module 表可以继续用于 silicon；其 SOL 公式明确用 `sites=2` 聚合两个 mHC 站点。接入 granular 时仍要核对 collector 的 pre/post 边界是否完整包含 mixing、Sinkhorn 和 residual，并保证“两个站点的聚合 module”与“分站点子项”只取一种表达。

## 6. 组件复用矩阵

| V4 逻辑 | 首选复用 | 是否建议新采集/模型 |
| --- | --- | --- |
| q_a/q_b/wkv/wo_a/wo_b、router、shared FFN | GEMM | 仅补齐 V4 特殊 FP8 recipe 或小 N shape |
| q/kv/output norm、RoPE、cache transform | ElementWise/已有 cache 公式 | 无需首期新拟合 |
| CSA indexer score | DSA index-MQA | 先做语义/shape 兼容检查；V4 可单独校准 |
| CSA top-k | DSA TopK | top-k 值必须来自 V4 config；不可复用 2048 的耦合参数 |
| CSA sparse attention | FA/MLA | 需要支持 compressed KV、selected pairs、head padding |
| HCA sparse attention | FA/MLA | window/compressed pair 规则不同，必要时新增轻量 adapter |
| mHC mixing/Sinkhorn/post | GEMM + ElementWise | 首期不新建拟合模型；module 表做 silicon 基准 |
| 普通 MoE dispatch/combine/expert | 现有 MoE/通信 | 默认普通 MoE；MegaMoE 单独隔离 |
| MegaMoE / DeepEP | 现有 WideEP 数据或 module | 不纳入首期 analytical，避免口径混淆 |

复用时必须区分“同类数学工作量”和“同一 kernel 实现”。DSA index 模型的启动项、chunk、head padding 和 FP8 量化参数来自有限 H100 采集，不能直接宣称适用于 V4 全部硬件；若只是代理，结果 source 应写为 `analytical:dsa_proxy`。

## 7. 当前 AIC 缺口

1. `DeepSeekV4Model._attention_ops()` 当前以 module operation 为主，且将 ratio 0 归入 ratio 128；这会阻断真正的 granular analytical。
2. `dsv4.py` 已有完整 SOL 公式和 sparse sidecar 查询，但它是 module/CP 辅助模型，不是由 GEMM、index、TopK、FA 逐项组成的 analytical graph。
3. 已有 `dsv4_*_module_perf` 和 `mhc_module_perf` 是 silicon 数据，不能作为无表 analytical 的隐式 fallback。
4. CSA TopK correction 是对 module 表的校正机制。granular 模式中若已单独计 TopK，必须禁止再次扣除 delta，避免 double count。
5. DSV4 attention 在 TP>1 时为 FlashMLA 对 head 数做 64/128 padding 后再 slice。AIC 需要一个显式 `attention_head_padding` 参数/规则：默认不指定，在 Hopper/Blackwell + SGLang 时可指定 64 或 128；不能把 64 当作普适硬件常数。
6. module/collector 可能包含 multi-stream overlap、fused cache write、online compression；逐项相加只代表逻辑工作量，不保证等于 module wall time。
7. V4 官方模型是 FP4 expert + FP8 mixed；AIC 当前 analytical 的 FP8 GEMM 和普通 MoE 不能自动代表 FP4 DeepGEMM/MegaMoE。

## 8. 实施阶段

### P0：语义和边界

- 固定 SGLang 版本、模型 config、compress ratio 层计数和 attention dtype。
- 为 context/generation 建立 V4 granular operation builder，至少覆盖 ratio 0/4/128。
- 将 `prefix`、`compressed_len`、`index_topk`、`window_size` 作为显式查询参数。
- 增加 operation-level source 和 boundary 标记，保证 analytical 不查询 silicon module 表。

### P1：复用现有模型

- 将所有线性投影接入 GEMM KernelSim，严格按 TP/local heads/groups 计算。
- CSA indexer 先复用 DSA index-MQA；增加 V4 形状快照与 head-padding 规则测试。
- CSA TopK 先复用 DSA TopK；对 Pro topk=1024、Flash topk=512 单独验证。
- CSA/HCA attention 接入 FA/MLA，使用实际 selected/windowed pair 数和单份 compressed KV storage。
- mHC 用 GEMM + launch-aware ElementWise 组合；不叠加 module 表。

### P2：针对性采集

优先级为：

1. V4 CSA index score，context/decode、BF16/FP8、不同 batch/past/compressed length。
2. V4 CSA TopK，512/1024 两个 top-k，区分 context/decode。
3. V4 CSA/HCA sparse attention，head padding、page size、selected pairs 和 cache dtype。
4. mHC pre/post，tokens、hidden、hc_mult、Sinkhorn iterations，并记录 TileLang/DeepGEMM/torch backend。
5. fused compressor/cache write，仅当 P1 结果对 module 差异超过目标误差时采集。

采集必须遵循 AIC collect 约定：纯 GPU kernel 计时、预热、CUDA event、独立 shape、清晰 boundary；不得把 host 调度、CUDA Graph capture、初始化或跨 stream 等待混入单算子数据。module 数据仅用于端到端校验。

### P3：模型接入

- 扩展 `AnalyticalConfig` 的 V4 attention 参数，保持默认值兼容旧模型。
- 通过 operations 层组合查询，不修改进程级共享 database。
- 非 SGLang backend 允许运行但警告一次；SGLang-specific index/cache assumptions 不向 vLLM/TRT-LLM 自动传播。
- unsupported FP4/MegaMoE 明确报错或走已声明的粗略 fallback，不伪装成 V4 analytical 精确支持。

## 9. 验证与验收标准

- 结构：Pro/Flash 的层数、ratio 分布、head/group/rank、top-k 和 MoE 参数均从 config 正确推导。
- 边界：ratio 0 不含 index/topK；ratio 4 含 index/topK/CSA；ratio 128 不含 index/topK；mHC 不重复计 module。
- 数值：手算 snapshot 覆盖 FLOPs、KV bytes、compressed pairs、index pairs、top-k 和 head padding。
- 模式：ANALYTICAL 无任何 silicon data 目录仍能执行；SILICON module 可正常查表；两者 source 标识不混淆。
- 误差：在已有 module silicon 点上比较总 attention、CSA、HCA、mHC、MoE 和通信；先分别报告，不用端到端抵消掩盖局部误差。
- 回归：Llama、Qwen3、DSV3.2、GLM5.2、M3 的既有 analytical/silicon/empirical/SOL 路径不改变。

## 10. 风险与适用性警告

- V4 kernel 代码和模型仓库均处于快速演进状态，本文以 SGLang 0.5.14 和上述 HF commit 为准。
- DSA index 代理不能覆盖 V4 indexer 的所有 FP8/FP4、page、head padding 和 multi-stream 特性；首期结果是有依据的粗略理论值，不是 silicon 替代。
- c4 的 overlap compressor、c128 online compressor、fused norm/rope/cache 写入会使逻辑 granular sum 与 module wall time 不相等。需要用模块校验而不是强行要求逐 kernel 之和相等。
- mHC 的 Sinkhorn 20 次和小矩阵 GEMM 对启动/融合极敏感，roofline 只给下界；没有多硬件采集前不提供跨 GPU 精确参数。
- FP4 expert、MegaMoE/DeepEP、后端非 SGLang、未验证的 attention head padding 和百万 token 外推均应标记 unsupported 或 low-confidence。

## 11. 推荐决策

先实施 P0/P1，以已有模型完成可运行的 V4 analytical；同时保留 module silicon 作为对照。只有在 CSA/HCA 或 mHC 的误差成为端到端主导项时，才进入 P2 新采集和拟合。这样能复用现有组件，避免为每个 V4 fused kernel 建立过多脆弱类，也能清楚区分“数学同类代理”和“真实后端 kernel”。
