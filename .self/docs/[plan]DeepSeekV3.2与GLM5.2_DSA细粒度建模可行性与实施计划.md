# DeepSeek V3.2 与 GLM-5.2 DSA 细粒度建模可行性与实施计划

## 1. 目的与结论

本文评估将 AIC 中 DeepSeek V3.2、GLM-5.2 的粗粒度 DSA attention module 重建为 granular 逐算子图，并在无实测表时复用 KernelSim/ANALYTICAL 的可行性。

结论是：**可行且值得实施，但不能把源码中的逻辑步骤机械地逐项相加。** 推荐建立两层表示：

1. **DSA 语义图**：表达投影、索引生成/共享、稀疏选择、attention、prefix、CP 和跨层依赖；它应尽量与硬件和 kernel 实现解耦。
2. **后端执行 recipe**：表达某个 SGLang 版本和 attention backend 如何融合、并行或省略语义步骤，防止重复计算 fused kernel，允许未来加入 vLLM、TRT-LLM 和非 NVIDIA 实现。

现有 KernelSim 可直接承担 projection GEMM、部分 BMM、普通小算子和经扩展后的 MLA 主体；FP8 ragged/paged MQA index score 与 TopK/index transform 必须增加 DSA 专用模型。跨层 IndexShare 是模型图调度语义，不应做成一个 kernel 效率系数。

首版建议只实现和验证 **SGLang + NVIDIA CUDA 的 DSA granular recipe**，保留 module 作为 SILICON/HYBRID 的优先实测路径；ANALYTICAL 则始终走 granular，保证完全无表可运行。

## 2. 审查基线与资料

本次结论基于以下版本和官方资料（审查日期：2026-08-06）：

- AIC 工作树：`0cb74b6d55f2c1b8d42e546059e9ccfe8657d4d8`
- SGLang：`579b359a18b9ed655d803727baa473ec445816fe`
- [DeepSeek-V3.2-Exp 官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp)
- [DeepSeek-V3.2-Exp 官方配置](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/config.json)
- [DeepSeek-V3.2-Exp 官方 inference 源码](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/tree/main/inference)
- [DeepSeek Sparse Attention 技术报告](https://arxiv.org/abs/2512.02556)
- [GLM-5.2 官方模型卡](https://huggingface.co/zai-org/GLM-5.2)
- [GLM-5.2 官方配置](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json)
- [IndexShare 技术报告](https://arxiv.org/abs/2603.12201)
- SGLang `deepseek_v2.py`、`forward_mha.py`、`dsa_indexer.py`、`dsa_backend.py`
- AIC `sdk/models/deepseek_v32.py`、`sdk/operations/dsa.py`
- AIC `collector/sglang/collect_mla_module.py` 和 `glm5_dsa_sparse_modules.py`

论文/模型卡负责确认算法意图和公开结构；官方 config 负责确定实际 checkpoint 参数；SGLang 是本设计判断当前推理边界、融合和 dtype 的主要准绳；AIC collector 只能说明当前数据库采集了什么，不能反向定义真实执行语义。

## 3. 两款模型的结构差异

| 参数 | DeepSeek V3.2-Exp | GLM-5.2 | 建模影响 |
| --- | ---: | ---: | --- |
| hidden size | 7168 | 6144 | projection GEMM 形状不同 |
| layers | 61 | 78 | GLM 的 IndexShare 必须按真实层 pattern 聚合 |
| attention heads | 128 | 64 | Q/attention/BMM 并行度不同 |
| Q LoRA rank | 1536 | 2048 | Q-B 与 indexer Q projection 不同 |
| KV LoRA rank | 512 | 512 | latent KV 主维相同 |
| QK no-PE/rope dim | 128/64 | 192/64 | QK 维分别为 192/256 |
| V head dim | 128 | 256 | PV/BMM/output 形状不同 |
| index heads/dim | 64/128 | 32/128 | index score 和 cache 流量不同 |
| index TopK | 2048 | 2048 | 稀疏 attention 饱和点相同 |
| index sharing | 每层生成 | 4 层一组共享 | producer/reuse 是两种不同 layer graph |
| producer 层数 | 61/61 | 21/78 | GLM 不能简单使用 `1/4` |
| max context | 163840 | 1048576 | GLM index score/cache 长上下文压力更大 |
| 原生 dtype | BF16 权重中带 FP8 block quant config | BF16 checkpoint | projection dtype 必须逐 checkpoint 解析 |

GLM-5.2 的 `index_topk_freq=4`、`index_skip_topk_offset=3` 使前 3 层和后续周期 producer 合计为 21 层，57 层复用上一 producer 的 TopK。AIC 当前已正确推导 `21/78=0.2692`，但 module 内以 full/skip 时延加权；granular 图应更直接地生成两种 layer recipe，并按真实 layer pattern 或准确计数聚合。

## 4. DSA 算法的建模抽象

### 4.1 为什么 DSA 不等于普通 FA/MLA

普通 dense attention 对每个 query 访问所有可见 KV，prefill 工作量随 causal pair 数增长，decode 随 context 线性增长。DSA 在 full context 超过 2048 后增加索引器：

1. 生成低维 index query/key 和每头权重；
2. 用 FP8 MQA score 计算候选相关度；
3. 每个 query 选出最多 2048 个 KV 位置；
4. 稀疏 attention core 只访问选中位置。

因此其主成本由三组不同规律构成：

- projection：随 fresh token 数线性变化；
- index score：约随 `fresh * full_context` 变化，且执行 ragged/paged MQA；
- sparse attention：随 `fresh * min(visible_context, topk)` 变化，超过 TopK 后饱和。

当 `full_context <= topk` 时，SGLang 有 K-only 快路径：保存 index K，并直接生成连续位置，不执行昂贵的 logits；这不是把 logits FLOPs 变小，而是整个 kernel 分支消失。

### 4.2 Prefix 下的精确 pair 数

设 batch 为 `B`，fresh query 长度为 `S`，prefix 为 `P`，`F=P+S`，TopK 为 `K`。稀疏 attention 的有效 KV pair 数应为：

```text
pairs = B * sum(i=1..S, min(P+i, K))
```

即：

```text
F <= K: pairs = B * ((F*(F+1) - P*(P+1)) / 2)
P >= K: pairs = B * S * K
otherwise:
  pairs = B * ((K*(K+1) - P*(P+1)) / 2 + (F-K)*K)
```

这一公式已存在于 AIC module SOL，迁移时应成为 `DSASparseAttention` 的统一语义，而不是 module 特有逻辑。

## 5. SGLang 的真实执行序列

### 5.1 公共 projection 与 MLA 数据路径

SGLang `forward_mha.py` 的正常 prefill 路径可抽象为：

```text
hidden
 -> fused_qkv_a_proj_with_mqa
 -> split(q_lora, kv_lora, k_rope)
 -> q_a_rmsnorm
 -> q_b_proj
 -> indexer(hidden, unquantized q_lora)
 -> kv_a_rmsnorm
 -> RoPE(q_pe, k_pe)
 -> kv_b_proj
 -> split K_nope/V + concat K_rope (+ optional cast/cache preparation)
 -> sparse attention backend
 -> o_proj
```

几个重要边界：

- indexer 需要未量化的 `q_lora`；即使 Q-B projection 使用 FP8 fused norm/quant，也可能同时产出 BF16 q_lora。
- backend 可通过 `prepare_prefill_qkv` 接管 RoPE、BF16 到 FP8 转换和 cache write；不能再额外添加独立 cast/cache op。
- `kv_b_proj` 在部分 MXFP4/FP8 路径可与 split、concat 和 cast 融合。
- DSA 的 indexer RoPE 与 MLA RoPE 布局要求不同。DeepSeek 官方曾专门修复 indexer non-interleaved 与 MLA interleaved 的差异，granular graph 必须保留两个语义标签。

### 5.2 Index producer 路径

完整 indexer 在 CUDA 路径执行：

```text
hidden --weights_proj---------------------> per-index-head gate
q_lora --wq_b-----------------------------> index Q
hidden/qkv_a fused output --index K proj--> index K
index Q/K norm + indexer RoPE
index Q FP8 quant + scale
index K FP8 quant + scale + paged cache write
FP8 ragged/paged MQA logits
TopK + causal/index offset transform
optional TP/CP topk broadcast
```

decode/paged 和 prefill/ragged 使用不同 kernel/metadata。小于等于 TopK 的普通 prefill走 K-only 快路径；decode 仍需要从历史 cache 中选择位置。CP prefill 又会按 local query chunk 构造 full-context K 范围。

SGLang 在小 token decode/CUDA graph 条件下可能使用 alternate stream：gate projection 与 index Q/K/cache 工作有并发，随后 event 同步。故生产模型需要 `OverlapOp` 或 fused recipe，而不是无条件求和。

### 5.3 Index reuse 路径（GLM-5.2）

共享层接收前一 producer 层产生的 `prev_topk_indices`：

```text
producer layer: projections + index producer + sparse attention -> topk_indices
shared layer:   projections + reuse(topk_indices) + sparse attention
```

reuse 不是“所有 indexer 工作乘 0.25”：

- producer 层仍承担完整 index score/TopK/cache；
- shared 层不执行本层 index score 与 TopK；
- shared 层自身的 MLA projections、KV cache 与 sparse attention 仍执行；
- index tensor 存在跨层生命周期和潜在 broadcast/copy；
- MTP 还受 `index_share_for_mtp_iteration` 影响；
- SGLang TBO 路径对 topk indices 的传递存在单独限制，首版不应默认与普通路径等价。

因此建议在模型层生成 producer/shared layer pattern，而不是只给一个平均 `full_frac`。仅在最终聚合结果中才可用 `21*T_producer + 57*T_shared` 加速批量计算。

### 5.4 Sparse attention backend

SGLang DSA backend 支持 `flashmla_sparse`、`flashmla_kv`、`fa3`、`tilelang`、`trtllm` 等变体。它们会改变：

- 是否直接消费 sparse indices；
- head padding 和 tile/split 策略；
- KV cache 存储 dtype 与 attention compute dtype；
- prefill/decode 的 kernel 选择；
- projection absorption 和 BMM 是否显式存在；
- metadata/schedule 初始化边界。

因此“attn_core 都使用 FA”只能作为算法层类比，不能直接调用普通 FA2/FA3 模型。首版应新增 sparse MLA shape，并以现有 MLA roofline 为底座扩展 selected-KV、index gather 和 task 数，而不是冒充 dense FA。

## 6. AIC 当前实现与缺口

### 6.1 当前模型层

`DeepSeekV32Model` 在 context/generation 都只放置一个 `ContextDSAModule` 或 `GenerationDSAModule`。MoE 已经相对 granular：shared expert GEMM、router、dispatch、MoE core、combine 和 decode overlap 分开表示。因此本轮主要重建 attention，不需要重写普通 MoE 图。

### 6.2 当前 DSA operation

`dsa.py` 声明 module 包含：

```text
kv_a projection
Q norm/Q-B projection
indexer wq_b/weights projection
FP8 MQA logits
TopK
sparse MLA
BMM pre/post
o_proj
```

其 SOL 已分别计算 GEMM、FP8 index logits、sparse attention FLOPs，但最终仍是：

```text
sol_math = gemm/peak_gemm + index/peak_fp8 + sparse/peak_attn
sol_mem  = aggregate_bytes / HBM_BW
T        = max(sol_math, sol_mem)
```

这一聚合存在四个问题：

1. 串行 kernel 的启动、低 occupancy 和中间张量成本被一个全局 roofline 隐藏；
2. 多组 weight/cache 流量被合并后，无法分别应用已校准的 GEMM/MLA/BMM 模型；
3. fusion、dual-stream overlap 和分支省略无法准确表达；
4. DSA query 没有 ANALYTICAL 专用分支，不能满足“无实采数据”的稳定语义。

### 6.3 当前 collector

`collect_mla_module.py` 直接调用完整 `self_attn`，module 数据天然包含实际 backend 的融合、cast、缓存、metadata 和同步行为，适合 SILICON 查表，却不能提供逐子项 attribution。

AIC 已额外采集部分 standalone kernel：

- `deep_gemm.fp8_mqa_logits`
- `fast_topk_transform_fused` / `fast_topk_v2`
- `flash_mla_sparse_fwd`

这证明 DSA 核心可拆，也为后续验证专用模型提供数据；但 standalone 微基准边界不必然等于整模块中的热缓存、stream 和 metadata 边界，不能简单相加后要求与 module 完全相等。

## 7. 建议的 granular 语义图

### 7.1 Operation 结构

建议新增组合 operation，而不是把所有步骤直接铺在 model 文件中：

```text
ContextDSAGranular
GenerationDSAGranular
  |- DSAProjectionGroup
  |- DSAIndexerProducer | DSAIndexReuse
  |- DSASparseAttention
  |- DSAOutputProjection
```

基础原子 operation：

- `DSAIndexScore`：FP8 ragged/paged MQA score；
- `DSATopKSelect`：causal TopK、padding 和 index transform；
- `DSASparseAttention`：selected-KV sparse MLA core；
- `DSAIndexCacheUpdate`：仅在 recipe 未融合时显式计时；
- `DSAIndexReuse`：默认零计算，但保留 tensor lifetime/broadcast 语义；
- `DSASmallOp`：明确的 norm/RoPE/quant/concat/cache-write fallback。

### 7.2 逐阶段映射

| 逻辑阶段 | 主要 shape/工作量 | 可复用 AIC/KernelSim | 新模型 | 关键 caveat |
| --- | --- | --- | --- | --- |
| fused Q/KV-A | `[B*S,H] x [H,q_lora+kv_lora+rope(+indexK)]` | GEMM | 否 | collector 声明和 SGLang fused 输出必须对齐 |
| Q RMSNorm | `B*S*q_lora` | ElementWise/SOL | 否 | 可能与 FP8 quant 融合 |
| Q-B | `[B*S,q_lora] x [q_lora,Hq*dq]` | GEMM KernelSim | 否 | TP local heads；混合 quant exclusion |
| KV RMSNorm | `B*S*kv_lora` | ElementWise/SOL | 否 | 可能与 quant 融合 |
| KV-B | `[B*S,kv_lora] x [kv_lora,Hq*(dnope+dv)]` | GEMM KernelSim | 否 | 部分 backend/phase 采用 absorption，不执行该 dense 展开 |
| index Q (`wq_b`) | `[B*S,q_lora] x [q_lora,Hi*di]` | GEMM KernelSim | 否 | producer 层才执行 |
| index gate | `[B*S,H] x [H,Hi]` | GEMM KernelSim | 否 | 可与 scale/softmax 融合 |
| index K proj/norm/RoPE | token-linear | GEMM + small op | 轻量 recipe | indexer RoPE 布局独立 |
| index Q/K quant+cache | token-linear bytes | ElementWise/SOL | 轻量 recipe | 常与 K store 融合；FP8 scale 字节需计入 |
| index score | `2*B*S*F*Hi*di`（ragged 需精确 pairs） | 不可直接复用普通 GEMM | **是** | paged/ragged、FP8、causal bounds、启动/occupancy |
| TopK/index transform | 每 query 对可见 F 选 K | 无 | **是** | 不是 GEMM；算法/分段/饱和/metadata 主导 |
| index reuse | 跨层 TopK tensor | model graph | 结构 op | producer/shared pattern，可能 broadcast |
| sparse attention | `pairs` 上 QK/PV + gather | 扩展 MLA roofline | **扩展** | K 上限、随机 gather、backend task/split |
| BMM pre/post | absorption shape | BMM KernelSim | 否 | 只在真实 recipe 显式执行时加入 |
| concat/RoPE/cast | token-linear bytes | ElementWise/MLAConcatK | 否 | 融合时必须关闭独立项 |
| o_proj | `[B*S,Hq*dv] x [Hq*dv,H]` | GEMM KernelSim | 否 | 通常最大 projection，quant 可能与其他投影不同 |

### 7.3 Shape 语义必须显式保留

组合 op 的输入至少包括：

```text
phase, batch, fresh_seq, prefix_seq
hidden_size, local_attention_heads
q_lora_rank, kv_lora_rank
qk_nope_dim, qk_rope_dim, value_dim
index_heads, index_head_dim, index_topk
is_index_producer
tp_size, cp_size
projection dtypes, kv-cache dtype, attention compute dtype
backend_recipe
```

不能只使用 `(batch, full_seq, heads, dtype)`，否则无法区分相同 full length 下 fresh/prefix 的投影和 causal pair 数，也无法表达 GLM shared 层。

## 8. Backend recipe 设计

建议定义显式 recipe capability，而非根据 GPU 名称猜测：

```python
DSAExecutionRecipe(
    backend="sglang",
    implementation="flashmla_sparse",
    phase="decode",
    q_norm_quant_fused=True,
    kv_projection_mode="absorbed",
    index_k_quant_store_fused=True,
    index_gate_overlap_with_qk=True,
    sparse_attention_kind="flashmla_sparse",
)
```

recipe 决定语义 stage 如何映射到计时节点：

- `serial`：逐项相加；
- `fused`：只调用一个融合模型或将小项并入其 bytes/FLOPs；
- `overlap`：用 `max(branch_a, branch_b)+sync`；
- `elided`：当前分支根本不执行；
- `fallback`：缺专用模型时使用可解释的 SOL 并标记低可信度。

这使架构语义和 kernel 版本解耦。将来 SGLang 从 FlashMLA 换到 TRT-LLM backend 时，只增加 recipe，不需要重写 DeepSeek/GLM model graph。

## 9. Prefill、Decode 与 CP 的差异

### 9.1 Prefill

- `S` 可大于 1，必须使用精确 causal pair 数；
- `F<=K` 时 index logits/TopK 可走 K-only 快路径；
- prefix cache 下 projection 只处理 fresh token，index score 和 attention 访问 full context；
- FP8 prefill backend 可能融合 Q/K RoPE、quant 和 cache write；
- CP 的 index MQA 使用 full-context K 范围，不能把所有序列轴都除以 CP。

### 9.2 Decode

- 通常 `S=1`，projection GEMM 极小，启动和 weight read 占比高；
- index score 访问 paged index-K cache，随 context 增长；
- sparse core 在 `F>=K` 后 attention KV 长度固定为 2048，但 index score 仍随 F 增长；
- alternate stream、CUDA graph 和 metadata 对实际时延影响更突出；
- 使用普通 dense MLA decode 模型会错误地让 attention core 随 full context 持续增长。

### 9.3 Context Parallel

现有 AIC 文档给出的执行抽象是：

```text
AG_KV -> full-context MQA/index score -> TopK -> AG_LSE -> sparse FMHA
```

granular 迁移需保留以下不变量：

- projection/token 工作按 local fresh tokens 计算；
- index score 的 K 轴需要 full context；
- sparse attention 每 query 最多读取 K=2048；
- TopK 的复杂度和输出不能按 `1/cp` 线性缩放；
- `AG_KV`、`AG_LSE` 是明确通信节点；
- shared-index 层是否仍需 topk broadcast 应按 backend recipe 决定。

## 10. Module 与 granular 的模式选择

建议统一选择规则：

| Database mode | DSA 路径 |
| --- | --- |
| SILICON | 精确 module 表优先；缺 shape/backend/dtype 时 granular fallback |
| HYBRID | module 实测优先；缺失项 granular，且逐项保留 source |
| ANALYTICAL | **强制 granular**，禁止读取 module 或 standalone 实测表 |
| SOL | granular 的无校准理论下界/工程估算 |
| EMPIRICAL | 按新版数据驱动语义处理；不可作为无表保证 |

若用户显式要求 `force_granular`，SILICON 也应逐项查询数据/公式，用于 decomposition 验证。ANALYTICAL 的测试必须断言所有 DSA 子项 source 均非 `silicon`，防止重现旧版本的隐式查表问题。

## 11. MoE 部分的迁移判断

DeepSeek V3.2 与 GLM-5.2 的 MoE 主骨架相对稳定，AIC 已拆为：

```text
shared gate/up GEMM -> activation -> shared down GEMM
router GEMM -> dispatch -> routed MoE -> combine
```

decode 还使用 `OverlapOp` 表达 shared/routed 分支重叠。无需因 DSA granular 化重建 MoE，但必须保留以下模型特异配置：

- router 通常保持 BF16；
- checkpoint quant exclusion 可能令 attention projection、shared expert 与 routed expert 使用不同 dtype；
- GLM-5.2 NVFP4 checkpoint 的 shared experts 可保持 BF16；
- WideEP/DeepEP 与普通 MoE 仍是不同执行边界，不能复用同一 analytical MoE 结果冒充；
- CP/attention-DP 改变进入 dispatch 的 token 分布，应继续由模型层和通信 op 处理。

因此本项目第一阶段只需确保 attention 拆解没有破坏现有 MoE op 次序、overlap 和 dtype 传播。

## 12. 实施阶段

### 阶段 0：语义快照与边界锁定

- 固定 AIC、SGLang、官方 config 和 collector commit/hash；
- 为 V3.2 和 GLM-5.2 导出 prefill/decode 的 SGLang kernel trace；
- 分别覆盖 `F<K`、`F=K`、`F>K`，以及 GLM producer/shared 层；
- 标记每个 projection、fusion、cache write、index score、TopK 和 sparse attention 的真实边界；
- 明确首版支持的 `flashmla_sparse`/`trtllm` backend 之一，不混合校准数据。

### 阶段 1：构建纯语义 granular 图

- 新建 `ContextDSAGranular`、`GenerationDSAGranular`；
- 将现有 module SOL 的 FLOPs/bytes 公式迁移到独立 stage helper；
- 实现准确的 fresh/prefix/pairs、local heads、producer/shared pattern；
- 先以串行 SOL 运行，确保任何硬件无表可执行；
- model 层从 config 传递真实维度和逐 projection dtype，不使用 architecture 常量兜底覆盖有效配置。

### 阶段 2：接入已有 KernelSim

- projection 全部接 GEMM standard/low/high；
- 可见的 absorption BMM 接 BMM model；
- sparse core 从 MLA roofline 派生 selected-KV 模型；
- norm/RoPE/quant/cache 先使用显式 bytes + launch floor 的小算子模型；
- ANALYTICAL 强制走 granular，并增加 source 审计。

### 阶段 3：新增 DSA 专用模型

- `DSAIndexScore`：区分 ragged prefill 与 paged decode，建模 FP8 score、cache bytes、CTA/task waves 和启动项；
- `DSATopKSelect`：按 query 数、可见长度和 K 建模，允许算法分段；
- `DSASparseAttention`：建模 selected index gather、TopK 饱和、head/task parallelism；
- 使用 standalone collector 做校准，但以 full module decomposition 做外部一致性验证；
- 不增加 shape 查表系数来掩盖结构错误。

### 阶段 4：后端 fusion/overlap recipe

- 按 SGLang trace 合并 fused norm/quant、split/concat/cache；
- decode alternate stream 用 overlap 表达并加入必要同步；
- 分别验证 producer/shared 两条路径；
- 再扩展其他 DSA backend，不通过硬件名称自动推断。

### 阶段 5：模型级集成

- V3.2 生成全 producer layer 图；
- GLM-5.2 按 `indexer_types` 或 offset/frequency 生成 21 producer + 57 shared；
- MTP、TBO、CP 分别作为显式 execution feature；
- SILICON/HYBRID 保留 module primary；ANALYTICAL/SOL 使用 granular；
- 输出 attention 子项 breakdown，支持端到端差异归因。

## 13. 验证计划

### 13.1 公式与 shape 单测

- V3.2/GLM 所有官方维度快照；
- projection 的 M/N/K、TP local head 和 dtype 快照；
- prefix causal sparse pairs 在 TopK 前、交界和饱和后的手算测试；
- index score full-context 轴与 sparse core selected-KV 轴分离；
- index FP8 data/scale cache bytes；
- output、KV cache、index cache 和临时 TopK bytes；
- 小于 TopK 的 K-only 分支不计算 score/TopK kernel。

### 13.2 模型结构单测

- V3.2 61 层全部 producer；
- GLM-5.2 精确产生 21 producer、57 shared，不使用简单 `78/4`；
- shared 层仍包含 projection、sparse attention 和 output，不包含本层 score/TopK；
- `index_topk_pattern` 优先于 frequency/offset；
- MTP index sharing 开关单独验证；
- TBO 未支持时明确报错或降级，不静默套普通路径。

### 13.3 模式和 source 单测

- ANALYTICAL 在空数据目录可执行；
- ANALYTICAL 每个 DSA stage 均无 silicon source；
- standard/low/high 结果有序且可区分；
- SILICON module 精确命中；缺失时逐 query granular fallback；
- force-granular 的 breakdown source 与 query mode 一致。

### 13.4 与实机/collector 的验证

- standalone index score、TopK、sparse attention 分项误差；
- granular 总和与 module trace 的差值，并归因于 fusion/overlap/metadata，而非只报告 MAPE；
- producer 与 shared layer 分开验证；
- prefill/decode、短/长 context、batch/TP/CP sweep；
- 检查 TopK=2048 附近是否连续但有正确斜率变化；
- 以未参与校准的 shape 和另一块同代 GPU 验证外推。

## 14. 风险、假设与边界

### 高风险

- **fusion 重复计时**：逻辑图与 backend kernel 不一一对应，必须由 recipe 消解；
- **TopK 不可辨识**：其复杂度受实现、数据分布和 metadata 影响，纯 FLOPs 不够；
- **sparse gather 效率**：选中 KV 的离散访存不能直接套 dense MLA eta；
- **跨层共享生命周期**：PP、TP/CP、MTP、TBO 可能改变 indices 的存储和通信；
- **dtype 名实不符**：KV 存储 FP8 不代表 sparse attention 一定用 FP8 compute。

### 中风险

- standalone collector 热缓存/stream 边界与模块执行不一致；
- context 与 decode 使用不同 DeepGEMM/FlashMLA kernel；
- 小 token projection 和 small op 的 launch floor 影响总时延；
- `kv_b_proj`/BMM absorption 是否存在取决于 backend recipe。

### 当前假设

- 首版以固定 SGLang commit 的正常非 TBO 路径为准；
- DSA TopK 为 2048，但接口保留可配置值；
- GLM-5.2 的 `indexer_types`/pattern 优先于近似频率；
- module collector 的边界用于总体校验，不作为 granular 公式定义；
- MoE 沿用现有 granular 模型，不在本轮重建；
- 未经 trace 证明的融合和 overlap 默认关闭，并在结果中降低可信度。

## 15. 最终判断

将 DSA module 拆回 granular 并不是对模型结构做无意义的展开，而是 ANALYTICAL 模式获得可解释性、无表能力和跨硬件外推的必要条件。DeepSeek V3.2 与 GLM-5.2 可以共享同一套 DSA 语义 operation；二者差异应来自官方 config、producer/shared layer graph 和 checkpoint dtype，而不是复制两个私有模型。

最关键的工程原则是：**共享语义图，显式后端 recipe，专门建模 index score/TopK/sparse core，禁止用 module 数据或普通 FA 模型悄悄填补缺口。** 按上述阶段推进后，已有 GEMM/BMM/MLA/小算子 KernelSim 可以覆盖大部分线性投影和数据变换，新建模型只集中在 DSA 真正新增的稀疏索引机制上，工作量和长期维护成本均可控。
