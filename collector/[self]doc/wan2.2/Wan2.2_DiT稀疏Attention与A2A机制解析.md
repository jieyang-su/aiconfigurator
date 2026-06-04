# Wan2.2 DiT 稀疏 Attention 与 A2A 机制解析

本文聚焦 SGLang `0.5.10` 官方安装版 Wan DiT 主干中两条非常规 self-attention 路径：`MinimalA2AAttnOp` 与 `UlyssesAttention_VSA`。它们都服务于长视频 token 序列下的 self-attention 加速，但设计目标不同：前者用 Sparse-Linear Attention 替换 dense FA 计算，后者用视频三维 tile 稀疏模式替换 dense FA 计算。

## 代码入口

| 模块 | 关键源码 | Wan 中的位置 | 触发方式 |
|---|---|---|---|
| `MinimalA2AAttnOp` | `runtime/layers/attention/turbo_layer.py` | `WanTransformerBlock.__init__` 的 `self.attn1` | `config.attention_type in ("sla", "sagesla")` |
| `UlyssesAttention_VSA` | `runtime/layers/attention/layer.py` + `backends/video_sparse_attn.py` | `WanTransformerBlock_VSA.__init__` 的 `self.attn1` | 全局 `--attention-backend video_sparse_attn` |
| 普通 `USPAttention` | `runtime/layers/attention/layer.py` | 默认 `WanTransformerBlock.__init__` 的 `self.attn1` | 默认或 FA/SageAttention/SDPA 后端 |

`WanTransformer3DModel.__init__` 会根据全局 server args 选择 block 类：当 `attention_backend.lower() == "video_sparse_attn"` 时使用 `WanTransformerBlock_VSA`，否则使用普通 `WanTransformerBlock`。普通 block 内如果 `attention_type` 是 `sla/sagesla`，self-attention 才进一步替换为 `MinimalA2AAttnOp`。

## 常规 USPAttention 背景

普通 Wan self-attention 的基本输入是 `q/k/v: [B, S_local, H_local, D]`。其中 `D=128`；A14B 全局 heads 为 `40`，TI2V 5B 全局 heads 为 `24`，TP 后 `H_local=model_heads/tp_size`。如果启用 SP，`USPAttention` 可能包含两段通信：

| SP 算法 | 通信前本地形状 | 通信后本地 attention 形状 | collector 处理 |
|---|---|---|---|
| 无 SP | `[B, S, H/tp, D]` | `[B, S, H/tp, D]` | 直接测本地 attention |
| Ulysses | `[B, S/sp, H/tp, D]` | `[B, S, H/tp/sp, D]` | 剥离 all-to-all，只测后者 |
| Ring | `[B, ceil(S/sp), H/tp, D]` | ring 内多轮局部 attention | 用本地 `ceil(S/sp)` shape 近似单卡 compute |

这个原则同样适用于本轮两个模块：collector 不把多卡 All2All/Ring 通信算入单算子 latency，只测通信后的单卡计算核。

## MinimalA2AAttnOp

### 顶层设计

`MinimalA2AAttnOp` 继承自 `DistributedAttention`。`DistributedAttention` 的结构是：

1. 如果没有 context-parallel process group，直接执行 `local_attn(query,key,value,metadata)`。
2. 如果 process group size 大于 1，先对 Q/K/V 做 `_SeqAllToAllQKV`。
3. 在 A2A 后的本地 head/seq 布局上执行 `local_attn`。
4. 对输出再做 `_SeqAllToAll` 还原布局。

在 Wan collector 中我们按单卡算子原则不建立多卡 process group，因此 `MinimalA2AAttnOp` 会落到第 1 条路径：只测真实的本地 SLA/SageSLA attention 实现。多卡 A2A 只通过 `sp_size/sp_algorithm` 影响测试 shape。

### 底层算子

`MinimalA2AAttnOp` 不使用普通 FA，而是根据 `attention_type` 选择：

| attention_type | 后端 | 核心计算 |
|---|---|---|
| `sla` | `SparseLinearAttentionBackend` | Triton mean-pool/block-map + Triton block sparse attention + torch matmul linear attention + `proj_l` |
| `sagesla` | `SageSparseLinearAttentionBackend` | block map + Q/K INT8 量化 + V FP8/FP16 block sparse attention kernel + linear attention + `proj_l` |

SLA 的关键思想是把 dense `S x S` 注意力拆成两部分：

- 稀疏 softmax attention：先按 block 对 Q/K 做平均池化，计算 block 级相似度，按 `topk_ratio` 选出每个 Q block 需要看的 K blocks，再用自定义 Triton kernel 只算这些 blocks。
- 线性 attention 补偿：对 Q/K 应用 feature map，再用 `K^T V` 与归一化项构造低成本全局分量。

SageSLA 在 H100/Blackwell 上进一步使用量化稀疏 attention kernel。源码中会按 CUDA arch 分支：`sm90` 使用一套 block size 与 FP8 kernel，其他新架构走另一路 FP8/threshold kernel。若未安装 `spas_sage_attn`，`sagesla` 会在实例化时失败；collector 默认只生成 `sla`，需要设置 `COLLECTOR_WAN_ENABLE_SAGESLA=1` 才纳入 SageSLA。

### Wan 推理触发条件

| 条件 | 行为 |
|---|---|
| `config.attention_type == "original"` | 不触发，使用 `USPAttention` |
| `config.attention_type == "sla"` | 使用 `MinimalA2AAttnOp` + `SLA_ATTN` |
| `config.attention_type == "sagesla"` | 使用 `MinimalA2AAttnOp` + `SAGE_SLA_ATTN` |
| 同时设置 `--attention-backend video_sparse_attn` | 优先切到 `WanTransformerBlock_VSA`，不走普通 block 的 `MinimalA2AAttnOp` |

## UlyssesAttention_VSA

### 顶层设计

`UlyssesAttention_VSA` 继承自 `UlyssesAttention`，但 forward 明确要求不支持 replicated text tokens。Wan VSA block 只把它用于 DiT self-attention；cross-attention 会把 sparse backends 过滤掉，继续走非稀疏 backend。

VSA forward 的主要流程：

1. 输入 `q/k/v/gate_compress: [B, S_local, H, D]`。
2. 把四个张量在 batch 维拼接成 `qkvg`。
3. 通过 `sequence_model_parallel_all_to_all_4D(scatter_dim=2,gather_dim=1)` 执行 Ulysses 风格重分布。
4. 调用 `VideoSparseAttentionImpl.preprocess_qkv`：按三维视频 tile 重新排列并 padding。
5. 调用 `VideoSparseAttentionImpl.forward`：执行外部 `vsa.video_sparse_attn` kernel。
6. 调用 `postprocess_output` untile，再 all-to-all 还原布局。

collector 中用单卡 `SingleRankGroup` 替代 SP group，因此 all-to-all 是恒等操作，保留真实 `preprocess_qkv -> video_sparse_attn -> postprocess_output` 计算链路。

### 底层算子

`VideoSparseAttentionImpl` 使用视频三维结构感知的稀疏注意力：

| 阶段 | 含义 |
|---|---|
| tile partition | 将 DiT token grid `(T, H, W)` 按固定 `VSA_TILE_SIZE=(4,4,4)` 切成 3D tiles |
| padding/index | 对最后不完整 tile 做 padding，并维护 `non_pad_index` 与反向 index |
| topk 推导 | `topk = ceil((1 - VSA_sparsity) * total_seq_length / 64)` |
| kernel | 调用外部 `vsa.video_sparse_attn(query,key,value,variable_block_sizes,topk,block_size,compress_attn_weight)` |
| gate_compress | 来自 `WanTransformerBlock_VSA.to_gate_compress`，作为压缩注意力权重输入 |

这条路径依赖外部 `vsa` 包。SGLang CUDA platform 在选择 `VIDEO_SPARSE_ATTN` 时会尝试 import `vsa.block_sparse_attn`；backend forward 时还要求 `vsa.video_sparse_attn` 存在。没有该包时，collector 不做 torch fallback，而是让任务显式失败，以避免“测到了错误 kernel”。

### Wan 推理触发条件

| 条件 | 行为 |
|---|---|
| `--attention-backend video_sparse_attn` | `WanTransformer3DModel` 构造 `WanTransformerBlock_VSA` |
| `attention_backend_config.VSA_sparsity` | denoising 阶段构建 `VideoSparseAttentionMetadata`，控制 VSA topk |
| SP > 1 | VSA 使用 Ulysses all-to-all 重分布；collector 只测通信后的单卡 compute |
| cross-attention | sparse backend 被过滤，VSA 不用于 text/image cross-attention |

## Collector 对照

本轮新增 `wan_sparse_attention`，输出 `wan_sparse_attention_perf.txt`。

| collector 行 | 对应模块 | 是否测通信 | 核心 kernel/source | 关键字段 |
|---|---|---|---|---|
| `op_variant=minimal_a2a_sla, backend=sla` | `MinimalA2AAttnOp` + `SparseLinearAttentionImpl` | 否 | `sglang_minimal_a2a_sla` | `seq_len,num_heads,tp_size,sp_size,sla_topk` |
| `op_variant=minimal_a2a_sla, backend=sagesla` | `MinimalA2AAttnOp` + `SageSparseLinearAttentionImpl` | 否 | `sglang_minimal_a2a_sagesla` | 同上；需 `COLLECTOR_WAN_ENABLE_SAGESLA=1` |
| `op_variant=ulysses_vsa, backend=video_sparse_attn` | `UlyssesAttention_VSA` + `VideoSparseAttentionImpl` | 否 | `sglang_ulysses_video_sparse_attn` | `raw_latent_t/h/w,patch_t/h/w,vsa_sparsity` |

执行示例：

```bash
docker run --gpus '"device=7"' --ipc=host --rm \
  -v /home/ai_lab/ljc/scale-up-sim/aiconfigurator:/workspace \
  -w /workspace 10.110.181.132:5000/sglang:0.5.10.post1 \
  python collector/collect.py --backend sglang --ops wan_sparse_attention --num-processes 1 --limit 8
```

可选环境变量：

| 变量 | 默认 | 作用 |
|---|---:|---|
| `COLLECTOR_WAN_ENABLE_VSA` | `1` | 是否生成 VSA 测试用例 |
| `COLLECTOR_WAN_VSA_SPARSITIES` | `0.0,0.3,0.5` | VSA sparsity 列表 |
| `COLLECTOR_WAN_ENABLE_SAGESLA` | `0` | 是否生成 SageSLA 测试用例 |
| `COLLECTOR_WAN_MAX_SPARSE_ATTN_OUTPUT_ELEMS` | `80000000` | 防止长序列大 head case OOM |

## Hopper 与 Blackwell 注意点

| 模块 | Hopper sm90 | Blackwell sm100 | 兼容性处理 |
|---|---|---|---|
| SLA | Triton sparse kernel，head_dim `64/128` 支持 | 理论上走 Triton 编译；需对应 Triton/CUDA wheel 支持 | collector 不指定 arch 分支，复用 SGLang backend |
| SageSLA | 源码对 `sm90` 有专门 FP8 kernel 分支 | 非 `sm90` 走另一路 FP8/threshold kernel，依赖 `spas_sage_attn` 支持 | 默认不启用，显式 env 开关避免环境缺包导致大面积失败 |
| VSA | 依赖 `vsa` 外部 CUDA 扩展 | 依赖 `vsa` 是否提供 Blackwell wheel/源码编译支持 | 不 fallback；缺包或不支持时显式报错 |
| 普通 FA 对照 | SGLang 平台默认 H100 用 FA3 | Blackwell 默认设置 FA4 | 本文新增 collector 不测普通 FA，普通路径仍由 `wan_attention` 覆盖 |

核心原则：如果 SGLang 真实推理需要外部稀疏 kernel，而目标服务器没有对应 kernel，collector 应失败并暴露环境问题；不能退回 PyTorch SDPA/torch matmul 后继续写性能数据。
