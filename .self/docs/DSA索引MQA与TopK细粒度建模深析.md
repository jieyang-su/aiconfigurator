# DSA 索引 MQA 与 TopK 细粒度建模深析

## 1. 问题与结论

本文进一步回答两个问题：

1. DSA indexer 中的 FP8 ragged/paged MQA logits 是否能视为普通 MQA/FA 同类算子，直接使用已有 FA roofline？
2. TopK/index transform 是否只是 norm、RoPE 一类轻量流式算子，可用一次访存量粗略估算？

结论如下：

- **数学上，index MQA 与 attention 的 QK score 阶段同源；执行上，它不是完整 FA。** 它只做 FP8 QK、index-head gate/reduction并物化 FP32 logits，不做 softmax、在线归一化、PV、partial output 或最终输出。因此不能原样调用当前 FA latency 结果。
- **不需要从零建立完全独立的 MQA roofline。** 最合适的工程方案是从现有 FA 模型抽出 tiled-QK、FP8 matrix peak、HBM/L2、CTA/task-wave 和 launch floor 基座，新增 `score_only/index_mqa` execution kind。
- **TopK 可以采用轻量 analytical recipe，但不能退化为 norm/RoPE 式单遍 `bytes/BW`。** 它需要在很长的 FP32 score row 上做选择/归约，并将局部索引转换为 paged/ragged 物理索引；至少应包含有效多遍访存、比较/选择吞吐、输出/metadata 和 launch/wave floor。
- MQA logits 与 TopK 在 SGLang 0.5.12 中是两个独立 kernel，二者之间存在一个完整 FP32 score workspace。除非未来 backend 提供 fused score+selection kernel，否则模型也应保持两个阶段。

## 2. 审查版本

本次以本机镜像为主要源码基线：

```text
image: booleimg.myaddr.io/lmsysorg/sglang:v0.5.12
image id: 79d577610f50
source root: /sgl-workspace/sglang
DeepGEMM: /usr/local/lib/python3.12/dist-packages/deep_gemm
```

SGLang 0.5.12 已把较早代码中的 DSA 路径统一命名为 NSA：

- `sglang/srt/layers/attention/nsa/nsa_indexer.py`
- `sglang/srt/layers/attention/nsa_backend.py`
- DeepGEMM `sm90_fp8_mqa_logits.cuh`
- DeepGEMM `sm90_fp8_paged_mqa_logits.cuh`

命名变化不改变本文讨论的算法对象：它仍是 DeepSeek Sparse Attention 的 index score、TopK 和 sparse attention 流程。

## 3. Index MQA 究竟计算什么

### 3.1 数学语义

对每个 query token、每个 index head 和候选 KV token，先计算：

```text
dot[q, h, k] = Q_index[q,h,:] dot K_index[k,:]
```

然后 DeepGEMM kernel 对每个 head 的 dot 执行激活和 gate 加权，并沿 index-head 归约：

```text
score[q,k] = sum_h relu(dot[q,h,k]) * weight[q,h] * k_scale[k]
```

SGLang 0.5.12 对 DeepSeek V3.2 的典型参数是：

```text
index_n_heads = 64
index_head_dim = 128
Q dtype = FP8 E4M3
K dtype = FP8 E4M3 + FP32 scale
weight dtype = FP32
score output dtype = FP32
```

GLM-5.2 的 index heads 为 32，head dim 同为 128。

主要矩阵工作量可写为：

```text
F_matrix = 2 * valid_pairs * H_index * D_index
```

另有约 `valid_pairs * H_index` 量级的 ReLU、gate multiply 和 head reduction。这里 `valid_pairs` 是所有 query 的有效 `[ks, ke)` 范围之和，而不是简单的 `B*S*max_context`。

### 3.2 Ragged prefill

`deep_gemm.fp8_mqa_logits` 接收：

```text
Q       [Nq, H_index, D_index] FP8
K       [sum(K_request), D_index] FP8
K scale [sum(K_request)] FP32
weights [Nq, H_index] FP32
ks/ke   [Nq] int32
```

它把多个 request 的 K 拼接到同一缓冲区，每个 query 用 `ks/ke` 指定自己的 causal 区间。输出为：

```text
logits [Nq, concatenated_K_width] FP32
```

无效区域可能不清理，后续 TopK 根据 lengths/row_starts 忽略。SGLang 在 `Nq*Kwidth` 较大时按 query row 分 chunk，以限制 FP32 workspace；各 chunk 都会独立执行 MQA 和 TopK。

### 3.3 Paged decode

`deep_gemm.fp8_paged_mqa_logits` 接收 page table、context lengths 和预计算 schedule metadata。典型布局是：

```text
Q         [batch, next_n=1, H_index, D_index]
KV page   page_size=64, one KV head, D=128 + scale storage
weights   [batch, H_index]
logits    [batch, aligned_max_context] FP32
```

DeepGEMM 使用 paged task scheduler，让多个 warpgroup 按 KV block 服务 query；SGLang 0.5.12 已将 schedule metadata 在 forward metadata 初始化阶段预计算并跨层复用。这一点与当前 FA 模型的抽象 `kv_splits` 有相似性，但不是同一个调度公式。

## 4. 它与普通 MQA/FA 的相同点

两者可以共享的底层抽象很多：

- 都沿 Q 和 KV 两个序列轴分 tile；
- 核心 FLOPs 都是 `2*Q*K*head_dim*heads`；
- 都需要从 HBM/L2 读取 Q/K，使用 shared memory 和 WGMMA/Tensor Core；
- decode 都可能沿长 KV 轴拆成多个 task，以填满 SM；
- 都受 SM 数、时钟、shared memory、L2/HBM 带宽、FP8 peak 和 CTA waves 影响；
- ragged/paged 元数据都会影响有效 pair 数和调度利用率；
- 小 batch decode 都存在低并行度、launch floor 和尾波问题。

因此，现有 FA roofline 的以下实现值得直接复用：

```text
HardwareSpec
FP8 matrix peak 选择
tile/padding 与 exact valid score 统计框架
HBM/L2 resource roof
parallel efficiency 与 wave rounding
fixed launch overhead
profile standard/high/low 档位机制
诊断输出结构
```

## 5. 它为什么不能直接调用当前 FA 模型

当前 FA 模型明确假设 Q-outer/KV-inner online-softmax dataflow，并计算：

```text
QK FLOPs
+ softmax/vector FLOPs
+ PV FLOPs
+ online state update
+ final normalize
+ optional split reduction
```

同时其 IO 包含 V、output、LSE 和 partial state。Index MQA 则不同：

| 项目 | 普通 FA | Index MQA logits |
| --- | --- | --- |
| QK | 有 | 有 |
| softmax | 有 | 无 |
| PV/V read | 有 | 无 |
| online max/sum | 有 | 无 |
| output vector | BF16/FP8 attention output | 无 |
| score materialization | 通常不落 HBM | **完整 FP32 logits 落 HBM** |
| index-head gate | 无 | **有** |
| head reduction | attention head独立 | **多个 index head 归约为一个 score** |
| split reduction | 合并 partial O/LSE | 通常各 KV tile直接写不同 score 区间 |
| KV layout | dense/paged K+V 或 latent KV | paged/ragged index K + scale |

直接把 index MQA 填成普通 FA shape 会产生三个方向的结构误差：

1. 错加 PV、softmax、normalize、partial state；
2. 漏掉巨大的 FP32 logits 写流量；
3. 错用 FA 的 `N_task=B*Hkv*query_tiles*kv_splits`。Index MQA 的工作按 query/KV block 调度，index heads 在 CTA 内融合和归约，并不是 64 个独立 attention head task。

所以“同属 attention”不足以证明可直接复用完整模型输出。

## 6. 推荐的 MQA 建模方式

### 6.1 共享基座，而非复制模型

建议将现有 `estimate_attention()` 内部拆出一个共享 QK roofline helper：

```text
estimate_tiled_qk(
    valid_pairs,
    executed_pairs,
    q_heads,
    kv_heads,
    head_dim,
    q_dtype,
    k_dtype,
    k_scale_bytes,
    score_output_bytes,
    scheduling_kind,
)
```

普通 FA 和 index MQA 分别在其上组合：

```text
FA = tiled_qk + online_softmax + tiled_pv + output/reduction
IndexMQA = tiled_qk + gate/head_reduce + FP32_score_store
```

公开接口可使用：

```python
AttentionExecutionKind.FUSED_ATTENTION
AttentionExecutionKind.INDEX_MQA_SCORE
```

这样属于扩展 FA 模型家族，而不是创建互不相关的经验模型。

### 6.2 Index MQA 的资源公式

首版可采用：

```text
T_matrix = F_matrix / (FP8_peak * eta_matrix * eta_parallel)

T_vector = F_gate_reduce / (vector_peak * eta_reduce * eta_parallel)

Bytes_Q = Nq * H_index * D_index * 1
Bytes_K = K_loaded * D_index * 1
Bytes_K_scale = K_loaded * 4
Bytes_gate = Nq * H_index * 4
Bytes_meta = O(Nq + page_table_entries) * 4
Bytes_score = executed_score_slots * 4

T_mem = max(Bytes_HBM/(BW_HBM*eta_hbm),
            Bytes_L2/(BW_L2*eta_l2))

T = launch_floor + max(T_matrix, T_vector, T_mem) + task_service
```

这里必须区分：

- `valid_pairs`：真实计算区间，用于语义 FLOPs；
- `executed_score_slots`：受 tile padding/aligned stride 影响的实际 score 写入；
- `K_loaded`：考虑 query tile 内 K 复用后的读取量；
- ragged chunk 数：每个 chunk 都增加 launch 与尾波；
- paged scheduler task 数：由 `context_lens/page_size/SPLIT_KV` 决定，而不是 FA 的 KV-head task 数。

### 6.3 参数能否沿用 FA standard/high/low

硬件事实和大部分结构参数可以沿用，但 FA 校准效率不能原样套用：

- FP8 matrix peak、HBM/L2 peak、SM/clock 可以共享；
- FA 的 `hbm_efficiency/l2_efficiency` 可作为初始 prior；
- FA 的 `kv_task_cycles` 不能直接复用，因为 DeepGEMM MQA scheduler、head reduction 和 score store 不同；
- fixed overhead 可用同量级 prior，但需要 standalone MQA 数据检查；
- 最终仍建议给 index-MQA 自己的 standard/high/low recipe，参数数量控制在 launch、matrix/memory efficiency 和 task service 四类以内。

## 7. TopK/index transform 到底做什么

SGLang 0.5.12 默认开启 fused TopK：

```text
SGLANG_NSA_FUSE_TOPK = true
```

输入为 MQA 产生的 FP32 score、每行有效长度和可选 row start。输出并不保证是原始 score 的局部 TopK 下标，而是 attention backend 可直接消费的索引表示。

### 7.1 PAGED transform

`fast_topk_transform_fused` 大体承担：

```text
对每个 query 的有效 score span 选 TopK
 -> 处理 padding/无效位置
 -> 根据 cu_seqlens_q 找到所属 request
 -> 通过 page_table_size_1 将逻辑 token 位置映射为物理 KV 位置
 -> 输出固定宽度 [Nq, K] int32 indices
```

### 7.2 RAGGED transform

`fast_topk_transform_ragged_fused` 承担：

```text
TopK selection
 -> 加上该 query 对应的 concatenated-KV offset
 -> 输出 sparse attention 可直接索引的绝对位置
```

### 7.3 Unfused fallback

关闭 fused TopK 时使用 `fast_topk_v2`，只返回 selection 结果，后续 backend 可能还需单独转换。模型必须用 recipe 标识 fused/unfused，不能同时计入 selection 和 transform 两次。

### 7.4 Select-all 快路径

当每个 request 的完整可见长度 `<=K` 时，无需真正排序选择。SGLang 的 K-only indexer 路径构造 dummy logits，TopK kernel 可直接生成：

```text
[0, 1, ..., length-1, -1, ...]
```

这时 MQA logits 整体消失，TopK 也应使用 select-all transform 的线性成本，而不是按一般 TopK 公式。

## 8. TopK 是否只是一个小访存算子

### 8.1 为什么不是 norm/RoPE

Norm/RoPE 对每个元素执行固定次数的读取、算术和写回，通常一两遍流式访问即可描述。TopK 有明显不同：

- 输入 `[Nq,F]` FP32，F 可达 16 万乃至 100 万；
- 需要从 F 个候选中保留 K=2048，存在多轮局部选择和跨 block 合并；
- 比较、交换、heap/radix/partition 等工作量由具体算法决定；
- 输出 `[Nq,2048]` int32，不一定比某些小算子输出小；
- paged transform 还有 page-table gather 和不规则索引；
- 单 query decode 时可并行 row 数很少，不能达到整卡 HBM 峰值；
- SGLang/AIC collector 已观察到不同 score 分布可带来约 3% 到 22% 的时延差异。

所以单纯使用：

```text
T = (4*Nq*F + 4*Nq*K) / HBM_peak
```

只能作为极乐观下界。

### 8.2 可接受的轻量模型

TopK 不必上升到 FA 那样复杂的完整 roofline。建议首版使用一个参数很少的 selection recipe：

```text
Bytes_input  = 4 * executed_score_slots
Bytes_output = 4 * Nq * K
Bytes_meta   = lengths + row_starts + page-table/index-offset traffic

Bytes_effective = pass_factor(F,K,algorithm) * Bytes_input
                + Bytes_output + Bytes_meta

Comparisons = comparison_factor(F,K,algorithm) * valid_score_elements

T_mem = Bytes_effective / (HBM_BW * eta_mem * eta_parallel)
T_cmp = Comparisons / (vector_compare_rate * eta_cmp * eta_parallel)
T = launch_floor + wave_floor + max(T_mem, T_cmp)
```

其中：

- `pass_factor>=1` 表示选择算法对 score 的有效扫描/中间流量；
- `comparison_factor` 不强行写成理论 `F*log(K)`，应按实际 kernel 算法设 prior；
- `eta_parallel` 必须依赖 query rows 和分块数，尤其避免 batch=1 decode 假装跑满 HBM；
- flat/top-last 两种 score 分布可定义 high/low 边界，standard 采用代表分布；
- paged/ragged transform 通过 metadata bytes 和轻量 gather penalty 区分。

这仍然是可解释的理论模型，不需要按 shape 查表。

### 8.3 推荐三档参数含义

为了跨硬件外推，三档应描述执行效率而不是绑定 H100：

| 档位 | 含义 |
| --- | --- |
| low latency | 清晰 winner、较少有效 pass、较高并行效率 |
| standard | 常规 logits 分布和中等 selection/gather 效率 |
| high latency | 大量 tie/近似值、更多 pass、低 query 并行度 |

命名应与现有 analytical 的 low/standard/high 延迟语义一致，避免“high estimate”被误解为高性能。

## 9. MQA 与 TopK 的边界和组合

在 SGLang 0.5.12 当前实现中，组合时延是：

```text
T_index_score_select =
    sum_over_chunks(T_mqa_logits(chunk) + T_topk_transform(chunk))
```

理由是 MQA 物化 FP32 logits，TopK 随后读取该 workspace；大 shape 还按 query row 交替执行多个 chunk。建模时需注意：

- score 写流量计入 MQA，score 读流量计入 TopK，不是重复计算；
- chunk 会重复支付 kernel launch、尾波和部分 metadata 成本；
- schedule metadata 的初始化若在 layer 外复用，不应每层重复计时；
- CUDA Graph 下 Python loop 不属于 kernel 时延，但多次 graph node/kernel launch 仍存在；
- 将来若出现 fused MQA+TopK，recipe 应关闭 score HBM write/read并改为单一 fused operation。

## 10. 推荐实施调整

对原 DSA granular 计划作如下收敛：

1. 不创建完全孤立的 `DSAIndexScore` roofline；先重构 FA 的共享 tiled-QK 基座。
2. 在共享基座上实现 `IndexMQAScore` execution kind，拥有自己的 IO、head reduction 和 scheduler。
3. 保留 `DSATopKSelect` 轻量独立 op，首版采用有效多遍访存+比较+launch/wave 模型。
4. `F<=K` 显式切换到 `IndexKCacheUpdate + SelectAllTransform`，不执行一般 MQA/TopK。
5. producer 层执行 MQA+TopK；GLM-5.2 shared 层只复用 indices，不重复执行这两项。
6. ANALYTICAL 仅使用 standard/high/low recipe；standalone 实测表只用于校准和验证，不在 analytical 查询时读取。

建议的代码关系为：

```text
kernelsim/fa/qk_roofline.py       # 共享 tile/resource/task 基座
kernelsim/fa/model.py             # 完整 FA2/FA3
kernelsim/dsa/index_mqa.py        # score-only 组合与 paged/ragged scheduler
kernelsim/dsa/topk.py             # selection/transform 轻量模型
operations/dsa_granular.py        # SGLang execution recipe 与模型图
```

## 11. 验证重点

- 手算 index MQA FLOPs、Q/K/scale/gate/score bytes；
- ragged/paged 的 valid pairs、aligned score slots 和 chunk 数；
- 验证关闭 softmax、PV、output、partial state 后不残留 FA 成本；
- 验证 index heads 在 CTA 内归约，不进入 FA 的独立 head task 数；
- TopK select-all、一般 fused PAGED、一般 fused RAGGED、unfused 四条分支；
- TopK flat/top-last 分布形成合理上下界；
- `Nq=1` 长 context 不错误使用整卡 HBM 峰值；
- MQA score write 与 TopK score read各计一次；
- chunked 与 unchunked 在工作量上守恒，但 chunked 增加 launch/尾波；
- GLM producer/shared 层不会重复支付 MQA/TopK；
- ANALYTICAL 在空数据库下完整运行且 source 不出现 silicon。

## 12. 最终评价

用户提出“index attention 本质上属于 MQA”是正确的，但复用层级应落在 **QK roofline 基座**，而不是当前 **完整 FA 时延函数**。Index MQA 是一个 QK-only、跨 index-head 融合归约、物化 FP32 score 的 attention-family kernel；对它完全另建黑盒模型没有必要，直接套 FA 又会出现结构性错误。

TopK/index transform 的确可以比 FA 使用更轻的模型，但它不是普通 norm/RoPE。一个包含有效多遍流量、比较吞吐、并行度和启动下限的轻量 recipe 已足够支撑跨硬件粗略评估，也符合当前项目“不追求 backend 绝对复刻、但要求理论边界正确和稳健”的定位。
