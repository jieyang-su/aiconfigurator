# MiniMax-M3 MSA 细粒度 ANALYTICAL 建模可行性与实施计划

## 1. 目标与结论

本文评估在 AIC 中为 `MiniMaxAI/MiniMax-M3` 建立完全不依赖实测查表的 ANALYTICAL 路径，并给出后续实施顺序。范围限定为文本 decoder；视觉塔、训练路径和 DeepEP 特化暂不纳入首版。

结论如下：

1. **细粒度实现可行，但当前 AIC 模型拓扑必须先修正。** MiniMax-M3 的 60 层并非全部为 MSA+MoE：前 3 层是 dense attention+dense FFN，后 57 层才是 block-sparse MSA+MoE。当前 AIC 将 60 层统一近似为 MSA+MoE，会同时污染 attention、FFN、MoE 和通信计数。
2. **投影、dense GQA、输出投影、dense/shared FFN、router、MoE core 和 EP 通信可复用现有组件。** 无需重新建立一套 MiniMax 专属 GEMM/MoE 模型。
3. **MSA index 不能直接套用 DSA 的 Index MQA 和 TopK 参数。** DSA 对 token 打分并选择 2048 个 token；MSA 对 128-token block 打分并选择 16 个 block，还包含 block 内 max reduction、index-head reduction及可融合的 page-table transform。两者形状、并行粒度和 kernel regime 均不同。
4. **selected-block 主 attention 可复用 FA/AttentionShape 的计算与流量语义，但不能直接冒充连续 dense FA。** 它是 64Q/4KV 的 GQA，访问离散 block/page，decode 还使用 split-K 和 partial merge。应建立薄的 MSA adapter，复用 FA roofline 主体并补充有效 pair、间接寻址、block gather 和 task-wave 语义。
5. **首版新增模型应控制在三个核心边界：MSA block index score、MSA block TopK、MSA selected-block GQA。** QK norm、partial RoPE、cache store、metadata 等先用已有 ElementWise/SOL recipe；仅在 module 级验证显示其不可忽略时再专项采集。

ANALYTICAL 始终走 granular；现有粗粒度 `ContextMSAModule/GenerationMSAModule` 可保留给 SOL/HYBRID 的兼容路径，但不得继续作为 ANALYTICAL 的 DSA-util transfer 实现。

## 2. 审查基线与证据边界

本次审查交叉使用以下本地代码：

- AIC：`sdk/models/minimax_m3.py`、`sdk/operations/msa.py`、`sdk/kernelsim/analytical.py`、模型 config 和 collector cases；
- 既有 DSA granular：`sdk/models/deepseek_v32.py`、`sdk/operations/dsa.py` 及 DSA KernelSim；
- SGLang 0.5.14 源码快照：`minimax_sparse_ops/`、`minimax_decode_topk.py`、`minimax_store_kv_index.py`、`minimax_qknorm_rope.py`；
- 本机较新 vLLM MiniMax-M3 实现：`vllm/models/minimax_m3/`，用于补足完整 model class、layer pattern、融合投影和 router 语义。

本轮联网访问因临时 DNS 解析失败，未能重新核对远端 SGLang HEAD；本机 `sglang-business:0.5.9` 又早于 MiniMax-M3 支持。因此下述 recipe 以本地 0.5.14 快照和较新 vLLM 实现为主要执行证据，正式编码前应固定目标 SGLang commit，并再次确认其完整 MiniMax-M3 model class。论文/config 用于确认算法和 checkpoint 参数，SGLang 执行源码才用于定义 kernel 边界。

## 3. 模型真实拓扑与当前 AIC 偏差

### 3.1 真实层结构

关键配置为：

| 项目 | MiniMax-M3 |
| --- | ---: |
| decoder layers | 60 |
| hidden size | 6144 |
| attention heads / KV heads | 64 / 4 |
| QK/V head dim | 128 / 128 |
| rotary dim | 64 |
| dense layers | 0-2，共 3 层 |
| sparse MoE layers | 3-59，共 57 层 |
| dense FFN intermediate | 12288 |
| routed/shared intermediate | 3072 / 3072 |
| experts / TopK experts | 128 / 4 |
| index heads / dim | 4 / 128 |
| sparse block / TopK blocks | 128 tokens / 16 blocks |
| selected-token ceiling | 2048 |
| local blocks | 1 |
| score reduction | block 内 `max` |

真实 layer graph 应是：

```text
layer 0..2:
  norm -> dense GQA -> o_proj -> norm
       -> dense gate_up -> SwiGLU-OAI -> down_proj

layer 3..59:
  norm -> MSA index + selected-block GQA -> o_proj -> norm
       -> shared expert || router/dispatch/routed MoE/combine
```

首版仍可沿用 AIC 当前的 MTP scale 表达，但必须明确 MTP module 是否复用相同稀疏路径；在未核对前不能默认把全部 layer latency 机械乘到 speculative token 数。

### 3.2 当前 AIC 问题

当前 `MiniMaxM3Model`：

- 将 `ContextMSAModule/GenerationMSAModule` 按 60 层计数；
- 将 shared expert、router、dispatch、MoE、combine 也按 60 层计数；
- 没有前三层 dense FFN；
- `MSAModule` 没有 silicon 表，SILICON 明确失败；
- HYBRID/EMPIRICAL 使用 DSA 的测得 utilization 乘人工比例；
- SOL 把所有 projection、index、attention FLOPs 和总 bytes 聚合为一次 `max(math, mem)`。

最后一项不是逐 kernel 模型：它会隐藏串行启动、低 occupancy、TopK regime、间接 cache 访问和 split-K merge，也无法给 ANALYTICAL 提供可信 breakdown。collector cases 当前也明确只有 base-op coverage，没有 MSA module collector，因而不能把 DSA utilization transfer 视为经过 MiniMax 实测验证的模型。

配置抽取还应补齐或保留：`first_k_dense_replace`、`dense_intermediate_size`、逐层 `sparse_attention_freq`、`moe_layer_freq`、index value 开关、`init_blocks/local_blocks`、score type、router sigmoid/correction bias/renormalize/routed scale。若 checkpoint 未提供逐层数组，可在 MiniMax-M3 的已确认默认值上生成，但不得在通用 parser 中默认为所有层 sparse/MoE。

## 4. MSA 的实际 granular 语义

### 4.1 Sparse layer 公共前处理

较新实现使用一个融合线性投影产生：

```text
[main Q | main K | main V | index Q | index K]
```

随后融合或相邻 kernel 完成：

```text
per-head Gemma RMSNorm
+ main Q/K partial NeoX RoPE
+ index Q/K partial RoPE
+ main K/V cache store
+ index-K side-cache store
```

SGLang 快照中 `store_kv_index` 可在一次 launch 写 main K/V 和 index K；`minimax_qknorm_rope_grouped` 则融合 main/index 各组 QK norm 与 RoPE。AIC 语义层应保留这些逻辑量，但 backend recipe 必须避免将融合 kernel 重复计时。

TP 下 Q heads 通常切分；KV heads 在 `TP>4` 时不能简单整数除尽后继续缩小，可能复制 KV heads 或采用特定 layout。index heads 也需依据框架实际 TP mapping 处理，不能统一使用 `max(1, heads//tp)` 掩盖复制流量。

### 4.2 Prefill

```text
fused projection
 -> norm/RoPE/cache store
 -> index attention: 对可见 KV token 形成 block score
 -> 每 block 按 score_type=max 聚合
 -> 每 index head 选 Top-16 blocks
 -> 必要时在 index heads/KV heads间 reduce/union
 -> selected-block GQA attention
 -> o_proj
```

prefill 使用 ragged metadata；prefix 场景必须使用 fresh query 与 full KV 的实际 causal 可见范围。`full_len <= 2048` 时可能进入 full-attention/K-only 快路径，此时昂贵的 index score/TopK 应按实际后端分支省略，而不是把其 FLOPs 缩小后仍计一次 launch。

MSA score 的基本工作不是 `Q×num_blocks` 的一个 128 维 dot。每个 query/index head 仍需与可见 index keys 形成 token score，随后按 128-token block 做 max reduction，最终 TopK 的候选轴才是 block。建模必须分别记录 token pairs、有效 block 数和 TopK blocks，防止低估 index score。

### 4.3 Decode

```text
fused projection + cache append
 -> paged index score over historical index-K cache
 -> block max/reduction
 -> Top-16 block selection
 -> [optional] fused block-id -> paged page-table transform
 -> selected-page GQA attention
 -> split-K partial merge
 -> o_proj
```

SGLang 可在 FA3/TRT-LLM dense-paged main backend 下让 indexer 直接输出 page table 和有效 KV 长度；另一条路径输出 block IDs，再执行 sparse Triton/MSA main kernel。两条 recipe 的 TopK 边界和 attention cache layout 不同，不能只靠一个 `attention_algorithm` 标签模糊处理。

decode selected-block kernel 会按 batch、KV head 和 TopK chunk 建 task，低 batch 时通过 split-K 提升并发，并产生 partial output/LSE merge。这一部分可沿用 FA3 task-service 和 rounded-wave 思路，但 task 数应由 MSA 的 block chunks 推导，而不是 DSA latent attention 的 task 公式。

## 5. 组件复用矩阵

| 子阶段 | 复用结论 | 首版实现 |
| --- | --- | --- |
| 前 3 层 dense QKV/O | 直接复用 | `GEMM` + 普通 GQA `Attention` |
| 前 3 层 dense FFN | 直接复用 | gate/up GEMM、activation、down GEMM |
| sparse fused Q/K/V/index projection | 复用 GEMM 模型 | 一个 fused GEMM shape；若目标 SGLang 实际拆分则按其 recipe 拆分 |
| Gemma QK norm + partial RoPE | 复用访存模型 | `ElementWise`/轻量 fused recipe |
| main/index cache store | 复用访存模型 | 按真实 dtype 和 K-only/index-K bytes 计量 |
| MSA block index score | **需新模型** | `MSABlockIndexScore`，区分 ragged/paged |
| block max + TopK + transform | **需新模型** | `MSABlockTopK`，区分 plain/fused page-table variant |
| index-head reduce/union | 部分复用 | 首版 memory/launch recipe；超出 1% module 时延再专项模型 |
| selected-block main GQA | 部分复用 FA | `MSASelectedBlockAttention` adapter，补 block gather/task/merge |
| o_proj | 直接复用 | `GEMM` |
| shared expert | 直接复用 | 现有 GEMM+activation 路径 |
| router | 计算主体复用 | FP32/BF16 router GEMM；sigmoid/bias/topk/renorm 作为小 recipe |
| routed MoE core | 直接复用 | 现有 `MoE` KernelSim，保留适用性警告 |
| dispatch/combine | 直接复用 | 现有通信模式及显式 dtype |
| decode shared/routed overlap | 直接复用 | `OverlapOp`，但需对齐目标框架真实 stream 行为 |

已有 DSA Index MQA/TopK 模型仅可作为 API、三档参数和测试组织方式的模板。其 H100 拟合参数与 `TopK=2048 token`、DSA paged/ragged kernel 强耦合，不得作为 MSA 默认参数。

## 6. 新增采集与建模计划

### 6.1 P0：MSA block index score

临时 collector 必须直接调用生产 backend 的纯 kernel 路径，并遵守 AIC collect 习惯：warmup、CUDA event、预分配、固定 metadata、计时区间内无 host 初始化/同步、记录 kernel source 和完整 shape。至少覆盖：

- prefill ragged：batch、fresh、prefix、index heads、head dim、block size；
- decode paged：batch、context、MTP query length、page/block size；
- BF16 与生产支持的 FP8 index-cache/compute 组合；
- `full<=2048` 快路径边界及其上下邻域；
- TP 后 local/replicated index-head 语义。

特征至少保留 token score pairs、candidate blocks、query blocks、task count、CTA waves、cache bytes。候选模型比较固定开销+task service、roofline max/sum 和 wave-rounded 版本；不能仅凭 H100 小样本删去 FLOPs/bytes 后宣称跨硬件有效。

### 6.2 P0：MSA block TopK

分别采集：

- prefill block TopK；
- decode plain block TopK；
- decode fused TopK + page-table transform；
- 可选 index-head reduce/union。

shape 轴为 `queries × index_heads × candidate_blocks`，TopK 是 blocks（默认 16），不是 selected tokens（2048）。必须覆盖不同 TopK blocks，避免模型与 16 强耦合；至少包含 8/16/32/64。若生产 kernel 只支持固定 16，则模型接口仍保留参数，并对外推发 warning。

### 6.3 P0：selected-block GQA

以现有 FA roofline 的非对称 Q/KV heads、QK/V dim、KV cache dtype 和 task-service 为底座，新增：

- 每 query 实际 selected block/token pair；
- 4 KV heads及 TP 后复制/切分规则；
- block/page table 与间接地址读取；
- decode TopK chunk split-K、partial output/LSE 和 merge；
- prefill causal block 边界与最后不完整 block；
- full-attention 快路径。

先用临时 H100 module/Nsight 验证 adapter；若连续 FA 估算在长上下文 decode 或小 batch 下系统误差超过 20%，再进行 standalone selected-block kernel 专项拟合。

### 6.4 P1：module 归因验证

新增实验性 sparse-layer module collector，不正式并入生产 collector。边界应固定为 projection 至 o_proj，并同时输出 CUDA-event module latency 与 Nsight kernel trace。用途是：

1. 检查 granular kernel 是否缺项或重复；
2. 量化 fusion、双 stream overlap 和 metadata 开销；
3. 比较逐项和与 module，而不是用 module 总时延直接拟合每个子模型；
4. 验证前三层 dense recipe 与后 57 层 sparse recipe 的 layer pattern。

## 7. AIC 实施方案

### 阶段 A：修正模型配置与 layer graph

1. 扩展 MiniMax config 解析，保留 dense/sparse/MoE layer pattern 和 MSA 参数。
2. 将 60 层拆为 `3*dense + 57*sparse`，分别构造 context/generation ops。
3. dense 层使用普通 GQA+FFN；sparse 层才加入 MSA+MoE+EP 通信。
4. 对 TP 下 Q/KV/index heads 添加显式校验和 replicated-head 语义，禁止静默 `max(1, heads//tp)`。
5. 保留 text-only、非 DeepEP 的当前范围，并对 vision/MTP 未验证路径明确 warning。

### 阶段 B：建立 MSA granular operations

建议仅新增组合层和三个必要 primitive：

```text
ContextMSAGranular / GenerationMSAGranular
  |- existing GEMM / ElementWise
  |- MSABlockIndexScore
  |- MSABlockTopK
  |- MSASelectedBlockAttention
  `- existing GEMM(o_proj)
```

operation 负责 shape/phase/backend 语义，KernelSim adapter 负责硬件 profile 和 low/standard/high 参数。不要为 norm、RoPE、cache store、router recipe 各自新增 MiniMax 专属 class。

### 阶段 C：接入 ANALYTICAL 与模式分派

- ANALYTICAL：始终使用 granular，所有核心项返回 `source="analytical"`；
- SOL/EMPIRICAL：保留现有粗粒度 MSA 公式以兼容历史结果，但标明 cross-op transfer；
- SILICON：当前没有 MSA 表，应明确报缺表，不伪装成 silicon；
- HYBRID：可按现有策略回退，但 provenance 必须能区分 DSA transfer、SOL 和真实数据；
- 后续若加入 MSA module 表，仍应逐 query 尝试 module，不得改变 ANALYTICAL 的无表语义。

### 阶段 D：参数固化

完成采集后按旧 KernelSim 规范输出 `low/standard/high` 三档。三档代表跨硬件无数据时的工程不确定性，不应由同一 H100 拟合协方差机械生成。归档必须记录 GPU、SGLang commit、kernel source、dtype、shape 域、collector hash 和不支持范围。

## 8. 测试与验收

### 8.1 语义与 shape 测试

- 精确验证 3 个 dense layer、57 个 sparse/MoE layer；
- dense FFN intermediate=12288，shared/routed intermediate=3072；
- index TopK=16 blocks 与 selected ceiling=2048 tokens 不混用；
- prefill 使用 fresh/prefix/full 的 causal block pair；decode 使用 paged context；
- `full<=2048` 快路径正确省略 index score/TopK；
- TP 下 64Q/4KV/4 index heads 的切分或复制符合目标 SGLang；
- BF16/FP8 的 main KV、index-K、dispatch/combine dtype 独立传播。

### 8.2 数值快照

对 prefill、prefix prefill、decode、MTP decode 各选代表 shape，手算并快照：

- projection GEMM M/N/K；
- token score pairs、candidate blocks、selected blocks/tokens；
- main/index cache bytes；
- GQA QK/PV FLOPs和 output bytes；
- split-K tasks、partial/merge bytes；
- MoE token/expert workload与通信 bytes。

### 8.3 模式和端到端验收

- 无任何 perf parquet 时 ANALYTICAL 可跑通 MiniMax-M3 context/generation；
- low/standard/high 可区分且参数隔离；
- SILICON 缺 MSA 表时 provenance/错误明确；
- 旧模型和旧 database modes 回归不变；
- 最小聚合与 PD 分离 sweep 可运行；
- 与临时 H100 module collector 比较 attention layer 总时延及逐 kernel attribution。

首阶段建议目标：三个 P0 kernel 在校准域 median ratio 位于 `[0.9,1.1]`，OOF MAPE 不高于 20%；完整 sparse attention granular 对 module 的误差不高于 25%。未达标应保留失败报告，不用 shape 查表系数掩盖。

## 9. 风险与明确限制

1. **后端版本风险：** MSA kernel 和融合边界仍快速演进，SGLang 0.5.14 快照不等于未来生产版本。
2. **跨硬件风险：** block TopK、Triton autotune、split-K chunks 和 page-table fusion 对 Hopper/Blackwell 特化明显；单 H100 参数只能视为阶段性模型。
3. **TP/CP 风险：** KV/index head 复制及 CP 下 prefix/block ownership 尚需从目标 SGLang model class确认。
4. **融合风险：** standalone kernel 相加不会自动复现 fused projection、cache store和多 stream overlap，必须用 module trace校验边界。
5. **量化风险：** main KV cache、index-K cache、projection GEMM、MoE 和通信 dtype 是独立维度；不能用一个“FP8 model”开关统一替代。
6. **MoE 精度风险：** 现有 MoE 模型可复用主要计算量，但 sigmoid routing、correction bias、SwiGLU-OAI clamp 和实际 expert imbalance 仍是近似项。
7. **范围限制：** 视觉塔、DeepEP、MTP 的专门执行优化、非 CUDA backend 和完整跨卡校准不属于首版验收。

## 10. 推荐执行顺序

```text
P0. 固定目标 SGLang commit并补齐模型配置证据
P1. 修正 3 dense + 57 sparse/MoE 的 AIC layer graph
P2. 用现有组件完成 dense recipe 和 sparse projection/small-op recipe
P3. 实采并建模 MSA block index score 与 block TopK
P4. 实现 selected-block GQA FA adapter并做 H100 kernel验证
P5. 接入 ANALYTICAL granular、provenance和三档参数
P6. 运行 module/Nsight归因、单测和无表端到端 sweep
P7. 固化适用范围；再评估 Blackwell、CP、MTP和DeepEP扩展
```

这一顺序先消除模型层数和模块类型的确定性错误，再投入新的 kernel 采集。否则即使 MSA 单算子拟合准确，端到端结果仍会因 3/57 layer graph 错误而失真。
