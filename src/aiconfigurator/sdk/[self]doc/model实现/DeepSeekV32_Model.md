# DeepSeek-V3.2 (GLM-5) 模型仿真拆解

**快速概括**：`DeepSeekV32Model` 主要用来推演 DeepSeek-V3.2 系列（包括以此架构衍生的 `GlmMoeDsaForCausalLM`）网络。该类别的核心特征是拥有与 DeepSeek V2/V3 类似的宽专家路由，但是**其注意力机制不再局限重构为早期的单一 MLA 计算，而是基于更细颗粒度的 DSA (Distillation/Dynamic Sparse Attention) 或者其近似变体**，从而在仿真模型中通过 `ContextDSAModule` / `GenerationDSAModule` 进行时延构建。

## 1. 结构初始化配置解析

在 `DeepSeekV32Model.__init__` 环节中，模型会进行一系列宏观和并行的校验装配：

*   **并行结构拓扑检验 (Asserts)**：
    ```python
    assert (
        self.config.tp_size * self.config.attention_dp_size == self.config.moe_tp_size * self.config.moe_ep_size
    )
    assert num_experts >= self.config.moe_ep_size
    ```
    这里是一个典型的分离并行域校验。在深求系列模型特有的并行机制中，**MoE 层往往有着不同于 Attention 层的通信切环方式**。模型保证其基础算子集群规模（TP切分 × Attention 级 DP 切分）必须对等于其 MoE 结构的并行拓扑（MoE 的 TP × Expert 并行度 `moe_ep_size`）。同时确保切割到单卡上的 EP Expert 数量不会超出现有网络设计的上限。

*   **多步投机系数计算 (`_mtp_scale_factor`)**：
    包含 MTP (Multi-Token Prediction) 或类似多节点联合推演的系数推演：
    $$
    Scale = \frac{1}{1 + E[n]} \times \frac{nextn + L}{L}
    $$
    该系数（`_mtp_scale_factor`）后续直接乘在了 `generation_ops` 的各项层数里。意为当开启投机解码（多 Token 跳级生成）时，其解码期的**实际模型循环轮次折算期待值**。

## 2. Ops 仿真图纸表 (Ops Table)

在这个阶段，该计算图谱分为全维度的 `context_ops` (首字计算阶段，往往表现为矩阵乘) 回合，以及带有 `_mtp_scale_factor` 并降级为向量核的 `generation_ops` (自回归解码阶段)。
大体算子在两端重合，故而在下面表中统合说明；二者差异直接置于 “差异 / 注释” 列中。

| Ops 算子图节点名称 | 对应的 Ops 系统算力类 (Class) | 功能简述与模型维度 | 阶段差异 / 并行相关注释 |
| :--- | :--- | :--- | :--- |
| **`embedding`** | `ops.Embedding` | 词表到隐含层的映射转化。维度：`vocab_size -> hidden_size`。 | Gen 期带有 `_mtp_scale_factor` 层厚乘数；因为词表维只做向量查询，故无显式通讯开销。 |
| **`add_norm_1`** | `ops.ElementWise` | Attention 层前端的 Add & Norm/RMSNorm 组件。处理 `2 * h` 吞吐。 | 经典逐元素算力。 |
| **`attention`** | **`ops.ContextDSAModule`** <br/> **`ops.GenerationDSAModule`** | 替代了原来粗犷的 Attention 计算。根据提供的 `architecture` 在内部进一步展开或执行 DSA 特化结构。 | **[架构特化核心]**。<br/>此处是此类的灵魂分水岭：Context 期的计算量跟 $SeqLen^2$ 有关，Gen 期只跟 $SeqLen$ 成比例。且两者使用了不同针对性的 Quant 模式（如传入 `fmha_quant_mode`）。 |
| **`add_norm_2`** | `ops.ElementWise` | FFN (不论 Shared 或 Routed) 层前端的归一化。 |  |
| **`shared_gate_up_gemm`** | `ops.GEMM` | **Shared Expert** 内的共享宽路由 Gate+Up 合法映射组合乘。维度：`2 * inter_size // tp_size` | 在 MoE 切分中，**共享专家一般跟随正常的 TP 进行权重切分**（因此除了 `/ tp_size`）。 |
| **`shared_act_gate`** | `ops.ElementWise` | **Shared Expert** 内部的非线性激活 (通常为 SiLU 打底)。 |  |
| **`shared_ffn2_gemm`** | `ops.GEMM` | **Shared Expert** 内通过非线性层后的 Down 回归隐层乘操作。 | 同 Gate/Up 均被 `tp_size` 除摊。 |
| **`router_gemm`** | `ops.GEMM` | **Routed Experts (MoE 动态路由)** 的大门票结算器。将隐状态求积运算各个专家的分数。维度：`num_experts` | 写死使用 `common.GEMMQuantMode.bfloat16`（保持 Gate 计算精度以防断流坍塌）。层厚度受投机规模约束。 |
| **`moe_pre_dispatch`** | `ops.MoEDispatch` | MoE All-to-All 的前向发射组。负责将打分最高的 Tokens 发往别的卡。 | 其通信代价严重受到 `moe_tp_size`、`moe_ep_size` 的耦合。Context 期由于批次多因此发送密集。 |
| **`moe`** | `ops.MoE` | 被激活专家的纯净计算矩阵群（一般内置了 Gate/Up/Down Gemm 计算块）。 | 本地 Expert 被命中后的实体测算，代入了基于幂律的 `workload_distribution` 失衡常数，仿真排队。 |
| **`moe_post_dispatch`** | `ops.MoEDispatch` | 发射组计算完后 All-to-All (汇聚) 归还本地卡的算子。 | 功能与前向对应（由 `False` 参数表明归聚向）。 |
| **`logits_gemm`** | `ops.GEMM` | Unembedding 的 Logits 输出头。 | Gen 期无此显式记录算子。Context 期则输出全词表。写死采用 `bfloat16` 以防输出坍塌。 |
