# DeepSeek-V4 模型仿真拆解

**快速概括**：`DeepSeekV4Model` 是专门针对 DeepSeek-V4 高级架构的算力与显存仿真的模型封装。该架构在前作（如 V3/V3.2 甚至 DeepSeek V2）的宽专家（WideEP MoE）路由和 MLA 注意力基础上，创新融合了 **mHC（Multiple Head Compression）结构**，并通过引入含有 **SWA (Sliding Window Attention) / CSA / HCA 降本降速衰减技术的压缩组合层 `compress_ratios`**，彻底改变了原本粗犷的单类型 Attention 测算范式。

## 1. 结构初始化配置解析

在 `DeepSeekV4Model.__init__` 初始化期间，特有的设置和校验极其严密：

*   **专有的配置加载限制 (`DeepSeekV4Config`)**：
    由于引入 mHC（前后的汇聚/路由）和大量非统一压缩 Attention 块，框架强制检查 `self.extra_params` 必须解析为合法的 `DeepSeekV4Config` 类对象。
    
*   **压缩比例白名单判定 (`_SUPPORTED_COMPRESS_RATIOS`)**：
    ```python
    _SUPPORTED_COMPRESS_RATIOS: ClassVar[set[int]] = {0, 4, 128}
    ```
    强制约定压缩粒率只能是预定义的这几种模式：如 `0` 代表 SWA 层（不压缩或近似直接截断滑动窗口），`128` 等代表典型的高段 HCA (Heuristically Compressed Attention)。对于未知比率框架直接抛出非法异常，以防没有对应的算力测定数据库可用。
    
*   **拓扑并行断言 (Assert) 及多步预测补偿 (`_mtp_scale_factor`)**：
    这部分保留了 DeepSeek V3.2 同样的验证逻辑：确保整体的 `tp_size * attention_dp_size == moe_tp_size * moe_ep_size` 的通讯环闭合前提，并且运用投机期模型生成加速比测算单次 Decode Token 需要被放大或折叠的模型层厚 `_mtp_scale_factor`。
    
*   **Attention 块代理分发层 (`_attention_ops`)**：
    这是一个特殊的内置闭包构造函数。在推演 Attention 时不只插入单一的 `ops`，而是通过对配置文件中携带的 `compress_ratios`（比如总共 60 层，有部分是 SWA，有部分是 128 压缩）按计数值（`Counter()`）统计，然后再把这些特殊的异构 Attention 模块**拼接为一个混合列表插入仿真图纸**中。这也是极其罕见的**按属性频率动态组合长算力 Ops 链**的设计。 

## 2. Ops 仿真图纸表 (Ops Table)

该层也是典型的 `context` 阶段 (矩阵密集算力) 与 `generation` 阶段 (向量寻址算力并受 MTP 折算乘子约束) 的组合。下面用总表予以提炼。

| Ops 算子图节点名称 | 对应的 Ops 系统算力类 (Class) | 功能简述与模型维度 | 阶段差异 / 并行相关注释 |
| :--- | :--- | :--- | :--- |
| **`embedding`** | `ops.Embedding` | 将 Vocabulary ID 转换为模型的隐层宽度特征 $h$。 | Gen 期厚度倍乘了 `_mtp_scale_factor` 下同。 |
| **`mhc_pre`** | **`ops.DeepSeekV4MHCModule`** | **[特化架构组件] Multiple Head Compression 路由前端模块**。对隐层进行预压缩，缩减下限或抽取特定频率。通过 Sinkhorn 等矩阵迭代收敛。（参数传的是 `pre`）。 | 此时依赖特定的乘积尺度 `hc_mult` 以及迭代限度 `hc_sinkhorn_iters` 参数。 |
| **`attn_norm`** | `ops.ElementWise` | 为输入接下来各种压缩注意力的激活流执行 RMSNorm 的简单常数缩放。 |  |
| **`attention`** | **`ops.ContextDeepSeekV4AttentionModule`**<br/>且/或 **`ops.GenerationDeepSeekV4AttentionModule`** | **[特化架构组件]** 根据不同压缩比例混合生成的异构 Attention。可能包含了 `0`（近似 HCA 数据库占用）, 或 `128` 等特殊 `o_groups` 注意力开销测定。 | 此项在图中可能会连续生成多个子列表项（例如 SWA 连续 10 层，HCA 连续 20 层）。注意在 Gen 期引入压缩不仅能够缩小计算量，更重要的是 **大幅折损 KV Cache 的读取带宽代价**。 |
| **`mhc_post`** | **`ops.DeepSeekV4MHCModule`** | **[特化架构组件] Multiple Head Compression 的还原或重分配（后置端）**。参数传入 `post`。 |  |
| **`ffn_norm`** | `ops.ElementWise` | 在进入复杂的 FFN（或者 MoE 领域）前的常数缩放平摊操作。 |  |
| **`shared_gate_up_gemm`** | `ops.GEMM` | **Shared Expert** 共享侧计算。由于没有动态路由损耗而单纯依靠 TP 切分。维度被折算为：`2 * local_moe_inter_size, h`。 | Gen 期带有折算。此结构与更早模型等价。 |
| **`shared_act_gate`** | `ops.ElementWise` | **Shared Expert** 内通过 Gate/Up 相乘后发生的 SiLU 或类似非线性激活的纯元素操作。 |  |
| **`shared_ffn2_gemm`** | `ops.GEMM` | **Shared Expert** 数据从大宽幅 `moe_inter` 下维到 `hidden_size` 的矩阵乘积。 |  |
| **`router_gemm`** | `ops.GEMM` | **Routed Experts (MoE)** 网络的专家投票分配机制的乘数矩阵。将维度发射到 `num_experts` 之上获取路由倾向分。 | 数据精度严控采用纯血 `bfloat16` 计算。 |
| **`moe_pre_dispatch`** | `ops.MoEDispatch` | MoE All-to-All 通讯组前端器。将 Router 择优出的数据按 TP/EP 环向别的显卡投流。 | Context 阶段属于密集投射，通常伴随巨量大流量突发。 |
| **`moe`** | `ops.MoE` | **Routed Experts** 目标端本体。执行真正的激活和惩耗分配。 | 带上了偏置项 `power_law_alpha` 对可能产生的通信长尾 / 计算堆积进行模拟。 |
| **`moe_post_dispatch`** | `ops.MoEDispatch` | MoE All-to-All 的反流回聚器。收敛到原卡隐层。 |  |
| **`logits_gemm`** | `ops.GEMM` | 最顶部的分类反词表查询，将特征转换为概率，输出受 TP 切分的词表规模。 | 仅存在于 Context，或者在特定实现下的单一提取阶段。同样锁紧 `bfloat16` 类型。 |