# 三款核心后端实例 (Backends) 异同比对分析

本篇对 `aiconfigurator/sdk/backends/` 下的核心三大实现子类（`vLLMBackend`, `SGLANGBackend`, `TRTLLMBackend`）进行结构与运作特点的深度比对。

## 1. 结构与继承一致性 (相同点)

无论底层实现差异多大，这三款 Backend 全部严依其父类 `BaseBackend` 设定的规范运转：
*   **统一的骨架**: 都极度细化实现了 `run_agg`、`find_best_agg_result_under_constraints` 以及 `_get_memory_usage`。
*   **微观查算复用 (Static Evaluation)**: 三款本身都不手写静态算力分析。全部一致地反向调用了父类实现的 `self._run_static_breakdown()` 函数，按层获取时间开销后重新整合其框架对应的并发排期。
*   **管线惩罚（Pipeline Penalty）模型一致**: 令人惊讶的是，`vLLM`、`SGLang` 乃至 `TRTLLM` 在 `run_agg` 里推演 Continuous Batching 管线耗时，统一采用了 `-3`（`num_mix_steps_for_tpot_calc = max(1, num_mix_steps - 3)`）的新队列更替惩罚。

## 2. 核心功能及框架逻辑映射 (不同点)

不同推断引擎的工程与架构特性被极其精妙地映射在了代码配置上：

| 评测维度 | vLLM (`vllm_backend`) | SGLang (`sglang_backend`) | TRT-LLM (`trtllm_backend`) |
| :--- | :--- | :--- | :--- |
| **主要定位** | 学术界标杆，连续批处理（Chunked Prefill）的主流。 | 面向特异化前缀复用（RadixAttention）的极优框架。 | NVIDIA 原生自研工业级产品，具备高度硬编码预分配。 |
| **`run_agg` 区别** | 完全基于动态切分的短平快微排期。利用 `min(2 + ..., 4)` 函数为高并发补上饥饿调度罚时（TTFT 惩罚修正）。 | 完全同构于 vLLM 测算，但其 Attention 消耗在第二步通过前缀平摊给单步消除了。 | `run_agg` 流程近似，但其所需前置常量极多（如 `max_num_tokens`, `max_seq_len`）。 |
| **`_get_memory_usage` 账本记算差异** | *(当前代码态中，其功能完全指向并委托利用了 `TRTLLMBackend()_get_memory_usage` 计算以求保守安全界限)* | **复用加点（Radix/Prefix-Aware）**<br>利用前缀直接截去了激活大小`num_tokens=isl-prefix`；但追加了高达`15%`与`20%`分别赋予激活区与系统的 Python 调度附加开销常驻。 | **静态硬池峰留存（AOT Pre-allocation）**<br>放弃请求伸缩计算。强上`TRTLLM_DEFAULT_MAX_NUM_TOKENS=8192`。KV Pool 硬抠 `1.5%` 安全预留并强制上锁最高显存只能吃 `90%`。 |

## 3. 总体结论

*   **执行调度（Time & Pipeline)**: 三大框架对于时间步距的仿真算法呈现高度趋同（Chunked 算力拼装+经验排队惩罚常数修正）。
*   **内存资源管辖 (Memory)**: 三者发生了强烈的理念分化。SGLang 选择以**激进复用降低峰值但承认极高调度底噪（Overhead）** 的方式；而 TRT-LLM 与（目前的） vLLM 评估线采用**重装备工业级兜底保留预锁算盘**（直接将 `tokens` 放至 8192 等假定最大值并用硬系数吃掉 10% 左右内存）来提早拉断 OOM 界限，保障不出错。

#### 💡 补充注解：关于 P/D 分离架构（Disaggregated）下的边界退化
值得重点强调的是：上述关于**基于 `mix_step`、Chunked 及排队惩罚常数的 `run_agg` 机制探讨**，全部局限于“同构单集群”混合部署的语境下。
当仿真工程进入 **P/D 分离（Prefill & Decode Separated）** 模式（即交由外层 `DisaggInferenceSession` 调度时），这三种 Backend **引以为傲的 `run_agg` 混合队列算法将被彻底剥离与架空**。
*   在分离模式中，Context 与 Generation 被拆分在不同节点的纯时域运行，微观算力阻断消失。因此 `DisaggInferenceSession` 只会调取它们基类（`BaseBackend`）的纯底层算力计算接口 `run_static`。
*   此时，实例化具体三个子类的唯一核心价值，**就是完全依赖它们在第二点（内存资源管辖）中分化极其强烈的 `_get_memory_usage` 账本计算法，去排查单节点 OOM 安全水位线**（以及提供各自 `name` 供底层进行 C++ 运行时的静态查表）。后端框架在混合时域调度上的魔力则并在 P/D 分离的高维重构中被结构性消解。
