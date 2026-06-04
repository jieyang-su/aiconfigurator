# 📋 [Agent Task] `aiconfigurator/collector` 组件代码库深度分析需求说明

## 一、 项目背景与 `collector` 的生态定位

在使用本说明分析代码前，请先理解 `aiconfigurator` 框架的整体宏观逻辑及被测模块的生态位。

### 1.1 `aiconfigurator` 整体逻辑
`aiconfigurator` 是一个大模型端到端推理性能仿真与部署配置搜优框架。它的核心逻辑是在用户给定的硬件约束下，利用数学公式（SOL）和**经验性能库表**，在巨大的并行策略搜索空间（TP、PP、EP、Batch Size 等）中通过推演寻找到最优的部署配置。

### 1.2 `collector` 库的定位与作用
如果说 `sdk` 模块是负责根据公式寻找最优解的“大脑”，那么 `collector` 就是下放至真实物理 GPU 上采集物理真理的“探针”。
*   **用途定位**：它由一系列用于物理真机环境压测的 Python/Shell 脚本组成（分为 sglang、trtllm、vllm 等特定引擎目录）。通过在真实硬件上模拟不同的数据类型、张量形状，来获取**极高精度的算子/模块运行耗时、功耗等硬件微观指标**。
*   **与 `PerfDatabase` 的配合闭环**：`collector` 压测结束后，会输出扁平化的纯文本表格（如 `gemm_perf.txt`）。在随后的架构仿真中，`sdk/perf_database.py`（仿真主程序的性能数据库引擎）会直接挂载并解析这些文件。当仿真遇到复杂或融合度极高、难以用纯数学理论（Speed of Light, SOL）精准刻画内存与算力重叠开销的低比特算子时，能够直接**查表插值**，用 `collector` 提供的“物理金标准”代替数学估算，从而极大缩小理论仿真与物理真机之间的误差。

---

## 二、 任务目标与执行要求

请作为高级系统架构师与性能优化专家，对当前代码仓库下的 `collector` 库源码进行地毯式分析，并严格按照以下两个阶段输出结构化的分析报告。

### 📌 阶段一：提炼 `collector` 库的共性架构与基础设施

不同引擎的 `collect_xxx.py` 脚本虽然测试的算子不同，但共享了一套极其严谨的 Benchmark 基础设施。请概括这套共性结构及实现方式，重点分析包括且不限于以下几点：
1.  **全局控制流框架**：测试用例（Test Cases）的生成机制（如使用 `common_test_cases.py` 定义的枚举空间），以及如何通过嵌套循环将参数传递给测试主体。
2.  **核心基准测试方法（`helper.py` 的精妙运用）**：详细剖析 `benchmark_with_power` 上下文管理器的行为原理（如 `CUDA Graph` 捕获、降频侦测、自适应迭代次数 `num_runs`、GPU L2 Cache 刷新策略、Warmup 机制等）。
3.  **身份与元数据对齐（`registry_types.py` 等的联动）**：代码结果如何对接 `log_perf` 函数，并利用 `PerfFile` 枚举值（如 `GEMM`, `CONTEXT_MLA`）强规范输出文件名，以便保障与上游 `PerfDatabase` 消费端的数据格式对齐。

将该总体上的总结输出为一个md文档。

---

### 📌 阶段二：分推理后端（Backend）输出独立算子实测分析档案

请按主要推理引擎（如 `TRT-LLM`、`SGLang`、`vLLM` 等）作为一级标题/独立文档进行整理。
考虑到代码中有大量的 `collect_` 脚本，许多脚本不仅是在测**细粒度的单算子**（如 GEMM），甚至是在测**宏大的模块**（如 Attention Context/Generation, MoE, 甚至是整个 Layer Module）。

对于每个后端下的每个 `collect` 代码文件，请使用**Markdown 高级表格**的形式整理，并严格包含且考证以下列（表项）：

| 评估维度列名 (Table Columns) | 数据要求与分析重点说明 (Guide for Agent) |
| :--- | :--- |
| **所属文件与注册标识** | 1. 脚本文件名（如 `collect_gemm.py`）；<br>2. 对应的 `registry_types.PerfFile` 映射表名标识。 |
| **测试粒度界定** | 判断究竟是 **底层单算子**（如纯矩阵乘 Base GEMM），还是 **复合大模块**（如 Attention 包含 RoPE+QKV_Proj+Softmax，或 MoE 可能包含 Routing+GEMM）。结合具体代码具体分析。 |
| **核心执行载体** | 实际承载计算的核心 API/代码引用（如 `tensorrt_llm...Linear`、`sglang...fp8_scaled_mm` 或 `FlashAttentionBackend`）。 |
| **包含的量化类型与前置转化全景** | **核心重点！** 列出涵盖的量化格式（FP16/FP8/FP4 等）。<br>必须详细指出：在低比特量化下，是直接测试算子内核，还是**包含了从显存加载高精度（如 BF16）并执行动态量化的前置小算子开销**（如寻找 Scale 的 Overhead）。 |
| **输入变量与保存的数据表** | 记录测试时的特定张量形状生成方式（如 M, N, K, Head 的排布），以及其他输入参数等。整理代码输出保存数据表的包含内容。 |
| **代码执行过程要点解析** | 使用精炼的技术语言按时序总结。<br>例如“张量初始化分配 -> (可选的离线重排) -> `kernel_func` 构建与量化闭包封装 -> 交由 `benchmark` 执行测量。”。需要突出代码中的关键细节 |
| **特殊设计** | 例如存在 `outside_loop_count` 的参数，它如何构建多份不连续地址的 Tensor 数组（`op_list`）以打破 L2 Cache 从而取得真实的 HBM 访存延时。对各测试用例中存在类似的一些特殊设计进行解析。 |

每个后端输出为一个md文档。