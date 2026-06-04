# SGLang Backend 代码详尽解析

`sglang_backend.py` 专门为基于 SGLang 框架的执行逻辑设计，并映射了其在模型服务中引入的 `RadixAttention` 前缀复用技术、以及高度优化的连续批处理（Continuous Batching）等特性。该子类实现了 `BaseBackend` 的三大核心接口，每一处内部细节和数据流转都对应着原生的物理架构特征。

---

## 1. `run_agg` 函数：吞吐与时延生命周期测算

`run_agg` 负责单一配置组合之下，基于 SGLang 的宏观时间分配模拟（Chunked Prefill 与 Decode 步调度计算），从而得出请求总体 TTFT（首字延迟）、TPOT（单字延迟）以及并发吞吐（Tokens/s）。

### 1.1 函数输入输出 (I/O)
*   **输入 (Inputs)**:
    *   `model` (`BaseModel`): 仿真模型对象（含网络大小、形状配置等）。
    *   `database` (`PerfDatabase`): 硬件性能数据库，包含基础算符和通信的 latency mapping。
    *   `runtime_config` (`RuntimeConfig`): 运行配置，关键提取 `isl` (Input Sequence Length), `osl` (Output Sequence Length), `batch_size` (b), `prefix` (公共前缀长度)。
    *   `kwargs`: 最核心包含 `ctx_tokens` (表示 SGLang 的 Chunked 粒度步长，表示单次 Context 处理的最大 token 发送限额)。
*   **输出 (Outputs)**:
    *   返回 `InferenceSummary` 对象：填充了包含 TTFT、TPOT、总功耗平均值、并记录资源墙报警状态（OOM 等）和综合维度的统计结果 DataFrame。

### 1.2 核心执行流与子函数拆解

SGLang 具备极强的短步长混合预测调度机制（基于 Chunked Prefill）。该逻辑在代码中由动态推演步骤切分体现。

#### > 步骤 A: Pipeline 宏观推演算法
计算 `steps_to_finish_ctx = np.ceil(isl * b / ctx_tokens)` 决断 Context 处理所需总步数。
区分 `mix_step`（混合步：需要把 Prefill 的 Chunk 塞进去同时处理可能已经开始生成的请求）和 `genonly_step`（纯生成步）。
*   **如果 $b > 1$ 且 $steps \ge osl$**: 表示生成的 Token 全部混在 Context 中消化完毕，`num_mix_steps` 等于完成 Context 所需步数，`num_genonly_steps = 0`。
*   **如果 $b > 1$ 且 $steps < osl$**: 此时引出了 **原生 SGLang 的 Pipeline Pipeline Correction (经验修正)** 设计 
    `num_mix_steps_for_tpot_calc = max(1, num_mix_steps - 3)`。这反映由于 SGLang 的引擎对于新进入请求列队存在排队开销，采用 `-3` 的延迟管道校正来模拟实际上不能无缝“无黑障”立刻填充新任务。

#### > 步骤 B: 调用子函数 `_get_mix_step_latency()` 计算混部耗时
此函数负责解析当同时在预填充 (prefill) 和生成 (decode) 的 step 消耗多少 Latency 与 Energy：
*   **输入**: Chunk 过后的临时 `ctx_tokens` 以及参与随跑生成的 `gen_tokens`。
*   **逻辑特征**: 
    对静态底层计算 (`self.run_static`) 放出了 3 遍扫描！这是因为 SGLang 的 Attention 在混部时拆成了极其复杂的重组：
    1.  **第一遍**: 获取非 Attention 算子的公共底座延迟和功耗 (`non_attention_latency_ms`)。
    2.  **第二遍**: 利用 `batch_size = np.ceil(ctx_tokens / isl)` 送入 Context Prefix，算出 `context_attention` 大小后，再平摊到这几步 (`/ scale_factor`)。**反映了 SGLang 在 RadixAttention 处理下由于前缀匹配，实际执行时间按块被均摊化解算的原生逻辑**。
    3.  **第三遍**: 追加随行并跑生成的 `generation_attention_latency_ms`。

#### > 步骤 C: 调用子函数 `_get_genonly_step_latency()` 计算独立 decode 耗时
*   **逻辑特征**: 非常简单的纯自回归查库，单纯将 `osl` 等切为纯 `static_gen` 送入 `run_static` 计算。

#### > 步骤 D: 聚合汇总
最后将得到的 `mix_step_latency_ms` 和 `genonly_step_latency_ms` 根据排好的 `num_mix_steps` 与 `num_genonly_steps` 加权。
*   **TTFT 的额外修正**: `ttft * min(2 + (steps - 3) / 20, 4)`，体现出 SGLang 在面临极端高并发 Chunked 下，最左侧的首批 Request TTFT 会出现队列恶化带来的经验拖尾效应。 

---

## 2. `find_best_agg_result_under_constraints` 函数：动态约束寻优

SGLang 在并行调度时对内存极为敏感。此函数专门用于在 `b` (Batch) 与 `ctx_tokens` 组合矩阵构成的空间下，找出在 OOM 边际内最高吞吐的执行配置。

### 2.1 函数输入输出 (I/O)
*   **输入**: `model`, `database`, `runtime_config`, `max_batch_size`, `ctx_stride` 以及 SLA（SLA限制：`ttft` 和 `tpot`）。
*   **输出**: 最好的 `InferenceSummary` （拥有最高 `seq/s` 的运行配置）。

### 2.2 执行过程与原生对应逻辑
1.  **生成扫描维度配置**: 首先列出激进分布的 `b_list` （自 1, 2, ..., 1024 翻倍跃升）。并通过 `self._get_ctx_tokens_list_for_agg_sweep(isl, ctx_stride)` 获取基于 SGLang 特性的候选 Chunk 分块集合。
2.  **平衡修正与 OOM 爬坡截断**:
    *   在双重循环内部计算 `balance_score = isl * b / ctx_tokens / osl`。
    *   **原生逻辑设计反光**: sglang 对请求具有特定的截断保护。当在某个 Batch 下如果测试到 `summary.check_oom()` 或 `check_kv_cache_oom()` 返回 True，意味着在 SGLang 的 Paged Cache/Radix Pool 被当前组合耗干，由于资源随 $b$ 和 $ctx\_tokens$ 单调递升，代码直接执行 `break` 跳过后续更庞大浪费时间的维度搜索。
3.  **最终筛选**: 判断 SLA 达标（$\le tpot, \le ttft$），之后由最高吞吐量（`seq/s`）夺魁。

---

## 3. `_get_memory_usage` 函数：SGLang 特供显存记账逻辑

这是这支代码最体现 SGLang/RadixAttention 全局框架生命周期分配哲学的函数。相比 TRT-LLM 硬扣内存的做法，SGLang 因为其基于 Python 启动调度栈（如 vLLM 也类似，但 SGLang overhead 会单独校定），显存划分充满了动态乘数。

### 3.1 函数输入输出 (I/O)
*   **输入**: 模型硬件基本常识(`beam_width`, `isl`, `osl`, `batch_size`) 以及 `prefix` (前缀数)。
*   **输出**: 包含详细分解项字典的 `dict` (`{"total": xxx, "weights": yyy, "activations": zzz, "kvcache": kkk, ...}` 单位换算为 GiB)

### 3.2 逻辑特征深度分析

整个内存记账被割分为 `[权重 + 激活内存 + KVCache + NCCL内存 + 杂项操作系统内存]`。其中极具特色的地方在于：

#### > `Activations` (激活内存池) - SGLang 特性的高位保底机制
*   不同于严格只给当前激活算子算内存的做法，代码在计算 `num_tokens = (isl - prefix) * batch_size` 后（**这里完美折射了 RadixAttention 结构对于公共 Prefix 前缀由于已被前置处理复用而不需要再度耗费大量活体中间激活计算区域的特征**），按照模型系列（如 GPT, MOE, DEEPSEEK）去乘上极大的经验常数因子 `c_dict`。
*   **额外的开销系数**: 
    `activations = max(activations, 90 * 1024 * 1024)`: 保底 90MB。
    `sglang_overhead = activations * 0.15`: 追加 **15%** 的原生 Overhead！这是因为 SGLang 运行时包含的 CUDA Graph 以及复杂状态机系统（特别是负责 Radix Tree 派发查找和状态转移的控制栈）有着比同类型框架稍高的 Python/CUDA 常驻激活调度消耗假设。

#### > `KV Cache` 与 `System` 余量
*   **KV Cache**: 虽然前边计算 Activation 扣除了 Prefix，但在存储静态持久化的 KV 池时：`seq_tokens = isl + beam_width * osl` 被直接送入公式，乘以全局的 `batch_size` 占用计算。这主要是保守估算其槽内仍要对每个批次实际装载引用的 Cache Block 进行索引持有的容量需求。 
*   **Others/System 开销**: 
    额外叠加了 `sglang_system_overhead = others_mem * 0.2`（惩罚 20%的额外节点后台进程开销），精准反应其实际物理机跑分时由于 Radix 服务化管理后端等辅助线程对剩余 VRAM 空间的占用。
