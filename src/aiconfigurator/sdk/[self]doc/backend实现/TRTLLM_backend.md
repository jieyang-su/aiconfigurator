# TRT-LLM Backend 代码详尽解析

`trtllm_backend.py` 展现的是 NVIDIA 最经典的 TensorRT-LLM 在构建 (Engine Build) 和执行时的预分配（Pre-allocation）重工业策略，它的算例模型充满着“未算先留存 (AOT - Ahead of time constraint)”的设计哲学。

---

## 1. 核心常量与全局参数定调

相比 vLLM 这种完全拥抱 JIT 和运行时动态池技术的框架，TRT-LLM 在最前端就锁死了数个极其保守的全局硬边界（极具 C++ 高性能隔离系统的特色）：

*   **`KV_CACHE_MEMORY_RESERVED_FRACTION = 0.015`**: KV Cache 池禁止占满。模拟器必须为底层分配模块（Block allocator）留下 **1.5%** 的不可侵犯管理余量。
*   **`KV_CACHE_MEMORY_TOLERANCE = 0.02`**: 测算时的 2% 浮动容限带误差。
*   **`TRTLLM_DEFAULT_FREE_GPU_MEMORY_FRACTION = 0.9`**: 引擎默认在系统内吃尽可用显存的限制线锁死在 **90%**（这有别于原生 Python 引擎可以激进侵占至 95% 以上）。
*   **`TRTLLM_DEFAULT_MAX_NUM_TOKENS = 8192`**: 全局最大的 `BuildConfig.max_num_tokens`。非常重要！在 TRT-LLM 中，显存预锁（特别是 Activation Memory）高度依赖这个静态构建期间抛出的顶峰并发上限，而非实时的请求流水。

---

## 2. `run_agg` 函数解析

### 2.1 修改的输入特性
相较于其他框架，除了传 `ctx_tokens` 这些步进约束，TRT-LLM 还直接通过 `kwargs` 获取 `max_seq_len` 和 `free_gpu_memory_fraction` 与 `max_num_tokens`，如果外部没有给，它会在这层覆盖上刚刚声明的全局硬参数。

### 2.2 宏观测算逻辑（部分与基准融合）
*   **步骤排期**: `num_mix_steps` 依然沿用了和 vLLM 同源的 pipeline queue 惩罚机制 (如由于 C++ In-flight scheduling 造成的换出惩罚，利用 `num_mix_steps_for_tpot_calc` 控制)。
*   **内部计算**: 也是由第一遍非 Attention 层、第二遍上下文 Attention、第三遍生成 Attention 对 `self.run_static` 重复调用并乘以此步长得到。

---

## 3. `find_best_agg_result_under_constraints` 

执行流程与 SGLang/vLLM 大同小异。最大的区别体现在 `run_agg` 中关于内存的 OOM 判断 `if summary.check_oom() or summary.check_kv_cache_oom()` 的触发壁垒极不相同由于预分配的特征，在 Batch 放大时它会极其早地触碰到 OOM 天花板然后退出。

---

## 4. `_get_memory_usage` 极严格的重度分配算法

这是 TRTLLMBackend 的灵魂所在，也是该文件与 SGLang，vLLM 差异最为巨大的模块（目前甚至 vLLM 也在借助 TRT 的此函数）。

### 4.1 输入的重置
TRT-LLM **完全摒弃了按需缩放模式**。在 `run_agg` 调用 `_get_memory_usage` 时，传入的不再是当前正欲请求量，而是：
*   `num_tokens=max_num_tokens`（也就是前面提过的全局常数 `8192` 或者外部限制极值）。
*   `max_seq_len=max_seq_len` 序列被推到极致边界计算。

### 4.2 显存组成计算逻辑
1.  **权重视图 (Weights)**: `weights /= pp_size` (张量并行 TP 不切分模型参数厚度而 PP 管道并行需要平摊模型载体大小)。
2.  **活体激活空间 (Activations)** - **按最大能力留空而非动态加成**:
    *   以 `min(70MB, max(2 * max_num_tokens * h * c_dict[tp_size]))` 保底限度来框定。 
    *   如果算到了 MoE 系列如 DeepSeek：在激活空间里还要叠加非常恐怖的 `moe_workspace` (用于专家路由下发)，大小跟 `num_tokens * h * num_experts * topk / moe_ep_size` 的张量位字节成正比。
    *   **MTP (NextN Token) 惩罚**: 对于能进行推测解码（Speculative Decoding）或包含投机头特性的， `activations = activations * (model.config.nextn + 1)` 激活内存池还要等比例翻倍。这一切都在生成前全额备好。
3.  **KV Cache 算顶池 (Block reserved)**:
    直接通过 `seq_tokens = max_seq_len if max_seq_len is not None else isl + beam_width * osl` 进行最粗线的生命周期测算。
4.  **底层通信协议与驱动预留**:
    加入了强硬的 `nccl_mem`（由于 TP 大小导致 NCCL Ring/Tree 构建时内存开销跃迁）和额外的 `other_mem`（CUDA Context、cuBLAS handle 的初始起跳消耗）。全部相加后向 OOM 校验器送出结果。
