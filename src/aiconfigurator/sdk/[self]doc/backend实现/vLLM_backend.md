# vLLM Backend 代码详尽解析

`vllm_backend.py` 试图在宏观上还原 vLLM 在执行 LLM 服务时基于 Continuous Batching (连续批处理) 和 Chunked Prefill 的多步调度特性。

---

## 1. `run_agg` 函数：吞吐与时延推演

### 1.1 函数输入输出 (I/O)
*   **输入 (Inputs)**:
    *   `model`, `database`, `runtime_config`
    *   最核心：`ctx_tokens`，这里充当限制一次调度向 GPU 放行的极大 Token 宽度（Chunked 限额）。
*   **输出 (Outputs)**:
    *   `InferenceSummary`（携带 `ttft`, `tpot`, 吞吐量和 GPU 并发等全维度测算结果）。

### 1.2 核心执行过程与设计逻辑

vLLM 和 SGLang 极其相似，同样是在处理 Chunked Context 生成的步骤切片：
1.  **宏观步计算**: 
    当 `b > 1` 时，判断完成所有 Context Token 是否需要多于输出步（$steps \ge osl$）。
    计算出 `num_mix_steps` (预填充阶段与历史解码请求混合并行的步数) 与 `num_genonly_steps` (单纯自回归步数)。
2.  **调取经验公式**:
    *   如果 `steps < osl`，说明 Prefill 非常短，采用 `num_mix_steps_for_tpot_calc = max(1, num_mix_steps - 3)`。这 3 步就是为了弥补由于 `Continuous Batching` 引擎中当老请求完成后，系统队列弹出与补入新请求的 `Python Queue Overhead`（新批构建惩罚）。
3.  **_get_mix_step_latency 逻辑**:
    与 SGLang 完全一致，将单步混合（`mix_step`）剥离出“纯计算（non_attn）”、“处理新进入的前缀加权 Attention”、和“生成 Token 的并行 Attention”。这里就是通过 `run_static` 按模块分别查库获得微观周期相加得到的单步总管线时延落后。
4.  **整体 TTFT 的并发退退化**: 
    `correction_factor = min(2 + (steps_to_finish_ctx - 3) / 2 / 10, 4)`，对于 `ttft` 施加惩罚修正，代表 vLLM 架构下在高 Batch 时处理分块，首个请求被拖后返回文本所遭遇的长队饥饿开销（Starvation latency penalty）。

---

## 2. `find_best_agg_result_under_constraints` 

该函数几乎原封不动地被 SGLang 继承。
它是通过对 `b_list` （自 1 到 max_batch_size）与 `ctx_tokens_list` 嵌套双重 `for` 循环暴力扫优。
*   **提前退出机制 (Pruning)**: `if summary.check_oom() or summary.check_kv_cache_oom(): break`。由于系统在不断放大 Batch 时必然会导致内存触顶，为了加速评估速度，当 vLLM 发现当前 Batch Configuration 报了 OOM 后，直接截断内部步长循环。

---

## 3. `_get_memory_usage` 显存记账（特别注意项）

在当前库的代码版本中，`VLLMBackend._get_memory_usage` 被硬编码挂入了指向 `TRTLLMBackend` 的委派：

```python
    def _get_memory_usage(self, model, database, batch_size, beam_width, isl, osl, num_tokens=0, prefix=0) -> dict[str, float]:
        # TODO
        from aiconfigurator.sdk.backends.trtllm_backend import TRTLLMBackend
        return TRTLLMBackend()._get_memory_usage(...)
```

*   **逻辑折射**: 
    当前由于 `vLLM` 本身的 Python 显存动态伸缩账本在微观边界极其难以完美标定，开发组在此处留下了一个 Todo（将回退到使用最保守、安全边界最高的 TRT-LLM 硬限制公式）。在跑 vLLM 的评估时，**其实际计算落点在于将最大可能的并发限制于常量 TRTLLM AOT 分配下内。** （详细的解析请参考 TRTLLM_backend.md 关于 `_get_memory_usage` 的拆解。）
