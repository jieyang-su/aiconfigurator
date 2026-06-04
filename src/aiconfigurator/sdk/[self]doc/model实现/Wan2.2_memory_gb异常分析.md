# Wan2.2 仿真 memory_gb 异常分析

## 结论

当前 RTX PRO 6000 sweep 中 `memory_gb` 出现数百到上千 GiB，基本可以判定为仿真后端的显存记账错误，不代表 Wan2.2 实际推理显存需求。

主要问题不是 collector 数据，也不是通信估算，而是 SDK 中 Wan 模型的 `Operation.get_weights()` 与 latency `scale_factor` 复用了同一个倍率，导致 denoising steps 和 layer 次数被错误计入常驻权重显存。

## 现象

在 `summary_table.csv` 中可见：

- A14B `tp=1` 时 `memory_gb` 约 `1350–1550 GiB`。
- `tp=2/4/8` 时近似按 `1/2、1/4、1/8` 下降。
- SP、Ulysses、Ring 变化对 `memory_gb` 影响很小。

这说明当前显存大头来自 `weights`，且这些 weights 被 TP 切分；但绝对值比 Wan2.2 权重规模高出一个数量级以上。

## 代码路径

`memory_gb` 来自 SGLang backend：

- `src/aiconfigurator/sdk/backends/sglang_backend.py:544` 遍历 `model.context_ops`。
- `src/aiconfigurator/sdk/backends/sglang_backend.py:546` 执行 `weights += op.get_weights()`。
- `src/aiconfigurator/sdk/backends/sglang_backend.py:625` 输出 `total = weights + activations + kvcache + nccl + others`。

Wan 的 latency 构图在：

- `src/aiconfigurator/sdk/models.py:1085` 起构造 DiT denoising path。
- 大量 DiT GEMM 使用 `steps * self._num_layers` 作为 `scale_factor`，例如 `src/aiconfigurator/sdk/models.py:1116`、`src/aiconfigurator/sdk/models.py:1170`。
- `WanStaticGEMM.get_weights()` 在 `src/aiconfigurator/sdk/operations.py:2068` 返回 `self._weights * self._scale_factor`。
- `WanMeasuredOp.get_weights()` 同样在 `src/aiconfigurator/sdk/operations.py:2036` 返回 `self._weights * self._scale_factor`。

这对 latency 是合理的：同一个 DiT block 在 40 层、50 个 denoise step 中反复执行，时延要乘 `steps * layers`。

但对显存是不合理的：权重是常驻参数，只加载一份；同一层权重不会因为 50 个 denoise step 重复加载 50 份，也不会因为一次仿真中重复执行同一 op 而变成多份常驻显存。

## 错误量级解释

以 A14B 为例，DiT 主体包含大量 `WanStaticGEMM`：

- self-attn q/k/v/out
- cross-attn q/k/v/out
- FFN fc_in/fc_out
- text/time/image embedding linear

这些 op 的 `scale_factor` 往往是 `steps * num_layers = 50 * 40 = 2000`。当前 `get_weights()` 把单个 GEMM 权重也乘以 2000，于是常驻参数显存被当成“执行次数累计显存”。

这解释了为什么结果会到 TB 级，也解释了为什么它随 TP 增大近似成倍下降：GEMM 的 `n` 或 `k` 已按 TP 切分，但错误放大的仍是权重项。

## 当前 activation 估算不是主因

Wan 分支的 activation 估算在 `src/aiconfigurator/sdk/backends/sglang_backend.py:558`：

```python
activations = 2 * batch_size * seq_len * hidden * 8 / tp_size
activations += 2 * batch_size * latent_frames * hidden * 4
activations += overhead
```

对 A14B 典型 `seq_len=111600, hidden=5120`，该项约为十几 GiB 级别；对 TI2V-5B 约为数 GiB级别。它可能仍需进一步校准峰值 workspace、attention backend 临时缓存、VAE decode 峰值等，但不能解释 `1000+ GiB` 的异常。

## 修正建议

建议把 Wan latency 倍率和常驻权重倍率分离：

1. **修改 Wan op 的权重语义**
   - `WanStaticGEMM.query()` 保持 `latency * scale_factor`。
   - `WanStaticGEMM.get_weights()` 改为只返回单份权重，或返回显式 `weight_scale_factor`。
   - `WanMeasuredOp.get_weights()` 同理不要默认乘 latency `scale_factor`。

2. **区分“层复用”与“层实例”**
   - `steps` 不应进入权重显存。
   - `num_layers` 是否进入权重显存取决于 op 表达方式：如果一个 op 代表单层模板，则权重应乘 `num_layers`；如果构图中每层没有单独建 op，则需要专门的 `weight_scale_factor=num_layers`。
   - 当前 `steps * num_layers` 应拆为 `latency_scale_factor=steps * num_layers`、`weight_scale_factor=num_layers`。

3. **优先实现 Wan 专用权重账本**
   - 更稳妥的方式是在 `WanVideoModel` 中显式计算模块级参数量：T5、CLIP、DiT、VAE。
   - 然后在 SGLang backend 的 Wan 分支直接读取 `model.get_wan_weight_bytes_per_gpu()`，避免从执行 op 反推权重。
   - 这样也更容易表达 A14B high/low noise expert 切换：同一时刻只需加载/激活一组 DiT 专家，或按实际 SGLang 是否常驻双 expert 来配置。

4. **修正后验收标准**
   - A14B BF16 单卡 `tp=1` 权重显存应落在几十 GiB量级，而不是 TB 级。
   - `memory_gb` 应随 TP 下降，但基线应合理。
   - SP/Ring/Ulysses 主要影响通信与局部 activation/workspace，不应显著切分模型权重。
   - denoising steps 改变应显著影响 latency，但不应按比例影响 weights memory。

## 暂不建议

不建议通过改 `summary_table.csv`、collector 数据或系统 YAML 的显存容量来掩盖该问题。当前异常来自 SDK 的模型显存账本语义，应在 `operations.py`/`models.py`/`sglang_backend.py` 中修正。
