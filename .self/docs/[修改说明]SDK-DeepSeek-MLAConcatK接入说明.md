# SDK DeepSeek MLAConcatK 接入说明

## 背景

SGLang DeepSeek V3/R1 在 prefill MHA 路径中，会在 `kv_b_proj` 之后把 `k_nope` 与 `k_rope` 拼成 FA3/MHA 使用的完整 `K`。该小算子只属于 prefill/context prepare 路径，decode 的 FlashMLA/MQA 路径不使用它。

此前 collector 与 H100 0.5.9 数据已经补齐：

- collector 入口：`collector/sglang/collect_mla.py::run_mla_concat_k`
- 数据文件：`src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.9/mla_concat_k_perf.txt`
- 数据规模：1540 条，其中 `concat_mla_k` 110 条，`concat_and_cast_mha_k_triton` 1430 条

## 本次 SDK 改动

- `src/aiconfigurator/sdk/common.py`
  - 新增 `PerfDataFilename.mla_concat_k = "mla_concat_k_perf.txt"`。

- `src/aiconfigurator/sdk/perf_database.py`
  - 新增 `load_mla_concat_k_data()`，按 `kernel_source -> local_num_heads -> num_tokens` 加载。
  - 在 `PerfDatabase` 初始化中加载 `_mla_concat_k_data`。
  - 新增 `query_mla_concat_k()`。
  - 查询时默认按 SGLang 0.5.9 源码分支选择 kernel：
    - `num_heads == 128` 使用 `concat_mla_k`
    - 其他 2 的幂 local heads 使用 `concat_and_cast_mha_k_triton`
    - 非 2 的幂 local heads 保留 `torch_cat_assign` 分支名
  - 0.5.9 有实测表时走 SILICON 查表；没有该文件的旧数据库版本会退到内存读写 SOL/empirical 估算，并标记 `source="empirical"`。

- `src/aiconfigurator/sdk/operations.py`
  - 新增 `MLAConcatK` operation。
  - `num_tokens = batch_size * (s + prefix)`，对齐 SGLang MHA one-shot/full K concat 的输入规模。

- `src/aiconfigurator/sdk/models/deepseek.py`
  - 仅普通 `DeepSeekModel` 接入，不改 DeepSeek V3.2/V4，也不改 WideEP 专用模型。
  - 仅在 `backend_name == "sglang"` 的 context fallback 中，于 `context_kv_b_proj_gemm` 和 `context_attention` 之间插入 `context_mla_concat_k`。
  - 修正普通 DeepSeek `create()` 传递 `backend_name`，确保 SGLang/vLLM 分支能正确区分。

## 验证

已执行：

```bash
python -m py_compile \
  src/aiconfigurator/sdk/common.py \
  src/aiconfigurator/sdk/operations.py \
  src/aiconfigurator/sdk/perf_database.py \
  src/aiconfigurator/sdk/models/deepseek.py
```

已执行轻量查询：

- `h100_sxm/sglang/0.5.9` 可加载 `_mla_concat_k_data`
- `query_mla_concat_k(8192, 128)` 命中 `concat_mla_k` 实测表，返回约 `0.2356 ms`
- `query_mla_concat_k(8192, 64)` 命中 Triton 实测表，返回约 `0.1200 ms`
- 模拟缺表时返回 empirical 估算，不抛异常

已执行模型路径检查：

- `get_model(..., backend="sglang")` 的普通 DeepSeek context fallback 包含 `context_mla_concat_k`
- `get_model(..., backend="vllm")` 不包含 `context_mla_concat_k`

## 注意事项

- `MLAConcatK` 是 prefill-only 小算子，未接入 generation/decode。
- `num_heads` 表示 TP 切分后的 local heads，与 collector 数据文件中的 `num_heads` 字段一致。
- 当前只有 H100 SGLang 0.5.9 有实测 `mla_concat_k_perf.txt`；旧版本缺表时使用 empirical 兜底以保持 SDK 可运行。

## 补充：为什么需要单独建模 concat_kv

### 1. SGLang 执行边界与 RadixAttention 采集边界

在 SGLang DeepSeek V3/R1 的 prefill MHA 路径中，`concat_kv` 更准确地说是 `concat K`：它把 `kv_b_proj` 产生的 `k_nope` 与 RoPE 后的 `k_pe/k_rope` 拼成 MHA/FA3 可直接消费的完整 `K`。该动作发生在 `DeepseekV2AttentionMLA` 的 `self_attn` 模块内，但位于 `RadixAttention(attn_mha)` 调用之前。

源码关系如下：

- `sglang/srt/models/deepseek_v2.py` 中 `DeepseekV2AttentionMLA` 创建两个 attention wrapper：
  - `attn_mqa = RadixAttention(... num_kv_heads=1 ...)`，服务 decode/absorbed MLA/MQA 路径。
  - `attn_mha = RadixAttention(... num_kv_heads=num_local_heads ...)`，服务 prefill MHA 路径。
- `forward_prepare()` 根据调度选择 `AttnForwardMethod.MHA`、`MHA_ONE_SHOT`、`MHA_CHUNKED_KV` 或 `MLA` 等路径。
- `sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py::forward_normal_prepare()` 中：
  - 先从 latent KV cache 或当前 fresh latent 中得到 `kv_a/k_pe`。
  - 执行 `kv_b_proj(kv_a)`，得到按 head 展开的 `kv`。
  - 拆分 `kv` 为 `k_nope` 与 `v`。
  - 调用 `self._concat_and_cast_mha_k(k_nope, k_pe, forward_batch)` 生成完整 `k`。
  - 返回 `q, k, v, forward_batch`。
- 随后 `forward_normal_core()` 才调用 `self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)`。
- `RadixAttention.forward()` 的职责是接收已经准备好的 `q/k/v`，reshape 后调用 `get_attn_backend().forward(...)` 或 unified attention custom op；它不负责把 `k_nope` 与 `k_rope` 拼起来。

因此，`concat K` 的边界处于：

- 模块层级：`model.layers[i].self_attn` 内部。
- 子阶段：`kv_b_proj` 之后、`attn_mha/RadixAttention` 之前。
- prefill 变体：普通 MHA、MHA one-shot 和 chunked-prefix MHA 的准备路径中都可能出现；decode FlashMLA/MQA 路径不使用它。

这解释了为什么需要单独采集和建模。AIC 原有 `collect_mla` 的 prefill attention 核心采集边界是 `RadixAttention`/attention backend，本身要求传入已经拼好的完整 `q/k/v`。`collector/sglang/collect_mla.py::_validate_context_mha_inputs()` 明确 fail-closed：prefill MHA 必须传入 fully-concatenated `q/k`，不能传 MLA rope kwargs。这说明 `collect_mla` 的 attention 行只覆盖 FA/MHA attention kernel 本体，不会自动包含 `concat_mla_k_kernel` 这类前置数据整理 kernel。

实机 nsys 也能看到同样边界：`concat_mla_k_kernel(...)` 通常出现在 `kv_b_proj` 后、`attn_mha` 前，且在 self_attn 的 kernel 序列中独立存在。若 AIC 的 prefill granular fallback 只累加 `q_b_proj/kv_b_proj/context_attention/o_proj`，就会系统性漏算这段 K 拼接开销；大 prefix 或总 K token 较多时误差会非常明显。

### 2. 算子本身做了什么

SGLang 0.5.9 CUDA 路径中的 `_concat_and_cast_mha_k()` 有三个分支：

- `concat_mla_k` 特化分支：当 `num_local_heads == 128`、`qk_nope_head_dim == 128`、`qk_rope_head_dim == 64` 时使用 `sgl_kernel`/JIT kernel。DeepSeek V3/R1 TP1 正好落在这个分支。
- `concat_and_cast_mha_k_triton` 分支：当 local heads、nope dim、rope dim 都是 2 的幂时使用 Triton kernel。TP 后 local heads 为 64/32/16/... 时通常落在这里。
- fallback 分支：普通 tensor slice assign，即 `k[..., :qk_nope] = k_nope` 与 `k[..., qk_nope:] = k_pe`。

输入/输出形状：

- 输入 `k_nope`: `[num_tokens, local_num_heads, qk_nope_head_dim]`，DeepSeek V3/R1 中 `qk_nope_head_dim=128`。
- 输入 `k_rope/k_pe`: `[num_tokens, 1, qk_rope_head_dim]`，DeepSeek V3/R1 中 `qk_rope_head_dim=64`，RoPE 部分在所有 heads 之间共享。
- 输出 `k`: `[num_tokens, local_num_heads, qk_nope_head_dim + qk_rope_head_dim]`，即 `[num_tokens, local_num_heads, 192]`。

这里的 `num_tokens` 是本次 MHA attention 需要使用的 K token 数。无 prefix 的普通 prefill 中，它等于 `batch_size * fresh_len`；MHA one-shot prefix 场景中，它等于 fresh 与 prefix 的总 K token；chunked-prefix 场景中，fresh 段和每个 prefix chunk 可能分别触发各自的 K 准备。

算子性质：

- 主要是数据搬运/重排/可能的 dtype cast，不是 GEMM 或 attention 这类高算术强度计算。
- 工作量近似随 `num_tokens * local_num_heads * (qk_nope_head_dim + qk_rope_head_dim)` 线性增长。
- 访存行为由三部分组成：读取 `k_nope`、读取共享的 `k_rope`、写出完整 `k`。因为 `k_rope` 只有 1 个 head 维度但需要广播到所有 local heads，实际 kernel 的访存/写出模式对 head 数敏感。
- 对大 prefix 场景很重要：attention 之前必须把 prefix latent KV 也恢复为 MHA 需要的 full K，`num_tokens` 可远大于 fresh token 数。

### 3. Collector 侧如何采集并与真实推理对齐

采集入口：

- registry：`collector/sglang/registry.py` 中注册 `op="mla_concat_k"`。
- 测试生成：`collector/sglang/collect_mla.py::get_mla_concat_k_test_cases()`。
- 实际执行：`collector/sglang/collect_mla.py::run_mla_concat_k()`。
- 输出文件：`mla_concat_k_perf.txt`。

测试 shape 的设计：

- `n_list = [64, 128]` 表示原始 total heads，用于覆盖 TP 后的 local heads。
- `tp_size = [1, 2, 4, 8, 16, 32, 64]`，要求 `num_heads % tp_size == 0`，实际写入数据库的是 `local_num_heads = num_heads // tp_size`。
- `b_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]`。
- `s_list = [1, 16, 32, ..., 32768]`。
- 过滤 `b * s > 65536`，让表覆盖常见 prefill K token 区间，同时避免单次采集过大。

`run_mla_concat_k()` 直接构造与 SGLang 实际 `_concat_and_cast_mha_k()` 一致的张量：

- `num_tokens = batch_size * input_len`。
- `k`: `[num_tokens, local_num_heads, 128 + 64]`，BF16。
- `k_nope`: `[num_tokens, local_num_heads, 128]`，BF16。
- `k_rope`: `[num_tokens, 1, 64]`，BF16。

然后按与 SGLang 分支一致的逻辑选择 kernel：

- 若可导入 `sgl_kernel.concat_mla_k` 且 `local_num_heads == 128`、`qk_nope=128`、`qk_rope=64`，记录 `kernel_source="concat_mla_k"`。
- 否则若 local heads 和维度均为 2 的幂，调用 `concat_and_cast_mha_k_triton()`，记录 `kernel_source="concat_and_cast_mha_k_triton"`。
- 否则走 PyTorch slice assign，记录 `kernel_source="torch_cat_assign"`。

当前 H100/SGLang 0.5.9 数据文件规模：

- 总计 1540 行。
- `concat_mla_k`: 110 行，覆盖 `local_num_heads=128`。
- `concat_and_cast_mha_k_triton`: 1430 行，覆盖 `local_num_heads=1/2/4/8/16/32/64`。
- `num_tokens` 范围为 `1..65536`。

这种采集方式与真实推理对齐的关键点是：真实 SGLang prefill 在 `attn_mha` 前调用的就是同一类 K concat kernel，且输入规模由 `num_tokens` 与 `local_num_heads` 决定。Collector 不再通过完整模型跑一遍，而是将该纯准备 kernel 单独抽出来，以同样的 shape 和分支规则压测，正好补齐 RadixAttention attention kernel 采集之外的前置开销。

### 4. SDK / PerfDatabase 如何查询与兜底估算

SDK operation：

- `src/aiconfigurator/sdk/operations.py::MLAConcatK`。
- 构造参数只有 `name`、`scale_factor`、`num_heads`，其中 `num_heads` 是 TP 后 local heads。
- `query()` 从上下文读取：
  - `batch_size`
  - `s`，即当前 fresh/extend token 长度
  - `prefix`，默认为 0
- 查询规模为 `num_tokens = batch_size * (s + prefix)`。

这里使用 `s + prefix` 是为了匹配 MHA one-shot/full-K concat 语义：进入 attention 的 K 包含 fresh 与 prefix 两部分。对于当前实机对比中未触发 chunked-kv 的用例，这个简化与 SGLang 实际 one-shot K 准备一致。若未来需要精确模拟 `MHA_CHUNKED_KV`，则需要按 SGLang 的 chunk 策略拆成 fresh 段和多个 prefix chunk 多次查询，而不是只用一次 `batch_size * (s + prefix)`。

模型接入：

- `src/aiconfigurator/sdk/models/deepseek.py` 中普通 `DeepSeekModel` 的 context prefix/granular path 插入：
  - `context_q_b_proj_gemm`
  - `context_kv_b_proj_gemm`
  - `context_mla_concat_k`
  - `context_attention`
  - `context_proj_gemm`
- 仅 `backend_name == "sglang"` 时插入该 op；vLLM 分支不使用这个 SGLang 特有的 concat kernel 建模。
- decode/generation 路径不接入该 op，因为 decode 通常走 FlashMLA/MQA，不经过 prefill MHA 的 full-K concat。

PerfDatabase 加载：

- `src/aiconfigurator/sdk/common.py` 中 `PerfDataFilename.mla_concat_k = "mla_concat_k_perf.txt"`。
- `src/aiconfigurator/sdk/perf_database.py::load_mla_concat_k_data()` 将数据组织为：
  - `kernel_source -> local_num_heads -> num_tokens -> {latency, power, energy}`。
- `PerfDatabase.__init__` 加载为 `_mla_concat_k_data`。

PerfDatabase 查询：

- `query_mla_concat_k(num_tokens, num_heads, kernel_source=None, database_mode=None)`。
- 若未显式传 `kernel_source`，按 SGLang 0.5.9 分支选择：
  - `num_heads == 128` -> `concat_mla_k`
  - `num_heads` 为 2 的幂 -> `concat_and_cast_mha_k_triton`
  - 其他 -> `torch_cat_assign`
- 若命中实测表，按 `num_heads` 与 `num_tokens` 查询。
- 若目标 `num_heads` 不在表中，会先在 head 维找邻近点，再分别对 token 维插值，最后在 head 维插值。
- token 维使用 `_interp_1d_tokens()`，允许边界外最近点外推/插值辅助逻辑；这对 `num_tokens` 超过 65536 的预测有意义，但准确性会低于表内。
- `_extrapolate_data_grid()` 对该二维 surface 额外扩展 head/token 网格，方便常见大 token 点查询。

兜底/empirical：

- SOL 估算把该算子视为纯访存型：
  - 读 `k_nope`: `num_tokens * num_heads * 128 * 2`
  - 读 `k_rope`: `num_tokens * 64 * 2`
  - 写 `k`: `num_tokens * num_heads * (128 + 64) * 2`
  - 用系统 `mem_bw` 计算理论内存时间，数学时间为 0。
- empirical 在 SOL 基础上除以 `scale_factor=0.7`，相当于考虑实际 kernel 不可能达到理论满带宽，给出更保守的估算。
- 当旧数据库没有 `mla_concat_k_perf.txt` 时，`query_mla_concat_k()` 不抛错，而返回 empirical 并标记 `source="empirical"`，保证旧版本系统数据仍能跑通；有 H100 0.5.9 实测数据时优先返回 `source="silicon"`。

整体上，`MLAConcatK` 是一个典型的小而关键的“边界补丁”算子：它不属于 attention 数学核心，却处在 SGLang prefill MHA 的真实 critical path 上；单独建模后，AIC 的 granular prefill MLA 路径才能同时覆盖 `kv_b_proj` 的 latent-KV 恢复、K concat 准备、FA/MHA attention 与 `o_proj`。在大 prefix 用例中，纳入该算子后实机与 AIC 的误差显著下降，也从侧面验证了这个边界划分是必要的。
