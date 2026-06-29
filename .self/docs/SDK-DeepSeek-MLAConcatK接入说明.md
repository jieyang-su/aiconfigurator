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
