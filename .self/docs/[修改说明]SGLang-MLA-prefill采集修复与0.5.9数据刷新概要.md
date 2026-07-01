# SGLang MLA Prefill 采集修复与 0.5.9 数据刷新概要

## 背景与目标

本轮修复针对 `collector/sglang/collect_mla.py` 中 `mla_context` 单算子采集的形状错误。问题根因是旧代码在 H100/FA3 backend 下，prefill 阶段仍按 decode/absorbed MLA 形态构造输入，即 latent KV 维度 `512 + 64` 且 KV head 为 `1`；但 SGLang DeepSeek V3 实际 prefill 路径会走非 absorbed MHA 形态，核心 attention 输入应为：

- `q/k head_dim = qk_nope_head_dim + qk_rope_head_dim = 128 + 64 = 192`
- `v_head_dim = 128`
- `num_kv_heads = local_num_heads`
- prefill no-prefix 单算子采集时强制 `attn_attend_prefix_cache=False`，避免落回 MLA 分支

同时，实机 nsys 中 prefill attention core 前存在 `_concat_and_cast_mha_k` 对应的 K 拼接/转换 kernel。该部分不属于旧 `collect_mla` 的 attention core 计时边界，因此本轮新增了独立 `mla_concat_k` 单算子采集，输出 `mla_concat_k_perf.txt`。

## 核心代码改动

提交：`4fa83267 fix sglang mla prefill collection`

提交文件：

- `collector/sglang/collect_mla.py`
- `collector/sglang/registry.py`
- `collector/registry_types.py`
- `src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.9/context_mla_perf.txt`
- `src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.9/generation_mla_perf.txt`
- `src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.9/mla_concat_k_perf.txt`

主要 diff：

- `collect_mla.py` 的 `__compat__` 从 `sglang>=0.5.10rc0` 调整为 `sglang>=0.5.9`，使 0.5.9 Docker 正式采集入口可用。
- `run_mla()` 在 context/prefill 且非 Triton 后端时，强制使用 DeepSeek MHA 维度：`head_dim=192`、`v_head_dim=128`、`num_kv_heads=local_num_heads`。
- 新增 `_validate_prefill_mha_or_raise()`，对 prefill 采集做 fail-closed 校验，若无法确认处于 MHA 形态则直接报错，避免静默采到 absorbed MLA 数据。
- FP8 KV cache 情况下，prefill MHA 的 `k` dtype 按 SGLang `_concat_and_cast_mha_k` 行为对齐到 KV pool dtype；FA3 内部再按需要 cast 回 `q.dtype`。
- 新增 `get_mla_concat_k_test_cases()` 与 `run_mla_concat_k()`，复用原 context MLA 的 batch/seq/head/tp sweep。
- `mla_concat_k` 按 SGLang 0.5.9 源码分支区分 `kernel_source`：
  - `concat_mla_k`：local heads 为 128 时使用 SGLang 专用 CUDA kernel。
  - `concat_and_cast_mha_k_triton`：TP 后 local heads 变小等场景使用 Triton fallback。
- `registry.py` 新增 `mla_concat_k` op。
- `registry_types.py` 新增 `PerfFile.MLA_CONCAT_K = "mla_concat_k_perf.txt"`。

## 采集过程

运行环境：

- Docker 镜像：`booleimg.myaddr.io/lmsysorg/sglang:v0.5.9`
- GPU：H100 SXM，使用后 4 张空闲卡，即容器映射 `device=4,5,6,7`
- 采集入口：`collector/collect.py --backend sglang --ops mla_context mla_generation mla_concat_k`

采集前 smoke：

- 手动调用 `run_mla()` 验证 context BF16、context FP8、generation BF16。
- 手动调用 `run_mla_concat_k()` 验证 `concat_mla_k` 和 `concat_and_cast_mha_k_triton` 两条分支。
- registry 入口 `--limit 2` 验证通过。

正式采集结果：

- `mla_context`：生成 3080 个 test case，完成成功。
- `mla_generation`：生成 4648 个 test case，完成成功。
- `mla_concat_k`：生成 1540 个 test case，完成成功。
- `collection_summary_sglang.json` 中 `total_errors = 0`。

## 新数据概况

落盘目录：`src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.9/`

数据文件：

- `context_mla_perf.txt`：3080 rows，`kernel_source = flash_attention`
- `generation_mla_perf.txt`：4648 rows，`kernel_source = flash_attention`
- `mla_concat_k_perf.txt`：1540 rows，`kernel_source` 覆盖 `concat_mla_k` 与 `concat_and_cast_mha_k_triton`

CSV 校验：

- 三个文件表头符合预期。
- 行数与 test case 生成数量一致。
- 所有 latency 均为正数。
- 按各自 key 去重后无重复项。
- `context_mla_perf.txt` 中所有行均为 `step=0`、`mla_dtype=bfloat16`，KV dtype 覆盖 `bfloat16` 与 `fp8`。

## 与 0.5.10 旧数据的快速对照

由于本轮只刷新了 `0.5.9`，`0.5.10/context_mla_perf.txt` 仍保留旧的错误 MLA shape 采集结果。对两个版本按 `mla_dtype, kv_cache_dtype, num_heads, batch_size, isl, tp_size, step` join 后：

- `0.5.9` 行数：3080
- `0.5.10` 行数：3080
- key 集合完全一致：交集 3080，差异 key 为 0
- overall `0.5.10 / 0.5.9` latency median：约 `2.45x`
- overall mean：约 `2.48x`
- `batch_size * isl >= 4096` 的大 shape 中，median：约 `3.52x`，mean：约 `3.32x`

这说明两个版本的数据覆盖形状一致，但未修正的 0.5.10 由于仍按错误 absorbed MLA/latent shape 采集，时延显著偏大；刷新后的 0.5.9 更符合 SGLang prefill MHA 实际执行路径。

## 提交与范围控制

本轮已创建本地提交：

```text
4fa83267 fix sglang mla prefill collection
```

提交只包含 collector 修复和 H100 0.5.9 目标数据文件。工作树中原本存在的 `.self/`、文档、`bench_data/`、其他版本 data 目录、`uv.lock` 等未纳入该提交。

