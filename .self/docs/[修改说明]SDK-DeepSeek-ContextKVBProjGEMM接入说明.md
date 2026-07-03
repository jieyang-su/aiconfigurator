# SDK DeepSeek ContextKVBProjGEMM 接入说明

## 背景

DeepSeek V3/R1 的 MLA 架构在 KV cache 中保存的是 latent KV，而不是已经展开到每个 head 的 full K/V。SGLang prefill 走 MHA/FA3 路径时，attention 需要消费完整的 `K/V`，因此 `kv_b_proj` 不只作用于本轮 fresh token，也会作用于从 prefix cache 命中的 latent KV token。

旧 AIC SDK 中 `context_kv_b_proj_gemm` 使用普通 `GEMM`：

- 后端传入的 `x` 是本轮 fresh token 数，即 `batch_size * fresh_len`。
- 普通 `GEMM.query()` 直接用 `x` 作为 GEMM 的 `m` 维查表。
- 对于 prefix 较大的 prefill 请求，这会严重低估 `kv_b_proj` 的实际输入 token 数。

实机 nsys 结果显示，大 prefix 场景中 `kv_b_proj` 的真实 kernel 时间明显随 `fresh + prefix` 总 token 增长，而不是只随 fresh token 增长。因此需要在 SDK 层给 DeepSeek context `kv_b_proj` 增加专门的查询语义。

## SGLang 侧语义

在 SGLang DeepSeek prefill MHA 路径中，关键流程位于 `sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`：

- 当前 fresh token 会先生成 latent KV，并写入/更新 MLA KV cache。
- 若存在 prefix cache 命中，MHA one-shot 或 chunked-prefix 路径会从 KV pool 中取出 prefix 对应的 latent KV。
- `kv_b_proj(kv_a)` 将 latent KV 从 `kv_lora_rank=512` 展开为每个 local head 的 `k_nope + v`。
- 随后再将 `k_nope` 与 RoPE 部分 `k_pe/k_rope` 拼成 full K，传给 `attn_mha`。

因此，对于未触发 chunked-kv 的 MHA one-shot prefill，单次 `kv_b_proj` 的 `m` 维应近似为：

```text
m = sum_i(prefix_len_i + fresh_len_i)
```

而旧 SDK 使用的是：

```text
m = sum_i(fresh_len_i)
```

如果未来触发 `MHA_CHUNKED_KV`，更严格的语义应按 SGLang chunk 策略拆成多次 `kv_b_proj` 查询：fresh 段一次，加上每个 prefix chunk 多次。当前已对超长 chunked-kv 做运行侧拦截/提示；本次 SDK 修复先覆盖当前正式对比中未触发 chunked-kv 的 one-shot/full-K 场景。

## 本次 SDK 改动

### `src/aiconfigurator/sdk/operations.py`

新增 `ContextKVBProjGEMM(GEMM)`：

- 继承普通 `GEMM`，复用底层 `PerfDatabase.query_gemm()` 和已有 GEMM 数据表。
- 在 `query()` 入口复制 kwargs，并将 `x` 从 fresh tokens 修正为 fresh + prefix tokens。
- `prefix` 支持两种形式：
  - 标量：按 `batch_size * prefix` 转成 prefix token 数。
  - list/tuple：按 `sum(prefix)` 转成 prefix token 数，便于后续支持变长请求。

核心逻辑：

```python
fresh_tokens = int(corrected_kwargs.get("x") or 0)
corrected_kwargs["x"] = fresh_tokens + self._prefix_tokens_from_kwargs(**corrected_kwargs)
return super().query(database, **corrected_kwargs)
```

这个设计刻意不修改 `PerfDatabase`：`kv_b_proj` 本质仍是 GEMM，特殊点只是 DeepSeek MLA context 阶段的输入 token 语义不同。把修正放在 `Operation` 层，可以复用所有现有 GEMM interpolation、energy、source 标记和 quant mode 逻辑。

### `src/aiconfigurator/sdk/models/deepseek.py`

将 DeepSeek context 路径中的 `context_kv_b_proj_gemm` 从普通 `ops.GEMM` 替换为 `ops.ContextKVBProjGEMM`。

已覆盖位置：

- 普通 `DeepSeekModel` 的 context prefix/granular MLA 路径。
- `TrtllmWideEPDeepSeekModel` 的 context 路径。

未改动位置：

- generation/decode 路径：decode 的 FlashMLA/MQA 语义不同，不使用这个 context full-K prefill 修正。
- 底层 GEMM 数据库：仍查询 `gemm_perf.txt`，不新增 perf file。

## 与 MLAConcatK 的关系

`ContextKVBProjGEMM` 与 `MLAConcatK` 是两个相邻但不同的修复点：

- `ContextKVBProjGEMM` 修复 `kv_b_proj` GEMM 的输入 token 数，从 fresh-only 改为 fresh+prefix。
- `MLAConcatK` 补上 `kv_b_proj` 后、`attn_mha` 前的 K 拼接 kernel。

两者都属于 SGLang DeepSeek prefill MHA 路径的 granular fallback 口径，但边界不同。`kv_b_proj` 是 GEMM 展开 latent KV；`concat_k` 是数据整理/拼接 kernel。完整对齐时二者都需要计入。

## 验证与收益

### 单算子孤立验证

分析目录：

```text
bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/analysis/kv_b_proj_prefix_token_sensitivity
```

该分析用 H100/SGLang 实机 `kv_b_proj` kernel-sum 作为 baseline，对比三种 AIC 查询语义：

- `old_fresh_only`: 普通 GEMM 旧行为，`m=sum(fresh_lens)`。
- `direct_total_tokens`: 本次修复后的 prefix-aware 行为，`m=sum(prefix_lens + fresh_lens)`。
- `sglang_chunk_policy`: 按 SGLang chunked-kv 策略拆分；当前用例未触发 chunked-kv，因此与 `direct_total_tokens` 等价。

结果摘要：

| 数据集 | 旧 fresh-only MAPE | prefix-aware MAPE | 说明 |
| --- | ---: | ---: | --- |
| full refresh prefill cases | 43.91% | 5.76% | 26 个 case，覆盖无 prefix、短 prefix、大 prefix |
| large-prefix prefill cases | 92.12% | 2.28% | 8 个大 prefix case，修复收益最明显 |

典型行为：

- 旧逻辑在大 prefix 下只按 fresh token 查 `kv_b_proj`，系统性低估，平均误差接近 -92%。
- 修复后按 fresh+prefix 查表，单算子误差下降到约 2% 到 6%。
- 当前 TP1 的 `(n=32768, k=512)` 在 `gemm_perf.txt` 中仍缺直接采集 shape，因此修复后仍依赖 3D interpolation；误差剩余部分主要来自数据表覆盖而非 token 语义。

### 完整 prefill AIC 对比收益

在完整 prefill MLA kernel-envelope 对比中，重新生成了不覆盖旧成果的新目录：

- `analysis/mla_aic_compare_kernel_envelope_refresh_kvfix`
- `analysis/mla_aic_compare_large_prefix_kvfix`

只看 `kv_b_proj` 修复后的第一版结果：

| 数据集 | 旧 DeepSeek fallback MAPE | kv_b_proj 修复后 DeepSeek fallback MAPE | 旧 Rule MAPE | kv_b_proj 修复后 Rule MAPE |
| --- | ---: | ---: | ---: | ---: |
| full refresh prefill cases | 17.29% | 11.32% | 14.89% | 8.91% |
| large-prefix prefill cases | 30.61% | 12.41% | 30.61% | 12.41% |

随后补入 `MLAConcatK` 后的最终 `_kvfix` 结果：

| 数据集 | 最终 DeepSeek fallback MAPE | 最终 Rule MAPE | 说明 |
| --- | ---: | ---: | --- |
| full refresh prefill cases | 9.28% | 6.07% | 26 cases / 104 layer rows |
| large-prefix prefill cases | 2.68% | 2.68% | 8 cases / 32 layer rows |

因此，`ContextKVBProjGEMM` 解决的是主要的 token 语义低估问题；`MLAConcatK` 进一步补齐相邻前置 kernel 后，大 prefix 场景的整体 AIC/实机对齐显著改善。

## 查询示例

以单请求 `fresh=4096, prefix=32000` 为例：

```text
old x = 4096
new x = 4096 + 32000 = 36096
GEMM shape: m=36096, n=32768, k=512, quant=fp8_block
```

旧普通 GEMM 查询只会查 `m=4096`；`ContextKVBProjGEMM` 会自动把 `x` 修正为 `36096` 后再调用普通 GEMM 查询。

## 兼容性与风险

- 该修改不改变 GEMM 数据库格式，不需要新增 collector 数据文件。
- 对 `prefix=0` 的请求，`ContextKVBProjGEMM` 与普通 `GEMM` 行为一致。
- 对 `prefix` 为 list/tuple 的变长请求，按总 prefix token 数求和；这与当前 AIC GEMM 查表粒度一致，但不表达每个请求内部的 chunked-kv 拆分。
- 若真实 SGLang 运行触发 `MHA_CHUNKED_KV`，单次 `fresh+prefix` 查询会偏向 one-shot 近似。更精细的后续方案应在 SDK op 或模型层引入 SGLang chunk policy，多次查询并累加 `kv_b_proj`。
- `gemm_perf.txt` 缺少某些 DeepSeek 特殊 shape，例如 TP1 下 `512 -> 32768` 的精确采样点，当前仍依赖插值/外推；这会带来剩余误差。

## 后续建议

- 为 `context_kv_b_proj_gemm` 补齐常见 DeepSeek shape 的 GEMM 采集点，尤其是 `(n=32768/tp, k=512)`。
- 在正式支持 chunked-kv 超长场景时，把 `ContextKVBProjGEMM` 从“一次 fresh+prefix 查询”扩展为“按 SGLang chunk policy 多次查询累加”。
- 在对比图或报告中明确标记 `context_kv_b_proj_gemm` 的 policy，避免后续误以为它仍是普通 context GEMM。
