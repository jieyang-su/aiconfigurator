# SDK DeepSeek MLA 路径校准说明

## 背景

本轮修改承接前序 SGLang 0.5.9 H100 实采校准工作：`context_mla` 单算子 prefill 形状错误已修复并重新采集，随后用 Nsight 实机数据对比了 AIC 的单算子口径与 module 口径。

对普通 DeepSeek V3/R1 的 MLA 路径，得到的可执行结论是：

- prefill/context 阶段，当 `prefix == 0` 时，AIC 应使用 MLA module 级数据。
- prefill/context 阶段，当 `prefix > 0` 时，AIC 应使用 granular 单算子路径，由 attention/MLA 单算子查表承担 prefix correction。
- decode/generation 阶段，应使用 MLA module 级数据。

旧代码通过 `FallbackOp(primary=MLAModule, fallback=[...])` 决定路径，实际是以“module 数据是否可查到”作为隐式分流条件。这会把语义规则和数据可用性兜底混在一起，也会在 module 路径和 fallback 路径之间产生 `qkv_a/downscale` 边界不一致的问题。

## 修改范围

本提交只修改 SDK 仿真侧普通 `DeepSeekModel` 的 MLA 相关 ops 路径，不修改 WideEP 模型本身，也不修改 collector 或 systems/data 数据。

核心文件：

- `src/aiconfigurator/sdk/models/deepseek.py`
- `src/aiconfigurator/sdk/operations.py`
- `tests/unit/sdk/database/test_fallback_op.py`

## 行为变化

### Context / Prefill

普通 `DeepSeekModel` 的 context MLA block 改为显式按 `prefix` 分流：

- `prefix == 0`：`context_downscale_gemm + context_mla_module`
- `prefix > 0`：`context_downscale_gemm + q_b_proj + kv_b_proj + concat_k(sglang) + context_attention + proj`

其中 `context_downscale_gemm` 被提升为共享 op，因为 SGLang module collector 边界不包含 qkv_a/downscale；这样 module 路径和 granular 路径都会且只会计入一次该算子。

### Decode / Generation

普通 `DeepSeekModel` 的 generation MLA block 改为：

- `generation_downscale_gemm + generation_mla_module`

旧的 generation granular fallback 路径被注释说明替代，不再作为默认执行路径。原因是当前实机/AIC 对比结论显示 decode 与 module 级 MLA 数据口径对应更稳定。

### 新增 Operation

新增 `PrefixConditionalOp`，仅按 `prefix == 0` 或 `prefix > 0` 选择一组 ops。它不同于 `FallbackOp`：

- `PrefixConditionalOp` 表达请求语义分流。
- `FallbackOp` 表达性能数据缺失时的兜底。

这次修改刻意避免继续用数据可用性来决定 DeepSeek MLA 的语义路径。

## 与前序采集校准的关系

前序 collector/systems/data 修改解决的是底层数据口径问题：

- `context_mla` 单算子 prefill 形状修正后，与实机 prefill attention kernel 口径更一致。
- module 数据用于无 prefix 的 prefill module 口径和 decode module 口径。

本次 SDK 修改解决的是模型仿真路径选择问题：

- prefix prefill 不再误走 module 路径。
- decode 不再依赖单算子 fallback 路径。
- `qkv_a/downscale` 不再因 module/fallback 边界差异出现漏计或重复语义。

## 兼容性与风险

- 该修改面向普通 `DeepSeekModel`；WideEPDeepSeekModel 和 TrtLLM WideEPDeepSeekModel 未改动。
- 如果某个系统缺少 `mla_context_module` 或 `mla_generation_module` 数据，普通 DeepSeek 的 `prefix == 0` context 或 generation 路径将更早暴露数据缺失，而不是静默 fallback 到 granular 估计。这是有意为之，便于发现数据口径不完整。
- vLLM 分支在 `prefix > 0` context granular 路径仍保留原有 `ContextAttention` 逻辑；`prefix == 0` context 和 generation 现在会走普通 DeepSeek 的 module 规则，后续如需为 vLLM 保留独立 module/granular 行为，需要另行评估。

## 验证

已执行：

```bash
python -m py_compile src/aiconfigurator/sdk/models/deepseek.py \
  src/aiconfigurator/sdk/operations.py \
  tests/unit/sdk/database/test_fallback_op.py
```

以及：

```bash
source /home/ai_lab/fjw/miniforge3/etc/profile.d/conda.sh
conda activate ljc01
PYTHONPATH=src python -m pytest -q tests/unit/sdk/database/test_fallback_op.py
```

结果：`19 passed`。

## Review 重点

- 确认普通 DeepSeek 的 `prefix == 0` / `prefix > 0` 规则是否符合后续 benchmark 对比口径。
- 确认 `context_downscale_gemm` 与 `generation_downscale_gemm` 作为 module 外共享 op 是否与当前 collector 边界一致。
- 若后续支持更多 backend 或 vLLM 的 DeepSeek MLA module 数据，需要重新审视 `DeepSeekModel` 中 backend 分支是否应进一步拆分。
