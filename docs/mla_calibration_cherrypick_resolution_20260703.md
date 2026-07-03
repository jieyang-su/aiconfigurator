# FriskFreeman MLA calibration 分支合入 dev 说明 2026-07-03

## 合并目标

本次目标是把远程 `origin/mla_calibration` 中 FriskFreeman 的 MLA calibration 相关提交，以 cherry-pick 方式合入 `dev`。合并原则是：

- 以 `dev` 当前代码框架为主。
- 保留 FriskFreeman 提交的功能语义和提交作者信息。
- 兼容旧 SGLang `0.5.9` 数据，恢复旧数据表。
- 不恢复旧的单文件 `src/aiconfigurator/sdk/operations.py` 架构。
- 对旧分支中改 `operations.py` 的内容，迁移到当前 `src/aiconfigurator/sdk/operations/` 分模块架构。

## Cherry-pick 提交列表

从 `origin/mla_calibration` 按提交顺序 cherry-pick 以下 FriskFreeman 提交：

1. `402d91b` chore: transplant self docs onto v0.9.0
2. `4fa8326` fix sglang mla prefill collection
3. `ab77840` feat: add sdk support for sglang mla concat k
4. `43f0c26` fix sdk deepseek mla path selection
5. `418ab3d` docs: add mla calibration notes
6. `7270927` fix deepseek mla module table lookup
7. `e6147e4` fix sdk: account for DeepSeek context kv_b_proj prefix tokens

临时分支为 `cherrypick_frisk_mla_to_dev`，基线是 `origin/dev`。

## 冲突解决原则

### 1. Collector 与旧 0.5.9 数据

`4fa8326` 在 collector MLA 采集和旧数据表上与当前 `dev` 有差异。

解决方式：

- `collector/sglang/collect_mla.py` 使用 FriskFreeman 兼容 SGLang `0.5.9` 的采集逻辑。
- 保留当前 `dev` 中已有的说明性 docstring。
- 恢复旧数据表：
  - `src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.9/context_mla_perf.txt`
  - `src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.9/generation_mla_perf.txt`
  - `src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.9/mla_concat_k_perf.txt`

这样可以继续兼容旧 `v0.5.9` 数据，同时不影响新数据目录。

### 2. `MLAConcatK` 迁移

`ab77840` 原分支把 `MLAConcatK` 写在旧的 `src/aiconfigurator/sdk/operations.py` 中，但当前 `dev` 已经迁移为 `operations/` 分模块架构。

解决方式：

- 不恢复旧 `operations.py`。
- 新增 `MLAConcatK` 到 `src/aiconfigurator/sdk/operations/mla.py`。
- 在 `src/aiconfigurator/sdk/operations/__init__.py` 中导出 `MLAConcatK`。
- `PerfDataFilename` 新增 `mla_concat_k = "mla_concat_k_perf.parquet"`。
- 保留当前 base loader 的 `.parquet` 到 `.txt` fallback，因此旧 `mla_concat_k_perf.txt` 仍可读取。
- `PerfDatabase.query_mla_concat_k` 只保留薄转发，具体查询逻辑归属 `MLAConcatK._query_mla_concat_k_table`。
- 在 DeepSeek SGLang context granular path 中插入 `ops.MLAConcatK("context_mla_concat_k", ...)`。

### 3. DeepSeek MLA 路径选择

`43f0c26` 原提交新增 `PrefixConditionalOp` 并修改 DeepSeek MLA path：

- `prefix=0` 使用 module-level MLA。
- `prefix>0` 使用 granular MLA path，让 prefix 修正在 attention kernel 层处理。
- generation path 固定使用 module-level MLA。

解决方式：

- 不恢复旧 `operations.py`。
- 将 `PrefixConditionalOp` 放入当前 composite op 文件 `src/aiconfigurator/sdk/operations/overlap.py`。
- 在 `operations/__init__.py` 导出 `PrefixConditionalOp`。
- 保留 FriskFreeman 对 `DeepSeekModel` 的路径调整：
  - context 先单独计 `context_downscale_gemm`。
  - `context_mla_block` 使用 `PrefixConditionalOp`。
  - generation 先单独计 `generation_downscale_gemm`，再走 module-level MLA。
- 对应单测合并到 `tests/unit/sdk/database/test_fallback_op.py`。

### 4. SGLang 普通 DeepSeek module table lookup

`7270927` 把普通 DeepSeek SGLang 的 module-level MLA lookup 指向现有 WideEP MLA 表。

解决方式：

- 保留自动合并到 `src/aiconfigurator/sdk/models/deepseek.py` 的逻辑。
- `context_mla_module` 使用 `WideEPContextMLA` 包装器。
- `generation_mla_module` 使用 `WideEPGenerationMLA` 包装器。
- 使用 `FMHAQuantMode.fp8_block` 和 `attention_backend` 对齐 SGLang MLA module 表。
- 测试文件 import 冲突按并集解决：保留 `dev` 中已有的 `ops`、模型类、`PerformanceResult`，同时加入 FriskFreeman 新增的 `operations` 模块导入。

### 5. Context KV B projection prefix token 修正

`e6147e4` 原提交新增 `ContextKVBProjGEMM`，用于修正 DeepSeek context `kv_b_proj` 在 prefix-cache 场景下的 token 数。

解决方式：

- 不恢复旧 `operations.py`。
- 将 `ContextKVBProjGEMM` 迁移到当前 `src/aiconfigurator/sdk/operations/gemm.py`。
- 在 `operations/__init__.py` 导出 `ContextKVBProjGEMM`。
- `ContextKVBProjGEMM` 继承 `GEMM`，仅在 query 前把 `x` 从 fresh tokens 修正为 `fresh_tokens + prefix_tokens`。
- `prefix` 支持标量和 list/tuple 两种形式。
- DeepSeek 普通 path 和 TRT-LLM WideEP path 中的 `context_kv_b_proj_gemm` 都替换为 `ops.ContextKVBProjGEMM`。

## 验证情况

已执行：

- `git diff --check` 通过。
- 关键 Python 文件 `py_compile` 通过，包括：
  - `src/aiconfigurator/sdk/common.py`
  - `src/aiconfigurator/sdk/models/deepseek.py`
  - `src/aiconfigurator/sdk/operations/mla.py`
  - `src/aiconfigurator/sdk/operations/gemm.py`
  - `src/aiconfigurator/sdk/operations/overlap.py`
  - `src/aiconfigurator/sdk/operations/__init__.py`
  - `src/aiconfigurator/sdk/perf_database.py`
  - 相关测试文件

尝试执行 `tests/unit/sdk/database/test_fallback_op.py` 时，当前临时 worktree 没有安装 editable package metadata，报错为 `PackageNotFoundError: No package metadata was found for aiconfigurator`。这是测试环境安装状态问题，不是本次代码语法错误。

## 当前结果

最终临时分支 `cherrypick_frisk_mla_to_dev` 在 `origin/dev` 基础上新增 7 个 FriskFreeman cherry-pick 提交，并额外补充本说明文档。核心代码路径已经按当前 `dev` 分模块 SDK 架构完成迁移。
