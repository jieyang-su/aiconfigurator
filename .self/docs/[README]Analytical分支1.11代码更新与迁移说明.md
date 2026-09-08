# Analytical 分支 1.11 代码更新与迁移说明

## 文档定位

本文将当前工作区源码状态视为 `analytical/1.11`，继续以已归档并推送的
`analytical/1.1`（HEAD `7c63e9d7`）为基线，整理两者之间的代码和配置差异。
本文用于后续代码迁移、功能合并、版本归档和回归测试，不代表当前工作区已经完成
新的 Git 提交；本次归档提交将继续落在现有 `analytical/1.1` 分支上，不另建分支。

本轮差异同时包含已追踪文件的修改和尚未追踪的新增 AIC 文件。`.self` 下的实验目录、
结果数据和 Web 资产属于工作材料，不应因为本文的源码变更说明而自动纳入代码提交。

## 变更性质总览

### Analytical 模式专属或主要服务于 Analytical 的改动

1. 新增通信 `analytical` 子模式及其 KernelSim 代理模型。
2. 更新 W8A16 GEMM 的兼容性 proxy，使其不再直接乐观复用 BF16 参数。
3. 修正 MiniMax MSA Analytical 的稀疏 selected-pair 计算，并加入 selected-block
   attention 的专用估算入口。
4. 扩展 Analytical 的 CLI、Task 和 operation 路径，使通信模式可以选择
   `analytical`、旧 `empirical` 或 `silicon`。
5. 增加和补充 Analytical、KernelSim、MSA、国产系统及通信模型的单元测试和风险说明。

这些改动的共同特点是：大多用于无完整 Silicon 数据时的理论估算，不能被解释为新硬件
的实测结果。部分改动位于共享 operation 或 KernelSim 入口，因此会影响调用 Analytical
路径的上层模型，但不应改变显式使用 `SOL`、`EMPIRICAL` 或 `SILICON` 的既有语义。

### 更广泛的 AIC 功能、数据和兼容性改动

1. Rust PerfDatabase 增加 `nvfp4` 缺失时到 `w4a16_mxfp4` 的数据回退。
2. 增加 MiniMax M3-MXFP8 模型配置、默认模型注册和量化算法归一化。
3. 增加国产 GPU 系统 YAML、空数据目录和新的硬件容量/带宽描述。
4. 更新 KLX-M300 兼容 system ID 的超节点容量和机内带宽元数据。
5. 扩展针对模型配置、国产系统、Task、CLI、MoE 数据库和 KernelSim 的测试覆盖。

这些内容并非全部属于 Analytical 专属能力。尤其是 Rust 数据库回退和模型量化归一化
可能被其他数据库模式或模型解析路径复用，迁移时不能只复制 Analytical operation 文件。

## Analytical 专属改动

### 1. 通信 Analytical 代理模型

新增目录：

```text
aic-core/src/aiconfigurator_core/sdk/kernelsim/communication/
```

核心模型位于 `communication/model.py`，当前固定采用：

```text
latency = 7 us + SOL_latency / 0.75
```

其中 `SOL_latency` 仍由 AIC 原有通信公式计算，包含消息大小、collective 类型、参与
rank 数、ring 因子、通信 dtype 和已经选择的拓扑带宽；新增模型只负责增加启动项并使用
基于 Hopper 机内通信数据得到的效率系数。SOL 为零时不支付启动项。

模型提供 `CommunicationParameters`、`CommunicationEstimate` 和版本标识，便于后续
替换参数或记录 provenance，但当前没有按硬件、collective、rank 或 dtype 分别拟合参数。
国产超节点和机外通信仍然只是通过底层 SOL 的拓扑带宽和统一系数进行估算。

对应接入位置：

- `sdk/kernelsim/analytical.py`：暴露 `communication_latency_ms()` 和配置校验；
- `sdk/operations/communication.py`：接入 `CustomAllReduce`、`NCCL` 和 `P2P`；
- `sdk/operations/moe.py`：接入 MoE dispatch/combine 及 WideEP 相关通信查询；
- `src/aiconfigurator/cli/main.py`：允许命令行选择 `--analytical-communication-mode analytical`；
- `src/aiconfigurator/sdk/task_v2.py` 和 CLI API：传播并说明该选项。

模式语义如下：

| 上层数据库模式 | communication mode | 实际路径 |
|---|---|---|
| `ANALYTICAL` | `analytical` | 7 us + SOL / 0.75 代理 |
| `ANALYTICAL` | `empirical` | 原有 SOL / 0.8 经验公式 |
| `ANALYTICAL` | `silicon` | 原有通信实测表及其插值/外推 |
| `SOL`、`EMPIRICAL`、`SILICON` 等 | 不适用 | 保持原有独立模式行为 |

迁移时必须同时迁移 AnalyticalConfig 校验、operation 分派和测试，否则配置可以被 CLI
接受但执行时仍会错误落到旧路径。

### 2. W8A16 GEMM proxy 更新

`sdk/kernelsim/gemm/model.py` 中的 W8A16 模型从早期的 BF16 参数转移模型更新为
`w8a16_int8wo_bf16_fused_transition_proxy_v4`。当前标准参数为：

```text
t_launch_us   = 5.0
eta_mem       = 0.70
eta_compute   = 0.80
rho_transition = 1.5
```

模型仍然使用 BF16 activation/output 和 BF16 计算峰值，但主体 GEMM 的权重访存按照
INT8 权重计算；不再把量化视为一个独立 kernel，也不额外计入独立 `quant_bytes`。
原因是当前目标 SGLang W8A16 路径将权重反量化和 GEMM 融合在 kernel 内，不能直接照搬
原始 FP8 collector 中“前序 quant + GEMM”合并采集的字节流量语义。

新增 transition 项是经验性保守修正，用于表达融合反量化、tile 和小规模矩阵调度造成的
额外损失，不代表物理上存在一个独立可观测的通信或量化阶段。`low`、`standard`、
`high` 三档用于工程敏感性分析，不是统计置信区间；`precise` 当前有意与 standard
相同，也不代表已经有纯 W8A16 实测拟合数据。

该 proxy 只适用于 Analytical 估算，不应标记为 Silicon，也不应直接推广到其他
backend、small-M、不同 scale granularity、特殊 packing 或非 NVIDIA 硬件的真实性能。
W4A16 MoE 等其他兼容路径不因本次 GEMM 模型更新而自动获得新的拟合模型。

### 3. MiniMax MSA 稀疏 attention 估算

`sdk/operations/msa.py` 新增 `_msa_selected_pairs()`，统一计算 MSA 在 prefill 和
decode 下的因果 selected-token pair 数。该函数同时被 MSA 的 SOL 近似路径和 Analytical
路径使用，因此这部分是共享语义修正，而不是完全隔离的 Analytical 代码。

Analytical 路径不再使用普通 dense FA 的 `kv_length` 近似，而调用
`msa_sparse_attention_latency_ms()`。该入口：

- 使用真实 selected pair 数计算 QK 和 AV 的工作量；
- 按 query head、KV head、head dimension 和 value dimension 计算矩阵工作量；
- 按 KV cache 元素字节数估算 selected block 的 HBM gather 流量；
- 计入 block index 元数据流量；
- 复用现有 Analytical 的计算效率、访存效率和固定启动项 profile。

当前仍是假设性模型，随机 block gather、cache reuse、sparse task wave、split-K merge、
跨 GPU 行为和不同 MSA backend 没有重新校准。原有 MSA Index 的 H=4 Triton 风格模型
仍是专用模型，不能因为数学上与 DSA index 相似就替换为 DSA 模型。

## 通用 AIC 改动

### 1. MoE PerfDatabase 的量化回退

Rust 文件：

```text
aic-core/rust/aiconfigurator-core/src/perf_database/moe.rs
```

在常规查询和可选查询中，当请求 `nvfp4` 但数据库没有对应数据时，尝试查询
`w4a16_mxfp4`。该行为是数据库兼容性回退，不是 Analytical KernelSim proxy。

迁移时需要关注：

- Python 和 Rust 查询结果的 parity；
- fallback 后的量化名称、数据分布和错误信息；
- 有原生 `nvfp4` 数据时必须优先使用原生数据；
- fallback 数据不能被报告为原生 NVFP4 Silicon 数据。

### 2. MiniMax M3-MXFP8 配置和量化解析

新增配置：

```text
aic-core/src/aiconfigurator_core/model_configs/MiniMaxAI--MiniMax-M3-MXFP8_config.json
```

同时在 `sdk/common.py` 注册模型，在 `sdk/utils.py` 将 `mxfp8` 归一化为已有的
`fp8_block` 代理。该配置保留 M3 的 dense/MSA/MoE 结构，只改变 checkpoint 的量化
描述；KernelSim 没有单独的 MXFP8 拟合模型，因此这种映射不能理解为 MXFP8 kernel
已经被独立校准。

### 3. 国产 GPU 系统配置

新增系统配置包括：

```text
_dom_ascend_950dt_1024.yaml
_dom_br200_64.yaml
_dom_br200_128.yaml
_dom_klx_p800_256.yaml
_dom_zte_128.yaml
```

每个系统同时有对应的空 `systems/data/.../.gitkeep` 目录。新增 YAML 使用国产硬件
标识和显式算力能力字段，微架构字段仍可能借用 NVIDIA 代理值供 KernelSim 使用；这些
字段不表示国产硬件具备对应 NVIDIA ISA 或 backend。

已有 `_dom_klx_m300_512.yaml` 的元数据也有变化：system ID 继续保留历史的 `512` 后缀，
但实际超节点容量为 256 GPU，机内带宽更新为 600 GB/s。迁移时不能只按文件名推断
有效容量。

这些系统当前没有对应 Silicon 性能数据库，主要服务于 Analytical/无卡估算。其通信、
显存、SM、L2 和向量参数都应在结果中保留“代理/估算”语义。

### 4. CLI、Task 和测试覆盖

除通信 CLI 参数外，`Task v2` 和 CLI API 文档补充了三种 Analytical communication
path 的说明。新增或更新的测试覆盖：

- Analytical communication proxy、旧 empirical 公式和 silicon 分派；
- W8A16 GEMM 的参数档位、transition 和零 quant traffic；
- MSA selected pair 和 sparse attention；
- 国产系统容量、带宽、能力和代理 SM 字段；
- M3-MXFP8 配置和 `mxfp8` 归一化；
- Task v2、CLI 参数和数据库查询。

## 当前不应误归类的内容

### Collector 文件与待办

`collector/sglang/collect_moe.py` 和相关 collector 测试在工作区时间戳上较新，但截至
当前检查，其工作区内容 hash 与 `analytical/1.1` HEAD 一致，Git 没有形成可提交的内容
差异。因此本轮不能把它们描述为新的 collector 功能；时间戳变化可能来自实验环境或
文件重新生成。

当前仍有明确 TODO：`collector/sglang/collect_moe.py` 的 Triton 后端存在错误的
clamp 行为，尤其需要继续核对 EP>1、远端 expert ID 和本地权重过滤语义。该问题不在
本次 1.11 归档中修复；后续版本应先修复并补充针对不同 EP/后端的实机回归，再决定是否
将相关采集数据纳入模型拟合或数据库。

### `.self` 实验与 Web

`.self/analytical-mode/`、`.self/domestic-gpu/` 和 `.self/moe-ep-clamp-repro/` 中的
Pareto 运行、通信代理实验、W8A16 实验、collector 快照、日志、CSV、Parquet、PNG、
分析脚本和 Pareto Web 页面，均属于本地研究资产，不是本轮正式 AIC 源码差异。

当前 `.self` 约 670 MB，其中日志约 238 MB、CSV 约 229 MB、Parquet 约 95 MB；Web
页面主要位于 `.self/domestic-gpu/pareto-web/`，其页面和数据没有被 AIC 主仓库追踪。
后续如需归档，应单独选择必要的结果文档，不应使用 `git add -A` 将整棵实验目录提交。

## 锁文件与生成物

根目录 `uv.lock` 的大规模差异主要是 Python 官方源 URL 被替换为清华镜像 URL，当前
没有证据表明这是功能依赖升级。`aic-core/uv.lock` 是本地新增锁文件，也应与功能代码
分开判断。

`.venv/`、`.so`、`target/`、`__pycache__/`、`.pytest_cache/` 和 `.ruff_cache/` 属于
本地环境或构建生成物，不应作为 1.11 功能变更归档。

## 迁移建议

### 从 1.1 迁移到 1.11

推荐按以下依赖顺序迁移：

1. 先迁移 `kernelsim/communication` 包和 `AnalyticalConfig` 的通信模式校验。
2. 再迁移 `communication.py`、`moe.py` 的 operation 分派，以及 CLI/Task 参数传播。
3. 迁移 W8A16 GEMM proxy 和 MSA sparse attention 入口，同时迁移对应单元测试。
4. 迁移 Rust MoE `nvfp4` fallback、M3-MXFP8 配置和量化归一化。
5. 迁移国产 YAML 及空数据目录，并确认目标分支已有相同的 system loader 语义。
6. 最后选择性迁移 `.self/docs` 中的结果性说明，不迁移实验结果目录和 Web 运行资产。

不要整体覆盖 `sdk/operations/communication.py`、`moe.py` 或 `msa.py`。这些文件包含
跨模式共享逻辑，应以目标分支版本为基线，按函数和分派分支逐项合并。

### 回归检查

至少应执行：

- YAML 解析和国产 system loader 测试；
- KernelSim communication、GEMM W8A16、MSA 测试；
- Analytical database/operation、Task v2 和 CLI 测试；
- Rust MoE 数据库测试和 Python/Rust parity；
- `ruff check`、`git diff --check`；
- 显式验证 `ANALYTICAL + analytical`、`ANALYTICAL + empirical`、
  `ANALYTICAL + silicon` 三条通信路径的结果来源标识。

## 风险摘要

- 通信 `7 us / 0.75` 是以 Hopper 机内数据为主的统一代理，不是逐硬件校准结果。
- W8A16 GEMM 没有纯 W8A16 实测拟合数据，transition 是经验性保守项。
- MSA sparse attention 使用 selected-pair 和 HBM gather 抽象，未覆盖真实稀疏 kernel 的
  cache、wave、split-K 和 backend 差异。
- `nvfp4 -> w4a16_mxfp4` 是数据回退，不能冒充原生 NVFP4 性能。
- 国产系统的 SM、L2、启动和部分通信参数是代理值，结果只适合作为 Analytical 估算、
  相对排序和敏感性分析。
- `uv.lock`、实验日志和 Web 数据的本地变化不能作为功能已经验证的依据。
