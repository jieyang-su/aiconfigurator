# 新版 KernelSim 与 ANALYTICAL 模式实现差异详解

## 1. 文档目的

本次迁移不是把旧版 `kernelsim` 目录和若干 `if ANALYTICAL` 分支机械复制到新版 AIC。KernelSim 的核心公式和三档参数基本沿用既有成果，但 AIC 主干已经重构了包边界、性能数据库生命周期、operation 组织方式、任务配置和执行引擎。迁移工作的重点因此从“新增一组公式”变成“让公式遵守新版运行时契约”。

本文专门解释：

- 新旧实现有哪些结构性差异；
- 这些差异分别由新版 AIC 的哪些特性导致；
- 为什么某些旧做法在新版中不应继续使用；
- 后续扩展 ANALYTICAL 模式时需要维持哪些不变量。

## 2. 总体架构差异

### 2.1 旧版：单层 SDK 内直接集成

旧版主要代码集中在：

```text
src/aiconfigurator/sdk/
  common.py
  operations.py
  perf_database.py
  task.py
  kernelsim/
```

典型调用链为：

```text
CLI / TaskConfig
  -> 获取共享 PerfDatabase
  -> set_analytical_config(...)
  -> Operation 或 PerfDatabase.query_* 中的 ANALYTICAL 分支
  -> kernelsim adapter
```

该方式能够工作，但 analytical 参数属于可变 database 状态。同一个缓存 database 被多个任务复用时，后一个任务调用 `set_analytical_config()` 可能改变前一个任务后续查询的含义。并发 sweep、聚合/PD 双数据库以及同进程比较 `low/standard/high` 时尤其容易发生配置串扰。

### 2.2 新版：core 与上层产品接口分层

新版将估算核心拆入独立的 `aiconfigurator_core`：

```text
aic-core/src/aiconfigurator_core/sdk/
  common.py
  operations/
  perf_database.py
  kernelsim/

src/aiconfigurator/
  cli/
  sdk/task_v2.py
  sdk/kernelsim/       # 兼容导出，不承载模型实现
```

新版调用链变为：

```text
CLI / Python API
  -> Task v2
  -> get_database_view(..., analytical_config=...)
  -> immutable configured database view
  -> operation.query(database, shape)
  -> aiconfigurator_core.sdk.kernelsim adapter
  -> 算子模型
```

这决定了生产模型必须放在 core 层。若仍放在上层 `aiconfigurator.sdk.kernelsim`，core operation 将反向依赖产品包，破坏包分层，也会使独立安装 `aiconfigurator_core` 时 ANALYTICAL 不可用。上层现在只保留 `AnalyticalConfig` 的兼容导出。

## 3. Database 配置：从共享可变状态到 immutable view

这是新旧实现最重要的差异。

### 3.1 旧版行为

旧版通过类似下述过程设置参数：

```python
database.set_analytical_config(level="high", ...)
```

`PerfDatabase` 同时承担：

- 实测表缓存；
- 当前 database mode；
- analytical 参数；
- 查询和插值缓存。

因此“数据模板”和“本次任务的查询策略”没有彻底分开。

### 3.2 新版行为

新版已有 configured database view 机制。根 database 是共享数据模板，view 才保存本次查询的：

- `DatabaseMode`；
- transfer policy；
- `AnalyticalConfig`；
- 与该配置绑定的 lazy support matrix 和查询缓存。

`AnalyticalConfig` 使用 frozen dataclass，并参与 `_cached_configured_database_view()` 的 cache key：

```text
(root_database, mode, transfer_policy, analytical_config)
```

直接效果是：

- `standard` 和 `high` 得到不同 view；
- FA2 和 FA3 不共享错误结果；
- SGLang FP8 GEMM 与 DeepGEMM recipe 相互隔离；
- prefill/decode 可以拥有独立 view；
- 不需要临时修改共享 database，再在 `finally` 中恢复。

### 3.3 维护约束

以后增加 analytical 参数时，必须把字段加入 immutable `AnalyticalConfig`。不能在 operation 查询期间给 database 动态挂载可变属性，也不能把参数放进进程级全局变量。否则会绕开 view cache key，产生难以复现的跨任务污染。

## 4. Operation 组织：从大文件内联到按算子族分层

旧版大量 operation 集中在 `operations.py`，部分 analytical 分支也直接写在 `PerfDatabase.query_*()` 中。新版将逻辑拆成：

```text
operations/gemm.py
operations/attention.py
operations/mla.py
operations/moe.py
operations/communication.py
operations/overlap.py
```

`PerfDatabase.query_*()` 更多承担兼容入口和委托职责，真正的 mode dispatch 位于具体 operation。此次迁移因此采用：

```text
Operation.query()
  -> 根据 DatabaseMode 选择 silicon / empirical / analytical
  -> analytical adapter 统一做 AIC enum、单位和硬件字段转换
```

这与旧版相比有三个好处：

1. 算子形状语义靠近 operation 定义，避免 database 继续膨胀。
2. `PerformanceResult` 的 latency、energy 和 source 由同一层统一返回。
3. model operation、granular fallback 和 engine-step 都能复用同一个 query 契约。

需要注意，未建立专用 KernelSim 的小算子并不会伪装为 analytical。它们可委托已有 SOL/EMPIRICAL 公式，返回实际的 `source="empirical"` 或 `source="sol"`。这使最终 breakdown 能区分专用 KernelSim 与理论 fallback。

## 5. MLA：新版 module/granular 机制改变了接入方式

### 5.1 旧版方案的背景

旧版为 WideEP/module 缺表和 analytical 模式引入过专门的 fallback 包装。其思路是：module 查询失败，或 analytical 不适合 module 时，将 module 拆成 granular operation 序列。

旧实现还曾存在两类风险：

- 一次 primary miss 后永久禁用 primary，导致后续可命中 shape 也不再查询；
- 为迫使 primary 走 SILICON 而临时修改共享 database mode。

### 5.2 新版已有的基础能力

新版 `FallbackOp` 已支持：

- 每个 shape 独立重试 primary；
- HYBRID primary 使用 SILICON configured view，而不修改原 database；
- `silicon_primary_only=True` 时，理论模式直接使用 granular sequence；
- fallback source 和各子算子结果按 `PerformanceResult` 组合。

因此迁移时没有恢复旧 `AnalyticalFallbackOp`。DeepSeek V3 MLA 使用：

```text
downscale
+ FallbackOp(module, granular_without_downscale,
             silicon_primary_only=True)
```

ANALYTICAL 自然进入 prefix-aware granular 路径。这样 module 实测边界、granular 理论边界和 downscale 的单次计数保持一致。

### 5.3 Prefix 语义的影响

新版 module collector 已经能够采 prefix，但旧 module 数据和部分 loader 曾丢失 `step/prefix` 维度。对整个 module 乘一个 attention 二次项比例仍然不正确，因为：

- `q_b_proj/o_proj/downscale` 随 fresh token 变化；
- SGLang 的 `kv_b_proj/concat` 使用 fresh + prefix；
- attention 才按 `full^2 - prefix^2` 变化。

因此 ANALYTICAL 必须复用已完成的 granular prefix 语义修正，不能为了减少 operation 数重新走 module blanket correction。这是新版 MLA 迁移与旧 ANALYTICAL 实现相比最重要的功能边界调整。

## 6. TaskConfig v2 与前端传播

旧版使用 `TaskConfig`/factory，常在任务执行前取得 database 并调用 setter。新版 `Task` v2 是扁平用户配置对象，聚合与 PD 分离使用不同字段集合，并由 `_load_database()` 创建对应 view。

八项 analytical 参数现在是 Task v2 的正式字段：

```text
analytical_level
analytical_fp8_gemm_recipe
analytical_attention_algorithm
analytical_communication_mode
analytical_moe_dispatch_dtype
analytical_moe_combine_dtype
analytical_wideep_dispatch_dtype
analytical_wideep_combine_dtype
```

传播链为：

```text
default/recommend/estimate CLI
  -> cli API / build_default_tasks
  -> Task v2
  -> per-role get_database_view
  -> AnalyticalConfig
```

新版特别需要验证 PD 分离，因为 prefill 与 decode 各自加载 database view。参数必须通过 Task 自身传播，而不是依赖“两个角色恰好引用同一个已修改 database”。这也是 immutable view 设计的实际收益之一。

## 7. Rust engine-step：新增的执行边界

老版本的主要仿真路径以 Python operation 为中心。新版已经具备可将 operation 编译为 `OpSpec`、序列化并交给 Rust engine-step 执行的完整路径。Rust 当前实现 SILICON、HYBRID 和 EMPIRICAL，并要求 Python 与 Rust 对同一模式给出等价答案。

KernelSim 模型目前只有 Python 实现。若仅给 Rust enum 增加 `ANALYTICAL`，但没有把全部模型公式、参数 profile、硬件解析和错误边界移植到 Rust，会造成两条路径同名不同义。

本次采取显式能力门控：

```text
用户请求 engine_step_backend=rust
+ database_mode=ANALYTICAL
=> should_use_rust_engine_step() 返回 False
=> 自动执行 Python engine-step
```

这不是功能降级失败，而是新版“双引擎结果一致”契约下的保守选择。未来若实现 Rust KernelSim，需要同时完成：

- Rust operation/spec 和参数 schema；
- 四类模型公式及 profile 的单一来源或代码生成；
- Python/Rust 数值快照一致性；
- 错误边界、warning 和 source provenance 对齐。

在这些条件满足前，不应把 ANALYTICAL 加入 `_RUST_SUPPORTED_DATABASE_MODES`。

## 8. 硬件配置：从私有 JSON 选择转为 SystemSpec 驱动

旧建模阶段曾通过硬件名称选择 KernelSim 私有 JSON。该方式对实验方便，但不适合新版系统注册机制：

- system YAML 已是 AIC 的硬件事实来源；
- 新系统可以由外部 `systems_paths` 注入；
- 同一 GPU 可能有不同板卡、时钟、带宽和节点拓扑；
- 按名称推断架构会使别名和用户自定义系统失效。

新版 FA/MLA adapter 从 `database.system_spec["gpu"]` 读取：

```text
sm_count
clock_hz
shared_memory_per_sm_bytes
l2_capacity_bytes
l2_bandwidth_bytes_s
vector_peak_flops
mem_bw
各 dtype tensor-core peak
```

缺失必要字段时明确报错，并列出字段名。迁移已为当前主要 NVIDIA 系统补充这些信息。后续增加国产 GPU 或其他架构时，应提供同一抽象字段，而不是在 KernelSim 中加入按 vendor/name 分支。

这里仍有一项模型局限：L2 bandwidth 和 vector peak 在公开规格中通常不是稳定标称值，可能来自微架构推导或经验估算。它们应在 system 配置中标明依据，跨架构使用时也应把结果视为工程估计。

## 9. 通信：新版按策略委托，而不是把所有结果标成 analytical

ANALYTICAL 的目标是无算子实测表运行，但通信已有独立建模体系。本次没有复制一套 NCCL/DeepEP KernelSim，而是使用 `communication_mode`：

```text
empirical（默认）:
    topology SOL / 固定效率
    不读取通信实测表

silicon:
    委托原有通信查表
```

普通 MoE 和 WideEP 分别拥有 dispatch/combine dtype。CustomAllReduce 的 SOL 也从旧硬编码 2 bytes 改为读取 dtype bytes，并保留对旧测试中非 enum 参数的兼容默认值。

值得特别注意：

- communication `source` 保留为 empirical 或 silicon，而不是 analytical；
- `communication_mode=silicon` 会重新引入数据依赖；
- WideEP 的无表公式只是粗粒度拓扑估计，不代表完整 DeepEP pipeline；
- silicon 表中的 dtype 往往是采集语义的隐含属性，前端 dtype 不能改写既有表的真实采集口径。

## 10. 新版新增模型族与覆盖边界

新版主干比 0.9.0 增加了 DSA、MSA、Mamba、DeepSeek V4/稀疏 attention 等 operation。此次迁移的 KernelSim 只覆盖既有四类：

- GEMM；
- FA/MLA attention；
- BMM；
- 普通 SGLang MoE。

不能因为 database mode 名为 ANALYTICAL，就默认所有新版 operation 已被四类模型覆盖。未覆盖 operation 应采取以下顺序：

1. 若已有可信 SOL/EMPIRICAL，显式委托；
2. 若可由已建模 granular ops 严格组合，使用 `FallbackOp`/组合 operation；
3. 若边界和算法不同，明确报不支持；
4. 不用形状相近的旧模型静默替代新算子。

例如普通 MoE 模型不能严格代表 WideEP MoE core；FA 也不能在没有语义确认时替代 DSA/MSA。

## 11. Source、单位与错误语义

新版 `PerformanceResult` 和 breakdown 更重视 provenance。迁移维持以下规则：

- 专用 KernelSim：`source="analytical"`；
- 委托 SOL/EMPIRICAL：保留真实 source；
- `communication_mode=silicon`：保留 silicon source；
- 多来源组合：由 `PerformanceResult` 合成为 mixed。

KernelSim 内部统一使用微秒，AIC adapter 统一转换为毫秒。operation 层不得再次换算。错误处理也采用 fail-loud：

- MLA FP8 明确失败；
- 缺少 FA/MLA 硬件字段明确失败；
- 不支持的 GEMM/MoE quant mode 明确失败；
- FP8 BMM 保留可靠性 warning，但仍返回估计。

这些边界比静默 fallback 更适合新版 support matrix 和自动化 sweep，因为调用方能够区分“可估计但精度有限”和“模型语义不成立”。

## 12. 缓存、导入和包兼容注意事项

### 12.1 View cache

任何会改变预测数值的参数都必须可哈希并进入 view key。不要把 list/dict 直接存入 frozen config；需要先规范化为 enum、字符串或 tuple。

### 12.2 Lazy database

ANALYTICAL 允许 `allow_missing_data=True`，但仍需要 system YAML。无表运行不等于无 system spec 运行。lazy support matrix 在 view 上重新绑定，数据表对象则保持共享只读。

### 12.3 导入方向

允许：

```text
aiconfigurator upper layer -> aiconfigurator_core
core operation -> core kernelsim
```

不允许：

```text
aiconfigurator_core -> aiconfigurator CLI/Task
```

### 12.4 兼容入口

`src/aiconfigurator/sdk/kernelsim/__init__.py` 只是历史 import path 的兼容 facade。新增生产模型必须先进入 core，不能只添加到 facade。

## 13. 新旧实现对照表

| 主题 | 旧版实现 | 新版实现 | 迁移原因 |
| --- | --- | --- | --- |
| 包位置 | `src/aiconfigurator/sdk/kernelsim` | `aic-core/.../sdk/kernelsim` | core/产品层拆包 |
| 配置状态 | `set_analytical_config()` 修改 database | frozen config + database view | 并发和缓存隔离 |
| mode dispatch | 大量位于 PerfDatabase/单体 operations | 分算子 operation query | operation 模块化 |
| MLA fallback | 专用 analytical fallback | 新版通用 `FallbackOp` | 已支持逐 shape retry 和 immutable view |
| Prefix | 容易使用 module 总体比例 | granular fresh/full/attention 分项 | 新版 collector/loader 语义审查 |
| 任务入口 | TaskConfig/factory | flat Task v2 + per-role view | 新任务系统与 PD 分离 |
| Rust | Python 为主 | Python/Rust 双 engine-step | 新版 compiled engine |
| Rust analytical | 无对应问题 | 自动回退 Python | 保证双引擎等价 |
| 硬件来源 | 私有 JSON/名称分支 | system YAML/SystemSpec | 可扩展系统注册 |
| 通信 | analytical 配置混入共享对象 | config-driven 委托 | 无表与查表策略显式化 |
| provenance | 较粗 | analytical/empirical/silicon/mixed | 新版 breakdown 和支持矩阵 |

## 14. 后续开发建议

新增 KernelSim 算子时，推荐按以下顺序实施：

1. 先确认 operation 与 collector/backend 的准确计时边界。
2. 将纯模型放入 `aiconfigurator_core.sdk.kernelsim`，避免依赖 database 和 CLI。
3. 在 adapter 中完成 enum、dtype、硬件字段和微秒/毫秒转换。
4. 在具体 operation 的 ANALYTICAL 分支接入，不继续堆积 `PerfDatabase`。
5. 决定未覆盖子项应组合、委托还是报错，并保留真实 source。
6. 若增加配置字段，同步 immutable config、view key、Task v2、CLI 和 Python API。
7. 分别测试聚合与 PD 分离，确认两个角色配置不串扰。
8. 只有在 Rust 完整等价实现后才扩展 Rust supported mode。
9. 增加无数据目录测试，证明默认 empirical communication 不依赖 silicon 表。
10. 用既有 SILICON/HYBRID/EMPIRICAL/SOL 回归确认旧行为未变化。

## 15. 结论

KernelSim 的数学模型不是此次迁移中变化最大的部分。真正的特殊之处是：新版 AIC 已把性能数据、查询策略、operation 图、任务配置和执行引擎拆成更清晰的层次。ANALYTICAL 必须成为这一架构中的一种受约束 query policy，而不能继续作为对共享 `PerfDatabase` 的临时状态修改。

当前实现的核心价值在于：模型参数与任务隔离、硬件信息统一进入 SystemSpec、MLA prefix 使用正确 granular 边界、通信依赖显式可选、Rust 不支持时可靠回退，同时保留新版已有模式和 operation 的行为。后续维护时只要持续遵守这些边界，KernelSim 可以逐步扩展，而不会重新形成旧版那种全局状态、单体文件和隐式 fallback 相互耦合的结构。
