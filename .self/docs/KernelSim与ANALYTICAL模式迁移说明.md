# KernelSim 与 ANALYTICAL 模式迁移说明

## 迁移目标

本次迁移在新版 AIC 架构中引入 GEMM、FA/MLA、BMM 和 MoE KernelSim，并新增 `ANALYTICAL` 数据库模式。该模式用于没有实采表时的快速理论评估，相比既有 SOL/EMPIRICAL 增加了启动开销、资源效率、attention task-service 等算子特性，但不改变 SILICON、HYBRID、EMPIRICAL、SOL 和 SOL_FULL 的行为。

生产实现位于 `aic-core/src/aiconfigurator_core/sdk/kernelsim/`。上层 `src/aiconfigurator/sdk/kernelsim/` 仅提供兼容导出；实验图表、校准脚本、私有硬件 JSON 和独立包配置未迁入生产代码。

## 新版架构接入

`AnalyticalConfig` 是不可变配置，随 `PerfDatabase` view 传播并参与 view cache key。不同参数档位、GEMM recipe 或 attention 算法不会修改共享 database，也不会共享错误的缓存结果。

默认配置为：

```python
AnalyticalConfig(
    level="standard",
    fp8_gemm_recipe="sglang",
    attention_algorithm="fa2",
    communication_mode="empirical",
    moe_dispatch_dtype="half",
    moe_combine_dtype="half",
    wideep_dispatch_dtype="half",
    wideep_combine_dtype="half",
)
```

配置已通过 TaskConfig v2、Python CLI API 和 `default`、`recommend`、`estimate` 命令行入口传播。非 SGLang backend 可以运行，但每个 database view 会警告一次，因为参数校准和采集边界来自 SGLang。

新版 system YAML 直接提供 `sm_count`、`clock_hz`、单 SM shared memory、L2 容量/带宽及 vector peak。FA/MLA 从当前 system spec 构建硬件模型，不再根据硬件名称读取私有配置。

## 算子与通信语义

- GEMM 支持 `low/standard/high` 三档。FP8 默认使用 SGLang GEMM；DeepGEMM 需要显式选择 Hopper 或 Blackwell recipe。
- FA 显式选择 FA2 或 FA3，默认 FA2。正的分数 token 按一次真实 kernel launch 向上取整，零工作保持零时延。
- MLA 仅支持 BF16。ANALYTICAL 沿用 prefix-aware granular 序列，不对 module 总时延恢复统一 prefix 比例。
- BMM 和 MoE 使用迁移后的工程参数；FP8 BMM 保留数据可靠性警告。
- 未覆盖的小型逐元素算子继续使用新版已有 SOL/EMPIRICAL 公式。
- `communication_mode="empirical"` 使用无表 topology SOL 和固定效率；`"silicon"` 委托实测表查询。普通 MoE 与 WideEP 的 dispatch/combine dtype 独立配置。

所有 KernelSim 算子返回 `source="analytical"`；委托的通信保留实际来源，以便识别结果是否依赖数据表。

## Rust 边界

ANALYTICAL 首版只在 Python engine-step 中执行。Rust operation enum 和二进制协议没有增加第二套 KernelSim 实现；即使请求 Rust engine-step，运行时也会自动选择 Python。该选择避免 Python/Rust 两份参数和公式逐渐漂移，并保持已有 Rust 模式完全兼容。

## 能力边界

- MLA FP8 会明确报错，不静默退化到其他精度。
- H100 上的 FP8 GEMM 默认不是 DeepGEMM；硬件特异 recipe 必须由用户确认。
- 通信 empirical 是粗粒度 topology 模型，不等同于特定 NCCL/DeepEP kernel 的实测性能。
- 新版 DSA、MSA、Mamba、DeepSeek V4 专属算子不属于此次四类 KernelSim 的校准范围，继续使用它们原有的理论或数据库路径。
- 跨 backend、跨 GPU 架构的 `standard/low/high` 是工程区间估计，不应解释为对应硬件的 precise 校准。

## 验证范围

迁移测试覆盖模型数值快照、参数档位、FA2/FA3、GEMM recipe、MLA FP8 错误、通信 dtype、无表通信、database view 隔离、非 SGLang 警告及 Rust 自动回退。SDK 回归和 Rust crate 测试用于确认旧 database mode 与 Rust 路径保持不变；Llama、Qwen3 和 DeepSeek V3 的最小运行用于验证模型级算子覆盖。
