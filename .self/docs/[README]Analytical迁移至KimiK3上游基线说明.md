# Analytical 迁移至 Kimi-K3 上游基线说明

## 基线与目标

本次集成以 `jieyang-su/sync/upstream-main-20260730` 的 `17ade35b` 为代码基线，而非将旧 `analytical/1.0` 整体合并进来。目标是在保留后续官方主干修改和 Kimi-K3 Analytical no-GPU 支持的前提下，按新版接口迁入 DSA、DeepSeek V4、MiniMax M3、KernelSim 和 Pareto-v2 能力。

原 `analytical/1.0` 的 `b4c5514c` 保留为独立归档和回退点。本次集成不覆盖远端同步分支，也不纳入 `.self/analytical-mode` 下的实验数据、图表、日志和临时采集脚本。

## 远端实现优先保留

以下能力直接沿用 `17ade35b`，没有用旧版文件整体覆盖：

- Kimi-K3 模型、KDA operation、KernelSim、collector、数据表和 Rust 实现。
- Rust engine-step 默认策略及 Rust/Python parity 机制。
- operation-owned performance data loading、shared layer、strict provenance、插值和缓存结构。
- MLA module 的 native/local head key、CP、WideEP 和新版 quant transfer 行为。
- 最新 Task v2、CLI、support matrix 和模型配置解析。

Analytical 不在 Rust engine-step 的支持模式集合内。即使显式指定 Rust，也会在路由阶段回退 Python；SILICON、HYBRID 和 EMPIRICAL 的 K3/KDA Rust 路径不受影响。

## 迁入能力

本次按新版 operation/model 结构迁入：

- DSA Index MQA、DSA TopK 和 sparse attention 的 granular Analytical 组合。
- DeepSeek V4 的 SWA、CSA、HCA、mHC、CP granular Analytical 路径和 DSV4 TopK v1/v2。
- MiniMax M3 的 dense/MSA 分层语义、MSA Index 和轻量 TopK 估算。
- KernelSim 的 DSA 修正、MSA Index、DSV4 TopK 和 attention causal 配置。
- 普通 MoE/WideEP 公式通信的 dispatch/combine 显式 dtype。
- 与 Pareto-v1 平行的 PD 分离 Pareto-v2。

DSV4 和 MSA 的 module 表、calibration table 与 silicon 行为仍由远端路径负责；新增 KernelSim 只服务 Analytical 无实采数据路径。

## MLA Prefix 语义

Context MLA module 表现在保留以下维度：

```text
fmha -> kv -> gemm -> native_heads -> local_heads -> prefix -> fresh_s -> batch
```

loader 读取 collector 的 `step` 作为 prefix，不再让相同 fresh shape 的不同 prefix 行互相覆盖。查询只接受 prefix 精确命中或有上下界的数据内插；缺少可靠 bracket 时抛出数据缺失并转 granular，不再对整个 module 时延应用统一的 prefix 比例。

SGLang prefill granular 使用 full-K 语义：`kv_b_proj` 和 K concat 按 `fresh + prefix` 计算；q_b、o_proj 和 downscale 按 fresh token 计算。vLLM/TRT-LLM 不自动继承该 SGLang 假设。

module collector 接收预构造 latent QKV，因此 downscale 位于 wrapper 外：

```text
downscale + FallbackOp(module, granular_without_downscale)
```

这保证 module 命中和 granular fallback 都恰好计算一次 downscale，并保留远端 native/local head 选择。

## 模式边界

- `SILICON/HYBRID`：可优先查询新版 module 表，数据缺失后逐 shape 转 granular。
- `ANALYTICAL`：DSA 使用 granular；V4 显式排除 module primary；DeepSeek MLA 使用 `silicon_primary_only` 进入 granular。
- `EMPIRICAL/SOL`：保留远端既有路径，除明确使用 granular 的 wrapper 外不重新定义数据驱动含义。
- 通信 dtype 只改变 Analytical 下公式驱动的普通 MoE/WideEP 通信；silicon 表仍代表采集时固定 dtype。

## 验证与限制

定向回归覆盖 Kimi-K3/KDA、MLA prefix/native-local key、DSA、V4、M3、MoE 通信、Task v2、CLI、Pareto-v2 和 Rust fallback。完整测试应在合并提交前重新执行。

DSA、MSA 和 DSV4 TopK 仍包含 H100/SGLang 绑定的阶段性参数，跨硬件、backend、版本和 shape 的结果属于工程估计。`low/standard/high` 是敏感性档位，不是统计置信区间。
