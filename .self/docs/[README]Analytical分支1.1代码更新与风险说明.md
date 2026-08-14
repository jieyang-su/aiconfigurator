# Analytical 分支 1.1 代码更新与风险说明

## 基线与范围

`analytical/1.1` 基于 `analytical/1.0-k3-integration`，归档以下四类生产代码更新：

1. mixed-precision KernelSim transfer proxy；
2. 国产系统、H20、架构能力解耦与单超节点拓扑；
3. `independent/tp_first` 通信 placement 和 Silicon 通信外推；
4. Pareto-v2 GPU accounting、前沿完整性和 provenance 修正。

`.self/analytical-mode/**`、`.self/domestic-gpu/**` 的 CSV、图、日志、脚本和 Nsight
资产不进入 Git。`uv.lock` 的本地清华镜像 URL 变化也不提交。

## Mixed precision

新增或修正：

- W8A16 dense GEMM 与 SGLang MoE BF16 transfer proxy；
- W4A16 MXFP4 SGLang MoE BF16 transfer proxy；
- FP8 KV + BF16 FA 的 compute/cache dtype 分离；
- MLA FP8 KV 的 BF16 compute proxy；
- DSA Index MQA BF16 resource-scaled proxy；
- Python/Rust `int8_wo` MoE enum parity；
- `fmha_quant_mode` 从模型图传到 generation attention operation。

这些路径都只改变理论工作量或资源比例，不是重新拟合，也不生成 Silicon 数据。
通用 INT4 GEMM、`int4_wo` MoE 和 backend-specific W4A16 仍未覆盖。

## Kimi K3 状态

官方 K3 模型适配与个人 Analytical 跟随适配必须分开评价：

- 官方图提供 KDA/MLA/LatentMoE/DSPARK 等 granular 结构和 Silicon route；
- Analytical 为 KDA core 建模并复用 GEMM、MLA/BMM、MoE 和小算子模型；
- Hopper K3 已能通过 W4A16 transfer proxy 执行，不再是 recipe 缺失导致的硬失败；
- Blackwell W4A8 仍借 NVFP4 proxy；
- fused KDA onorm ownership、Analytical fused route、MoE norm 顺序、AttnRes、双池内存
  与 PP stage imbalance 尚未解决。

因此 K3 路径“可运行”不等于“多硬件成熟”。

## 架构与拓扑

国产 YAML 使用 `architecture_family: domestic` 和 capability override，避免代理 SM90
触发 Hopper/Blackwell 专用 backend。`single_supernode` 限制所有有效 TP/PP/DP/CP 和
MoE TP/EP 配置位于一个超节点，并让 PP P2P 使用域内带宽。

`tp_first` 在 8 卡 NVIDIA node 上按逻辑 rank 判断通信组是否跨 node；`independent`
保留原行为。Rust engine 暂不支持这些元数据，因此相关任务回退 Python。

## Pareto-v2

本轮修正：

- P/D worker 与最终吞吐效率统一使用 `num_total_gpus`；
- CP/EP/PP/DP 不再通过 `pp*tp*dp` 的旧投影丢失卡数；
- rate-match 后保留 prefill/decode operation source；
- 最终 objective 空间重新去重并拒绝 dominated、非有限和非单调前沿；
- 验证结果卡数属于请求的 `num_gpu_list`；
- Kimi K3 在大模型场景加入 16/32 卡 worker 和 PP 候选；
- Task v2 可显式传播 `workload_distribution`。

Pareto-v1 未修改。旧 Pareto-v2 结果可能遗漏最优点或产生非规则前沿，严谨比较应使用
1.1 重新搜索；只对旧 CSV 后处理不能恢复此前未搜索到的候选。

## Source 与数据语义

- `silicon`：来自表或其插值/拓扑外推，不保证所有小算子均有实采；
- `empirical`：可能是本 quant utilization，也可能是 sibling transfer；
- `analytical`：KernelSim、recipe 或显式 proxy；
- `estimated/sol`：理论公式，不是实采；
- KDA/DSA/MSA/DSV4 TopK 和 mixed precision proxy 都有各自范围限制。

作图和报告应保留 module/granular 路径、operation source、recipe、fallback 与 warning，
不能只按顶层 database mode 着色。

## 已知风险

- 当前环境缺少 `torch` 时无法收集 `test_moe_dispatch.py`；需要在完整依赖环境复验；
- 国产硬件完全无 Silicon 数据，SM/L2/launch 使用代理参数；
- W4A16/W8A16、FP8 KV BF16 compute 和 DSA BF16 均为未重新拟合 proxy；
- KDA fused route 与官方图中的 onorm ownership 仍可能重复；
- MoE collector 的部分 EP>1 数据有已知异常，不能作为拟合真值；
- `tp_first` 不模拟真实 NCCL 分层算法或拥塞；
- Rust 只保留 `independent`、非国产兼容路径，其他情况回退 Python；
- Pareto-v2 的严谨前沿需要重新执行全搜索，不能复用旧搜索遗漏。

## 验证要求

归档至少运行 YAML parse、Ruff、`git diff --check`、KernelSim/Analytical/Task/拓扑/
Pareto-v2 Python 回归和 Rust core `cargo test`。缺少依赖导致的未执行项必须在提交说明
中记录，不得以跳过测试等同于通过。
