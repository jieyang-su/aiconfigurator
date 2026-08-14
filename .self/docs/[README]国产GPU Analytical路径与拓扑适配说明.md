# 国产 GPU Analytical 路径与拓扑适配说明

## 定位

国产系统当前只用于无实采数据的 Analytical 估算。YAML 中的 SM 数、时钟、共享内存、
L2 和部分功耗字段是 KernelSim 微架构代理，不代表 NVIDIA ISA、CUDA backend 或实际
kernel 可执行性。Silicon、collector 和真实国产通信 backend 均不在本轮范围内。

## 系统配置

| system id | 有效超节点卡数 | 显式 MMA 能力 | 域内带宽 |
|---|---:|---|---:|
| `_dom_br100_64` | 64 | BF16；FP8/FP4 false | 400 GB/s |
| `_dom_br100_128` | 128 | BF16；FP8/FP4 false | 400 GB/s |
| `_dom_ascend_910c` | 384 | BF16/FP16 proxy；FP8/FP4 false | 392 GB/s |
| `_dom_ascend_950dt` | 96 | BF16、FP8、FP4 | 840 GB/s |
| `_dom_klx_m300_512` | 256 | BF16、FP8；FP4 false | 400 GB/s |

`_dom_klx_m300_512` 保留既有 system ID，避免破坏本地任务与结果引用；当前 YAML 和实验
实际使用的有效超节点容量是 256，不应按文件名解释为 512 卡。

所有国产 YAML 都设置：

```yaml
gpu:
  architecture_family: domestic
  capability_overrides: ...
node:
  topology_scope: single_supernode
  inter_node_bw: 0
```

## SM 与能力解耦

`sm_version` 只服务 KernelSim 的代理调度参数。SDK 的架构路由改用集中 helper：

- `is_hopper_spec()`、`is_blackwell_spec()` 和 `is_sm100_spec()` 会先检查
  `architecture_family`，国产 SM90 proxy 不会被误判为 Hopper；
- `supports_fp8_mma()`、`supports_fp4_mma()` 和 `supports_mnnvl()` 优先使用 YAML
  capability override；
- 未声明新字段的 NVIDIA 系统继续按历史 SM 规则推导，保持兼容。

因此国产卡不会进入 NVIDIA MegaMoE、MNNVL、SM100 DeepGEMM 等专用 Silicon 路由。
collector 中读取真实 NVIDIA SM 的语义没有修改。

## 精度路径

- 不支持 FP8 MMA 的系统遇到 FP8 KV 时，普通 FA 可采用 FP8 cache traffic + BF16
  compute proxy；反量化假定融合且忽略显式开销。
- MLA 全局仍可使用 FP8 KV 做容量估算，MLA/BMM 时延临时按 BF16 compute proxy。
- DSA Index MQA 没有 FP8 peak 时，使用 BF16-to-FP8 theoretical resource ratio 缩放
  H100 FP8 task term；这是低可信 proxy。
- W8A16 GEMM/MoE 和 W4A16 MXFP4 MoE 只迁移 BF16 参数并修改权重/scale 流量。
- 不存在对应 peak 字段且没有显式 proxy 的 dtype 继续报错，不通过伪造 SM100 绕过。

## 单超节点约束

对 `single_supernode` 系统，每个显式或枚举候选必须满足：

```text
worker_gpus    = tp * pp * attention_dp * cp <= capacity
attention_group = tp * attention_dp * cp      <= capacity
moe_group       = moe_tp * moe_ep              <= capacity
```

约束在 Task v2、候选枚举、sweep 和 InferenceSession 多层生效。AFD 当前按多通信域的
node-granular A/F placement 实现，因此国产单超节点模式明确拒绝 AFD，而不是使用
`inter_node_bw=0` 产生无穷或除零。

## Communication placement

`independent` 保留历史 size-only 判断。普通 NVIDIA PP P2P 仍沿用 inter-node pipeline
近似；国产单超节点 PP 使用 intra-node bandwidth。

`tp_first` 使用 TP、CP、attention-DP、PP 的逻辑 rank 顺序，分别判断 TP、CP、DP、
完整 attention、MoE TP/EP 和相邻 PP edge 是否跨 8 卡 node。它不模拟 NCCL 分层算法、
contention 或真实 rank remap，只决定使用 intra/inter bandwidth。

Silicon communication table 在 `tp_first` 下保留实采 launch/algorithm 行为，再按 ring
参与比例和目标 domain bandwidth 外推。国产卡没有通信实采表，仍使用公式路径。

Rust engine-step 尚未携带 placement/capability metadata。国产系统、单超节点系统和
非 `independent` placement 会回退 Python，并按 system/placement 只发一次 warning。

## H20 对照系统

新增 `h20_sxm` Analytical YAML：96 GiB、148 TFLOPS BF16、296 TFLOPS FP8、8 卡
node、450 GB/s 域内和 50 GB/s scale-out proxy。数据目录只有 `.gitkeep`，表示当前
没有 H20 Silicon 表；不能把其结果解释为实采。

## 风险

- 国产计算、缓存和 launch 行为继承代理微架构，跨硬件误差未知；
- 单超节点带宽只是一阶 per-GPU 带宽模型，不含 topology contention；
- mixed-precision transfer proxy 没有国产 kernel 数据；
- `tp_first` 是理论 what-if placement，不等同于框架/NCCL 实际 placement；
- 结果适合相对筛选、瓶颈分解和敏感性分析，不适合直接容量承诺。
