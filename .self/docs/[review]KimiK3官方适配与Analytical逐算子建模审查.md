# Kimi K3 官方适配与 Analytical 逐算子建模审查

## 1. 审查范围与责任边界

本文把 Kimi K3 的 AIC 支持严格分为两个层级：

> 2026-08-14 状态更新：Hopper K3 已可通过 W4A16 MXFP4 的 BF16 transfer proxy 完成
> Analytical 执行。该路径未重新拟合，不是 W4A16 Silicon 模型；本文原先“因缺少
> recipe 完全无法运行”的结论已被本轮修复取代。KDA fusion、MoE norm、PP stage 和
> backend fidelity 等结构风险仍然成立。

1. **AIC 官方 K3 适配层**：负责解析模型配置、构造 KDA/MLA/LatentMoE/DSPARK operation 图、确定并行与量化语义、路由 Silicon 数据，以及估算权重和运行态内存。
2. **Analytical 跟随适配层**：不重新定义 K3 模型结构，而是在官方 operation 图上为 GEMM、attention/MLA、MoE、KDA kernel、通信和小算子提供无实采数据估算。

提交边界可以清楚证明二者并非同一轮实现：

- `a77804e3`：官方 K3 支持，加入 `KimiK3Model`、KDA operation/collector/table、DSPARK、多后端及多硬件数据。
- `17ade35b`：加入 Analytical no-GPU 模型；`kimi_k3.py` 未被修改，主要新增 KernelSim，并在 `KDAKernel` 查询层接入 Analytical。

因此，operation 图的漏项、错误顺序、PP/内存近似属于官方适配层；KernelSim recipe 缺失、代理模型偏差、Analytical 未复现 Silicon 动态融合路由属于 Analytical 跟随层。Analytical 会继承官方图的问题，但不应被误认为问题的来源。

本文证据分级：

- **已确认**：由代码、HF 示意实现和实际 AIC 实例化结果直接证明。
- **仓库已知问题**：官方 bring-up ledger 已明确记录。
- **高可信风险**：代码行为明确，但缺少目标 SGLang/Kimi 分支完整源码或实机拆解确认影响量。
- **待验证**：存在合理疑点，但当前证据不足以判定为缺陷。

## 2. K3 真实结构与 AIC operation 总图

K3 文本模型共有 93 层：69 层 KDA、24 层 MLA。主维度为 hidden=7168；KDA 为 96 heads x 128，短卷积宽度 4；MLA 为 96 heads，`q_lora=1536`、`kv_lora=512`、`qk_nope=128`、`qk_rope=64`、`v=128`，启用 NoPE 和 output gate。

FFN 方面，第 0 层为 dense MLP，后 92 层为 LatentMoE：896 experts、top-16，hidden 7168 先投影到 latent 3584，专家 intermediate=3072，再投影回 hidden；另有 2 个 full-hidden shared experts。模型使用 SiTU 激活，并以 12 层为 AttnRes block。

AIC 当前主干可概括为：

```text
KDA layer:
  input norm
  q/k/v/full-rank output-gate projection
  f_a + beta projection
  f_b projection
  QKV short conv
  KDA scan/recurrent kernel
  gated RMSNorm
  output projection
  TP all-reduce

MLA layer:
  input norm
  fused q_a + kv_a downscale
  q_b projection
  kv_b projection (prefill)
  MLA/attention core or decode absorb BMMs
  output-gate projection + sigmoid multiply
  output projection
  TP all-reduce

LatentMoE layer:
  FFN norm
  router in hidden space
  hidden -> latent
  latent norm (current AIC placement)
  dispatch -> routed experts -> combine
  latent -> hidden
  shared expert MLP
```

这一拆解方向是正确的。KDA collector 也只采集独特的 conv/scan/recurrent/fused kernel，明确排除 projection GEMM、norm 和通信，与逐算子 Analytical 组合的责任边界一致。bring-up 中模块级原型显示 B300 TP8、batch 1/8/64 的 KDA layer 约为 34.5/36.6/66.0 us，而 fused KDA core 仅约 6.7/6.7/17.0 us，证明 KDA core 绝不能代表整层，当前 granular 组合是必要的。

## 3. 第一层：AIC 官方 K3 适配审查

### 3.1 做到位的部分

**配置与层结构：整体可靠。** AIC 正确解析 1-based `kda_layers`，校验越界，得到 69/24 层分布；KDA head 必须整除 TP，CP 因递归状态不可顺序切分而显式拒绝，避免静默错价。

**KDA 语义：主体完整。** q/k/v、forget gate、beta、full-rank output gate、三路 causal conv、chunk/recurrent delta-rule core、fp32 recurrent state、gated output norm、output projection 和 AR 均有对应项。SGLang 与 vLLM 的 prefill/decode/verify kernel source 分开处理。

**MLA 语义：主体完整。** AIC 保留 K3 的真实 96-head geometry、NoPE MLA、output gate、SGLang granular MLA，以及 vLLM 的 MLA-as-attention 路径。decode absorb BMM 使用 96/48/24/12 exact-first 查询，缺表时才退到 next-pow2 head slice，而非永久把 K3 当 DeepSeek 128-head模型。

**LatentMoE 主边界：总体完整。** router 保持 hidden=7168，routed experts 在 latent=3584 内执行，专家 intermediate=3072，前后 latent projection、dispatch/combine、shared experts 和 dense 第 0 层均被表示。量化解析也符合 checkpoint ignore 规则：实测实例化中 attention/dense/shared GEMM 均为 BF16；H100 routed MoE 为 `w4a16_mxfp4`，B200 SGLang routed MoE 为 `w4a8_mxfp4_mxfp8`。

**DSPARK：已区分两套草稿模型。** SGLang 使用 RadixArk 5-layer GQA draft，vLLM 使用 Inferact 5-layer MLA-style draft；AIC 没有把 target 的 `nextn` 当传统 MTP layer 数直接放大，而是按 verify width 和 draft token progress 分开缩放。

**内存项：至少覆盖主要物理对象。** MLA 按 24 x (512+64) elements/token 计 token-linear KV；KDA 按 fp32 recurrent state、bf16 conv window 和每请求固定 slot 计费；DSPARK 额外 KV 也按 backend 几何计入。

### 3.2 已确认或官方已知的问题

#### P0/P1：SGLang fused KDA onorm 重复计时

这是仓库 ledger 已记录的 issue #1463。SGLang TP8 12-head shard 的 `kda_fused_decode` 已融合：

```text
conv update + recurrence + gated RMSNorm
```

Silicon 查询层会把静态图中的 conv+recurrent pair 动态重路由到 fused row，并把 conv 返回 0；但 `KimiK3Model` 对 SGLang 仍追加 `generation_kda_onorm`。ledger 量化为约 **0.2 ms/step double count**。

DSPARK fused verify 同样需要按实际 shard 判断 onorm 是否融合。当前图只对 vLLM no-spec 无条件移除 onorm，未把 SGLang 的 per-shard fusion contract 上移到模型组合层。修复应使用显式 kernel-boundary metadata，而不是继续依赖表项缺失推断。

责任：**官方 K3 operation 图/融合路由**。Analytical 若复用同一图也会继承，但不是 Analytical 创造的重复项。

#### P1：LatentMoE norm 顺序与 HF 语义不一致

HF 示意实现为：

```text
hidden -> latent
routed experts + weighted combine
latent RMSNorm
latent -> hidden
```

AIC 当前为：

```text
hidden -> latent
latent RMSNorm
dispatch -> experts -> combine
latent -> hidden
```

两者访存量接近，所以粗粒度总时延未必变化很大；但语义位置会改变：

- norm 是否处于 EP 通信边界内部；
- combine 输出 dtype 与 norm 输入 dtype；
- 能否与 latent-up 或 combine 融合；
- collector/module 边界如何解释。

建议将 norm 移到 post-dispatch/combine 后、latent-up 前，并增加 operation-order 契约测试。责任：**官方 K3 operation 图**。

#### P1：TP/PP 通信仍有已知乐观项

bring-up ledger 已记录 K3 AR 项在 bs8 约 **-34% optimistic**，约少计 **0.9 ms/step**，并标为 `k3_ar_fusion WON'T-DO`。当前 shared experts 的 gate/up 和 down GEMM 按 TP 分片，但图中没有独立 shared-expert AR；latent-up 也标为 replicated/unsharded。真实框架可能将部分归约与 routed/shared 路径组合，但当前代码没有可审计的 fusion contract。

这不一定能仅凭缺少一个 `CustomAllReduce` 判定具体漏了哪次通信，但“当前 AR 总体乐观”已由端到端测量确认。建议以 SGLang layer trace 明确：shared down output、latent-up output及两路相加前后的 rank-local/global tensor ownership，再决定加入独立 AR 或 fused communication operation。

责任：**官方 K3 并行/通信建模**。

#### P1/P2：PP 使用平均层成本，未建模最忙 stage

AIC 对全模型 op 总和按 `pp_size` 归一，不显式构造 K3 每个 pipeline stage 的真实层序列。K3 的 KDA/MLA 周期分布、dense layer 0、最后 logits 和 MoE 层并不均匀。

按连续层均分估算：

| PP | 平均 MLA/stage | 最大 MLA/stage | 最大/平均 |
| ---: | ---: | ---: | ---: |
| 2 | 12.0 | 13 | 1.08x |
| 4 | 6.0 | 7 | 1.17x |
| 8 | 3.0 | 4 | 1.33x |
| 16 | 1.5 | 2 | 1.33x |
| 32 | 0.75 | 2 | 2.67x |

PP32 时部分 stage 没有 MLA，而最后 stage 有 2 个 MLA；dense 首层和 logits 又只落在特定 stage。因 MLA/KDA/MoE 成本不同，平均模型可能低估 bottleneck stage，尤其影响高 PP 帕累托点。ledger 的“93%pp rounding”只描述层数不能整除，尚未解决异构层类型不均衡。

责任：**AIC 官方通用 PP 执行模型，对 K3 尤其敏感**，Silicon 与 Analytical 同受影响。

### 3.3 粗化但可接受的部分

#### AttnRes 不是普通 elementwise，当前 recipe 过粗

HF `_apply_attn_res()` 对 2..9 个候选 residual 执行：候选维 RMS 统计、hidden->scalar learned projection、softmax、再对 hidden 维加权 reduction。AIC 把它统一写为：

```python
ElementWise(..., scale=2 * 93, dim_in=4h, dim_out=2h)
```

`186` 次的计数基本合理：92 次 pre-attention、93 次 post-attention，加 1 次 final output aggregation。但固定 `4h -> 2h` 不表达 block 内候选数从 2 增到 9，也不表达 softmax和 reduction launch。

该项不能视为零成本。B200 TP8/EP8、8K prefill、batch=1 的 Analytical 代表点中，AttnRes 为 **21.83 ms，占 6.8%**；decode batch=32 为 **655.86 ms，占 1.9%**。无需马上建立重型 KernelSim，但值得新增 launch-aware reduction recipe，输入至少包含 tokens、hidden、candidate_count，并按每个 block 的真实 candidate count 求和。

责任：**官方图采用小算子粗化；Analytical/SOL/EMPIRICAL 共用该公式**。

#### KDA/MLA 双池内存是有意近似

AIC 把 MLA KV 和 KDA state 合并为一个弹性预算，并用 `KDA_STATE_SLOTS_PER_REQUEST=5` 表示 SGLang radix state pool。它接近 `--enable-unified-memory`，但默认 serving 往往是两个独立池：一个池先耗尽时，另一个池的剩余空间不能互借。

因此该模型会：

- 对短上下文、高并发场景因 5 slots/request 偏保守；
- 对默认双池比例不匹配场景偏乐观；
- 无法表达 prefix reuse/radix checkpoint pool 的动态共享程度。

这是已注释的工程近似，当前可保留，但帕累托显存边界点应标为中等可信度。

### 3.4 待验证项

#### `bfa_gemm` 的 TP 输出维度

AIC 对任意 TP 都使用：

```text
n = global_kda_heads + head_dim = 96 + 128 = 224
```

HF 语义是 `b_proj: hidden->num_heads` 和 `f_a_proj: hidden->head_dim`。若框架将 beta head 维按 TP 切分、而 f_a 的 128 维复制，则 TP8 更可能是 `12+128=140`，不是 224；若二者在 SGLang 中被特殊复制/融合，则当前值可能正确。

当前公开分支源码不足，不能判为确认缺陷。建议从目标镜像中读取 parallel-linear weight metadata，或用 Nsight/CUPTI 记录该 GEMM 的 `(M,N,K)`。若 N=140，当前会系统性高估 skinny GEMM；若 N=224，则保留现状并补契约注释。

## 4. 第二层：Analytical 跟随适配审查

### 4.1 架构设计是正确的

Analytical 没有复制一套 K3 layer/module 模型，而只在 `KDAKernel._query_kda_table()` 的 `ANALYTICAL` 分支调用 `estimate_kernel()`。其余 projection GEMM、MLA/BMM、MoE、elementwise 与通信继续使用现有 Analytical hooks。

这是正确的 ownership：

```text
官方 K3 operation graph
  + GEMM KernelSim
  + MLA/attention/BMM KernelSim
  + KDA-specific KernelSim
  + MoE KernelSim
  + communication delegation
  + small-op formula
```

它保持 KDA collector 的纯 kernel 边界，也避免把 KDA core 当整层重复计算 projection 和通信。

### 4.2 KDA KernelSim 的覆盖与成熟度

当前 KDA 模型覆盖：

- SGLang prefill：三路 causal conv + `chunk_kda`；
- SGLang decode：conv update + packed recurrent；
- SGLang verify：Triton pair，并具备 fused DSPARK estimator；
- vLLM prefill：`flashkda_fwd`；
- vLLM no-spec decode：fused decode；
- vLLM verify：fused recurrent；
- fp32 recurrent state 流量、backend fusion boundary、low/standard/high 三档。

模型成熟度应定为**中等、局部高可信**：

- fused decode/verify 单 kernel 边界相对清晰；
- SGLang prefill 的 v4 workload-saturation 依赖固定 64-token chunk、SM 数和 K3 geometry，跨 kernel 版本外推风险较高；
- vLLM prefill collector 含约 140 us/次的 host metadata 项，模型自己已标 low confidence；
- runtime autotune、workspace materialization 和跨硬件残差尚未完全校准。

因此 KDA KernelSim 适合 K3 当前 geometry 的规划级评估，不应宣称为任意 delta attention 的通用高精度模型。

### 4.3 确认缺口：Analytical 未跟随 SGLang 动态 fused 路由

Silicon 的 SGLang operation 图静态构造 conv+recurrent pair，查询时根据数据表是否存在对应 fused row，动态执行：

```text
conv -> 0
recurrent -> kda_fused_decode / fused_kda_decode_mtp_dspark
```

Analytical 分支在读取表和执行上述路由之前直接返回 `estimate_kernel()`，所以：

- no-spec SGLang TP8 始终按 conv+recurrent pair 估算，不使用已有 fused-decode estimator；
- SM100 DSPARK 始终按 conv+verify pair 估算，不使用 fused DSPARK estimator；
- 也无法根据 fused kernel 是否包含 onorm，消除官方图中的重复项。

这不是 KDA 模型缺少 fused estimator，模型已经实现；问题是 Analytical adapter 没有得到明确的 backend dispatch decision。建议把 fused route 从“Silicon 表项是否缺失”改为由 system/backend/shape/draft width 解析的显式 route contract，Silicon 和 Analytical 共用，然后分别查表或调用对应 estimator。

责任：**Analytical 跟随适配不完整，同时暴露官方 routing 依赖数据缺失的脆弱设计**。

### 4.4 已修复阻塞：Hopper K3 可运行，但 W4A16 仍是低可信代理

实际实例化结果：

| 系统/backend | GEMM | routed MoE |
| --- | --- | --- |
| H100 SGLang | BF16 | `w4a16_mxfp4` |
| H100 vLLM | BF16 | `w4a16_mxfp4` |
| B200 SGLang | BF16 | `w4a8_mxfp4_mxfp8` |
| B200 vLLM | BF16 | `w4a16_mxfp4` |

当前 `moe_latency_ms()` 已把 `w4a16_mxfp4` 和 `w4a16_mxfp4_cutlass` 映射到
`w4a16_mxfp4_bf16_transfer`。它复用 BF16 Sum-3P 的 launch、计算效率和访存效率，
保留 BF16 activation/intermediate/output，只把权重值改成 MXFP4 packed bytes，并加入
每 32 个权重一个 E8M0 scale 的逻辑流量。H100/H200 K3 因而不再在首个 MoE 查询失败。

这项修复解决的是“能否无卡执行”，不是“是否已有 W4A16 精确模型”：

1. 参数没有用 H100/H200 W4A16 数据重新拟合，Marlin/Triton/CUTLASS 共用一个 proxy；
2. fused unpack/dequant、tile、small-M 分派和 backend launch floor 没有单独建模；
3. EP>1 仍按理想均匀 workload 外推，不能吸收 collector 的 EP 异常；
4. 每次首次使用会发出低可信 warning，结果必须解释为 Analytical transfer proxy。

因此 Hopper K3 当前属于“路径闭环但低成熟度”，独立 W4A16 recipe 仍是 P0 校准工作。

### 4.5 Blackwell MoE 可运行，但代理精度有限

`w4a8_mxfp4_mxfp8` 当前被映射为 `nvfp4_cutedsl`。这只利用“低比特权重 + FP4 tensor-core peak”近似，不等价于真实 SGLang `flashinfer/trtllm-gen` MXFP4+MXFP8、SiTU、896 experts top-16 的调度。

B200 TP8/EP8 代表点：

| Phase | Silicon total | Analytical total | Silicon LatentMoE | Analytical LatentMoE |
| --- | ---: | ---: | ---: | ---: |
| 8K prefill, bs1 | 406.97 ms | 322.07 ms | 224.39 ms | 169.51 ms |
| decode, bs32 | 33337.83 ms | 35090.74 ms | 24515.59 ms | 27504.88 ms |

prefill Analytical 的 LatentMoE 低约 **24.5%**，decode 则高约 **12.2%**。这并不证明 Silicon 是绝对真值，小算子与通信仍含公式/共享层，但至少说明一个 NVFP4 proxy 无法稳定描述 K3 MoE 的 token 域、backend和分布变化。

建议新建 K3 MXFP4 MoE recipe 时优先覆盖：

- W4A8 MXFP4/MXFP8 trtllm-gen；
- W4A16 MXFP4 Marlin/Triton；
- latent hidden=3584、inter=3072、896x top-16；
- balanced/power-law 和 EP 维度；
- SiTU epilogue及小 token launch regime。

### 4.6 小算子仍通过通用公式，不等于 Analytical 实测级模型

代表点的 source 显示 AttnRes、norm、部分通信仍标为 empirical/公式来源，而非 KernelSim 专模。例如 8K prefill 中 KDA 27.2%、MLA 10.6%、AttnRes 6.8%，说明 Analytical 是“关键大算子 KernelSim + 小算子/通信估算”的组合，不是所有 operation 都达到相同可信度。

这符合 Analytical 的设计目标，但报告和 provenance 应避免把整个结果描述为统一精度的 `analytical`。建议在 module breakdown 中保留 operation source，并对 proxy/empirical/analytical 分别汇总。

### 4.7 backend 可信度不对称

KDA dispatch 已区分 SGLang/vLLM，但通用 GEMM、attention 和 MoE KernelSim 仍主要按 SGLang recipe 校准。vLLM K3 又使用 FlashKDA prefill、fused decode、MLA-as-attention、不同 DSPARK draft 和 W4A16 MoE，因此 vLLM Analytical 不能仅因 KDA source 匹配就视为全模型 backend-faithful。

当前建议：

- SGLang Blackwell、无 spec：中等可信；
- SGLang Blackwell、DSPARK：融合路由修复前中低可信；
- Hopper：可执行，但 W4A16 transfer proxy 为低可信；vLLM 仍需单独核对 backend 边界；
- 未采硬件、非 96-head/128-dim KDA 外推：低可信。

## 5. 逐组件责任与建模状态

| 组件 | 官方图还原 | Analytical 覆盖 | 主要问题 | 责任层 |
| --- | --- | --- | --- | --- |
| KDA projections | 基本完整 | GEMM KernelSim | `bfa_gemm` TP shape 待确认 | 官方待验证 |
| KDA conv/scan | 完整 granular | KDA KernelSim | prefill 外推/版本风险 | Analytical 模型 |
| KDA fused decode/verify | Silicon 可动态路由 | 未跟随动态路由 | pair/fused 边界不一致 | 两层接口 |
| KDA output norm | 独立 op | 通用 mem formula | fused shard 重复计时 | 官方图 |
| MLA NoPE/output gate | 完整 | MLA/BMM/GEMM复用 | vLLM 走 attention proxy | 官方近似 + Analytical backend 风险 |
| AttnRes | 次数基本正确 | 通用 mem formula | 未建模候选数/softmax/reduction | 官方粗化 |
| LatentMoE projections | 边界大体完整 | BF16 GEMM KernelSim | norm 顺序错误 | 官方图 |
| Routed MoE | shape/quant 解析正确 | Hopper 使用 W4A16 transfer proxy，Blackwell 使用 NVFP4 proxy | 均未 backend 专模校准 | Analytical |
| Shared experts | 计算量已建 | BF16 GEMM + mem formula | AR/fusion ownership不清 | 官方图 |
| Dispatch/combine | 已建 | 按 communication mode | dtype/真实框架融合需继续核对 | 两层 |
| DSPARK draft | 两 backend 几何已分开 | 复用 GEMM/attention | KDA fused verify 跟随不足 | Analytical 路由 |
| KV/KDA state memory | 主要对象齐全 | 两模式共用 | 单弹性池与5-slot近似 | 官方内存 |
| PP | 通用平均归一 | 两模式共用 | 不建模异构最忙 stage | 官方执行模型 |

## 6. 新专用模型的优先级

### P0：MXFP4 MoE 独立校准

必须覆盖 W4A16 与 W4A8 两条真实 K3 serving lane。现有 transfer proxy 已解决 Hopper
可运行性，但 K3 P/D 的第一大项仍缺少 backend-specific 参数；相比继续微调 KDA，
为 Marlin/Triton/W4A8 分开校准的全局收益更高。

### P1：显式 KDA dispatch/fusion contract

这不是再拟合一个模型，而是让官方图、Silicon 和 Analytical 共享同一个 route resolver。输入应至少包含 backend、system/SM、local heads、head dim、phase、draft width 和 fused-kernel capability；输出应包含 kernel sequence 以及 folded operations（conv/onorm）。

### P1/P2：AttnRes launch-aware recipe

建议先做轻量模型，不需要独立复杂 KernelSim：

```text
latency = launch(candidate_count)
        + max(reduction_flops / effective_compute,
              bytes(tokens, hidden, candidate_count) / effective_bw)
```

按 block 内实际 candidate count 求和。若实机显示占比持续低于 2%，可保持 recipe；若 8K prefill 类场景维持 5%+，再补 H100/B200 小规模采集。

### P2：K3 stage-aware PP

该能力应作为通用 heterogeneous-layer pipeline 功能实现，而非 K3 私有修正。至少用连续 layer assignment 构造每 stage 的 KDA/MLA/MoE/dense/logits op 总和，并以最大 stage latency和最大 stage weights做瓶颈估算。

### 暂不建议新建的模型

- KDA/MLA projection GEMM：已有 GEMM KernelSim 可复用，优先校正 shape/fusion contract。
- latent norm、SiTU elementwise：先使用小算子公式，除非 Nsight 证明融合边界或占比显著。
- router/top-k：现有 router GEMM 与 MoE recipe 已覆盖主体；K3 top-16 本身无需像 DSA sparse top-k 那样单独建立 attention-index 模型。

## 7. 推荐修复与测试计划

### 第一批：正确性与可运行性

1. 修正 LatentMoE norm 的 operation 顺序。
2. 建立 KDA route resolver，Silicon/Analytical 共用，显式 fold conv/onorm。
3. 补 `w4a16_mxfp4` Analytical MoE recipe；在此之前让 unsupported error 在 Task validate 阶段提前暴露。
4. 增加 H100/H200/B200，SGLang/vLLM，no-spec/DSPARK 的 K3 Analytical smoke tests。

### 第二批：并行和量化契约

1. 从目标 SGLang 镜像确认 `bfa_gemm` 的 TP-local N。
2. 用 layer trace确认 shared expert、latent-up 与 routed output 的 AR ownership。
3. 测试 attention/dense/shared 保持 BF16，routed MoE按系统/backend解析成正确 MXFP4 mode。
4. 对 KDA fused row 禁止继续依赖 donor table 的 absence 作为永久语义；现有 absence-sensitive test可保留为兼容保护。

### 第三批：精度和可解释性

1. 建立 AttnRes recipe并采集少量 P/D shape。
2. 加入 stage-aware PP breakdown。
3. 报告 KDA KernelSim 的 central/low/high及 confidence，而非只丢弃边界信息返回 central。
4. breakdown 同时输出 operation source 和 proxy recipe，例如 `analytical:mxfp4-via-nvfp4-proxy`。

## 8. 总结判断

### 官方 K3 适配

官方适配并非粗粒度 module 代替，而是较完整的 granular operation 图。69/24 hybrid layer、KDA 独特 kernel、MLA NoPE/output gate、LatentMoE、DSPARK、量化和主要内存对象均已还原，整体基础质量较高。

但它尚不能评价为“完全到位”：fused KDA onorm 重复计时和 AR 乐观已被官方 ledger 确认；LatentMoE norm 顺序与 HF 语义不一致；AttnRes、双池内存和 PP stage 是有影响的结构近似；`bfa_gemm` TP shape仍需框架源码/实机确认。

### Analytical 跟随适配

Analytical 的接入架构选择正确：KDA 只承担独特 core kernel，其余继续复用现有组件，避免整层重复建模。KDA KernelSim 也已覆盖主要 backend/phase，并具备合理的三档与不确定性说明。

但当前不能称为多硬件、全路径成熟支持：H100/H200 已由 W4A16 BF16 transfer proxy
打通，但没有独立拟合；Blackwell MoE 使用 NVFP4 proxy，P/D 误差方向不稳定；
SGLang fused decode/verify 的动态路由没有传递到 Analytical，导致实际执行边界与
Silicon 不一致，并继承 onorm 重复项。

综合成熟度建议：

- **官方 K3 granular 图：中高成熟度，存在若干明确需修项。**
- **KDA KernelSim 单模型：中等成熟度，校准 geometry 内可用于规划。**
- **K3 全模型 Analytical：Blackwell SGLang 为中等成熟度；Hopper 已可运行但仍为低成熟度 proxy；vLLM 和 DSPARK 融合路径仍不完整。**

当前最优先的工作不是继续增加更多 KDA 公式，而是补齐 MXFP4 MoE、统一 fused route contract，并修正官方 operation 图中的 norm/通信边界。完成这三项后，K3 Analytical 才能达到与现有 DSV3.2/V4 granular 路径相近的工程可用性。
