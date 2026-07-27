# MoE Recorded AIC 迁移、Truth 验证与跨硬件校准方法草稿

本文是后续生成 Codex skills 的源材料。它沉淀的是一套方法，而不是某一次
DeepSeek-V3 修 bug 的记录。

核心目标：

- 以 AIC 现有 recorded 模式为主驱动。
- 以实机 truth 测算代码和方法为辅助验证手段。
- 将 recorded 模式迁移到不同 MoE 模型。
- 在同一模型语义和同一 SGLang/backend runtime 语义内，证明 recorded 模式能
  跨不同硬件泛化。
- SGLang/backend runtime 版本变化时，先做兼容性迁移和打点验证；新版本应作为
  独立校准 scope，不和旧版本混在一起拟合同一套校准。
- 在误差不收敛时，按 source / materializer / truth / gate 的边界定位问题，
  并通过平台无关、模型语义清晰的方式收敛误差。

后续应拆成两个 skill：

1. `moe-recorded-aic-porting`
2. `moe-recorded-cross-hardware-calibration`

第一个处理“新 MoE 模型或新 runtime 版本如何接入 recorded AIC”。第二个处理
“某个 recorded AIC 模型在一个固定 runtime 语义内如何跨硬件验证、校准、
产出数据”。如果 runtime 版本变了，应先回到第一个 skill 做兼容性迁移。

## 1. 方法总览

Recorded AIC 的基本链路：

```text
模型能力识别
  -> case matrix 生成
  -> collector source 采集
  -> source health / source contract 检查
  -> materializer 生成 final compact latency
  -> clean latency 安装到顶层 *_perf.txt
  -> 离线 gate 对比实机 truth
  -> 误差归因与收敛
  -> 产出可发布数据
```

几个核心原则：

- AIC recorded 是主线；实机 truth 是离线验证和校准设计的辅助。
- AIC source 不是最终 latency。
- AIC final compact latency 才和 truth 对比。
- collector 运行时不使用 truth。
- 不写 H20/H100 或其他平台名特判。
- 不写 SGLang/backend 版本名特判；版本差异只能作为明确 API/path 兼容 guard。
- 不用厚策略把 bad point 直接修成 truth。
- 模型 case 要因模型而异，不是所有 MoE 模型都必须有 WideEP context /
  WideEP generation。

## 2. 模型能力与 case matrix

迁移 recorded AIC 到一个 MoE 模型时，第一步不是直接套 DeepSeek-V3 的四模式，
而是先识别模型与后端实际支持的 MoE case。

### 2.1 必查模型能力

需要明确：

- 模型是否是 MoE。
- routed expert 数量。
- shared expert 是否存在。
- topk / routing 方式。
- MoE layer 范围。
- context/prefill 是否走 MoE。
- generation/decode 是否走 MoE。
- 是否支持 EP。
- 是否支持 WideEP/DeepEP。
- 是否支持 EPLB。EPLB 通常和 WideEP/DeepEP 或特定后端策略绑定，
  应作为可选能力，不应要求所有 ordinary MoE 模型都有。
- 是否开启 cuda graph。
- 当前 SGLang/backend package 版本、git commit、镜像和实际 Python path。
- 当前 MoE runtime 路径是否仍是旧版本假设的路径。
- 是否可走 single-card EP simulation。当前 AIC recorded 迁移默认使用
  single-card EP simulation，不把多卡真实执行作为默认 AIC 测算路径。

### 2.2 case matrix 不是固定四模式

Recorded AIC 的 family group 应按模型能力裁剪：

| family group | 什么时候需要 | 什么时候不需要 |
| --- | --- | --- |
| ordinary MoE | 模型存在普通 routed MoE 路径 | 模型不是 MoE，或不需要普通 MoE recorded AIC |
| WideEP/DeepEP MoE | 模型和后端支持 WideEP/DeepEP MoE | 模型不支持 WideEP/DeepEP，或本次 AIC 不覆盖 WideEP |

因此：

- DeepSeek-V3 可以展开成 ordinary context、ordinary generation、
  WideEP context、WideEP generation 四个 family。
- 其他 MoE 模型可能只有 ordinary MoE，即只展开 ordinary context 和
  ordinary generation。
- 有的模型可能有 ordinary MoE，但没有 WideEP。
- ordinary MoE 是一个成对能力：需要 ordinary recorded 时，
  `ordinary_context` 和 `ordinary_generation` 都应作为同一组进入 case matrix；
  不需要时两者都不进入。
- WideEP/DeepEP MoE 也是一个成对能力：需要 WideEP recorded 时，
  `wideep_context` 和 `wideep_generation` 都应作为同一组进入 case matrix；
  不需要时两者都不进入。
- 如果未来确实遇到只有 context 或只有 generation 的特殊模型，应在 skill
  执行时显式记录为模型例外，而不是作为默认设计。

skill 生成时必须把这个规则写成“按模型能力生成 case matrix”，不要写成
“固定跑四个模式”。

### 2.3 case matrix 的输出

每个模型应形成一张明确的 case matrix：

```text
family
phase
EP list
EPLB list，可选；没有 EPLB 的模型不生成 EPLB 维度
token list
backend
backend version / runtime image / sglang commit
model path / model config
runtime MoE path
single-card EP simulation 默认启用
是否需要 source health guard
是否需要 truth gate
```

示例：

```text
Model A:
  ordinary_context: yes
  ordinary_generation: yes
  wideep_context: no
  wideep_generation: no
  EPLB: no

Model B:
  ordinary_context: yes
  ordinary_generation: yes
  wideep_context: yes
  wideep_generation: yes
  EPLB: optional, 根据 WideEP/后端能力决定
```

### 2.4 同模型新 Runtime 版本迁移

第一个 skill 不只是“跨模型”。同一个模型升级 SGLang/backend 版本时，也应走
第一个 skill，因为版本变化可能改变：

- MoE runtime path。
- WideEP/DeepEP dispatcher。
- cuda graph capture/replay 行为。
- collector source schema。
- materializer 输入字段。
- SGLang truth marker 位置。
- nsys parser evidence。

如果同模型的新 runtime 版本在 bring-up 硬件上 recorded AIC 验证失败，不要
直接进入跨硬件校准。应先在第一个 skill 内处理：

1. 确认对比点一致：
   - model config。
   - layer truncation。
   - family/phase/token/EP/EPLB。
   - public token 语义。
   - quant/backend flags。
   - cuda graph mode。
2. diff runtime code path：
   - ordinary MoE 路径。
   - WideEP/DeepEP dispatcher。
   - `run_moe_core` call site。
   - piecewise cuda graph path。
   - EPLB flag 处理。
3. diff collector source schema：
   - compact rows。
   - raw/source columns。
   - replay manifest。
   - source health guard 决策。
   - no-keep 与 keep-source final latency。
4. diff materializer 输入：
   - 必需字段是否存在。
   - 是否误用二级/debug 输出。
   - source 到 final latency 的转换是否变化。
   - 是否误把 truth 派生值写入 source 或 final latency。
5. 重验 truth instrumentation：
   - marker 是否进入 server worker。
   - marker 是否只包 routed compute。
   - parser evidence 是否属于当前版本。
  - selected kernels 是否排除 dispatch/combine/通信/attention/shared/topk。
6. 决策：
   - 如果 runtime kernel 没变但 AIC 变了，修 collector/materializer 兼容。
   - 如果 source 没变但 truth 变了，查打点和 parser。
   - 如果 runtime kernel 确实变化且 truth 证实，给该版本设计薄的、
     语义驱动适配。
   - 大多数版本升级理论上不需要改变 recorded 方法，因为算子语义通常不会大幅
     改变；必须先证明原因，再增加兼容逻辑。
   - 如果新版本的 recorded AIC 语义在 bring-up 硬件上验证通过，新版本才可以
     作为稳定 runtime scope 进入跨硬件校准。

禁止：

- 写版本名特判去追某个误差点。
- 把 truth 作为 collector runtime 输入。

## 3. AIC Recorded 端到端语义

### 3.1 Collector source

source 是当前硬件上测出来的原始材料，用于给 materializer 提供输入。

source 可能包括：

- raw latency
- rank-local replay latency
- rank max latency
- rank mean latency
- rank spread
- token distribution shape
- expert/rank assignment shape
- active experts
- masked_m
- tiny scaled source
- WideEP source shape
- kernel / backend metadata

不同模型、不同 family 的 source 可以不同，但必须满足：

- 字段语义清晰。
- 与 materializer 输入契约一致。
- no-keep 和 keep-source 路径都能生成同样 final latency。
- 是否落盘不能影响 source feature 是否参与 materializer。
- SGLang/backend 版本变化后，需要重新审 source schema；不能默认旧版本的
  中间字段、CSV 名称或 replay shape 仍然成立。

### 3.2 Materializer

materializer 把 source 转成 final compact latency。

职责：

- 按 family/phase 选择对应转换逻辑。
- 使用 token、EP、EPLB、source shape、rank shape 等通用特征。
- 产出最终 `latency`。
- 不使用实机 truth。
- 不写硬件平台名特判。

materializer 可以有模型特定逻辑，但必须来自模型结构和 source 语义。例如：

- expert 数量不同。
- topk 不同。
- shared expert 是否存在。
- WideEP 路径是否存在。
- context/generation kernel 语义不同。

materializer 不应该把某一次 H20/H100 的误差点写死。也不应该把某一个
SGLang/backend 版本的中间字段或路径当成所有版本恒定成立；换 runtime 版本时
必须重新审 source schema、MoE 调用路径和 materializer 输入契约。

### 3.3 Clean latency

clean latency 负责把 materializer 输出安装到顶层 compact 表。

顶层 compact 文件是 AIC 对外产物：

- `moe_perf.txt`
- `moe_token_distribution_perf.txt`
- `wideep_context_moe_perf.txt`，仅模型需要 WideEP context 时存在
- `wideep_generation_moe_perf.txt`，仅模型需要 WideEP generation 时存在

默认运行不应保留中间调试文件。只有排查时才开启 keep-source/audit。

## 4. Source Health Guard 是通用保护层，不是校准策略

source health guard 的定位：

- 保护 source 采集稳定性。
- 只选择更健康的完整 source row。
- 不直接修 final latency。
- 不使用 truth。
- 不使用平台名。

适用场景：

- 小 token kernel 太短，单次测量容易抖动。
- 长流程全量 collector 中某些 case 被运行态污染。
- no-keep 默认命令需要稳定 source，但不希望落 debug 文件。

不适用场景：

- final latency 与 truth 系统性偏差。
- materializer 语义错误。
- truth 采法不对。
- 模型 case matrix 错了。

默认策略可以是：

```text
small token:
  context token <= 32
  generation token <= 32
sessions:
  default 3
  max 5
retries:
  max 2
selection:
  stable -> median source row
  unstable -> conservative row + audit mark
```

但具体 token 阈值、session 数和 retry 数应该是模型/后端 defaults，不应写死成
所有模型统一规则。

## 5. 实机 Truth 是辅助验证，不是 AIC 运行时依赖

### 5.1 Truth 的作用

实机 truth 用于：

- 验证 AIC final latency 是否接近真实运行。
- 判断模型 case matrix 是否合理。
- 判断 source 是否稳定。
- 判断 materializer 是否需要调整。
- 判断某个 recorded 方法能否跨硬件泛化。
- 判断某个 recorded 方法在 SGLang/backend 版本变化后是否仍保持同一语义。

truth 不用于：

- collector 运行时。
- source row 选择。
- materializer 在线反推。
- 直接给新硬件打平台 scale。

### 5.2 Truth 也要因 family 而异

truth 不是只记录一个 CSV 数值。对于可复用 skill 来说，truth 必须同时记录：

- 实机测试脚本。
- 启动命令。
- 模型裁剪/加载配置。
- GPU 可见卡数和 EP 约束。
- SGLang 代码打点位置。
- 打点代码 diff 或 patch。
- profiler/nsys/torch profiler 使用方式。
- SGLang/backend 版本迁移检查结果。
- kernel/window 选择规则。
- 过滤规则。
- 解析脚本和输出目录。

这样后续 SGLang 版本迭代时，才能先判断路径是否变了，再把同一套语义的
打点重新打到新版本里，并复现同一语义的 truth。

SGLang/backend 版本变化的风险：

- MoE Python 路径可能从 `fused_moe_triton/layer.py` 移到模型专属文件或
  piecewise cuda graph 实现。
- WideEP/DeepEP dispatcher 可能换类名、参数或默认路径。
- `/start_profile` 的 step/profile 语义可能变化。
- cuda graph capture/replay 行为可能变化。
- NVTX marker 可能只在 client 侧存在，没有进入 server worker。
- kernel 名称、`graphNodeId`、`CUDA_GRAPH_NODE_EVENTS` 结构可能变化。

因此换 SGLang/backend 版本时，不能直接沿用旧 truth 采点假设；至少需要做
instrumentation smoke 和 parser evidence 检查。

ordinary MoE truth：

- 通常可以沿用已有实机 MoE truth 测试方法。
- 要明确 context/generation、token、EP/EPLB、rank-local/global 口径。
- 要记录 SGLang 中普通 MoE routed compute 的打点位置，至少包括：
  - Python 调用入口。
  - fused MoE kernel 调用附近。
  - context/generation 分支差异。
  - 是否开启 cuda graph。
- 如果使用 torch profiler，要记录 profiler window 如何包住 routed compute，
  以及如何排除 attention、shared expert、routing/topk 等非目标部分。

WideEP context truth：

- 需要确认实机运行路径确实走 WideEP/DeepEP context。
- 需要确认没有把普通 MoE 或其他路径混进来。
- 要记录 WideEP/DeepEP context 在 SGLang 里的调用路径：
  - DeepEP dispatch 前后。
  - routed expert compute 前后。
  - combine 前后。
  - EPLB on/off 对路径的影响。
- 如果 truth 目标是 routed/compute，需要明确是否排除 dispatch/combine；
  如果目标是端到端 WideEP context，也要明确包含哪些部分。

WideEP generation truth：

- 如果开启 cuda graph，torch profiler 可能抓错窗口。
- 这类场景应优先考虑 nsys semantic profile-window。
- truth 应对齐 routed/compute，不应包含 dispatch/combine/attention/shared/topk/
  NCCL/NVSHMEM。
- 要记录 SGLang 打点和 nsys 解析的完整方法：
  - 进入 WideEP generation/decode MoE 的 Python/CUDA 调用位置。
  - NVTX range 或可识别的 profile-window 命名。
  - 目标 layer 范围。
  - routed compute kernel 白名单。
  - dispatch/combine/DeepEP/NCCL/NVSHMEM/attention/shared/topk 黑名单。
  - cuda graph 开启时如何确认窗口语义没有跨 step。
  - nsys 输出文件、解析脚本、candidate 选择方法。

如果某个模型没有 WideEP generation，就不需要 WideEP generation truth。

truth 记录模板：

```text
model:
backend:
sglang version / commit:
sglang package version:
container image:
runtime image digest:
hardware:
visible GPUs:
family:
phase:
EP list:
EPLB list:
token list:
model loading / layer truncation:
truth target semantic:
sglang instrumentation files:
instrumentation patch / diff:
profiler:
profile command:
parser:
parser assumptions:
kernel include rules:
kernel exclude rules:
aggregation:
output dir:
known limitations:
```

### 5.3 Truth 覆盖与冻结

每轮校准前应明确：

- 使用哪份 truth。
- truth 覆盖哪些 family/EP/EPLB/token。
- 哪些点缺失。
- 缺失点是否影响当前结论。

一旦确认为 latest frozen truth，后续 gate 应固定使用，不要边调 AIC 边随意换。

## 6. Gate 与误差归因

Gate 输入：

- AIC final compact latency。
- frozen truth。

Gate 输出：

- overall MAPE。
- per platform MAPE。
- per family MAPE。
- p90 / max error。
- over-threshold 点。
- missing truth points。
- extra AIC points。

遇到误差点时按顺序归因：

1. AIC 点和 truth 点是否真的是同一语义？
2. family/phase/token/EP/EPLB 是否匹配？
3. 模型是否真的支持这个 case？
4. AIC source 是否稳定？
5. no-keep 与 keep-source final latency 是否一致？
6. source feature 是否完整进入 materializer？
7. materializer 是否错误使用了二级输出或错误字段？
8. SGLang/backend 版本是否改变了 runtime path、source schema 或 profiler API？
9. truth 是否不稳定或采法不对？
10. 是否缺硬件或 runtime 版本覆盖导致外推不稳？
11. 是否需要新增薄的、平台无关且版本兼容的 source/shape/curve 约束？

不要一上来就：

- 写平台名特判。
- 改 truth 来适配 AIC。
- 用一次异常点拟合厚策略。
- 把 source 直接当 final latency。

## 7. 跨硬件泛化策略，以及 Runtime 版本边界

目标：

- 新硬件原则上只跑 AIC collector，就能得到可用 recorded latency。
- 新 SGLang/backend 版本原则上也应先走 recorded AIC 默认路径，但必须先做
  runtime compatibility 检查；通过后作为该版本自己的校准 scope。
- truth 用于证明和持续校准，而不是新硬件运行 AIC 的必要前置。

判断一个方法是否具有泛化方向：

- 不含硬件平台名特判。
- 依赖当前硬件实际 source，而不是固定 H20/H100 参数。
- materializer 使用模型语义和 source shape。
- source/materializer 契约在当前 runtime 语义内稳定。
- source health guard 只看测量稳定性。
- H20/H100 或更多硬件 fit-all 表现稳定。
- leave-one-platform 差时能解释是否来自 truth 缺失或外推覆盖不足。

有限硬件下的现实判断：

- 两类硬件只能说明初步泛化。
- 如果 H100 缺 EP8 truth，不能声称 EP8 已完全验证。
- 如果新 SGLang/backend 版本没有重新验证 marker/parser，不能声称 truth 语义
  已验证。
- 不把旧 runtime 和新 runtime 的点混在一起拟合同一个 calibration；除非已经
  证明 source/materializer/truth 语义完全等价，并且目标是明确的兼容性研究。
- fit-all 收敛但 leave-one-platform 不理想时，要继续补硬件 truth 或减少外推。

## 8. 两个 Skill 的职责

### 8.1 `moe-recorded-aic-porting`

这个 skill 负责把 AIC recorded 模式迁移到不同 MoE 模型，或迁移到同模型的
新 SGLang/backend runtime 版本。迁移完成不只是 collector 跑通，还要在
bring-up 硬件上完成 AIC final latency vs 实机 truth 的语义验证。

输入：

- 目标模型。
- 后端。
- 当前 collector 代码。
- 模型 config。
- 需要支持的 case matrix。

输出：

- 模型 recorded case matrix。
- collector source 覆盖检查。
- materializer 输入契约检查。
- compact 文件 schema 检查。
- 默认命令或模型专用命令。
- bring-up 硬件上的 AIC vs truth gate 结果。
- 不收敛点的 source/materializer/truth 归因。

核心动作：

1. 读取模型结构和后端能力。
2. 如果是新 SGLang/backend 版本，先做 runtime compatibility 检查。
3. 生成模型/版本专属 case matrix。
4. 判断哪些 family 需要启用。
5. 接入或复用 `moe_token_distribution`。
6. 接入 ordinary MoE source。
7. 如果模型支持 WideEP，再接入 WideEP source。
8. 接入 materializer。
9. 接入 clean latency。
10. 验证 no-keep 默认输出。
11. 在 bring-up 硬件上用实机 truth 验证 final compact latency。
12. 如果同模型新版本在 bring-up 硬件上 recorded AIC 验证失败，先按 2.4 的
    流程解决，暂不进入跨硬件。

特别强调：

- 不是所有模型都需要 `wideep_context`。
- 不是所有模型都需要 `wideep_generation`.
- 不要固定 DeepSeek-V3 的 token/EP 列表。
- 不要把 DSV3 小 token 修复当成所有模型默认规则。
- 新 runtime 版本验证失败时，先当作迁移/兼容问题处理，不要交给跨硬件校准。
- 允许模型相关参数调整，但必须来自模型结构、backend source 或 runtime source
  语义；不能改掉 recorded AIC 的主方法，也不能做点级 truth 拟合。

适配后 truth gate 不收敛时，按以下顺序处理：

1. case 是否对齐：model config、family/phase/token/EP/EPLB、quant/backend flag、
   cuda graph mode。
2. source 是否对齐：source schema、replay shape、source health guard、
   no-keep/keep-source final latency。
3. materializer 是否对齐：模型参数、expert/topk/shared expert 假设、source
   到 final latency 的转换。
4. truth 是否可信：SGLang marker 是否在 server worker、是否只包 routed compute、
   parser evidence 是否属于当前 runtime、kernel 过滤是否正确。
5. runtime/kernel 是否真的变化：如果算子实现确实变化且 truth 证实，再为该
   模型/runtime 语义增加薄兼容适配。

建议 references：

- `references/model-capability-matrix.md`
- `references/collector-source-contract.md`
- `references/token-ep-eplb-semantics.md`
- `references/materializer-contract.md`

### 8.2 `moe-recorded-cross-hardware-calibration`

这个 skill 负责在固定 SGLang/backend runtime 语义内，跨不同硬件验证 recorded
AIC 的泛化性，并收敛与实机 truth 的误差。

它和 `moe-recorded-aic-porting` 使用同一套 recorded 方法论：

- 先确认模型能力。
- 再确认 case matrix。
- 再跑 AIC source/materializer/clean latency。

区别在于目标不同：

- porting 的目标是“这个模型能不能正确产生 recorded AIC 数据”。
- calibration 的目标是“这个 recorded AIC 数据在当前 runtime 版本内、不同硬件上
  是否能代表实机，误差能否收敛，并最终形成可发布的数据”。

因此 calibration skill 需要更重视：

- 实机 truth 方法本身是否正确。
- SGLang 打点是否可复用。
- 不同硬件的 AIC 数据是否同口径。
- 当前 SGLang/backend 版本的 AIC source 和 truth marker 是否同语义；如果版本
  变化且尚未验证，先回到 porting/compatibility 检查，不能在本 skill 内修
  版本问题。
- gate 结果是否能解释。
- 收敛策略是否平台无关。

输入：

- 已经接入 recorded AIC 的模型。
- 一个或多个硬件环境。
- 一个固定 SGLang/backend runtime 版本；多个版本应拆成多个校准 scope。
- AIC 输出目录。
- 实机 truth 测算脚本、SGLang 打点 patch 和 truth 结果。
- gate 脚本或 gate 方法。
- 当前要验证的 family group：
  - ordinary MoE
  - WideEP/DeepEP MoE，可选
- 当前硬件的资源限制：
  - 可见 GPU 数。
  - 可测 EP 上限。
  - 是否能测 EP8/EP16 等大 EP。
- 当前 runtime 版本应是确定输入；如果 MoE 路径、cuda graph、profile API 或
  nsys parser evidence 尚未确认，本 skill 暂停，回到 porting skill。

输出：

- 当前 runtime 版本内每个硬件的 AIC compact 数据。
- frozen truth manifest。
- SGLang truth instrumentation manifest。
- gate 结果。
- 误差归因报告。
- 可合入系统数据目录的 compact 文件。

核心动作：

1. 固定代码、SGLang/backend 版本、镜像、容器、硬件、数据目录。
2. 读取或生成模型 case matrix。
3. 以 DeepSeek-V3 这类已接入模型作为参考例子跑通硬件泛化流程。
   后续换模型时，先由 porting skill 改 case matrix 和 source contract。
4. 确认已有硬件 baseline：
   - 已经确认过的 H20/H100 AIC compact 数据直接复用。
   - 已经确认过的 H20/H100 frozen truth 直接复用。
   - 不因为来了新硬件就默认重跑 H20/H100。
   - 如果 runtime 语义尚未验证，先回到 porting/compatibility 流程；本 skill
     不处理版本迁移。
   - 只有发现旧 baseline 口径错误或 truth 覆盖不足时，才安排旧硬件补测或重测。
5. 在新硬件上按模型 case matrix 跑 AIC。
6. 检查新硬件 AIC 默认行为：
   - total_errors。
   - compact 文件是否齐全。
   - no-keep 是否不落 debug/source/audit。
   - keep-source 是否只影响落盘，不影响 final latency。
7. 在新硬件上设计并执行对应实机 truth。
8. 对新硬件 truth 方法留痕：
   - SGLang 版本。
   - SGLang package 版本、commit、镜像 digest。
   - SGLang 打点文件。
   - patch/diff。
   - profiler 命令。
   - parser。
   - parser 对 runtime 版本的假设。
   - kernel include/exclude 规则。
9. 冻结新硬件 truth。
10. 做 final-output-only gate：
   - 新硬件 AIC vs 当前 runtime 对应 truth。
   - 旧硬件 AIC vs 旧硬件 frozen truth。
   - 同一 runtime 内的新旧硬件放在同一套 gate 报告里看泛化性。
11. 同时对比 recorded 与旧模式：
   - balanced。
   - power_law。
   - uniform。
   - 其他历史 baseline。
12. 判断是否收敛：
   - 新硬件误差可以适度放宽。
   - 但 recorded 应明显优于旧模式，至少不能比旧模式更差。
   - max error 不能非常大；若出现极端点，必须先归因。
13. 对误差点做 source/materializer/truth 归因。
14. 设计同一 runtime 内新硬件 + 旧硬件都适用的泛化 MoE 测算方式。
15. 先做离线回放：
   - 使用当前 runtime 的新硬件 AIC source。
   - 使用当前 runtime 的旧硬件 AIC source。
   - 使用所有 frozen truth。
   - 验证新策略在同一 runtime 的新旧硬件上同时收敛。
16. 离线回放收敛后，再合入正式 collector/materializer 逻辑。
17. 重跑 AIC 默认命令。
18. 再做 final-output-only gate。
19. 收敛后发布 compact 数据。

推荐执行粒度：

```text
P0: 固定旧硬件 baseline，不重跑已确认 H20/H100
P1: 新硬件按当前 runtime 的 case matrix 跑 AIC 默认命令
P2: 新硬件跑对应实机 truth，并沉淀 SGLang 打点/脚本/parser
P3: 新硬件 AIC vs truth gate，同时复用同 runtime 旧硬件 frozen gate
P4: recorded vs balanced/power_law/uniform 等旧模式对比
P5: 误差归因，设计同 runtime 新旧硬件都适用的泛化策略
P6: 离线回放验证新策略
P7: 离线收敛后合入正式逻辑，重跑 AIC
P8: final gate 收敛后发布 compact 数据和 PR 白名单
```

DeepSeek-V3 可作为硬件泛化验证模板：

```text
AIC command:
  python3 collect.py --backend sglang --model-path deepseek-ai/DeepSeek-V3 \
    --ops moe_token_distribution moe wideep_moe --keep-csv

ordinary MoE:
  context + generation

WideEP/DeepEP MoE:
  context + generation

single-card EP simulation:
  默认启用

EPLB:
  按 WideEP/后端能力展开

truth:
  ordinary 使用普通 MoE truth 方法
  WideEP generation 使用 nsys semantic routed/compute truth
```

换到其他模型时：

- 不要照抄 DSV3 token/EP/WideEP 列表。
- 先用 porting skill 生成该模型自己的 case matrix。
- calibration skill 只负责在这个 case matrix 上做硬件泛化验证。

特别强调：

- 该 skill 可以使用 truth，但 truth 只用于离线验证。
- 它不是“给每个硬件拟合一个 scale”。
- 它要证明 recorded 方法在不同硬件上尽量泛化。
- 缺 truth 覆盖的点要明确标注，不能假装已验证。
- 如果新硬件暂时不能跑 truth，可以先产出 AIC 数据，但只能说“未完成
  truth gate”，不能声称已校准通过。
- 如果某个模型没有 WideEP，就不要生成 WideEP truth/gate 工作项。

建议 references：

- `references/truth-strategy.md`
- `references/nsys-wideep-generation-truth.md`
- `references/sglang-instrumentation.md`
- `references/gate-and-error-attribution.md`
- `references/cross-hardware-generalization.md`
- `references/data-publishing.md`

## 9. 可归档脚本目录

为了后续把这套方法沉淀成 skill，临时日期脚本应该收敛到一个无日期、
按职责划分的小目录。当前建议归档目录为：

```text
tools/moe_calibration/recorded_moe/
  truth/
    run_truth_family.sh
    run_truth_stage.sh
    presets/
      deepseek_v3.sh
    sglang_instrumentation/
      sglang_moe_profile_markers.md
      fused_moe_triton_layer_reference.patch
    nsys/
      run_wideep_generation_client.py
      parse_wideep_generation_event_nodes.py
      select_wideep_generation_candidate.py
  gate/
    evaluate_clean_latency.py
    verify_generalization_manifest.py
  replay/
    analyze_source_contract.py
    analyze_generalization_flow.py
```

这些脚本的定位如下：

- `truth/run_truth_family.sh`：truth 入口封装。按 `PLATFORM`、`FAMILY`、
  `PRESET` 调起具体 truth stage。
- `truth/run_truth_stage.sh`：普通 MoE / WideEP MoE 的 stage-based truth
  执行骨架。它保留 EP、EPLB、token、session、profile layer、parse worker
  等关键控制项，后续迁移模型时可以复用骨架，只替换 preset 和 profile
  语义。
- `truth/presets/deepseek_v3.sh`：DeepSeek-V3 的参考 preset。它记录 DSV3
  当前已确认的 token/EP/default layer 选择，但不代表所有 MoE 模型都要照抄。
- `truth/sglang_instrumentation/sglang_moe_profile_markers.md`：SGLang MoE
  truth 打点契约。它明确默认 SGLang 可能没有 AIC marker，换版本或换镜像时
  必须重新定位 MoE 路径、重新打 marker，并用 nsys SQLite 验证 marker 没打错。
- `truth/sglang_instrumentation/fused_moe_triton_layer_reference.patch`：
  `fused_moe_triton/layer.py` 的参考 patch。它不是所有版本都能无脑 apply，
  但记录了 helper、env flag、`dispatch/compute/combine` 和
  `aic_nsys/layer_N/routed/compute` 应该如何放置。
- `truth/nsys/run_wideep_generation_client.py`：WideEP generation nsys truth
  客户端参考。用于开 cuda graph 时绕开 torch profiler 窗口错配问题，通过
  SGLang server + NVTX/profiler 控制采集 generation routed/compute 语义窗口。
- `truth/nsys/parse_wideep_generation_event_nodes.py`：从 nsys 导出的事件节点中
  解析 WideEP generation 候选窗口。
- `truth/nsys/select_wideep_generation_candidate.py`：从候选窗口中选择 semantic
  routed/compute truth 值。
- `gate/evaluate_clean_latency.py`：final-output-only gate。只读取 collector
  顶层 compact `*.txt`，不走离线 materializer 反推，避免 gate 口径和线上
  collector 口径不一致。
- `gate/verify_generalization_manifest.py`：检查 frozen truth、AIC 输入目录、
  raw/source/compact 数据是否存在，适合在正式 gate 前做输入审计。
- `replay/analyze_source_contract.py`：source/materializer 契约审计。它用于
  离线确认 source 表能被正式 materializer 正确读取，并可选地对 frozen truth
  做误差归因。
- `replay/analyze_generalization_flow.py`：泛化回放 gate。它把 recorded compact
  和 balanced/power_law/uniform 等 baseline 点放在同一套 truth 上比较，用于
  合入正式 collector 前的离线验证。

使用原则：

- 这些脚本可以直接作为后续 skill 的参考实现。
- 新模型迁移时优先改 preset/case matrix，不要改 gate 口径。
- 新硬件校准时优先新增 manifest 和输入目录，不要写平台名特判。
- 如果目标镜像的 SGLang 没有 AIC marker，先按
  `truth/sglang_instrumentation/` 重新打点和验证，再跑 truth；不要只凭 nsys
  里有 kernel 就选 truth。
- DSV3 的 nsys WideEP generation truth 是一个可复用范例，但其他模型如果没有
  WideEP generation，就不需要执行这部分。
- 旧的带日期脚本可以保留为历史追溯；正式 skill 和后续文档应优先引用
  `recorded_moe/` 目录。

## 10. 两个 Skill 的协作方式

推荐流程：

1. 先用 `moe-recorded-aic-porting`：
   - 让模型支持 recorded AIC。
   - 明确 case matrix。
   - 跑通 collector/source/materializer/clean latency。
2. 再用 `moe-recorded-cross-hardware-calibration`：
   - 旧硬件数据已经确认时，直接复用旧硬件 AIC compact 和 frozen truth。
   - 不默认重跑 H20/H100。
   - 在新硬件上跑 AIC。
   - 在新硬件上跑对应实机 truth。
   - 把新硬件 truth 与旧硬件 frozen truth 放进同一套 gate 里。
   - 对比 recorded 与 balanced/power_law/uniform 等旧模式。
   - 判断 recorded 是否比旧模式更好，误差是否在可接受范围内。
   - 如果误差不收敛，分析 source/materializer/truth 归因。
   - 设计新硬件 + 旧硬件都适用的泛化 MoE 测算方式。
   - 先做离线回放，确认新旧硬件同时收敛。
   - 离线回放收敛后再合入正式 collector/materializer 逻辑。
   - 重跑 AIC 默认命令并 gate。
   - 收敛后发布 compact 数据。

增量硬件验证流程：

```text
已确认旧硬件:
  复用 H20/H100 AIC compact
  复用 H20/H100 frozen truth
  不重跑，除非发现口径错误或代码语义变化

新硬件:
  跑 recorded AIC
  跑对应实机 truth
  记录 truth 脚本、SGLang 打点、patch、parser、profile 命令
  做 AIC vs truth gate

对比:
  recorded vs truth
  recorded vs balanced
  recorded vs power_law
  recorded vs uniform
  新硬件 + 旧硬件统一看泛化性

收敛:
  先离线回放
  离线收敛后合入正式逻辑
  重跑 AIC
  gate 收敛后发布 compact 数据
```

判断该用哪个：

- “这个模型怎么接 recorded？”用 porting。
- “这个硬件上准不准？”用 calibration。
- “WideEP 是否需要测？”先用 porting 判断模型能力，再用 calibration 设计 truth。
- “某个点误差大怎么办？”用 calibration。
- “source 字段缺失/口径错？”通常先回 porting 修契约。

收尾要求：

- 原有 frozen truth、truth 测试脚本、SGLang 打点 patch、profile/parser
  方法要单独整理并提交到仓库，作为后续复现和版本迁移参考。
- 离线回放脚本也要固化，避免每次校准都临时写一次。
- 发布 compact 数据时，只提交项目需要读取的数据文件；临时 profile 输出、
  raw run 目录和中间 CSV 不进主线数据目录。

## 11. 从本次 DSV3 经验抽出的通用教训

这些经验可以作为例子，但不要变成所有模型的硬编码规则。

1. `COLLECTOR_DSV3_KEEP_LATENCY_SOURCES` 只能控制是否落盘，不能影响 final latency。
2. no-keep 与 keep-source 不一致时，优先查 source feature 是否被错误 gate。
3. 小 token 容易被运行态污染，source health guard 是合理保护层。
4. WideEP generation 开 cuda graph 时，torch profiler truth 容易抓错窗口。
5. WideEP generation truth 应优先用 nsys semantic routed/compute。
6. AIC source 不一定接近 truth，final compact latency 才用于 gate。
7. token/EP/EPLB 口径错配会导致“看起来像校准问题”的假问题。
8. 文档、数据目录、容器路径、硬件可见 GPU 数都必须留痕。
9. PR 中只提交项目运行所需代码和数据；实验流水、gate 脚本、skill 草稿可拆开提交。

本次 DSV3 结果可以作为后续 skill 的示例材料：

```text
overall MAPE ~= 4.88%
主要残差集中在 WideEP context token8
H100 缺 EP8 truth，外推结论需标注
```

但 skill 主体不应依赖这些固定数字。
