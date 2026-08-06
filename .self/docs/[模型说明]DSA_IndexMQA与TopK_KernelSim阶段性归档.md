# DSA Index MQA 与 TopK KernelSim 阶段性归档说明

## 1. 归档目的

本次将 DSA 稀疏索引阶段的两个 kernel 模型归档到：

```text
aic-core/src/aiconfigurator_core/sdk/kernelsim/dsa/
```

包括：

- FP8 Index MQA logits；
- FP32 score TopK/index transform；
- `standard/high/low` 三档工程参数；
- shape、结果 breakdown、运行时范围警告和单元测试。

这些模型供 DeepSeek V3.2、GLM-5.2 及后续稀疏 attention 架构的 granular 建模使用。本轮只完成独立 KernelSim 成品归档，尚未接入 AIC operation、collector 或 ANALYTICAL 自动分发。

## 2. 成熟度声明

该版本是**阶段性的半成熟方案**，不是已经完成跨硬件验收的通用模型。

校准依据仅包括：

```text
GPU:       单张 NVIDIA H100 SXM 80GB
Framework: SGLang 0.5.12
MQA:       DeepGEMM FP8 ragged/paged MQA logits
TopK:      sgl-kernel FP32 score -> int32 index
Heads:     32 / 64
Head dim:  128
```

目前缺少：

- A100/H200/B200/RTX PRO 及国产 GPU 验证；
- SGLang 其他版本、vLLM 和其他 backend 验证；
- BF16 Index MQA 同源 kernel 数据；
- 不同 head dim、index heads 和 TopK 大小的系统验证；
- Nsight HBM/L2/TMA/Tensor Core 计数器；
- 真实模型 logits 分布，当前只有 flat/top-last 两个构造端点。

因此每次模型调用都会发出 `DsaIndexModelWarning`，结果的 `scope` 和 `warnings` 也保留相同限制。该 warning 不应在接入 operation 时被静默删除；如上层需要降噪，应按 database view 或任务只汇报一次。

## 3. Index MQA 模型

### 3.1 API 与边界

```python
from aiconfigurator_core.sdk.kernelsim.dsa import (
    IndexMqaShape,
    estimate_index_mqa,
)

shape = IndexMqaShape(
    layout="ragged",
    batch_size=4,
    query_length=256,
    context_length=32768,
    index_heads=64,
)

result = estimate_index_mqa(
    shape,
    sm_count=132,
    clock_hz=1.83e9,
    fp8_peak_flops_s=1.978e15,
    hbm_bandwidth_bytes_s=3.35e12,
    parameter_level="standard",
)
```

模型只支持当前有证据的 FP8 MQA。Paged `query_length` 对应 `next_n`，严格限制为 1 或 2。Ragged/paged 必须显式指定，不根据 phase 或硬件名称自动推断。

### 3.2 公式

Paged：

```text
T = floor(heads)
  + max(QK_flops / (FP8_peak * eta_compute),
        logical_bytes / (HBM_BW * eta_memory))
```

Ragged：

```text
T = chunk_count * floor(heads)
  + max(executed_task_flops / (FP8_peak * eta_compute),
        executed_task_bytes / (HBM_BW * eta_memory))
  + critical_kv_blocks * control_cycles(heads) / clock
```

Ragged 保留 `BLOCK_Q=16`、`BLOCK_KV=64`、`ks/ke` causal span、8M score-slot chunking 和 SM critical service。它没有把 Index MQA错误建成包含 softmax/PV 的完整 FA。

### 3.3 三档参数

| 参数 | Low | Standard | High |
| --- | ---: | ---: | ---: |
| `eta_compute` | 0.98 | 0.85 | 0.65 |
| `eta_memory` | 1.00 | 0.95 | 0.75 |
| Paged floor H32/H64 | 20/29 us | 25/37 us | 32/46 us |
| Ragged floor H32/H64 | 17/21 us | 22/27 us | 28/34 us |
| Ragged control H32/H64 | 245/180 cycles | 330/245 cycles | 430/320 cycles |

Standard 参考 H100 扩展拟合结果后进行了圆整和轻度保守化。High/low 通过同时调整 floor、效率和 control service 构造，保证跨代表 shape 的时延单调。它们是工程场景，不是统计置信区间。

非 32/64 heads 使用线性插值/外推，并在结果中附加范围警告。

## 4. TopK 模型

### 4.1 API 与边界

```python
from aiconfigurator_core.sdk.kernelsim.dsa import (
    IndexTopKShape,
    estimate_index_topk,
)

shape = IndexTopKShape(
    layout="paged",
    batch_size=128,
    query_length=1,
    context_length=65536,
    topk=2048,
    variant="fused",
    score_distribution="standard",
)

result = estimate_index_topk(
    shape,
    hbm_bandwidth_bytes_s=3.35e12,
    parameter_level="standard",
)
```

TopK 的输入固定为 FP32 score，输出为 int32 index。它没有 FP8/BF16 compute 分支。`variant` 支持 `plain` 或与 layout 对应的 fused kernel。

`score_distribution` 支持：

- `flat`：构造的平坦低分 score；
- `top_last`：高分集中在有效行尾；
- `standard`：两个端点 recipe 参数的中值，作为不知道真实 logits 分布时的默认值。

### 4.2 公式

```text
row_efficiency = min(1, query_rows / row_saturation)

T = floor
  + effective_pass_factor
    * (FP32 score bytes + int32 output bytes)
    / HBM_BW
    / row_efficiency
```

基础 recipe 按 `layout × plain/fused × score_distribution` 区分，不按具体 shape 查表。

### 4.3 三档参数

三档通过基础 recipe 的统一倍率构造：

| 参数倍率 | Low | Standard | High |
| --- | ---: | ---: | ---: |
| Floor | 0.80 | 1.00 | 1.25 |
| Effective passes | 0.75 | 1.00 | 1.30 |
| Row saturation | 0.80 | 1.00 | 1.20 |

High 同时增加启动、扫描遍数和达到充分并行所需 rows；low 做相反调整。

## 5. 归档回放结果

成品 API 已逐点回放扩展数据的 678 个 MQA 和 1196 个 TopK 测量：

| 模型 | 档位 | 全量 MAPE | Median APE | Median ratio |
| --- | --- | ---: | ---: | ---: |
| Index MQA | Low | 31.32% | 30.59% | 0.741 |
| Index MQA | Standard | 30.08% | 20.20% | 0.947 |
| Index MQA | High | 43.40% | 22.62% | 1.202 |
| TopK | Low | 35.85% | 27.68% | 0.783 |
| TopK | Standard | 35.69% | 16.78% | 1.034 |
| TopK | High | 61.51% | 41.50% | 1.370 |

这里是包含极端 shape 的全量回放，不是 holdout。High/low 的用途是给出方向明确的工程上下界，不以各自最小 MAPE 为目标。

实验阶段的固定 holdout 结果仍是：

```text
Index MQA shared task-roofline: 27.42% MAPE
TopK grouped lightweight recipe: 28.51% MAPE
```

较大的误差再次说明该版本只适合无实测数据时的量级评估、架构排序和敏感性分析，不适合替代同后端同硬件的 silicon 数据。

## 6. 测试与后续接入

新增测试：

```text
tests/unit/sdk/kernelsim/test_dsa_index.py
```

覆盖 shape 验证、ragged 手算、256-chunk 极端点、paged alignment、三档单调性、TopK layout/variant 和范围 warning。KernelSim 全量回归为 `60 passed`。

后续接入 ANALYTICAL 时建议：

1. 新增独立的 `DSAIndexScore` 与 `DSAIndexTopK` granular operation；
2. 从模型实现传递真实 ragged/paged、index heads、head dim、TopK 和 score layout；
3. 不把二者映射为普通 FA 或 norm/RoPE；
4. 每个 analytical database view 至少警告一次阶段性适用范围；
5. module fallback 必须避免将 Index MQA、TopK 与后续 sparse attention core 重复计时；
6. 多硬件数据完成前，不将这三档参数宣传为跨 GPU 已校准 profile。
