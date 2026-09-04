# MiniMax-M3 MSA KernelSim 接入说明

## 已接入内容

MiniMax-M3 的 MSA ANALYTICAL 路径现已接入 `aic-core` KernelSim：

- `sdk/kernelsim/msa/model.py`：H100 单卡 BF16 Triton index-score 模型；
- `sdk/kernelsim/analytical.py`：AIC 单位转换和硬件字段适配；
- `sdk/operations/msa.py`：ANALYTICAL 下的 granular 组合路径；
- `sdk/models/minimax_m3.py`：显式区分前 3 个 dense layer 和后 57 个 sparse MSA+MoE layer。

ANALYTICAL MSA 的组合边界为：

```text
fused Q/K/V/index projection GEMM
+ selected-block GQA（复用 FA 代理）
+ MSA index-score（含 128-token block max）
+ TopK/page-table 的 launch-aware ElementWise 近似
+ output projection GEMM
```

上下文长度不超过 M3 的 2048-token selected ceiling 时，index-score 和 TopK 走后端的 select-all 快路径，不重复计入稀疏选择成本。

## 有意保留的限制

MSA index 参数来自单张 H100 SXM 的 BF16、`head_dim=128`、`block_size=128` 实验，只提供 `low/standard/high` 三个工程档位。它不宣称适用于其他 GPU、SGLang 版本、后端或 dtype。

TopK 没有建立独立拟合模型。由于其单次成本相对 index 和主 attention 较小，当前用带启动 floor 的 ElementWise/访存近似处理；这不是 TopK 实测查表，也不应解读为排序 kernel 的精确性能。

selected-block GQA 当前复用 FA 模型，未显式拟合随机 block gather、页表访问和 sparse split-K 的 backend 特性。

执行 MSA ANALYTICAL 时会发出 `MsaAnalyticalApproximationWarning`，明确提示上述三项限制。H100 校准参数也会发出 `MsaIndexModelWarning`。

SOL、EMPIRICAL、HYBRID 和 SILICON 的原有 MSA 行为保持不变；本次没有用新的 H100 index 模型伪造 silicon 数据，也没有把 DSA TopK 参数迁移到 M3。

## 验证

```text
236 passed
git diff --check passed
```

验证覆盖 MSA index 的 prefill/decode 快照、三档参数、decode 固定 head tile、M3 的 3/57 层计数、ANALYTICAL source 标记，以及既有 DSA/FA/MLA analytical 回归。
