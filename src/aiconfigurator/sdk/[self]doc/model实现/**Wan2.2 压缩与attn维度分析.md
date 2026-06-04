我们可以将整个视频序列的维度变化过程、SGLang 的 Config 设计体系以及二者的联动机制剖析清楚。

### 一、 视频压缩及维度映射全过程（从像素到 100K 序列）

以你提到的 **Wan 2.2 视频生成 (720x1280, 121帧)** 为例，数据从像素空间走向 Transformer 的序列空间，共经历了 **VAE 压缩** 和 **DiT Patch 化** 两个核心步骤。

#### 1. VAE 压缩 (像素空间 -> 潜空间 Latent Space)
在原始 `Wan-AI--Wan2.2-T2V-A14B-VAE_config.json` 和 SGLang 的 `WanVAEArchConfig` 中：
*   **输入维度**: $(B, 3, 121, 720, 1280)$ 即 (Batch, 通道, 帧数, 高, 宽)
*   **空间下采样 (Spatial)**:
    由 `dim_mult: [1, 2, 4, 4]` 决定。序列长度为 4 意味着网络包含 3 个下采样阶段（通常每个阶段长宽各缩小一半），因此空间整体下采样率为 $2^3 = 8$。
    所以：$H_{latent} = 720 / 8 = 90$，$W_{latent} = 1280 / 8 = 160$。
*   **时间下采样 (Temporal)**:
    由 `temperal_downsample: [false, true, true]` 决定。3 个下采样块中，后两个块在时间维度进行了下采样，因此时间整体下采样率为 $2^2 = 4$。
    这里有一层因果卷积的边界处理，其推导公式为 $(T - 1) / 4 + 1$。
    所以：$T_{latent} = (121 - 1) / 4 + 1 = 31$。
*   **通道维度 (Channels)**:
    由 `z_dim: 16` 决定，也就是输出 16 个通道。
*   **VAE 输出**: 一个形状为 $(B, 16, 31, 90, 160)$ 的 5D Tensor。

#### 2. DiT Patch 化 (潜空间 -> Token 序列空间)
潜变量输入到 `WanTransformer3DModel`。根据原生 DiT config `Wan-AI--Wan2.2-T2V-A14B_config.json` 和 SGLang 中的配置：
*   **Patch切分**: `patch_size: [1, 2, 2]`，意味着时间维度不再进行 patch 合并，但空间维度每 2x2 个网格合并为一个 Token。
*   因此，最终用于 Self-Attention 的 Token 维度：
    *   时间帧数：$31 / 1 = 31$
    *   高度 Token 数：$90 / 2 = 45$
    *   宽度 Token 数：$160 / 2 = 80$
*   **最终序列长度**: $Seq\_Len = 31 \times 45 \times 80 = 111,600$。
*   **通道维度投影**: Patch 内部的 $1 \times 2 \times 2 \times 16(\text{in\_channels}) = 64$ 维数据会被通过 Linear 层直接投影为 DiT d 的模型隐藏维度（即 `dim: 5120` 和 `num_heads: 40 * 128`）。计算时张量形为 `(B, 111600, 5120)`。
