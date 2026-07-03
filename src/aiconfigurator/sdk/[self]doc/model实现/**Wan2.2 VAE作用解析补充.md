这是一个非常关键且经典的疑问！直接给出结论：

**在推理管线中，绝对不会去生成原始视频规格（121x720x1280）的随机噪声像素张量再做 VAE Encode。**
真实的流程是：**直接在内存中初始化一个潜空间随机 Latent 张量**，然后直接送入 DiT 进行去噪。对 A14B T2V/I2V 来说典型 latent grid 是 `31x90x160`；对 TI2V-5B，SGLang `Wan2_2_TI2V_5B_Config.prepare_latent_shape()` 明确使用 `vae_stride=(4,16,16)`，典型 DiT 输入 grid 是 `31x44x80`。VAE 在纯文本生成视频（T2V）任务中，**只参与最后一步的 Decode（逆向解压回像素）**。

只有在**图生视频（I2V）**或**视频生视频（V2V）**中，VAE 的 Encode 功能才会被调用，用来把你的“参考图”或者“参考视频”压进潜空间作为 Condition 条件。

下面我结合 SGLang 的核心代码管线为你做透彻的证据推演：

### 证据 1：噪声的直接潜空间初始化 (Latent Preparation)

整个推理的拓扑定义在 `wan_pipeline.py` 中，请求依次经过 `LatentPreparationStage` -> `DenoisingStage` -> `DecodingStage`。

我们在 `sglang/multimodal_gen/runtime/pipelines_core/stages/latent_preparation.py` 可以看到随机噪声是如何被创造出来的：
```python
class LatentPreparationStage(PipelineStage):
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        # 1. 并没有基于像素空间生成 tensor
        if latents is None:
            # 获取的是 latent 专属的 shape 尺寸
            shape = server_args.pipeline_config.prepare_latent_shape(
                batch, batch_size, num_frames
            )
            # 2. 直接根据潜空间 shape 生成了符合正态分布的随机 Latent (randn_tensor)
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )

        batch.latents = latents
        return batch
```
这个 `prepare_latent_shape` 的具体定义需要分两类看。A14B T2V/I2V 走 `PipelineConfig.prepare_latent_shape()`，空间压缩来自 `vae_config.arch_config.spatial_compression_ratio=8`；TI2V-5B 在 `pipeline_configs/wan.py` 中重写了该函数：
```python
    def prepare_latent_shape(self, batch, batch_size, num_frames):
        F = num_frames
        z_dim = self.vae_config.arch_config.z_dim
        vae_stride = self.vae_stride
        oh = batch.height
        ow = batch.width
        # TI2V-5B 中 vae_stride=(4,16,16)
        shape = (batch_size, z_dim, F, oh // vae_stride[1], ow // vae_stride[2])
        return shape
```
**结论证明**：DiT 拿到手的初始画布，生来就是抽象高维噪声；A14B 常见为 `(B,16,31,90,160)`，TI2V-5B 常见为 `(B,48,31,44,80)`。根本不存在所谓的“生成纯随机像素然后再压缩”的脱裤子放屁开销。

---

### 证据 2：VAE Decoding (解压回现实)

上述的随机 `latents` 在 `DenoisingStage` 中被 DiT 循环去噪 50 步之后，最终会流转到管线的最后一环 `DecodingStage`。
查看 `pipelines_core/stages/decoding.py`：
```python
class DecodingStage(PipelineStage):
    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        # 调用 VAE Decoder，将去噪完毕的 Latent 解码为 Pixel
        frames = self.decode(batch.latents, server_args)
        ...
```
此时，调用你刚才看过的 `wanvae.py` 里面的 `decode` 方法（通常会触发 `parallel_tiled_decode` / `spatial_tiled_decode`），将潜空间特征彻底反卷积、上采样（Upsample），还原出肉眼可见像素帧。这里要特别注意：VAE arch 本身仍是 `(4,8,8)` 压缩/解压结构；TI2V-5B 的 `(4,16,16)` 是 DiT latent 准备路径里的 profile stride，不能和 VAE kernel 维度混为一谈。

---

### 证据 3：VAE Encoding 什么时候起作用？(I2V 条件注入)

你可能会问：既然 T2V 不需要，那写了那么长 VAE Encode 代码干嘛？答案是给 **条件控制 (Conditioning)** 用的。
如果在 I2V 任务中，你需要提供一张起手图（比如 1 张 720x1280 的图片），引擎会调用 `ImageVAEEncodingStage` (位于 `image_encoding.py`)：

```python
class ImageVAEEncodingStage(PipelineStage):
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if batch.condition_image is None:
            return batch
        ...
        for image in images: # 原图 720x1280
            # 将那一张图复制延展为伪视频形状或用 0 填充其它帧 (num_frames-1)
            video_condition = torch.cat(
                [image, image.new_zeros(..., num_frames - 1, ...)], dim=2
            )
            # 通过真刀真枪的 VAE Encode 将原图压缩进潜空间
            video_condition = self.vae.encode(video_condition)
            condition_latents.append(video_condition)
```
**看，只有作为“现实约束条件”存在的图片或视频，才是真实的像素输入，必须经过 `VAE Encode` 压缩成条件潜变量 `condition_latents`。**

### 全局推演流程图

**Text-to-Video (T2V) 的生命周期：**
1. (随机生成阶段) $\to$ **直接** `randn` 出 Latent 噪声；A14B 典型 `(16,31,90,160)`，TI2V-5B 典型 `(48,31,44,80)`
2. (去噪阶段) $\to$ 注入文本，经过 50 次 DiT 算出干净的 Latent
3. (解码阶段) $\to$ 干净的 Latent 投入 **VAE Decode**，输出最终像素 `(3, 121, 720, 1280)`。

**Image-to-Video (I2V) 的生命周期：**
1. (条件编码) $\to$ 用户输入参考大图，投入 **VAE Encode** 得到 Condition Latent
2. (随机生成阶段) $\to$ **直接** `randn` 出 Latent 噪声
3. (去噪阶段) $\to$ 将 Condition Latent 和 噪声 Latent 拼接/结合，通过 DiT 去噪
4. (解码阶段) $\to$ 干净的 Latent 投入 **VAE Decode**，输出最终像素。

因此，你的疑惑解开了：框架非常聪明，不会徒增巨大且无意义的编码算力。

### 2026-05-19 校正：为什么会出现 `31*45*80` 与 `31*22*40` 两种答案？

- A14B T2V/I2V：VAE arch stride `(4,8,8)`，DiT patch `(1,2,2)`，所以 `720x1280x121 -> 31x90x160 -> 31*45*80=111600` tokens。
- TI2V-5B：SGLang 官方 `Wan2_2_TI2V_5B_Config.prepare_latent_shape()` 使用 profile 字段 `vae_stride=(4,16,16)`，并且 HF DiT config 为 `in_dim=48,out_dim=48`，所以 `704x1280x121 -> 31x44x80 -> 31*22*40=27280` tokens。
- VAE encode 的职责仍只服务 I2V/TI2V 条件图像；VAE decode 仍负责最终视频还原。AIC 代码已拆分 `latent_prepare_stride` 与 `vae_stride`，避免把 TI2V DiT 输入 shape 错套到 VAE kernel，或把 A14B `31*45*80` 错套到 TI2V。
