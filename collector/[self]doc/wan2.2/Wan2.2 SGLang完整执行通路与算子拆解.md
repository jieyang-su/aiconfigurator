# Wan2.2 SGLang 完整执行通路与算子拆解

本文基于 SGLang `0.5.10.post1` 官方安装版源码静态分析，不启动推理服务。参考官方 Wan2.2 cookbook：`https://docs.sglang.io/cookbook/diffusion/Wan/Wan2.2`，其部署说明覆盖 `Wan-AI/Wan2.2-TI2V-5B-Diffusers`、`Wan-AI/Wan2.2-T2V-A14B-Diffusers`、`Wan-AI/Wan2.2-I2V-A14B-Diffusers`，并强调 `--tp-size`、`--sp-degree`、`--ulysses-degree`、`--ring-degree`、CPU offload 与 Cache-DiT 等运行选项。

## 1. Pipeline 总览

| 阶段 | SGLang 0.5.10 源码入口 | Wan2.2 相关任务 | 主要张量 | 已有/新增 collector 覆盖 |
|---|---|---|---|---|
| 配置解析 | `multimodal_gen/registry.py`、`configs/pipeline_configs/wan.py` | T2V、I2V、TI2V | 采样参数、模型组件路径、precision/offload | 文档说明，不测 |
| Pipeline 组装 | `runtime/pipelines/wan_pipeline.py`、`wan_i2v_pipeline.py` | T2V 用 `WanPipeline`，I2V/TI2V 用 `WanImageToVideoPipeline` | 模块字典：text encoder、tokenizer、VAE、DiT、scheduler、可选 image encoder | 文档说明，不测 |
| 文本编码 | `stages/text_encoding.py` + `runtime/models/encoders/t5.py` | 三类任务均执行 | `input_ids/attention_mask -> prompt_embeds[B,512,4096]` | `wan_t5` + GEMM |
| 图像编码 | `stages/image_encoding.py` + `runtime/models/encoders/clip.py` | I2V/TI2V 条件图像执行 | CLIP hidden states，Wan I2V cross-attn 取 257 image tokens | `wan_clip` + GEMM |
| 图像 VAE encode | `stages/image_encoding.py::ImageVAEEncodingStage` + `wanvae.py` | I2V/TI2V 条件图像执行 | `condition_image -> image_latent` | `wan_vae` / `wan_vae_attention` / `wan_vae_elementwise` |
| Latent 准备 | `stages/latent_preparation.py` | 三类任务均执行 | A14B latent `B,16,F_lat,H/8,W/8`；TI2V-5B 官方 `prepare_latent_shape()` 为 `B,48,F_lat,H/16,W/16`，`F_lat=(F-1)//4+1` | 文档说明，不测随机数 |
| Timesteps | `stages/timestep_preparation.py` | 三类任务均执行 | scheduler timesteps/sigmas；TI2V 可扩展到 token 级 timestep | 文档说明，不测 |
| DiT denoise | `stages/denoising.py` + `runtime/models/dits/wanvideo.py` | 三类任务均执行；A14B 有 high/low expert | latent tokens、text tokens、image tokens、timestep modulation | `wan_patch_embed` / `wan_rope` / `wan_attention` / `wan_elementwise` + GEMM |
| VAE decode | `stages/decoding.py` + `runtime/models/vaes/wanvae.py` | 三类任务均执行 | `latents -> video[B,3,F,H,W]` | `wan_vae` / `wan_vae_attention` / `wan_vae_elementwise` |

## 2. 模型配置事实

| Profile | SGLang 配置类 | Task | 分辨率/帧数采样 | VAE | DiT token 设计 |
|---|---|---|---|---|---|
| Wan2.2 TI2V 5B | `Wan2_2_TI2V_5B_Config` | `TI2V` | collector 覆盖 `704x1280`，`121/81/49` 帧 | encoder+decoder；VAE arch stride `(4,8,8)`；DiT latent prepare stride `(4,16,16)` | hidden `3072`、heads `24`、head_dim `128`、FFN `14336`、patch input/output `48` |
| Wan2.2 T2V A14B | `Wan2_2_T2V_A14B_Config` | `T2V` | `720x1280`、`480x832`，`121/81/49` 帧 | decoder only | hidden `5120`、heads `40`、FFN `13824`、patch input `16` |
| Wan2.2 I2V A14B | `Wan2_2_I2V_A14B_Config` | `I2V` | `720x1280`、`480x832`，`121/81/49` 帧 | encoder+decoder | hidden `5120`、heads `40`、FFN `13824`、patch input `36`、image tokens `257` |

## 3. 文本编码器 T5 拆解

`TextEncodingStage` 调 tokenizer 后调用 `T5EncoderModel/UMT5EncoderModel`，Wan 的 `t5_postprocess_text()` 会按 attention mask 截断并 pad 到 512 tokens。

| 算子类别 | SGLang 源码载体 | 维度设计 | collector |
|---|---|---|---|
| Token embedding | `VocabParallelEmbedding` | vocab `32128`，hidden `4096`，seq `128/256/512` | `wan_t5: embedding` |
| QKV/O/FFN GEMM | `QKVParallelLinear`、`MergedColumnParallelLinear`、`RowParallelLinear` | `d_model=4096`，`num_heads=64`，`d_kv=64`，`d_ff=10240`，TP 按输出/输入 shard | `gemm` Wan-only 维度 |
| T5 attention compute | `T5MultiHeadAttention` | `q/k/v[B,S,64,64]`，`einsum(q,k)` + bias/mask + FP32 softmax + `einsum(p,v)` | `wan_t5: attention_compute_bias_softmax` |
| RMSNorm | `runtime.layers.layernorm.RMSNorm` | `B,S,4096` | `wan_t5: rmsnorm` |
| Gated FFN elementwise | `T5DenseGatedActDense.forward` | `gelu_new(wi_0) * wi_1`，shape `B,S,10240` | `wan_t5: ffn_gated_act_mul` |

## 4. 图像编码器 CLIP 拆解

I2V/TI2V pipeline 在有 `image_encoder` 与 `image_processor` 时执行 `ImageEncodingStage`。Wan 配置使用 `CLIPVisionConfig`，并在 config 中 `postprocess_image()` 取倒数第二层 hidden states。

| 算子类别 | SGLang 源码载体 | 维度设计 | collector |
|---|---|---|---|
| Patch embedding | `CLIPVisionEmbeddings.patch_embedding` | `Conv2d(3 -> 768, kernel=stride=32)`，`224x224 -> 49 patches + cls = 50 tokens` | `wan_clip: vision_patch_embed` |
| LayerNorm | `pre_layrnorm`、block LN | `B,50,768` | `wan_clip: layernorm` |
| Attention compute | `CLIPAttention.attn = LocalAttention` | heads `12`，head_dim `64`，causal=True；实际 backend 由 SGLang platform 选择 | `wan_clip: attention_compute`（collector 直接测源码等价 `torch_sdpa`，避免单测环境缺 server args） |
| MLP activation | `CLIPMLP.activation_fn=quick_gelu` | `B,50,3072` | `wan_clip: mlp_activation` |
| QKV/O/MLP GEMM | `QKVParallelLinear`、`ColumnParallelLinear`、`RowParallelLinear` | `hidden=768`，`intermediate=3072`，TP shard | `gemm` Wan-only 维度 |

## 5. Wan DiT 主干拆解

DiT collector 已覆盖主干：`PatchEmbed`、RoPE、attention local compute、fused layernorm/residual/modulation，以及 Linear/GEMM 维度。SP/USP/Ring 的通信不纳入 latency，只用通信后的本地 shape：Ulysses 近似为 `seq_len=global_seq`、`heads=heads/tp/sp`；Ring 近似为 `seq_len=ceil(global_seq/sp)`、`heads=heads/tp`。A14B 的 DiT token grid 来自 VAE arch `(4,8,8)` 后再 patch `(1,2,2)`；TI2V-5B 则按官方 `prepare_latent_shape()` 使用 `(4,16,16)` 的 DiT latent prepare grid，例如 `704x1280x121 -> 31x44x80 -> 27280 tokens`。

| 算子类别 | SGLang 源码载体 | collector |
|---|---|---|
| Latent patch embed | `WanTransformer3DModel.patch_embedding` / `WanModel.patch_embedding` | `wan_patch_embed` |
| RoPE | `rotary_embedding.apply_flashinfer_rope_qk_inplace` | `wan_rope` |
| USPAttention 本地 compute | `USPAttention` 经 platform backend 选 FA/SDPA/Sage/SLA；按 A14B `40` heads 与 TI2V `24` heads 生成本地 shape | `wan_attention` |
| LayerNorm/Scale/Residual fused | `LayerNormScaleShift`、`ScaleResidualLayerNormScaleShift`、`MulAdd`、`RMSNorm`；覆盖 hidden `5120/3072` 与 RMSNorm TP shard | `wan_elementwise` |
| Linear/GEMM | `to_q/k/v/out`、FFN、text/time/image embedder；覆盖 A14B `ffn=13824` 与 TI2V `ffn=14336` | `gemm` Wan-only 维度 |

## 6. WanVAE encode/decode 拆解

`AutoencoderKLWan` 使用 `use_feature_cache=True`，decode 按 latent frame 逐帧过 decoder；encode 对条件图像构造首帧+零帧视频，并按首帧、后续 4 帧 chunk 编码。collector 不模拟跨 chunk cache 调度的整体模块耗时，而测其中反复出现的单算子。

| 算子类别 | SGLang 源码载体 | 维度设计 | collector |
|---|---|---|---|
| Causal Conv3d | `WanCausalConv3d.forward` | encode/decode 的 `1x1`、`3x1x1`、`3x3x3`；含 `cache_x` 时序拼接路径 | `wan_vae` |
| 2D resample conv | `WanResample.resample` | `Conv2d` over `B*T` frames，空间 up/down | `wan_vae` |
| VAE attention compute | `WanAttentionBlock` 的 `scaled_dot_product_attention` | 单头 SDPA，tokens=`H*W`，逐帧执行 | `wan_vae_attention` |
| VAE RMSNorm/SiLU | `WanRMS_norm`、`SiLU` | channel-first 5D tensor | `wan_vae_elementwise` |
| Avg/Dup shortcut | `AvgDown3D`、`DupUp3D` | residual down/up block shortcut reshape/reduce/repeat | `wan_vae_elementwise` |

## 7. 并行与维度变化

| 并行因素 | 代码位置 | 对 collector 的处理 |
|---|---|---|
| DiT TP | `ColumnParallelLinear`/`RowParallelLinear`、attention heads shard | GEMM 记录 TP 后 `(M,N,K)`；attention 记录 `tp_size` |
| DiT SP/USP | `sequence_shard_enabled`、`USPAttention`、`sequence_model_parallel_all_gather` | 不测通信；`sp_size/sp_algorithm` 只影响本地 `seq_len` 或 heads |
| Text encoder TP folding | `T5Config.parallel_folding` 与 `_get_folding_tp_group()` | collector 单卡测试默认 world size 1；GEMM 维度补 TP shape 作建模输入 |
| VAE parallel encode/decode | `WanDist*`、`split_for_parallel_encode/decode`、height split/gather | 不测分布式通信；VAE collector 使用单卡局部 height/width shape，文档记录可按服务器切片策略扩展 |
| CPU offload | stage `load_model/offload_model` | 不测 H2D/D2H 与 offload，只测 GPU kernel |
| Cache-DiT/TeaCache | `DenoisingStage`、`CachableDiT` | 会改变执行 block 次数，不改变单算子 shape；collector 不测 cache policy |

## 8. 新增 collector 输出

| OpEntry | 输出文件 | 覆盖范围 | 是否接入 PerfDatabase |
|---|---|---|---|
| `wan_patch_embed` | `wan_patch_embed_perf.txt` | DiT patch Conv3d；区分 `in_channels` 与 `hidden_size` | 是 |
| `wan_rope` | `wan_rope_perf.txt` | q/k RoPE inplace；按 profile heads 生成 | 是 |
| `wan_attention` | `wan_attention_perf.txt` | USPAttention 本地 compute | 是 |
| `wan_elementwise` | `wan_elementwise_perf.txt` | DiT norm/residual/modulation/RMSNorm | 是 |
| `wan_t5` | `wan_t5_perf.txt` | T5 embedding、attention compute、RMSNorm、FFN gated elementwise | 是 |
| `wan_clip` | `wan_clip_perf.txt` | CLIP patch embedding、attention compute、LayerNorm、quick_gelu | 是 |
| `wan_vae` | `wan_vae_perf.txt` | VAE causal Conv3d 与 2D resample conv | 是 |
| `wan_vae_attention` | `wan_vae_attention_perf.txt` | VAE mid-block SDPA | 是 |
| `wan_vae_elementwise` | `wan_vae_elementwise_perf.txt` | VAE norm、activation、AvgDown/DupUp | 是 |

## 9. 静态校验结果

- 已在 Docker `10.110.181.132:5000/sglang:0.5.10.post1` 中通过 `py_compile`。
- 已在本机 `ljc01` 环境中静态校验当前 registry 测试用例生成：`wan_patch_embed=15`、`wan_rope=198`、`wan_attention=505`、`wan_elementwise=144`、`wan_sparse_attention=384`；`wan_t5=12`、`wan_clip=4`、`wan_vae=165`、`wan_vae_attention=15`、`wan_vae_elementwise=120`。
- 已用 GPU 7 做 smoke：`wan_t5` 与 `wan_clip` 均通过 collector `--smoke`；`wan_vae`、`wan_vae_attention`、`wan_vae_elementwise` 通过 collector `--smoke`；同时对 VAE 小 shape 做过直接 run 函数 smoke。

## 10. 复合 SP 与通信补充

SGLang 0.5.10 中 Wan 的总 SP 满足 `sp_degree = ulysses_degree * ring_degree`。如果用户只给 `sp_degree>1`，SGLang 默认采用纯 Ulysses：`ulysses_degree=sp_degree, ring_degree=1`。本仓库 collector 与 SDK 已按以下事实对齐：

| 模块 | SGLang 行为 | collector / SDK 处理 |
|---|---|---|
| DiT self-attn | Q/K/V 先 Ulysses AllToAll，再在 Ring 分组上做本地 attention，最后输出 AllToAll 回 sequence shard | collector 测 post-A2A/Ring local compute；SDK 额外加 AllToAll 与 Ring P2P empirical 通信 |
| DiT cross-attn | `skip_sequence_parallel=True`，Q 保持 SP sequence shard，KV text/image replicated，不做 Ulysses head A2A | collector 使用 `cross_local` shape；SDK cross attention heads 只除 TP，不除 Ulysses |
| DiT GEMM/elementwise | block 内非 self-attn 输入是 SP 后 local tokens | collector 和 SDK 均使用 `ceil(global_seq/sp_size)` |
| T5 encoder | 默认跟 TP group；`tp_size==1 && sp_degree>1` 时 parallel linear/embedding 复用 SP group | GEMM collector 补 group size；SDK 加 embedding/attention/FFN all-reduce empirical 通信 |
| VAE encode/decode | 按 height 维做 split/gather，distributed conv 有 halo P2P | VAE collector 补 local height shape；SDK 加 height all-gather 与 halo P2P empirical 通信 |

本轮仍不考虑 VSA/SLA 稀疏 attention 路径；`collect_wan_sparse_attention.py` 保留为专项实验入口，不纳入主 Wan 仿真链路。
