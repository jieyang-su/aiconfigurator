# Wan2.2 SGLang 推理流程与仿真映射

本文面向后续维护者解释 SGLang `0.5.10.post1` 中 Wan2.2 视频生成的完整数据流，以及 SDK 当前如何把它映射成可查性能库的静态算子序列。本文以本仓库缓存的 HF config、SGLang 0.5.10.post1 源码和 `h100_sxm/sglang/0.5.10.post1-wan2.2-main0515` 实测数据为准。

## 模型配置入口

| 模型 | SGLang DiT 类 | HF 主 config 关键字段 | SDK 解析结果 |
|---|---|---|---|
| `Wan-AI/Wan2.2-T2V-A14B` | `WanTransformer3DModel` | `num_attention_heads=40`、`attention_head_dim=128`、`ffn_dim=13824`、`in_channels=16`、`out_channels=16`、`patch_size=[1,2,2]` | hidden `5120`、layers `40`、patch input `16`、latent `16` |
| `Wan-AI/Wan2.2-I2V-A14B` | `WanModel` / MOVA DiT | `dim=5120`、`num_heads=40`、`ffn_dim=13824`、`in_dim=36`、`out_dim=16`、`text_len=512` | hidden `5120`、layers `40`、patch input `36`、latent `16` |
| `Wan-AI/Wan2.2-TI2V-5B` | `WanModel` / MOVA DiT | `dim=3072`、`num_heads=24`、`ffn_dim=14336`、`in_dim=48`、`out_dim=48`、`text_len=512` | hidden `3072`、layers `30`、patch input `48`、latent `48` |

`utils.py` 优先读取 `src/aiconfigurator/model_configs/Wan-AI--Wan2.2-*_config.json`。T5、VAE、Scheduler 等模块配置若存在则作为补充；缺少模块专属配置时复用 SGLang 0.5.10 的公共默认值。

## 端到端数据流

| 阶段 | SGLang 运行时职责 | 典型 tensor 形状 | SDK 映射 |
|---|---|---|---|
| Prompt 编码 | tokenizer 后进入 UMT5 Encoder | text tokens `[B,512]`，encoder hidden `[B,512,4096]` | `_build_t5_ops()`：`wan_t5_perf.txt` + GEMM + T5 parallel comm |
| 图像条件编码 | I2V/TI2V 读取输入图像，进入 CLIP 视觉塔 | CLIP tokens 约 `[B,50,768]`，投影后 image context 约 `257` tokens | `_build_clip_ops()` 与 image embed GEMM |
| 条件 VAE encode | I2V/TI2V 把参考图像/条件转 latent | RGB `[B,3,F,H,W]` 到 latent/条件通道 | `_build_vae_encode_ops()` 查询 `wan_vae*.txt`，并加入 height split/gather/halo 通信 |
| latent 初始化 | scheduler 根据步数生成噪声 latent | A14B DiT latent `[B,C,F_lat,H/8,W/8]`；TI2V-5B 官方 `prepare_latent_shape()` 使用 `[B,48,F_lat,H/16,W/16]` | 不单独计时，体现在 DiT patch shape |
| DiT denoising | 每个 timestep 调用 Wan DiT 主干 | A14B token 数 `F_lat*(H/16)*(W/16)`；TI2V-5B token 数 `F_lat*(H/32)*(W/32)` | `_build_wan_pipeline()` 按 `denoising_steps*num_layers` 放大 |
| scheduler 更新 | 根据 DiT 输出更新 latent | 与 latent 同形状 | 当前未单独计时，视为控制逻辑开销 |
| VAE decode | 最终 latent 解码成视频帧 | latent frames `ceil((frames-1)/4)+1` | `_build_vae_decode_ops()` 查询 `wan_vae*.txt`，并加入 gather/halo 通信 |

## DiT 主干维度

设输出视频为 `frames × height × width`，Wan VAE arch 压缩率为 `(4,8,8)`，DiT patch size 为 `(1,2,2)`。需要把 **VAE arch stride** 和 **DiT latent prepare stride** 分清：

| 维度 | 公式 | 示例 |
|---|---|---|
| A14B latent grid | `F_lat=(frames-1)//4+1`、`H_lat=height//8`、`W_lat=width//8` | `720x1280x121 -> 31x90x160` |
| A14B DiT tokens | `seq_len=F_lat*(H_lat//2)*(W_lat//2)` | `31*45*80=111600` |
| TI2V-5B latent prepare grid | SGLang `Wan2_2_TI2V_5B_Config.prepare_latent_shape()` 使用 `vae_stride=(4,16,16)` | `704x1280x121 -> 31x44x80` |
| TI2V-5B DiT tokens | `seq_len=31*(44//2)*(80//2)` | `31*22*40=27280` |
| block local tokens | `ceil(seq_len/sp_size)` | A14B SP=4 时 `27900`；TI2V-5B SP=4 时 `6820` |
| head_dim | `hidden_size/num_heads` | A14B `5120/40=128`，TI2V `3072/24=128` |
| TP 后 heads | `num_heads/tp_size` | A14B TP=4 时 local heads `10`，TI2V TP=4 时 local heads `6` |
| patch 输入通道 | `WanTransformer3DModel.in_channels` 或 `WanModel.in_dim` | T2V `16`，I2V `36`，TI2V `48` |

两个容易踩坑的点：第一，`WanModel.in_dim` 不是 VAE latent channels，I2V 为 `36`、TI2V 为 `48`。第二，TI2V-5B 的 DiT latent prepare stride 是 `(4,16,16)`，但 VAE encode/decode 仍按 VAE arch `(4,8,8)` 组织算子。SDK 用 `patch_in_channels` 表示 DiT patch embed 输入通道，用 `latent_channels` 表示 DiT/VAE 输出通道，用 `latent_prepare_stride` 表示 DiT 输入 grid，用 `vae_stride` 表示 VAE 算子与解码路径，两者不再混用。

## 并行与通信

| 并行项 | SGLang 行为 | 对计算 shape 的影响 | SDK 当前处理 |
|---|---|---|---|
| TP | Column/Row Parallel Linear 切分 QKV、FFN、输出投影 | local heads 与 GEMM `n/k` 按 `tp_size` 缩放 | GEMM 维度按 `tp_size` 计算，RowParallel 后加入 AllReduce |
| Ulysses SP | attention 内部做 AllToAll，在 head/sequence 维间重排 | self-attn heads 再除 `ulysses_degree` | `_local_attention_shape()` 返回 `heads=heads/tp/ulysses`，加入 input/output AllToAll |
| Ring SP | PyTorch templated ring attention 会先发起一次 K/V all-gather，再消费本地切片 | self-attn seq 变为 `ceil(global_seq/ring_degree)` | `_local_attention_shape()` 返回 ring local seq，加入 `KV all_gather + first self-attn overlap` |
| Ulysses×Ring | SGLang 使用 `sp_degree = ulysses_degree * ring_degree` | self-attn 同时切 heads 与 ring seq | `sp_algorithm="usp"`，collector 与 SDK 均生成复合 shape |
| Cross attention | `skip_sequence_parallel=True`，不做 Ulysses A2A | Q 为 `ceil(global_seq/sp_size)`，heads 只按 TP 切 | SDK/collector 用 `sp_algorithm="cross_local"` |
| T5 encoder | 默认 TP group；`tp_size==1 && sp>1` 时 folding 到 SP group | T5 GEMM 按 parallel group 切，attention heads/local FFN 同步变化 | SDK 用 `_t5_parallel_group_size()`，collector GEMM 补 folding shape |
| VAE | parallel encode/decode 按 height split，distributed conv 有 halo，末尾 gather | conv/attention/elementwise 使用 local height | SDK 和 collector 统一使用 SP local height，并加入 P2P/gather 通信 |

Wan 通信使用 `WanParallelComm`：NCCL 类 collective 调 `query_nccl()` 并继承当前数据库模式，目标 system 提供 `nccl/<version>/nccl_perf.txt` 时可在 `HYBRID` 下命中实测；VAE halo P2P 调 `query_p2p(..., DatabaseMode.EMPIRICAL)`，按通信组规模选择 system YAML 中的 intra/inter node 带宽。当前 Wan Ring attention 已按 SGLang 实际行为建模为一次性 `all_gather`，不再使用旧的逐 hop P2P 假设。因此 Hybrid 模式下，单卡算子与 NCCL collective 都优先来自 silicon/插值，当前主要保留 empirical 的是 VAE halo 一类 P2P。

## SDK 算子映射

| SGLang 模块 | 关键源码行为 | SDK op | 性能文件 |
|---|---|---|---|
| `UMT5EncoderModel` | embedding、RMSNorm、self-attn、gated FFN | `WanMeasuredOp(query_wan_t5)`、`WanStaticGEMM`、T5 AllReduce | `wan_t5_perf.txt`、`gemm_perf.txt`、NCCL/P2P 通信查询 |
| CLIP 视觉塔 | patch embed、LN、MHSA、MLP | `query_wan_clip`、GEMM | `wan_clip_perf.txt`、`gemm_perf.txt` |
| Wan VAE | 3D/2D conv、RMSNorm/SILU、局部 attention | `query_wan_vae*` + VAE parallel comm | `wan_vae_perf.txt`、`wan_vae_attention_perf.txt`、`wan_vae_elementwise_perf.txt` |
| DiT patch embed | `PatchEmbed` 或 `Conv3dLocalIsland` | `query_wan_patch_embed` | `wan_patch_embed_perf.txt` |
| DiT RoPE | 3D RoPE 作用在 Q/K | `query_wan_rope` | `wan_rope_perf.txt` |
| DiT self/cross attention | USPAttention/LocalAttention compute | `query_wan_attention` | `wan_attention_perf.txt` |
| DiT norm/residual/modulation | FP32 LN、RMSNorm、ScaleResidual 类 fused op | `query_wan_elementwise` | `wan_elementwise_perf.txt` |
| DiT Linear/FFN/head | Column/Row/Replicated Linear | `WanStaticGEMM` | `gemm_perf.txt` |

## Collector 对齐策略

- `wan_common.valid_parallel_cases()` 生成 `tp_size, sp_size, ulysses_degree, ring_degree, sp_algorithm`，覆盖纯 Ulysses、纯 Ring 与 Ulysses×Ring 复合情况。
- `collect_wan_rope.py` 与 self-attn collector 使用 post-Ulysses/Ring 的本地 compute shape。
- `collect_wan_attention.py` 对 self-attn 与 cross-attn 分开生成：cross 不除 Ulysses heads，使用 `cross_local` 标签。
- `collect_wan_elementwise.py` 使用 `ceil(seq_len/sp_size)`，避免 SDK 在 SP 下查询全局 token 旧 shape。
- `collect_gemm.py` 继续复用 GEMM collector，并补 T5 parallel folding group 的 GEMM shape。
- `collect_wan_vae.py` 补齐 SP local height 的 conv/attention/elementwise shape。

## 调用示例

```python
from aiconfigurator.cli.api import cli_estimate

result = cli_estimate(
    "Wan-AI/Wan2.2-TI2V-5B",
    "h100_sxm",
    backend_name="sglang",
    backend_version="0.5.10.post1-wan2.2-main0515",
    database_mode="HYBRID",
    isl=1,
    osl=1,
    batch_size=1,
    tp_size=1,
    video_task="ti2v",
    video_height=704,
    video_width=1280,
    video_frames=121,
    denoising_steps=50,
    sp_size=8,
    ulysses_degree=2,
    ring_degree=4,
)
```

`HYBRID` 是当前推荐模式：已有 Wan 实测表和 NCCL 实测表优先查表，P2P 通信固定 empirical，GEMM 或 elementwise 缺少精确点时回退到插值/经验模型，避免整条视频链路因为单个未采样形状中断。

## CFG 与 cfg_parallel 映射

Wan2.2 在 SGLang 中先由采样参数决定请求是否启用 CFG，再由 `enable_cfg_parallel` 决定 cond/uncond 两路如何在 CFG 并行组内分配。AIC 仿真把它作为顶层调度倍率，而不是新的底层 kernel shape。

| SGLang 执行路径 | Tensor/并行影响 | SDK 表达 |
|---|---|---|
| `cfg_scale<=1` 或负向 prompt 为空 | 只跑 conditional DiT forward | DiT `1x`，无 CFG 通信 |
| CFG 串行 | 每个 denoise step 先 cond 后 uncond，两路 shape 相同 | DiT 主干 scale 乘 2；T5 正/负 prompt 编码 scale 乘 2 |
| CFG parallel | CFG rank0 跑 cond，另一个 CFG rank 跑 uncond | DiT 主干保持 `1x`，`num_total_gpus` 乘 `2`，parallel 标签追加 `cfg2` |
| CFG parallel 合并 | 两 rank 输出 partial noise，CFG 组 all-reduce 求和 | 每 denoise step 增加 `wan_dit_cfg_noise_all_reduce`，message 约为 `seq_len * latent_channels * patch_t * patch_h * patch_w` |

这意味着 CFG 不改变 TP/SP 下单卡 GEMM、attention、elementwise 的 shape；它改变的是相同 shape 的执行次数，以及 cfg_parallel 场景下的额外 CFG 组通信和资源占用。对于吞吐 sweep，若开启 `enable_cfg_parallel=True`，单视频并行组占用 GPU 数会翻倍，因此 DP 并发数需要按 `cluster_gpus / (tp*sp*cfg_degree)` 理解。
