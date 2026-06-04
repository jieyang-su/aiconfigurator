# Wan2.2 推理流程与 Collector 测试用例对照

本文给真人迁移与复测使用，说明 SGLang Wan2.2 推理通路中的关键算子如何映射到 `collector/sglang` 测试脚本。collector 输出的 `wan_*.txt` 与 `gemm_perf.txt` 已被 SDK Wan 仿真路径消费；新增或迁移硬件数据时应尽量保持本表维度精确覆盖，减少最近邻插值与 `HYBRID` 回退。

## 1. 推荐执行命令

```bash
IMAGE=10.110.181.132:5000/sglang:0.5.10.post1
REPO=/home/ai_lab/ljc/scale-up-sim/aiconfigurator
OUT=${REPO}/src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.10.post1-wan2.2-$(date +%Y%m%d_%H%M%S)-full
mkdir -p ${OUT}

docker run --gpus '"device=7"' --ipc=host --rm \
  -v ${REPO}:/workspace \
  -w /workspace/src/aiconfigurator/systems/data/h100_sxm/sglang/$(basename ${OUT}) \
  ${IMAGE} bash -lc 'python /workspace/collector/collect.py \
    --backend sglang \
    --ops wan_t5 wan_clip wan_patch_embed wan_rope wan_attention wan_elementwise wan_vae wan_vae_attention wan_vae_elementwise \
    --checkpoint-dir checkpoints'
```

Wan-only GEMM 单独跑，避免覆盖通用数据：

```bash
GEMM_OUT=${REPO}/src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.10.post1-wan2.2-$(date +%Y%m%d_%H%M%S)-gemm-wan
mkdir -p ${GEMM_OUT}

docker run --gpus '"device=7"' --ipc=host --rm \
  -v ${REPO}:/workspace \
  -w /workspace/src/aiconfigurator/systems/data/h100_sxm/sglang/$(basename ${GEMM_OUT}) \
  ${IMAGE} bash -lc 'COLLECTOR_GEMM_ONLY_WAN=1 python /workspace/collector/collect.py \
    --backend sglang --ops gemm --checkpoint-dir checkpoints'
```

## 2. 流程到算子表

| 推理阶段 | Runtime 代码 | 张量/维度 | Collector | 输出文件 |
|---|---|---|---|---|
| Tokenize + T5 embedding | `TextEncodingStage.encode_text`、`T5Stack.embed_tokens` | `input_ids[B,S] -> [B,S,4096]`，`S=128/256/512` | `wan_t5: embedding` | `wan_t5_perf.txt` |
| T5 self-attn | `T5Attention.forward`、`T5MultiHeadAttention` | `q/k/v[B,S,64,64]`，bias `[B,64,S,S]` | `wan_t5: attention_compute_bias_softmax` + `gemm` | `wan_t5_perf.txt`、`gemm_perf.txt` |
| T5 FFN | `T5DenseGatedActDense` | `d_model=4096`，`d_ff=10240` | `wan_t5: ffn_gated_act_mul` + `gemm` | `wan_t5_perf.txt`、`gemm_perf.txt` |
| CLIP patch | `CLIPVisionEmbeddings` | image `224x224`，patch `32`，tokens `50` | `wan_clip: vision_patch_embed` | `wan_clip_perf.txt` |
| CLIP transformer | `CLIPAttention`、`CLIPMLP` | hidden `768`，heads `12`，intermediate `3072`；attention collector 使用源码等价 SDPA | `wan_clip` + `gemm` | `wan_clip_perf.txt`、`gemm_perf.txt` |
| Condition image VAE encode | `ImageVAEEncodingStage`、`AutoencoderKLWan.encode` | image/video condition `B,3,F,H,W`；feature cache chunks | `wan_vae`、`wan_vae_attention`、`wan_vae_elementwise` | `wan_vae*.txt` |
| Latent patch embed | `WanTransformer3DModel.patch_embedding` / `WanModel.patch_embedding` | T2V `C=16,hidden=5120`；I2V `C=36,hidden=5120`；TI2V `C=48,hidden=3072`；patch `(1,2,2)` | `wan_patch_embed` | `wan_patch_embed_perf.txt` |
| DiT RoPE | `rotary_emb` + q/k apply | local `seq_len` + local heads | `wan_rope` | `wan_rope_perf.txt` |
| DiT attention | `USPAttention` | self/cross text/cross image，head_dim `128` | `wan_attention` | `wan_attention_perf.txt` |
| DiT elementwise | block norm/residual/modulation | A14B hidden `5120`；TI2V hidden `3072`；q/k RMSNorm 记录 TP shard hidden | `wan_elementwise` | `wan_elementwise_perf.txt` |
| DiT Linear/MLP | `to_q/k/v/out`、FFN、embedders | Wan-only `(M,N,K)` including TP/SP local M | `gemm` with `COLLECTOR_GEMM_ONLY_WAN=1` | `gemm_perf.txt` |
| VAE decode | `DecodingStage.decode`、`AutoencoderKLWan.decode` | latent frames `ceil((F-1)/4)+1`；decode 按帧输出 | `wan_vae`、`wan_vae_attention`、`wan_vae_elementwise` | `wan_vae*.txt` |

## 3. Tensor 维度设计

| 维度来源 | 代码事实 | collector 取值 |
|---|---|---|
| 文本长度 | Wan `t5_postprocess_text()` pad 到 512 | `128/256/512` 覆盖短/中/满长；DiT cross-attn 固定 `512` |
| CLIP image tokens | `224/32=7`，`7*7+cls=50`；Wan DiT image context 代码切 `257` tokens | CLIP collector 测 encoder `50`；DiT cross image attention 仍测 `257` |
| VAE spatial stride | Wan2.2 VAE 8x spatial compression | `height//8,width//8` 进入 latent，再由 DiT patch `(1,2,2)` 进一步压缩 |
| VAE temporal stride | WanVAE temporal compression ratio 4 | decode latent frames `ceil((F-1)/4)+1`；TI2V DiT config 特殊使用 full `F` latent shape |
| DiT token 数 | `patch_size=(1,2,2)` | `seq_len=F_lat*(H/32)*(W/32)`，已有 `wan_common.seq_len_from_video()` |
| DiT hidden | A14B `40 heads * 128 = 5120`；TI2V `24 heads * 128 = 3072` | A14B FFN `13824`；TI2V FFN `14336`；text dim `4096` |
| DiT patch input | HF config `in_channels` 或 `in_dim` | T2V `16`，I2V `36`，TI2V `48`；`wan_patch_embed_perf.txt` 记录 `hidden_size` 字段 |

## 4. 并行算法对 collector 的影响

| 并行项 | 真实运行影响 | collector 建模 |
|---|---|---|
| TP | Linear 输出/输入按 rank shard；attention heads 变少 | GEMM 生成 TP 后 `(N,K)`；attention `num_heads=model_heads/tp` 再考虑 SP |
| Ulysses SP | all-to-all 后单卡保留全局序列、heads 再 shard | `seq_len=global_seq_len`，`num_heads=(model_heads/tp)/sp` |
| Ring SP | attention 计算按局部 sequence shard 轮转 KV | `seq_len=ceil(global_seq_len/sp)`，`num_heads=model_heads/tp` |
| Sequence shard | DiT forward 先 padding 再按 SP 切序列，最后 all-gather | 通信不测；只测 local compute shape，并记录 `sp_algorithm` |
| VAE parallel encode/decode | height split/gather 与 `WanDist*` | 本轮只测单卡算子；迁移时可把 `height` 改为本地 height shard 后复测 |
| CFG parallel | 改变 rank 上执行正/负 prompt 的职责 | 单算子 shape 基本不变，不在 collector 中单独建模 |

## 5. Blackwell 迁移检查

| 检查项 | 期望 | 异常处理 |
|---|---|---|
| `wan_attention_perf.txt.kernel_source` | SM90 为 `sglang_fa`；SM100/SM103 应由 SGLang selector 选到对应 FA 实现 | 若回退 SDPA，检查镜像是否含 Blackwell FA4/sgl-kernel |
| `gemm_perf.txt.gemm_dtype` | Blackwell 应额外出现 `nvfp4` | 若无 `nvfp4`，检查 FlashInfer/sgl-kernel 版本 |
| `fp8_block` | 只出现 `K % 128 == 0` case | 不要强行跑不对齐 DeepGEMM case |
| VAE/CLIP/T5 | 允许 PyTorch CUDA/SDPA，因为源码本身使用 PyTorch primitive 或 SGLang wrapper | 不把这些视为错误 fallback；重点确认 DiT fused/FA 不 fallback |

## 6. 验收命令

```bash
for f in wan_t5_perf.txt wan_clip_perf.txt wan_patch_embed_perf.txt wan_rope_perf.txt wan_attention_perf.txt wan_elementwise_perf.txt wan_vae_perf.txt wan_vae_attention_perf.txt wan_vae_elementwise_perf.txt; do
  echo "== ${f} =="
  test -f "${f}" && { wc -l "${f}"; head -2 "${f}"; } || echo missing
done
find . -name collection_summary_sglang.json -maxdepth 2 -print -exec cat {} \;
```

当前代码期望 registry case 数：`wan_t5=12`、`wan_clip=4`、`wan_patch_embed=15`、`wan_rope=198`、`wan_attention=505`、`wan_elementwise=144`、`wan_sparse_attention=384`、`wan_vae=165`、`wan_vae_attention=15`、`wan_vae_elementwise=120`。实际行数可能因 `--smoke`、OOM 过滤、稀疏 attention 外部包缺失或手动调低 `COLLECTOR_WAN_VAE_MAX_INPUT_ELEMS` 而减少。

## 7. 维度补齐迭代记录

SDK 读取 HF config 后暴露出旧 collector 主要按 A14B 维度采样，导致 TI2V/I2V 仿真容易触发最近邻插值或 `HYBRID` 回退。本轮已修正：

| 缺口 | 修正 |
|---|---|
| `wan_patch_embed` 只测 `in_channels=16,hidden=5120` | 按 profile 生成 T2V `16/5120`、I2V `36/5120`、TI2V `48/3072`，CSV 新增 `hidden_size` |
| `wan_rope`/`wan_attention` 只覆盖 A14B `40` heads 派生值 | 按 `profile.num_heads` 生成，新增 TI2V local heads `24/12/6/3` |
| `wan_elementwise` 只覆盖 `5120` 及 TP shard | 新增 TI2V hidden `3072` 与 RMSNorm shard `1536/768/384` |
| Wan-only GEMM 只覆盖 A14B `hidden=5120,ffn=13824` | 新增 TI2V `hidden=3072,ffn=14336`、I2V patch/context 相关形状与 final DiT head 输出形状 |
| `PerfDatabase` 对 Wan 表过度最近邻 | `query_wan_patch_embed/rope/attention/elementwise` 提高精确前缀，数据缺失时在 `SILICON` 模式显式报缺表，促使补采 |
