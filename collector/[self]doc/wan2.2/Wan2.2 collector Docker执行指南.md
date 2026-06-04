# Wan2.2 SGLang Collector Docker 执行指南

本文记录 Wan2.2 文/图生视频模型在 `aiconfigurator/collector` 中的 SGLang 单卡算子采集方法、已完成的 H100 验收结果，以及迁移到 Hopper/Blackwell 服务器时需要注意的兼容点。

## 1. 本轮 H100 验收结论

### 1.1 Wan 独有算子结果目录

本轮正式结果写入：

```bash
aiconfigurator/src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.10.post1-wan2.2-20260512_171000-full
```

| 文件 | 行数（不含表头） | 预期 case 数 | kernel_source | 验收状态 |
|---|---:|---:|---|---|
| `wan_patch_embed_perf.txt` | 15 | 15 | `torch_conv3d` | 通过 |
| `wan_rope_perf.txt` | 132 | 132 | `flashinfer_rope` | 通过 |
| `wan_attention_perf.txt` | 485 | 485 | `sglang_fa` | 通过 |
| `wan_elementwise_perf.txt` | 96 | 96 | `sglang_layernorm_scale_shift` / `sglang_rmsnorm` / `sglang_cutedsl_scale_residual_layernorm_scale_shift` / `sglang_mul_add` | 通过 |

采集 summary：

```text
collection_summary_sglang.json: total_errors = 0
SGLang version = 0.5.10.post1
device = NVIDIA H100 80GB HBM3
```

### 1.2 Wan-only GEMM 结果目录

Wan GEMM 单独写入，避免覆盖既有通用 `gemm_perf.txt`：

```bash
aiconfigurator/src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.10.post1-wan2.2-20260512_171000-gemm-wan-full
```

| 文件 | 可用行数（不含表头） | dtype 覆盖 | 状态 |
|---|---:|---|---|
| `gemm_perf.txt` | 1148 | `bfloat16`: 391；`fp8`: 391；`fp8_block`: 366 | 可用，存在已解释的 DeepGEMM shape 约束 |

`fp8_block` 曾有 25 个失败 case，全部为 `K=1728` 时触发 `sglang_per_token_group_quant_fp8(group_size=128)` 的 `K % 128 == 0` 约束。代码已增加过滤逻辑：`fp8_block` 仅生成 `k % 128 == 0` 的 case；按要求未继续重跑全量 DeepGEMM。

## 2. Docker 运行环境

推荐使用项目文档指定的官方 Docker 环境，避免本机 conda 依赖差异：

```bash
IMAGE=10.110.181.132:5000/sglang:0.5.10.post1
REPO=/home/ai_lab/ljc/scale-up-sim/aiconfigurator
```

本机 H100 环境中前 6 卡被占用，因此示例固定使用 GPU 7：

```bash
docker run --gpus '"device=7"' --ipc=host --rm \
  -v ${REPO}:/workspace \
  -w /workspace \
  ${IMAGE} bash
```

迁移到其他服务器时，将 `device=7` 改成目标空闲 GPU；若希望多进程并行，传入多个 GPU，例如 `--gpus '"device=6,7"'`，collector 会按容器内可见 GPU 数启动 worker。

## 3. 推荐执行流程

### 3.1 创建独立输出目录

不要在已有版本目录中直接运行，以免追加覆盖历史数据。建议每次用时间戳创建新目录：

```bash
OUT=/workspace/src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.10.post1-wan2.2-$(date +%Y%m%d_%H%M%S)-full
mkdir -p ${OUT}
```

硬件名可按服务器实际情况替换，例如 `b200_sxm`、`gb200`。

### 3.2 先跑 smoke

```bash
docker run --gpus '"device=7"' --ipc=host --rm \
  -v /home/ai_lab/ljc/scale-up-sim/aiconfigurator:/workspace \
  -w ${OUT} \
  10.110.181.132:5000/sglang:0.5.10.post1 \
  bash -lc 'python /workspace/collector/collect.py \
    --backend sglang \
    --ops wan_patch_embed wan_rope wan_attention wan_elementwise \
    --smoke \
    --checkpoint-dir checkpoints'
```

smoke 通过后再跑全量。

### 3.3 跑 Wan 独有算子全量

```bash
docker run --gpus '"device=7"' --ipc=host --rm \
  -v /home/ai_lab/ljc/scale-up-sim/aiconfigurator:/workspace \
  -w ${OUT} \
  10.110.181.132:5000/sglang:0.5.10.post1 \
  bash -lc 'python /workspace/collector/collect.py \
    --backend sglang \
    --ops wan_patch_embed wan_rope wan_attention wan_elementwise \
    --checkpoint-dir checkpoints'
```

输出文件：

```text
wan_patch_embed_perf.txt
wan_rope_perf.txt
wan_attention_perf.txt
wan_elementwise_perf.txt
wan_patch_embed+wan_rope+wan_attention+wan_elementwise_*/collection_summary_sglang.json
```

### 3.4 跑 Wan-only GEMM

GEMM 使用通用 `collect_gemm.py`，不要直接跑全量通用 GEMM。需要启用 Wan-only 过滤：

```bash
GEMM_OUT=/workspace/src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.10.post1-wan2.2-$(date +%Y%m%d_%H%M%S)-gemm-wan
mkdir -p ${GEMM_OUT}

docker run --gpus '"device=7"' --ipc=host --rm \
  -v /home/ai_lab/ljc/scale-up-sim/aiconfigurator:/workspace \
  -w ${GEMM_OUT} \
  10.110.181.132:5000/sglang:0.5.10.post1 \
  bash -lc 'COLLECTOR_GEMM_ONLY_WAN=1 python /workspace/collector/collect.py \
    --backend sglang \
    --ops gemm \
    --checkpoint-dir checkpoints'
```

如只做迁移验证，可先加 `--smoke`。

## 4. 算子与真实 SGLang kernel 对齐

| collector | 测试对象 | 对齐方式 | 备注 |
|---|---|---|---|
| `collect_wan_patch_embed.py` | Wan `PatchEmbed.proj` 的 `Conv3d(kernel=stride=patch_size)` | 使用 PyTorch/CUDA `Conv3d` | 单算子粒度；CSV 额外记录 `hidden_size`，以区分 A14B 与 TI2V |
| `collect_wan_rope.py` | q/k RoPE inplace | 调用 `apply_flashinfer_rope_qk_inplace` | 与 Wan CUDA 路径一致；按 profile 生成 `head_dim=128`、`num_heads=40/24` 的本地 shape |
| `collect_wan_attention.py` | attention 本地 compute | 通过 SGLang `current_platform.get_attn_backend_cls_str()` 选择后端并实例化 impl | H100 实测为 `sglang_fa`；TI2V 额外覆盖 `24` heads 的 Ulysses/Ring 组合 |
| `collect_wan_elementwise.py` | Wan block 非 GEMM fused op | 调用 SGLang `LayerNormScaleShift`、`RMSNorm`、`ScaleResidualLayerNormScaleShift`、`MulAdd` | A14B 与 TI2V 都纳入；`rmsnorm_qk` 使用 `hidden_size=hidden/tp` |
| `collect_gemm.py` | Wan DiT Linear/GEMM shape | 复用 SGLang GEMM collector，增加 Wan shape 与 `COLLECTOR_GEMM_ONLY_WAN` | `fp8_block` 已过滤 `K % 128 != 0` case |

`SP/USP/Ring` 通信不计入单卡算子 latency；collector 只使用通信后本地 compute shape，并在数据列中记录 `tp_size`、`sp_size`、`sp_algorithm`。

### 4.1 Config 驱动维度更新

SDK 接入 HF config 后发现旧数据对 TI2V/I2V 维度覆盖不足：`wan_patch_embed` 只有 `in_channels=16`，attention/rope 只覆盖 A14B `40` heads 派生的 local heads，elementwise 只覆盖 `hidden=5120`。当前 collector 已按 `collector/sglang/wan_common.py::WAN_PROFILES` 生成真实 profile 维度：

| Profile | patch embed | heads/head_dim | FFN | elementwise |
|---|---|---|---|---|
| T2V A14B | `in_channels=16,hidden=5120` | `40/128` | `13824` | `5120` 与 RMSNorm TP shard |
| I2V A14B | `in_channels=36,hidden=5120` | `40/128` | `13824` | `5120` 与 RMSNorm TP shard |
| TI2V 5B | `in_channels=48,hidden=3072` | `24/128` | `14336` | `3072` 与 RMSNorm TP shard `1536/768/384` |

复合 SP 更新后，当前期望 case 数约为 `wan_patch_embed=15`、`wan_rope=270`、`wan_attention=1117`、`wan_elementwise=576`；`wan_sparse_attention` 不纳入主链路。旧 H100 `20260512_171000-full` 数据仍可作为历史验收记录，但用于新 SDK 仿真时建议补采新目录，避免 `SILICON` 缺表或 `HYBRID` 回退。

## 5. Blackwell 兼容性说明

当前脚本面向 Hopper/Blackwell 做了以下兼容处理：

| 模块 | Hopper 行为 | Blackwell 兼容处理 | 潜在风险 |
|---|---|---|---|
| attention | SGLang `FA` 后端，Hopper 使用 FA3 | `collect_wan_attention.py` 复用 `current_platform.get_attn_backend_cls_str()`；SGLang 在 Blackwell 上会设置 FA4 | Docker/sgl-kernel 必须包含对应 SM100/SM103 FA4 kernel |
| GEMM dtype | SM90 跑 `fp8_block`、`bfloat16`、`fp8` | `collect_gemm.py` 对 SM100/SM103 增加 `nvfp4`；对 SM120 跳过 `fp8_block` | Blackwell 服务器需使用支持 NVFP4/FA4 的 SGLang 镜像 |
| DeepGEMM FP8-block | 依赖 TMA 与 group quant | 过滤 `k % 128 != 0`，避免不满足 `group_size=128` 的 case | 部分 Wan FFN TP shape 无 `fp8_block` 数据，保留 BF16/FP8 |
| RoPE | `flashinfer_rope` | FlashInfer JIT/预编译 kernel 按实际 GPU 架构加载 | 若镜像缺少对应 arch cubin，会在 smoke 阶段暴露 |
| elementwise fused | CUTeDSL / Triton / sgl-kernel | 默认直接调用 SGLang fused module，不再使用 torch native fallback | 镜像需带 `cutlass`/CUTeDSL；缺失时应修环境，不建议静默 fallback |
| patch embed | CUDA `Conv3d` | 依赖 PyTorch/cuDNN 对架构支持 | 需要匹配驱动与 CUDA 版本 |

迁移到 B200/GB200 时建议先跑：

```bash
python /workspace/collector/collect.py --backend sglang \
  --ops wan_patch_embed wan_rope wan_attention wan_elementwise \
  --smoke --checkpoint-dir checkpoints
```

再检查 `wan_attention_perf.txt` 的 `kernel_source` 是否为预期的 `sglang_fa`，并确认日志中没有 FA 回退到 SDPA 的提示。

## 6. 验收命令

Wan 独有算子验收：

```bash
OUT=aiconfigurator/src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.10.post1-wan2.2-20260512_171000-full
for f in wan_patch_embed_perf.txt wan_rope_perf.txt wan_attention_perf.txt wan_elementwise_perf.txt; do
  echo "== ${f} =="
  wc -l "${OUT}/${f}"
  head -2 "${OUT}/${f}"
done
cat "${OUT}"/wan_patch_embed+wan_rope+wan_attention+wan_elementwise_*/collection_summary_sglang.json
```

Wan-only GEMM 验收：

```bash
GEMM=aiconfigurator/src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.10.post1-wan2.2-20260512_171000-gemm-wan-full
wc -l "${GEMM}/gemm_perf.txt"
python - <<'PY'
import csv, collections
p = "aiconfigurator/src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.10.post1-wan2.2-20260512_171000-gemm-wan-full/gemm_perf.txt"
with open(p) as f:
    rows = list(csv.DictReader(f))
print(collections.Counter(r["gemm_dtype"] for r in rows))
PY
```

## 7. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `Global sgl_diffusion args is not set` | 单算子 collector 没有启动 SGLang diffusion server | 已在 `collect_wan_attention.py` 中绕开 server args，直接复用 platform backend selector |
| attention 变成 `sdpa` | 当前镜像/硬件不满足 FA 条件或强制设置了 SDPA | 检查日志、SGLang 镜像、`SGLANG_DIFFUSION_ATTENTION_BACKEND` 环境变量 |
| `fp8_block` 报 `K % 128` | DeepGEMM group quant 要求 `group_size=128` 对齐 | 已过滤不对齐 case；使用 BF16/FP8 数据补充 |
| elementwise 缺 `cutlass` | Docker 环境不完整 | 更换官方完整镜像；不要把 native fallback 当真实数据 |
| 输出文件追加到旧目录 | collector 使用追加写入 | 每次创建新时间戳输出目录 |

## 8. Ulysses × Ring 并行维度更新

本轮 collector 已按 SGLang `sp_degree = ulysses_degree * ring_degree` 补齐复合并行 shape。核心规则如下：

| 场景 | collector 维度 | 说明 |
|---|---|---|
| self-attn / RoPE | `seq_len=ceil(global_seq_len/ring_degree)`，`num_heads=(heads/tp_size)/ulysses_degree` | 对应 USPAttention 内部 A2A 后的本地 attention compute |
| cross-attn text/image | `q_seq_len=ceil(global_seq_len/sp_size)`，`num_heads=heads/tp_size`，`sp_algorithm=cross_local` | SGLang cross attention 设置 `skip_sequence_parallel=True`，不执行 Ulysses head A2A |
| DiT elementwise | `seq_len=ceil(global_seq_len/sp_size)` | block 输入已 sequence shard，避免用全局 token shape 查询 |
| T5 GEMM | `tp_size==1 && sp_size>1` 时额外生成 `group_size=sp_size` 的 QKV/FFN GEMM | 对应 SGLang `parallel_folding_mode="sp"` |
| VAE | conv/attention/elementwise 额外覆盖 SP local height | 对应 parallel encode/decode 的 height split |

新增 `wan_rope_perf.txt` / `wan_attention_perf.txt` 新采集行会写入 `ulysses_degree`、`ring_degree` 两列。`PerfDatabase` 当前 key 仍保持旧格式，用实际 `seq_len/num_heads/sp_algorithm` 区分计算 shape；新增列主要用于人工验收和后续迁移。

通信仍不由 collector 实测。SDK 侧使用 `WanParallelComm` 按 system YAML 带宽做 empirical 估算：DiT Ulysses AllToAll、Ring KV P2P、TP AllReduce、T5 folding AllReduce、VAE height gather/halo P2P 都在仿真端补齐。

## 9. 2026-05-19 扩展采集维度建议

本轮代码已把主链路 collector 的合法并行 envelope 扩展到 `tp*sp<=32`，其中 `sp=ulysses_degree*ring_degree`：

| collector | 新增/修正 | 目的 |
|---|---|---|
| `wan_common.valid_parallel_cases()` | `sp_size` 扩展到 `1/2/4/8/16/32`，并过滤 `tp*sp>32` 与非法 head 切分 | 覆盖 32 卡内 Ulysses×Ring 组合 |
| `collect_gemm.py` | Wan DiT GEMM local token M 增加 SP=16/32 shape | 避免 SDK 在大 SP 下 GEMM 只能插值 |
| `collect_wan_t5.py` | T5 attention compute 与 gated activation 增加 `group_size=1/2/4/8/16/32` 的 local heads / local d_ff | 对齐 SGLang `tp=1 && sp>1` 时 `parallel_folding_mode="sp"` |
| `collect_wan_vae.py` | VAE local height 增加 SP=16/32 | 支持 VAE parallel encode/decode 的 height split |

注意：已有 `main0519` 数据未包含全部 SP=16/32 维度，SDK 在 HYBRID 下会用 warning 标注最近邻或 GEMM 插值。若要减少 warning 并提高 32 卡评估可信度，应优先补采 `wan_rope/wan_attention/wan_elementwise/wan_t5/wan_vae/wan_vae_attention/wan_vae_elementwise`，再补采 `COLLECTOR_GEMM_ONLY_WAN=1` 的 Wan GEMM shape。

`batch_size>1` 暂不建议直接扩展 collector 全量采集。当前 SDK 将多视频 batch 视为多条单视频 pipeline 串行累加；如果后续确认 SGLang 实际会把多视频合成 batch 进入同一 kernel，再为 patch/attention/elementwise/T5/VAE 增加 `batch_size=2/4/...` 分层采集。
