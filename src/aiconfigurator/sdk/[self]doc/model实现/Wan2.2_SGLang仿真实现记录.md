# Wan2.2 SGLang 仿真实现记录

本文记录 SDK 侧 Wan2.2 视频模型仿真实现。实现目标是使用 SGLang `0.5.10.post1-wan2.2-main0515` 单卡算子数据，近似还原 SGLang Wan2.2 主推理通路；当前覆盖 T5、CLIP、VAE、Wan DiT 主干、TP/SP 通信估算，SLA/VSA/稀疏分支仍不纳入主仿真。

## 入口与配置

| 组件 | 实现内容 | 说明 |
|---|---|---|
| `common.py` | 新增 `ModelFamily=WAN`、Wan profile/别名、Wan collector 文件枚举 | 支持 Wan2.2 TI2V 5B、T2V A14B、I2V A14B |
| `utils.py` | 读取本地 HF config 与模块 config，解析为 SDK 统一模型字段 | `in_dim/out_dim/d_model/ffn_dim/num_heads` 等不再硬编码 |
| `config.py` | `RuntimeConfig` 增加 `video_*`、`denoising_steps`、`sp_size`、`ulysses_degree`、`ring_degree`、`sp_algorithm`、`attention_backend` | 这些字段只被 Wan 路径消费，LLM 路径保持兼容 |
| `models.py` | `get_model()` 接入 `WanVideoModel` | Wan 被表示为 `context_ops` 静态视频 pipeline，使用 `run_static(..., mode="static_ctx")` |
| `operations.py` | 新增 Wan 实测 op、静态 GEMM/memory op、`WanParallelComm` | NCCL collective 走数据库默认模式；VAE halo P2P 固定 empirical；Wan ring-attn 改按实际 SGLang 路径建模为 all-gather |
| `cli/api.py` | `cli_estimate()` 可传视频与 SP 参数；Wan 自动走 `static_ctx` | 支持程序接口单点估计 Ulysses/Ring 组合 |

## 推理阶段到算子映射

| Wan 阶段 | SDK op 表达 | 性能数据 |
|---|---|---|
| T5 文本编码 | `wan_t5_*` 实测 op + `WanStaticGEMM` + T5 parallel group AllReduce | `wan_t5_perf.txt` + Wan GEMM 表 + NCCL/P2P 通信查询 |
| CLIP 图像编码 | `wan_clip_*` 实测 op + `WanStaticGEMM`，仅 I2V/TI2V | `wan_clip_perf.txt` + Wan GEMM 表 |
| 条件图像 VAE encode | `wan_vae`、`wan_vae_attention`、`wan_vae_elementwise`，仅 I2V/TI2V | `wan_vae*.txt` + height halo/gather 通信 |
| DiT patch embed | `wan_patch_embed` | `wan_patch_embed_perf.txt` |
| DiT RoPE | `wan_rope`，使用 Ulysses×Ring 后的 self-attn local shape | `wan_rope_perf.txt` |
| DiT USPAttention compute | `wan_attention`，self/cross 分别建模 | `wan_attention_perf.txt` |
| DiT norm/residual/modulation | `wan_elementwise`，SP 后 local token shape | `wan_elementwise_perf.txt` |
| DiT Linear/FFN/condition embed | `WanStaticGEMM` 与少量 `WanStaticMemOp` | Wan GEMM 表；简单 elementwise 走 memory 模型 |
| TP/SP 通信 | `WanParallelComm` | `query_nccl(...)` 使用默认数据库模式；`query_p2p(..., database_mode=EMPIRICAL)` |
| VAE decode | `wan_vae`、`wan_vae_attention`、`wan_vae_elementwise` | `wan_vae*.txt` + height gather/halo 通信 |

## 并行维度公式

| 维度 | 公式/默认值 | 备注 |
|---|---|---|
| latent shape | `F_lat=(frames-1)//4+1`，`H_lat=height//8`，`W_lat=width//8` | 与 collector `wan_common.latent_shape()` 对齐 |
| DiT token 数 | `seq_len=F_lat*(H_lat//2)*(W_lat//2)` | patch size 为 `(1,2,2)` |
| TP 后 heads | `heads_after_tp=num_heads//tp_size` | A14B `40` heads，TI2V `24` heads |
| SP 分解 | `sp_size = ulysses_degree * ring_degree` | 若只给 `sp_size`，按 SGLang 默认解析为 `ulysses_degree=sp_size, ring_degree=1` |
| self-attn local shape | `q_seq_len=ceil(global_seq_len/ring_degree)`，`num_heads=heads_after_tp//ulysses_degree` | Ulysses 切 heads，Ring 切 sequence；`sp_algorithm` 记录为 `ulysses/ring/usp/none` |
| cross-attn local shape | `q_seq_len=ceil(global_seq_len/sp_size)`，`num_heads=heads_after_tp` | SGLang cross attention 使用 `skip_sequence_parallel=True`，不做 Ulysses head all-to-all |
| DiT GEMM/elementwise M | `ceil(global_seq_len/sp_size)` | patch 后 sequence shard 再进入 block 内非 self-attn 计算 |
| T5 parallel group | `tp_size`；若 `tp_size==1 && sp_size>1` 则为 `sp_size` | 对应 SGLang `parallel_folding_mode="sp"` |
| VAE encode height | `padded_height/(sp_size)`，`padded_height` 对齐 `sp_size*2**downsample_count` | 对应 `split_for_parallel_encode()` 的 height padding/split |
| VAE decode height | `ceil(height_or_latent_height/sp_size)` | decode 按 height 分片，末尾 all-gather |

## 通信建模

| 通信点 | SGLang 行为 | SDK 表达 |
|---|---|---|
| DiT TP RowParallel 输出 | Row parallel linear 后 AllReduce | `wan_dit_*_tp_all_reduce`，NCCL silicon/HYBRID |
| DiT Ulysses input | Q/K/V 在 sequence/head 维 AllToAll | `wan_dit_usp_input_{q,k,v}_alltoall`，NCCL silicon/HYBRID |
| DiT Ring attention | K/V 一次性 all-gather，并与首个 local self-attn overlap | `wan_dit_ring_attention_kv_all_gather`，NCCL silicon/HYBRID，外层 `WanRingAttentionOverlap` 建模 |
| DiT Ulysses output | attention 输出 AllToAll 回到 sequence shard | `wan_dit_usp_output_alltoall`，NCCL silicon/HYBRID |
| DiT output gather | block 完成后 sequence all-gather | `wan_dit_output_sequence_all_gather`，NCCL silicon/HYBRID |
| T5 folding/TP | embedding/out/FFN RowParallel AllReduce | `wan_t5_*_all_reduce`，NCCL silicon/HYBRID |
| VAE parallel encode/decode | height split后 gather，distributed conv halo exchange | `wan_vae_*_height_all_gather` 走 NCCL silicon/HYBRID；`*_height_halo_p2p` 走 P2P empirical |

Wan 通信现在区分 collective 与点对点：AllReduce/AllGather/AllToAll 等 NCCL collective 尊重 PerfDatabase 默认模式，目标硬件提供 `nccl/<version>/nccl_perf.txt` 时在 `HYBRID` 下优先使用实测；Ring KV 与 VAE halo 这类 P2P 仍用 system YAML 的带宽/latency 经验估算，并按通信组规模选择 intra/inter node 带宽。

## PerfDatabase 查询

| 数据文件 | 查询方法 | 匹配策略 |
|---|---|---|
| `wan_patch_embed_perf.txt` | `query_wan_patch_embed` | `model/task/batch/in_channels/hidden_size` 精确，shape 最近邻 |
| `wan_rope_perf.txt` | `query_wan_rope` | 全 key 精确；旧数据缺 `usp` 时允许回退到同 shape 的 `ulysses` 历史标签 |
| `wan_attention_perf.txt` | `query_wan_attention` | `model/task/attn_kind/backend/batch/q/k/head` 精确，`tp/sp/algorithm` 最近邻；旧 cross SP 可从 `cross_local` 回退历史 `ring` 标签 |
| `wan_elementwise_perf.txt` | `query_wan_elementwise` | `op_name/batch/seq/hidden/tp` 精确 |
| `wan_t5_perf.txt` | `query_wan_t5` | `op_name/batch` 精确，shape 最近邻 |
| `wan_clip_perf.txt` | `query_wan_clip` | `op_name/batch` 精确，shape 最近邻 |
| `wan_vae*.txt` | `query_wan_vae*` | model/task/path/stage 或 op 前缀精确，shape 最近邻 |
| `gemm_perf.txt` | `query_gemm` | 直接读取 `main0515` 同级 GEMM 数据 |

## 使用示例

```python
from aiconfigurator.cli.api import cli_estimate

result = cli_estimate(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    "h100_sxm",
    backend_name="sglang",
    backend_version="0.5.10.post1-wan2.2-main0515",
    database_mode="HYBRID",
    isl=1,
    osl=1,
    batch_size=1,
    tp_size=1,
    video_height=720,
    video_width=1280,
    video_frames=121,
    denoising_steps=50,
    sp_size=4,
    ulysses_degree=2,
    ring_degree=2,
)
print(result.raw)
```

建议默认使用 `HYBRID`。Wan 专用算子优先查实测表；NCCL collective 优先查通信实测表；P2P、GEMM 表外形状、少量 memory op 或缺失文件会回退到经验/SOL 模型。若目标是新硬件/新 profile 的精确评估，优先补齐 collector 与 NCCL 数据，再把 `SILICON` 作为验收手段。

## 当前边界

- 不考虑 `MinimalA2AAttnOp`、`UlyssesAttention_VSA`、SLA/VSA 稀疏路径。
- Scheduler 更新、CPU offload、Cache-DiT 控制逻辑未单独计时。
- Blackwell 可复用同一代码路径，但需要对应 `b200_sxm/sglang/<version>/wan_*.txt`、GEMM 数据与可选 `nccl/<version>/nccl_perf.txt`；缺少 NCCL 实测时 collective 在 HYBRID 下回退 empirical，P2P 始终按目标 system YAML 带宽估算。

## 2026-05-20 RTX6000PRO 通信实测修正记录

- `WanParallelComm` 对 NCCL collective 不再强制 `DatabaseMode.EMPIRICAL`，而是调用 `query_nccl()` 继承当前数据库模式；在 `rtxpro6000_server/supernode` 的 `HYBRID` 运行中，AllReduce/AllGather/AllToAll 能使用 `nccl/sim/nccl_perf.txt` 实测数据。
- `WanParallelComm` 对 P2P 仍显式使用 `DatabaseMode.EMPIRICAL`，因为当前没有 P2P 专项实测表；`query_p2p()` 新增 `num_gpus` 参数，用 `_get_p2p_bandwidth(num_gpus)` 区分普通 2 卡节点跨节点链路与 32 卡 supernode 节点内链路。
- `cli_estimate()` 在 Wan/static 路径下把 `context_source_dict` 回填到 `EstimateResult.per_ops_source`，便于 sweep 日志统计 silicon/empirical/mixed 来源。
- `rtxpro6000_server.yaml` 与 `rtxpro6000_supernode.yaml` 的 `misc.nccl_version` 指向 `sim`，对应 `systems/data/<system>/nccl/sim/nccl_perf.txt`。

## 2026-05-19 并行与 Batch 修正记录

- `run.py` 现在在调用 `cli_estimate()` 前先写入完整 Wan 输入参数；即使运行中报错，日志也会保留 `batch_size/tp_size/sp_size/ulysses_degree/ring_degree` 等关键 debug 信息。
- `run_static()` 对 Wan 结果补充 `sp/sp_size/ulysses_degree/ring_degree/sp_algorithm/wan_batch_model`，并将 `num_total_gpus` 修正为 `tp*pp*dp*sp`；`parallel` 字符串追加 `sp{sp}u{ulysses}r{ring}`。
- Wan 并行合法性增加 32 卡上限检查：`tp*pp*dp*sp <= 32`，且 DiT 要求 `num_heads % tp == 0`、`(num_heads/tp) % ulysses_degree == 0`。
- `batch_size` 当前按 `serial_single_video_collect` 处理：collector 只采单视频 kernel，SDK 将整条 Wan pipeline latency/energy 乘以 batch，以保守近似多视频请求；日志会显式 warning。若未来要模拟真实 batched video kernel，需要新增 batch>1 的 Wan collector 数据并切换为 batched shape 查询。
- `PerfDatabase` 对 Wan `rope/attention/elementwise` 放宽为同语义前缀最近邻兜底，并保留 warning 输出，避免 HYBRID 在扩展 SP=16/32 但旧数据未补采时直接失败。
- VAE `avg_down3d` 与 decode 起始 height/width 对齐 collector 的偶数空间维度逻辑，避免 SGLang `AvgDown3D` 仅 pad 时间维导致的奇偶 mismatch。

## 2026-05-22 CFG / cfg_parallel 仿真实现记录

SGLang Wan 的 CFG 是请求级逻辑：当最终请求的 `cfg_scale`（优先 `true_cfg_scale`，否则 `guidance_scale`）大于 1 且 `negative_prompt` 非空时，`Req.do_classifier_free_guidance=True`。`enable_cfg_parallel` 不会主动开启 CFG，只改变已开启 CFG 请求的 cond/uncond 执行方式。

SDK 已新增 `RuntimeConfig` 与 `cli_estimate()` 参数：`do_classifier_free_guidance`、`enable_cfg_parallel`、`guidance_scale`、`true_cfg_scale`、`negative_prompt`。其中 `do_classifier_free_guidance=None` 表示沿用模型 profile 默认采样参数推导；Wan2.2 T2V/I2V/TI2V 默认 `guidance_scale>1` 且默认负向 prompt 非空，因此默认会启用 CFG。

| 场景 | DiT forward 建模 | T5 text encoder 建模 | 额外通信 | GPU 计数 |
|---|---|---|---|---|
| 无 CFG，无 cfg_parallel | `1x` | `1x` | 无 | `tp*pp*dp*sp` |
| 无 CFG，有 cfg_parallel | `1x`，非 CFG rank 空转 | `1x` | 当前未建模 barrier 小开销 | `tp*pp*dp*sp*2` |
| CFG，无 cfg_parallel | cond/uncond 串行，`2x` DiT | 正/负 prompt 编码，`2x` T5 | 无 | `tp*pp*dp*sp` |
| CFG，有 cfg_parallel | cond/uncond 分到 CFG rank，单 rank `1x` DiT | 正/负 prompt 编码仍按 `2x` T5 | 每 denoise step 一次 CFG 组 AllReduce | `tp*pp*dp*sp*2` |

实现位置：

- `config.py::RuntimeConfig` 增加 CFG 请求/执行字段。
- `models.py::WanVideoModel._resolve_cfg_flags()` 根据显式开关或 Wan 默认 profile 推导 CFG。
- `models.py::_build_wan_pipeline()` 对 DiT 主干使用 `cfg_denoise_scale`，串行 CFG 下对 patch、block、final norm、proj out 等 DiT forward 算子整体翻倍；cfg_parallel 下保持单倍并新增 `wan_dit_cfg_noise_all_reduce`。
- `models.py::_build_t5_ops(scale_factor)` 在 CFG 请求下把正/负 prompt text encoder 编码按 `2x` 计入一次性条件阶段。
- `base_backend.py::run_static()` 将 `cfg_degree` 乘入 `num_total_gpus`，并在 `parallel` 字符串追加 `cfg2`；结果 raw 增加 `do_classifier_free_guidance/enable_cfg_parallel/cfg_degree`。
- `cli/main.py`、`cli/api.py::cli_estimate()` 与调试 `run.py` 增加 CFG 输入字段；聚合估计和 disagg 估计都会把 CFG/通信参数写入 `RuntimeConfig`，其中 Wan 正常仍推荐走 `mode="agg"` 的 `static_ctx` 路径。
- RTX6000PRO sweep 脚本增加 CFG 输入字段，summary 表输出 `cfg/cfg_parallel/cfg_degree`；若开启 `enable_cfg_parallel`，脚本按 `cfg_degree=2` 预留 CFG 组 GPU，避免 32 卡扫描时把 `tp*sp*cfg_degree` 算超。

验证记录：

- `py_compile` 已覆盖 `config.py`、`models.py`、`base_backend.py`、`operations.py`、`cli/api.py`、`cli/main.py` 与 Wan 调试/sweep 脚本。
- 轻量 smoke：`no_cfg`、串行 CFG、`cfg_parallel` 三种 T2V 配置均可运行；串行 CFG 请求时延约为无 CFG 的 `2x`，`cfg_parallel` 的请求时延接近无 CFG，且 raw 中 `num_total_gpus=2`、`parallel` 追加 `cfg2`。

当前边界：未单独建模 `enable_cfg_parallel=True` 但请求无 CFG 时的 executor barrier 小开销；未建模 guidance rescale 额外 broadcast/std 计算，因为 Wan 默认主路径通常不开启该项，且其成本远小于 DiT 主干。
