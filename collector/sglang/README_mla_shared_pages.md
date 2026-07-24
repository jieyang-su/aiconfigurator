# MLA 共享页局部性诊断

`diagnose_mla_shared_pages.py` 是用于复现 `B-32` / `B-mixed-32` MLA
decode 差异的单卡诊断脚本。它固定测量同一个
`bs=32, s=32768, heads=32` MLA 模块，只改变
`ReqToTokenPool.req_to_token` 中的物理 KV 映射：

| 拓扑 | KV 映射方式 |
|---|---|
| `independent` | 32 个请求使用完全独立的 KV 映射项 |
| `pair_shared_shuffled` | 16 组、每组 2 个请求；组内请求共享前 29,491 个映射项（约 90%），请求顺序采用 seed 11 的 mixed workload shuffle |
| `all_shared` | 32 个请求全部共享相同的前 29,491 个映射项 |

脚本从 `collect_mla_module.py` 导入 `load_model_runner()`，使用相同的
第 0 层 attention 模块、dummy latent 输入、FA3 backend、预热和 CUDA Graph
回放路径。它不是 serving workload，也不会更新 AIC 性能数据库。

## 固定测量契约

精确复现时必须保持以下配置：

- batch size：32，当前脚本会主动拒绝其他 batch size
- 历史 KV 长度：32,768
- 准备新 token 后的有效 decode 序列长度：32,769
- 本地 attention heads：32
- 共享前缀长度：29,491，即 `floor(32768 * 0.9)`
- mixed 拓扑：16 组、每组 2 个请求、seed 11
- attention backend：FA3
- collector 接收的输入和 KV cache dtype：BF16
- 计时单位：单层 MLA 模块每次调用的毫秒数

`SGLANG_TEST_NUM_LAYERS=2` 只用于限制 dummy 模型的构建规模；实际 benchmark
仍然只调用 `model.model.layers[0].self_attn`。只有在与目标 sweep 的 20 层
MLA 聚合值比较时，才将单层结果乘以 20。

这里测量的是完整 MLA 模块，包含 projection、normalization、rotary/KV write
和 attention kernel。虽然 collector 参数记录的是 BF16 compute、KV 和 GEMM
标签，但 DeepSeek 模型路径仍可能为 projection 选择 FP8 DeepGEMM kernel。
实际执行了哪些 kernel，应以 Chrome trace 为准。

## 环境要求与 GPU 预检

使用 HiSim 虚拟环境，并确保目标 H100 空闲且有足够显存运行这个固定 shape。
参考测试使用宿主机 5 号卡，当时的 UUID 为
`GPU-08b65028-397c-269b-e1e7-f6c0aacd51c2`，运行时设备名为
`NVIDIA H100 80GB HBM3`。

首先确认当前 GPU index、UUID 和设备名：

```bash
nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader,nounits
```

然后确认目标 UUID 上没有活动的计算进程：

```bash
nvidia-smi \
  --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader,nounits
```

如果 5 号卡被占用，应直接停止测试，不要终止其他进程，也不要静默切换到其他
GPU。如果 5 号卡当前的 UUID 已变化，需要同步修改下方命令中的
`TARGET_GPU_UUID`。`CUDA_VISIBLE_DEVICES=5` 负责实际绑定 GPU；
`TARGET_GPU_HOST_INDEX` 和 `TARGET_GPU_UUID` 只是写入 `results.json`
的来源追踪字段。

如果在 sandbox 或容器内运行 `nvidia-smi` 时无法访问驱动，需要切换到具有
宿主机 GPU 访问权限的环境。不能把 `nvidia-smi` 失败当作 GPU 空闲。

## 复现带 profiler 的测试

在 collector 目录运行。请使用新的输出目录，因为脚本会覆盖同名的
`results.json` 和 topology trace。

```bash
cd /workspace/hisim_align/aiconfigurator/collector/sglang

CUDA_VISIBLE_DEVICES=5 \
TARGET_GPU_HOST_INDEX=5 \
TARGET_GPU_UUID=GPU-08b65028-397c-269b-e1e7-f6c0aacd51c2 \
FLASHINFER_WORKSPACE_BASE=/workspace/hisim_align/.cache/flashinfer \
XDG_CACHE_HOME=/workspace/hisim_align/.cache/xdg \
TORCH_HOME=/workspace/hisim_align/.cache/torch \
SGLANG_LOAD_FORMAT=dummy \
SGLANG_TEST_NUM_LAYERS=2 \
PYTHONUNBUFFERED=1 \
/workspace/hisim_align/hisim_sglang_align/.venv-hisim/bin/python \
  diagnose_mla_shared_pages.py \
  --output-dir \
  /workspace/hisim_align/diagnostics/mla_shared_pages_gpu5_repro_01
```

上面的命令使用以下默认参数：

```text
--batch-size 32
--seq-len 32768
--num-heads 32
--shared-prefix-len 29491
--cycles 3
--samples-per-cycle 5
--replays-per-sample 100
--profile-replays 5
```

每个 cycle 都会改变 topology 的执行顺序，以降低固定执行顺序造成的偏差。
每个样本是 100 次 CUDA Graph replay 的平均值。每种 topology 在第一轮计时
结束后执行一次 profiler。

## 低噪声计时测试

用于数值比较时，建议增加 replay 次数并关闭 profiler：

```bash
cd /workspace/hisim_align/aiconfigurator/collector/sglang

CUDA_VISIBLE_DEVICES=5 \
TARGET_GPU_HOST_INDEX=5 \
TARGET_GPU_UUID=GPU-08b65028-397c-269b-e1e7-f6c0aacd51c2 \
FLASHINFER_WORKSPACE_BASE=/workspace/hisim_align/.cache/flashinfer \
XDG_CACHE_HOME=/workspace/hisim_align/.cache/xdg \
TORCH_HOME=/workspace/hisim_align/.cache/torch \
SGLANG_LOAD_FORMAT=dummy \
SGLANG_TEST_NUM_LAYERS=2 \
PYTHONUNBUFFERED=1 \
/workspace/hisim_align/hisim_sglang_align/.venv-hisim/bin/python \
  diagnose_mla_shared_pages.py \
  --output-dir \
  /workspace/hisim_align/diagnostics/mla_shared_pages_gpu5_repro_timing_01 \
  --cycles 6 \
  --samples-per-cycle 5 \
  --replays-per-sample 200 \
  --profile-replays 0
```

关闭 profiler 后不会生成 `.trace.json` 文件，但所有计时和 KV 映射字段仍会
写入 `results.json`。

## 输出与结果校验

输出目录包含：

- `results.json`：测试配置、设备来源、mixed 请求顺序、物理映射计数、
  原始计时样本、汇总统计以及 profiler 中耗时最高的 kernel
- `independent.trace.json`、`pair_shared_shuffled.trace.json` 和
  `all_shared.trace.json`：当 `--profile-replays` 大于 0 时生成的
  Chrome/PyTorch profiler trace

分析时延前，先确认 `results.json` 中的以下约束：

| 字段 | 期望值 |
|---|---:|
| `cuda_visible_devices` | `5` |
| `sglang_attention_backend` | `fa3` |
| `active_seq_len` | 32,769 |
| independent 逻辑映射数 / 唯一物理映射数 | 1,048,608 / 1,048,608 |
| pair-shared 逻辑映射数 / 唯一物理映射数 | 1,048,608 / 576,752 |
| all-shared 逻辑映射数 / 唯一物理映射数 | 1,048,608 / 134,387 |

可以使用以下只读命令快速检查：

```bash
rg -n \
  '"(device_name|cuda_visible_devices|active_seq_len|sglang_attention_backend|logical_page_count|unique_page_count|unique_over_logical|mean_ms|median_ms|stdev_ms)"' \
  /workspace/hisim_align/diagnostics/mla_shared_pages_gpu5_repro_01/results.json
```

需要同时检查 `mean_ms` 和 `median_ms`。参考测试中的 pair-shared 场景出现过
多个稳定的计时档位，因此不能根据单个样本下结论。如果标准差异常偏大，应重新
确认 GPU 仍然空闲，然后重复低噪声计时测试。

## 5 号卡参考结果

关闭 profiler、运行 6 个 cycle 的参考结果如下：

| 拓扑 | 单层均值（ms） | 单层中位数（ms） | 均值 x20（ms） |
|---|---:|---:|---:|
| 完全独立 | 0.454402 | 0.453918 | 9.088 |
| 两请求一组共享并打乱顺序 | 0.313206 | 0.313234 | 6.264 |
| 32 请求全部共享 | 0.269123 | 0.268630 | 5.382 |

调查过程中使用的对照值分别为：AIC `B-32 MLA = 9.115 ms`、实机
`B-mixed-32 MLA = 6.154 ms` 和实机 `B-32 MLA = 5.534 ms`。这些值只用于
对照，不是针对不同 GPU、驱动、SGLang commit 或 kernel build 的通过阈值。

带 profiler 的参考测试中，主导 FA3 kernel 每次 replay 的耗时约为：
完全独立 0.405727 ms、pair-shared 0.252148 ms、all-shared 0.217810 ms。
projection kernel 的耗时基本不变，因此本次观测到的差异集中在 MLA 模块的
attention 部分。

参考产物保存在：

- `/workspace/hisim_align/diagnostics/mla_shared_pages_gpu5_bs32_s32768_h32_seed11_20260722/`
- `/workspace/hisim_align/diagnostics/mla_shared_pages_gpu5_bs32_s32768_h32_seed11_20260722_repeat2/`

## 结论边界

该实验用于判断：在 shape、请求顺序、模块代码和测量路径保持不变时，物理 KV
映射复用是否会改变 FA3 MLA 模块时延。它本身不能进一步区分 L2 cache 复用、
TLB/页表影响、HBM 流量或 FA3 调度行为。若要继续拆分这些因素，需要使用
Nsight Compute/System 的硬件计数器。

它也不是 TP4 serving benchmark，而是单卡、本地 heads 的模块级微基准。不能
把这里的时延直接替换为 TTFT/TPOT，也不能在没有独立校准决策的情况下用该脚本
改写 AIC 数据。

## 常见问题

- `NVIDIA-SMI has failed`：当前进程无法正常访问 GPU 驱动，应停止测试并切换
  到宿主机 GPU 环境。
- FlashInfer cache 权限错误：保留命令中的 `FLASHINFER_WORKSPACE_BASE` 和
  `XDG_CACHE_HOME`，它们指向 workspace 内可写的缓存目录。
- CUDA OOM：重新检查目标 GPU 上的活动进程。修改 batch size 或 sequence
  length 后将不再是本次要求复现的测试点。
- SGLang import 或版本错误：使用上面指定的 `.venv-hisim` 解释器，不要使用
  系统 Python。
- 没有生成 trace 文件：当 `--profile-replays 0` 时属于预期行为，检查
  `results.json` 是否已完整写入。
