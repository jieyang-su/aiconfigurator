# DeepSeekV3 MoE H100 Calibration Plan

本文档描述在 SGLang v0.5.9 上对 DeepSeekV3 MoE 相关算子做 AIC 仿真校准的计划。目标是让 H100 实机完整推理 trace 与 AIC collector/simulator 在 MoE 模块和底层算子边界上可对齐、可复现、可逐项分析误差。

## 目标

1. 在 H100 机器上用 SGLang v0.5.9 执行 DeepSeekV3 形状的完整 offline/Engine 推理，并保存 torch profiler chrome trace。
2. 在 trace 中提取 DeepSeekV3 MoE 模块级和子阶段级时延，包括 router、topk、shared experts、routed experts、dispatch、expert compute、combine、output all-reduce。
3. 用 AIC 现有 `moe` / `wideep_moe` collector 数据和仿真结果对齐实机观测值，量化直接命中、插值、不同 token/expert 分布下的误差。
4. 保持实机运行路径接近真实 SGLang 模型推理，不用 standalone MoE microbenchmark 替代实机 trace。

## 范围

本阶段固定硬件为 H100，后续再扩展 H20。模型形状固定 DeepSeekV3：

- `hidden_size = 7168`
- `moe_intermediate_size = 2048`
- `num_experts = 256`
- `topk = 8`
- `n_shared_experts` 按 SGLang DeepSeekV3 config 保持原值；普通 `moe_perf` 校准命令必须显式加 `--disable-shared-experts-fusion`，让 shared experts 保持在 `shared_experts` 事件内，避免混入 `collector/moe`

AIC 中已有模型 case 入口：

- `collector/cases/models/DeepseekV3ForCausalLM_cases.yaml`
- `collector/cases/base_ops/moe.yaml`
- `collector/sglang/collect_moe.py`
- `collector/wideep/sglang/collect_deepep_moe.py`

## 实机 Profile 边界

SGLang 侧新增可开关的 torch profiler `record_function` 打点，命名与现有 profiler 方法保持一致，使用 `torch.profiler.record_function` 进入 chrome trace，不引入新的计时系统。

启用方式：

```bash
export SGLANG_AIC_MOE_PROFILE=1
export SGLANG_TORCH_PROFILER_DIR=/path/to/profile_out
```

打点名称：

- `aic_moe/layer_{layer_id}/module`
- `aic_moe/layer_{layer_id}/router`
- `aic_moe/layer_{layer_id}/collector/moe`
- `aic_moe/layer_{layer_id}/topk`
- `aic_moe/layer_{layer_id}/shared_experts`
- `aic_moe/layer_{layer_id}/routed_experts`
- `aic_moe/layer_{layer_id}/routed/dispatch`
- `aic_moe/layer_{layer_id}/routed/dispatch_a`
- `aic_moe/layer_{layer_id}/routed/dispatch_b`
- `aic_moe/layer_{layer_id}/routed/compute`
- `aic_moe/layer_{layer_id}/routed/combine`
- `aic_moe/layer_{layer_id}/routed/combine_a`
- `aic_moe/layer_{layer_id}/routed/combine_b`
- `aic_moe/layer_{layer_id}/routed/all_reduce`
- `aic_moe/layer_{layer_id}/output_postprocess`
- `aic_moe/layer_{layer_id}/output_all_reduce`

默认不开启该环境变量时，profile 上下文退化为 `nullcontext()`，正常推理路径不应有额外 trace 事件。

## 实机运行设计

1. 使用 `booleimg.myaddr.io/lmsysorg/sglang:v0.5.9` 容器。
2. H100 上最多使用最后 2 张卡；第一阶段建议先单卡，确认边界和数值稳定后再做 2 卡 TP/EP。
3. 使用 DeepSeekV3 config + dummy weight，通过 `--json-model-override-args '{"num_hidden_layers":6,"first_k_dense_replace":3}'` 或调整层数/稀疏层起点，不需要复制/修改 config。DeepSeekV3 默认前 `first_k_dense_replace=3` 层是 dense MLP，因此 `num_hidden_layers=3` 不会产生 MoE profile；第一轮至少用 6 层，得到 3 个 MoE 层样本。
4. 关闭 cudagraph 或确保 profiler 边界仍可见；第一阶段建议关闭 cudagraph，避免图捕获折叠 Python-side record_function 边界。
5. 普通 MoE `moe_perf` 校准必须关闭 shared experts fusion：所有 SGLang 启动命令都加 `--disable-shared-experts-fusion`。这样 shared experts 独立进入 `shared_experts` 事件；正式对齐 AIC 普通 collector 时使用 `topk+routed/compute`，`collector/moe` 用来检查 SGLang routed wrapper 额外开销。
6. 通过 SGLang Engine/offline batch 入口执行完整推理，profile 从请求前启动，到请求结束后停止。

建议先覆盖这些 token 规模：

- 小 decode/短 prefill：`num_tokens = 1, 2, 4, 8, 16, 32`
- 常用 prefill：`64, 128, 256, 512, 1024, 2048, 4096`
- 大块 prefill：`8192, 12288, 16384`，视 H100 显存和 reduced config 可行性选择

每个配置至少跑 5 次，前 1 次作为 warmup，不进入误差统计。trace 提取时按 `layer_id` 聚合，优先比较 MoE 层，跳过 dense 层。

## AIC 仿真与 Collector 对齐

AIC collector 侧不要新增 DeepSeekV3 MoE shape，优先复用已有 YAML：

- DeepSeekV3 model case 给出 MoE 静态形状。
- `moe.yaml` 给出 token count、TP/EP/GPU count、balanced/power-law expert 分布。
- SGLang H100 上 `fp8_block` 和 `bfloat16` 是主要校准对象；`int4_wo` 可作为后续低优先级。

对齐字段：

- 模型：`model_path=deepseek-ai/DeepSeek-V3`
- 硬件：`sm=90`
- 框架：`framework=sglang`
- MoE 类型：`bfloat16` 或 `fp8_block`
- 形状：`num_tokens, hidden_size, inter_size, topk, num_experts`
- 并行：`tensor_parallel_size, expert_parallel_size, gpu_count`
- 分布：`token_expert_distribution, power_law_alpha`

普通 AIC `collect_moe.py` 的 timed loop 是 `select_experts(...) + fused_moe(...)`，不包含 DeepSeek 外层 gate/router/shared experts，也不包含完整 `FusedMoE.forward` 的 dispatcher/combine wrapper。因此普通单卡 `query_moe(..., moe_backend=None)` 的严格对齐目标是实机 trace 的组合 stage `topk+routed/compute`。`collector/moe` 仍然会输出，它表示 SGLang routed wrapper 口径：`topk + self.experts(...)`，可用于检查 dispatcher/combine wrapper 开销，但不要把它作为修 AIC fused_moe 表的第一证据。SGLang 命令必须加 `--disable-shared-experts-fusion`；如果不关闭 fusion，DeepSeek shared experts 可能被融合进专家路径，任一普通 MoE 对齐口径都会被污染。`routed/compute` 单独还可对齐 WideEP collector 中直接调用 `run_moe_core(...)` 的 compute 表。`routed/dispatch` 和 `routed/combine` 对齐 WideEP/DeepEP 通信路径；`output_postprocess` 解释 routed scaling/shared-add 等尾部 elementwise 开销；非 TBO 的 `module` 对齐完整 DeepSeek MoE forward 闭包，只用于模块闭合检查，不能直接和单个 AIC MoE 表比较。

## 分阶段实验

### 阶段 1：同形状直接命中

选取 collector 已覆盖的 token count，使用 balanced 分布和 full-miss/prefill 场景。目标是建立 1 个 AIC 仿真结果对 N 个实机 trace 样本的误差分布。

输出：

- 每层 MoE module latency 的均值、p50、p90、std
- `router/topk/shared/routed` breakdown
- `dispatch/compute/combine` breakdown
- AIC vs 实机的绝对误差和相对误差

### 阶段 2：插值误差

选取 collector token count 中间值，例如 `96, 160, 384, 768, 1536, 3072, 6144`。AIC 侧使用 PerfDatabase 插值，实机侧按同样方法提取 trace。目标是分离模型插值误差与 kernel 误差。

### 阶段 3：expert 分布误差

使用 `balanced`、`power_law(alpha=1.01)`、`power_law(alpha=1.2)` 三类分布。实机侧如果完整推理难以稳定控制路由分布，先记录实际 topk 直方图，再按实际分布匹配或修正 AIC 查询。

### 阶段 4：多卡 DeepEP/WideEP

在单卡结果稳定后扩展到最后 2 张 H100。关注 `routed/dispatch`、`routed/combine`、`output_all_reduce` 是否成为主要误差来源。该阶段需要记录 NCCL/DeepEP 后端配置、TP/EP 设置、可见 GPU 列表。

## 数据处理建议

trace 解析脚本应按以下规则工作：

1. 读取 chrome trace JSON 或 JSON.GZ。
2. 过滤 `name` 以 `aic_moe/` 开头且 `ph == "X"` 的事件。
3. 按 `layer_id`、阶段名、trace 文件聚合 `dur`。
4. 对同一阶段嵌套事件分别保留 inclusive duration；对 module breakdown 使用子阶段求和时要避免重复计算。
5. `parse-trace` 输出事件级 CSV，字段为：`run, trace, phase, num_tokens, layer_id, stage, duration_us, ts, pid, tid`。TP/EP、dtype、distribution、model_path、hardware 等运行配置不重复写进每一行事件，而是保存在 `${OUT_DIR}/configs/run_manifest.env`、`gpu_info.csv`、`aic_moe_predictions.csv` 和最终 report 中。

## 远程执行检查清单

1. 确认容器内 SGLang checkout 包含本次 `SGLANG_AIC_MOE_PROFILE` 打点。
2. 设置 `SGLANG_AIC_MOE_PROFILE=1` 和 `SGLANG_TORCH_PROFILER_DIR`。
3. 使用 `--json-model-override-args` 控制 DeepSeekV3 层数，确认 MoE 层存在且 trace 中能看到 `aic_moe/` 事件。
4. 关闭 cudagraph 做第一轮 trace。
5. 保存 profile trace、运行配置 JSON、SGLang commit、AIC commit、GPU 型号、CUDA/NCCL/driver 版本。
6. 跑 AIC collector/sim 时使用相同 token count、TP/EP、dtype、distribution。
7. 每轮实验结束生成误差表和 breakdown 图，再决定是否扩展下一个阶段。

## 可执行 Runbook

本节按“远程 H100 上可以直接照着执行”的粒度写。推荐优先用脚本模板：

- `tools/moe_calibration/run_h100_dsv3_moe.sh`：串联环境记录、SGLang trace、trace 解析、AIC query、误差表、collector smoke。
- `tools/moe_calibration/aic_moe_calibrate.py`：提供 `self-test`、`parse-trace`、`validate-trace`、`summarize-trace`、`breakdown-trace`、`summarize-expert-distribution`、`query-aic`、`query-deepep-dispatch`、`compare`、`make-report` 等子命令。
- `docs/deepseekv3_moe_h100_smoke_checklist.md`：H100 上第一轮 smoke 的最短验收清单。

### 0. 变量约定

后续命令统一使用这些变量；远程机器上按实际路径改：

```bash
export SGLANG_SRC=/workspace/sglang
export AIC_SRC=/workspace/aiconfigurator-jieyang
export MODEL_PATH=deepseek-ai/DeepSeek-V3
export CUDA_VISIBLE_DEVICES=7
export OUT_DIR=/workspace/moe_calibration_runs/h100_dsv3_moe_$(date +%Y%m%d_%H%M%S)
export BACKEND_VERSION=0.5.9
export DEEPEP_DISPATCH_BACKEND_VERSION="${BACKEND_VERSION}"
export NUM_HIDDEN_LAYERS=6
export FIRST_K_DENSE_REPLACE=3
export EXPECTED_MOE_LAYERS=$(( NUM_HIDDEN_LAYERS > FIRST_K_DENSE_REPLACE ? NUM_HIDDEN_LAYERS - FIRST_K_DENSE_REPLACE : 0 ))
export TP_SIZE=1
export EP_SIZE=1
export MOE_TP_SIZE=1
export MOE_EP_SIZE=1
export HIDDEN_SIZE=7168
export INTER_SIZE=2048
export TOPK=8
export NUM_EXPERTS=256
export QUANT_MODE=bfloat16
export DISTRIBUTION=balanced
export WIDEEP_DISTRIBUTION=uniform
export DEEPEP_NODE_NUM=1
export DEEPEP_SMS=20
mkdir -p "${OUT_DIR}"/{profile,logs,aic,parsed,configs}
```

`MODEL_PATH` 可以是 HuggingFace id，也可以是远程机器上已有的本地模型/config 目录。`--load-format dummy` 只表示权重走 dummy load，SGLang 仍需要能读到 DeepSeekV3 config/tokenizer。

`DISTRIBUTION=balanced` 对应普通 `collector/sglang/collect_moe.py` 和 `moe.yaml` 的 balanced 分布；`WIDEEP_DISTRIBUTION=uniform` 对应 `collector/wideep/sglang/collect_deepep_moe.py` 写入 WideEP 表时的均匀分布名。两者不要混用，否则 `query_moe(..., moe_backend=deepep_moe)` 可能查不到同一行。

`HIDDEN_SIZE/INTER_SIZE/TOPK/NUM_EXPERTS` 来自 AIC 的 DeepSeekV3 MoE case：
`collector/cases/models/DeepseekV3ForCausalLM_cases.yaml` 中 `hidden_size=7168`、`inter_size=2048`、`topk=8`、`num_experts=256`。这些变量必须和 AIC query、SGLang config、collector plan 保持一致。

`EXPECTED_MOE_LAYERS` 是本轮实际应该出现 profile 的 MoE 层数，不等于总 `NUM_HIDDEN_LAYERS`。DeepSeekV3/SGLang 的稀疏层判定是 `layer_id >= first_k_dense_replace`，所以默认 `NUM_HIDDEN_LAYERS=6`、`FIRST_K_DENSE_REPLACE=3` 时，期望 MoE 层是 `3,4,5` 共 3 层。

层数不需要手动复制/修改 config，SGLang 直接用：

```bash
--json-model-override-args "{\"num_hidden_layers\":${NUM_HIDDEN_LAYERS},\"first_k_dense_replace\":${FIRST_K_DENSE_REPLACE}}"
```

如果要跑 40 层：

```bash
export NUM_HIDDEN_LAYERS=40
```

### 1. Docker 启动命令

单卡第一轮只暴露最后一张 H100：

```bash
docker run --gpus '"device=7"' --ipc=host --net=host --rm -it \
  -v /workspace:/workspace \
  -v /models:/models \
  booleimg.myaddr.io/lmsysorg/sglang:v0.5.9 \
  /bin/bash
```

两卡 DeepEP/WideEP 阶段暴露最后两张 H100：

```bash
docker run --gpus '"device=6,7"' --ipc=host --net=host --rm -it \
  -v /workspace:/workspace \
  -v /models:/models \
  booleimg.myaddr.io/lmsysorg/sglang:v0.5.9 \
  /bin/bash
```

容器内确认 GPU：

```bash
nvidia-smi
nvidia-smi --query-gpu=index,name,uuid,compute_cap,driver_version,memory.total --format=csv \
  | tee "${OUT_DIR}/configs/gpu_info.csv"
```

### 2. 代码和 Python 静态检查

```bash
cd "${SGLANG_SRC}"
git rev-parse HEAD | tee "${OUT_DIR}/configs/sglang_commit.txt"
grep -R "SGLANG_AIC_MOE_PROFILE" -n \
  python/sglang/srt/models/deepseek_v2.py \
  python/sglang/srt/layers/moe/fused_moe_triton/layer.py
grep -R "aic_moe/layer" -n \
  python/sglang/srt/models/deepseek_v2.py \
  python/sglang/srt/layers/moe/fused_moe_triton/layer.py
python -m py_compile \
  python/sglang/srt/models/deepseek_v2.py \
  python/sglang/srt/layers/moe/fused_moe_triton/layer.py
```

```bash
cd "${AIC_SRC}"
git rev-parse HEAD | tee "${OUT_DIR}/configs/aic_commit.txt"
python -m py_compile \
  collector/collect.py \
  collector/sglang/collect_moe.py \
  collector/wideep/sglang/collect_deepep_moe.py \
  tools/moe_calibration/aic_moe_calibrate.py
python tools/moe_calibration/aic_moe_calibrate.py self-test

# 可选：只跑 runner 内置检查，不启动 SGLang/collector。
bash tools/moe_calibration/run_h100_dsv3_moe.sh check-only

```

### 3. 记录环境版本

```bash
python - <<'PY' | tee "${OUT_DIR}/configs/python_packages.txt"
import importlib.metadata as m
for pkg in ["torch", "sglang", "transformers", "flashinfer-python"]:
    try:
        print(pkg, m.version(pkg))
    except Exception as exc:
        print(pkg, "NA", exc)
PY
```

### 4. 一键脚本路径

如果你只想跑单卡 smoke + token sweep + AIC query + compare：

```bash
cd "${AIC_SRC}"
chmod +x tools/moe_calibration/run_h100_dsv3_moe.sh
SGLANG_SRC="${SGLANG_SRC}" \
AIC_SRC="${AIC_SRC}" \
MODEL_PATH="${MODEL_PATH}" \
CUDA_VISIBLE_DEVICES=7 \
NUM_HIDDEN_LAYERS=6 \
FIRST_K_DENSE_REPLACE=3 \
EXPECTED_MOE_LAYERS=3 \
TP_SIZE=1 \
EP_SIZE=1 \
MOE_TP_SIZE=1 \
MOE_EP_SIZE=1 \
QUANT_MODE=bfloat16 \
DISTRIBUTION=balanced \
HIDDEN_SIZE=7168 \
INTER_SIZE=2048 \
TOPK=8 \
NUM_EXPERTS=256 \
TOKEN_SWEEP="128 512 2048 4096" \
INTERP_TOKEN_SWEEP="96 160 384 768 1536 3072 6144" \
OUT_DIR="${OUT_DIR}" \
bash tools/moe_calibration/run_h100_dsv3_moe.sh
```

脚本最终生成：

- `${OUT_DIR}/profile/**.trace.json*`：SGLang 实机 trace。
- `${OUT_DIR}/parsed/sglang_aic_moe_events.csv`：trace 中所有 `aic_moe/` 事件。
- `${OUT_DIR}/parsed/sglang_aic_moe_trace_validation.csv`：按 token/stage 检查关键 profile 点是否存在。
- `${OUT_DIR}/parsed/sglang_aic_moe_summary.csv`：按 `num_tokens, stage` 聚合后的实机均值/p50/p90。
- `${OUT_DIR}/parsed/sglang_aic_moe_module_breakdown.csv`：按单个 `module` 事件做子阶段闭合分析，输出组件和、残差和缺失组件。
- `${OUT_DIR}/parsed/sglang_aic_moe_module_breakdown_summary.csv`：按 `num_tokens` 聚合后的 module 闭合残差均值/p50/min/max。
- `${OUT_DIR}/expert_distribution/expert_distribution_recorder_*.pt`：SGLang expert distribution recorder 原始输出。
- `${OUT_DIR}/parsed/sglang_expert_distribution_summary.csv`：按层聚合后的实际 expert 分布摘要。`logical_layer_index` 是 recorder 内部 MoE 层序号，`layer_id = logical_layer_index + FIRST_K_DENSE_REPLACE`，用于和 trace 的真实 DeepSeekV3 layer id 对齐；`count_kind=physical_to_logical` 表示工具已用 recorder 的 `last_physical_to_logical_map` 将 physical expert 计数聚合回 logical expert。该表用于判断 AIC 查询应使用 balanced、power-law 还是实际分布修正。
- `${OUT_DIR}/parsed/aic_moe_predictions.csv`：AIC `query_moe` 预测。
- `${OUT_DIR}/parsed/aic_vs_sglang_topk_compute.csv`：实机 `topk+routed/compute` vs AIC 普通 `moe_perf` 的严格对齐误差。
- `${OUT_DIR}/parsed/aic_vs_sglang_collector_moe.csv`：实机 `collector/moe` vs AIC 普通 `moe_perf` 的 SGLang routed-wrapper sanity check。
- `${OUT_DIR}/parsed/aic_wideep_compute_predictions.csv`：AIC `query_moe(..., moe_backend=deepep_moe)` 的 WideEP compute 预测，需在 WideEP 阶段执行。
- `${OUT_DIR}/parsed/aic_vs_sglang_wideep_compute.csv`：实机 `routed/compute` vs AIC WideEP compute 误差，需在 WideEP 阶段执行。
- `${OUT_DIR}/parsed/deepseekv3_moe_calibration_report.md`：自动汇总 AIC 误差、module 闭合残差和实际 expert 分布热点的 Markdown 报告。
- `${OUT_DIR}/aic/collector_moe_smoke/**`：AIC collector smoke 输出。
- `${OUT_DIR}/aic_systems_overlay/data/h100_sxm/sglang/0.5.9/*moe*perf.parquet`：本轮 collector 产物转换后的 PerfDatabase overlay。
- `${OUT_DIR}/configs/aic_systems_root.txt`：本轮 `query-aic` 实际读取的 AIC systems root。
- `${OUT_DIR}/configs/aic_overlay_files.txt`：overlay 中实际存在的 MoE parquet 文件。
- `${OUT_DIR}/configs/stage_alignment_map.csv`：SGLang profile stage 到 AIC query/table 的对齐映射。
- `${OUT_DIR}/configs/run_manifest.env`：本轮关键参数记录。
- `${OUT_DIR}/logs/commands.xtrace.log`：脚本实际执行命令记录。

默认脚本只跑单卡普通 MoE 边界。DeepEP dispatch+combine 的
`${OUT_DIR}/parsed/aic_deepep_dispatch_predictions.csv` 和
`${OUT_DIR}/parsed/aic_vs_sglang_deepep_dispatch_combine.csv` 需要在第 10、11 节的两卡 DeepEP/WideEP 阶段单独执行。

如果只想打印 server 启动和 curl profile 命令，不跑实验：

```bash
cd "${AIC_SRC}"
SGLANG_SRC="${SGLANG_SRC}" \
AIC_SRC="${AIC_SRC}" \
MODEL_PATH="${MODEL_PATH}" \
CUDA_VISIBLE_DEVICES=7 \
NUM_HIDDEN_LAYERS=6 \
FIRST_K_DENSE_REPLACE=3 \
EXPECTED_MOE_LAYERS=3 \
OUT_DIR="${OUT_DIR}" \
bash tools/moe_calibration/run_h100_dsv3_moe.sh print-server-commands
```

### 5. SGLang 实机 trace：bench_one_batch

这是最小、最可控的完整模型 forward 路径。注意 `bench_one_batch.py` 的参数名是 `--batch-size`。

```bash
cd "${SGLANG_SRC}/python"
export PYTHONPATH="${SGLANG_SRC}/python:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=7
export SGLANG_AIC_MOE_PROFILE=1
export SGLANG_TORCH_PROFILER_DIR="${OUT_DIR}/profile/smoke_prefill_isl_128"
mkdir -p "${SGLANG_TORCH_PROFILER_DIR}"

python -m sglang.bench_one_batch \
  --model-path "${MODEL_PATH}" \
  --load-format dummy \
  --tp-size 1 \
  --ep-size 1 \
  --json-model-override-args "{\"num_hidden_layers\":${NUM_HIDDEN_LAYERS},\"first_k_dense_replace\":${FIRST_K_DENSE_REPLACE}}" \
  --disable-cuda-graph \
  --disable-shared-experts-fusion \
  --batch-size 1 \
  --input-len 128 \
  --output-len 1 \
  --profile \
  --profile-stage prefill \
  --profile-filename-prefix dsv3_moe_smoke_prefill_isl_128 \
  2>&1 | tee "${OUT_DIR}/logs/sglang_smoke_prefill_isl_128.log"
```

直接命中 collector token 点：

```bash
for ISL in 128 512 2048 4096; do
  export SGLANG_TORCH_PROFILER_DIR="${OUT_DIR}/profile/direct_prefill_isl_${ISL}"
  mkdir -p "${SGLANG_TORCH_PROFILER_DIR}"
  python -m sglang.bench_one_batch \
    --model-path "${MODEL_PATH}" \
    --load-format dummy \
    --tp-size 1 \
    --ep-size 1 \
    --json-model-override-args "{\"num_hidden_layers\":${NUM_HIDDEN_LAYERS},\"first_k_dense_replace\":${FIRST_K_DENSE_REPLACE}}" \
    --disable-cuda-graph \
    --disable-shared-experts-fusion \
    --batch-size 1 \
    --input-len "${ISL}" \
    --output-len 1 \
    --profile \
    --profile-stage prefill \
    --profile-filename-prefix "dsv3_moe_direct_prefill_isl_${ISL}" \
    2>&1 | tee "${OUT_DIR}/logs/sglang_direct_prefill_isl_${ISL}.log"
done
```

插值点：

```bash
for ISL in 96 160 384 768 1536 3072 6144; do
  export SGLANG_TORCH_PROFILER_DIR="${OUT_DIR}/profile/interp_prefill_isl_${ISL}"
  mkdir -p "${SGLANG_TORCH_PROFILER_DIR}"
  python -m sglang.bench_one_batch \
    --model-path "${MODEL_PATH}" \
    --load-format dummy \
    --tp-size 1 \
    --ep-size 1 \
    --json-model-override-args "{\"num_hidden_layers\":${NUM_HIDDEN_LAYERS},\"first_k_dense_replace\":${FIRST_K_DENSE_REPLACE}}" \
    --disable-cuda-graph \
    --disable-shared-experts-fusion \
    --batch-size 1 \
    --input-len "${ISL}" \
    --output-len 1 \
    --profile \
    --profile-stage prefill \
    --profile-filename-prefix "dsv3_moe_interp_prefill_isl_${ISL}" \
    2>&1 | tee "${OUT_DIR}/logs/sglang_interp_prefill_isl_${ISL}.log"
done
```

Decode smoke：

```bash
export SGLANG_TORCH_PROFILER_DIR="${OUT_DIR}/profile/decode_ntok_1_isl_128_osl_64"
mkdir -p "${SGLANG_TORCH_PROFILER_DIR}"
python -m sglang.bench_one_batch \
  --model-path "${MODEL_PATH}" \
  --load-format dummy \
  --tp-size 1 \
  --ep-size 1 \
  --json-model-override-args "{\"num_hidden_layers\":${NUM_HIDDEN_LAYERS},\"first_k_dense_replace\":${FIRST_K_DENSE_REPLACE}}" \
  --disable-cuda-graph \
  --disable-shared-experts-fusion \
  --batch-size 1 \
  --input-len 128 \
  --output-len 64 \
  --profile \
  --profile-stage decode \
  --profile-filename-prefix dsv3_moe_decode_ntok_1_isl_128_osl_64 \
  2>&1 | tee "${OUT_DIR}/logs/sglang_decode_ntok_1_isl_128_osl_64.log"
```

Decode 阶段的 MoE `num_tokens` 是本次 decode forward 的 token 数；这里 `batch-size=1`，所以显式用 `ntok_1`。不要从 `isl_128` 推断 decode compute 的 token 数。

### 6. SGLang 实机 trace：Engine offline throughput

`bench_offline_throughput` 更接近 offline batch Engine 入口，仍不启动 HTTP server。

```bash
cd "${SGLANG_SRC}/python"
export PYTHONPATH="${SGLANG_SRC}/python:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=7
export SGLANG_AIC_MOE_PROFILE=1
export SGLANG_TORCH_PROFILER_DIR="${OUT_DIR}/profile/engine_prefill_isl_512"
mkdir -p "${SGLANG_TORCH_PROFILER_DIR}"

python -m sglang.bench_offline_throughput \
  --model-path "${MODEL_PATH}" \
  --load-format dummy \
  --tp-size 1 \
  --ep-size 1 \
  --json-model-override-args "{\"num_hidden_layers\":${NUM_HIDDEN_LAYERS},\"first_k_dense_replace\":${FIRST_K_DENSE_REPLACE}}" \
  --disable-cuda-graph \
  --disable-shared-experts-fusion \
  --backend engine \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 1 \
  --num-prompts 1 \
  --profile \
  --skip-warmup \
  --result-filename "${OUT_DIR}/logs/engine_prefill_isl_512.jsonl" \
  2>&1 | tee "${OUT_DIR}/logs/sglang_engine_prefill_isl_512.log"
```

### 7. SGLang server 启动命令

如果需要验证 HTTP server 路径，用下面命令启动服务：

```bash
cd "${SGLANG_SRC}/python"
export PYTHONPATH="${SGLANG_SRC}/python:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=7
export SGLANG_AIC_MOE_PROFILE=1
export SGLANG_TORCH_PROFILER_DIR="${OUT_DIR}/profile/server"
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR="${OUT_DIR}/expert_distribution"
mkdir -p "${SGLANG_TORCH_PROFILER_DIR}"
mkdir -p "${SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR}"

python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --load-format dummy \
  --tp-size 1 \
  --ep-size 1 \
  --json-model-override-args "{\"num_hidden_layers\":${NUM_HIDDEN_LAYERS},\"first_k_dense_replace\":${FIRST_K_DENSE_REPLACE}}" \
  --disable-cuda-graph \
  --disable-shared-experts-fusion \
  --expert-distribution-recorder-mode stat \
  --expert-distribution-recorder-buffer-size -1 \
  --host 0.0.0.0 \
  --port 30000 \
  2>&1 | tee "${OUT_DIR}/logs/sglang_server.log"
```

另开一个终端启动/停止 profile、记录实际 expert 分布并发请求：

```bash
curl -X POST http://127.0.0.1:30000/start_profile \
  -H 'Content-Type: application/json' \
  -d '{"profile_by_stage": false, "profile_prefix": "dsv3_moe_server"}'

curl -X POST http://127.0.0.1:30000/start_expert_distribution_record

curl -X POST http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text": "hello", "sampling_params": {"temperature": 0, "max_new_tokens": 1}}'

curl -X POST http://127.0.0.1:30000/stop_expert_distribution_record
curl -X POST http://127.0.0.1:30000/dump_expert_distribution_record
curl -X POST http://127.0.0.1:30000/stop_profile
```

把 expert distribution recorder 的 `.pt` 文件转成 CSV 摘要：

```bash
cd "${AIC_SRC}"
export PYTHONPATH="${AIC_SRC}/src:${AIC_SRC}:${PYTHONPATH:-}"
python tools/moe_calibration/aic_moe_calibrate.py summarize-expert-distribution \
  --input-dir "${OUT_DIR}/expert_distribution" \
  --first-moe-layer-id "${FIRST_K_DENSE_REPLACE}" \
  --output "${OUT_DIR}/parsed/sglang_expert_distribution_summary.csv"
```

摘要字段里重点看：

- `active_experts`：该层实际被命中的 expert 数。
- `cv`：expert assignment 的变异系数，越大说明越偏斜。
- `max_over_mean`：最热 expert 相对平均值的倍数。
- `max_expert_id/max_assignments`：热点 expert 位置和命中数。

如果 `cv` 和 `max_over_mean` 都很低，可以先按 `balanced` 查 AIC；如果明显偏斜，应切到 power-law/实际分布修正路线，不能把这部分误差直接归因给 kernel 或插值。

### 8. 解析 trace

```bash
cd "${AIC_SRC}"
export PYTHONPATH="${AIC_SRC}/src:${AIC_SRC}:${PYTHONPATH:-}"

python tools/moe_calibration/aic_moe_calibrate.py parse-trace \
  --trace-root "${OUT_DIR}/profile" \
  --output "${OUT_DIR}/parsed/sglang_aic_moe_events.csv"
```

解析器会把 trace 路径中的 `ntok_*`、`isl_*` 或 `input*` 推断成 `num_tokens`，并根据路径中的 `decode/generation` 推断 `phase=generation`，否则默认为 `phase=context`。如果同一个目录中同时有 prefill 和 decode，后续 summary、validation、breakdown 会按 `phase,num_tokens` 分开输出；普通 `compare` 默认只比较 `phase=context`。

如果是 server/curl 手工 profile，路径里没有 token 信息时不要和 sweep trace 混在一起解析。要么把 profile 目录命名成 `server_ntok_...`，要么单独解析并显式传 `--num-tokens` 和 `--phase`：

```bash
python tools/moe_calibration/aic_moe_calibrate.py parse-trace \
  --trace-root "${OUT_DIR}/profile/server" \
  --num-tokens 1 \
  --phase generation \
  --output "${OUT_DIR}/parsed/sglang_aic_moe_server_generation_events.csv"
```

先检查关键 profile 点是否齐全：

```bash
python tools/moe_calibration/aic_moe_calibrate.py validate-trace \
  --trace-csv "${OUT_DIR}/parsed/sglang_aic_moe_events.csv" \
  --expected-layers "${EXPECTED_MOE_LAYERS}" \
  --output "${OUT_DIR}/parsed/sglang_aic_moe_trace_validation.csv"
```

`sglang_aic_moe_trace_validation.csv` 中：

- `samples`：该 token/stage 的事件数。
- `observed_layers`：该 token/stage 实际覆盖的不同 layer 数。
- `expected_layers`：本轮期望出现 profile 的 MoE 层数，也就是 `EXPECTED_MOE_LAYERS`。
- `layer_ids`：实际出现的 layer id。若 required stage 的 `observed_layers < expected_layers`，状态会变成 `incomplete_layers`。

如果只验证普通单卡非 SBO 路径，可以把 `collector/moe` 放进 required stage：

```bash
python tools/moe_calibration/aic_moe_calibrate.py validate-trace \
  --trace-csv "${OUT_DIR}/parsed/sglang_aic_moe_events.csv" \
  --required-stage module router collector/moe topk shared_experts routed_experts routed/compute output_postprocess \
  --expected-layers "${EXPECTED_MOE_LAYERS}" \
  --output "${OUT_DIR}/parsed/sglang_aic_moe_trace_validation_strict.csv" \
  --fail-on-missing
```

生成实机 summary：

```bash
python tools/moe_calibration/aic_moe_calibrate.py summarize-trace \
  --trace-csv "${OUT_DIR}/parsed/sglang_aic_moe_events.csv" \
  --output "${OUT_DIR}/parsed/sglang_aic_moe_summary.csv"
```

只看关键 stage 的 summary：

```bash
python tools/moe_calibration/aic_moe_calibrate.py summarize-trace \
  --trace-csv "${OUT_DIR}/parsed/sglang_aic_moe_events.csv" \
  --stage module router collector/moe topk shared_experts routed_experts routed/dispatch routed/compute routed/combine routed/dispatch_a routed/dispatch_b routed/combine_a routed/combine_b routed/all_reduce output_postprocess output_all_reduce \
  --output "${OUT_DIR}/parsed/sglang_aic_moe_summary.csv"
```

检查原始 stage：

```bash
cut -d, -f5 "${OUT_DIR}/parsed/sglang_aic_moe_events.csv" | sort | uniq -c
```

合成 stage 也可以直接用于 summary/compare。例如：

```bash
python tools/moe_calibration/aic_moe_calibrate.py summarize-trace \
  --trace-csv "${OUT_DIR}/parsed/sglang_aic_moe_events.csv" \
  --stage routed/dispatch+routed/combine routed/dispatch_a+routed/dispatch_b+routed/combine_a+routed/combine_b \
  --output "${OUT_DIR}/parsed/sglang_aic_moe_deepep_comm_summary.csv"
```

生成普通单卡 MoE module 闭合表。默认口径避免重复计算：`collector/moe` 已经包含 `topk + routed_experts`，所以这里不再额外加 `topk` 或 `routed_experts`：

```bash
python tools/moe_calibration/aic_moe_calibrate.py breakdown-trace \
  --trace-csv "${OUT_DIR}/parsed/sglang_aic_moe_events.csv" \
  --component router collector/moe shared_experts output_postprocess output_all_reduce \
  --output "${OUT_DIR}/parsed/sglang_aic_moe_module_breakdown.csv" \
  --summary-output "${OUT_DIR}/parsed/sglang_aic_moe_module_breakdown_summary.csv"
```

如果是 DeepEP/WideEP 路径，`collector/moe` 通常不会出现，改用更细的组件：

```bash
python tools/moe_calibration/aic_moe_calibrate.py breakdown-trace \
  --trace-csv "${OUT_DIR}/parsed/sglang_aic_moe_events.csv" \
  --component router topk shared_experts routed_experts output_postprocess output_all_reduce \
  --output "${OUT_DIR}/parsed/sglang_aic_moe_deepep_module_breakdown.csv" \
  --summary-output "${OUT_DIR}/parsed/sglang_aic_moe_deepep_module_breakdown_summary.csv"
```

如果 trace 是 TBO/overlap 分段路径，需要看通信拆分项：

```bash
python tools/moe_calibration/aic_moe_calibrate.py breakdown-trace \
  --trace-csv "${OUT_DIR}/parsed/sglang_aic_moe_events.csv" \
  --component router topk shared_experts routed/dispatch_a routed/dispatch_b routed/compute routed/combine_a routed/combine_b output_postprocess output_all_reduce \
  --output "${OUT_DIR}/parsed/sglang_aic_moe_tbo_module_breakdown.csv" \
  --summary-output "${OUT_DIR}/parsed/sglang_aic_moe_tbo_module_breakdown_summary.csv"
```

注意：TBO 路径通过 `OperationsStrategy` 直接调度 `layer.mlp.op_gate/op_select_experts/op_dispatch_*/op_experts/op_combine_*/op_output`，不一定经过 `DeepseekV2MoE.forward` 外层的 `module` profile。TBO 校准应优先使用分段 stage 和 `dispatch_a+dispatch_b+combine_a+combine_b`，不要强行用普通 `module` 闭合残差判定 compute 表。

闭合表中的 `residual_us = module_duration_us - component_sum_us`。因为有 overlap 和 inclusive duration，`residual_us` 不一定必须接近 0；它的用途是定位还未解释的外围开销或重叠导致的负残差。`missing_components` 非空时，先确认该路径是否本来就不会产生对应 stage。

关键 stage 对齐关系：

- `module`：非 TBO 路径的完整 DeepSeek MoE forward 闭包，包含 router/topk/shared/routed/output_postprocess 和可能的 all-reduce。它用于检查模块闭合，不直接对齐普通 `moe_perf` 或 WideEP compute 单表。
- `topk+routed/compute`：普通 AIC `moe_perf` 的严格对齐边界，对应 collector timed loop 中的 `select_experts + fused_moe`。这是修普通 AIC MoE 表时的第一证据。
- `collector/moe`：SGLang routed wrapper 口径，包含 `topk + self.experts(...)`，不包含 router/shared experts/output_postprocess。普通校准必须通过 `--disable-shared-experts-fusion` 保证 shared experts 不被融合进该边界；该值可能比 `topk+routed/compute` 多出 dispatcher/combine wrapper 开销。
- `routed_experts`：SGLang `FusedMoE.forward` 整体，包含 dispatch/compute/combine。
- `routed/compute`：专家 GEMM/core compute，更适合用于拆解普通 MoE 误差，或对齐 WideEP `run_moe_core(...)` compute。
- `routed/dispatch` 与 `routed/combine`：多卡 DeepEP/WideEP 通信校准重点。
- `routed/dispatch_a`、`routed/dispatch_b`、`routed/combine_a`、`routed/combine_b`：DeepEP/TBO 分段执行路径的通信拆分事件；如果 trace 出现这些事件，用 `dispatch_a+dispatch_b` 视作 dispatch，用 `combine_a+combine_b` 视作 combine。
- `output_postprocess`：routed scaling、shared expert add 等 MoE 尾部 elementwise 开销。它不对齐 AIC 普通 `moe_perf`，主要用于解释 `module` 与主要子阶段求和之间的剩余量。

注意：某些执行路径里 `output_postprocess` 可能由多段同名事件组成，例如 routed scaling 和 shared-output add 分开落点。`summarize-trace` 会按事件样本统计，`samples` 可能大于 MoE 层数乘运行次数；需要看每层尾部总开销时，以 `breakdown-trace` 在 `module` 窗口内的组件求和为准。

组合 stage（例如 `topk+routed/compute`、`routed/dispatch+routed/combine`）会先按同一个 trace、同一个 layer、同一个 `pid/tid` 分组，再按 `ts` 排序和 occurrence 顺序配对：第 1 个 `topk` 加第 1 个 `routed/compute`，第 2 个加第 2 个。这样 server 或多请求 trace 不会把多个 forward 折叠成一个假样本，也尽量避免跨线程错配；如果某个组件缺少对应 occurrence，多出来的事件不会进入组合样本。

### 9. AIC collector：计划、smoke、正式采集、overlay

先确认 DeepSeekV3 MoE case plan：

```bash
cd "${AIC_SRC}"
export PYTHONPATH="${AIC_SRC}/src:${AIC_SRC}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=7
export COLLECTOR_LOG_DIR="${OUT_DIR}/aic/collector_plan"
mkdir -p "${COLLECTOR_LOG_DIR}"

python collector/collect.py \
  --backend sglang \
  --model-path "${MODEL_PATH}" \
  --ops moe \
  --sm 90 \
  --limit 16 \
  --plan-only \
  2>&1 | tee "${OUT_DIR}/logs/aic_plan_moe.log"
```

检查 plan 里是否包含 DeepSeekV3 MoE 形状：

```bash
grep -E "DeepseekV3|moe|num_tokens|hidden_size|inter_size|topk|num_experts" \
  "${OUT_DIR}/logs/aic_plan_moe.log" | head -200
```

先跑 smoke。这里保留 `--keep-csv`，因为后续 overlay 步骤会显式把 `_perf.txt` 转成 parquet，同时保留 txt 便于排查：

```bash
export COLLECTOR_LOG_DIR="${OUT_DIR}/aic/collector_moe_smoke"
mkdir -p "${COLLECTOR_LOG_DIR}"
python collector/collect.py \
  --backend sglang \
  --model-path "${MODEL_PATH}" \
  --ops moe \
  --sm 90 \
  --limit 4 \
  --keep-csv \
  2>&1 | tee "${OUT_DIR}/logs/aic_collect_moe_smoke.log"
```

确认 smoke 产物：

```bash
find "${OUT_DIR}/aic/collector_moe_smoke" -maxdepth 2 -type f | sort \
  | tee "${OUT_DIR}/logs/aic_collect_moe_smoke_files.txt"
grep -R "ERROR\\|Traceback\\|failed" -n "${OUT_DIR}/aic/collector_moe_smoke" "${OUT_DIR}/logs/aic_collect_moe_smoke.log" | head -100
```

正式采集。如果 smoke 通过，再去掉 `--limit` 跑完整点集：

```bash
export COLLECTOR_LOG_DIR="${OUT_DIR}/aic/collector_moe_full"
mkdir -p "${COLLECTOR_LOG_DIR}"
python collector/collect.py \
  --backend sglang \
  --model-path "${MODEL_PATH}" \
  --ops moe \
  --sm 90 \
  --keep-csv \
  2>&1 | tee "${OUT_DIR}/logs/aic_collect_moe_full.log"
```

把本轮 collector 输出接到 `PerfDatabase`。AIC 的 `PerfDatabase.query_moe` 默认读取 `${systems_root}/data/${system}/${backend}/${version}/*_perf.parquet`，而 collector 的 `--keep-csv` 输出是 `_perf.txt` staging 文件，所以必须执行这一步：

```bash
export AIC_COLLECTOR_DIR="${OUT_DIR}/aic/collector_moe_smoke"
export AIC_SYSTEMS_ROOT="${OUT_DIR}/aic_systems_overlay"
export AIC_SYSTEMS_TARGET="${AIC_SYSTEMS_ROOT}/data/h100_sxm/sglang/${BACKEND_VERSION}"
export AIC_DISPATCH_SYSTEMS_TARGET="${AIC_SYSTEMS_ROOT}/data/h100_sxm/sglang/${DEEPEP_DISPATCH_BACKEND_VERSION}"

case "${AIC_SYSTEMS_ROOT}" in
  "${OUT_DIR}"/*) ;;
  *) echo "Refusing to remove overlay outside OUT_DIR: ${AIC_SYSTEMS_ROOT}" >&2; exit 1 ;;
esac
rm -rf "${AIC_SYSTEMS_ROOT}"
mkdir -p "${AIC_SYSTEMS_ROOT}" "${AIC_SYSTEMS_TARGET}" "${AIC_DISPATCH_SYSTEMS_TARGET}"
cp -a "${AIC_SRC}/src/aiconfigurator/systems/." "${AIC_SYSTEMS_ROOT}/"

cd "${AIC_SRC}"
export PYTHONPATH="${AIC_SRC}/src:${AIC_SRC}:${PYTHONPATH:-}"
python - <<PY
from collector.helper import finalize_perf_outputs
finalize_perf_outputs("${AIC_COLLECTOR_DIR}", recursive=True, delete_source=False)
PY

mapfile -t MOE_FILES < <(find "${AIC_COLLECTOR_DIR}" -name "moe_perf.parquet" -print)
if (( ${#MOE_FILES[@]} == 0 )); then
  echo "No moe_perf.parquet found under ${AIC_COLLECTOR_DIR}" >&2
  exit 1
fi
cp "${MOE_FILES[0]}" "${AIC_SYSTEMS_TARGET}/moe_perf.parquet"

mapfile -t WIDEEP_CONTEXT_FILES < <(find "${AIC_COLLECTOR_DIR}" -name "wideep_context_moe_perf.parquet" -print)
if (( ${#WIDEEP_CONTEXT_FILES[@]} > 0 )); then
  cp "${WIDEEP_CONTEXT_FILES[0]}" "${AIC_SYSTEMS_TARGET}/wideep_context_moe_perf.parquet"
fi
mapfile -t WIDEEP_GENERATION_FILES < <(find "${AIC_COLLECTOR_DIR}" -name "wideep_generation_moe_perf.parquet" -print)
if (( ${#WIDEEP_GENERATION_FILES[@]} > 0 )); then
  cp "${WIDEEP_GENERATION_FILES[0]}" "${AIC_SYSTEMS_TARGET}/wideep_generation_moe_perf.parquet"
fi
mapfile -t DEEPEP_NORMAL_FILES < <(find "${AIC_COLLECTOR_DIR}" -name "wideep_deepep_normal_perf.parquet" -print)
if (( ${#DEEPEP_NORMAL_FILES[@]} > 0 )); then
  cp "${DEEPEP_NORMAL_FILES[0]}" "${AIC_DISPATCH_SYSTEMS_TARGET}/wideep_deepep_normal_perf.parquet"
fi
mapfile -t DEEPEP_LL_FILES < <(find "${AIC_COLLECTOR_DIR}" -name "wideep_deepep_ll_perf.parquet" -print)
if (( ${#DEEPEP_LL_FILES[@]} > 0 )); then
  cp "${DEEPEP_LL_FILES[0]}" "${AIC_DISPATCH_SYSTEMS_TARGET}/wideep_deepep_ll_perf.parquet"
fi

echo "${AIC_SYSTEMS_ROOT}" | tee "${OUT_DIR}/configs/aic_systems_root.txt"
find "${AIC_SYSTEMS_TARGET}" "${AIC_DISPATCH_SYSTEMS_TARGET}" -maxdepth 1 -type f -name "*moe*perf.parquet" -print \
  | sort -u | tee "${OUT_DIR}/configs/aic_overlay_files.txt"
```

如果要让 query 使用完整采集结果，把 `AIC_COLLECTOR_DIR` 改成：

```bash
export AIC_COLLECTOR_DIR="${OUT_DIR}/aic/collector_moe_full"
```

如果暂时不跑本轮 collector，也可以直接用仓库已有数据库：

```bash
export AIC_SYSTEMS_ROOT="${AIC_SRC}/src/aiconfigurator/systems"
```

WideEP/DeepEP 计划和 smoke：

```bash
python collector/collect.py \
  --backend sglang \
  --model-path "${MODEL_PATH}" \
  --ops wideep_moe \
  --sm 90 \
  --plan-only \
  2>&1 | tee "${OUT_DIR}/logs/aic_plan_wideep_moe.log"

export CUDA_VISIBLE_DEVICES=6,7
export COLLECTOR_LOG_DIR="${OUT_DIR}/aic/collector_wideep_moe_smoke"
mkdir -p "${COLLECTOR_LOG_DIR}"
python collector/collect.py \
  --backend sglang \
  --model-path "${MODEL_PATH}" \
  --ops wideep_moe \
  --sm 90 \
  --limit 4 \
  --keep-csv \
  2>&1 | tee "${OUT_DIR}/logs/aic_collect_wideep_moe_smoke.log"
```

### 10. AIC 查询与校准误差表

本节是校准核心，执行顺序固定为：

1. 确认 `AIC_SYSTEMS_ROOT` 指向本轮 overlay 或仓库已有 systems 目录。
2. 用 AIC `PerfDatabase.query_moe(...)` 对相同 token/shape/TP/EP/dtype/distribution 生成预测 CSV。
3. 用 trace summary 中的实机 `topk+routed/compute` 作为普通 AIC `moe_perf` 的严格对齐目标，同时保留 `collector/moe` 检查 SGLang routed wrapper 开销。
4. 用 `compare` 生成误差表。
5. 根据误差表判断应该修 collector 数据、插值策略，还是只修 AIC 模型后处理。

对齐关系不要混淆：

| 实机 trace stage | 对齐对象 | 用途 |
| --- | --- | --- |
| `topk+routed/compute` | `PerfDatabase.query_moe(..., moe_backend=None)` | 单卡普通 AIC `moe_perf` 严格对齐，第一优先级 |
| `collector/moe` | `PerfDatabase.query_moe(..., moe_backend=None)` | SGLang routed wrapper sanity check，可能包含 dispatcher/combine wrapper 开销 |
| `topk` | 普通 AIC `moe_perf` 中的 `select_experts(...)` 部分 | 拆解 topk 偏差 |
| `routed/compute` | 普通 AIC `moe_perf` 中的 `fused_moe(...)` 部分；也对齐 WideEP `run_moe_core(...)` | 专家 GEMM/core compute 偏差 |
| `routed_experts` | SGLang `FusedMoE.forward` 整体 wrapper | 检查 dispatcher/compute/combine 包装后的 routed path 偏差 |
| `module` | 非 TBO forward 闭包：router + topk + shared + routed + output_postprocess + all-reduce | 检查完整 DeepSeek MoE module 偏差，不直接对齐单个 AIC MoE 表 |
| `output_postprocess` | 无直接普通 `moe_perf` 对齐对象 | 解释 routed scaling/shared-add 等尾部 elementwise 开销 |
| `routed/dispatch` | WideEP/DeepEP dispatch 表或通信模型 | 多卡通信校准 |
| `routed/combine` | WideEP/DeepEP combine 表或通信模型 | 多卡通信校准 |
| `routed/dispatch_a + routed/dispatch_b` | DeepEP/TBO dispatch 分段 | overlap 路径通信校准 |
| `routed/combine_a + routed/combine_b` | DeepEP/TBO combine 分段 | overlap 路径通信校准 |

普通单卡 MoE compute 查询。建议先读本轮 overlay：

```bash
cd "${AIC_SRC}"
export PYTHONPATH="${AIC_SRC}/src:${AIC_SRC}:${PYTHONPATH:-}"
export AIC_SYSTEMS_ROOT="$(cat "${OUT_DIR}/configs/aic_systems_root.txt")"

python tools/moe_calibration/aic_moe_calibrate.py query-aic \
  --systems-root "${AIC_SYSTEMS_ROOT}" \
  --system h100_sxm \
  --backend sglang \
  --backend-version 0.5.9 \
  --database-mode SILICON \
  --num-tokens 128 512 2048 4096 96 160 384 768 1536 3072 6144 \
  --hidden-size "${HIDDEN_SIZE}" \
  --inter-size "${INTER_SIZE}" \
  --topk "${TOPK}" \
  --num-experts "${NUM_EXPERTS}" \
  --moe-tp-size "${MOE_TP_SIZE}" \
  --moe-ep-size "${MOE_EP_SIZE}" \
  --quant-mode "${QUANT_MODE}" \
  --distribution "${DISTRIBUTION}" \
  --phase context \
  --output "${OUT_DIR}/parsed/aic_moe_predictions.csv"
```

检查 AIC 预测来源和命中情况：

```bash
head -5 "${OUT_DIR}/parsed/aic_moe_predictions.csv"
cut -d, -f1,8,18,20 "${OUT_DIR}/parsed/aic_moe_predictions.csv" | column -s, -t
```

和实机 `topk+routed/compute` 做严格对比，同时输出 `collector/moe` wrapper sanity check：

```bash
python tools/moe_calibration/aic_moe_calibrate.py compare \
  --trace-csv "${OUT_DIR}/parsed/sglang_aic_moe_events.csv" \
  --aic-csv "${OUT_DIR}/parsed/aic_moe_predictions.csv" \
  --real-stage topk+routed/compute \
  --distribution "${DISTRIBUTION}" \
  --phase context \
  --output "${OUT_DIR}/parsed/aic_vs_sglang_topk_compute.csv"

python tools/moe_calibration/aic_moe_calibrate.py compare \
  --trace-csv "${OUT_DIR}/parsed/sglang_aic_moe_events.csv" \
  --aic-csv "${OUT_DIR}/parsed/aic_moe_predictions.csv" \
  --real-stage collector/moe \
  --distribution "${DISTRIBUTION}" \
  --phase context \
  --output "${OUT_DIR}/parsed/aic_vs_sglang_collector_moe.csv"
```

如果采了 decode/generation trace，单独查询 generation 表并显式按 `phase=generation` 对比。下面命令以 `batch-size=1` decode 为例，MoE token 数是 `1`：

```bash
python tools/moe_calibration/aic_moe_calibrate.py query-aic \
  --systems-root "${AIC_SYSTEMS_ROOT}" \
  --system h100_sxm \
  --backend sglang \
  --backend-version 0.5.9 \
  --database-mode SILICON \
  --num-tokens 1 \
  --hidden-size "${HIDDEN_SIZE}" \
  --inter-size "${INTER_SIZE}" \
  --topk "${TOPK}" \
  --num-experts "${NUM_EXPERTS}" \
  --moe-tp-size "${MOE_TP_SIZE}" \
  --moe-ep-size "${MOE_EP_SIZE}" \
  --quant-mode "${QUANT_MODE}" \
  --distribution "${DISTRIBUTION}" \
  --phase generation \
  --output "${OUT_DIR}/parsed/aic_moe_generation_predictions.csv"

python tools/moe_calibration/aic_moe_calibrate.py compare \
  --trace-csv "${OUT_DIR}/parsed/sglang_aic_moe_events.csv" \
  --aic-csv "${OUT_DIR}/parsed/aic_moe_generation_predictions.csv" \
  --real-stage topk+routed/compute \
  --distribution "${DISTRIBUTION}" \
  --phase generation \
  --output "${OUT_DIR}/parsed/aic_vs_sglang_generation_topk_compute.csv"

python tools/moe_calibration/aic_moe_calibrate.py compare \
  --trace-csv "${OUT_DIR}/parsed/sglang_aic_moe_events.csv" \
  --aic-csv "${OUT_DIR}/parsed/aic_moe_generation_predictions.csv" \
  --real-stage collector/moe \
  --distribution "${DISTRIBUTION}" \
  --phase generation \
  --output "${OUT_DIR}/parsed/aic_vs_sglang_generation_collector_moe.csv"
```

生成本轮校准总览报告：

```bash
python tools/moe_calibration/aic_moe_calibrate.py make-report \
  --compare-csv \
    "${OUT_DIR}/parsed/aic_vs_sglang_topk_compute.csv" \
    "${OUT_DIR}/parsed/aic_vs_sglang_collector_moe.csv" \
    "${OUT_DIR}/parsed/aic_vs_sglang_generation_topk_compute.csv" \
    "${OUT_DIR}/parsed/aic_vs_sglang_generation_collector_moe.csv" \
    "${OUT_DIR}/parsed/aic_vs_sglang_wideep_compute.csv" \
    "${OUT_DIR}/parsed/aic_vs_sglang_deepep_dispatch_combine.csv" \
    "${OUT_DIR}/parsed/aic_vs_sglang_deepep_tbo_dispatch_combine.csv" \
  --breakdown-summary-csv "${OUT_DIR}/parsed/sglang_aic_moe_module_breakdown_summary.csv" \
  --expert-distribution-csv "${OUT_DIR}/parsed/sglang_expert_distribution_summary.csv" \
  --stage-map-csv "${OUT_DIR}/configs/stage_alignment_map.csv" \
  --output "${OUT_DIR}/parsed/deepseekv3_moe_calibration_report.md"
```

如果本轮没有通过 server 路径采集 expert distribution，报告会跳过 expert distribution 部分，只保留 AIC 误差和 module 闭合分析。

如果要拆解普通 MoE 误差，不要直接拿普通 `aic_moe_predictions.csv` 和 `routed_experts` 做正式误差表；普通 AIC `moe_perf` 的口径是 `select_experts + fused_moe`，正式对齐目标是 `topk+routed/compute`。这里应生成拆解 summary，再用 `topk`、`routed/compute`、`collector/moe`、`routed_experts` 的相对大小判断误差来自 select_experts、专家核心还是 SGLang wrapper：

```bash
python tools/moe_calibration/aic_moe_calibrate.py summarize-trace \
  --trace-csv "${OUT_DIR}/parsed/sglang_aic_moe_events.csv" \
  --stage topk+routed/compute collector/moe topk routed_experts routed/compute \
  --output "${OUT_DIR}/parsed/sglang_aic_moe_collector_breakdown_summary.csv"
```

解释误差表：

- `real_mean_us`：实机 trace 中同 token、同 stage、所有 MoE 层的平均时延。
- `aic_latency_us`：AIC `PerfDatabase.query_moe` 返回值。
- `abs_error_us = aic_latency_us - real_mean_us`。
- `rel_error_pct > 0` 表示 AIC 预测偏慢，`rel_error_pct < 0` 表示 AIC 预测偏快。
- 第一轮最重要的是 `topk+routed/compute`，因为 AIC 普通 SGLang `moe` collector 的 timed region 包含 `select_experts + fused_moe`，但不包含 DeepSeek 外层 router/shared/module overhead，也不包含 SGLang routed wrapper 的 dispatcher/combine 包装开销。

建议判定阈值：

- `|rel_error_pct| <= 10%`：该 token 点先认为可接受，继续看更多 token。
- `10% < |rel_error_pct| <= 25%`：标记为中等偏差，优先检查 collector 命中点和 dtype/backend 是否一致。
- `|rel_error_pct| > 25%`：进入误差定位，按下面顺序排查。

误差定位命令：

```bash
column -s, -t < "${OUT_DIR}/parsed/aic_vs_sglang_topk_compute.csv" | less -S
column -s, -t < "${OUT_DIR}/parsed/aic_vs_sglang_collector_moe.csv" | less -S
column -s, -t < "${OUT_DIR}/parsed/sglang_aic_moe_summary.csv" | less -S
grep -R "kernel_source\\|distribution\\|fp8\\|bfloat16" -n "${OUT_DIR}/aic" "${OUT_DIR}/logs" | head -200
```

排查顺序：

1. 先看 `sglang_aic_moe_trace_validation.csv`，确认关键 stage 没有缺失；普通单卡非 SBO 路径尤其要确认 `collector/moe` 存在。
2. 再确认 `aic_moe_predictions.csv` 中 `quant_mode/distribution/moe_tp_size/moe_ep_size` 与 SGLang 命令一致。
3. 再看 `sglang_aic_moe_summary.csv` 里同一 token 的 `topk`、`routed/compute`、`collector/moe` 样本数是否等于 MoE 层数乘运行次数。样本明显偏少说明 trace 没覆盖完整 forward 或进入了 piecewise/TBO 路径。
4. 如果 `topk+routed/compute` 不准，先拆成 `topk` 和 `routed/compute` 看偏差来自 select_experts 还是专家核心；如果只有 `collector/moe` 偏慢，则优先看 dispatcher/combine wrapper。
5. 如果 `routed/compute` 本身不准，优先查 collector 的 `moe_perf` 是否直接命中该 token；插值点偏差大则看相邻 token 的 collector 数据。
6. 如果 `module` 不准但 `topk+routed/compute` 准，问题在 router/shared experts/output_postprocess/all-reduce 或 wrapper/overlap，不应直接修改 MoE compute 表。

WideEP 查询需要切换到 `moe_backend=deepep_moe`：

```bash
python tools/moe_calibration/aic_moe_calibrate.py query-aic \
  --systems-root "${AIC_SYSTEMS_ROOT}" \
  --system h100_sxm \
  --backend sglang \
  --backend-version 0.5.9 \
  --database-mode SILICON \
  --num-tokens 512 2048 4096 \
  --hidden-size "${HIDDEN_SIZE}" \
  --inter-size "${INTER_SIZE}" \
  --topk "${TOPK}" \
  --num-experts "${NUM_EXPERTS}" \
  --moe-tp-size "${MOE_TP_SIZE}" \
  --moe-ep-size "${MOE_EP_SIZE}" \
  --quant-mode "${QUANT_MODE}" \
  --distribution "${WIDEEP_DISTRIBUTION}" \
  --phase context \
  --moe-backend deepep_moe \
  --output "${OUT_DIR}/parsed/aic_wideep_compute_predictions.csv"
```

这张表来自 AIC 的 `wideep_context_moe_perf` / `wideep_generation_moe_perf`，对齐实机 trace 的 `routed/compute`，不是 dispatch/combine 通信：

```bash
python tools/moe_calibration/aic_moe_calibrate.py compare \
  --trace-csv "${OUT_DIR}/parsed/sglang_aic_moe_events.csv" \
  --aic-csv "${OUT_DIR}/parsed/aic_wideep_compute_predictions.csv" \
  --real-stage routed/compute \
  --distribution "${WIDEEP_DISTRIBUTION}" \
  --phase context \
  --output "${OUT_DIR}/parsed/aic_vs_sglang_wideep_compute.csv"
```

DeepEP dispatch+combine 通信表单独查询。AIC loader 中 `wideep_deepep_normal_perf` 的 latency 是 `dispatch_transmit + dispatch_notify + combine_transmit + combine_notify`，所以实机侧也要用 `dispatch+combine` 合成 stage 对齐。

注意：当前仓库中 H100/SGLang DeepEP dispatch 表可能只有 `0.5.6.post2`，未必有 `0.5.9`。如果 `0.5.9` 查询报缺文件，有两种选择：

- 正式校准：先采集或导入 `0.5.9` 的 `wideep_deepep_normal_perf.parquet` / `wideep_deepep_ll_perf.parquet`。
- 临时参考：显式设置 `DEEPEP_DISPATCH_BACKEND_VERSION=0.5.6.post2`，并在结果中标记通信表不是同版本数据。

查询命令：

```bash
export DEEPEP_DISPATCH_BACKEND_VERSION="${DEEPEP_DISPATCH_BACKEND_VERSION:-0.5.9}"
python tools/moe_calibration/aic_moe_calibrate.py query-deepep-dispatch \
  --systems-root "${AIC_SYSTEMS_ROOT}" \
  --system h100_sxm \
  --backend sglang \
  --backend-version "${DEEPEP_DISPATCH_BACKEND_VERSION}" \
  --database-mode SILICON \
  --num-tokens 512 2048 4096 \
  --hidden-size "${HIDDEN_SIZE}" \
  --topk "${TOPK}" \
  --num-experts "${NUM_EXPERTS}" \
  --node-num "${DEEPEP_NODE_NUM}" \
  --sms "${DEEPEP_SMS}" \
  --deepep-mode normal \
  --distribution "${WIDEEP_DISTRIBUTION}" \
  --output "${OUT_DIR}/parsed/aic_deepep_dispatch_predictions.csv"
```

非 TBO/非分段 trace 对比：

```bash
python tools/moe_calibration/aic_moe_calibrate.py compare \
  --trace-csv "${OUT_DIR}/parsed/sglang_aic_moe_events.csv" \
  --aic-csv "${OUT_DIR}/parsed/aic_deepep_dispatch_predictions.csv" \
  --real-stage routed/dispatch+routed/combine \
  --distribution "${WIDEEP_DISTRIBUTION}" \
  --phase context \
  --output "${OUT_DIR}/parsed/aic_vs_sglang_deepep_dispatch_combine.csv"
```

TBO/overlap 分段 trace 对比：

```bash
python tools/moe_calibration/aic_moe_calibrate.py compare \
  --trace-csv "${OUT_DIR}/parsed/sglang_aic_moe_events.csv" \
  --aic-csv "${OUT_DIR}/parsed/aic_deepep_dispatch_predictions.csv" \
  --real-stage routed/dispatch_a+routed/dispatch_b+routed/combine_a+routed/combine_b \
  --distribution "${WIDEEP_DISTRIBUTION}" \
  --phase context \
  --output "${OUT_DIR}/parsed/aic_vs_sglang_deepep_tbo_dispatch_combine.csv"
```

### 11. 两卡 SGLang trace

容器启动时暴露 `6,7`，容器内：

```bash
export CUDA_VISIBLE_DEVICES=6,7
export TP_SIZE=2
export EP_SIZE=2
export SGLANG_AIC_MOE_PROFILE=1
export SGLANG_TORCH_PROFILER_DIR="${OUT_DIR}/profile/tp2_ep2_prefill_isl_512"
mkdir -p "${SGLANG_TORCH_PROFILER_DIR}"
cd "${SGLANG_SRC}/python"
export PYTHONPATH="${SGLANG_SRC}/python:${PYTHONPATH:-}"

python -m sglang.bench_one_batch \
  --model-path "${MODEL_PATH}" \
  --load-format dummy \
  --tp-size 2 \
  --ep-size 2 \
  --json-model-override-args "{\"num_hidden_layers\":${NUM_HIDDEN_LAYERS},\"first_k_dense_replace\":${FIRST_K_DENSE_REPLACE}}" \
  --disable-cuda-graph \
  --disable-shared-experts-fusion \
  --batch-size 1 \
  --input-len 512 \
  --output-len 1 \
  --profile \
  --profile-stage prefill \
  --profile-filename-prefix dsv3_moe_tp2_ep2_prefill_isl_512 \
  2>&1 | tee "${OUT_DIR}/logs/sglang_tp2_ep2_prefill_isl_512.log"
```

两卡对齐时重点看：

- `routed/compute`：专家计算是否仍然接近 AIC compute。
- `routed/dispatch`、`routed/combine`：通信项是否是主要误差来源。
- `routed_experts - routed/compute`：粗略表示 dispatch/combine/permute 等非 compute 开销。

### 12. 归档

```bash
tar -czf "${OUT_DIR}.tar.gz" -C "$(dirname "${OUT_DIR}")" "$(basename "${OUT_DIR}")"
echo "${OUT_DIR}.tar.gz"
```

必须保留：

- `${OUT_DIR}/configs/*`
- `${OUT_DIR}/profile/**/*.trace.json*`
- `${OUT_DIR}/parsed/*.csv`
- `${OUT_DIR}/aic/**`
- `${OUT_DIR}/logs/**`

## 风险

- 普通 `moe_perf` 校准必须关闭 `--disable-shared-experts-fusion`；如果为了研究默认融合路径而打开 fusion，该 trace 只能用于模块拆解或单独建模，不能直接和普通 AIC `moe_perf` 表比较。
- cudagraph 可能让 Python `record_function` 边界不可见，第一轮先关闭。
- 完整推理的真实路由分布不一定等于 collector 的合成 balanced/power-law 分布，需要记录实机 topk 分布后再做严格对齐。
- 多卡场景下 `dispatch/combine` 受网络、拓扑、NCCL/DeepEP 配置影响，不能用单卡误差直接外推。
