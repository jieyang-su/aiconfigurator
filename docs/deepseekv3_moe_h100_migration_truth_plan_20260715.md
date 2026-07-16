# DeepSeek-V3 MoE H100 迁移与实机 Truth 计划

本文档记录如何把 H20 上已经收敛的 DeepSeek-V3 MoE / WideEP AIC
collector 方案迁移到 H100，并用 H100 实机 truth 验证硬件可迁移性。

核心目标不是为 H100 重新做一套特化校准，而是证明：H20 上收敛后的
AIC collector 逻辑在 H100 上也能复用。默认不做 H100 专属 policy。

## 环境信息

- SSH host: `10.110.183.26`
- SSH user: `ai_lab`
- 镜像名: `Ai-configurator`
- 所有 collector、benchmark、实机 truth 命令都必须进入镜像执行。
- 不要把密码写入仓库文件、脚本或文档。
- DeepSeek-V3 model config/cache 在 H100 上已经存在，不需要迁移。

进入镜像：

```bash
ssh ai_lab@10.110.183.26
docker exec -it Ai-configurator bash
```

先只做环境确认，不要直接改远端文件或系统数据。

当前 H100 环境探测结果：

- 宿主侧没有 `/cold/tair-kvcache`。
- `Ai-configurator` 容器内存在 `/cold/tair-kvcache`，当前内容是
  tair-kvcache 根目录，但还没有 `aiconfigurator/` 子目录。
- `Ai-configurator` 容器内当前 SGLang 代码路径是 `/sgl-workspace/sglang`。
- 当前可见 GPU 为 2 张 H100。

因此迁移目标路径暂定为：

```text
/cold/tair-kvcache/aiconfigurator/    # 新增/覆盖 AIC 相关代码
/sgl-workspace/sglang/                # 覆盖 H20 镜像中的 SGLang 代码并重新安装
```

## 需要迁移的内容

需要迁移代码、脚本和数据：

```text
collector/
tools/moe_calibration/
src/aiconfigurator/
sglang/
ShareGPT 数据
LongBench 数据
```

ShareGPT 和 LongBench 尽量放到 H20 镜像里相同的路径，减少脚本改动。
DeepSeek-V3 model config/cache 不迁移。

其中 `sglang/` 也需要从 H20 镜像里的当前代码迁移到 H100 镜像并覆盖安装。
这是为了保证实机 truth 测算时使用相同的 SGLang profiling、DeepEP/WideEP
路径和辅助改动。覆盖与安装也必须在 `Ai-configurator` 镜像内完成。

建议迁移后在 H100 镜像内执行类似流程：

```bash
cd /sgl-workspace/sglang
python3 -m pip install -e python
```

安装后做轻量确认：

```bash
python3 - <<'PY'
import sglang
print(sglang.__file__)
PY
```

如果 H100 镜像里已有运行中的 SGLang server 或 collector 进程，先停止再覆盖
`sglang/`，避免半新半旧代码混用。

## AIC 与实机的范围划分

AIC collector 是单卡/profile-free 模拟，不受 H100 两卡实机限制，应保持全量：

- ordinary MoE: EP1/2/4/8/16/32
- WideEP MoE: EP2/4/8

H100 实机 truth 当前只有两张卡，因此最多测到 EP2：

- ordinary MoE truth: EP1/2
- WideEP context truth: EP2
- WideEP generation low_latency truth: EP2
- 不测 EP4/EP8 实机 truth

AIC 的 EP4/8/16/32 输出可以保留为系统数据候选，但不能纳入本轮
H100 实机误差证据。第一轮迁移性证明只看有实机 truth 的 EP<=2。

## Step 1: H100 上跑 AIC Collector

进入 `Ai-configurator` 后执行：

```bash
cd /cold/tair-kvcache/aiconfigurator/collector

COLLECTOR_DSV3_KEEP_LATENCY_SOURCES=1 \
python3 collect.py \
  --backend sglang \
  --model-path deepseek-ai/DeepSeek-V3 \
  --ops moe_token_distribution wideep_moe \
  --keep-csv
```

本轮 H100 已有对应的 `0.5.9-dsv3` ordinary MoE 数据，且 H20 侧本次主要修改
集中在 WideEP clean-latency 逻辑；因此普通 `moe` 暂不重测、不覆盖系统数据。

正式 AIC 输出只使用 run 目录顶层 compact 文件中的 WideEP 相关表：

```text
moe_token_distribution_perf.txt
wideep_context_moe_perf.txt
wideep_generation_moe_perf.txt
```

`aic_latency_source_bundle/final_full_header` 只作为诊断文件，不作为正式系统数据。

## Step 2: H100 WideEP Context 实机 Truth

所有命令进入 `Ai-configurator` 执行。

配置口径：

- DeepEP mode: `normal`
- phase: `context`
- EP: 2
- EPLB: off/on
- dataset: ShareGPT 为主，LongBench 做交叉验证
- metric: `rank_max_us`
- session 数: 3
- 每个 session 有效 repetition: 3
- warmup: 3 次，不进入统计
- truth: 每个点先取 session 内 median，再取跨 session median

tokens:

```text
8,16,32,64,128,512,640,1536,2048,2560,4096,5120,8192,10240,12288,14336,16384,18888
```

`18888` 作为右外推验证点，保留在诊断里，但第一轮不要让它主导是否加
policy。

## Step 3: H100 WideEP Generation Low-Latency 实机 Truth

所有命令进入 `Ai-configurator` 执行。

配置口径：

- DeepEP mode: `low_latency`
- phase: `generation`
- EP: 2
- EPLB: off/on
- dataset: ShareGPT 为主，LongBench 做交叉验证
- metric: `rank_max_us`
- session 数: 3
- 每个 session 有效 repetition: 5
- warmup: 3 次，不进入统计
- truth: 每个点先取 session 内 median，再取跨 session median

tokens:

```text
4,8,32,40,64,128,288,512,896,1024,1280
```

`token=4` 是左外推敏感点。需要测出来，但第一轮 evidence 里单独标注，
不要因为这个点直接加 H100 特化 policy。

## Step 4: H100 Ordinary MoE 实机 Truth

当前 H100 只有两卡，因此 ordinary MoE 实机 truth 只测 EP1/2：

- EP: 1, 2
- `moe_tp_size=1`
- `moe_ep_size=EP`
- context 和 generation 分开
- 使用 phase-qualified recorded distributions
- metric: `rank_max_us`
- context: 3 sessions x 3 effective repetitions
- generation: 如果抖动明显，用 3 sessions x 5 effective repetitions

注意：AIC ordinary MoE 仍然可以生成 EP1/2/4/8/16/32；只是实机 truth
只验证 EP1/2。

## Step 5: 生成 H100 Frozen Truth

H100 truth 单独沉淀到新目录：

```text
results/h100_frozen_truth_<date>/
  h100_frozen_truth_points_<date>.csv
  h100_frozen_truth_summary_<date>.csv
  README.md
```

至少保留字段：

```text
family,dataset,phase,moe_ep_size,eplb,distribution,num_tokens,
truth_metric,truth_us,truth_policy,server_samples,
session_medians_us,low_platform_us,high_platform_us,source_file,notes
```

`truth_policy` 需要写清楚：

- ordinary MoE: `rank_max_us`，EP1/2，两卡实机可测范围内
- WideEP context: DeepEP normal，EP2，3 sessions x 3 reps，warmup 后 median
- WideEP generation low_latency: EP2，3 sessions x 5 reps，warmup 后 median

## Step 6: 对比与迁移性验证

对比对象：

- ordinary MoE: 使用 H100 现有 `0.5.9-dsv3/moe_perf.txt`
- WideEP: 使用本轮 H100 AIC collector 顶层 compact 文件
- H100 frozen truth

第一轮正式 evidence 只看实机有 truth 的点：

```text
ordinary_moe / sharegpt / context,generation / EP1,2
wideep_context / sharegpt,longbench / EP2
wideep_generation_low_latency / sharegpt,longbench / EP2
```

解释原则：

- ShareGPT 是主目标。
- LongBench 是交叉验证。
- AIC 的 EP4/8/16/32 行不纳入本轮实机误差结论。
- 如果 EP<=2 的误差与 H20 同级，或能被实机 truth 抖动解释，则认为
  H20 collector policy 具备硬件可迁移性。

## Step 7: H100 是否需要特化校准

默认不做 H100 特化校准。

只有同时满足以下条件，才考虑改 policy：

- ShareGPT EP<=2 出现系统性大误差。
- LongBench 也出现同方向误差。
- 已排除 warmup、session 抖动、DeepEP mode、EPLB 配置、token 口径、
  compare 插值口径等问题。
- AIC 诊断字段显示与 H20 相同机制，但确实存在跨硬件系数偏移。

即使需要修改，也优先做基于 AIC 诊断字段的硬件泛化 thin policy。
不要写 H100 truth anchor，也不要做点位硬编码。

## Step 8: H100 系统数据安装

只有迁移性验证通过后，才考虑安装 H100 系统数据：

```text
src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.9-dsv3/
```

安装来源是 H100 AIC collector 顶层 compact 文件：

```text
moe_perf.txt
moe_token_distribution_perf.txt
wideep_context_moe_perf.txt
wideep_generation_moe_perf.txt
```

Evidence 单独保留：

```text
results/h100_frozen_truth_<date>/h100_dsv3_calibration_evidence_<run_id>/
```

最终结论应明确区分：

- EP<=2: 有 H100 实机 truth，可作为硬件迁移性证据
- EP4/8/16/32: AIC 可生成，但本轮没有 H100 实机 truth，不作为误差证据

## 实操记录

### 2026-07-15 Step A: H100 只读环境探测

已完成，只读探测，没有修改远端文件。

本地执行方式：

```bash
ssh ai_lab@10.110.183.26
docker ps
docker exec Ai-configurator bash -lc 'nvidia-smi -L; python3 -c "import torch; print(torch.cuda.device_count())"'
```

探测结果：

- SSH 可连，远端 host 为 `pod-hpc-04`。
- `Ai-configurator` 容器存在，镜像为 `booleimg.myaddr.io/lmsysorg/sglang:v0.5.9`。
- 容器内可见 2 张 H100。
- 容器内当前工作目录是 `/sgl-workspace/sglang`。
- 容器内存在 `/cold/tair-kvcache`，但还没有 `/cold/tair-kvcache/aiconfigurator/`。
- 宿主侧没有 `/cold/tair-kvcache`。

### 2026-07-15 Step B: 本地打包并上传到远端临时目录

已完成。只上传到远端宿主临时目录，没有解压、没有覆盖容器路径、没有安装。

本地临时目录：

```text
/tmp/aic_h100_migration_20260715_105311/
```

远端宿主临时目录：

```text
/home/ai_lab/aic_h100_migration_20260715_105311/
```

本地打包命令口径：

```bash
tar \
  --exclude='aiconfigurator/collector/*_2026*' \
  --exclude='aiconfigurator/collector/*+*' \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  --exclude='.git' \
  -czf /tmp/aic_h100_migration_20260715_105311/aiconfigurator_code_20260715_105311.tar.gz \
  aiconfigurator/collector \
  aiconfigurator/tools/moe_calibration \
  aiconfigurator/src/aiconfigurator \
  aiconfigurator/docs/deepseekv3_moe_h100_migration_truth_plan_20260715.md

tar \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  --exclude='.git' \
  -czf /tmp/aic_h100_migration_20260715_105311/sglang_code_20260715_105311.tar.gz \
  sglang
```

归档文件：

```text
aiconfigurator_code_20260715_105311.tar.gz  89M
sglang_code_20260715_105311.tar.gz          9.3M
```

远端校验：

```text
befadf5780928527d5344d9787c3b5d8def73a1f03efa243ae4e7d9d0f143431  aiconfigurator_code_20260715_105311.tar.gz
306ad086c9ffc62387583a797ef4ac0fcdc345ac522872d622d507a2eff041e9  sglang_code_20260715_105311.tar.gz
```

下一步建议先进入容器创建备份目录并解压到临时 staging 目录，仍然不覆盖正式路径；
确认 staging 内容无误后，再由用户 review 是否覆盖 `/cold/tair-kvcache/aiconfigurator/`
和 `/sgl-workspace/sglang/`。

### 2026-07-15 Step C: 覆盖 H100 容器代码并安装

已完成。覆盖目标均在 `Ai-configurator` 容器内：

```text
/cold/tair-kvcache/aiconfigurator/
/sgl-workspace/sglang/
```

执行前检查了容器内没有运行中的 collector / SGLang server / WideEP benchmark 进程。

归档复制进容器：

```bash
docker exec Ai-configurator bash -lc 'mkdir -p /tmp/aic_h100_migration_20260715_105311'
docker cp /home/ai_lab/aic_h100_migration_20260715_105311/aiconfigurator_code_20260715_105311.tar.gz Ai-configurator:/tmp/aic_h100_migration_20260715_105311/
docker cp /home/ai_lab/aic_h100_migration_20260715_105311/sglang_code_20260715_105311.tar.gz Ai-configurator:/tmp/aic_h100_migration_20260715_105311/
```

容器内校验：

```text
befadf5780928527d5344d9787c3b5d8def73a1f03efa243ae4e7d9d0f143431  aiconfigurator_code_20260715_105311.tar.gz
306ad086c9ffc62387583a797ef4ac0fcdc345ac522872d622d507a2eff041e9  sglang_code_20260715_105311.tar.gz
```

覆盖解压：

```bash
tar -xzf /tmp/aic_h100_migration_20260715_105311/aiconfigurator_code_20260715_105311.tar.gz -C /cold/tair-kvcache
tar -xzf /tmp/aic_h100_migration_20260715_105311/sglang_code_20260715_105311.tar.gz -C /sgl-workspace
```

SGLang 安装过程：

- `python3 -m pip install -e /sgl-workspace/sglang/python` 失败，因为 build
  isolation 尝试访问默认 PyPI 拉取 `setuptools>=61.0`，远端 DNS 不可用。
- `python3 -m pip install --no-build-isolation -e python -i
  https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple` 长时间无输出，已中断。
- 最终使用镜像已有依赖，执行成功：

```bash
cd /sgl-workspace/sglang
python3 -m pip install --no-deps --no-build-isolation -e python \
  -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

因为第一次 AIC 代码包没有包含根目录 `pyproject.toml`，补传了 metadata 包：

```text
aiconfigurator_metadata_20260715_105311.tar.gz
d52f0206bf884815e458f8490c2097bbae315448e23d55e466bc6109526955cf
```

metadata 内容：

```text
pyproject.toml
README.md
LICENSE
```

AIC 安装命令：

```bash
cd /cold/tair-kvcache/aiconfigurator
python3 -m pip install --no-deps --no-build-isolation -e . \
  -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

安装后验证：

```text
aiconfigurator_file= /cold/tair-kvcache/aiconfigurator/src/aiconfigurator/__init__.py
sglang_file= /sgl-workspace/sglang/python/sglang/__init__.py
```

下一步建议先做轻量 smoke：

```bash
cd /cold/tair-kvcache/aiconfigurator/collector
python3 collect.py --backend sglang --model-path deepseek-ai/DeepSeek-V3 --ops moe_token_distribution wideep_moe --smoke --keep-csv
```

如果 smoke 通过，再跑正式 AIC collector。正式全量本轮只跑：

```bash
cd /cold/tair-kvcache/aiconfigurator/collector
COLLECTOR_DSV3_KEEP_LATENCY_SOURCES=1 \
python3 collect.py --backend sglang --model-path deepseek-ai/DeepSeek-V3 \
  --ops moe_token_distribution wideep_moe --keep-csv
```

后续与实机 truth 对比时：

- ordinary MoE 使用 H100 原有 `0.5.9-dsv3/moe_perf.txt`
- WideEP context / generation 使用本轮新 collector run 的顶层 compact 文件

### 2026-07-15 Step D: H100 collector smoke

已完成。命令在 `Ai-configurator` 容器内执行：

```bash
cd /cold/tair-kvcache/aiconfigurator/collector
python3 collect.py --backend sglang --model-path deepseek-ai/DeepSeek-V3 \
  --ops moe_token_distribution wideep_moe --smoke --keep-csv
```

输出目录：

```text
/cold/tair-kvcache/aiconfigurator/collector/moe_token_distribution+wideep_moe_20260715_032301
```

结果：

```text
Total errors: 0
```

说明：

- `moe_token_distribution` smoke 通过。
- `wideep_moe` smoke 通过；H100 上 WideEP smoke case 较慢，4 个 case
  总计约 12 分钟。
- clean-latency 在 `--smoke` 下跳过是预期行为，因为 sampled WideEP source
  不完整。

### 2026-07-15 Step E: H100 AIC WideEP 正式全量测算

开始时间：2026-07-15 11:37 CST。

执行前再次确认 `Ai-configurator` 容器内没有残留进程：

```bash
ps -ef | grep -E 'collect.py|run_moe_benchmark' | grep -v grep || true
```

正式命令在 `Ai-configurator` 容器内执行，只跑 WideEP 相关 collector：

```bash
cd /cold/tair-kvcache/aiconfigurator/collector
COLLECTOR_DSV3_KEEP_LATENCY_SOURCES=1 \
python3 collect.py --backend sglang --model-path deepseek-ai/DeepSeek-V3 \
  --ops moe_token_distribution wideep_moe --keep-csv
```

本轮不跑 ordinary MoE；后续对比时 ordinary MoE 使用 H100 现有
`0.5.9-dsv3/moe_perf.txt`，WideEP 使用本轮生成的顶层
`wideep_context_moe_perf.txt` / `wideep_generation_moe_perf.txt`。

完成时间：2026-07-15 11:47 CST。

输出目录：

```text
/cold/tair-kvcache/aiconfigurator/collector/moe_token_distribution+wideep_moe_20260715_033808
```

顶层 compact 文件：

```text
moe_token_distribution_perf.txt  485 行
wideep_context_moe_perf.txt      243 行
wideep_generation_moe_perf.txt   181 行
```

结果：

```text
Total errors: 0
```

本地只拉回了 AIC 输出副本用于查看：

```text
aiconfigurator/results/h100_dsv3_wideep_aic_20260715_033808/
```

### 2026-07-15 Step F: H100 实机 truth 容器与数据确认

H100 原始 `Ai-configurator` 容器是在 `/home/ai_lab/fjw/models` 成为
model 挂载点之前启动的。虽然 `docker inspect` 里有
`/home/ai_lab/fjw/models:/model`，但容器内 `/model` 实际看不到
DeepSeek-V3 权重，只能看到早先复制进去的 ShareGPT。

为了不停止、不删除、不重建原始 `Ai-configurator`，本轮实机 truth 单独创建
一个新容器：

```bash
docker run --name=Ai-configurator-truth \
  --gpus '"device=5,7"' \
  --cpuset-cpus="48-95,144-191" \
  --cpuset-mems="1" \
  --ipc=host \
  --network=host \
  -itd \
  -v /home/cold:/cold \
  -v /home/ai_lab/fjw/models:/model \
  booleimg.myaddr.io/lmsysorg/sglang:v0.5.9
```

LongBench 数据从 H20 `/model/data` 迁移到了 H100 model 挂载目录：

```text
/home/ai_lab/fjw/models/data/longbench_sharegpt_format_for_moe_profile.json
sha256: 120b8071e8eb7c9db7384811673ae5289cf02ed3ab0c58d5c8602c6c4c88743c
```

`Ai-configurator-truth` 内已确认可见：

```text
/model/DeepSeek-V3/config.json
/model/DeepSeek-V3/model-00001-of-000163.safetensors
/model/ShareGPT_V3_unfiltered_cleaned_split.json
/model/data/longbench_sharegpt_format_for_moe_profile.json
```

`Ai-configurator-truth` 内安装口径：

```bash
cd /sgl-workspace/sglang
python3 -m pip install --no-deps --no-build-isolation -e python \
  -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

cd /cold/tair-kvcache/aiconfigurator
python3 -m pip install --no-deps --no-build-isolation -e . \
  -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

安装后确认：

```text
sglang_file=/sgl-workspace/sglang/python/sglang/__init__.py
aiconfigurator_file=/cold/tair-kvcache/aiconfigurator/src/aiconfigurator/__init__.py
```

H100 网卡和 IB 设备名与 H20 不同。H20 脚本里的 `bond0` /
`mlx5_bond_*` 不应直接照搬到 H100。H100 truth smoke 先使用不显式指定
HCA 的最小环境，确认 DeepEP normal / low_latency 是否能在当前两卡上正常跑。

### 2026-07-15 Step G: H100 实机 truth smoke

目标：先只跑 ShareGPT、EP2、小 token、单 session，确认完整链路可以启动、
profile、parse。

Context smoke 口径：

```text
container: Ai-configurator-truth
dataset: ShareGPT
phase: context
DeepEP mode: normal
EP/TP: 2/2
tokens: 8,128
EPLB: off
session/repetition: 1 session, 1 effective repetition, 1 warmup
metric: rank_max_us
```

Generation low_latency smoke 口径：

```text
container: Ai-configurator-truth
dataset: ShareGPT
phase: generation
DeepEP mode: low_latency
EP/TP: 2/2
tokens: 8,128
EPLB: off/on
session/repetition: 1 session, 1 effective repetition, 1 warmup
metric: rank_max_us
```

smoke 通过后再进入正式 H100 frozen truth：

1. ShareGPT context/generation EP2 全 token，3 sessions。
2. LongBench context/generation EP2 全 token，3 sessions。
3. 生成 H100 frozen truth CSV，并在 H100 上直接与 AIC compact 输出对比。

### 2026-07-15 Step H: H100 smoke 实际执行记录

执行前确认：

```text
sglang=/sgl-workspace/sglang/python/sglang/__init__.py
aiconfigurator=/cold/tair-kvcache/aiconfigurator/src/aiconfigurator/__init__.py
/model/DeepSeek-V3/config.json 可见
/model/ShareGPT_V3_unfiltered_cleaned_split.json 可见
/model/data/longbench_sharegpt_format_for_moe_profile.json 可见
两张 H100 均为空闲，显存占用 4 MiB，无残留 profile/server 进程
```

第一次 context smoke 使用默认 server warmup，server 能完成权重加载和
DeepEP normal 初始化，但 SGLang 内置 server warmup 在 DeepGEMM grouped GEMM
JIT 阶段超过 600s，触发：

```text
Initialization failed. warmup error:
requests.exceptions.ReadTimeout: HTTPConnectionPool(host='127.0.0.1', port=33120): Read timed out. (read timeout=600)
```

这不是 profile truth 的有效样本失败，而是 server ready 前的内置 warmup 超时。
因此给 `profile_dsv3_moe_dense_refresh.sh` 增加了一个通用排障开关：

```bash
SERVER_EXTRA_ARGS="${SERVER_EXTRA_ARGS:-}"
```

并在启动 `sglang.launch_server` 时追加该参数。冷启动排障时可使用：

```bash
SERVER_EXTRA_ARGS="--skip-server-warmup"
```

这样跳过 SGLang 内置 server warmup，但仍保留 profile 脚本自己的 warmup
repetitions；也就是只绕过启动阶段超时，不改变 truth 的统计 warmup 口径。

后续在 DeepGEMM / CUDA graph 缓存热起来后，重新执行 no-skip smoke，
context 和 generation 均通过。因此正式 H100 truth 推荐口径是：

```text
正式 truth 不默认设置 SERVER_EXTRA_ARGS。
如果容器/缓存完全冷启动，先跑一次 smoke 或预热，让 DeepGEMM JIT 缓存落好。
只有冷启动反复卡在 SGLang 内置 server warmup 超时时，才临时使用
SERVER_EXTRA_ARGS="--skip-server-warmup" 做排障或继续验证链路。
```

Context smoke 通过，命令核心口径：

```text
container: Ai-configurator
dataset: ShareGPT
phase: context
DeepEP mode: normal
EP/TP: 2/2
EPLB: off
tokens: 8,128
SERVER_EXTRA_ARGS: --skip-server-warmup
CONTEXT_WARMUP_REPETITIONS: 1
SAMPLES: 1
```

输出目录：

```text
/cold/tair-kvcache/aiconfigurator/results/h100_truth_smoke_20260715/sharegpt/wideep_context/ep2/session1_skip_server_warmup
```

smoke 结果：

```text
context / eplb_off / token 8:   rank_max_us=548.512
context / eplb_off / token 128: rank_max_us=2279.52
```

Generation low_latency smoke 通过，命令核心口径：

```text
container: Ai-configurator
dataset: ShareGPT
phase: generation
DeepEP mode: low_latency
EP/TP: 2/2
EPLB: off,on
tokens: 8,128
SERVER_EXTRA_ARGS: --skip-server-warmup
GENERATION_WARMUP_REPETITIONS: 1
GENERATION_REPETITIONS: 1
SAMPLES: 1
GENERATION_OCCURRENCE_AGGREGATION: max_chunk
GENERATION_TIMING_SOURCE: auto
```

输出目录：

```text
/cold/tair-kvcache/aiconfigurator/results/h100_truth_smoke_20260715/sharegpt/wideep_generation_low_latency/ep2/session1_skip_server_warmup
```

smoke 结果：

```text
generation / eplb_off / token 8:   rank_max_us=1167.724
generation / eplb_off / token 128: rank_max_us=1158.745
generation / eplb_on  / token 8:   rank_max_us=891.897
generation / eplb_on  / token 128: rank_max_us=1415.453
```

low_latency smoke 期间出现 NVSHMEM/IBGDA 提示：

```text
neither nv_peer_mem, or nvidia_peermem detected. Skipping transport.
init failed for transport: IBGDA
```

但本次 smoke 没有被该提示阻断，eplb off/on 均产出 parsed 结果。正式 truth
仍需保留这条环境提示作为 H100 复现实操记录。

smoke 结束后确认：

```text
无残留 profile/server/scheduler 进程
GPU0/GPU1 显存占用均回到 4 MiB
```

缓存热后，不加 `SERVER_EXTRA_ARGS` 的 no-skip smoke 也通过：

```text
context / eplb_off / token 8:   rank_max_us=559.552
context / eplb_off / token 128: rank_max_us=2275.808

generation / eplb_off / token 8:   rank_max_us=1174.905
generation / eplb_off / token 128: rank_max_us=1073.088
generation / eplb_on  / token 8:   rank_max_us=1687.681
generation / eplb_on  / token 128: rank_max_us=1041.954
```

这些 no-skip smoke 点与 H100 AIC compact recorded 行的临时对比：

```text
wideep_context / eplb_off / token 8:   truth 559.552 us,  AIC 1135.385 us,  error +102.91%
wideep_context / eplb_off / token 128: truth 2275.808 us, AIC 2732.938 us,  error +20.09%

wideep_generation / eplb_off / token 8:   truth 1174.905 us, AIC 751.352 us, error -36.05%
wideep_generation / eplb_off / token 128: truth 1073.088 us, AIC 806.498 us, error -24.84%
wideep_generation / eplb_on  / token 8:   truth 1687.681 us, AIC 745.652 us, error -55.82%
wideep_generation / eplb_on  / token 128: truth 1041.954 us, AIC 804.786 us, error -22.76%
```

注意：上面只是 smoke 级别的单 session、单 effective repetition 临时对比，
不能作为最终校准误差结论。正式结论仍需使用 3 sessions、session 内有效
repetitions 取 median、跨 session 再取 median 的 frozen truth。
```text
/cold/tair-kvcache/aiconfigurator/collector/moe_token_distribution+wideep_moe_20260715_033808
```

镜像外对应路径：

```text
/home/cold/tair-kvcache/aiconfigurator/collector/moe_token_distribution+wideep_moe_20260715_033808
```

注意：H100 上同时存在旧目录 `/home/cold/aiconfigurator/collector`，本轮迁移和
执行使用的是 `/home/cold/tair-kvcache/aiconfigurator/collector`。如果在旧目录下
查看，只能看到 2026-07-12 的历史 run，看不到本轮 2026-07-15 的输出。

结果：

```text
Total errors: 0
```

本轮最终生成：

```text
moe_token_distribution_perf.txt
wideep_context_moe_perf.txt
wideep_generation_moe_perf.txt
collection_summary_sglang.json
recorded_materialized_source/
clean_latency_debug/candidate_full/
```

clean-latency 行为：

- `wideep_context_moe_perf.txt` 替换 102 行 Recorded latency。
- `wideep_generation_moe_perf.txt` 替换 54 行 Recorded latency。
- ordinary MoE source 缺失时只安装 WideEP tables，这是预期行为；本轮没有重跑
  ordinary MoE。

### 2026-07-15 Step F: H100 frozen truth smoke 前置检查

H20 可复用脚本：

```text
/cold/tair-kvcache/aiconfigurator/tools/moe_calibration/profile_dsv3_moe_dense_refresh.sh
/cold/tair-kvcache/aiconfigurator/tools/moe_calibration/run_h20_wideep_context_session_truth_20260714.sh
/cold/tair-kvcache/aiconfigurator/tools/moe_calibration/run_h20_wideep_generation_session_truth_20260714.sh
/cold/tair-kvcache/aiconfigurator/tools/moe_calibration/summarize_wideep_context_session_truth.py
/cold/tair-kvcache/aiconfigurator/tools/moe_calibration/summarize_wideep_generation_session_truth.py
```

H20 truth 口径：

- WideEP context：`DEEPEP_MODE=normal`，`RUN_CONTEXT=1`，
  `RUN_GENERATION=0`，`DISABLE_CUDA_GRAPH=1`，
  `PROFILE_STAGE=routed_experts`，`CONTEXT_TIMING_SOURCE=kernel_external_id`，
  多 session 后取 `median(session median(rank_max_us))`。
- WideEP generation：`DEEPEP_MODE=low_latency`，`RUN_CONTEXT=0`，
  `RUN_GENERATION=1`，`DISABLE_CUDA_GRAPH=0`，
  `GENERATION_TOKEN_SCOPE=global`，`GENERATION_TIMING_SOURCE=auto`，
  `GENERATION_OCCURRENCE_AGGREGATION=max_chunk`，多 session 后取
  `median(session median(rank_max_us))`。
- H100 实机 truth 只跑 EP2，因为当前容器只暴露 2 张 H100。

H100 当前环境检查结果：

- `Ai-configurator` 容器内脚本已存在。
- 容器内 `/model` 当前为空，未看到 `/model/DeepSeek-V3`。
- 容器内仅存在 AIC/SGLang 临时 config cache，例如
  `/tmp/aic_model_config_deepseek-ai--DeepSeek-V3/config.json`，这不能用于
  启动真实 SGLang server。
- H100 host 上可见 ShareGPT 数据：
  `/home/cold/ShareGPT_V3_unfiltered_cleaned_split.json`。
- H100 host/container 里暂未找到
  `longbench_sharegpt_format_for_moe_profile.json`。
- H100 host 上存在 HuggingFace cache：
  `/home/ai_lab/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V3`，
  但当前容器没有挂载 `/home/ai_lab/.cache`，因此容器内不可见。

因此，当前还不能启动真正的 H100 truth smoke；否则
`profile_dsv3_moe_dense_refresh.sh` 会在 `sglang.launch_server` 阶段因模型路径
不可见而失败。下一步需要先恢复容器可见的真实模型路径，并补齐/确认
ShareGPT + LongBench 数据路径。

已完成的可确定迁移动作：

```bash
mkdir -p /model/data
cp -f /cold/ShareGPT_V3_unfiltered_cleaned_split.json \
  /model/ShareGPT_V3_unfiltered_cleaned_split.json
```

迁移后：

```text
/model/ShareGPT_V3_unfiltered_cleaned_split.json 642M
```

进一步检查到的候选模型目录：

```text
/home/ai_lab/fjw/hisim_align/hisim_sglang_align/model_cache/DeepSeek-V3_layers6_maxpos163840 7.6M
/home/ai_lab/fjw/hisim_align/hisim_sglang_align/model_cache/DeepSeek-V3_layers4_maxpos163840 7.6M
/home/ai_lab/ljc/LLM/HFmodel/Model_files/deepseek_v3 136K
/home/ai_lab/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V3 212K
```

这些目录都只是 config/tokenizer 级别，不包含完整权重，不能作为真实 truth smoke
的 `MODEL_PATH`。H100 旧结果显示 2026-07-12 当时使用的是
`/model/DeepSeek-V3`，但当前容器中该路径不存在。

已完成 dry-run 命令生成检查，未启动 server：

```bash
cd /cold/tair-kvcache/aiconfigurator

env DRY_RUN=1 \
  RESULTS_ROOT=/cold/tair-kvcache/aiconfigurator/results/h100_truth_smoke_dryrun_20260715/context \
  DATASETS=sharegpt EPS=2 SESSIONS=1 SAMPLES_PER_SESSION=1 \
  CONTEXT_WARMUP_REPETITIONS=1 WIDEEP_CONTEXT_TOKENS="8 128" \
  MODEL_PATH=/model/DeepSeek-V3 TOKENIZER_PATH=/model/DeepSeek-V3 \
  bash tools/moe_calibration/run_h20_wideep_context_session_truth_20260714.sh

env DRY_RUN=1 \
  RESULTS_ROOT=/cold/tair-kvcache/aiconfigurator/results/h100_truth_smoke_dryrun_20260715/generation \
  DATASETS=sharegpt EPS=2 SESSIONS=1 REPS_PER_SESSION=1 WARMUP_REPS=1 \
  WIDEEP_GENERATION_TOKENS_EP24="8 128" \
  MODEL_PATH=/model/DeepSeek-V3 TOKENIZER_PATH=/model/DeepSeek-V3 \
  bash tools/moe_calibration/run_h20_wideep_generation_session_truth_20260714.sh
```

dry-run 输出位置：

```text
/cold/tair-kvcache/aiconfigurator/results/h100_truth_smoke_dryrun_20260715
```

注意：

- generation session 脚本当前在 `run_profile` 参数中固定
  `RUN_EPLB_OFF=1 RUN_EPLB_ON=1`，外部 `RUN_EPLB_ON=0` 不会覆盖。
- H20 脚本默认注入 `bond0` / `mlx5_bond_0,mlx5_bond_1` 等 DeepEP 网卡环境。
  H100 当前容器网卡不是这个命名：

```text
net: enp155s0np0, enp170s0np0, enp187s0np0, enp218s0np0, ...
ib:  mlx5_0 ... mlx5_9
```

因此 H100 真实 smoke 建议先参考 2026-07-12 H100 旧脚本，不强制继承 H20 的
`deep_ep_env_args`，让 SGLang/DeepEP 按 H100 环境自动选择，或单独确认 H100
对应的 `NCCL_IB_HCA` / socket iface 后再显式设置。

### 2026-07-15 Step G: 重新挂载后的 H100 smoke 与执行口径

用户重新启动 `Ai-configurator` 容器后，容器内模型与数据路径恢复：

```text
/model/DeepSeek-V3/config.json
/model/ShareGPT_V3_unfiltered_cleaned_split.json
/model/data/longbench_sharegpt_format_for_moe_profile.json
```

SGLang 与 AIC 已在容器内重新安装，确认 import 路径：

```text
sglang -> /sgl-workspace/sglang/python/sglang/__init__.py
aiconfigurator -> /cold/tair-kvcache/aiconfigurator/src/aiconfigurator/__init__.py
```

`SERVER_EXTRA_ARGS="--skip-server-warmup"` 的结论：

- 初次冷启动时，SGLang built-in server warmup 可能因为 DeepGEMM JIT 超过 600s。
- 加 `--skip-server-warmup` 可以作为冷启动 fallback，帮助 smoke 先走通。
- 正式 frozen truth 不默认加 `--skip-server-warmup`；正确流程是先 smoke/prewarm，
  再跑不带 skip 的正式 profile。

H100 smoke 通过后，ShareGPT context/generation 的正式 truth 均不带
`SERVER_EXTRA_ARGS`。

### 2026-07-15 Step H: ShareGPT EP2 frozen truth 已完成

H100 当前容器只暴露 2 张卡，因此实机 frozen truth 只覆盖 EP2。

#### WideEP context truth

执行口径：

- dataset: ShareGPT
- phase: WideEP context
- DeepEP mode: `normal`
- EP/TP: `2/2`
- EPLB: off/on
- sessions: 3
- 每 session: warmup 后 3 次有效 sample
- final truth: `median(session median(rank_max_us))`
- server extra args: none

输出目录：

```text
/cold/tair-kvcache/aiconfigurator/results/h100_frozen_truth_20260715/sharegpt_wideep_context_session_median_20260715_0716
```

关键输出：

```text
wideep_context_session_median_summary.csv
wideep_context_unstable_points.csv
```

#### WideEP generation low_latency truth

第一次直接跑全 token 时，在 `token=896` 附近失败，server log 报：

```text
RuntimeError: Failed: Assertion error ... x.size(0) <= num_max_dispatch_tokens_per_rank
```

定位结果：

- SGLang 默认 `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128`。
- DeepEP low_latency 代码允许最大设置到 `1024`。
- EP2/global token=896/1024/1280 会产生较大的本地 decode batch，默认 128 的
  per-rank dispatch 容量不足。
- 这不是 AIC policy 校准，而是实机 low_latency buffer 容量参数；H100 正式
  generation truth 需要显式设置：

```bash
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024
```

验证 smoke：

```text
token=896, eplb_off, EP/TP=2/2, global scope, max_dispatch=1024: pass
```

正式执行口径：

- dataset: ShareGPT
- phase: WideEP generation low_latency
- DeepEP mode: `low_latency`
- EP/TP: `2/2`
- EPLB: off/on
- sessions: 3
- 每 session: warmup 后 5 次有效 repetition
- generation token scope: `global`
- generation timing source: `auto`
- occurrence aggregation: `max_chunk`
- final truth: `median(session median(rank_max_us))`
- server extra args: none
- `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024`

输出目录：

```text
/cold/tair-kvcache/aiconfigurator/results/h100_frozen_truth_20260715/sharegpt_wideep_generation_session_median_20260715_0755_maxdispatch1024
```

关键输出：

```text
wideep_generation_session_median_summary.csv
wideep_generation_unstable_points.csv
```

### 2026-07-15 Step I: ShareGPT EP2 与 H100 AIC compact 顶层文件对比

对比输入必须是 H100 collector 顶层 compact 文件，不使用 keep-latency-sources 的
中间源：

```text
/cold/tair-kvcache/aiconfigurator/collector/moe_token_distribution+wideep_moe_20260715_033808/wideep_context_moe_perf.txt
/cold/tair-kvcache/aiconfigurator/collector/moe_token_distribution+wideep_moe_20260715_033808/wideep_generation_moe_perf.txt
```

对比口径：

- 只比较 `moe_ep_size=2`。
- `recorded_no_eplb` 对齐 truth `eplb=off`。
- `recorded_eplb` 对齐 truth `eplb=on`。
- AIC compact 表的 `latency` 按 ms 读取，乘 1000 后与 truth us 比较。
- truth 使用 `rank_max_final_median_us`。

当前 ShareGPT / EP2 结果：

```text
wideep_context:
  matched=34, missing=2
  MAPE=24.26%, max=154.34%

wideep_generation_low_latency:
  matched=18, missing=4
  MAPE=34.73%, max=48.61%
```

context 的主要问题集中在小 token：

```text
recorded_eplb token=8:   AIC 1470.449 us, truth 578.145 us,  +154.34%
recorded_eplb token=16:  AIC 1724.799 us, truth 982.400 us,   +75.57%
recorded_eplb token=32:  AIC 2176.904 us, truth 1420.962 us,  +53.20%
recorded_no_eplb token=8:  AIC 1135.385 us, truth 579.200 us, +96.03%
recorded_no_eplb token=16: AIC 2051.298 us, truth 982.207 us, +108.85%
```

context 中大 token 基本收敛到 2%-15% 区间，但 token=18888 在 AIC compact
中缺失，后续如果需要完整实机覆盖，要决定是否给 AIC collector 增加该点或从 truth
集合剔除该点。

generation low_latency 当前是系统性低估：

```text
recorded_eplb token=8:     AIC 745.652 us, truth 1450.965 us, -48.61%
recorded_eplb token=32:    AIC 798.344 us, truth 1456.664 us, -45.19%
recorded_eplb token=128:   AIC 804.786 us, truth 1401.594 us, -42.58%
recorded_no_eplb token=8:  AIC 751.352 us, truth 1414.089 us, -46.87%
recorded_no_eplb token=128:AIC 806.498 us, truth 1408.531 us, -42.74%
```

generation 中 `token=4` 和 `token=1280` 在 AIC compact 中缺失。这与 H20
阶段讨论过的左外推/边界 token 问题一致；H100 复现时需要明确这些点是否纳入
正式 compare 口径。

当前判断：

- H20 的 collector/AIC 逻辑与实机 truth 流程已经可以迁移到 H100 并跑通。
- 但以 ShareGPT EP2 frozen truth 看，当前 H100 AIC compact 表还不能算校准通过。
- 不建议立刻做 H100 特有厚校准；下一步先补 LongBench cross-validation，并确认
  H100 collector 是否应该增加很薄的硬件无关修正，还是把 H100 作为“迁移发现偏差”
  的证据单独记录。

### 2026-07-15 Step J: H20 WideEP generation low_latency 最小复测

目的：

- 暂不修改 AIC collector / policy。
- 复查 H20 frozen truth 中 `WideEP generation low_latency / EP2 / ShareGPT`
  是否稳定落在 `~0.85ms` 主平台，还是也会稳定复现 H100 当前的 `~1.4ms`
  平台。
- 只做最小 4 点复测：`eplb_off/on x token=8/128`。

执行约束：

- 必须进入 H20 `Ai-configurator` 镜像执行，不在宿主 Python 环境直接跑。
- H20 镜像名：`Ai-configurator`。
- AIC 路径：`/cold/tair-kvcache/aiconfigurator`。
- SGLang 路径：`/sgl-workspace/sglang`。
- 模型路径：`/model/DeepSeek-V3`。
- ShareGPT 数据：`/model/ShareGPT_V3_unfiltered_cleaned_split.json`。
- H20 当前可见 8 张 H20 GPU。
- H20 当前网卡/IB 与历史脚本默认一致，可使用 `bond0` 和
  `mlx5_bond_0,mlx5_bond_1`。

环境确认命令：

```bash
docker exec Ai-configurator bash -lc '
  nvidia-smi -L
  ls -ld /cold/tair-kvcache/aiconfigurator /sgl-workspace/sglang /model/DeepSeek-V3
  ls -l /model/ShareGPT_V3_unfiltered_cleaned_split.json
'
```

计划执行命令：

```bash
docker exec Ai-configurator bash -lc '
  cd /cold/tair-kvcache/aiconfigurator
  RESULTS_ROOT=/cold/tair-kvcache/aiconfigurator/results/h20_wideep_generation_lowlat_ep2_sharegpt_recheck_20260715_1805
  DATASETS=sharegpt EPS=2 SESSIONS=3 REPS_PER_SESSION=3 WARMUP_REPS=3 \
  WIDEEP_GENERATION_TOKENS_EP24="8 128" \
  RESULTS_ROOT="${RESULTS_ROOT}" \
  bash tools/moe_calibration/run_h20_wideep_generation_session_truth_20260714.sh
'
```

本次复测完成后，需要记录：

- `wideep_generation_session_median_summary.csv` 中 4 个点的
  `rank_max_session_medians_us` / `rank_max_final_median_us` / `stability`。
- 与 H20 frozen truth 原始值对比：
  - off token=8: `849.474;858.210;850.882`, median `850.882us`
  - off token=128: `885.426;978.641;853.222`, median `885.426us`
  - on token=8: `873.842;862.974;1498.625`, median `873.842us`
  - on token=128: `901.257;876.863;1005.834`, median `901.257us`
- 与 H100 当前复测值对比：H100 在 `cpu/gpu handler`、`max_dispatch=256`
  下仍稳定约 `1.4ms`。

执行结果：

```text
输出目录:
/cold/tair-kvcache/aiconfigurator/results/h20_wideep_generation_lowlat_ep2_sharegpt_recheck_20260715_1805

summary:
/cold/tair-kvcache/aiconfigurator/results/h20_wideep_generation_lowlat_ep2_sharegpt_recheck_20260715_1805/wideep_generation_session_median_summary.csv
```

复测结果：

```text
off token=8:
  session medians = 859.130;888.407;867.402 us
  final median    = 867.402 us
  stability       = stable

off token=128:
  session medians = 1466.260;830.540;844.423 us
  final median    = 844.423 us
  stability       = unstable

on token=8:
  session medians = 878.189;905.340;917.162 us
  final median    = 905.340 us
  stability       = stable

on token=128:
  session medians = 889.832;880.467;876.989 us
  final median    = 880.467 us
  stability       = stable
```

与 H20 frozen truth 对比：

```text
off token=8:
  frozen median 850.882 us, recheck median 867.402 us

off token=128:
  frozen median 885.426 us, recheck median 844.423 us
  recheck 中出现一个 1466.260 us 高 session，但 median 仍回到 0.84ms 平台。

on token=8:
  frozen median 873.842 us, recheck median 905.340 us

on token=128:
  frozen median 901.257 us, recheck median 880.467 us
```

结论：

- H20 复测确认 `WideEP generation low_latency / EP2 / ShareGPT` 的主平台仍是
  `0.84ms~0.91ms`。
- H20 会偶发 `~1.45ms` 高 session，但不是主平台；median 不会稳定落在高平台。
- H100 当前在同类 4 点 smoke 中，无论 `NVSHMEM_IBGDA_NIC_HANDLER=cpu/gpu`、
  `max_dispatch=256/1024`，均稳定落在 `~1.4ms`。
- 因此，H100 当前 WideEP generation low_latency 的误差不是 H20 frozen truth
  偶然取错 median，而是 H100 DeepEP low_latency 路径表现出不同硬件/环境平台。

## Step K：H20 复测任务状态复查

时间：`2026-07-15 18:16 CST`

复查命令：

```bash
docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
docker exec Ai-configurator bash -lc "
  ps -ef | grep -E 'run_h20_wideep_generation|sglang|bench|python' | grep -v grep | head -50
"
docker exec Ai-configurator bash -lc "
  cd /cold/tair-kvcache/aiconfigurator &&
  ls -l results/h20_wideep_generation_lowlat_ep2_sharegpt_recheck_20260715_1805 &&
  sed -n '1,80p' results/h20_wideep_generation_lowlat_ep2_sharegpt_recheck_20260715_1805/wideep_generation_session_median_summary.csv &&
  sed -n '1,80p' results/h20_wideep_generation_lowlat_ep2_sharegpt_recheck_20260715_1805/wideep_generation_unstable_points.csv
"
```

复查结果：

- H20 `Ai-configurator` 容器仍在运行。
- 未发现正在运行的 `run_h20_wideep_generation` / bench / sglang 相关复测进程，本轮任务已经结束。
- 输出目录完整，包含：
  - `MANIFEST.md`
  - `wideep_generation_session_median_summary.csv`
  - `wideep_generation_unstable_points.csv`
- 唯一 unstable 点：

```text
sharegpt, EP2, eplb=off, token=128
session medians = 1466.260;830.540;844.423 us
final median    = 844.423 us
spread          = 75.28%
```

复查结论：

- H20 复测数据已落盘且完整。
- H20 `token=128/off` 的确有一次高 session，但最终 median 仍在 `0.84ms` 平台。
- 这进一步支持当前判断：H100 WideEP generation low_latency 稳定落到 `~1.4ms`，不是 H20 truth 由偶发高 session 造成的口径问题。

## Step L：WideEP generation low_latency profile 初步归因

时间：`2026-07-15 18:xx CST`

目标：

- 对比 H20 与 H100 的 `WideEP generation low_latency / EP2 / ShareGPT` profile。
- 判断 H100 `~1.4ms` 平台是哪个模块主导，以及它和 H20 `~0.85ms` 平台的差异来源。

分析口径：

- 先看 parsed rank aggregate：
  - `generation_dense_refresh_eplb_off_rank_aggregate.csv`
  - `generation_dense_refresh_eplb_off_per_layer_rank_aggregate.csv`
- 再解析 `.trace.json.gz` 中：
  - `gpu_user_annotation`
  - `kernel`
  - `gpu_memcpy`
  - `gpu_memset`
- 聚合时只作为 profile 归因参考，不直接拿 kernel dur 总和等价 wall time，因为 trace 中不同 stream/kernel 会有重叠。

H20 侧检查：

```bash
python3 - <<'PY'
import csv, pathlib
base=pathlib.Path(
  'aiconfigurator/results/h20_wideep_generation_lowlat_ep2_sharegpt_recheck_20260715_1805/sharegpt/wideep_generation_low_latency/ep2'
)
for sess in [1,2,3]:
    f=base/f'session{sess}/parsed/generation_dense_refresh_eplb_off_per_layer_rank_aggregate.csv'
    print('SESSION', sess)
    for r in csv.DictReader(f.open()):
        if r['num_tokens']=='128' and r['repetition']=='0':
            print(r)
PY
```

H20 `off/token=128` 典型结果：

```text
session1 rep0:
  layer3 rank_min/rank_max = 1047.462 / 1742.457 us
  layer4 rank_min/rank_max = 848.004  / 1467.217 us
  layer5 rank_min/rank_max = 899.992  / 1432.095 us

session2 rep0:
  layer3 rank_min/rank_max = 1017.545 / 1047.138 us
  layer4 rank_min/rank_max = 811.729  / 825.216 us
  layer5 rank_min/rank_max = 813.306  / 830.540 us

session3 rep0:
  layer3 rank_min/rank_max = 1030.067 / 1035.336 us
  layer4 rank_min/rank_max = 817.744  / 838.032 us
  layer5 rank_min/rank_max = 811.363  / 826.093 us
```

H20 结论：

- H20 的高 session 是 rank 间尾部差异，不是所有 rank 都稳定变高。
- H20 的正常平台仍是 `~0.82ms~0.90ms`。
- H20 偶发高点对应某些 layer/rank 的通信/同步尾部，而不是 DeepGEMM 本身整体变慢。

H20 trace 里 `off/token=128/session2/rep0/rank1` 的主要 GPU annotation：

```text
aic_moe/layer_4/routed/compute      ~1.713 ms
aic_moe/layer_5/routed/compute      ~1.574 ms
aic_moe/layer_3/routed/compute      ~1.394 ms
nccl:_reduce_scatter_base           ~1.954 ms total
aic_moe/layer_5/routed/dispatch     ~0.866 ms
aic_moe/layer_4/routed/dispatch     ~0.774 ms
aic_moe/layer_3/routed/dispatch     ~0.689 ms
aic_moe/layer_5/routed/combine      ~0.637 ms
```

H20 trace 里主要 kernel：

```text
deep_gemm::sm90_fp8_gemm_1d2d_impl  dominates compute
deep_ep::internode_ll::dispatch     appears in comm path
deep_ep::internode_ll::combine      appears in comm path
ncclDevKernel_ReduceScatter         appears in sync path
ncclDevKernel_AllGather             appears in sync path
sglang::cross_device_reduce_1stage  appears in reduction path
```

H100 侧检查：

```bash
ssh ai_lab@10.110.183.26
docker exec Ai-configurator bash -lc '
cd /cold/tair-kvcache/aiconfigurator
sed -n "1,80p" \
  results/h100_frozen_truth_20260715/sharegpt_wideep_generation_session_median_20260715_0755_maxdispatch1024/wideep_generation_session_median_summary.csv
'
```

H100 `off/token=8` 与 `off/token=128` 的 parsed 特征：

```text
off token=8:
  session medians = 1414.089;1440.301;1403.259 us
  final median    = 1414.089 us
  stability       = stable

off token=128:
  session medians = 1408.531;1419.316;1404.869 us
  final median    = 1408.531 us
  stability       = stable
```

H100 `session3/off/token=128/rep0` parsed：

```text
layer3 rank_min/rank_max = 1792.858 / 1869.791 us
layer4 rank_min/rank_max = 1398.861 / 1416.790 us
layer5 rank_min/rank_max = 1393.989 / 1436.969 us
median(layer3,4,5) rank_max = 1436.969 us
```

H100 与 H20 的关键不同：

- H100 不是 H20 那种单 session / 单 rank 偶发高，而是 token=8、token=128 都稳定在 `~1.4ms`。
- H100 rank_min 也经常已经在 `~1.39ms`，说明不是单 rank 尾部问题，而是整体平台抬高。
- H100 `token=8` 的固定开销明显更高：
  - `shared_experts` 每层约 `0.60ms~0.65ms`
  - `routed/compute` 每层约 `0.54ms~0.66ms`
  - `dispatch/combine + NCCL` 合计比 H20 小 token 更重
- H100 `token=128` 的主导项仍包含：
  - `routed/compute`
  - `shared_experts`
  - `DeepEP dispatch/combine`
  - `NCCL all_gather/reduce_scatter`

H100 `off/token=8/session3/rep0/rank1` 主要 GPU annotation：

```text
aic_moe/layer_3/routed/compute      ~0.665 ms
aic_moe/layer_5/shared_experts      ~0.647 ms
aic_moe/layer_4/shared_experts      ~0.645 ms
aic_moe/layer_3/shared_experts      ~0.625 ms
aic_moe/layer_4/routed/compute      ~0.572 ms
aic_moe/layer_5/routed/compute      ~0.545 ms
nccl:_all_gather_base               ~0.480 ms total
nccl:_reduce_scatter_base           ~0.445 ms total
aic_moe/layer_3/routed/dispatch     ~0.361 ms
aic_moe/layer_4/routed/dispatch     ~0.229 ms
aic_moe/layer_3/routed/combine      ~0.191 ms
```

H100 `off/token=128/session3/rep0/rank1` 主要 GPU annotation：

```text
nccl:_reduce_scatter_base           ~1.414 ms total
aic_moe/layer_5/routed/compute      ~1.316 ms
aic_moe/layer_4/routed/compute      ~1.203 ms
nccl:_all_gather_base               ~1.048 ms total
aic_moe/layer_3/routed/compute      ~0.964 ms
aic_moe/layer_3/shared_experts      ~0.666 ms
aic_moe/layer_3/routed/dispatch     ~0.653 ms
aic_moe/layer_4/shared_experts      ~0.651 ms
aic_moe/layer_5/shared_experts      ~0.647 ms
aic_moe/layer_4/routed/dispatch     ~0.600 ms
aic_moe/layer_5/routed/dispatch     ~0.555 ms
aic_moe/layer_3/routed/combine      ~0.537 ms
aic_moe/layer_5/routed/combine      ~0.478 ms
```

H100 trace 里的主要 kernel：

```text
deep_gemm::sm90_fp8_gemm_1d2d_impl       routed/shared compute 主体
deep_ep::internode_ll::dispatch          WideEP/DeepEP dispatch 路径
deep_ep::internode_ll::combine           WideEP/DeepEP combine 路径
ncclDevKernel_AllGather_RING_LL          同步/聚合路径
ncclDevKernel_ReduceScatter_Sum_bf16     同步/归约路径
sglang::cross_device_reduce_1stage       cross-device reduce
```

初步归因：

- H100 WideEP generation low_latency 的 `~1.4ms` 平台不是单个 DeepGEMM kernel
  独立造成的。
- 更像是 `routed compute + shared_experts + DeepEP dispatch/combine + NCCL
  sync/reduce` 的固定底座一起抬高。
- 对小 token，GEMM 计算量本身不大，`shared_experts`、DeepEP dispatch/combine、
  NCCL all_gather/reduce_scatter 这类固定开销会主导 wall time。
- 对 token=128，routed compute 开始明显，但 H100 仍有更高的通信/同步底座。
- 普通 MoE 正常而 WideEP 异常，说明问题集中在 WideEP/DeepEP low_latency
  的通信/同步路径，以及 shared/routed expert 路径的组合开销，而不是普通
  FP8 GEMM 或普通 MoE 计算口径整体错误。

后续如果要继续定位：

1. 做同一 token 下的 rank0/rank1 trace 差分，确认 H100 是否所有 rank 都同步抬高。
2. 做 `token=8/128` 的 layer3/4/5 NVTX span 级别表格，把每层拆成：
   `router/topk/dispatch/compute/combine/shared_experts/NCCL`。
3. 如果需要更强证据，再跑一次 H100 `NCCL_DEBUG=INFO` / DeepEP 相关 debug，
   但这一步可能会扰动性能，暂时不作为校准输入。

## 2026-07-15 WideEP 原始 AIC 口径与实机 truth 口径核查

本节只核查 AIC 原先 `uniform/power_law` 方式，不看新提出的 `recorded`
方式。目标是排除 H100 WideEP 误差是否来自“实机 truth 多测了额外模块”。

代码证据：

- 实机 truth 脚本使用 `PROFILE_STAGE=routed_experts`。
- SGLang `routed_experts` 包的是 `self.experts(hidden_states, topk_output)`。
- 在 WideEP `experts.forward_impl()` 里，这一段包含：
  - `routed/dispatch`: `self.dispatcher.dispatch(...)`
  - `routed/compute`: `self.run_moe_core(...)`
  - `routed/combine`: `self.dispatcher.combine(...)`
  - 以及满足条件时的 `routed/all_reduce`
- `routed_experts` 不包含 `shared_experts`、`router`、`topk`、
  `output_postprocess`。

AIC 原先 `uniform/power_law` WideEP collector 的关键口径：

- context 路径在构造 `DeepEPNormalDispatchOutput` 后，计时区间只调用
  `moe_layer.experts.run_moe_core(dispatch_output)`。
- generation 路径在构造 `DeepEPLLDispatchOutput` 后，`kernel_func()` 里也只调用
  `moe_layer.experts.run_moe_core(dispatch_output)`。
- 输出行写成 `kernel_source=deepepmoe`，但这个原始 `uniform/power_law`
  计时并不是完整 `self.experts()`，而是“已经有 dispatch output 之后的
  expert compute core”。

因此当前结论：

- 实机 truth 没有把 `shared_experts/router/topk/output_postprocess` 这些外层
  模块算进去；这一点可以排除。
- 但是实机 truth 的 `routed_experts` 比 AIC 原先 `uniform/power_law` 多了
  `dispatch + combine` 这两个 DeepEP 通信/整理阶段。
- 所以如果拿 AIC 原先 `uniform/power_law` 行直接对实机
  `routed_experts`，口径并不完全一致；H100 WideEP 偏大不应该直接归因于
  AIC compute core 校准失败。
- 新的 `recorded` 方案正是为了让 AIC 的输入分布和真实请求更接近，但它仍然要
  明确区分是校准 `run_moe_core`，还是校准完整 `routed_experts`。

下一步建议：

1. 先用已有 profile 拆 H100/H20 的 `routed_experts`：
   `routed/dispatch + routed/compute + routed/combine (+ all_reduce)`。
2. AIC 原始 `uniform/power_law` 只能优先对齐 `routed/compute`，不要直接对齐
   `routed_experts`。
3. 如果最终目标是预测实机 `routed_experts`，需要在 AIC 侧显式补一层很薄的
   WideEP communication/runtime 项，或者把 truth 对齐到 `routed/compute`。
4. 在 H100 上先不改 AIC，先做分项对比表，确认误差主要来自
   `dispatch/combine` 固定底座，还是 `run_moe_core` 本身。

## 2026-07-15 WideEP routed_experts 分项拆解

拆解目的：

- 只看实机 profile 里的 `routed_experts` 内部组成。
- 确认 AIC 原始 `uniform/power_law` 更接近 `routed/compute`，还是完整
  `routed_experts`。
- 当前不修改 AIC。

拆解方法：

- 输入是 PyTorch profiler 的 `*.trace.json.gz`。
- 每个 trace 文件按 layer 3/4/5 提取：
  `routed_experts`、`routed/dispatch`、`routed/compute`、`routed/combine`、
  `routed/all_reduce`。
- 同一个 trace 内同名 stage 可能出现多个 occurrence，因此先取该 stage 的最大
  occurrence。
- 每个 rep 内跨 rank 取最大 stage-sum 的 rank/layer，rep 间取 median。
- `stage_sum = dispatch + compute + combine + all_reduce` 只是分项强弱估计。
  因为 stage 之间可能 overlap，`stage_sum` 不能当作最终 wall latency。
- `routed_experts_span` 是 profiler annotation 里的 span 参考值，也可能和
  `stage_sum` 不完全一致。

### H20 EP2 WideEP generation low_latency 拆解

数据来源：

- `aiconfigurator/results/h20_wideep_generation_ep2_low_latency_ibgda_cpu_truth_20260713_061744/profiles_eplb_off`
- `aiconfigurator/results/h20_wideep_generation_ep2_low_latency_ibgda_cpu_truth_20260713_061744/profiles_eplb_on`

| dataset | token | routed_experts_span ms | stage_sum ms | dispatch ms | compute ms | combine ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H20 EP2 off | 8 | 1.105 | 1.141 | 0.301 | 0.632 | 0.217 |
| H20 EP2 off | 128 | 0.831 | 2.561 | 0.267 | 1.854 | 0.375 |
| H20 EP2 on | 8 | 1.159 | 1.257 | 0.298 | 0.639 | 0.411 |
| H20 EP2 on | 128 | 1.038 | 2.750 | 0.293 | 1.853 | 0.443 |

H20 观察：

- token=8 时，`dispatch+combine` 是明显固定底座：
  - off: `0.301 + 0.217 = 0.518ms`
  - on: `0.298 + 0.411 = 0.709ms`
- token=128 时，`compute` 开始主导 stage_sum，但 `dispatch/combine`
  仍有 `0.64ms~0.74ms` 的量级。
- 因为 H20 trace 里存在 overlap，token=128 的 `stage_sum` 大于
  `routed_experts_span`，所以这里只看构成强弱，不把它当最终 latency。

### H100 EP2 WideEP generation low_latency 拆解

数据来源：

- `/home/cold/aiconfigurator/results/h100_dsv3_moe_ep2_formal_truth_20260712/generation/profiles_eplb_off`
- `/home/cold/aiconfigurator/results/h100_dsv3_moe_ep2_formal_truth_20260712/generation/profiles_eplb_on`
- `/home/cold/aiconfigurator/results/h100_dsv3_moe_ep2_sharegpt_full_calibration_20260712/wideep/profiles_eplb_off`
- `/home/cold/aiconfigurator/results/h100_dsv3_moe_ep2_sharegpt_full_calibration_20260712/wideep/profiles_eplb_on`

| dataset | token | routed_experts_span ms | stage_sum ms | dispatch ms | compute ms | combine ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H100 formal off | 8 | 2.883 | 2.602 | 0.231 | 2.314 | 0.057 |
| H100 formal off | 128 | 2.141 | 1.898 | 0.190 | 1.649 | 0.058 |
| H100 formal on | 8 | 2.886 | 2.606 | 0.194 | 2.358 | 0.057 |
| H100 formal on | 128 | 2.156 | 1.903 | 0.166 | 1.682 | 0.055 |
| H100 ShareGPT off | 8 | 6.017 | 11.545 | 3.242 | 2.313 | 5.216 |
| H100 ShareGPT off | 128 | 5.535 | 5.722 | 2.981 | 2.236 | 0.495 |
| H100 ShareGPT on | 8 | 3.584 | 8.836 | 1.136 | 2.198 | 0.310 |
| H100 ShareGPT on | 128 | 4.217 | 6.543 | 2.170 | 2.179 | 0.638 |

H100 观察：

- H100 formal 这套更像 `compute` 主导，`dispatch/combine` 很小。
- H100 ShareGPT full calibration 这套里，`dispatch/combine` 明显抬高，
  尤其 off/token=8：
  - `dispatch ~= 3.24ms`
  - `combine ~= 5.22ms`
  - `compute ~= 2.31ms`
- 这说明 H100 WideEP 的误差不能只看 expert compute。不同实机路径/运行配置下，
  DeepEP communication/runtime 底座可能变化很大。

### 和 AIC 原始 uniform 的关系

AIC 原始 `uniform/power_law` collector 计时边界是：

```text
DeepEPNormalDispatchOutput / DeepEPLLDispatchOutput 已经构造好
-> moe_layer.experts.run_moe_core(dispatch_output)
```

所以它更接近 `routed/compute`，而不是完整：

```text
routed_experts = dispatch + compute + combine (+ all_reduce)
```

这次拆解后的判断：

- 如果拿原始 AIC `uniform/power_law` 对实机 `routed_experts`，H100 上会把
  `dispatch/combine` 的平台开销全部算成 AIC 误差，口径不公平。
- 如果目标是校准 `run_moe_core`，应该对齐 profile 的 `routed/compute`。
- 如果目标是校准实机端到端 routed path，就必须显式建模或补偿
  `dispatch/combine`，尤其是 H100 ShareGPT 这类通信底座抬高的情况。
- 当前先不改 AIC，下一步应先做 H100 当前正式实机 truth 的同口径 profile：
  固定 ShareGPT、EP2、WideEP generation low_latency，保留
  `routed/dispatch`、`routed/compute`、`routed/combine` 分项，再决定是否需要
  一层很薄的 communication/runtime policy。

## 2026-07-15 H100 当前环境 WideEP generation profile 重跑

目的：

- 固定 H100 当前容器环境，重新跑 ShareGPT / EP2 / WideEP generation
  low_latency 的关键点。
- 只跑 token `8,128`，覆盖 EPLB off/on。
- 保留 profile trace，用于拆 `routed/dispatch`、`routed/compute`、
  `routed/combine`。

执行环境：

```text
container: Ai-configurator
workdir: /cold/tair-kvcache/aiconfigurator
GPU: 2 x H100
MODEL_PATH: /model/DeepSeek-V3
dataset: /model/ShareGPT_V3_unfiltered_cleaned_split.json
```

这次没有继续使用 H20 默认的 `bond0/mlx5_bond_*` 变量，而是显式使用 H100
当前容器可见环境：

```text
GLOO_SOCKET_IFNAME=enp86s0f0
NCCL_SOCKET_IFNAME=enp86s0f0
NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=enp86s0f0
NVSHMEM_SOCKET_IFNAME=enp86s0f0
NCCL_IB_HCA=mlx5_0,mlx5_1
NVSHMEM_HCA_PE_MAPPING=mlx5_0:1:2,mlx5_1:1:2
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024
```

执行口径：

```text
DATASETS=sharegpt
EPS=2
SESSIONS=3
REPS_PER_SESSION=5
WARMUP_REPS=3
WIDEEP_GENERATION_TOKENS_EP24="8 128"
MOE_A2A_BACKEND=deepep
DEEPEP_MODE=low_latency
DISABLE_CUDA_GRAPH=0
PROFILE_STAGE=routed_experts
GENERATION_TIMING_SOURCE=auto
GENERATION_OCCURRENCE_AGGREGATION=max_chunk
GENERATION_TOKEN_SCOPE=global
```

输出目录：

```text
/cold/tair-kvcache/aiconfigurator/results/h100_frozen_truth_20260715/sharegpt_wideep_generation_profile_rerun_20260715_1912_tokens8_128_h100env
```

关键文件：

```text
wideep_generation_session_median_summary.csv
wideep_generation_unstable_points.csv
wideep_generation_routed_stage_breakdown.csv
wideep_generation_routed_stage_breakdown_by_session.csv
```

### 重跑 truth 摘要

| dataset | EP | eplb | token | session medians us | final median us | spread | stability |
| --- | ---: | --- | ---: | --- | ---: | ---: | --- |
| sharegpt | 2 | off | 8 | 1382.612;1425.759;1452.538 | 1425.759 | 4.90% | stable |
| sharegpt | 2 | off | 128 | 1406.795;1341.282;1438.946 | 1406.795 | 6.94% | stable |
| sharegpt | 2 | on | 8 | 1465.781;1517.756;1486.135 | 1486.135 | 3.50% | stable |
| sharegpt | 2 | on | 128 | 1448.822;1453.806;1603.933 | 1453.806 | 10.67% | stable |

结论：

- H100 当前环境下，4 个关键点仍稳定在 `~1.4ms~1.5ms`。
- 这与前一版 H100 frozen truth 的 `~1.4ms` 平台一致。
- 所以 H100 偏高不是 H20 truth 偶发抖动，也不是 H100 单次 smoke 偶然值。

### routed_experts 分项拆解

拆解方法同上一节：每个 trace 里取 layer 3/4/5 的 stage 最大 occurrence，
rep/session 取 median。注意 `stage_sum` 仍然只能表示分项强弱，不等于 wall
latency，因为 profiler stage 之间存在 overlap，且不同 stage 的最大 occurrence
不一定来自同一个 chunk。

| dataset | EP | eplb | token | routed_experts span ms | stage_sum ms | dispatch ms | compute ms | combine ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sharegpt | 2 | off | 8 | 1.918 | 4.939 | 2.467 | 1.146 | 0.920 |
| sharegpt | 2 | off | 128 | 1.206 | 3.566 | 1.952 | 1.102 | 0.446 |
| sharegpt | 2 | on | 8 | 1.772 | 5.186 | 3.201 | 0.919 | 1.096 |
| sharegpt | 2 | on | 128 | 1.322 | 4.267 | 2.603 | 1.394 | 0.320 |

分项结论：

- H100 当前 ShareGPT / WideEP generation low_latency 下，
  `routed/dispatch` 是非常明显的底座，常常比 `routed/compute` 还大。
- `combine` 在 token=8 也不是小项，尤其 off/on 的小 token 分别约
  `0.92ms` / `1.10ms`。
- 因此，如果 AIC 原始 `uniform/power_law` 只测 `run_moe_core`，拿它直接对齐
  实机 `routed_experts` 会系统性低估，这是口径差异，不是单纯 compute 校准失败。

### 对“实机 truth 是否多算”的当前定义

相对 AIC 原始 `uniform/power_law`：

- 是的，当前 `PROFILE_STAGE=routed_experts` 的实机 truth 多算了
  `dispatch/combine`。
- 但它没有多算 `shared_experts`、`router`、`topk`、`output_postprocess`。

更精确地说：

```text
AIC 原始 uniform/power_law:
  run_moe_core(dispatch_output)

实机 PROFILE_STAGE=routed_experts:
  dispatcher.dispatch + run_moe_core + dispatcher.combine (+ all_reduce)
```

因此后续必须先选定校准目标：

1. 如果目标是校准原始 AIC WideEP 所需算子，则 truth 应对齐
   `routed/compute`，`recorded` 也应该主要服务于真实 workload 下的
   `run_moe_core` 输入分布校准。
2. 如果目标是预测实机 `routed_experts` 端到端 routed path，则 AIC 需要额外补
   一个很薄的 `dispatch/combine` communication/runtime 项。
3. 不建议把 `dispatch/combine` 的误差硬塞给 recorded compute replay，否则会把
   边界问题伪装成 compute policy 问题。

## 2026-07-15 H20 WideEP 口径修正计划

当前先暂停 H100 侧实机工作，优先在 H20 上把 WideEP 的校准口径重新钉牢。

### 新的边界判断

AIC 原始 WideEP `uniform/power_law` 与当前 `recorded` 的核心测量对象都是：

```text
moe_layer.experts.run_moe_core(dispatch_output)
```

collector 侧主 latency 由 CUDA event 包住完整 `run_moe_core(dispatch_output)`：

```text
measurement_boundary = run_moe_core_cuda_critical_path
```

因此 recorded 模式不应该承担 DeepEP `dispatch/combine` 通信校准。通信相关先不纳入
本轮目标，也不对比 `wideep_deepep_normal_perf` / `wideep_deepep_ll_perf`。

实机 profiling 里的当前关系是：

```text
routed_experts = routed/dispatch + routed/compute + routed/combine (+ routed/all_reduce)
routed/compute = run_moe_core 附近的 profiler annotation
```

所以旧的 `PROFILE_STAGE=routed_experts` 对 AIC compute replay 来说确实多算了
`dispatch/combine`。后续 WideEP compute truth 的第一目标，是把实机测量边界
切回与 AIC 原始 `uniform/power_law` 相同的 compute-only 算子数量，而不是先要求
recorded 数值已经校准准确。

### H20 离线 reparse 留痕

未启动新的 GPU 任务，只复用 H20 已有的 ShareGPT WideEP generation low_latency
三 session trace，按 `PROFILE_STAGE=routed/compute` 重新解析：

```text
source:
  aiconfigurator/results/h20_wideep_generation_lowlat_session_median_20260714/sharegpt/wideep_generation_low_latency/

output:
  aiconfigurator/results/h20_wideep_compute_truth_20260715/generation_compute/
  aiconfigurator/results/h20_wideep_compute_truth_20260715/compare/
```

生成的关键 CSV：

```text
aiconfigurator/results/h20_wideep_compute_truth_20260715/compare/h20_sharegpt_wideep_generation_compute_truth.csv
aiconfigurator/results/h20_wideep_compute_truth_20260715/compare/h20_sharegpt_wideep_generation_compute_vs_recorded_detail.csv
aiconfigurator/results/h20_wideep_compute_truth_20260715/compare/h20_sharegpt_wideep_generation_compute_vs_recorded_summary.csv
```

### reparse 结果

若用 `routed/compute` annotation duration 作为 truth，再对比当前 compact
`wideep_generation_moe_perf.txt` 里的 `recorded_eplb/recorded_no_eplb`，AIC 会系统性偏高。
这组对比只用于说明 recorded 目前尚未校准，不作为 compute-only truth 边界错误的证据：

| dataset | phase | stage | EP | eplb | MAPE | median abs | max abs |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| sharegpt | generation | routed/compute | 2 | off | 83.12% | 83.74% | 96.61% |
| sharegpt | generation | routed/compute | 2 | on | 86.33% | 82.83% | 101.80% |
| sharegpt | generation | routed/compute | 4 | off | 73.73% | 75.10% | 82.27% |
| sharegpt | generation | routed/compute | 4 | on | 77.04% | 75.28% | 89.49% |
| sharegpt | generation | routed/compute | 8 | off | 80.35% | 81.71% | 97.75% |
| sharegpt | generation | routed/compute | 8 | on | 82.51% | 82.26% | 96.21% |

典型点：

```text
EP2 on token=288:
  routed/compute truth = 488.195 us
  AIC recorded         = 985.199 us
  error               = +101.80%

EP2 off token=288:
  routed/compute truth = 475.249 us
  AIC recorded         = 934.396 us
  error               = +96.61%
```

同一批点的 AIC/truth 比值中位数约为：

```text
EP2 off: 1.84x
EP2 on : 1.83x
EP4 off: 1.75x
EP4 on : 1.75x
EP8 off: 1.82x
EP8 on : 1.82x
```

这说明当前 `recorded` 还不能直接作为真值校准完成的证据。它本来就是后续要重新设计的
workload replay 方案，不能用它现在的误差反推 `routed/compute` 边界不成立。

补充诊断：对同一个 H20 trace 尝试用 `kernel_external_id` 解析 `routed/compute`，
也不能直接作为 truth。以 ShareGPT / EP2 / eplb off / token=288 / TP0 为例：

```text
rep0:
  annotation_duration layer3/4/5 = 581.389 / 457.590 / 451.683 us
  kernel_external_id  layer3/4/5 = 1728.646 / 1880.327 / 2499.754 us

rep1:
  annotation_duration layer3/4/5 = 646.005 / 460.881 / 453.740 us
  kernel_external_id  layer3/4/5 = 1431.109 / 1720.070 / 2179.017 us

rep2:
  annotation_duration layer3/4/5 = 617.847 / 478.621 / 457.759 us
  kernel_external_id  layer3/4/5 = 1706.726 / 1812.998 / 2342.696 us
```

当前 AIC recorded 对应点约为 `934.396 us`。这只能说明：

- `annotation_duration` 明显偏低，主要反映 CPU launch/annotation span，不保证等于 GPU
  完成时间；
- `kernel_external_id` 明显偏高，容易把同一区间内并发/重叠 kernel duration 求和；
- recorded 现阶段尚未校准，不应该用来决定 truth 的边界。

因此后续不能再用 `routed_experts` 冻结 WideEP compute truth；它应该作为端到端
routed path 观察值保留。WideEP compute truth 应先回到 `routed/compute` /
`run_moe_core` 这一类 compute-only 边界。

### 下一步计划

先不对 AIC recorded policy 做新的校准，也不把通信误差塞进 recorded。当前阶段只做
两件事：定义准 truth 口径，并用多次测量策略压掉抖动。

#### 1. 统一 WideEP truth 边界定义

WideEP truth 拆成两个层级：

```text
compute-only truth:
  routed/compute ~= run_moe_core(dispatch_output)
  用于对齐 AIC 原始 uniform/power_law WideEP 算子数量。

end-to-end routed observation:
  routed_experts = dispatch + compute + combine (+ all_reduce)
  只作为实机端到端观察值，不用于校准 AIC WideEP compute table。
```

DeepEP 通信单独处理：

```text
dispatch/combine:
  本轮不纳入 recorded 责任范围。
  后续若要预测端到端 routed path，再单独看 DeepEP comm/runtime table。
```

#### 2. H20 truth 测试策略

H20 先作为主校准平台，ShareGPT 为主，LongBench 做交叉验证。

WideEP context：

```text
DeepEP mode: normal
truth stage: routed/compute
EP: 2/4/8
EPLB: off/on
sessions: 3
effective repetitions per session: 3
warmup: 每个 session、每个点测量前先 warmup，warmup 不入统计
truth policy: session 内取 median，再跨 3 sessions 取 median
metric: rank_max_us 为主，同时保留 rank_min/rank_mean 诊断
```

WideEP generation：

```text
DeepEP mode: low_latency
truth stage: routed/compute
EP: 2/4/8
EPLB: off/on
sessions: 3
effective repetitions per session: 5
warmup: 每个 session、每个点测量前先 warmup，warmup 不入统计
truth policy: session 内取 median，再跨 3 sessions 取 median
metric: rank_max_us 为主，同时保留 rank_min/rank_mean 诊断
```

同时保留 routed path 拆解诊断：

```text
routed_experts
routed/dispatch
routed/compute
routed/combine
```

这组拆解的用途是证明旧 `routed_experts` truth 多算了通信，而不是用于 recorded 校准。

#### 3. H100 truth 测试策略

H100 用来证明这套边界定义和测试方法的硬件可迁移性，不优先做 H100 专属校准。
当前 H100 实机只有两卡，因此实机 truth 只到 EP2：

```text
WideEP context:
  DeepEP mode: normal
  truth stage: routed/compute
  EP: 2
  EPLB: off/on
  sessions: 3
  effective repetitions per session: 3
  warmup 后统计，session median -> cross-session median

WideEP generation:
  DeepEP mode: low_latency
  truth stage: routed/compute
  EP: 2
  EPLB: off/on
  sessions: 3
  effective repetitions per session: 5
  warmup 后统计，session median -> cross-session median
```

AIC 侧仍可全量生成：

```text
ordinary MoE: EP1/2/4/8/16/32
WideEP MoE: EP2/4/8
```

但 H100 evidence 只比较有实机 truth 的 EP2 点。ordinary MoE 可沿用已有 H100
数据做辅助对照；本轮重点是 WideEP 边界迁移。

#### 4. recorded 后续处理

recorded 目前不作为本阶段 blocker。它现在不准是预期现象，因为它仍带有原先
router/expert replay 的假设，未必能准确复现真实 DeepEP `dispatch_output`
形状。

后续顺序应是：

1. 先冻结 compute-only truth 边界。
2. 再用 AIC 原始 `uniform/power_law` 与 compute-only truth 做“算子数量一致”的证据。
3. 然后重新设计 recorded 的 materialization，让它服务于真实 workload 下的
   `run_moe_core(dispatch_output)`，但不承担 dispatch/combine 通信。
4. 如果未来需要端到端 routed path，再把 compute-only + DeepEP comm/runtime 组合起来，
   不把通信误差硬塞进 recorded compute policy。

## 2026-07-15 H20 routed/compute 小样本稳定性验证

本节记录一次 H20 镜像内实机验证，目的不是校准 recorded，而是确认：

1. `routed/compute` 作为 compute-only truth 边界是否能稳定测出；
2. 旧 `routed_experts` 口径到底是否多算了 `dispatch/combine`；
3. 多 session + warmup + median 的测试策略能否压掉明显抖动。

执行环境：

```text
container: Ai-configurator
model: /model/DeepSeek-V3
dataset: /model/ShareGPT_V3_unfiltered_cleaned_split.json
GPU: H20, CUDA_VISIBLE_DEVICES=0,1
EP/TP: 2/2
DeepEP env: 使用 H20 旧 truth 脚本中的 cpu handler / bond0 / mlx5_bond_* 配置
```

首次未带 `NVSHMEM_IBGDA_NIC_HANDLER=cpu` 时，server 在 NVSHMEM 初始化阶段失败：

```text
nvshmem setup connections failed
nvshmem initialization failed
```

因此 H20 WideEP truth 后续必须继续带旧脚本中的 DeepEP 环境变量，不能直接使用裸
DeepEP 默认环境。

输出目录：

```text
aiconfigurator/results/h20_wideep_compute_boundary_smoke_20260715_cpuenv/sharegpt/
```

### Generation Low-Latency EP2

配置：

```text
phase: generation
DeepEP mode: low_latency
stage: routed/compute
timing source: auto
tokens: 8,128,288,1024
EPLB: off/on
sessions: 3
effective repetitions per session: 5
warmup repetitions: 3
truth policy: session 内 rank_max median，再跨 session median
```

关键文件：

```text
wideep_generation_low_latency/ep2/routed_compute_stability_summary.csv
wideep_generation_low_latency/ep2/stage_breakdown_summary.csv
wideep_generation_low_latency/ep2/stage_breakdown_combo_summary.csv
```

`routed/compute` 稳定性：

| eplb | token | final median us | session medians us | spread | stability |
| --- | ---: | ---: | --- | ---: | --- |
| off | 8 | 483.629 | 484.391;481.231;483.629 | 0.65% | stable |
| off | 128 | 485.950 | 485.950;472.950;487.774 | 3.05% | stable |
| off | 288 | 470.871 | 470.871;467.879;485.801 | 3.81% | stable |
| off | 1024 | 480.264 | 469.416;480.264;487.640 | 3.79% | stable |
| on | 8 | 473.282 | 469.185;476.690;473.282 | 1.59% | stable |
| on | 128 | 477.815 | 469.192;477.815;478.348 | 1.92% | stable |
| on | 288 | 473.286 | 467.202;475.186;473.286 | 1.69% | stable |
| on | 1024 | 472.435 | 470.635;474.399;472.435 | 0.80% | stable |

同一批 trace 的 stage 拆解：

| eplb | token | dispatch us | compute us | combine us | routed_experts us | routed_experts / compute |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 8 | 179.570 | 483.629 | 90.249 | 867.631 | 1.79x |
| off | 128 | 179.160 | 485.950 | 98.317 | 879.545 | 1.81x |
| off | 288 | 173.145 | 470.871 | 90.274 | 849.099 | 1.80x |
| off | 1024 | 175.486 | 480.264 | 93.558 | 842.300 | 1.75x |
| on | 8 | 200.442 | 473.282 | 109.519 | 890.946 | 1.88x |
| on | 128 | 194.071 | 477.815 | 93.406 | 872.694 | 1.83x |
| on | 288 | 192.942 | 473.286 | 92.735 | 869.186 | 1.84x |
| on | 1024 | 194.966 | 472.435 | 93.581 | 857.473 | 1.82x |

结论：

- H20 generation low_latency 的 `routed/compute` 可以稳定测出；
- 跨 3 sessions 的 spread 全部小于 4%，没有看到明显抖动；
- 旧 `routed_experts` 口径约为 compute-only 的 `1.75x~1.88x`，确实多算了
  `dispatch/combine`；
- 因此 WideEP generation compute truth 不应继续使用 `routed_experts` 冻结。

### Context EP2

配置：

```text
phase: context
DeepEP mode: normal
stage: routed/compute
timing source: auto
tokens: 8,128,1024
EPLB: off/on
sessions: 3
effective samples per session: 3
warmup repetitions: 3
truth policy: session 内 rank_max median，再跨 session median
```

关键文件：

```text
wideep_context/ep2/routed_compute_stability_summary.csv
wideep_context/ep2/stage_breakdown_summary.csv
wideep_context/ep2/stage_breakdown_combo_summary.csv
```

`routed/compute` 稳定性：

| eplb | token | final median us | session medians us | spread | stability |
| --- | ---: | ---: | --- | ---: | --- |
| off | 8 | 1375.014 | 1374.211;1375.331;1375.014 | 0.08% | stable |
| off | 128 | 4992.735 | 4991.772;4992.735;4992.957 | 0.02% | stable |
| off | 1024 | 5612.091 | 5612.091;5612.607;5611.294 | 0.02% | stable |
| on | 8 | 1375.779 | 1375.395;1376.547;1375.779 | 0.08% | stable |
| on | 128 | 4995.293 | 4992.447;5012.765;4995.293 | 0.41% | stable |
| on | 1024 | 5609.501 | 5609.501;5610.112;5599.071 | 0.20% | stable |

同一批 trace 的 stage 拆解：

| eplb | token | dispatch us | compute us | combine us | routed_experts us | routed_experts / compute |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 8 | 3.136 | 1375.014 | 0.000 | 1378.467 | 1.00x |
| off | 128 | 8.193 | 4992.735 | 0.000 | 5000.799 | 1.00x |
| off | 1024 | 113.696 | 5612.091 | 0.000 | 5721.851 | 1.02x |
| on | 8 | 1.856 | 1375.779 | 0.000 | 1377.667 | 1.00x |
| on | 128 | 2.208 | 4995.293 | 0.000 | 4997.597 | 1.00x |
| on | 1024 | 93.056 | 5609.501 | 0.000 | 5693.375 | 1.01x |

结论：

- H20 context normal 的 `routed/compute` 也可以稳定测出；
- 跨 3 sessions spread 全部小于 0.5%；
- context 下 `dispatch/combine` 相对 compute 很小，`routed_experts` 与
  `routed/compute` 基本一致；
- 因此过去 context 旧 truth 的口径问题没有 generation low_latency 那么严重，
  但后续为了和 generation 保持统一，仍建议 frozen truth 明确使用
  `routed/compute` compute-only 口径。

### 小样本验证后的下一步

1. H20 上继续按同一方式扩到 EP4/EP8 和全 token 表；
2. ShareGPT 作为主校准数据，LongBench 做交叉验证；
3. 生成新的 compute-only WideEP frozen truth；
4. 旧 `routed_experts` truth 保留为 end-to-end routed observation，不再作为
   AIC WideEP compute 校准目标；
5. 等 compute-only truth 冻结后，再重新设计/校准 recorded materialization。

## 2026-07-15 H20 WideEP compute-only 全量实机 truth 采集

本轮按前面确认的算子边界重新采集 H20 WideEP compute-only truth：

- 实机测量边界：`PROFILE_STAGE=routed/compute`
- 对齐 AIC 原始 WideEP `uniform/power_law` 的核心算子：`moe_layer.experts.run_moe_core(dispatch_output)`
- 不把 DeepEP `dispatch/combine` 通信计入本轮 compute 校准目标
- `context`：DeepEP `normal`
- `generation`：DeepEP `low_latency`
- 数据集：ShareGPT 主校准，LongBench 交叉验证
- EP：2/4/8
- EPLB：off/on
- session：每点 3 sessions
- context：每 session warmup 3 次，每 token 3 个有效 sample，session 内取 median，跨 sessions 再取 median
- generation：每 session warmup 后，每 token 5 次有效 repetition，session 内取 median，跨 sessions 再取 median
- H20 DeepEP 环境：继续使用 `NVSHMEM_IBGDA_NIC_HANDLER=cpu` 以及 bond0/mlx5_bond_0/mlx5_bond_1 相关 NVSHMEM/NCCL/GLOO 环境，否则会出现 NVSHMEM connection setup 失败

产物目录：

```text
aiconfigurator/results/h20_wideep_compute_truth_full_20260715_cpuenv/
```

关键汇总文件：

```text
wideep_context_session_median_summary.csv
wideep_context_unstable_points.csv
wideep_generation_session_median_summary.csv
wideep_generation_unstable_points.csv
```

完整性检查：

```text
rank/per-layer aggregate files total: 144
context files:     72 = 2 datasets * 3 EP * 3 sessions * 2 EPLB * 2 rank/per-layer
generation files:  72 = 2 datasets * 3 EP * 3 sessions * 2 EPLB * 2 rank/per-layer
residual server/profile process: none
GPU memory after run: all 0 MiB
```

### Context 稳定性

Context 全量结果很稳，204 个点全部 stable，没有 mildly unstable/unstable。

| dataset | ep | eplb | points | max spread | avg spread | unstable |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| longbench | 2 | off | 17 | 0.83% | 0.27% | 0 |
| longbench | 2 | on | 17 | 0.64% | 0.25% | 0 |
| longbench | 4 | off | 17 | 1.09% | 0.30% | 0 |
| longbench | 4 | on | 17 | 0.63% | 0.19% | 0 |
| longbench | 8 | off | 17 | 1.07% | 0.39% | 0 |
| longbench | 8 | on | 17 | 0.84% | 0.26% | 0 |
| sharegpt | 2 | off | 17 | 1.23% | 0.30% | 0 |
| sharegpt | 2 | on | 17 | 0.55% | 0.30% | 0 |
| sharegpt | 4 | off | 17 | 0.88% | 0.31% | 0 |
| sharegpt | 4 | on | 17 | 0.81% | 0.36% | 0 |
| sharegpt | 8 | off | 17 | 0.62% | 0.34% | 0 |
| sharegpt | 8 | on | 17 | 0.56% | 0.26% | 0 |

Context 最差点仍然只有 `1.23%` 跨 session spread：

| dataset | ep | eplb | token | final median us | spread | session medians us |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| sharegpt | 2 | off | 16384 | 25119.011 | 1.23% | 24859.462;25119.011;25167.172 |
| longbench | 4 | off | 14336 | 12498.822 | 1.09% | 12442.792;12498.822;12579.273 |
| longbench | 8 | off | 18888 | 7516.585 | 1.07% | 7496.517;7576.872;7516.585 |

结论：H20 WideEP context 的 `routed/compute` 实机 truth 可以按当前多 session median 口径冻结为 compute-only truth 候选。

### Generation 稳定性

Generation 全量已完整采集，但 compute-only low_latency 在 EP8 和少数 tiny-token 点上出现明显跨 session 簇化。这个现象和小样本 smoke 的“generation dispatch/combine 占比很大”不是同一个问题；本轮已经排除了 dispatch/combine，剩下的是 compute-only low_latency 自身在极小绝对时延区域的 session 间模式差异。

| dataset | ep | eplb | points | max spread | avg spread | unstable | mildly unstable |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| longbench | 2 | off | 11 | 73.19% | 15.10% | 2 | 0 |
| longbench | 2 | on | 11 | 3.49% | 1.83% | 0 | 0 |
| longbench | 4 | off | 11 | 18.19% | 5.92% | 0 | 1 |
| longbench | 4 | on | 11 | 41.72% | 7.53% | 1 | 0 |
| longbench | 8 | off | 10 | 50.37% | 30.47% | 6 | 0 |
| longbench | 8 | on | 10 | 51.28% | 18.63% | 3 | 0 |
| sharegpt | 2 | off | 11 | 4.37% | 2.10% | 0 | 0 |
| sharegpt | 2 | on | 11 | 4.20% | 2.63% | 0 | 0 |
| sharegpt | 4 | off | 11 | 8.14% | 4.45% | 0 | 0 |
| sharegpt | 4 | on | 11 | 70.00% | 11.37% | 1 | 0 |
| sharegpt | 8 | off | 10 | 48.26% | 24.39% | 5 | 0 |
| sharegpt | 8 | on | 10 | 45.65% | 25.45% | 6 | 1 |

Generation 当前最明显的点：

| dataset | ep | eplb | token | final median us | spread | stability | session medians us |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| longbench | 2 | off | 4 | 487.268 | 73.19% | unstable | 835.971;487.268;479.339 |
| sharegpt | 4 | on | 4 | 493.806 | 70.00% | unstable | 493.806;490.718;836.402 |
| longbench | 2 | off | 8 | 483.154 | 61.08% | unstable | 773.697;483.154;478.566 |
| longbench | 8 | on | 896 | 563.630 | 51.28% | unstable | 814.195;525.156;563.630 |
| longbench | 8 | off | 288 | 559.530 | 50.37% | unstable | 811.707;529.887;559.530 |
| longbench | 8 | off | 128 | 549.964 | 50.31% | unstable | 536.063;812.756;549.964 |
| longbench | 8 | off | 1024 | 573.805 | 49.92% | unstable | 521.635;808.105;573.805 |
| sharegpt | 8 | off | 512 | 558.385 | 48.26% | unstable | 807.400;537.942;558.385 |

当前判断：

- H20 generation compute-only truth 已经测完，但 EP8 和 tiny-token 区域不能直接按单一 “stable truth” 理解；
- 这些点的绝对值集中在 `~500-850 us`，容易被 low_latency session 级初始化/路径模式放大成很大的百分比 spread；
- 后续冻结 generation truth 时，需要显式定义这些簇化点的处理策略，例如多数簇/中位簇、去除明显 warm path miss、或扩展到更多 sessions 后再定；
- 在策略冻结前，不把 recorded materialization 的误差作为 AIC 校准失败结论。

### Generation unstable 进一步诊断

本次继续拆了 `wideep_generation_session_median_summary.csv` 和原始 trace，结论如下：

1. 不是 `dispatch/combine` 口径问题。
   本轮测量边界已经是 `routed/compute`，没有把 DeepEP 通信计入。

2. 不是 `max_chunk` 聚合造成。
   generation README 里记录为 `generation occurrence aggregation: max_chunk`，但抽查原始 trace 后发现每个 repetition 实际只有 1 个 measured compute occurrence；因此 `max_chunk`、occurrence median、all occurrence median 在这些点上等价。

3. 主要是 `rank_max` 尾巴簇化，而不是全 rank 一起慢。
   EP8 unstable 点里，`rank_max` session spread 很大，但 `rank_mean` 通常只动几个百分点到十几个百分点：

```text
sharegpt ep8 off: session-level rank_max median s1=715.4, s2=538.8, s3=571.1, spread=30.9%
sharegpt ep8 on:  session-level rank_max median s1=790.6, s2=553.0, s3=807.4, spread=32.2%
longbench ep8 off: session-level rank_max median s1=673.9, s2=577.4, s3=562.1, spread=19.4%
longbench ep8 on:  session-level rank_max median s1=689.0, s2=557.3, s3=564.1, spread=23.4%
```

而 rank_mean 维度明显稳定很多：

```text
longbench ep8 off: max rank_mean spread 9.00%
longbench ep8 on:  max rank_mean spread 11.11%
sharegpt ep8 on:   max rank_mean spread 12.85%
```

少数 tiny-token 点连 rank_mean 也会明显变动，例如：

```text
longbench ep2 off token=4: rank_mean spread 36.23%, rank_max spread 73.19%
longbench ep2 off token=8: rank_mean spread 33.71%, rank_max spread 61.08%
sharegpt ep4 on token=4:   rank_mean spread 17.84%, rank_max spread 70.00%
```

4. per-rank trace 抽查显示，EP8 慢簇来自少数 rank 的偶发尾巴，不是固定某张卡一直慢。

示例 1：`sharegpt / ep8 / off / token=512`

```text
session1 rep max values: 810.3, 807.4, 520.7, 531.9, 810.5
session1 median rank_max: 807.4
慢 rank 主要是 r0，但不是所有 repetition 都慢。

session2 rep max values: 805.6, 537.9, 527.3, 555.4, 513.2
session2 median rank_max: 537.9
只有 1/5 次踩到 ~800us 尾巴，因此 session median 仍落在快簇。

session3 rep max values: 830.5, 558.4, 537.1, 830.2, 556.2
session3 median rank_max: 558.4
2/5 次踩到 ~830us 尾巴，但不足以把 session median 拉到慢簇。
```

示例 2：`longbench / ep8 / off / token=288`

```text
session1 rep max values: 811.7, 825.9, 827.6, 550.7, 568.6
session1 median rank_max: 811.7

session2 rep max values: 823.8, 835.6, 516.3, 529.9, 527.0
session2 median rank_max: 529.9

session3 rep max values: 559.5, 550.0, 561.8, 565.3, 551.4
session3 median rank_max: 559.5
```

示例 3：`longbench / ep2 / off / token=4`

```text
session1 rep max values: 826.3, 818.8, 841.5, 837.1, 836.0
session1 median rank_max: 836.0
慢 rank 基本固定在 r1。

session2 rep max values: 537.3, 480.1, 480.1, 487.3, 497.7
session2 median rank_max: 487.3

session3 rep max values: 474.7, 479.3, 469.8, 519.3, 485.5
session3 median rank_max: 479.3
```

当前判断：

- `context` 可以按 `rank_max` 冻结，因为 session spread 全部小于 1.23%；
- `generation low_latency` 的 compute-only `rank_max` 在 EP8 和 tiny-token 点上包含明显的随机尾巴；
- 如果 AIC 目标是校准“compute operator 本体”，`rank_mean`/中位 rank 更接近稳定 compute 基线；
- 如果 AIC 目标是校准“serving critical path”，则必须额外建 tail/envelope 策略，不能把随机 `rank_max` 尾巴直接混进 recorded compute materialization；
- 下一步建议先把 generation truth 拆成两层：`compute_base` 使用稳定的 rank_mean/majority fast cluster，`rank_tail_envelope` 单独记录 rank_max 尾巴，不直接作为 recorded 算子校准目标。

## 2026-07-16 pooled median frozen truth 候选与 AIC 离线验证

### pooled median 口径

新的候选聚合口径：

- generation：不再先取 session median，再跨 session median；改为把 3 sessions * 5 repetitions 的 15 个有效测量值直接 pooled 后取 median；
- context：当前已采数据是 3 sessions * 3 samples，因此先按 9 个有效测量值直接 pooled 后取 median；如果后续重测，也可以把 context 对齐成 3 sessions * 5 samples；
- 同时保留 `rank_max_pooled_median_us`、`rank_mean_pooled_median_us`、`rank_min_pooled_median_us`、IQR、slow count；
- `rank_max_pooled_median_us` 作为 serving critical-path truth 候选；
- `rank_mean_pooled_median_us` 作为 compute-base 辅助视角，不直接替代 serving rank_max。

离线产物：

```text
aiconfigurator/results/h20_wideep_compute_truth_full_20260715_cpuenv/pooled_median_truth_candidate/
  wideep_context_pooled_median_summary.csv
  wideep_generation_pooled_median_summary.csv
  wideep_generation_session_vs_pooled_median_compare.csv
```

初步观察：

- pooled median 能修正一部分 session 投票导致的慢簇问题；
- 例如 `sharegpt / ep8 / off / token=128`，旧 session-median final 是 `818.578us`，pooled median 后是 `578.130us`，下降 `29.37%`；
- `sharegpt / ep8 / on / token=128`，旧值 `808.418us`，pooled median 后 `587.569us`，下降 `27.32%`；
- 但 generation 的 pooled IQR 仍然很高，说明 rank tail 本身仍然存在；pooled median 是更合理的 frozen truth 聚合方式，但不是 recorded compute 校准已经自然解决。

### 当前 AIC recorded 与 pooled truth 的离线 exact-hit 对比

对比输入：

```text
AIC data:
aiconfigurator/src/aiconfigurator/systems/data/h20_sxm/sglang/0.5.9-dsv3/
  wideep_context_moe_perf.txt
  wideep_generation_moe_perf.txt

Truth:
aiconfigurator/results/h20_wideep_compute_truth_full_20260715_cpuenv/pooled_median_truth_candidate/
```

对比只看 exact table-hit，不混入插值/外推。

输出：

```text
aiconfigurator/results/h20_wideep_compute_truth_full_20260715_cpuenv/pooled_median_truth_candidate/aic_recorded_vs_pooled_truth_exact_hits/
  context_exact_hit_detail.csv
  context_exact_hit_summary.csv
  generation_exact_hit_detail.csv
  generation_exact_hit_summary.csv
```

Context 对 `rank_max_pooled_median_us` 的误差可接受：

| dataset | ep | eplb | points | MAPE | median APE | max APE |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| sharegpt | 2 | off | 16 | 4.35% | 2.42% | 12.04% |
| sharegpt | 2 | on | 16 | 3.28% | 2.56% | 10.01% |
| sharegpt | 4 | off | 16 | 4.57% | 1.15% | 22.64% |
| sharegpt | 4 | on | 16 | 4.57% | 1.82% | 22.84% |
| sharegpt | 8 | off | 16 | 3.84% | 2.89% | 9.93% |
| sharegpt | 8 | on | 16 | 3.85% | 3.30% | 10.98% |
| longbench | 2 | off | 16 | 4.35% | 2.40% | 12.07% |
| longbench | 2 | on | 16 | 3.27% | 2.51% | 10.08% |
| longbench | 4 | off | 16 | 4.47% | 1.44% | 22.71% |
| longbench | 4 | on | 16 | 4.43% | 1.87% | 22.90% |
| longbench | 8 | off | 16 | 3.89% | 2.98% | 9.86% |
| longbench | 8 | on | 16 | 3.86% | 3.43% | 10.96% |

Generation 对 `rank_max_pooled_median_us` 的误差非常大，说明当前 AIC recorded generation 与新的 compute-only truth 不是同一个目标：

| dataset | ep | eplb | points | MAPE | median APE | max APE |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| sharegpt | 2 | off | 9 | 81.98% | 79.27% | 94.14% |
| sharegpt | 2 | on | 9 | 87.23% | 79.61% | 106.53% |
| sharegpt | 4 | off | 9 | 77.27% | 76.30% | 85.05% |
| sharegpt | 4 | on | 9 | 92.52% | 83.49% | 177.50% |
| sharegpt | 8 | off | 9 | 89.00% | 83.62% | 141.75% |
| sharegpt | 8 | on | 9 | 85.54% | 89.43% | 175.59% |
| longbench | 2 | off | 9 | 81.07% | 79.87% | 94.97% |
| longbench | 2 | on | 9 | 88.07% | 82.08% | 108.27% |
| longbench | 4 | off | 9 | 81.02% | 82.21% | 88.30% |
| longbench | 4 | on | 9 | 94.87% | 84.29% | 174.69% |
| longbench | 8 | off | 9 | 87.94% | 78.94% | 145.88% |
| longbench | 8 | on | 9 | 97.24% | 90.68% | 152.90% |

结论：

- Context recorded 当前可以先保留，后续只需要处理少数 max APE 超 20% 的点；
- Generation recorded 必须重新设计，不应该继续在现有 recorded materialization 上叠厚 policy；
- 现有 generation recorded 更像旧 router/expert replay 路径的结果，不是新的 `routed/compute` pooled truth 对齐结果。

### AIC WideEP recorded 重新设计计划

原则：

1. 先冻结实机 truth 口径，再重做 AIC recorded；不要在旧 recorded 上继续叠厚 policy。
2. Context 与 generation 分开处理：context 当前路径基本可用；generation 需要重建。
3. WideEP generation 拆成两层：
   - `compute_base`：对齐 pooled median compute truth；
   - `rank_tail_envelope`：单独记录 rank_max 尾巴，不直接混进 compute latency。
4. 删除或旁路旧 generation recorded 的 thin policy，避免旧策略继续把另一条测算路径的误差“修”到新 truth 上。
5. 离线验证先只做 exact-hit，再做插值/外推；通过后也不直接写回系统数据目录，而是先把新逻辑写入 collector/materialization，重跑 AIC 算子生成正式 compact 数据，再用 frozen truth 验证。

具体执行计划：

1. 固化 pooled truth 生成脚本
   - 把当前一次性 pooled median 逻辑沉淀为工具脚本；
   - 输入为 full truth 目录；
   - 输出 context/generation pooled summary、session-vs-pooled compare、tail diagnostics。

2. 重新定义 WideEP generation recorded materialization
   - 不再用旧 router/expert replay 结果直接作为 generation recorded；
   - 以 AIC 原始 `uniform/power_law` synthetic rows 为底座；
   - 用 pooled truth exact-hit 拟合最薄的 phase/EP/eplb 级缩放；
   - tiny token 与 EP8 tail 不做点级厚 patch，只把 tail 写进 diagnostic/envelope。

3. 清理旧 policy
   - 检查 `moe_clean_latency.py` / `build_clean_latency_candidate.py` 中与 `wideep_generation` recorded 相关的局部 patch；
   - 对 generation 旧 thin policy 先旁路，不删除历史代码；
   - context 相关 thin policy 暂时保留，避免破坏当前已较好的 context 对齐。

4. 离线 candidate
   - 生成一个不安装到系统目录的 candidate 数据目录；
   - 比较 `recorded_no_eplb` / `recorded_eplb` 对 pooled truth 的 exact-hit MAPE；
   - ShareGPT 作为主指标，LongBench 作为交叉验证；
   - 目标先设为：context 不退化，generation exact-hit MAPE 显著低于当前 77%-97%。

5. 离线 candidate 通过后的代码落地
   - 把 candidate 中验证通过的新 materialization 逻辑写入 collector 侧代码；
   - 不直接复制离线 candidate 数据到系统目录；
   - 保留旧逻辑的旁路开关或清晰分支，方便回滚和复查。

6. 重跑 AIC 算子
   - 进入 `Ai-configurator` 镜像；
   - 重跑 WideEP 相关 collector，生成正式 compact 输出；
   - 推荐命令形态：

```bash
COLLECTOR_DSV3_KEEP_LATENCY_SOURCES=1 python3 collect.py \
  --backend sglang \
  --model-path deepseek-ai/DeepSeek-V3 \
  --ops moe_token_distribution wideep_moe \
  --keep-csv
```

   - 普通 MoE 暂时沿用旧数据，除非 collector 代码改动影响普通 MoE。

7. 用新 AIC 输出重新验证
   - 只使用重跑 collector 顶层 compact 文件：
     - `wideep_context_moe_perf.txt`
     - `wideep_generation_moe_perf.txt`
   - 与 pooled frozen truth 对比；
   - ShareGPT 作为主指标，LongBench 作为交叉验证；
   - 对比 exact-hit、插值、外推三类点；
   - 输出 detail/summary 到 evidence 目录。

8. 决定是否写回系统数据目录
   - 如果重跑 collector 的 compact 数据通过验证，再写入 `h20_sxm/sglang/0.5.9-dsv3`；
   - 如果离线 candidate 通过但重跑 collector 不通过，说明代码落地或 collector 生成链路仍有差异，不能写回；
   - 如果离线 candidate 本身不通过，则说明 AIC WideEP generation 原始测算路径需要更深的重做，不应通过 policy 修。

### 后续执行留痕要求

每一步都记录到本文档，至少包括：

1. 操作时间；
2. 使用的命令或脚本；
3. 输入目录和输出目录；
4. 关键参数；
5. 产物文件列表；
6. summary 指标；
7. 是否进入下一步的判断。

执行顺序：

1. 固化 pooled truth 生成脚本；
2. 运行脚本复现当前 pooled truth candidate；
3. 实现离线 AIC generation candidate；
4. 离线 exact-hit + 插值/外推验证；
5. 若离线通过，把新逻辑写入 collector/materialization；
6. 在镜像内重跑 `wideep_moe` collector；
7. 用重跑出的 compact 文件与 pooled frozen truth 重新验证；
8. 若通过，再同步到 `h20_sxm/sglang/0.5.9-dsv3`；
9. 最后整理 H20 frozen truth + AIC 校准证据，供 H100 迁移复现。

### Step 1 执行记录：固化 pooled truth 生成脚本

时间：2026-07-16

新增脚本：

```text
aiconfigurator/tools/moe_calibration/summarize_wideep_pooled_truth.py
```

脚本职责：

- 从 full multi-session truth 目录读取 context/generation 的 session parsed CSV；
- generation 直接 pooled `3 sessions * 5 repetitions = 15` 个有效值；
- context 当前直接 pooled `3 sessions * 3 samples = 9` 个有效值；
- 输出 pooled median summary；
- 输出 generation 的 session-median vs pooled-median 对比；
- 不改写历史 parsed CSV。

验证命令：

```bash
python3 aiconfigurator/tools/moe_calibration/summarize_wideep_pooled_truth.py \
  --root aiconfigurator/results/h20_wideep_compute_truth_full_20260715_cpuenv \
  --output-dir aiconfigurator/results/h20_wideep_compute_truth_full_20260715_cpuenv/pooled_median_truth_candidate_from_script
```

输出：

```text
pooled_median_truth_candidate_from_script/
  wideep_context_pooled_median_summary.csv
  wideep_generation_pooled_median_summary.csv
  wideep_generation_session_vs_pooled_median_compare.csv
```

复现检查：

```text
wideep_context_pooled_median_summary.csv: 与临时 one-off 版本一致
wideep_generation_pooled_median_summary.csv: 与临时 one-off 版本一致
```

判断：Step 1 通过，可以进入 Step 2/3，即基于 pooled truth 构建离线 AIC generation candidate。

### Step 2/3 执行记录：离线 generation candidate

时间：2026-07-16

新增脚本：

```text
aiconfigurator/tools/moe_calibration/build_wideep_generation_pooled_candidate.py
```

脚本职责：

- 输入当前系统数据目录和 pooled truth；
- context 文件原样复制，不做改动；
- generation 只对 `recorded_no_eplb` / `recorded_eplb` 做最薄的 EP/eplb 级 scale；
- scale 用 ShareGPT exact table-hit 的 `rank_max_pooled_median_us / AIC latency` 中位数拟合；
- 不写回系统数据目录，只生成离线 candidate。

生成命令：

```bash
python3 aiconfigurator/tools/moe_calibration/build_wideep_generation_pooled_candidate.py \
  --data-dir aiconfigurator/src/aiconfigurator/systems/data/h20_sxm/sglang/0.5.9-dsv3 \
  --truth-dir aiconfigurator/results/h20_wideep_compute_truth_full_20260715_cpuenv/pooled_median_truth_candidate_from_script \
  --output-dir aiconfigurator/results/h20_wideep_compute_truth_full_20260715_cpuenv/offline_aic_candidate_pooled_generation_scale_20260716
```

输出目录：

```text
aiconfigurator/results/h20_wideep_compute_truth_full_20260715_cpuenv/offline_aic_candidate_pooled_generation_scale_20260716/
```

拟合出的 scale：

| ep | eplb | scale |
| ---: | --- | ---: |
| 2 | off | 0.55780884 |
| 2 | on | 0.55674756 |
| 4 | off | 0.56722258 |
| 4 | on | 0.54498639 |
| 8 | off | 0.54459732 |
| 8 | on | 0.52790990 |

### Step 4 执行记录：离线 exact-hit 验证

验证输出：

```text
aiconfigurator/results/h20_wideep_compute_truth_full_20260715_cpuenv/offline_aic_candidate_pooled_generation_scale_20260716/aic_recorded_vs_pooled_truth_exact_hits/
  context_exact_hit_detail.csv
  context_exact_hit_summary.csv
  generation_exact_hit_detail.csv
  generation_exact_hit_summary.csv
```

Context 没有改动，指标与原始 recorded 一致：

| dataset | ep | eplb | points | MAPE | median APE | max APE |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| sharegpt | 2 | off | 16 | 4.35% | 2.42% | 12.04% |
| sharegpt | 2 | on | 16 | 3.28% | 2.56% | 10.01% |
| sharegpt | 4 | off | 16 | 4.57% | 1.15% | 22.64% |
| sharegpt | 4 | on | 16 | 4.57% | 1.82% | 22.84% |
| sharegpt | 8 | off | 16 | 3.84% | 2.89% | 9.93% |
| sharegpt | 8 | on | 16 | 3.85% | 3.30% | 10.98% |
| longbench | 2 | off | 16 | 4.35% | 2.40% | 12.07% |
| longbench | 2 | on | 16 | 3.27% | 2.51% | 10.08% |
| longbench | 4 | off | 16 | 4.47% | 1.44% | 22.71% |
| longbench | 4 | on | 16 | 4.43% | 1.87% | 22.90% |
| longbench | 8 | off | 16 | 3.89% | 2.98% | 9.86% |
| longbench | 8 | on | 16 | 3.86% | 3.43% | 10.96% |

Generation 从原始 recorded 的 77%-97% MAPE 显著下降：

| dataset | ep | eplb | points | MAPE | median APE | max APE |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| sharegpt | 2 | off | 9 | 4.09% | 4.29% | 8.29% |
| sharegpt | 2 | on | 9 | 6.33% | 8.42% | 14.99% |
| sharegpt | 4 | off | 9 | 2.63% | 3.16% | 5.03% |
| sharegpt | 4 | on | 9 | 8.12% | 2.87% | 51.24% |
| sharegpt | 8 | off | 9 | 11.38% | 2.11% | 35.67% |
| sharegpt | 8 | on | 9 | 20.11% | 28.92% | 45.49% |
| longbench | 2 | off | 9 | 3.78% | 3.04% | 8.75% |
| longbench | 2 | on | 9 | 6.77% | 8.23% | 15.95% |
| longbench | 4 | off | 9 | 3.88% | 3.93% | 6.81% |
| longbench | 4 | on | 9 | 7.01% | 1.12% | 49.70% |
| longbench | 8 | off | 9 | 19.47% | 26.84% | 33.90% |
| longbench | 8 | on | 9 | 13.17% | 6.12% | 33.51% |

最差点：

| dataset | ep | eplb | token | truth us | candidate us | APE | signed |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| sharegpt | 4 | on | 8 | 488.892 | 739.383 | 51.24% | +51.24% |
| longbench | 4 | on | 8 | 493.910 | 739.383 | 49.70% | +49.70% |
| sharegpt | 8 | on | 8 | 560.924 | 816.068 | 45.49% | +45.49% |
| sharegpt | 8 | off | 32 | 827.117 | 532.071 | 35.67% | -35.67% |
| sharegpt | 8 | on | 1024 | 553.562 | 744.510 | 34.49% | +34.49% |
| longbench | 8 | off | 8 | 599.805 | 803.163 | 33.90% | +33.90% |

判断：

- EP/eplb 常数缩放方向正确，足以证明当前 generation recorded 主要是尺度错位；
- 但这版离线 candidate 还不能算完全通过，因为 EP8 和少数 tiny-token / EPLB-on 点仍有明显 token 形状误差；
- 下一步不能直接写入 collector，也不能写回系统数据目录；
- 需要继续做一版“薄 token-regime 形状校正”或进一步确认 AIC WideEP generation 原始测算路径是否应该重做；
- 当前 candidate 作为 Step 3/4 证据保留，不进入 Step 5 代码落地。

### Step 4b 执行记录：rank_max serving truth 与 rank_mean compute-base 拆分验证

上一步的 `rank_max` bucket candidate 仍然在 EP8 交叉验证上不稳定：

- ShareGPT EP8/on MAPE `16.01%`，LongBench EP8/on MAPE `25.88%`；
- 最差点仍然在 EP8 rank tail 区域，且 ShareGPT/LongBench 有时方向相反；
- 这说明 pooled `rank_max` serving truth 中仍包含数据集/session 相关 tail，不能直接作为 deterministic AIC recorded compute latency。

因此补做 `rank_mean` compute-base candidate：

命令：

```bash
python3 aiconfigurator/tools/moe_calibration/build_wideep_generation_pooled_candidate.py \
  --data-dir aiconfigurator/src/aiconfigurator/systems/data/h20_sxm/sglang/0.5.9-dsv3 \
  --truth-dir aiconfigurator/results/h20_wideep_compute_truth_full_20260715_cpuenv/pooled_median_truth_candidate_from_script \
  --output-dir aiconfigurator/results/h20_wideep_compute_truth_full_20260715_cpuenv/offline_aic_candidate_pooled_generation_rankmean_split_tail_20260716 \
  --truth-metric rank_mean \
  --token-bucket-scale \
  --bucket-scheme split_tail
```

新增/更新脚本能力：

```text
aiconfigurator/tools/moe_calibration/build_wideep_generation_pooled_candidate.py
  --truth-metric rank_max|rank_mean|rank_min
  --bucket-scheme coarse|split_tail
```

`split_tail` bucket：

```text
tiny_le8
small_32_128
mid_288
tail_512
tail_ge896
```

验证输出：

```text
aiconfigurator/results/h20_wideep_compute_truth_full_20260715_cpuenv/offline_aic_candidate_pooled_generation_rankmean_split_tail_20260716/
  wideep_generation_pooled_scale_report.csv
  wideep_generation_pooled_bucket_scale_report.csv
  aic_recorded_vs_pooled_rankmean_truth_exact_hits/
    generation_rankmean_exact_hit_detail.csv
    generation_rankmean_exact_hit_summary.csv
```

`rank_mean` compute-base exact-hit 指标：

| dataset | ep | eplb | points | MAPE | median APE | max APE |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| sharegpt | 2 | off | 9 | 0.38% | 0.37% | 0.89% |
| sharegpt | 2 | on | 9 | 0.16% | 0.07% | 0.64% |
| sharegpt | 4 | off | 9 | 0.38% | 0.17% | 1.68% |
| sharegpt | 4 | on | 9 | 0.50% | 0.26% | 1.41% |
| sharegpt | 8 | off | 9 | 4.84% | 0.38% | 16.77% |
| sharegpt | 8 | on | 9 | 0.89% | 0.96% | 1.70% |
| longbench | 2 | off | 9 | 1.06% | 1.07% | 2.11% |
| longbench | 2 | on | 9 | 0.57% | 0.41% | 1.26% |
| longbench | 4 | off | 9 | 1.79% | 1.62% | 3.42% |
| longbench | 4 | on | 9 | 0.98% | 1.22% | 2.73% |
| longbench | 8 | off | 9 | 6.97% | 6.62% | 17.92% |
| longbench | 8 | on | 9 | 2.89% | 2.44% | 7.35% |

最差点：

| dataset | ep | eplb | token | truth us | candidate us | APE | signed |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| longbench | 8 | off | 1024 | 546.772 | 448.785 | 17.92% | -17.92% |
| sharegpt | 8 | off | 896 | 522.343 | 609.943 | 16.77% | +16.77% |
| longbench | 8 | off | 896 | 535.465 | 609.943 | 13.91% | +13.91% |
| sharegpt | 8 | off | 32 | 568.263 | 495.028 | 12.89% | -12.89% |
| sharegpt | 8 | off | 1024 | 513.240 | 448.785 | 12.56% | -12.56% |

判断：

- `rank_mean` compute-base 可以被很薄的 EP/eplb + token-bucket scale 校准；
- EP2/EP4 基本已经稳定，LongBench 交叉验证也很好；
- EP8/off 仍有少数 tail token 形状误差，但已经远小于 rank_max serving truth；
- 因此后续 collector/materialization 应该落的是 `compute_base` 逻辑，不应该用 `rank_max` tail 直接重写 recorded compute latency；
- `rank_max` pooled truth 仍保留为 serving critical-path evidence，并在 `rank_tail_envelope` 中单独记录。

是否进入 Step 5：

- 可以进入 Step 5 的前置实现：把 candidate 逻辑以“compute_base generation materialization”的形式落入代码；
- 但正式写回系统数据目录前，必须先重跑 collector，并同时输出 compute-base validation 与 rank-tail envelope 诊断；
- 如果重跑 collector 后 EP8/off 仍类似当前离线指标，可以接受为 compute-base 候选，但不能宣称 rank_max serving tail 已完全校准。

### Step 5 执行记录：WideEP generation compute-base 逻辑落地与离线验证

时间：2026-07-16 02:01 CST。

本步只落地 WideEP generation low_latency 的 compute-base 标定，不修改：

- ordinary MoE；
- WideEP context；
- DeepEP communication table；
- `rank_max` serving tail envelope 口径。

核心判断：

- AIC original WideEP generation 需要对齐的是 routed compute-base；
- H20 compute-only truth 中 `rank_mean` pooled median 更适合作为 compute-base 标定目标；
- `rank_max` pooled median 继续作为 serving critical-path / rank-tail envelope 证据，不直接写入 recorded compute latency。

代码改动：

```text
aiconfigurator/collector/moe_hybrid_policy.py
aiconfigurator/collector/moe_clean_latency.py
```

落地策略：

1. 复用原有 WideEP generation low_latency log model。
2. 在 log model 后追加 H20 ShareGPT 拟合出的 EP/EPLB compute-base scale。
3. 再追加 `split_tail` token bucket residual scale。
4. `moe_hybrid_policy.py` 与 `moe_clean_latency.py` 保持同一组 scale，避免
   `recorded_materialized_source` 与最终 compact 表口径偏移。
5. clean-latency 的 local envelope 跳过条件从精确等于
   `origin_latency_low_latency_log_model` 改为前缀匹配，避免新诊断 label 触发旧
   envelope。

`split_tail` bucket：

```text
tiny_le8
small_32_128
mid_288
tail_512
tail_ge896
```

语法检查：

```bash
python3 -m py_compile \
  aiconfigurator/collector/moe_hybrid_policy.py \
  aiconfigurator/collector/moe_clean_latency.py
```

结果：通过。

离线 clean-latency 重建命令：

```bash
PYTHONPATH=aiconfigurator/collector python3 - <<'PY'
from pathlib import Path
from moe_clean_latency import build_clean_latency_tables

source = Path("aiconfigurator/collector/moe+moe_token_distribution+wideep_moe_20260714_163952_combined_source")
out = Path("aiconfigurator/results/h20_wideep_compute_base_candidate_20260716")
build_clean_latency_tables(source_dir=source, candidate_dir=out, write_origin_dir=True)
print(out)
PY
```

输出目录：

```text
aiconfigurator/results/h20_wideep_compute_base_candidate_20260716/
```

说明：第一次把输出写到包含 `truth` 字样的目录时被 `_truth_guard` 拒绝，这是预期
防呆；正式候选目录改为不含 `truth` 字样。

离线候选 vs H20 pooled `rank_mean` compute-base truth，exact table-hit 指标：

| dataset | ep | eplb | points | MAPE | median APE | max APE | >10% | >20% |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sharegpt | 2 | off | 9 | 0.38% | 0.36% | 0.87% | 0 | 0 |
| sharegpt | 2 | on | 9 | 0.17% | 0.11% | 0.62% | 0 | 0 |
| sharegpt | 4 | off | 9 | 0.39% | 0.18% | 1.67% | 0 | 0 |
| sharegpt | 4 | on | 9 | 0.52% | 0.29% | 1.33% | 0 | 0 |
| sharegpt | 8 | off | 9 | 4.85% | 0.40% | 16.82% | 3 | 0 |
| sharegpt | 8 | on | 9 | 0.89% | 0.99% | 1.69% | 0 | 0 |
| longbench | 2 | off | 9 | 1.06% | 1.05% | 2.09% | 0 | 0 |
| longbench | 2 | on | 9 | 0.58% | 0.42% | 1.29% | 0 | 0 |
| longbench | 4 | off | 9 | 1.82% | 1.63% | 3.44% | 0 | 0 |
| longbench | 4 | on | 9 | 0.99% | 1.27% | 2.76% | 0 | 0 |
| longbench | 8 | off | 9 | 6.96% | 6.58% | 17.89% | 2 | 0 |
| longbench | 8 | on | 9 | 2.90% | 2.40% | 7.35% | 0 | 0 |

最差点：

| dataset | ep | eplb | token | truth us | candidate us | APE | signed |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| longbench | 8 | off | 1024 | 546.772 | 448.936 | 17.89% | -17.89% |
| sharegpt | 8 | off | 896 | 522.343 | 610.184 | 16.82% | +16.82% |
| longbench | 8 | off | 896 | 535.465 | 610.184 | 13.95% | +13.95% |
| sharegpt | 8 | off | 32 | 568.263 | 494.944 | 12.90% | -12.90% |
| sharegpt | 8 | off | 1024 | 513.240 | 448.936 | 12.53% | -12.53% |

结论：

- 代码路径复现了 Step 4b 离线 rank_mean split-tail 候选；
- ShareGPT 主目标 EP2/EP4 已经非常贴近，LongBench 交叉验证也稳定；
- EP8/off tail 仍有 12%-18% 的 compute-base 形状误差，但没有超过 20%，且集中在
  tail / tiny 局部点；
- 可以进入下一步：在镜像中重跑 AIC collector，使用顶层 compact
  `wideep_context_moe_perf.txt` / `wideep_generation_moe_perf.txt` 对 pooled truth
  复验；
- 复验通过前，不写回 `src/aiconfigurator/systems/data/.../0.5.9-dsv3`。

### Step 6 执行记录：镜像内重跑 AIC collector 并复验 compact 输出

时间：2026-07-16 02:03-02:10 CST。

执行位置：H20 本机 `Ai-configurator` 镜像内。

执行前确认：

- 容器内 `/cold/tair-kvcache/aiconfigurator/collector/moe_clean_latency.py`
  已包含 compute-base 新逻辑；
- 无残留 `collect.py` / `run_moe_benchmark` / `sglang.launch_server` 进程；
- 8 张 H20 显存均为空闲。

命令：

```bash
docker exec Ai-configurator bash -lc '
cd /cold/tair-kvcache/aiconfigurator/collector &&
COLLECTOR_DSV3_KEEP_LATENCY_SOURCES=1 \
python3 collect.py --backend sglang --model-path deepseek-ai/DeepSeek-V3 \
  --ops moe_token_distribution wideep_moe --keep-csv
'
```

输出目录：

```text
aiconfigurator/collector/moe_token_distribution+wideep_moe_20260715_180310/
```

结果：

```text
Total errors: 0
wideep_context_moe_perf.txt      49376 bytes
wideep_generation_moe_perf.txt   36434 bytes
moe_token_distribution_perf.txt  614031 bytes
```

clean-latency 行为：

```text
Installed DeepSeek-V3 profile-free Recorded latency into wideep_context_moe_perf.txt
  replaced 102 Recorded rows
Installed DeepSeek-V3 profile-free Recorded latency into wideep_generation_moe_perf.txt
  replaced 54 Recorded rows
DeepSeek-V3 clean latency will install WideEP tables only; ordinary MoE source files are absent.
```

新 compact 中一个 sanity 点：

```text
wideep_generation / EP2 / recorded_no_eplb / token=8
latency = 0.470770 ms
policy  = wideep_generation_low_latency_ep2_noeplb_group_0p958_compute_base_tiny_le8
```

复验输出目录：

```text
aiconfigurator/results/h20_wideep_compute_base_collector_rerun_20260715_180310/
  wideep_context_vs_pooled_rank_max_detail.csv
  wideep_context_vs_pooled_rank_max_summary.csv
  wideep_generation_vs_pooled_rank_mean_detail.csv
  wideep_generation_vs_pooled_rank_mean_summary.csv
```

WideEP generation compact vs H20 pooled `rank_mean` compute-base truth：

| dataset | ep | eplb | points | MAPE | median APE | max APE | >10% | >20% |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sharegpt | 2 | off | 9 | 0.42% | 0.39% | 0.99% | 0 | 0 |
| sharegpt | 2 | on | 9 | 0.21% | 0.14% | 0.52% | 0 | 0 |
| sharegpt | 4 | off | 9 | 0.39% | 0.18% | 1.64% | 0 | 0 |
| sharegpt | 4 | on | 9 | 0.51% | 0.29% | 1.41% | 0 | 0 |
| sharegpt | 8 | off | 9 | 4.84% | 0.40% | 16.81% | 3 | 0 |
| sharegpt | 8 | on | 9 | 0.86% | 0.93% | 1.69% | 0 | 0 |
| longbench | 2 | off | 9 | 1.03% | 0.95% | 1.99% | 0 | 0 |
| longbench | 2 | on | 9 | 0.61% | 0.45% | 1.39% | 0 | 0 |
| longbench | 4 | off | 9 | 1.81% | 1.66% | 3.44% | 0 | 0 |
| longbench | 4 | on | 9 | 0.99% | 1.24% | 2.77% | 0 | 0 |
| longbench | 8 | off | 9 | 6.95% | 6.58% | 17.89% | 2 | 0 |
| longbench | 8 | on | 9 | 2.95% | 2.39% | 7.34% | 0 | 0 |

Generation 最差点：

| dataset | ep | eplb | token | truth us | candidate us | APE | signed |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| longbench | 8 | off | 1024 | 546.772 | 448.967 | 17.89% | -17.89% |
| sharegpt | 8 | off | 896 | 522.343 | 610.171 | 16.81% | +16.81% |
| longbench | 8 | off | 896 | 535.465 | 610.171 | 13.95% | +13.95% |
| sharegpt | 8 | off | 32 | 568.263 | 495.463 | 12.81% | -12.81% |
| sharegpt | 8 | off | 1024 | 513.240 | 448.967 | 12.52% | -12.52% |

判断：

- 新代码经 collector 正式重跑后，顶层 compact generation 表仍然复现
  compute-base 离线候选指标；
- ShareGPT 主目标已经收敛；LongBench 交叉验证没有出现方向性崩坏；
- EP8/off tail 误差保留为诊断项，不继续加厚 policy；
- 本轮可以认为 WideEP generation compute-base collector 逻辑通过 H20 离线与
  重跑验证。

WideEP context 本轮未改逻辑。compact vs pooled `rank_max` context truth：

| dataset | ep | eplb | points | MAPE | median APE | max APE | >10% | >20% |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sharegpt | 2 | off | 16 | 4.48% | 3.00% | 11.99% | 2 | 0 |
| sharegpt | 2 | on | 16 | 5.95% | 2.56% | 29.69% | 2 | 2 |
| sharegpt | 4 | off | 16 | 4.65% | 1.32% | 22.49% | 2 | 1 |
| sharegpt | 4 | on | 16 | 4.61% | 1.78% | 22.33% | 2 | 1 |
| sharegpt | 8 | off | 16 | 3.66% | 2.95% | 10.37% | 1 | 0 |
| sharegpt | 8 | on | 16 | 3.73% | 3.33% | 11.14% | 1 | 0 |

Context 判断：

- context 整体 MAPE 仍在约 4%-6%；
- 大误差主要集中在小 token / EPLB 点，例如 EP2/on/token32、EP2/on/token8、
  EP4/off/on/token32；
- 这部分不是本轮 generation compute-base 逻辑引入，后续如果要继续收敛，可单独做
  context tiny-token thin policy；
- 当前不把 context tiny-token 问题混入 WideEP generation compute-base 结论。

### Step 7 执行记录：WideEP context tiny-token thin policy 离线验证

时间：2026-07-16。

目标：在不影响 generation compute-base 的前提下，只处理 context 中已经定位出的
tiny/small token 局部误差。

改动文件：

```text
aiconfigurator/collector/moe_clean_latency.py
```

策略范围：

- 只作用于 WideEP context；
- 只作用于 `token in {8, 32, 128}`；
- 只按 `EP + EPLB + token` 做很薄的 scale；
- 不修改 WideEP generation；
- 不修改 ordinary MoE；
- 不修改 DeepEP communication table；
- 不处理 EP2/off 的 2048/2560 大 token 边界点。

离线探索：

1. `bucket` 方案：`tiny_le8 / small_32 / small_64_128`
2. `exact_tiny` 方案：只对 `8 / 32 / 128` 做小 token scale

结论：`exact_tiny` 更稳，因为 `64` 和 `128` 放在同一个 bucket 会互相拉扯。
正式 clean-latency 路径的 pre-scale 基线和上一轮 compact 表略有差异，因此最终
factor 按 `build_clean_latency_tables` 的正式路径重新拟合。

离线 candidate 输出：

```text
aiconfigurator/results/h20_wideep_context_generation_compute_base_candidate_v2_20260716/
  wideep_context_moe_perf.txt
  wideep_generation_moe_perf.txt
  validation/
    wideep_context_vs_pooled_rank_max_detail.csv
    wideep_context_vs_pooled_rank_max_summary.csv
    wideep_generation_vs_pooled_rank_mean_detail.csv
    wideep_generation_vs_pooled_rank_mean_summary.csv
```

语法检查：

```bash
python3 -m py_compile aiconfigurator/collector/moe_clean_latency.py
```

结果：通过。

离线重建命令：

```bash
PYTHONPATH=aiconfigurator/collector python3 - <<'PY'
from pathlib import Path
from moe_clean_latency import build_clean_latency_tables

source = Path("aiconfigurator/collector/moe+moe_token_distribution+wideep_moe_20260714_163952_combined_source")
out = Path("aiconfigurator/results/h20_wideep_context_generation_compute_base_candidate_v2_20260716")
build_clean_latency_tables(source_dir=source, candidate_dir=out, write_origin_dir=True)
print(out)
PY
```

WideEP context candidate vs H20 pooled `rank_max` truth：

| dataset | ep | eplb | points | MAPE | median APE | max APE | >10% | >20% |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sharegpt | 2 | off | 16 | 2.89% | 1.04% | 10.44% | 1 | 0 |
| sharegpt | 2 | on | 16 | 1.88% | 1.50% | 6.85% | 0 | 0 |
| sharegpt | 4 | off | 16 | 2.24% | 0.87% | 8.20% | 0 | 0 |
| sharegpt | 4 | on | 16 | 2.27% | 1.53% | 7.90% | 0 | 0 |
| sharegpt | 8 | off | 16 | 2.22% | 1.65% | 6.21% | 0 | 0 |
| sharegpt | 8 | on | 16 | 2.33% | 1.96% | 6.41% | 0 | 0 |
| longbench | 2 | off | 16 | 2.89% | 1.10% | 10.43% | 1 | 0 |
| longbench | 2 | on | 16 | 1.87% | 1.59% | 6.76% | 0 | 0 |
| longbench | 4 | off | 16 | 2.19% | 0.65% | 8.18% | 0 | 0 |
| longbench | 4 | on | 16 | 2.14% | 0.87% | 7.91% | 0 | 0 |
| longbench | 8 | off | 16 | 2.28% | 1.75% | 6.26% | 0 | 0 |
| longbench | 8 | on | 16 | 2.39% | 2.09% | 6.37% | 0 | 0 |

Context 最差点：

| dataset | ep | eplb | token | truth us | candidate us | APE | signed |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| sharegpt | 2 | off | 2560 | 6509.567 | 5830.274 | 10.44% | -10.44% |
| longbench | 2 | off | 2560 | 6509.508 | 5830.274 | 10.43% | -10.43% |
| longbench | 2 | off | 2048 | 6036.949 | 5438.792 | 9.91% | -9.91% |
| sharegpt | 2 | off | 2048 | 6034.038 | 5438.792 | 9.86% | -9.86% |
| sharegpt | 4 | off | 1536 | 3011.042 | 2764.216 | 8.20% | -8.20% |

判断：

- 原先 20%-30% 的 context tiny/small token 点已经消除；
- ShareGPT 和 LongBench 都没有出现新的 tiny-token 反向崩坏；
- 剩余最大误差转移到 EP2/off 的 2560/2048 大 token，幅度约 10%，本轮不继续处理；
- generation compute-base 指标保持不变，说明 context thin policy 没有影响
  generation。

当前未完成事项：

- 还没有重新跑完整 collector 顶层 compact。
- 原因：当前 H20 GPU 被另一组不属于 `Ai-configurator` 容器的
  `sglang::scheduler_DP0_TP*_EP*` 进程占满显存。已确认这些进程不在
  `Ai-configurator` 容器内，本轮不主动 kill/抢占。
- 等 GPU 空闲后，需要再执行一次：

```bash
docker exec Ai-configurator bash -lc '
cd /cold/tair-kvcache/aiconfigurator/collector &&
COLLECTOR_DSV3_KEEP_LATENCY_SOURCES=1 \
python3 collect.py --backend sglang --model-path deepseek-ai/DeepSeek-V3 \
  --ops moe_token_distribution wideep_moe --keep-csv
'
```

重跑完成后，再用 run 顶层 compact 文件复验 context 与 generation。
