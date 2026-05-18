# DS-V4 Flash Topology Compare 命令参考

本文档汇总以下文件的常用执行命令，方便你在服务器直接复现：

- `tools/run_dsv4flash_topology_compare.sh`
- `tools/extract_final_metrics.py`
- `tools/plot_pareto_compare.py`
- 相关 system yaml 与实验合同文档

## 1) 前置检查

```bash
# 进入仓库
cd /path/to/aiconfigurator

# 确认新增系统文件存在
ls src/aiconfigurator/systems/rtx_pro_6000_scaleup_32.yaml
ls src/aiconfigurator/systems/rtx_pro_6000_scaleout_2x16.yaml

# 查看实验合同
cat docs/perf_database/dsv4flash_topology_compare_contract.md
```

## 2) 启动两组实验（scale-up + scale-out）

```bash
# 启动（默认参数已固定：model/backend/version/quant/SLA/isl/osl/total-gpus）
bash tools/run_dsv4flash_topology_compare.sh start

# 或用自定义输出目录
OUT_DIR=results/dsv4flash_topology_compare_run1 \
  bash tools/run_dsv4flash_topology_compare.sh start
```

## 3) 查看任务状态与等待结束

```bash
# 查看两个后台任务是否仍在运行
bash tools/run_dsv4flash_topology_compare.sh check

# 阻塞等待两个任务完成
bash tools/run_dsv4flash_topology_compare.sh wait
```

## 4) 日志与命令留痕文件

```bash
# 默认输出目录
OUT_DIR=results/dsv4flash_topology_compare

# 任务日志
ls "$OUT_DIR"/output_scaleup_32.log
ls "$OUT_DIR"/output_scaleout_2x16.log

# 自动记录的启动命令
cat "$OUT_DIR"/commands.sh

# PID 记录
cat "$OUT_DIR"/pids.env
```

## 5) 抽取单点对比结果（Phase 3）

```bash
python tools/extract_final_metrics.py \
  --scaleup-log results/dsv4flash_topology_compare/output_scaleup_32.log \
  --scaleout-log results/dsv4flash_topology_compare/output_scaleout_2x16.log \
  --output-csv results/dsv4flash_topology_compare/compare_single_point.csv

# 查看输出
cat results/dsv4flash_topology_compare/compare_single_point.csv
```

提取字段包括：

- `best_throughput`
- `per_gpu_throughput`
- `per_user_throughput`
- `ttft_ms`
- `tpot_ms`
- `request_latency_ms`
- `best_experiment`
- `tp`
- `pp`
- `replicas`
- `bs`

## 6) 画 Pareto 对比图（Phase 4）

> 建议使用 `uv run --frozen` 运行，避免系统 Python 依赖差异。

```bash
uv run --frozen python tools/plot_pareto_compare.py \
  --scaleup-csv results/dsv4flash_topology_compare/scaleup/pareto.csv \
  --scaleout-csv results/dsv4flash_topology_compare/scaleout/pareto.csv \
  --x-col "tokens/s/gpu" \
  --y-col "tokens/s/user" \
  --title "DS-V4 Flash Scale-up vs Scale-out Pareto" \
  --output results/dsv4flash_topology_compare/pareto_compare.png
```

说明：

- 运行 `start` 后，脚本会把两组任务分别写到：
  - `results/dsv4flash_topology_compare/scaleup/`
  - `results/dsv4flash_topology_compare/scaleout/`
- 脚本内部使用 `aiconfigurator cli default --save-dir ...`；AIC 会在该目录下生成随机结果子目录。
- 执行 `wait` 后，脚本会自动把找到的 Pareto 文件复制为标准路径：
  - `results/dsv4flash_topology_compare/scaleup/pareto.csv`
  - `results/dsv4flash_topology_compare/scaleout/pareto.csv`
- 如果 `pareto.csv` 文件名在你的 AIC 版本中不同，可用下面命令定位：
  ```bash
  find results/dsv4flash_topology_compare/scaleup -name "pareto.csv" -o -name "*pareto*.csv"
  find results/dsv4flash_topology_compare/scaleout -name "pareto.csv" -o -name "*pareto*.csv"
  ```
- 脚本支持常见列名自动兼容，不一定必须严格叫 `tokens/s/gpu` 与 `tokens/s/user`。
- 如果你导出的列名不同，先直接跑一次；若报列名错误，再按报错提示改 `--x-col/--y-col`。

## 7) 一组推荐完整流程

```bash
# 1) 启动
bash tools/run_dsv4flash_topology_compare.sh start

# 2) 查看状态（可循环执行）
bash tools/run_dsv4flash_topology_compare.sh check

# 3) 等待完成
bash tools/run_dsv4flash_topology_compare.sh wait

# 4) 抽取对比表
python tools/extract_final_metrics.py \
  --scaleup-log results/dsv4flash_topology_compare/output_scaleup_32.log \
  --scaleout-log results/dsv4flash_topology_compare/output_scaleout_2x16.log \
  --output-csv results/dsv4flash_topology_compare/compare_single_point.csv
```
