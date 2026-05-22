# DS-V4 Flash Topology Automation 说明

本文档说明 `tools/dsv4flash_topology_automation.py` 的用途、执行流程、配置项语义、环境变量覆盖规则，以及常见日志排查方法。

## 1. 这个脚本做什么

该脚本用于一键完成两套拓扑对比实验：

- scale-up 系统（由配置中的 `scaleup_system` 指定）
- scale-out 系统（由配置中的 `scaleout_system` 指定）

核心步骤：

1. 读取 JSON 配置。
2. 分别调用 AIC CLI 跑 scale-up 和 scale-out。
3. 收集每个模式（agg/disagg）的结果 CSV。
4. 生成标准化输出文件：
   - `scaleup/pareto_agg.csv`, `scaleup/pareto_disagg.csv`
   - `scaleout/pareto_agg.csv`, `scaleout/pareto_disagg.csv`
5. 画图：
   - 全量点图（all candidates）
   - Pareto 对比图
6. 导出对比单点汇总与最佳配置快照。

## 2. 运行方式

示例：

```bash
python3 tools/dsv4flash_topology_automation.py \
  --config docs/perf_database/dsv4flash_topology_compare_fixed_agg_example.json
```

脚本会在配置中的 `out_dir` 下产生完整结果目录，例如：

- `output_scaleup.log`
- `output_scaleout.log`
- `compare_single_point.csv`
- `best_configs/`
- `plots/`

## 3. 配置文件重点字段

以 `docs/perf_database/dsv4flash_topology_compare_fixed_agg_example.json` 为例：

- 基础实验：
  - `model`, `backend`, `backend_version`
  - `database_mode`
  - `isl`, `osl`, `ttft`, `tpot`
  - `total_gpus`
- 拓扑对比：
  - `scaleup_system`
  - `scaleout_system`
- 输出：
  - `out_dir`
- 图表列：
  - `x_col`, `y_col`
- 搜索空间：
  - `search_parallel`（agg/disagg）
  - 或 `fixed_parallel`（agg/disagg）

## 4. 环境变量与 JSON 的优先级

脚本在 `_run_aic()` 里会构造子进程环境 `env = os.environ.copy()`，然后按 JSON 继续覆盖：

- 若 `debug_comm_queries` 为真，会设置：
  - `AIC_DEBUG_COMM_QUERIES=1`
- 若 `prefer_nccl_for_custom_allreduce` 为真，会设置：
  - `AIC_PREFER_NCCL_FOR_CUSTOM_ALLREDUCE=1`

这意味着：

- 你在 shell 里先写 `AIC_PREFER_NCCL_FOR_CUSTOM_ALLREDUCE=0`，
- 但 JSON 里 `prefer_nccl_for_custom_allreduce: true`，
- 最终子进程仍会被脚本覆盖为 1。

建议：

- 若想禁用该开关，请在 JSON 中设为 `false`。
- 或修改脚本逻辑，改成“仅当环境变量未设置时才从 JSON 写入”。

## 5. 为什么会看到 query_nccl 日志

看到 `query_nccl` 并不等于触发了“custom allreduce -> NCCL 替代”。

原因：

- 很多算子本来就会调用 NCCL（例如 all_gather、reduce_scatter、alltoall）。
- 只有当 `query_custom_allreduce` 出现 `source=prefer_nccl_substitute`，才表示触发了“custom allreduce 替代到 NCCL”。

因此排查时请区分：

1. 普通 NCCL 调用（正常）：
   - `query_nccl ... operation=all_gather/reduce_scatter/alltoall`
2. custom 替代到 NCCL（你关心的）：
   - `query_custom_allreduce ... source=prefer_nccl_substitute`

## 6. 建议的最小排查命令

```bash
# 1) 是否触发 custom->nccl 替代
rg "prefer_nccl_substitute" results/<your_out_dir>/output_*.log

# 2) custom 分支真实来源
rg "query_custom_allreduce.*source=" results/<your_out_dir>/output_*.log

# 3) 普通 NCCL 调用分布
rg "query_nccl" results/<your_out_dir>/output_*.log
```

解释原则：

- 如果没有 `prefer_nccl_substitute`，但有大量 `query_nccl`，通常是正常通信算子路径，不是乱打印。

## 7. 相关文件

- 脚本：`tools/dsv4flash_topology_automation.py`
- 示例配置：`docs/perf_database/dsv4flash_topology_compare_fixed_agg_example.json`
- 合同说明：`docs/perf_database/dsv4flash_topology_compare_contract.md`
- 命令参考：`docs/perf_database/dsv4flash_topology_compare_commands_reference.md`
