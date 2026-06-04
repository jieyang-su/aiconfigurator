# Wan2.2 RTX6000PRO 仿真执行与对比

本文记录 Wan2.2 在 RTX6000PRO 普通服务器与 32 卡 supernode 两种系统配置上的 32 卡并行仿真执行方法、代码修改点和结果结论。执行基于 SGLang collector 数据 `0.5.10.post1-wan2.2-r6000pro-test` 与 NCCL 实测数据 `nccl/sim`。

## 输入数据与系统配置

| 项目 | 普通服务器 | 32 卡 Supernode |
|---|---|---|
| system yaml | `src/aiconfigurator/systems/rtxpro6000_server.yaml` | `src/aiconfigurator/systems/rtxpro6000_supernode.yaml` |
| 单卡算子数据 | `systems/data/rtxpro6000_server/sglang/0.5.10.post1-wan2.2-r6000pro-test` | `systems/data/rtxpro6000_supernode/sglang/0.5.10.post1-wan2.2-r6000pro-test` |
| NCCL 数据 | `systems/data/rtxpro6000_server/nccl/sim/nccl_perf.txt` | `systems/data/rtxpro6000_supernode/nccl/sim/nccl_perf.txt` |
| GPU 组织 | `num_gpus_per_node=2` | `num_gpus_per_node=32` |
| NCCL 版本键 | `misc.nccl_version: sim` | `misc.nccl_version: sim` |

`sglang/0.5.10.post1-wan2.2-r6000pro-test` 下的 Wan/GEMM 文件负责单卡算子查表；`nccl/sim` 负责 AllReduce、AllGather、AllToAll 等 collective 查表；P2P halo/ring 仍走 system YAML 中的 `intra_node_bw/inter_node_bw/p2p_latency` 经验估算。

## 代码修改备忘

| 文件 | 修改点 | 目的 |
|---|---|---|
| `src/aiconfigurator/sdk/operations.py` | `WanParallelComm` 的 NCCL 分支不再强制 `DatabaseMode.EMPIRICAL`；P2P 分支传入 `num_gpus` | NCCL 使用实测表，P2P 保留理论/经验估算 |
| `src/aiconfigurator/sdk/perf_database.py` | `query_p2p(message_bytes, num_gpus=2, ...)` 按 `_get_p2p_bandwidth(num_gpus)` 选择带宽 | 区分普通 server 跨节点 P2P 与 supernode 节点内 P2P |
| `src/aiconfigurator/cli/api.py` | Wan/static 路径把 `context_source_dict` 回填到 `EstimateResult.per_ops_source` | sweep 可以统计 silicon/empirical 来源 |
| `src/aiconfigurator/systems/rtxpro6000_*.yaml` | `misc.nccl_version: sim` | 指向本轮 NCCL 实测数据目录 |
| `src/aiconfigurator/sdk/[self]cli/wan2_2_rtxpro6000_sweep.py` | 新增 32 卡合法 TP/SP/Ulysses/Ring 全组合 sweep | 自动生成分模型日志、JSON 与 CSV 汇总 |

## 执行方法

在仓库 `aiconfigurator` 目录执行：

```bash
source /home/ai_lab/fjw/miniforge3/etc/profile.d/conda.sh
conda activate ljc01
PYTHONPATH=/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src \
  python src/aiconfigurator/sdk/[self]cli/wan2_2_rtxpro6000_sweep.py
```

脚本会遍历：

- 系统：`rtxpro6000_server`、`rtxpro6000_supernode`
- 模型：`T2V-A14B`、`I2V-A14B`、`TI2V-5B`
- 总 GPU 数：固定 `32`
- 合法并行：`tp_size * sp_size = 32`，且 `num_heads % tp_size == 0`、`(num_heads/tp_size) % ulysses_degree == 0`、`sp_size = ulysses_degree * ring_degree`

输出目录形如：

```text
src/aiconfigurator/sdk/[self]cli/logs/wan2_2_rtxpro6000_sweep_YYYYMMDD_HHMMSS/
```

每个系统/模型各有一份 `.log` 与 `.json`，全局汇总为 `summary.json` 和 `summary_table.csv`。

## 本轮执行结果

最新有效结果目录：

```text
src/aiconfigurator/sdk/[self]cli/logs/wan2_2_rtxpro6000_sweep_20260520_134116/
```

验收状态：

| 项目 | 结果 |
|---|---|
| 总任务数 | `2 systems × 3 models × 10 configs = 60` |
| 成功数 | `60/60` |
| 脚本 warning 计数 | `0` |
| 结果表 | `summary_table.csv` |
| 详细结果 | `summary.json` 与各模型 JSON |

## 最优配置对比

| 模型 | 系统 | 最优并行 | request latency ms | tokens/s/gpu | supernode 收益 |
|---|---|---|---:|---:|---:|
| T2V-A14B | server | `tp1/sp32/u8/r4/usp` | `41861.819` | `0.023888` | - |
| T2V-A14B | supernode | `tp1/sp32/u8/r4/usp` | `38660.637` | `0.025866` | `7.65%` latency 降低 |
| I2V-A14B | server | `tp1/sp32/u8/r4/usp` | `42708.414` | `0.023415` | - |
| I2V-A14B | supernode | `tp1/sp32/u8/r4/usp` | `39507.125` | `0.025312` | `7.50%` latency 降低 |
| TI2V-5B | server | `tp1/sp32/u8/r4/usp` | `5112.720` | `0.195591` | - |
| TI2V-5B | supernode | `tp1/sp32/u8/r4/usp` | `4584.432` | `0.218130` | `10.33%` latency 降低 |

三个模型的最优策略一致：`tp=1, sp=32, ulysses=8, ring=4, sp_algorithm=usp`。在当前 RTX6000PRO 数据与 Wan shape 下，高 SP、低 TP 更优；TP 增大虽然降低单卡显存，但引入的 TP 切分与通信收益不足以抵消局部算子效率下降。

## 数据来源验收

最优配置下来源统计：

| 模型 | system | silicon op 数 | empirical op 数 | mixed op 数 |
|---|---|---:|---:|---:|
| T2V-A14B | server/supernode | `78` | `14` | `0` |
| I2V-A14B | server/supernode | `123` | `26` | `0` |
| TI2V-5B | server/supernode | `123` | `26` | `0` |

`silicon` 包含 Wan 单卡算子、GEMM 与 NCCL collective 实测/插值查询；`empirical` 主要是 `wan_dit_ring_attention_kv_p2p` 和 VAE distributed conv height halo P2P。以 T2V 最优配置为例，NCCL 项 `wan_t5_*_all_reduce`、`wan_dit_usp_input_{q,k,v}_alltoall`、`wan_dit_usp_output_alltoall`、`wan_dit_output_sequence_all_gather`、`wan_vae_decode_height_all_gather` 均标记为 `silicon`；P2P 项仍标记为 `empirical`。

## 结论

- 在本轮数据下，`rtxpro6000_supernode` 对三个 Wan2.2 模型均优于 `rtxpro6000_server`，最优 latency 降低约 `7.5%~10.3%`。
- 差异主要来自 32 卡 supernode 对 collective 与 P2P 局部通信更友好；普通 server 的 2 卡节点组织在 SP/Ring/VAE halo 等通信上更容易落到跨节点链路。
- 当前仿真已经使用 RTX6000PRO 单卡算子实测与 NCCL 实测；剩余 empirical 来源是有意保留的 P2P 理论估算，不是 collector 维度缺失。
- 若后续取得 P2P 专项实测，可新增 P2P 表并替换 `query_p2p()` 的 empirical 分支，预计会进一步提高 server/supernode 差异判断的可信度。
