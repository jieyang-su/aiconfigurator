# `aiconfigurator/collector` 共性架构分析

## 1. 生态定位

`collector` 是 `aiconfigurator` 推理性能仿真闭环中的物理采样层：它在真实 GPU / XPU 后端上压测 GEMM、Attention、MLA、MoE、Mamba2、GDN 等算子或模块，将耗时和可选功耗写成扁平 CSV 文本。`sdk/perf_database.py` 再按 `PerfDataFilename` 加载这些文件，用查表和插值结果替代理论 SOL 难以刻画的融合 kernel、低比特量化和真实后端调度开销。

本报告只分析三大推理后端 collector：`sglang`、`trtllm`、`vllm`。根目录通信采集、`deep_collector`、`slurm_comm_collector` 不展开。

## 2. 全局控制流框架

### 2.1 注册表驱动

三大后端都通过 `collector.<backend>.registry.REGISTRY` 声明采集单元。每个 `OpEntry` 包含：

| 字段 | 作用 |
| :--- | :--- |
| `op` | CLI / 任务过滤使用的逻辑操作名，例如 `gemm`、`mla_context`、`dsa_generation_module`。 |
| `module` 或 `versions` | 无版本分叉时直接指定模块；有版本分叉时通过 `VersionRoute(min_version, module)` 按运行时版本选择实现。 |
| `get_func` | 返回测试用例列表的函数名。 |
| `run_func` | 执行单个测试用例并写 CSV 的函数名。 |
| `perf_filename` | 标准输出文件名，来自 `registry_types.PerfFile`。 |

`collect.py` 在 `collect_sglang`、`collect_trtllm`、`collect_vllm` 中完成以下动作：

1. 读取框架版本，例如 `sglang` package metadata、`tensorrt_llm.__version__`、`vllm.version.__version__`。
2. 调用 `version_resolver.build_collections()` 将 `REGISTRY` 解析为实际模块。
3. 对每个 collection 动态导入 `get_func` / `run_func`。
4. 用 `functools.partial(run_func, perf_filename=...)` 固定输出文件。
5. 将测试用例切分给 worker 进程，在每个 GPU 设备上执行。

TRT-LLM 和 vLLM 的部分算子存在版本路由：TRT-LLM 的 MLA 与 MoE 按版本选择 `collect_mla_v1/v2`、`collect_moe_v1/v2/v3`；vLLM 的 MoE 和 MLA/DSA module 按版本选择 v1/v2/v3。模块内可声明 `__compat__`，`collect.py` 会在运行时校验。

### 2.2 测试用例生成

`common_test_cases.py` 是跨后端的枚举中心，包含：

| 用例族 | 主要维度 | 典型消费者 |
| :--- | :--- | :--- |
| GEMM | `x/m`、`n`、`k`，覆盖小 batch、临界 `+1`、大矩阵插值点。 | 三后端 GEMM collector。 |
| MoE | `num_tokens`、模型配置、`tp`、`ep`、GPU 数、token 分布。 | 三后端 MoE collector。 |
| MLA | context / generation 的 `batch_size`、`input_len`、`num_heads`、LoRA rank、KV block。 | SGLang/TRT-LLM 细粒度 MLA。 |
| Mamba2 | context/generation、`d_model`、`d_state`、`d_conv`、`nheads`、`chunk_size`。 | TRT-LLM Mamba2。 |
| GDN | context/generation、Qwen3.5 GDN 维度集。 | 三后端 GDN。 |
| MHC / DSV4 Flash | DeepSeek-V4 Flash 的压缩注意力、稀疏 kernel、TP 和量化组合。 | SGLang V4 Flash/MHC collector。 |

测试用例通常通过嵌套 `itertools.product` 生成，并在生成时做合法性剪枝：例如 MoE 要求 `tp * ep == num_gpu`、`num_experts % ep == 0`、`inter_size % tp == 0`；Attention / MLA 限制 `batch_size * seq_len` 上界；低比特路径根据 `get_sm_version()` 过滤，如 FP8 多要求 SM89/90+，NVFP4/MXFP4 多要求 SM100+。

`COLLECTOR_MODEL_PATH` 可以过滤模型配置。`collect.py` 还支持 `--limit`、`--shuffle`、resume checkpoint，把大规模枚举变成可恢复的多进程任务流。

### 2.3 worker 与恢复机制

`collect.py` 的执行层提供：

- 多进程 worker：每个 worker 绑定设备并逐个运行 task，避免单进程长时间持有碎片化 GPU 内存。
- 任务 ID：`helper.create_test_case_id(test_case, test_type, module_name)` 生成稳定 ID。
- Resume checkpoint：记录 `done` / `failed`，支持跳过已完成任务或重试失败任务。
- 进程重启：部分 collector 在重资源任务后通过 `EXIT_CODE_RESTART` 触发 worker 重启，用 OS 回收 CUDA context 和框架私有缓存。
- 日志：主进程和 worker 写 `collector.log`、`collector_errors.log`，并压制部分第三方噪声日志。

## 3. 核心基准测试基础设施

### 3.1 `benchmark_with_power` 行为

`helper.benchmark_with_power()` 是大多数 collector 的统一测量上下文管理器。它封装了如下时序：

1. 读取环境变量：`COLLECTOR_MEASURE_POWER` 控制是否采功耗，`COLLECTOR_POWER_MIN_DURATION` 控制功耗采样最短时长。
2. Warmup：非功耗模式直接 warmup；功耗模式先估算单次耗时，并自适应放大 `num_runs`。
3. CUDA Graph 捕获：默认捕获 `repeat_n` 次 `kernel_func()`，后续通过 `g.replay()` 测量；`allow_graph_fail=True` 时捕获失败回退 eager。
4. 再次 warmup 实际执行路径：图模式 replay，eager 模式直接调用。
5. 功耗采样：用 `PowerMonitor` 后台线程按 100ms 通过 NVML 采 GPU power。
6. 计时：用 CUDA event 包裹 `actual_num_runs` 次 replay / eager。
7. 降频侦测：功耗模式下比较前后 SM clock，下降超过 10% 标记 `throttled=True`。
8. 清理：释放 `CUDAGraph` 和私有内存池，执行 `torch.cuda.empty_cache()`。

返回结果包含 `latency_ms`、`power_stats`、`throttled`、`num_runs_executed`、`used_cuda_graph`。

### 3.2 自适应迭代与功耗

当开启功耗采集时，helper 会估算 `single_iter_time`，目标是让总测量时长至少达到 `power_min_duration`。对极快 kernel，目标时长会收敛到最多 0.3s，`actual_num_runs` 最高限制为 3000，避免为了功耗采样无限增加图 replay 与内存压力。

`PowerMonitor` 使用 NVML 读取当前 device handle、power limit 和 power usage。CSV 中只有当 `power_stats` 非空时才追加 `power` 与 `power_limit` 列；`PerfDatabase` loader 对老数据格式兼容，缺失功耗列时默认 0 或空值。

### 3.3 Cache 与重复执行设计

collector 并非只测单次 Python 调用，而是经常构造“多份 op / 多份输入”的闭包：

- GEMM：`outside_loop_count` 创建多个 Linear / GEMM op，`kernel_func()` 顺序执行后再除以次数，减少 launch 抖动并弱化 L2 cache 命中偏差。
- MoE：SGLang 使用 `outside_loop_count = 5`；vLLM/TRT-LLM 在 power-law 或动态 token 分布下预生成多组 logits / workload，避免一直命中同一地址和同一 routing。
- GDN / Mamba2：默认预生成 input pool，每次取不同 pool entry，减少缓存复用；也可通过环境变量切换 cached input 便于稳定性对比。
- MLA/DSA module：部分 DSA context 显式关闭 CUDA Graph，避免图私有池持有十几 GiB scratch 导致后续任务 OOM。

低比特开销的判断原则：凡是动态量化、scale 计算、routing/topk 等步骤在 `kernel_func()` 内部，就计入表中 latency；若发生在构造阶段、warmup 前或权重预处理阶段，则不计入主测 latency。

## 4. 输出格式与数据库对齐

### 4.1 `log_perf` 基础格式

所有 collector 通过 `helper.log_perf()` 追加 CSV。基础列固定为：

```text
framework,version,device,op_name,kernel_source,...item字段...,power,power_limit
```

其中 `item字段` 由各 collector 的 `item_list[0]` 决定。`log_perf` 使用同名 `.lock` 文件做简易跨进程互斥；文件为空时写 header，非空时追加行并 `fsync()`，适配 NFS 场景。

### 4.2 `PerfFile` 与 `PerfDatabase`

`registry_types.PerfFile` 是 collector 侧规范文件名；`sdk/common.py::PerfDataFilename` 是 SDK 消费侧规范文件名。两者核心值一一对应，例如：

| collector `PerfFile` | 文件名 | `PerfDatabase` loader |
| :--- | :--- | :--- |
| `GEMM` | `gemm_perf.txt` | `load_gemm_data` |
| `CONTEXT_ATTENTION` | `context_attention_perf.txt` | `load_context_attention_data` |
| `GENERATION_ATTENTION` | `generation_attention_perf.txt` | `load_generation_attention_data` |
| `MOE` | `moe_perf.txt` | `load_moe_data` |
| `CONTEXT_MLA` / `GENERATION_MLA` | `context_mla_perf.txt` / `generation_mla_perf.txt` | `load_context_mla_data` / `load_generation_mla_data` |
| `MLA_BMM` | `mla_bmm_perf.txt` | `load_mla_bmm_data` |
| `MLA_CONTEXT_MODULE` / `MLA_GENERATION_MODULE` | `mla_context_module_perf.txt` / `mla_generation_module_perf.txt` | module-level MLA loaders |
| `DSA_CONTEXT_MODULE` / `DSA_GENERATION_MODULE` | `dsa_context_module_perf.txt` / `dsa_generation_module_perf.txt` | module-level DSA loaders |
| `MAMBA2` | `mamba2_perf.txt` | `load_mamba2_data` |
| `GDN` | `gdn_perf.txt` | `load_gdn_data` |
| `MHC_MODULE` | `mhc_module_perf.txt` | `load_mhc_module_data` |
| DSV4 Flash split files | `dsv4_flash_*_module_perf.txt` | DSV4 Flash split loaders and merge逻辑 |

`PerfDatabase.__init__()` 按系统 YAML 定位 `systems/data/<system>/<backend>/<version>/`，加载各 CSV 到嵌套 dict，并在 query 时按形状/量化/模型字段插值。SGLang WideEP 和 TRT-LLM WideEP 有后端专属加载分支。

### 4.3 与上游查询的字段约定

常见字段约定如下：

| 数据表 | 关键字段 |
| :--- | :--- |
| GEMM | `gemm_dtype,m,n,k,latency` |
| Attention | `batch_size,isl,num_heads,num_key_value_heads,head_dim,beam_width,kv_cache_dtype,attn_dtype,step,latency` |
| MLA | `mla_dtype,kv_cache_dtype,num_heads,batch_size,isl,tp_size,step,latency` |
| MLA BMM | `bmm_dtype,num_tokens,num_heads,latency`，用 `op_name` 区分 `mla_gen_pre/post`。 |
| MoE | `moe_dtype,num_tokens,hidden_size,inter_size,topk,num_experts,moe_tp_size,moe_ep_size,distribution,latency` |
| Module MLA/DSA | `model,architecture,mla_dtype,kv_cache_dtype,gemm_type,num_heads,batch_size,isl,tp_size,step,latency` |
| GDN/Mamba2 | `phase,batch_size,seq_len,num_tokens,模型维度...,latency`，用 `kernel_source` 区分子 kernel。 |

这些字段是后端文档逐项核对的主线。
