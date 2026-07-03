# PerfDatabase 代码解析文档

## 1. 快速总结

`PerfDatabase` 是 `aiconfigurator` SDK 中的**硬件性能底层查询引擎**。它的核心职责是基于给定的物理集群系统（System）、调度后端（Backend）和特定版本（Version），读取评测数据（Data Files），从离线的算子评测文件构建内存多维词典。

该类向上提供统一的、具备内存缓存机制（LRU Cache）、并支持高维平滑插值的查询接口（`query_*` 系列函数）。仿真过程中由 `models.py` 构建的各类 `Ops` 执行时，均会访问此数据库通过 `[Batch Size, Sequence Length, Model Dimensions ...]` 等特征参量索引获取其预期在对应架构上执行的时延（Latency）和能量消耗（Energy）。

## 2. 主要功能概括

1. **统一数据管理与生命周期**：根据物理系统配置文件（YAML）指定的数据目录寻址，将散落的各类底层 Benchmark（Gemm、Attention、Comm 集合通信）以及专门设计的大模型算子（MLA、Mamba2、DSA）CSV 数据统一纳入生命周期管控并在对象初始化期完成树形字典装载。
2. **多级性能求值模式 (DatabaseMode)**：
   - `SILICON`：只依赖真机的测绘数据，查询底层实测表格或进行网格插值。
   - `SOL / EMPIRICAL` (Speed of Light/经验折算)：不查询实测文件。依据架构配置文件（`mem_bw`, `tc_flops`）利用理论极限值计算 "数学/显存 bound 时延" 预估（结合 scale_factor 折算效率作为经验模型）。
   - `HYBRID`：混合模式查找，如果硬件采集表数据缺失范围超出可信度，将降级（Fallback）至理论模型计算避免直接崩溃抛出异常。
3. **高维插值与缺失弥补预演 (Extrapolation & Interpolation)**：具备 `1D/2D/3D` 不等间距样条插值或线性插值功能。针对于难以测尽的部分参数排列（例如特定大序列、罕见批量大小），通过邻界值的自动插值拟合预测近似性能结果。
4. **多模态结果对象包裹 (`PerformanceResult`)**：所有返回值统一采用包裹类 `PerformanceResult`。其本质继承扩展了基础 `float` 类型，既可以作为原生的浮点延迟直接参与加减乘除计算时间线混合流（mix_step 的时钟），同时也封装着 `.energy`/`.power` 参数，支撑向功耗维度的仿真扩张。

## 3. 细节深度解析

该组件内函数规模庞大，但主要代码框架由**数据构建映射（Initialization & Loaders）**与**接口共性查询范型（Query API）**构成。

### 3.1 初始化与数据加载流程（深度解析）

#### （1）初始化入口与系统配置读取

```python
def __init__(self, system: str, backend: str, version: str, systems_root: str = "./systems") -> None:
    # Step 1: 保存基本元数据
    self.system = system
    self.backend = backend
    self.version = version
    self.systems_root = systems_root

    # Step 2: 从YAML系统配置文件加载硬件规格
    with open(os.path.join(systems_root, system + ".yaml")) as f:
        self.system_spec = yaml.load(f, Loader=yaml.SafeLoader)
    # 系统规格包含: gpu.mem_bw, gpu.bfloat16_tc_flops 等硬件参数

    # Step 3: 初始化缓存与数据目录定位
    self._default_database_mode = common.DatabaseMode.SILICON
    self._extracted_metrics_cache = {}  # LRU缓存提取的指标数据

    # Step 4: 根据系统规格定位性能数据文件目录
    data_dir = os.path.join(systems_root, self.system_spec["data_dir"], backend, version)
    # 路径格式: systems_root/[data_dir]/[backend]/[version]/[operator_perf_files]
```

**关键点**：系统配置文件（如 `h100.yaml`）中的 `data_dir` 字段指定了性能评测数据的相对路径。这允许同一个系统支持多个后端版本的数据隔离存储。

#### （2）`PerfDataFilename` 枚举与 `func_map` 映射机制

`PerfDataFilename` 是一个枚举类，定义了所有支持的操作类型及其对应的CSV文件名：

```python
class PerfDataFilename(Enum):
    gemm = "gemm_perf.txt"
    context_attention = "context_attention_perf.txt"
    generation_attention = "generation_attention_perf.txt"
    moe = "moe_perf.txt"
    custom_allreduce = "custom_allreduce_perf.txt"
    nccl = "nccl_perf.txt"
    # ... 以及其他40+种操作类型
```

在 `__init__` 函数内定义的 `_load_op_data` 闭包中，存在一个关键的 `func_map` 字典：

```python
def _load_op_data(op_filename_enum: PerfDataFilename) -> LoadedOpData | tuple[LoadedOpData, ...]:
    func_map = {
        PerfDataFilename.gemm: load_gemm_data,              # 函数指针
        PerfDataFilename.context_attention: load_context_attention_data,
        PerfDataFilename.generation_attention: load_generation_attention_data,
        PerfDataFilename.moe: load_moe_data,                # 返回元组（两个值）
        PerfDataFilename.custom_allreduce: load_custom_allreduce_data,
        # ... 其他映射
    }

    # 根据op_filename_enum查找对应的加载函数
    data_filepath = os.path.join(perf_data_dir, op_filename_enum.value)
    data_dict: Optional[dict] = func_map[op_filename_enum](data_filepath)
```

**解耦设计的优势**：
- **单一职责**：每个 `load_*_data` 函数只负责解析特定格式的CSV文件
- **易扩展性**：增加新的操作类型只需编写新的加载函数并添加 `func_map` 条目
- **统一入口**：所有加载操作通过 `_load_op_data` 闭包完成，确保加载过程的一致性

#### （3）CSV文件读取与嵌套字典构建

以 `load_gemm_data` 为例，展示CSV读取与内存树构建的详细过程：

```python
def load_gemm_data(gemm_file):
    """Load GEMM performance data from CSV file"""
    if not os.path.exists(gemm_file):
        logger.debug(f"GEMM data file {gemm_file} not found.")
        return None

    # Step 1: 创建高度嵌套的defaultdict结构，用来存储多维性能参数
    gemm_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))

    # Step 2: 打开CSV文件并逐行读取
    with open(gemm_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)  # 使用字典读取器，自动识别列名
        rows = list(reader)

    # Step 3: 检查后向兼容性（旧数据格式可能没有power列）
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug(f"Legacy database format detected - power will default to 0.0")

    # Step 4: 逐行解析CSV数据
    for row in rows:
        # 从CSV行提取字段并进行类型转换
        quant_mode, m, n, k, latency = (
            row["gemm_dtype"],      # e.g., "bfloat16"
            int(row["m"]),           # 矩阵行数
            int(row["n"]),           # 矩阵列数
            int(row["k"]),           # 内维度
            float(row["latency"]),   # 延迟（毫秒）
        )

        # 读取功率数据（后向兼容）
        power = float(row.get("power", 0.0))

        # 计算能量值 = 功率 × 延迟（瓦特·毫秒）
        energy = power * latency

        # 转换字符串枚举到Python枚举对象
        quant_mode = common.GEMMQuantMode[quant_mode]

        # Step 5: 存储到嵌套字典中，叶子节点是包含latency/power/energy的字典
        gemm_data[quant_mode][m][n][k] = {
            "latency": latency,
            "power": power,
            "energy": energy,
        }

    return gemm_data
```

**数据结构示例**：
```
gemm_data = {
    GEMMQuantMode.bfloat16: {  # 第1层：量化模式
        4096: {                 # 第2层：M维度
            4096: {             # 第3层：N维度
                4096: {         # 第4层：K维度
                    "latency": 2.145,       # 叶子节点：性能指标
                    "power": 350.0,
                    "energy": 750.675,
                }
            }
        }
    }
}
```

**关键特性**：
- 采用 `defaultdict` 链式调用，防止KeyError异常，自动创建缺失层级
- 叶子节点存储为字典，包含 `latency`、`power`、`energy` 三个指标
- 支持后向兼容性：旧数据格式缺少power列时，自动用0.0填充

#### （4）`LoadedOpData` 安全包裹类的设计与作用

`LoadedOpData` 是继承自 `UserDict` 的专用包裹类：

```python
class LoadedOpData(UserDict):
    """Dictionary-like object that tracks source file information"""

    def __init__(self, dict_data: Optional[dict], op_name_enum: PerfDataFilename, filepath: str):
        # 保存元数据而非仅是数据字典本身
        self.op_name_enum = op_name_enum      # 操作类型枚举
        self.filepath = filepath               # CSV文件的物理路径
        self.loaded = dict_data is not None   # 加载成功标志

        super().__init__()
        if dict_data:
            super().update(dict_data)  # 将数据注入到字典中

    def raise_if_not_loaded(self):
        """在任何访问前验证数据是否正确加载"""
        if self.loaded:
            return

        # 生成诊断错误消息
        if not os.path.exists(self.filepath):
            raise PerfDataNotAvailableError(
                f"Error loading silicon data for op {self.op_name_enum}: "
                f"File does not exist at {self.filepath}."
            )
        raise PerfDataNotAvailableError(
            f"Unknown error loading {self.op_name_enum} data from {self.filepath}."
        )

    # 重写关键方法，任何访问都先检查加载状态
    def __getitem__(self, key):
        self.raise_if_not_loaded()  # 防护机制
        return super().__getitem__(key)

    def __contains__(self, key):
        self.raise_if_not_loaded()
        return super().__contains__(key)
```

**包裹类的核心优势**：
- **延迟错误报告**：不在加载时立即抛出异常，而是在实际查询时才报告
- **精准诊断**：当查询失败时，能精确告知哪个CSV文件没有找到或加载失败
- **元数据保留**：保存 `filepath` 和 `op_name_enum`，便于日志记录和调试

#### （5）数据装载到对象属性

在 `__init__` 中，通过大量的赋值语句，将 `LoadedOpData` 对象装载到PerfDatabase实例属性：

```python
# Core ops
self._gemm_data = _load_op_data(PerfDataFilename.gemm)
self._context_attention_data = _load_op_data(PerfDataFilename.context_attention)
self._generation_attention_data = _load_op_data(PerfDataFilename.generation_attention)

# 某些操作返回元组（如moe），需要特殊处理
self._moe_data, self._moe_low_latency_data = _load_op_data(PerfDataFilename.moe)

# Comm ops
self._custom_allreduce_data = _load_op_data(PerfDataFilename.custom_allreduce)
self._nccl_data = _load_op_data(PerfDataFilename.nccl)
```

这些属性在查询时被直接访问，触发 `LoadedOpData.__getitem__()` 的验证逻辑。

#### （6）前沿模型特化处理：DSV4-Flash数据合并

针对DeepSeek V4 Flash Attention根据压缩比分割的数据文件，使用特殊的合并逻辑：

```python
def _load_dsv4_flash_split(loaded_list):
    """Merge split DSV4-Flash module-level data from multiple compress_ratio files"""
    merged: dict = {}
    first_loaded = next((x for x in loaded_list if x is not None), None)
    if first_loaded is None:
        return None

    # 深度合并所有分割文件的数据
    for loaded in loaded_list:
        if loaded is None or not loaded.loaded:
            continue
        _deep_merge_dsv4_dicts(merged, loaded.data)

    if not merged:
        return None

    # 返回合并后的LoadedOpData
    return LoadedOpData(merged, first_loaded.op_name_enum, first_loaded.filepath)

# 分别加载context和generation模式的数据
ctx_split = [
    _load_op_data(PerfDataFilename.dsv4_flash_csa_context_module),      # CSA (ContextSelfAttn)
    _load_op_data(PerfDataFilename.dsv4_flash_hca_context_module),      # HCA (HeadCompressAttn)
]
gen_split = [
    _load_op_data(PerfDataFilename.dsv4_flash_csa_generation_module),
    _load_op_data(PerfDataFilename.dsv4_flash_hca_generation_module),
]

# 合并后的数据对外提供统一接口
self._context_deepseek_v4_attention_module_data = _load_dsv4_flash_split(ctx_split)
```

这样做的目的是**隐藏模型复杂度**：下游的查询函数（如 `query_context_deepseek_v4_attention_module`）无需关心内部数据来自哪个压缩比文件，只需操作单个统一的接口。

### 3.2 运行时查询流程（从Ops调用到Silicon查询）

#### （1）查询函数的统一签名与LRU缓存

```python
@functools.lru_cache(maxsize=32768)  # 第一层缓存：函数级别
def query_gemm(
    self,
    m: int,
    n: int,
    k: int,
    quant_mode: common.GEMMQuantMode,
    database_mode: common.DatabaseMode | None = None,
) -> PerformanceResult:
    """
    Ops执行时调用此接口：
    result = db.query_gemm(m=4096, n=4096, k=4096, quant_mode=GEMMQuantMode.bfloat16)
    """
```

**LRU缓存的作用**：
- 在Transformer推理中，同一层的多个Ops通常具有相同的 `(m, n, k, quant_mode)` 参数组合
- 缓存避免了重复的查表或插值计算，显著提升性能
- 缓存大小32768条记录，基本覆盖所有实际场景

#### （2）三层查询策略（理论 → 数据库 → 插值）

```python
def query_gemm(...) -> PerformanceResult:
    # 内部定义理论计算闭包
    def get_sol(m, n, k, quant_mode) -> tuple[float, float, float]:
        """Speed of Light：基于硬件理论极限计算"""
        tc_flops = self._get_quant_tc_flops(quant_mode)
        # 计算数学bound: 浮点计算的时间 = 操作数/张量核吞吐量
        sol_math = 2 * m * n * k / tc_flops * 1000  # 转换为毫秒
        # 计算访存bound: 内存访问的时间 = 数据量/显存带宽
        sol_mem = quant_mode.value.memory * (m * n + m * k + n * k) / \
                  self.system_spec["gpu"]["mem_bw"] * 1000
        # 实际时间 = max(math_bound, mem_bound)
        sol_time = max(sol_math, sol_mem)
        return sol_time, sol_math, sol_mem

    def get_empirical(m, n, k, quant_mode) -> float:
        """经验折算：使用scale_factor调整SOL估计"""
        sol_time = get_sol(m, n, k, quant_mode)[0]
        scale_factor = 0.8  # 80%的理论峰值
        return sol_time / scale_factor

    # 根据database_mode分支处理
    if database_mode is None:
        database_mode = self._default_database_mode

    # 路径1：纯SOL模式（仅理论计算）
    if database_mode == common.DatabaseMode.SOL:
        sol_time, _, _ = get_sol(m, n, k, quant_mode)
        return PerformanceResult(sol_time, energy=0.0)

    # 路径2：EMPIRICAL模式（理论折算）
    elif database_mode == common.DatabaseMode.EMPIRICAL:
        return PerformanceResult(get_empirical(m, n, k, quant_mode), energy=0.0)

    # 路径3：SILICON或HYBRID模式（数据库查询）
    else:
        def get_silicon():
            """核心：从性能数据库中查询GEMM性能"""
            # 第一步：验证数据是否成功加载
            self._gemm_data.raise_if_not_loaded()

            # 获取规范化的量化模式（某些模式共享数据表）
            table_quant_mode = self._normalize_gemm_quant_mode_for_table(quant_mode)

            # 验证表中是否存在该量化模式
            if table_quant_mode not in self._gemm_data:
                raise PerfDataNotAvailableError(
                    f"GEMM perf data not available for quant_mode='{quant_mode.name}'. "
                    f"Supported: {sorted([k.name for k in self._gemm_data])}"
                )

            gemm_data = self._gemm_data[table_quant_mode]

            # 查询策略1：精确匹配（数据库中恰好存在）
            if m in gemm_data and n in gemm_data[m] and k in gemm_data[m][n]:
                result = gemm_data[m][n][k]  # {'latency': 2.145, 'power': 350.0, 'energy': 750.675}
                return PerformanceResult(result["latency"], energy=result.get("energy", 0.0))

            # 查询策略2：一维插值（M维度不存在，但有相邻的M值）
            m_values = sorted(
                m_key for m_key in gemm_data
                if n in gemm_data[m_key] and k in gemm_data[m_key][n]
            )
            if len(m_values) >= 2:
                # 找M的左右邻界点
                m_left, m_right = self._nearest_1d_point_helper(m, m_values, inner_only=False)
                # 线性插值：result = left_val + (m - m_left) / (m_right - m_left) * (right_val - left_val)
                left_result = gemm_data[m_left][n][k]
                right_result = gemm_data[m_right][n][k]
                result = self._interp_1d(
                    [m_left, m_right],
                    [left_result, right_result],
                    m
                )
                return PerformanceResult(result["latency"], energy=result.get("energy", 0.0))

            # 查询策略3：三维插值（上述两种都失败，使用立方样条或其他高阶插值）
            result = self._interp_3d(m, n, k, gemm_data, method="cubic")
            # result = {'latency': 2.156, 'power': 0.0, 'energy': 0.0}
            return PerformanceResult(result["latency"], energy=result.get("energy", 0.0))

        # 调用统一的SILICON/HYBRID桥接器
        return self._query_silicon_or_hybrid(
            get_silicon=get_silicon,
            get_empirical=lambda: get_empirical(m, n, k, quant_mode),
            database_mode=database_mode,
            error_msg=f"Failed to query gemm data for {m=}, {n=}, {k=}, {quant_mode=}"
        )
```

#### （3）`_query_silicon_or_hybrid` 的HYBRID模式容错机制

```python
def _query_silicon_or_hybrid(
    self,
    get_silicon: Callable[[], PerformanceResult],
    get_empirical: Callable[[], float],
    database_mode: common.DatabaseMode,
    error_msg: str,
) -> PerformanceResult:
    """
    统一的SILICON/HYBRID查询路由器
    """
    try:
        # 尝试使用SILICON模式（从数据库查询）
        return get_silicon()

    except Exception as e:
        # HYBRID模式下，异常被捕获并自动降级
        if database_mode == common.DatabaseMode.HYBRID:
            logger.debug(f"{error_msg} Will try empirical mode.")
            # 从理论模型降级获取性能估计
            return PerformanceResult(get_empirical(), energy=0.0, source="empirical")

        # SILICON模式下，异常被重新抛出（失败快速原则）
        raise
```

**HYBRID模式的应用**：
- 当某个模型/系统/后端组合在数据库中缺失时，不直接崩溃，而是自动使用理论计算
- 用户可以通过 `set_default_database_mode(DatabaseMode.HYBRID)` 启用容错机制
- 便于处理新增模型或不常见的配置组合

#### （4）一维插值和三维插值的递推逻辑

**一维线性插值**：
```python
def _interp_1d(self, x_points: list, y_points: list, x_query: float) -> dict | float:
    """
    Linear interpolation for 1D case
    已知：(x_points[0], y_points[0]) 和 (x_points[1], y_points[1])
    求：x = x_query 对应的y值
    """
    x0, x1 = x_points
    y0, y1 = y_points

    # 线性插值公式：y = y0 + (x - x0) / (x1 - x0) * (y1 - y0)
    alpha = (x_query - x0) / (x1 - x0)

    if isinstance(y0, dict):
        # 对字典中的每个指标分别插值
        result = {}
        for key in y0:
            result[key] = y0[key] + alpha * (y1[key] - y0[key])
        return result
    else:
        # 标量值直接插值
        return y0 + alpha * (y1 - y0)
```

**三维立方样条插值**：
```python
def _interp_3d(self, x: int, y: int, z: int, data: dict, method: str = "cubic") -> dict:
    """
    3D interpolation for cases where exact [x][y][z] doesn't exist
    使用scipy的cubic spline或linear方法
    """
    # Step 1: 从嵌套字典中提取出三维坐标点和对应的latency/energy值
    # 构造类似 points: [(x0,y0,z0), (x0,y0,z1), ...], values: [lat0, lat1, ...]
    points = []
    latencies = []

    for x_key in data:
        for y_key in data[x_key]:
            for z_key in data[x_key][y_key]:
                value = data[x_key][y_key][z_key]
                points.append([x_key, y_key, z_key])
                latencies.append(self._get_value(value, "latency"))

    # Step 2: 使用scipy的插值函数
    if method == "cubic":
        # 构造cubic RBF插值器
        rbf = interpolate.Rbf(
            [p[0] for p in points],
            [p[1] for p in points],
            [p[2] for p in points],
            latencies,
            function='cubic'
        )
    else:  # linear
        rbf = interpolate.Rbf(
            [p[0] for p in points],
            [p[1] for p in points],
            [p[2] for p in points],
            latencies,
            function='linear'
        )

    # Step 3: 查询指定点的插值结果
    interpolated_latency = float(rbf(x, y, z))

    # 对energy也进行同样的插值
    # ...

    return {
        "latency": interpolated_latency,
        "power": 0.0,
        "energy": interpolated_energy
    }
```

#### （5）从Ops调用到Silicon的完整通路示例

```
示例场景：DeepSeekV4推理第5层的GEMM操作

[Ops执行层 (models.py)]
  ↓
GemmOp(m=4096, n=4096, k=4096, quant_mode=bfloat16).execute_time(db)
  ↓
[查询层 (perf_database.py)]
  ↓
database.query_gemm(
    m=4096, n=4096, k=4096,
    quant_mode=GEMMQuantMode.bfloat16,
    database_mode=DatabaseMode.SILICON
)
  ↓
[缓存检查层]
  @lru_cache检查是否曾查询过相同参数 → 缓存命中则返回
  ↓
[策略分支层]
  database_mode == SILICON → 调用 get_silicon()
  ↓
[数据验证层]
  self._gemm_data.raise_if_not_loaded()  ← 检查 'gemm_perf.txt' 是否成功加载
  ↓
[查表层1：精确匹配]
  if 4096 in gemm_data[bfloat16] and 4096 in gemm_data[bfloat16][4096] and 4096 in gemm_data[bfloat16][4096][4096]:
    → result = gemm_data[bfloat16][4096][4096][4096]
    → result = {'latency': 2.145, 'power': 350.0, 'energy': 750.675}
    → return PerformanceResult(2.145, energy=750.675)
  ↓
[查表层2：一维插值]（如果精确匹配失败）
  m_values = [4000, 4100, ...]  # 存在的M值
  m_left=4000, m_right=4100
  result = linear_interp(4000→val1, 4100→val2, query=4096)
  ↓
[查表层3：三维插值]（如果一维插值失败）
  使用cubic RBF基于所有已知点拟合到查询点(4096, 4096, 4096)
  ↓
[返回层]
  → PerformanceResult(2.146, energy=750.812)  ← 延迟可直接用于float运算
  ↓
[Ops消费层]
  self.latency_ms = float(result)  # 2.146
  self.energy_wms = result.energy  # 750.812
  total_time += 2.146
```

### 3.3 数据驱动的查询执行细节对比

| 查询场景 | 执行路径 | 特点 | 性能 |
|---------|---------|------|------|
| 精确匹配 | 直接字典查表 | 一次O(1)查找 | 最快 |
| 一维插值 | 邻界搜索 + 线性插值 | 需要排序+扫描 | 快 |
| 三维插值 | RBF基函数拟合 | 涉及scipy计算 | 慢（但被LRU缓存补偿） |
| 理论估计 | SOL/EMPIRICAL计算 | 无数据库依赖 | 快且鲁棒 |

### 4. 完整端到端执行流程总结

本部分从仿真启动到性能查询的全生命周期，展示PerfDatabase在系统中的实际位置与流量入口。

### 4.1 系统初始化阶段

```python
# 用户代码
from aiconfigurator.sdk import get_database
from aiconfigurator.sdk.models import get_model

# Step 1: 初始化性能数据库
db = get_database(
    system="h100_8gpu",           # 硬件系统
    backend="sglang",             # 调度框架
    version="0.20",               # 版本号
    systems_paths="/path/to/systems"
)
# 在get_database内部：
#   - 定位systems_root/h100_8gpu.yaml
#   - 读取其中的data_dir: "data/benchmark"
#   - 加载data/benchmark/sglang/0.20/下的所有CSV文件
#   - 创建PerfDatabase实例，执行__init__

# Step 2: 从database获取特定模型实例
model = get_model(
    model_name="deepseek-ai/DeepSeek-V4-Flash",
    db=db  # 注入数据库引用
)
# 在模型初始化中，将保存db的引用
```

### 4.2 模型编译与Ops生成阶段

```python
# model.py中的init过程
class DeepSeekV4Model(BaseModel):
    def __init__(self, ..., db):
        self.db = db  # 保存数据库引用

        # 根据配置构建Ops列表
        self.ops = self._build_ops()  # 包含数千个Ops对象

    def _build_ops(self):
        """为模型的每一层、每个计算阶段构建对应的Ops"""
        ops_list = []

        for layer_idx in range(num_layers):
            # 注意力模块Ops
            ctx_attn_ops = self._attention_ops(layer_idx, phase="context")
            # ctx_attn_ops 包含: [attention_pre_ops, qkv_gemm, attn_compute, attn_post_ops]

            # FFN模块Ops
            ffn_ops = self._ffn_ops(layer_idx)
            # ffn_ops 包含: [up_gemm, gate_gemm, down_gemm, ...]

            ops_list.extend(ctx_attn_ops)
            ops_list.extend(ffn_ops)

        return ops_list
```

### 4.3 模型执行与性能查询阶段

```python
# 在仿真引擎中执行模型推理
class InferenceSession:
    def __init__(self, model, db):
        self.model = model
        self.db = db

    def mix_step(self, prefill_tokens, decode_tokens):
        """执行一次混合推理步骤"""
        total_time = 0

        for layer_idx in range(self.model.num_layers):
            # 前缀填充阶段
            if prefill_tokens > 0:
                prefill_attn_latency = self._execute_attn_ops(
                    layer_idx,
                    phase="context",
                    n_tokens=prefill_tokens
                )
                total_time += prefill_attn_latency

            # 解码生成阶段
            if decode_tokens > 0:
                decode_attn_latency = self._execute_attn_ops(
                    layer_idx,
                    phase="generation",
                    n_tokens=decode_tokens
                )
                total_time += decode_attn_latency

        return total_time

    def _execute_attn_ops(self, layer_idx, phase, n_tokens):
        """执行注意力层Ops并查询性能数据"""
        layer_ops = self.model.ops[layer_idx][phase]["attention"]
        total_latency = 0

        for op in layer_ops:
            # Ops.query_ideal()方法触发数据库查询
            op_latency_ms = op.query_ideal(self.db)
            # 这里的op可能是：
            # - GemmOp(m=4096, n=32000, k=4096, ...)
            # - AttentionOp(seq_len=4096, num_heads=128, ...)
            # - CommOp(message_size=16MB, num_gpus=8, ...)

            total_latency += op_latency_ms

        return total_latency
```

### 4.4 Ops查询接口到PerfDatabase的调用链

```python
# operations.py中定义的Op基类
class Op:
    def query_ideal(self, database: PerfDatabase, **kwargs) -> float:
        """
        查询该操作在给定数据库中的理想性能（无调度开销）
        子类重写此方法以访问特定的query_*接口
        """
        return self.db.query_gemm(...)  # 或其他查询方法

class GemmOp(Op):
    def __init__(self, m, n, k, quant_mode):
        self.m = m
        self.n = n
        self.k = k
        self.quant_mode = quant_mode

    def query_ideal(self, database: PerfDatabase, **kwargs) -> float:
        """
        执行流程：
        1. 调用database.query_gemm()
        2. 触发LRU缓存检查
        3. 根据database_mode分支到SILICON/EMPIRICAL/SOL
        4. 在SILICON模式下执行查表/插值
        5. 返回PerformanceResult对象（实现了__float__接口）
        """
        result = database.query_gemm(
            m=self.m,
            n=self.n,
            k=self.k,
            quant_mode=self.quant_mode,
            database_mode=database._default_database_mode
        )
        return float(result)  # 自动转换为浮点数延迟值

class AttentionOp(Op):
    def __init__(self, seq_len, num_heads, head_dim, phase):
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.phase = phase  # "context" 或 "generation"

    def query_ideal(self, database: PerfDatabase, **kwargs) -> float:
        if self.phase == "context":
            return database.query_context_attention(
                seq_len=self.seq_len,
                num_kv_heads=self.num_heads,
                head_size=self.head_dim,
                ...
            )
        else:
            return database.query_generation_attention(
                batch_size=...,
                num_kv_heads=self.num_heads,
                ...
            )
```

### 4.5 缓存与性能优化

```python
# 缓存效果分析
# 场景：DeepSeek V4 推理 32层，每层2个注意力头，批处理大小8

# 不使用缓存：
#   第1层第1头: query_gemm(4096, 4096, 4096) → 进行3D插值 ≈ 10ms
#   第1层第2头: query_gemm(4096, 4096, 4096) → 进行3D插值 ≈ 10ms （重复计算）
#   第2层第1头: query_gemm(4096, 4096, 4096) → 进行3D插值 ≈ 10ms （重复计算）
#   ...
#   总耗时: 32 * 2 * 10ms = 640ms （仅用于查询）

# 使用LRU缓存：
#   第1层第1头: query_gemm(4096, 4096, 4096) → 缓存未命中 → 进行3D插值 ≈ 10ms
#   第1层第2头: query_gemm(4096, 4096, 4096) → 缓存命中 ≈ 0.01ms
#   第2层第1头: query_gemm(4096, 4096, 4096) → 缓存命中 ≈ 0.01ms
#   ...
#   总耗时: 10 + 63 * 0.01 = 10.63ms （性能提升60倍）
```

## 5. 关键约束条件与扩展指南

### 5.1 系统与后端的组合矩阵

PerfDatabase的加载受系统配置和后端约束：

```
系统配置文件 (system.yaml) 必须包含：
├── data_dir: "data/benchmark"
├── gpu:
│   ├── mem_bw: 2000.0  # GB/s (用于SOL计算)
│   ├── bfloat16_tc_flops: 1456.0e12  # FLOPS
│   └── ...
└── misc:
    ├── nccl_version: "2.18"
    └── oneccl_version: "2023.2"

后端目录结构：
systems_root/
  ├── h100_8gpu.yaml
  ├── data/
      ├── benchmark/
          ├── sglang/
          │   ├── 0.19/
          │   │   ├── gemm_perf.txt
          │   │   ├── context_attention_perf.txt
          │   │   ├── ... (其他CSV文件)
          │   │   └── INCOMPLETE.txt (若存在表示该版本未完成)
          │   └── 0.20/
          ├── trtllm/
          └── vllm/
      ├── nccl/
          └── 2.18/
              ├── nccl_perf.txt
```

### 5.2 扩展新操作类型的步骤

若要添加新的操作类型（例如假设新增 `custom_op_perf.txt`）：

**Step 1: 在 `common.py` 中扩展枚举**
```python
class PerfDataFilename(Enum):
    # ... 现有条目
    custom_op = "custom_op_perf.txt"  # 新增
```

**Step 2: 在 `perf_database.py` 中编写加载器**
```python
def load_custom_op_data(custom_op_file):
    """
    Load custom op data from CSV
    应遵循的约定：
    - 返回 dict 或 (dict, dict) 元组（某些操作返回两个字典）
    - 叶子节点应为 {'latency': float, 'power': float, 'energy': float}
    - 使用defaultdict避免KeyError
    - 对CSV缺失、格式错误等异常返回None，由LoadedOpData处理
    """
    if not os.path.exists(custom_op_file):
        logger.debug(f"Custom op file {custom_op_file} not found.")
        return None

    custom_op_data = defaultdict(lambda: defaultdict(...))
    with open(custom_op_file) as f:
        reader = csv.DictReader(f)
        for row in rows:
            # 解析逻辑...
            custom_op_data[key1][key2] = {
                "latency": float(row["latency"]),
                "power": float(row.get("power", 0.0)),
                "energy": float(...),
            }
    return custom_op_data
```

**Step 3: 在 `PerfDatabase.__init__()` 中添加映射**
```python
def _load_op_data(op_filename_enum: PerfDataFilename):
    func_map = {
        # ... 现有映射
        PerfDataFilename.custom_op: load_custom_op_data,  # 新增
    }
    # 剩余逻辑不变
```

**Step 4: 在 `PerfDatabase` 中添加查询方法**
```python
@functools.lru_cache(maxsize=32768)
def query_custom_op(self, param1: int, param2: int, ...) -> PerformanceResult:
    """
    模板化查询方法
    """
    def get_sol(param1, param2, ...):
        # 理论计算逻辑
        return sol_time

    if self._default_database_mode == DatabaseMode.SOL:
        return PerformanceResult(get_sol(...), energy=0.0)

    def get_silicon():
        self._custom_op_data.raise_if_not_loaded()

        # 查表逻辑（精确匹配 → 一维插值 → 三维插值）
        if param1 in self._custom_op_data and ...:
            result = self._custom_op_data[param1][...]
            return PerformanceResult(result["latency"], energy=result.get("energy", 0.0))

        # 插值逻辑...

    return self._query_silicon_or_hybrid(
        get_silicon=get_silicon,
        get_empirical=lambda: get_sol(...),
        database_mode=self._default_database_mode,
        error_msg=f"Failed to query custom_op for {param1=}, {param2=}, ..."
    )
```

### 5.3 性能数据库格式规范

CSV文件的列名约定（以GEMM为例）：

```csv
gemm_dtype,m,n,k,latency,power,energy
bfloat16,4096,4096,4096,2.145,350.0,750.675
bfloat16,4096,4096,8192,4.102,350.0,1435.7
fp8,4096,4096,4096,1.080,180.0,194.4
...
```

**必需列**：`latency`（延迟，单位毫秒）
**可选列**：`power`（功率，瓦特）、`energy`（能量，瓦特·毫秒）

后向兼容性处理：如果CSV中缺少 `power` 和 `energy` 列，加载器自动默认为0.0。

## 6. 总结评述

`PerfDatabase` 是串联 `Hardware Configuration` (提供理论预估基础与文件定位坐标) 与 `Models.py Layer` (提供运行需求与请求传参维度) 之间的 "血脉"。它避免了对所有组合进行枚举评测（极大的测绘工作量），转由优秀的局部网格扩展与插值平滑弥合了 Benchmark 文件中的非连续真空区。

其接口封装高度模式化：
- **数据层面**：通过 `PerfDataFilename` 枚举 + `func_map` 映射 + `LoadedOpData` 包裹，实现了操作类型到CSV加载器的自动分发与错误诊断
- **查询层面**：通过 `@lru_cache` + `_query_silicon_or_hybrid` + 三层查询策略，实现了缓存、容错与递推插值的统一框架
- **扩展层面**：新增操作类型仅需编写 `load_*_data` 函数、添加 `func_map` 条目、实现 `query_*` 方法这三段代码，无需修改核心逻辑

无论扩充何等冷门的高级计算核（如DSV4 Flash Attention、MoE Expert routing等），基本只需按照此模式编写两到三个适配方法即可接入主链路运算流中。这种设计充分体现了Python中"模式优先于规则"的设计哲学，使得系统具备高度的可维护性和可扩展性。
