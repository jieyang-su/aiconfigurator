# PerfDatabase 附录：初始化收尾流程解析

**快速概括**：`PerfDatabase` 在完成全部性能表加载之后，还会执行两个非常关键的收尾步骤：`_correct_data()` 与 `_update_support_matrix()`。前者负责用理论下限对已采集性能数据做一致性修正，防止数据库中的经验值低于硬件可达的 `SOL` 下界；后者负责基于已加载数据重建“当前数据库实际支持哪些算子量化模式”的支持矩阵，为后续 `Ops` 查询、校验和分支选择提供统一入口。

这两个步骤并不是简单的初始化尾注，而是决定了后续查询链路是否稳定、是否可诊断、以及支持矩阵是否与真实数据内容一致的核心环节。

## 1. 主要功能概括

1. **`_correct_data()`：数据一致性修正**
   - 以 `SOL` 模式查询结果作为理论基线。
   - 扫描已加载的 `gemm`、`generation_attention` 等关键性能表。
   - 如果数据库中的实测/插值结果低于理论下限，则直接抬升到 `SOL` 值。
   - 只修正 `latency`，保留原始 `power` 字段不变，避免破坏能耗信息。

2. **`_update_support_matrix()`：支持矩阵重建**
   - 根据当前已成功加载的数据，动态生成 `self.supported_quant_mode`。
   - 将不同 backend 下可查询的算子类型、量化模式、数据组织方式统一投影为一张支持矩阵。
   - 处理 backend 差异，例如 `sglang` / `trtllm` / `vllm` 的支持集合并不完全相同。
   - 对某些行为别名做兼容，例如 `trtllm` 下把 `fp8_static` 视作可复用 `fp8` GEMM 表的行为模式。

## 2. `_correct_data()` 的执行过程与作用

`_correct_data()` 位于 `PerfDatabase.__init__()` 的末尾，属于加载完成后的后处理阶段。它的目标不是“补数据”，而是“修正已经加载进来的数据，使其不违反硬件可达性约束”。

### 2.1 为什么需要修正

性能数据库中的数据来源并不总是完全一致：
- 有些数据来自真机测量。
- 有些来自外推插值。
- 有些来自不同 collector、不同版本的归档。

这会导致某些点出现一种不合理情况：**数据库记录的 latency 比理论 `SOL` 还小**。这在物理上通常意味着测量误差、采样噪声、或数据整合过程中的不一致。为避免后续仿真在这些点上产生“超越理论极限”的假象，初始化结束前会统一做一次修正。

### 2.2 修正 GEMM 数据

对 `self._gemm_data` 的修正逻辑是最直接的：

```python
sol = self.query_gemm(m, n, k, quant_mode, database_mode=common.DatabaseMode.SOL)
data = self._gemm_data[quant_mode][m][n][k]
current_latency = data["latency"] if isinstance(data, dict) else data
if sol > current_latency:
    self._gemm_data[quant_mode][m][n][k]["latency"] = float(max(sol, current_latency))
```

这一段体现出三个关键点：

- **修正基准来自 `query_gemm(..., SOL)`**：不是随便写死一个常数，而是根据系统规格里的 `tc_flops` 和 `mem_bw` 动态算出的理论下界。
- **只改 latency，不改 power**：如果原始条目是字典形式，则只替换 `latency`，能耗相关字段保留，这样不会破坏能量信息的可追溯性。
- **兼容旧格式**：如果叶子节点仍是旧版 float 结构，则直接写回 float。

### 2.3 修正 Generation Attention 数据

`generation_attention` 的修正比 GEMM 更复杂，因为其字典层级更深：

```python
for quant_mode in self._generation_attention_data:
    for n_kv in self._generation_attention_data[quant_mode]:
        for head_size in self._generation_attention_data[quant_mode][n_kv]:
            for window_size in self._generation_attention_data[quant_mode][n_kv][head_size]:
                for n in self._generation_attention_data[quant_mode][n_kv][head_size][window_size]:
                    for b in self._generation_attention_data[quant_mode][n_kv][head_size][window_size][n]:
                        for s in self._generation_attention_data[quant_mode][n_kv][head_size][window_size][n][b]:
```

这里说明两件事：

- **Generation Attention 的数据维度非常多**，包含 `quant_mode / n_kv / head_size / window_size / n / b / s` 等维度。
- **修正必须按原始索引逐点执行**，因为不同上下文长度、batch size 和 head 参数都可能落在不同性能区域。

修正时还存在一个特殊分支：

```python
if n_kv == 0:
    n_kv_local = n
else:
    n_kv_local = n_kv
```

这个设计说明数据表里有些条目把 `n_kv=0` 作为一种特殊占位符，实际查询时需要把它还原为真实的 `n`。这属于典型的数据库归一化兼容逻辑。

### 2.4 `_correct_data()` 的总体意义

可以把它理解成两个动作：

1. **对齐物理下界**：保证数据库中的任何性能点都不低于理论极限。
2. **防止后续查询失真**：后续 `Ops.query_*` 不会因为某个异常低值而输出比硬件极限更优的结果。

换句话说，`_correct_data()` 是整个 PerfDatabase 的“物理约束闸门”。

## 3. `_update_support_matrix()` 的执行过程与作用

如果说 `_correct_data()` 负责修正数据本身，那么 `_update_support_matrix()` 负责修正“系统看待这些数据的方式”。它会根据当前已加载的表动态生成 `self.supported_quant_mode`。

### 3.1 支持矩阵的本质

`self.supported_quant_mode` 是一个按算子类型分类的支持列表字典，典型结构类似：

```python
{
    "gemm": ["bfloat16", "fp8", "fp4"],
    "context_attention": ["bfloat16", "fp8"],
    "moe": ["bfloat16", "fp8_block"],
    ...
}
```

它的作用不是直接参与算力计算，而是为以下场景提供基础：

- 查询前的量化模式合法性检查。
- 搜索阶段的候选空间裁剪。
- UI 或 CLI 的支持能力展示。
- backend 不同路径下的数据可用性对齐。

### 3.2 `_enum_key_names()`：把数据顶层键规范化为字符串列表

```python
def _enum_key_names(data: dict | None) -> list[str]:
    if not data:
        return []
    names: list[str] = []
    for key in data:
        names.append(key.name if hasattr(key, "name") else str(key))
    return names
```

这个小工具函数很关键。因为不同加载器返回的数据顶层键可能是：
- `Enum` 类型，比如 `GEMMQuantMode.bfloat16`
- 也可能是字符串
- 或者数据为空 `None`

它把这些情况统一转换成字符串列表，便于最终 `supported_quant_mode` 的结构稳定。

### 3.3 `_merge_key_names()`：把多个数据源的支持项合并

一些算子不是只靠一个数据文件支持，而是多个来源共同构成。例如：
- `context_mla` 同时存在 granular 表和 module-level 表。
- `generation_mla` 需要融合 granular 和 module-level 的 kv-cache dtype 支持情况。

`_merge_key_names()` 会把多个来源的顶层键合并去重后排序，保证支持矩阵是“全局可见”的。

### 3.4 `_generation_mla_kv_modes()`：专门处理 generation MLA 的 kv_cache 维度

这个辅助函数说明了一个非常具体的结构差异：

- **granular generation MLA 数据**：顶层键就是 kv_cache dtype。
- **module-level generation MLA 数据**：kv_cache dtype 位于第二层，需要遍历 `fmha_mode -> kv_mode`。

因此它不能简单用 `_enum_key_names()` 一把抓，而必须同时扫描两个来源后统一合并。

### 3.5 不同 backend 的支持矩阵构建逻辑

#### `sglang`

`sglang` 分支会额外处理 `wideep_context_mla` 和 `wideep_generation_mla`：

- `wideep_context_mla` 的 quant_mode 需要从嵌套的 `kernel_source -> quant_mode` 结构里抽出。
- `wideep_generation_mla` 的 kv-cache dtype 则位于 `kernel_source -> kv_cache_dtype`。

这说明支持矩阵并不是简单的“把数据集顶层键列出来”，而是要理解每种数据表的真实组织方式。

#### `trtllm`

`trtllm` 分支的支持矩阵和 `sglang` 类似，但额外有一条特殊兼容逻辑：

```python
if common.GEMMQuantMode.fp8.name in gemm_modes and common.GEMMQuantMode.fp8_static.name not in gemm_modes:
    gemm_modes.append(common.GEMMQuantMode.fp8_static.name)
```

这表示：
- `fp8_static` 不是独立采集的数据路径。
- 它是一个行为模式，复用 `fp8` 的 GEMM perf table。

这是一个非常典型的“**行为别名**”处理，避免为了一个模式重复采集和维护一套数据库。

#### `vllm`

`vllm` 分支与 `trtllm` 类似，但 `nccl` 支持可能来自 `self._nccl_data` 或 `self._oneccl_data`，体现出不同通信库后端的兼容性。

#### 其他 backend

如果 backend 不在支持范围内，则 `self.supported_quant_mode = {}`。这说明支持矩阵是一个显式的、按 backend 约束的运行时能力视图。

### 3.6 `_update_support_matrix()` 的总体意义

它完成的是“**把加载成功的数据，转换成查询和搜索可直接使用的能力表**”。

换句话说：
- `_correct_data()` 修正的是“数据是否物理合理”。
- `_update_support_matrix()` 修正的是“系统该如何理解这些数据支持了什么”。

两者一个面向数值，一个面向语义。

## 4. 在初始化末尾的整体执行通路

PerfDatabase 初始化末尾的顺序非常关键：

```python
# 1. 读取全部CSV/文本数据并挂载到各个 _xxx_data 属性
# 2. 执行 _correct_data()，修正数据库中的异常低值
# 3. 执行 _update_support_matrix()，重建当前backend可支持的算子/量化模式视图
```

这个顺序不能反过来：

- 如果先更新支持矩阵，再修正数据，那么支持矩阵虽然还能用，但它对应的表内容还没被物理修正完毕。
- 先修正再更新，才能保证“支持矩阵中的能力”对应的是最终版本的数据。

因此，这两个收尾步骤本质上构成了 PerfDatabase 初始化的最后一道一致性闭环。

## 5. `_extrapolate_data_grid()` 的执行过程与作用

在完成原始数据装载后，PerfDatabase 并不会立刻进入 `_correct_data()` 与支持矩阵更新，而是会先对一批核心性能表执行 `_extrapolate_data_grid()`。这一步的本质不是“查询时再补点”，而是**在初始化阶段主动扩展查询网格密度**，把稀疏采样表预先补成更完整的可插值表。

### 5.1 这个函数解决的是什么问题

PerfDatabase 的原始数据文件通常只覆盖了有限的采样点，例如：
- 少量的 `num_heads / batch_size / seq_len` 组合；
- 少量的 `m / n / k` GEMM 形状；
- 少量的 `tp_size / b / s` 组合；
- 某些模型族还会在特定维度上只采到局部区域。

但上层 `Ops.query_*` 的参数空间往往更连续，模型配置和运行时 batch、上下文长度也可能落在“没有被原始采样直接命中”的位置。若完全依赖运行时临时插值，会导致：

- 查询路径更重，插值点更少且边界更脆弱；
- 部分高维表在边角区域容易因为缺少锚点而无法稳定插值；
- 大序列、大 batch、宽 token 维度的覆盖面不足。

因此 `_extrapolate_data_grid()` 的目标是：**在初始化期把数据网格补宽、补密、补长尾**，让后续运行时查询更多落在“已存在或已补全”的网格上。

### 5.2 `_extrapolate_data_grid()` 的核心执行过程

函数签名为：

```python
def _extrapolate_data_grid(
    self,
    data_dict: dict[int, dict[int, dict[int, float]]],
    target_x_list: list[int],
    target_y_list: list[int],
    target_z_list: list[int],
    sqrt_y_value: bool = False,
) -> None:
```

它假定输入的 `data_dict` 是一个三维嵌套字典，形式上类似 `x -> y -> z -> value`。函数内部按 **z 方向 → y 方向 → x 方向** 逐层扩展。

#### （1）先补 z 轴：在同一个 `(x, y)` 截面内扩密度

```python
for x in x_list:
    for y in sorted(data_dict[x].keys()):
        z_dict = data_dict[x][y]
        for z in target_z_list:
            if z not in z_dict:
                z_left, z_right = self._nearest_1d_point_helper(z, list(z_dict.keys()), False)
                value = self._interp_1d([z_left, z_right], [data_dict[x][y][z_left], data_dict[x][y][z_right]], z)
                z_dict[z] = value
```

这一步的作用是：
- 先在每个 `(x, y)` 平面内部，把 z 轴上缺失的点补齐；
- 使用左右边界点做一维插值；
- 插值结果直接写回原字典，后续阶段可以复用这些“补出来的点”继续插值。

这意味着 `_extrapolate_data_grid()` 不是一次性只补原始点之间的空缺，而是会**边补边扩**，逐步让网格变稠密。

#### （2）再补 y 轴：在同一个 x 切片内扩展第二维

```python
for y in target_y_list:
    if y not in data_dict[x]:
        y_left, y_right = self._nearest_1d_point_helper(y, y_keys, False)
        for z in z_list:
            value = self._interp_1d([y_left, y_right], [y_left_value, y_right_value], y)
            data_dict[x][y][z] = value
```

这一阶段利用的是同一 x 下左右两个 y 边界截面上的值，按 z 逐点复制和插值。它的意义在于：
- 把 batch、seq_len、head 等第二维上的采样空洞补上；
- 避免后续查询时因为第二维缺点而直接退化到更重的三维插值；
- 让“同一 x 切片下的曲面”尽量完整。

#### （3）最后补 x 轴：跨不同 x 切片做横向扩展

```python
for x in target_x_list:
    if x not in data_dict:
        x_left, x_right = self._nearest_1d_point_helper(x, x_keys, False)
        for y in sorted(data_dict[x_left].keys()):
            for z in sorted(data_dict[x_left][y].keys()):
                value = self._interp_1d([x_left, x_right], [x_left_value, x_right_value], x)
                data_dict[x][y][z] = value
```

这是最外层的扩展，负责把整个三维网格在 x 维上铺开。它会以左右相邻的 x 截面为锚点，把缺失的切片补出来。

### 5.3 `sqrt_y_value=True` 的特殊含义

这个参数只在少数数据表上开启，典型就是 attention / MLA 这类表。

当 `sqrt_y_value=True` 时，y 方向插值不会直接在线性空间中做，而会先对值开方，再插值，最后再平方回去：

```python
if sqrt_y_value:
    y_left_value = math.sqrt(y_left_value)
    y_right_value = math.sqrt(y_right_value)
    value = self._interp_1d([y_left, y_right], [y_left_value, y_right_value], y)
    value = value * value
```

这样做的目的通常是为了让某些本身更接近“平方增长”或“面积/长度类增长”的表，在 y 维上获得更平滑的外推效果。它不是一般性必需逻辑，而是对特定算子族的经验型拟合修正。

### 5.4 这一步具体用在哪些表上

从初始化代码可以看到，`_extrapolate_data_grid()` 被用于多类核心表：

- `context_attention_data`：补全 `n / s / b` 相关网格；
- `generation_attention_data`：补全 `n / b / s` 网格；
- `gemm_data`：补全 token / vocab 相关的大 GEMM 形状；
- `context_mla_data` 与 `generation_mla_data`：补全 MLA 相关的 `tp / b / s`；
- `context_mla_module_data` 与 `generation_mla_module_data`：补全 module-level MLA 表；
- `context_dsa_module_data` 与 `generation_dsa_module_data`：补全 DSA 相关模块表；
- wideep MLA / DSA 变体表：用于 SGLang / TRTLLM 特化路径。

这说明它不是某一个算子的“特殊修补工具”，而是 PerfDatabase 初始化中的**统一网格加密器**。

### 5.5 结果与后续查询的关系

执行完 `_extrapolate_data_grid()` 后，原本稀疏的数据表会变成更密集的查询图：

- 后续 `query_*` 更容易命中直接字典查表；
- 即使没有完全命中，也更容易在更近的邻域找到插值锚点；
- 大范围长序列、大 batch、极端 head 组合下的查询稳定性更好；
- 最终再配合 `_correct_data()`，保证外推后的值仍不低于理论 `SOL` 下界。

换句话说，`_extrapolate_data_grid()` 负责“把表铺开”，`_correct_data()` 负责“把值校正”，二者共同完成初始化阶段的数据整形。

## 6. 对外部查询链路的影响

这两个函数会直接影响后续的查询与搜索逻辑：

- `Ops.query_*` 进入 PerfDatabase 时，会依赖已经修正过的数据，避免返回低于理论下限的异常结果。
- 搜索器或外部调用者可以通过 `supported_quant_mode` 了解当前后端真正支持哪些量化模式，而无需自己扫描整个数据目录。
- 当某些数据缺失时，后续的 SILICON / HYBRID 逻辑也能更准确地判断是否属于“支持但缺表”还是“根本不支持”。

## 7. 小结

`_correct_data()` 和 `_update_support_matrix()` 是 PerfDatabase 初始化尾部最重要的两步：

- `_correct_data()` 把“原始性能表”修正为“物理上自洽的性能表”。
- `_update_support_matrix()` 把“已加载数据”整理为“可查询能力视图”。

前者保证数值可信，后者保证语义清晰。两者一起构成 PerfDatabase 从数据落盘到运行时查询之间的最后一道桥梁。
