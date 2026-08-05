# DeepSeek V3 MLA Prefix 语义审查与迁移说明

## 1. 修改目的

新版 AIC 的 `MLAModule` 已接收 `prefix`，collector 也会执行 prefix 请求并记录
`isl=fresh sequence length`、`step=prefix length`。但原查询链丢弃了 `step`，并把
attention 的二次工作量比例应用到整个 module 时延。这只能算接口层支持 prefix，不能
正确表达 module 内不同 kernel 的规模变化。

本次迁移把“实测 module 精确命中”和“逐算子理论回退”分开处理：有 prefix-aware
module 数据时查询实测表；没有可靠数据覆盖时按各子操作的真实 token 规模回退，不再
对 module 总时延使用统一比例修正。

## 2. Collector 与模型边界

SGLang context module collector 通过 dummy latent QKV 绕过 `qkv_a/downscale`，随后
调用完整 `self_attn`。因此边界为：

```text
module = q_b_proj + kv_b_proj + K concat/cast + attention + o_proj
outside = qkv_a/downscale
```

各部分对 fresh 长度 `s` 和 prefix 长度 `p` 的缩放规律不同：

| 子操作 | 正确规模 |
| --- | --- |
| qkv_a/downscale | `s`，module 外 |
| q_b_proj | `s` |
| kv_b_proj | SGLang prefill 路径为 `s+p` |
| K concat/cast | SGLang prefill 路径为 `s+p` |
| attention score/PV | `(s+p)^2-p^2 = s^2+2sp` |
| o_proj | `s` |

旧公式 `T_module(s+p, 0) * ((s+p)^2-p^2)/(s+p)^2` 会错误缩放 projection、concat、
固定开销，尤其会低估“大 prefix、小 fresh”请求。

## 3. Module 数据路径

`load_context_mla_module_data()` 现在保留 collector 的 `step`：

```text
data[fmha][kv][gemm][local_heads][prefix][fresh_s][batch]
```

这避免相同 `heads/isl/batch`、不同 `step` 的行按 first-source-wins 相互覆盖。module
SILICON 查询使用 `(local_heads, prefix, fresh_s, batch)` 四维原始网格：

1. prefix 精确值可直接查询；
2. prefix 位于已有样本范围内部时允许插值；
3. prefix 超出采样范围时拒绝外推，并抛出 `PerfDataNotAvailableError`；
4. 不再对返回的 latency 或 energy 乘统一 prefix ratio。

缺少 `step` 的旧数据按 `prefix=0` 载入，因此仍兼容无 prefix 查询，但不能用于
`prefix>0`。纯 `ContextMLA` 表仍是 attention kernel 的旧三维
`(heads, full_s, batch)` 语义，其二次项修正是合理的，未改成 module 四维口径。

## 4. Granular 路径

SGLang DeepSeek V3 context fallback 为：

```text
q_b_proj(fresh)
+ kv_b_proj(fresh + prefix)
+ concat_k(fresh + prefix)
+ ContextMLA(fresh, prefix)
+ o_proj(fresh)
```

新增 `ContextKVBProjGEMM` 在父类进行 CP 分片前加入 prefix token；prefix 为标量时按
`batch * prefix` 计算，为逐请求列表时按 `sum(prefix)` 计算。

新增 `MLAConcatK` 只用于已确认语义的 SGLang prefill。对每个 full-K token，它估算：

```text
read K-nope = local_heads * 128 * 2 bytes
read K-rope = 64 * 2 bytes
write K     = local_heads * 192 * 2 bytes
```

该操作通过 `query_mem_op()` 估算时延，并与 KV projection 一样在 CP 下按本 rank token
量处理。vLLM/TRT-LLM 继续使用原普通 GEMM 路径，不继承未经其 collector/后端验证的
SGLang full-K projection 和 concat 假设。

## 5. Module 与 Granular 选择

`FallbackOp` 新增可选的 `silicon_primary_only`，默认关闭，因此不改变其他调用方。
DeepSeek MLA wrapper 将其设为 true：

| Database mode | 行为 |
| --- | --- |
| SILICON | 尝试 prefix-aware module；数据缺失或无可靠 bracket 时回退 granular |
| HYBRID | 使用 SILICON database view 尝试 module；失败后使用原 HYBRID view 查询 granular |
| EMPIRICAL | 直接组合 granular |
| SOL | 直接组合 granular |

每个 shape 都重新尝试 primary，不设置永久 unavailable 状态，也不临时修改共享
`PerfDatabase`。这样一个 prefix miss 不会阻止后续可命中 shape 使用 module 数据。

EMPIRICAL/SOL 绕过 module 是有意设计：当前 module SOL 只描述 attention 主体，尚未
完整覆盖 projections、concat 和固定开销。逐算子组合虽然仍是理论近似，但至少保持了
正确的 fresh/full/attention 工作量边界。

## 6. Downscale 放置

context 和 generation 都采用：

```text
downscale + FallbackOp(module, granular_without_downscale)
```

这与 dummy latent collector 边界一致，并保证 primary 和 fallback 场景均只计算一次
downscale。context 保留 `seq_split=cp`；generation 使用 decode token 数。权重回归也按
“外置 downscale + wrapper”验证，避免因边界调整漏算模型权重。

## 7. 验证与限制

测试覆盖了 prefix loader 防覆盖、module 精确/内部插值/禁止外推、逐 shape fallback
重试、SOL/EMPIRICAL 绕过 module、KV projection 在 CP 前加入 prefix、concat 字节数、
SGLang backend 隔离和 downscale 单次计入。

当前限制如下：

- SGLang 当前数据目录尚无 `mla_context_module_perf.parquet`，实际 SILICON/HYBRID prefix
  请求仍主要依赖 granular 路径；实现为未来 module 数据接入准备了完整 key。
- 四维插值要求数据在 prefix 轴上具有可靠覆盖；本次明确禁止 prefix 外推，避免以不稳健
  的数值结果掩盖数据缺口。
- SGLang full-K `kv_b_proj` 与 concat 语义不自动推广到其他 backend。
- module 自身的 SOL 函数仍不是完整 module 理论模型，因此只能作为内部辅助，DeepSeek
  wrapper 的 SOL/EMPIRICAL 不直接使用它。
- Rust 扩展已在 `ljc01` 中成功构建并通过 `_build_smoke()`、Rust engine 公共 API 和
  crate 单元测试。`ContextKVBProjGEMM`、`MLAConcatK` 目前按主干的 `EXEMPT` 机制明确
  保持在 Python engine：完整 Rust parity 必须原子迁移四维 module 表、
  `silicon_primary_only` 策略及两个 granular op，不能只把新类型序列化后造成静默数值漂移。
  默认 Python 路径可完整执行；显式 Rust 路径遇到该算子图时应通过既有 unsupported 机制
  回退 Python，后续单独完成 MLA Rust parity。
