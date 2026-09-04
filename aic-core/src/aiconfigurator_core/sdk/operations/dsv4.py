# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 family (ISSUE-11 / AIC-1095).

Four op classes migrate from ``_legacy.py`` into ``operations/dsv4.py``:

- ``DeepSeekV4MHCModule`` — manifold-constrained hyper-connection pre/post.
  Owns ``_mhc_module_data``. Delegates to
  ``PerfDatabase.query_mhc_module`` which becomes a one-line forward.
- ``_BaseDeepSeekV4AttentionModule`` — shared weight metadata; not
  instantiated directly. Holds the shared SOL helper used by both
  context and generation phases.
- ``ContextDeepSeekV4AttentionModule`` — context-phase SWA/CSA/HCA. Owns
  ``_context_deepseek_v4_attention_module_data`` (merged from csa+hca
  split files), ``_raw_context_deepseek_v4_attention_module_data``
  (deepcopy used for topk piecewise lookup), and the
  ``_dsv4_sparse_kernel_data`` sidecar dict (paged_mqa_logits)
  used for prefix kernel-Δ correction.
- ``GenerationDeepSeekV4AttentionModule`` — decode-phase. Owns
  ``_generation_deepseek_v4_attention_module_data`` (merged from
  csa+hca split files).

No SOL clamping in the legacy ``_correct_data`` for DSV4 (the per-attn
SOL formula runs inside the query path). No grid extrapolation either —
Interpolation/fallback is handled by the engine's interpolation at query time.

Cache key matches every other migrated op:
``(systems_root, system, backend, version, enable_shared_layer)``.
"""

from __future__ import annotations

import logging
import math
import os
from typing import TYPE_CHECKING, ClassVar

import aiconfigurator_core._aiconfigurator_core as _core
from aiconfigurator_core.sdk.operations.base import OpShellKit, resolve_op_data_path

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from aiconfigurator_core.sdk.perf_database import PerfDatabase


def _cache_key(database: PerfDatabase) -> tuple:
    """Shared cache key — same shape as every other migrated op family."""
    return (
        database.systems_root,
        database.system,
        database.backend,
        database.version,
        database.enable_shared_layer,
    )


# ───────────────────────────────────────────────────────────────────────
# DeepSeekV4MHCModule
# ───────────────────────────────────────────────────────────────────────


class DeepSeekV4MHCModule(_core.DeepSeekV4MHCModule, OpShellKit):
    """DeepSeek-V4 manifold-constrained hyper-connection pre/post module."""

    _data_cache: ClassVar[dict] = {}

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Fetches the engine's mhc_module table view, binds
        ``database._mhc_module_data``."""
        from aiconfigurator_core.sdk.engine_table_view import load_view
        from aiconfigurator_core.sdk.perf_database import PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            cls._data_cache[key] = load_view(database, "_mhc_module_data", PerfDataFilename.mhc_module)
            cls._record_load()

        if "_mhc_module_data" not in database.__dict__:
            database._mhc_module_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_mhc_module)
    # ------------------------------------------------------------------

    @classmethod
    def _query_mhc_table(
        cls,
        database: PerfDatabase,
        num_tokens: int,
        hidden_size: int,
        hc_mult: int,
        sinkhorn_iters: int,
        op: str,
        quant_mode: common.GEMMQuantMode = common.GEMMQuantMode.bfloat16,
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult | tuple[float, float, float]:
        """Verbatim port of legacy ``PerfDatabase.query_mhc_module`` body.

        The SOL estimate models the combined attention-site and FFN-site mHC work
        inside one decoder layer, matching the collector's module boundary.
        """
        # Strict eager resolution (parity with the Rust engine, which resolves
        # flops with `?` at query entry): reject a missing *_tc_flops entry up
        # front — a SILICON exact hit never invokes the get_sol closure.
        common.get_quant_tc_flops(database.system_spec, quant_mode)

        cls.load_data(database)

        sites = 2
        hc_dim = hc_mult * hidden_size
        mix_hc = (2 + hc_mult) * hc_mult

        def get_sol(nt: int = num_tokens, op_name: str = op) -> tuple[float, float, float]:
            pre_ops = sites * (
                2 * nt * hc_dim * mix_hc
                + nt * hc_dim * 3
                + nt * (hc_mult * hc_mult + 2 * hc_mult) * sinkhorn_iters
                + 2 * nt * hc_mult * hidden_size
            )
            post_ops = sites * (2 * nt * hc_mult * hc_mult * hidden_size + 2 * nt * hc_mult * hidden_size)
            if op_name == "pre":
                ops = pre_ops
            elif op_name == "post":
                ops = post_ops
            elif op_name == "both":
                ops = pre_ops + post_ops
            else:
                raise ValueError(f"Unsupported DeepSeek-V4 mHC op: {op_name}")

            param_bytes = sites * (mix_hc * hc_dim + mix_hc + 3) * quant_mode.value.memory
            activation_bytes = sites * nt * hc_dim * quant_mode.value.memory * (3 if op_name == "both" else 2)
            if op_name in {"pre", "both"}:
                activation_bytes += sites * nt * (2 * hc_mult + hc_mult * hc_mult) * 4
            sol_math = ops / common.get_quant_tc_flops(database.system_spec, quant_mode) * 1000
            sol_mem = (param_bytes + activation_bytes) / database.system_spec["gpu"]["mem_bw"] * 1000
            return max(sol_math, sol_mem), sol_math, sol_mem

        def get_empirical() -> float:
            # SOL / util from own num_tokens curve (per op slice); raises if no data.
            mhc_data = getattr(database, "_mhc_module_data", None)

            def _emp_for_op(op_name: str) -> float:
                sol_q = get_sol(num_tokens, op_name)[0]

                def _slice():
                    if not mhc_data:
                        raise PerfDataNotAvailableError("No DeepSeek-V4 mHC data is loaded.")
                    return util_empirical.require_data_slice(mhc_data, op_name, hc_mult, hidden_size)

                grid = util_empirical.grid_for(
                    (
                        "dsv4_mhc",
                        database.system,
                        database.backend,
                        database.version,
                        op_name,
                        hc_mult,
                        hidden_size,
                        quant_mode.name,
                    ),
                    _slice,
                    lambda c: get_sol(c[0], op_name)[0],  # c=(num_tokens,)
                    depth=1,
                )
                lat, _ = util_empirical.estimate(sol_q, (num_tokens,), grid)
                return lat

            if op == "both":
                return _emp_for_op("pre") + _emp_for_op("post")
            return _emp_for_op(op)

        if database_mode is None:
            database_mode = database._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            return PerformanceResult(get_sol()[0], energy=0.0, source="sol")
        if database_mode == common.DatabaseMode.ANALYTICAL:
            return PerformanceResult(get_sol()[0], energy=0.0, source="analytical")
        if database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol()
        if database_mode == common.DatabaseMode.EMPIRICAL:
            return PerformanceResult(get_empirical(), energy=0.0, source="empirical")

        def get_silicon():
            mhc_data = getattr(database, "_mhc_module_data", None)
            if not mhc_data:
                raise PerfDataNotAvailableError(
                    f"DeepSeek-V4 mHC module data not loaded for system='{database.system}', "
                    f"backend='{database.backend}', version='{database.version}'."
                )

            def _lookup_single(op_name: str) -> PerformanceResult:
                # Validate bucket presence before chained indexing; mhc_data is
                # a nested defaultdict, so `mhc_data[op][hc_mult][hidden_size]`
                # would silently materialize empty dicts and then query an
                # empty table, surfacing as an opaque miss instead of a
                # structured PerfDataNotAvailableError.
                if (
                    op_name not in mhc_data
                    or hc_mult not in mhc_data[op_name]
                    or hidden_size not in mhc_data[op_name][hc_mult]
                    or not mhc_data[op_name][hc_mult][hidden_size]
                ):
                    raise PerfDataNotAvailableError(
                        f"No mHC silicon data for op='{op_name}', hc_mult={hc_mult}, hidden_size={hidden_size}."
                    )
                mhc_dict = mhc_data[op_name][hc_mult][hidden_size]
                # 1-D tokens curve: RAW lerp in range; boundary util-hold via
                # the per-op mHC SOL beyond it.
                config = perf_interp.OpInterpConfig(
                    axes=("num_tokens",),
                    resolver=perf_interp.Grid(),
                    sol_fn=lambda t: get_sol(t, op_name)[0],
                )
                result = perf_interp.query(config, mhc_dict, num_tokens)
                latency = perf_interp.get_value(result, "latency")
                energy = perf_interp.get_value(result, "energy")
                return database._interp_pr(latency, energy=energy)

            # Silicon tables only store "pre" and "post" rows. For op=="both"
            # (still a supported input in DeepSeekV4MHCModule), aggregate the
            # two silicon look-ups so callers don't need to know about the
            # storage layout.
            if op == "both":
                pre_result = _lookup_single("pre")
                post_result = _lookup_single("post")
                # Use PerformanceResult's __add__ to merge sources correctly
                # (silicon + silicon -> silicon, mismatch -> mixed) instead of
                # constructing a new PR that would default-tag as silicon.
                return pre_result + post_result

            return _lookup_single(op)

        return database._query_silicon_or_hybrid(
            get_silicon=get_silicon,
            get_empirical=get_empirical,
            database_mode=database_mode,
            error_msg=(
                f"Failed to query DeepSeek-V4 mHC module for {num_tokens=}, {hidden_size=}, "
                f"{hc_mult=}, {sinkhorn_iters=}, {op=}"
            ),
        )

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------


# ───────────────────────────────────────────────────────────────────────
# _BaseDeepSeekV4AttentionModule (shared metadata)
# ───────────────────────────────────────────────────────────────────────


# ───────────────────────────────────────────────────────────────────────
# ContextDeepSeekV4AttentionModule
# ───────────────────────────────────────────────────────────────────────


class ContextDeepSeekV4AttentionModule(_core.ContextDeepSeekV4AttentionModule, OpShellKit):
    """Context-phase DeepSeek-V4 SWA/CSA/HCA compressed attention module.

    Owns three class-level caches:
    - ``_data_cache`` — merged ctx table (csa + hca split files combined)
    - ``_raw_data_cache`` — deepcopy of the merged table, kept untouched
      so the topk-piecewise lookup can consult the original
      compress_ratio==4 rows for boundary correctness.
    - ``_sparse_kernel_cache`` — dict ``{"paged_mqa_logits"}``
      of ``LoadedOpData`` used for prefix kernel-Δ correction.
    """

    _data_cache: ClassVar[dict] = {}
    _raw_data_cache: ClassVar[dict] = {}
    _sparse_kernel_cache: ClassVar[dict] = {}

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Fetches the engine's merged csa+hca context table view
        and the three DSV4 sparse-kernel views.

        Binds:
        - ``database._context_deepseek_v4_attention_module_data``
        - ``database._raw_context_deepseek_v4_attention_module_data``
        - ``database._dsv4_sparse_kernel_data``
        """

        from aiconfigurator_core.sdk.engine_table_view import fetch_table_view
        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache or key not in cls._raw_data_cache or key not in cls._sparse_kernel_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])

            def _primary(filename_enum):
                return resolve_op_data_path(system_data_root, database.backend, database.version, filename_enum.value)

            # Locals first, commit last — a failed sparse fetch must not
            # leave only the merged view cached (see GEMM.load_data).
            # The csa+hca merge happens engine-side; an absent-or-empty merge
            # binds None, matching the retired split-merge semantics
            # (whose filepath came from the csa side, loaded first).
            merged_view = fetch_table_view(database, "_context_deepseek_v4_attention_module_data")
            if merged_view:
                merged_loaded = LoadedOpData(
                    merged_view,
                    PerfDataFilename.dsv4_csa_context_module,
                    _primary(PerfDataFilename.dsv4_csa_context_module),
                )
            else:
                merged_loaded = None

            def _load_sparse(sub_key, filename_enum):
                view = fetch_table_view(database, f"_dsv4_sparse_kernel_data.{sub_key}")
                return LoadedOpData(view, filename_enum, _primary(filename_enum))

            # paged_mqa_logits is the only sparse-kernel sidecar with a live
            # consumer (the CP chunk walk). The hca_attn/csa_attn sidecars
            # were design placeholders with no query path and were retired
            # (csa_attn never even shipped data).
            sparse_loaded = {
                "paged_mqa_logits": _load_sparse("paged_mqa_logits", PerfDataFilename.dsv4_paged_mqa_logits_module),
            }

            cls._data_cache[key] = merged_loaded
            # The raw wrapper stays a plain alias for backward compatibility.
            cls._raw_data_cache[key] = merged_loaded
            cls._sparse_kernel_cache[key] = sparse_loaded
            cls._record_load()

        if "_context_deepseek_v4_attention_module_data" not in database.__dict__:
            database._context_deepseek_v4_attention_module_data = cls._data_cache[key]
        if "_raw_context_deepseek_v4_attention_module_data" not in database.__dict__:
            database._raw_context_deepseek_v4_attention_module_data = cls._raw_data_cache[key]
        if "_dsv4_sparse_kernel_data" not in database.__dict__:
            database._dsv4_sparse_kernel_data = cls._sparse_kernel_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()
        cls._raw_data_cache.clear()
        cls._sparse_kernel_cache.clear()

    # ------------------------------------------------------------------
    # Sparse-kernel lookup helper (formerly PerfDatabase._lookup_dsv4_sparse_kernel)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_context_deepseek_v4_attention_module)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # NOTE(#1357 PR-5): the Python CP prefill model and chunked-mqa
    # decomposition that lived here retired with the per-call query stack;
    # their oracle is the compiled engine (operators/dsv4.rs).


# ───────────────────────────────────────────────────────────────────────
# GenerationDeepSeekV4AttentionModule
# ───────────────────────────────────────────────────────────────────────


class GenerationDeepSeekV4AttentionModule(_core.GenerationDeepSeekV4AttentionModule, OpShellKit):
    """Decode-phase DeepSeek-V4 SWA/CSA/HCA compressed attention module.

    Owns ``_generation_deepseek_v4_attention_module_data`` (merged from
    csa+hca split files).
    """

    _data_cache: ClassVar[dict] = {}

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Fetches the engine's merged csa+hca generation table
        view, binds ``database._generation_deepseek_v4_attention_module_data``.
        """

        from aiconfigurator_core.sdk.engine_table_view import fetch_table_view
        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            # The csa+hca merge happens engine-side; an absent-or-empty merge
            # binds None, matching the retired _load_dsv4_split semantics
            # (whose filepath came from the csa side, loaded first).
            merged_view = fetch_table_view(database, "_generation_deepseek_v4_attention_module_data")
            if merged_view:
                system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
                primary = resolve_op_data_path(
                    system_data_root,
                    database.backend,
                    database.version,
                    PerfDataFilename.dsv4_csa_generation_module.value,
                )
                cls._data_cache[key] = LoadedOpData(merged_view, PerfDataFilename.dsv4_csa_generation_module, primary)
            else:
                cls._data_cache[key] = None

            cls._record_load()

        if "_generation_deepseek_v4_attention_module_data" not in database.__dict__:
            database._generation_deepseek_v4_attention_module_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_generation_deepseek_v4_attention_module)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------



class DeepSeekV4KVAllGather(Operation):
    """V4 CP all-gather with window/compression-aware message sizing."""

    _CP_AWARE: ClassVar[bool] = True

    def __init__(
        self,
        name: str,
        scale_factor: float,
        *,
        kind: str,
        width: int,
        cp_size: int,
        window_size: int = 0,
        compress_ratio: int = 1,
    ) -> None:
        super().__init__(name, scale_factor)
        if kind not in {"window", "compressed", "index"}:
            raise ValueError("V4 KV all-gather kind must be window, compressed, or index")
        self._kind = kind
        self._width = width
        self._cp_size = cp_size
        self._window_size = window_size
        self._compress_ratio = compress_ratio

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        batch = int(kwargs.get("batch_size"))
        sequence = int(kwargs.get("s"))
        if self._kind == "window":
            entries = min(sequence, self._window_size)
        elif self._kind == "compressed":
            entries = sequence // self._compress_ratio
        else:
            entries = sequence
        result = database.query_nccl(
            common.CommQuantMode.half,
            self._cp_size,
            "all_gather",
            batch * entries * self._width,
        )
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "estimated"),
        )

    def get_weights(self, **kwargs):
        return 0.0


class DeepSeekV4SparseAttention(Operation):
    """No-table V4 SWA/CSA/HCA attention core for ANALYTICAL granular paths.

    The logical sparse/window pair count is converted to an average effective
    MQA KV length and evaluated by the existing FA KernelSim model. Projection,
    compression, index scoring and TopK are intentionally outside this boundary.
    """

    _CP_AWARE: ClassVar[bool] = True

    def __init__(
        self,
        name: str,
        scale_factor: float,
        *,
        layout: str,
        local_heads: int,
        head_dim: int,
        window_size: int,
        compress_ratio: int,
        index_topk: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        cp_size: int = 1,
    ) -> None:
        super().__init__(name, scale_factor)
        if layout not in {"ragged", "paged"}:
            raise ValueError("DeepSeek-V4 sparse attention layout must be ragged or paged")
        if compress_ratio not in {0, 4, 128}:
            raise ValueError("DeepSeek-V4 compress_ratio must be 0, 4, or 128")
        self._layout = layout
        self._local_heads = local_heads
        self._head_dim = head_dim
        self._window_size = window_size
        self._compress_ratio = compress_ratio
        self._index_topk = index_topk
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode
        self._cp_size = cp_size

    @staticmethod
    def _causal_limited_pairs(batch: int, query: int, prefix: int, limit: int) -> int:
        full = prefix + query
        if prefix >= limit:
            return batch * query * limit
        if full <= limit:
            return batch * (full * (full + 1) - prefix * (prefix + 1)) // 2
        ramp = batch * (limit * (limit + 1) - prefix * (prefix + 1)) // 2
        return ramp + batch * (full - limit) * limit

    @staticmethod
    def _floor_sum(n: int, divisor: int) -> int:
        """Return sum(floor(i/divisor), i=1..n) in O(1)."""
        if n <= 0:
            return 0
        groups = n // divisor
        return divisor * groups * (groups - 1) // 2 + groups * (n - groups * divisor + 1)

    @classmethod
    def _compressed_causal_pairs(cls, batch: int, query: int, prefix: int, ratio: int, limit: int) -> int:
        # Each fresh query sees floor(position / ratio) completed compressed
        # entries. Clamp at CSA top-k; HCA passes an effectively unbounded limit.
        unclamped_queries = min(query, max(0, ratio * limit - 1 - prefix))
        unclamped = cls._floor_sum(prefix + unclamped_queries, ratio) - cls._floor_sum(prefix, ratio)
        return batch * (unclamped + (query - unclamped_queries) * limit)

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        batch = int(kwargs.get("batch_size"))
        sequence = int(kwargs.get("s"))
        if batch <= 0 or sequence <= 0:
            return PerformanceResult(0.0, energy=0.0, source="analytical")

        if self._layout == "ragged":
            query = math.ceil(sequence / self._cp_size)
            prefix = int(kwargs.get("prefix", 0)) + max(0, sequence - query)
            pairs = self._causal_limited_pairs(batch, query, prefix, self._window_size)
            if self._compress_ratio:
                compressed_limit = self._index_topk if self._compress_ratio == 4 else 2**62
                pairs += self._compressed_causal_pairs(batch, query, prefix, self._compress_ratio, compressed_limit)
        else:
            query = 1
            pairs = batch * min(sequence, self._window_size)
            if self._compress_ratio:
                compressed = sequence // self._compress_ratio
                if self._compress_ratio == 4:
                    compressed = min(compressed, self._index_topk)
                pairs += batch * compressed

        effective_kv = max(1, math.ceil(pairs / (batch * query)))
        from aiconfigurator_core.sdk.kernelsim.analytical import attention_latency_ms

        # SGLang's DSV4 sparse FlashMLA contract is BF16 Q and BF16 WGMMA with
        # an FP8 KV cache. Hopper and Blackwell kernels dequantize the no-PE
        # cache values into BF16 shared memory before QK/PV; the cache itself
        # stores 448 FP8 no-PE bytes, 64 BF16 RoPE values, and 8 scale/padding
        # bytes per token.
        latency = attention_latency_ms(
            system=database.system,
            gpu=database.system_spec["gpu"],
            batch=batch,
            query_length=query,
            kv_length=effective_kv,
            query_heads=database._analytical_config.sparse_attention_executed_heads(self._local_heads),
            kv_heads=1,
            head_dim=self._head_dim,
            dtype="bf16",
            value_head_dim=self._head_dim,
            kv_storage_dim=self._head_dim,
            kv_cache_bytes_per_token=584,
            # Cache packing/writes are fused into the preceding DSV4 KV
            # norm/RoPE path and sit outside this attention-core boundary.
            include_kv_cache_update=False,
            config=database._analytical_config,
            # Pair-count reduction above already includes causal/window/sparse
            # masking; use an equivalent rectangular workload here.
            causal=False,
        )
        source = "analytical" if database._default_database_mode == common.DatabaseMode.ANALYTICAL else "estimated"
        return PerformanceResult(latency * self._scale_factor, energy=0.0, source=source)

    def get_weights(self, **kwargs):
        return 0.0


class DeepSeekV4MegaMoEModule(Operation):
    """
    SGLang DeepSeek-V4 MegaMoE routed module.

    This models the measured routed MegaMoE module boundary used by
    ``collector/sglang/collect_dsv4_megamoe.py``: prepared hidden states and
    top-k tensors -> SGLang pre-dispatch -> ``deep_gemm.fp8_fp4_mega_moe`` ->
    routed output scaling. Gate/top-k and shared experts are modeled outside
    this operation.
    """

    _data_cache: ClassVar[dict] = {}

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        from aiconfigurator_core.sdk.engine_table_view import load_view
        from aiconfigurator_core.sdk.perf_database import PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            # Single-primary semantics live in the engine view (it reads only
            # the head of the resolved source list, like the retired loader).
            cls._data_cache[key] = load_view(
                database, "_dsv4_megamoe_module_data", PerfDataFilename.dsv4_megamoe_module
            )
            cls._record_load()

        if "_dsv4_megamoe_module_data" not in database.__dict__:
            database._dsv4_megamoe_module_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    @classmethod
    def _query_megamoe_table(
        cls,
        database: PerfDatabase,
        num_tokens: int,
        hidden_size: int,
        inter_size: int,
        topk: int,
        num_experts: int,
        moe_tp_size: int,
        moe_ep_size: int,
        quant_mode: common.MoEQuantMode,
        workload_distribution: str,
        is_context: bool = True,
        source_policy: str = "random",
        pre_dispatch: str = "sglang_jit",
        num_fused_shared_experts: int = 0,
        kernel_source: str = "deepgemm_megamoe",
        kernel_dtype: str = "fp8_fp4",
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult:
        """
        Query DeepSeek-V4 MegaMoE full-module latency.

        This table is intentionally strict: it models only measured fused
        MegaMoE rows and does not fall back to uniform/random distributions or
        analytical constants when a row is missing. New databases use the
        unified ``dsv4_megamoe_module`` file for both context and generation;
        ``is_context`` selects the phase stored inside that table.
        """
        cls.load_data(database)

        if database_mode is None:
            database_mode = database._default_database_mode
        if database_mode not in (common.DatabaseMode.SILICON, common.DatabaseMode.HYBRID):
            raise PerfDataNotAvailableError(
                f"DSv4 MegaMoE module only supports measured SILICON data, got {database_mode=}."
            )

        if not isinstance(quant_mode, common.MoEQuantMode):
            quant_mode = common.MoEQuantMode[str(quant_mode)]
        phase = "context" if is_context else "generation"

        module_data = getattr(database, "_dsv4_megamoe_module_data", None)
        if module_data is None:
            raise PerfDataNotAvailableError(
                f"DSv4 MegaMoE module data not loaded for system='{database.system}', "
                f"backend='{database.backend}', version='{database.version}'."
            )
        module_data.raise_if_not_loaded()

        try:
            token_dict = module_data[phase][kernel_source][kernel_dtype][quant_mode][pre_dispatch][source_policy][
                workload_distribution
            ][topk][num_experts][num_fused_shared_experts][hidden_size][inter_size][moe_tp_size][moe_ep_size]
        except KeyError as exc:
            raise PerfDataNotAvailableError(
                f"No DSv4 MegaMoE {phase} module data for {kernel_source=}, {kernel_dtype=}, {quant_mode=}, "
                f"{pre_dispatch=}, {source_policy=}, {workload_distribution=}, {topk=}, {num_experts=}, "
                f"{num_fused_shared_experts=}, {hidden_size=}, {inter_size=}, "
                f"{moe_tp_size=}, {moe_ep_size=}."
            ) from exc

        # 1-D tokens curve. No analytic SOL is implemented for the fused
        # MegaMoE module, but util-hold only needs the SOL RATIO: routed-expert
        # work scales ~linearly with tokens at fixed topk/experts/hidden, so a
        # linear token proxy is ratio-equivalent (see the DeepEP note).
        config = perf_interp.OpInterpConfig(
            axes=("num_tokens",),
            resolver=perf_interp.Grid(),
            sol_fn=lambda t: float(t),
        )
        result = perf_interp.query(config, token_dict, num_tokens)
        latency = float(perf_interp.get_value(result, "latency"))
        energy = float(perf_interp.get_value(result, "energy"))
        return PerformanceResult(latency, energy=energy)

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query measured MegaMoE routed-module latency."""
        from aiconfigurator_core.sdk.system_spec import is_blackwell_spec

        if not is_blackwell_spec(database.system_spec):
            raise ValueError(
                "DeepSeek-V4 MegaMoE is only supported on Blackwell-class GPUs "
                "with the NVIDIA-specific MegaMoE backend."
            )

        # DSv4 MegaMoE perf rows are indexed by local-rank tokens. Do not
        # multiply by attention_dp_size here; the old decomposed MoE table is
        # indexed differently.
        x = int(kwargs.get("x"))
        overwrite_quant_mode = kwargs.get("quant_mode")
        quant_mode = self._quant_mode if overwrite_quant_mode is None else overwrite_quant_mode

        result = database.query_dsv4_megamoe_module(
            num_tokens=x,
            hidden_size=self._hidden_size,
            inter_size=self._inter_size,
            topk=self._topk,
            num_experts=self._num_experts,
            moe_tp_size=self._moe_tp_size,
            moe_ep_size=self._moe_ep_size,
            quant_mode=quant_mode,
            workload_distribution=self._workload_distribution,
            is_context=self._is_context,
            source_policy=self._source_policy,
            pre_dispatch=self._pre_dispatch,
            num_fused_shared_experts=self._num_fused_shared_experts,
            kernel_source=self._kernel_source,
            kernel_dtype=self._kernel_dtype,
        )

        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


# ───────────────────────────────────────────────────────────────────────
# Init-time split-file merge helper (formerly in PerfDatabase.__init__)
# ───────────────────────────────────────────────────────────────────────


def _load_dsv4_split(loaded_list):
    """Merge per-(attn_kind) loaded data into one combined ``LoadedOpData``.

    Each DSV4 context/generation module CSV is collected per attention kind
    (csa/hca). Each loader returns a nested dict scoped to one
    compress_ratio. We merge into one aggregate dict so downstream queries
    do not need to know which attention kind produced each row.
    """
    from aiconfigurator_core.sdk.perf_database import LoadedOpData

    merged: dict = {}
    first_loaded = next((x for x in loaded_list if x is not None), None)
    if first_loaded is None:
        return None
    for loaded in loaded_list:
        if loaded is None or not loaded.loaded:
            continue
        _deep_merge_dsv4_dicts(merged, loaded.data)
    if not merged:
        return None
    return LoadedOpData(merged, first_loaded.op_name_enum, first_loaded.filepath)


# ─────────────────────────────────────────────────────────
# CSV loaders (moved here from perf_database.py so each op family owns its data + parser)
# ─────────────────────────────────────────────────────────


def load_mhc_module_data(mhc_file: str):
    """Load DeepSeek-V4 mHC pre/post module-level performance data.

    CSV columns: framework, version, device, op_name, kernel_source,
    architecture, num_tokens, hc_mult, hidden_size, latency [, power]
    Optional metadata columns: num_sites, sinkhorn_iters
    Legacy rows may include a ``model`` column; it is ignored because mHC is
    selected by compute shape.

    ``op_name`` is ``pre`` or ``post``, matching the ``op`` arg of
    ``query_mhc_module``.

    Dict structure (matches query_mhc_module silicon path):
        data[op][hc_mult][hidden_size][num_tokens]
    """
    rows = _read_filtered_rows(mhc_file)
    if rows is None:
        logger.debug(f"mHC module data file {mhc_file} not found.")
        return None

    mhc_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))

    has_power = len(rows) > 0 and "power" in rows[0]

    for row in rows:
        op = row["op_name"]
        hc_mult = int(row["hc_mult"])
        hidden_size = int(row["hidden_size"])
        num_tokens = int(row["num_tokens"])
        latency = float(row["latency"])
        power = float(row.get("power", 0.0)) if has_power else 0.0
        energy = power * latency

        try:
            # Check for conflict: first source wins (shared-layer contract).
            mhc_data[op][hc_mult][hidden_size][num_tokens]
            logger.debug(f"value conflict in mhc module data: {op} {hc_mult} {hidden_size} {num_tokens}")
        except KeyError:
            mhc_data[op][hc_mult][hidden_size][num_tokens] = {
                "latency": latency,
                "power": power,
                "energy": energy,
            }

    return mhc_data


_DSV4_DTYPE_ALIASES = {
    # CSV columns use sglang naming; aic_dev enums use canonical short names.
    "fp8_e4m3": "fp8",
}


def _dsv4_normalize_dtype(name: str) -> str:
    return _DSV4_DTYPE_ALIASES.get(name, name)


# ───────────────────────────────────────────────────────────────────────
# DSV4 CSA topk DELTA calibration (SCHEME A correction).
#
# The CSA context-module collector runs the topK kernel on DEGENERATE scores
# (dummy weights + zero/uninitialized prefix KV -> near-constant logits -> the
# Small topK path falls into its O(n^2) tie-break). That inflates the measured
# module latency vs real silicon, where logits are spread. We measured the topK
# time standalone under a degenerate "flat" construction and a representative
# "top_last" construction for every (prefix, isl, batch_size) shape, phase-
# qualified (context runs the v1 selector, generation v2), stored as four rows
# (score_mode=v{1,2}_{flat,top_last}) per shape in dsv4_csa_topk_calib_perf;
# DELTA = flat.latency - top_last.latency per variant. At query time we
# SUBTRACT the matching variant's DELTA from the CSA (compress_ratio==4)
# module latency only. The DELTA is selector-geometry-specific (Flash
# index_topk 512 vs Pro 1024), so the calib keys by the row's native
# num_heads and applies ONLY to queries with the matching native identity
# (#1460 review) — an uncovered native (Pro today: calib collected for
# Flash only) is a logged no-op, never a borrowed correction.
# Gate: AIC_DSV4_TOPK_CORRECTION (default on; set "0" to disable).
# ───────────────────────────────────────────────────────────────────────
_TOPK_CORRECTION_ENABLED = os.environ.get("AIC_DSV4_TOPK_CORRECTION", "1") != "0"


def _build_topk_calib_from_rows(by_native):
    """Pair flat / top_last rows into per-native, per-variant DELTA tables.

    Returns ``{native: {"v1": table_or_None, "v2": table_or_None}}`` (or
    ``None`` when nothing pairs), each table
    ``{'exact': {(step, isl, bs): delta_ms}, 'by_pi': ...}``. ``by_native`` is
    the ``_TOPK_CALIB_KEYS`` nesting
    ``data[native][step][isl][bs][score_mode] = {"latency": ms}`` — the DELTA
    is selector-geometry-specific, so the row's native ``num_heads`` is the
    outermost key (#1460 review) and only matching queries consume it.
    """
    if not by_native:
        return None
    out = {}
    for native, by_mode in by_native.items():
        if not isinstance(native, int):
            # The generic loader keeps unparseable key cells as str (the
            # score_mode case); a malformed num_heads must not fail the load.
            # The Rust twin reads the column via u32_optional and skips too.
            continue
        variants = {}
        for variant in ("v1", "v2"):
            exact = {}
            by_pi = {}
            for step, isl_d in by_mode.items():
                for isl, bs_d in isl_d.items():
                    for bs, mode_d in bs_d.items():
                        flat = mode_d.get(f"{variant}_flat")
                        top_last = mode_d.get(f"{variant}_top_last")
                        if not isinstance(flat, dict) or not isinstance(top_last, dict):
                            continue
                        delta = max(0.0, float(flat["latency"]) - float(top_last["latency"]))
                        exact[(step, isl, bs)] = delta
                        by_pi.setdefault((step, isl), []).append((bs, delta))
            if not exact:
                variants[variant] = None
                continue
            for k in by_pi:
                by_pi[k].sort()
            variants[variant] = {"exact": exact, "by_pi": by_pi}
        if any(variants.values()):
            out[native] = variants
    return out or None


_TOPK_CALIB_MISS_WARNED: set = set()


def _dsv4_topk_calib_for_native(calib, native_heads, database):
    """Select the calib bucket matching the querying model's native identity.

    A mismatched bucket must never be borrowed (Flash index_topk 512 vs Pro
    1024 run different selector geometry); an uncovered native is a one-time
    logged no-op so the coverage gap is visible, not silent (#1460 review).
    """
    if not calib:
        return None
    bucket = calib.get(int(native_heads))
    if bucket is None:
        key = (int(native_heads), database.system, database.backend, database.version)
        if key not in _TOPK_CALIB_MISS_WARNED:
            _TOPK_CALIB_MISS_WARNED.add(key)
            logger.warning(
                "DSV4 CSA topk DELTA calibration has no bucket for native_heads=%s on "
                "%s/%s/%s (available: %s); CSA latencies for this model stay UNCORRECTED "
                "for the collector's degenerate-topk inflation until its calibration is "
                "collected.",
                native_heads,
                database.system,
                database.backend,
                database.version,
                sorted(calib),
            )
        return None
    return bucket


def _dsv4_interp_1d_from_points(points, x):
    """Linear interpolation with nearest-value extrapolation."""
    if not points:
        return None
    merged = defaultdict(list)
    for coord, value in points:
        merged[int(coord)].append(float(value))
    xs = sorted(merged)
    vals = {k: sum(v) / len(v) for k, v in merged.items()}
    if x in vals:
        return vals[x]
    if len(xs) == 1:
        return vals[xs[0]]
    if x <= xs[0]:
        return vals[xs[0]]
    if x >= xs[-1]:
        return vals[xs[-1]]
    left = max(v for v in xs if v < x)
    right = min(v for v in xs if v > x)
    if left == right:
        return vals[left]
    t = (float(x) - float(left)) / (float(right) - float(left))
    return vals[left] * (1.0 - t) + vals[right] * t


def _dsv4_topk_delta_ms(calib, prefix, isl, bs):
    """Return topK DELTA (flat_ms - top_last_ms) from measured calibration.

    Exact (prefix, isl, bs) rows are preferred; off-grid shapes are
    interpolated prefix-first within a fixed (isl, bs), then isl, then bs.
    Returns 0.0 when no calibration is available.
    """
    if not calib:
        return 0.0
    prefix = int(prefix)
    isl = int(isl)
    bs = int(bs)
    exact = calib.get("exact", {})
    direct = exact.get((prefix, isl, bs))
    if direct is not None:
        return max(0.0, float(direct))

    def _prefix_interp(query_prefix, anchor_isl, anchor_bs):
        points = [(p, d) for (p, i, b), d in exact.items() if int(i) == int(anchor_isl) and int(b) == int(anchor_bs)]
        return _dsv4_interp_1d_from_points(points, query_prefix)

    def _isl_interp(query_prefix, query_isl, anchor_bs):
        isl_values = sorted({i for (_, i, b) in exact if int(b) == int(anchor_bs)})
        points = []
        for i in isl_values:
            value = _prefix_interp(query_prefix, i, anchor_bs)
            if value is not None:
                points.append((i, value))
        return _dsv4_interp_1d_from_points(points, query_isl)

    bs_values = sorted({b for (_, _, b) in exact})
    points = []
    for b in bs_values:
        value = _isl_interp(prefix, isl, b)
        if value is not None:
            points.append((b, value))
    interpolated = _dsv4_interp_1d_from_points(points, bs)
    if interpolated is None:
        return 0.0
    return max(0.0, float(interpolated))


def _get_dsv4_topk_calib(database):
    """Load (and cache on ``database``) the CSA topK DELTA calibration through
    the same sparse-op loader + source resolution the other DSV4 ops use."""
    cached = getattr(database, "_dsv4_csa_topk_calib", _MISSING)
    if cached is not _MISSING:
        return cached
    import os as _os

    from aiconfigurator_core.sdk.perf_database import PerfDataFilename

    system_data_root = _os.path.join(database.systems_root, database.system_spec["data_dir"])
    enum = PerfDataFilename.dsv4_csa_topk_calib
    primary_path = resolve_op_data_path(system_data_root, database.backend, database.version, enum.value)
    sources = database._build_op_sources(enum, primary_path, system_data_root)
    by_mode = load_dsv4_sparse_op_data(sources, _TOPK_CALIB_KEYS)
    calib = _build_topk_calib_from_rows(by_mode)
    try:
        database._dsv4_csa_topk_calib = calib
    except Exception:
        pass
    return calib


_MISSING = object()


def _validate_dsv4_local_head_semantics(rows, file_path):
    """Reject rows still carrying the retired pre-#1131 NATIVE ``num_heads``
    semantics.

    The unified DSV4 module-row convention (issue #1429) is rank-LOCAL:
    ``num_heads`` is the head count the benchmarked module actually ran with
    on one rank, ``tp_size`` is persisted in every row, and the model-native
    count is derived as ``num_heads * tp_size``.  Within one artifact a
    genuine local sweep varies ``num_heads`` as ``native // tp``, so a group
    whose ``num_heads`` stays constant across several ``tp_size`` values can
    only be a stale pre-migration file (Flash/Flash-FP8 64, Pro 128 constant
    across tp 1/2/4/8).  Reading such a file as local would collapse distinct
    tp shards onto wrong (native, local) coordinates again — rows whose
    latencies differ 30-50% — so raise instead.  The shipped sglang 0.5.10
    tables were migrated in-place; external files must be migrated the same
    way (``num_heads //= tp_size``).

    The fingerprint is checked per ``(model, version)`` because the shared
    layer concatenates sibling-version files into one row stream — a
    migrated (local) primary pooled with a stale (native) sibling of the
    same model would otherwise blur both patterns and mask the stale rows.
    """
    observed: dict[tuple[str, str], set[tuple[int, int]]] = {}
    saw_tp_size = False
    missing_tp_rows = 0
    for row in rows:
        try:
            heads = int(row["num_heads"])
        except (TypeError, ValueError, KeyError):
            continue
        try:
            tp = max(1, int(row["tp_size"]))
            saw_tp_size = True
        except (TypeError, ValueError, KeyError):
            tp = 1
            missing_tp_rows += 1
        group = (str(row.get("model", "")), str(row.get("version", "")))
        observed.setdefault(group, set()).add((heads, tp))

    if saw_tp_size and missing_tp_rows:
        # A per-row tp_size fallback to 1 would derive native = num_heads and
        # file the row under a wrong native bucket (#1460 review): fail on any
        # unparseable tp_size once the file demonstrably carries the column.
        raise ValueError(
            f"DSV4 module file {file_path} has {missing_tp_rows} row(s) without a parseable "
            f"tp_size; the #1429 convention requires tp_size in every row "
            f"(native = num_heads * tp_size)."
        )

    if observed and not saw_tp_size:
        # Without tp_size every row collapses to tp=1 and the stale fingerprint
        # below can never trigger — a stale file would load silently with wrong
        # (native, local) coordinates. The #1429 convention makes tp_size a
        # mandatory column, so fail like the Rust loader does.
        raise ValueError(
            f"DSV4 module file {file_path} carries no parseable tp_size column; the #1429 "
            f"convention requires tp_size in every row (native = num_heads * tp_size)."
        )

    for (model, version), pairs in observed.items():
        tps = {tp for _, tp in pairs}
        heads_constant = len({h for h, _ in pairs}) == 1
        product_constant = len({h * tp for h, tp in pairs}) == 1
        if len(tps) > 1 and heads_constant and not product_constant:
            raise ValueError(
                f"DSV4 module rows for model={model!r} version={version!r} in {file_path} keep "
                f"num_heads constant across tp_size values {sorted(tps)}: that is the retired "
                f"pre-#1131 NATIVE semantics (#1429). Migrate the file to rank-local heads "
                f"(num_heads //= tp_size) before loading."
            )


def load_context_dsv4_kind_module_data(file_path: str):
    """Load ONE DeepSeek-V4 context CSV (single attn_kind / compress_ratio).

    Returns an 8-level prefix-resolved nested dict:
        data[fmha_quant][kv_quant][gemm_quant][num_heads_native][num_heads_local]
            [compress_ratio][prefix][s][b] = {"latency": ms, "power": W, "energy": J}

    The head identity is (native, rank-local) under the unified #1429
    convention: the ``num_heads`` column is the rank-LOCAL head count the
    benchmarked module ran with, and the model-native count is derived as
    ``num_heads * tp_size`` (``_validate_dsv4_local_head_semantics`` rejects
    stale pre-#1131 files that stored native heads instead).  The native value
    is the row's model identity (separates Pro rows from Flash rows) and the
    local count is the physical per-rank shape; both are key dimensions.
    Collapsing either axis merged rows whose latencies differ 30-50%
    (different model shapes / tp shards) into one coordinate, leaving an
    arbitrary row-order winner.

    ``prefix`` is the past-KV length, ``int(float(row["step"]))``; ``s`` is the
    context chunk length (``isl``).  Multiple files (csa/hca) merge cleanly
    because compress_ratio is a key dimension.
    """
    rows = _read_filtered_rows(file_path)
    if rows is None:
        logger.debug(f"DSV4 module data file {file_path} not found.")
        return None
    _validate_dsv4_local_head_semantics(rows, file_path)

    # 8-level nesting: fmha → kv → gemm → native → local → cr → prefix → s → b
    def _make_nested(depth: int):
        if depth == 0:
            return defaultdict()
        return defaultdict(lambda d=depth: _make_nested(d - 1))

    data = _make_nested(8)
    has_power = bool(rows) and "power" in rows[0]

    for row in rows:
        if row.get("batch_size") in (None, "", "batch_size"):
            continue  # skip duplicate header rows from appended runs
        try:
            b = int(row["batch_size"])
            s = int(row["isl"])
            prefix = int(float(row.get("step", 0) or 0))
            cr = int(row["compress_ratio"])
            latency = float(row["latency"])
            heads_col = int(row["num_heads"])
            tp_size = max(1, int(row.get("tp_size", 1) or 1))
        except (TypeError, ValueError, KeyError):
            continue
        power = float(row.get("power", 0.0)) if has_power else 0.0

        num_heads_local = heads_col
        num_heads_native = heads_col * tp_size
        gemm_mode = common.GEMMQuantMode[row["gemm_type"]]
        fmha_mode = common.FMHAQuantMode[_dsv4_normalize_dtype(row["mla_dtype"])]
        kv_dtype = common.KVCacheQuantMode[_dsv4_normalize_dtype(row["kv_cache_dtype"])]

        # NOTE: the topK DELTA correction (degenerate -> representative) is
        # applied ONCE at query time for compress_ratio==4 (CSA). Do NOT
        # subtract it here, or the CSA module latency would be double-corrected.
        try:
            # Check for conflict: first source wins (shared-layer contract).
            data[fmha_mode][kv_dtype][gemm_mode][num_heads_native][num_heads_local][cr][prefix][s][b]
            logger.debug(
                f"value conflict in context dsv4 module data: {fmha_mode} {kv_dtype} {gemm_mode} "
                f"{num_heads_native} {num_heads_local} {cr} {prefix} {s} {b}"
            )
        except KeyError:
            data[fmha_mode][kv_dtype][gemm_mode][num_heads_native][num_heads_local][cr][prefix][s][b] = {
                "latency": latency,
                "power": power,
                "energy": power * latency,
            }
    return data


def load_generation_dsv4_kind_module_data(file_path: str):
    """Load ONE DeepSeek-V4 generation CSV.

    Generation lookup uses absolute KV length ``s_total = isl + step`` (decode
    is q_len=1 with past_kv = step).  Dict shape (same (native, local) head
    identity as ``load_context_dsv4_kind_module_data``: rank-local ``num_heads``
    column, native derived as ``num_heads * tp_size``, stale NATIVE-semantics
    files rejected by ``_validate_dsv4_local_head_semantics``):
        data[kv_quant][gemm_quant][num_heads_native][num_heads_local]
            [compress_ratio][b][s_total]
    """
    rows = _read_filtered_rows(file_path)
    if rows is None:
        logger.debug(f"DSV4 module data file {file_path} not found.")
        return None
    _validate_dsv4_local_head_semantics(rows, file_path)

    # 6-level nesting: kv → gemm → native → local → cr → b → s_total
    def _make_nested(depth: int):
        if depth == 0:
            return defaultdict()
        return defaultdict(lambda d=depth: _make_nested(d - 1))

    data = _make_nested(6)
    has_power = bool(rows) and "power" in rows[0]

    for row in rows:
        if row.get("batch_size") in (None, "", "batch_size"):
            continue
        try:
            b = int(row["batch_size"])
            s_total = int(row["isl"]) + int(row["step"])
            cr = int(row["compress_ratio"])
            latency = float(row["latency"])
            heads_col = int(row["num_heads"])
            tp_size = max(1, int(row.get("tp_size", 1) or 1))
        except (TypeError, ValueError, KeyError):
            continue
        power = float(row.get("power", 0.0)) if has_power else 0.0

        num_heads_local = heads_col
        num_heads_native = heads_col * tp_size
        gemm_mode = common.GEMMQuantMode[row["gemm_type"]]
        kv_dtype = common.KVCacheQuantMode[_dsv4_normalize_dtype(row["kv_cache_dtype"])]

        try:
            # Check for conflict: first source wins (shared-layer contract).
            data[kv_dtype][gemm_mode][num_heads_native][num_heads_local][cr][b][s_total]
            logger.debug(
                f"value conflict in generation dsv4 module data: {kv_dtype} {gemm_mode} "
                f"{num_heads_native} {num_heads_local} {cr} {b} {s_total}"
            )
        except KeyError:
            data[kv_dtype][gemm_mode][num_heads_native][num_heads_local][cr][b][s_total] = {
                "latency": latency,
                "power": power,
                "energy": power * latency,
            }
    return data


def load_dsv4_megamoe_module_data(dsv4_megamoe_module_file):
    """
    Load DeepSeek-V4 MegaMoE full-module data.

    The collected latency is the SGLang/DeepGEMM MegaMoE routed path:
    prepared hidden states and top-k tensors -> pre-dispatch -> fused MegaMoE.
    Gate/top-k generation is intentionally outside the measured region.

    Returns:
        dict: Nested dict whose leaves contain latency, power, energy and
        routing metadata.
    """
    if dsv4_megamoe_module_file is None:
        return None

    if isinstance(dsv4_megamoe_module_file, list | tuple):
        raise TypeError("DSv4 MegaMoE data loader expects a single unified perf file path")

    source_label = os.fspath(dsv4_megamoe_module_file)
    rows = _read_filtered_rows(source_label)
    if rows is None:
        logger.debug(f"DeepSeek-V4 MegaMoE data file {source_label} not found.")
        return None

    def _to_bool(value: object) -> bool:
        return str(value).strip().lower() in {"1", "true", "yes", "y"}

    row_bool_invariants = [
        ("used_cuda_graph", True, None, "DSv4 MegaMoE perf row was not collected with CUDA Graph"),
        (
            "includes_gate_topk",
            False,
            "true",
            "DSv4 MegaMoE perf row includes gate/top-k outside the supported boundary",
        ),
        ("includes_routed_scale", True, None, "DSv4 MegaMoE perf row does not include SGLang routed output scaling"),
    ]

    def _row_phase(row: dict[str, str]) -> str:
        phase = row.get("phase", "").strip()
        if not phase:
            raise ValueError(f"DSv4 MegaMoE unified perf file requires a phase column: {source_label} {row}")
        if phase not in {"context", "generation"}:
            raise ValueError(f"DSv4 MegaMoE perf row has unsupported phase={phase!r}: {row}")
        return phase

    def _put_nested(root: dict, keys: list[object], value: dict) -> None:
        current = root
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        leaf_key = keys[-1]
        if leaf_key in current:
            raise ValueError(f"duplicate DSv4 MegaMoE data row for {source_label} {keys}")
        current[leaf_key] = value

    dsv4_megamoe_data: dict = {}
    logger.debug(f"Loading DeepSeek-V4 MegaMoE module data from: {source_label}")
    for row in rows:
        for field, expected_value, default, error in row_bool_invariants:
            if _to_bool(row.get(field, default)) != expected_value:
                raise ValueError(f"{error}: {source_label} {row}")

        kernel_source = row.get("kernel_source", "deepgemm_megamoe")
        kernel_dtype = row["kernel_dtype"]
        quant_mode = common.MoEQuantMode[row["moe_dtype"]]
        pre_dispatch = row["pre_dispatch"]
        source_policy = row["source_policy"]
        distribution = row["distribution"]
        topk = int(row["topk"])
        num_experts = int(row["num_experts"])
        num_fused_shared_experts = int(row.get("num_fused_shared_experts", 0))
        hidden_size = int(row["hidden_size"])
        inter_size = int(row["inter_size"])
        moe_tp_size = int(row.get("moe_tp_size", 1))
        moe_ep_size = int(row["moe_ep_size"])
        num_tokens = int(row["num_tokens"])
        latency = float(row["latency"])
        power = float(row.get("power") or 0.0)
        energy = power * latency
        num_max_tokens_per_rank = int(row.get("num_max_tokens_per_rank") or 0)
        effective_num_max_tokens_per_rank = int(row.get("effective_num_max_tokens_per_rank") or num_max_tokens_per_rank)

        entry = {
            "latency": latency,
            "power": power,
            "energy": energy,
            "global_num_tokens": int(row.get("global_num_tokens") or num_tokens * moe_ep_size),
            "num_max_tokens_per_rank": num_max_tokens_per_rank,
            "effective_num_max_tokens_per_rank": effective_num_max_tokens_per_rank,
            "used_cuda_graph": True,
            "kernel_dtype": kernel_dtype,
            "routed_scaling_factor": float(row["routed_scaling_factor"]),
            "includes_routed_scale": True,
            "includes_gate_topk": False,
            "buffer_policy": row.get("buffer_policy", ""),
            "includes_buffer_init": _to_bool(row.get("includes_buffer_init", "false")),
        }
        phase = _row_phase(row)
        entry["phase"] = phase
        _put_nested(
            dsv4_megamoe_data,
            [
                phase,
                kernel_source,
                kernel_dtype,
                quant_mode,
                pre_dispatch,
                source_policy,
                distribution,
                topk,
                num_experts,
                num_fused_shared_experts,
                hidden_size,
                inter_size,
                moe_tp_size,
                moe_ep_size,
                num_tokens,
            ],
            entry,
        )

    return dsv4_megamoe_data


# ───────────────────────────────────────────────────────────────────────
# DSV4 sparse-op family loader (ONE engine for all four)
# ───────────────────────────────────────────────────────────────────────
# csa_attn / hca_attn / paged_mqa_logits (FMLA & indexer kernels) and the
# csa_topk_calib DELTA rows share ONE column schema, so they all parse through
# ``load_dsv4_sparse_op_data``; each consumer just supplies the key columns it
# indexes on (declared here so callers stay in sync).
_SPARSE_KERNEL_KEYS = ("num_heads", "tp_size", "step", "isl", "batch_size")
_TOPK_CALIB_KEYS = ("num_heads", "step", "isl", "batch_size", "score_mode")


def load_dsv4_sparse_op_data(file_or_sources, key_columns):
    """Generic loader for the DeepSeek-V4 sparse-op family.

    Reads the shared perf schema (parquet or txt, single path or override
    ``(path, kernel_source_filter)`` sources — see ``_read_filtered_rows``) and
    nests every row under ``key_columns`` in order, leaf == ``{"latency": ms}``.

    Numeric key cells coerce to ``int``; non-numeric stay ``str`` (e.g.
    ``score_mode``). Rows with a blank or NaN/inf key cell are skipped.
    Returns ``None`` when no source file exists.

    Consumers:
      - sparse kernels: ``_SPARSE_KERNEL_KEYS`` -> data[heads][tp][past_kv][isl][bs]
      - topk calib:     ``_TOPK_CALIB_KEYS``    -> data[native][step][isl][bs][score_mode]
    """
    rows = _read_filtered_rows(file_or_sources)
    if rows is None:
        return None

    def _coerce(value):
        try:
            return int(float(value))
        except (TypeError, ValueError, OverflowError):
            return value

    def _is_bad_key(k):
        # A key cell that is blank or a NaN/inf sentinel must not become a dict
        # key: such rows are malformed and would misbucket (or KeyError) the
        # downstream calibration lookup. Legitimate non-numeric keys (e.g.
        # ``score_mode`` values like ``"default"``) are kept.
        if k is None:
            return True
        if isinstance(k, float):  # uncoerced float NaN/inf
            return k != k or k in (float("inf"), float("-inf"))
        if isinstance(k, str):
            return k.strip() == "" or k.strip().lower() in (
                "nan",
                "inf",
                "-inf",
                "+inf",
                "infinity",
                "-infinity",
            )
        return False

    root: dict = {}
    for row in rows:
        # Skip duplicate header rows (files may be appended to across runs).
        if row.get("batch_size") in (None, "", "batch_size"):
            continue
        try:
            keys = [_coerce(row[col]) for col in key_columns]
            latency = float(row["latency"])
        except (KeyError, TypeError, ValueError):
            continue
        if any(_is_bad_key(k) for k in keys):  # blank / NaN / inf key cell
            continue
        node = root
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        if keys[-1] in node:
            # Check for conflict: first source wins (shared-layer contract).
            logger.debug(f"value conflict in dsv4 sparse-op data: {keys}")
            continue
        node[keys[-1]] = {"latency": latency}
    return root or None


def load_dsv4_sparse_kernel_data(file_or_sources):
    """DSV4 sparse-kernel CSV (csa_attn / hca_attn / paged_mqa_logits).

    Thin wrapper over ``load_dsv4_sparse_op_data`` with the kernel key columns,
    yielding ``data[native_heads][tp_size][past_kv][isl][bs] = {"latency": ms}``.
    """
    return load_dsv4_sparse_op_data(file_or_sources, _SPARSE_KERNEL_KEYS)
