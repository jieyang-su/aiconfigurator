# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context + Generation attention ops (ISSUE-06 / AIC-543).

Both classes bind the engine's table views, SOL correction (generation
only — context attention has no SOL clamp in the legacy
``_correct_data``), and grid extrapolation.
``PerfDatabase.query_context_attention`` / ``query_generation_attention``
delegate here.

``ContextAttention.query`` keeps its three ``query_mem_op`` callers
(QK-norm, apply-RoPE, KV-write) pointed at ``database.query_mem_op`` —
deciding a long-term home for the analytical mem-op formula is deferred
to the post-refactor cleanup.

Cache key is ``(systems_root, system, backend, version,
enable_shared_layer)``, same as GEMM (and every other migrated op).
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, ClassVar

import aiconfigurator_core._aiconfigurator_core as _core
from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.attention_lanes import (
    UnsupportedAttentionBackendError,
    resolve_attention_lane_order,
    resolve_attention_override_lanes,
    split_attention_lane_tiers,
)
from aiconfigurator_core.sdk.operations.base import Operation, OpShellKit

if TYPE_CHECKING:
    from aiconfigurator_core.sdk.perf_database import PerfDatabase

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=256)
def _lane_order_cached(backend, version, sm_version, override, systems_root) -> tuple[str, ...]:
    """Memoized :func:`resolve_attention_lane_order`.

    The resolution reads a YAML map and builds a list; the engine-spec build
    needs it per attention op, so memoize on the full input tuple. One entry
    per (database identity x override) — a sweep resolves each order exactly
    once.
    """
    return resolve_attention_lane_order(backend, version, sm_version, override, systems_root)


def resolve_lane_order(database, override: str | None = None) -> tuple[str, ...]:
    """Attention lane precedence for *database* under an optional *override*.

    The override is the user-facing ``attention_backend`` knob carried by the
    op; everything else comes off the database handle (backend, version,
    ``sm_version``, systems root). ``"default"`` is always the last element.
    """
    sm_version = database.system_spec["gpu"].get("sm_version") or -1
    return _lane_order_cached(database.backend, database.version, sm_version, override, database.systems_root)


def lane_walk_order(density: dict[str, tuple[int, int]], lane_order: tuple[str, ...]) -> tuple[str, ...]:
    """The concrete walk order given each lane's ``(slice_count, row_count)``
    density: pinned lanes, then donors by density.

    ``density`` is ``{kernel_source: (slice_count, row_count)}`` for the
    REAL query-path table — fetched from the compiled engine via
    ``engine_table_view.fetch_attention_lane_density``
    (``perf_database/attention.rs::AttentionTable::context_lanes``/
    ``generation_lanes``), never the lane-blind Python enumeration view
    (``engine_table_view.fetch_table_view`` / ``table_view.rs``), which
    folds every kernel_source into one first-wins table for
    charts/support-matrix and cannot answer a density question at all.

    Three tiers, in order:

    1. **Pinned** — the override and the framework-default map lane, in the
       precedence the resolver produced. Never re-ordered: explicit intent wins.
       The boundary comes from the resolver itself (``LaneOrder.pinned_count``,
       read by ``split_attention_lane_tiers``) — a lane order that is not
       resolver-produced (hand-specified, or an already-expanded walk) is pinned
       in full and replayed verbatim.
    2. **Donor tier** — the remaining known lanes (plus ``"default"`` last, per
       the resolver's contract), ranked by measured coverage in THIS table
       (slices, then rows, then name) instead of alphabetically. Gap-fill should
       come from the data-richest lane: on gb200/sglang, plain ``sorted()`` let
       ``flashinfer`` (10 slices / 2 584 rows) preempt ``trtllm_mha`` (64 /
       31 141) on 5 context + 10 generation slices for no reason but its name.
    3. **Table leftovers** — lanes present in the table but outside the resolver
       vocabulary, same density ranking. The collected ``kernel_source`` labels
       are richer than the map (trtllm ships ``torch_flow*``, vllm ``vllm_*``,
       sglang also ``flash_attention``) and those backends have no ``"default"``
       lane at all, so without this tier none of their rows would be reachable.

    The ranking is a pure function of ``density``, so it is stable for a data
    set and identical at spec-build time — the ENGINE SPEC carries this
    extended order and the Rust twin replays it verbatim rather than
    re-deriving it.
    """
    if not density:
        return tuple(lane_order)
    pinned, donors = split_attention_lane_tiers(lane_order)

    def _rank(lane: str) -> tuple[int, int, str]:
        slices, rows = density.get(lane, (0, 0))
        return (-slices, -rows, lane)

    known = sorted((lane for lane in donors if lane != "default"), key=_rank)
    tail = ("default",) if "default" in donors else ()
    leftovers = sorted((lane for lane in density if lane not in lane_order), key=_rank)
    return pinned + tuple(known) + tail + tuple(leftovers)


def _source_tiered_lane_walk_order(
    density: dict[str, tuple[int, int]],
    primary_density: dict[str, tuple[int, int]],
    lane_order: tuple[str, ...],
) -> tuple[str, ...]:
    """Rank requested-version lanes before shared donor-version lanes.

    ``AttentionTable::by_lane`` merges sources under their bare lane labels,
    so the shared density alone cannot distinguish a requested-version lane
    from a denser inherited lane. Preserve the resolver's pinned intent first,
    then build requested-version and shared-only tiers with the same density
    ranking. Within a lane, Rust's existing first-source-wins fold still
    preserves requested-version rows when the same label also appears in a
    donor. A pinned lane remains first even when it exists only in a donor.
    """
    shared_order = lane_walk_order(density, lane_order)
    if not primary_density:
        return shared_order
    pinned, _ = split_attention_lane_tiers(lane_order)
    primary_order = tuple(
        lane for lane in lane_walk_order(primary_density, lane_order) if lane in primary_density and lane not in pinned
    )
    shared_only_order = tuple(lane for lane in shared_order if lane not in primary_density and lane not in pinned)
    return pinned + primary_order + shared_only_order


def resolved_lane_order_for_op(database, table_attr: str, override: str | None = None) -> list[str]:
    """Kernel-lane precedence for an attention op, RESOLVED python-side.

    Since the pyo3 op unification, ``ContextAttention``/``GenerationAttention``
    are constructed by the model layer WITHOUT a database handle (models are
    pure shape graphs; the database is only bound later, when
    ``engine.py::build_engine_spec_json`` walks a built model's op lists
    against a specific database — same place ``_wideep_moe`` pre-bakes its
    kernel_source). This is called from there, once the database is
    available, to set each attention op's ``_lane_order`` before
    serialization; it is NOT reachable from the op's own ``__init__``.

    ``table_attr`` is ``"_context_attention_data"`` or
    ``"_generation_attention_data"``. With no database, an unset or literal
    ``"default"`` override returns the always-valid ``["default"]``; a named
    override raises because it cannot be verified without database metadata.
    The engine spec must never carry an empty lane list.

    Table-aware extension (donor/leftover-lane density ranking) fires ONLY
    when there is EVIDENCE of intent — an explicit *override* (a non-empty
    pinned head), or a framework-default map entry for this exact (backend,
    floor-matched version, sm_version), including an entry whose lane is
    ``"default"`` (it pins nothing, yet it is a sourced statement about the
    framework default — e.g. vllm 0.24.0 on Blackwell). With NEITHER — no
    override and no map entry (unknown backend, unmapped shipped versions
    such as vllm 0.22.0/0.19.0, or a missing sm row) — donor density is no
    evidence of the framework default at all, so this FAILS CLOSED to the
    plain ``["default"]`` the pyo3 constructor already carries, relying only
    on the Rust-side ``lane_slice`` fallback (any other table lane, BTreeMap
    order) — unchanged from this op's behavior before AIC-1715/1716. Do not
    "fix" the unmapped case by extending the density walk to it; map the
    version with a verifiable source instead (PR #1519 review).

    An explicit named *override* is user intent: unsupported pairs and
    unexpected resolver/density failures both propagate rather than silently
    discarding it. Unset and literal ``"default"`` paths may safely degrade to
    ``["default"]``, with a WARNING so the fallback remains observable.
    """
    if database is None:
        if override not in (None, "default"):
            raise UnsupportedAttentionBackendError(
                f"attention_backend={override!r} cannot be resolved without an attention performance database"
            )
        return ["default"]
    try:
        order = resolve_lane_order(database, override)
        if getattr(order, "pinned_count", 0) == 0 and not getattr(order, "framework_default_matched", False):
            # No override, no framework-default map entry: fail closed (see
            # docstring) instead of density-ranking the whole vocabulary.
            return ["default"]
        from aiconfigurator_core.sdk.engine_table_view import fetch_attention_lane_density

        density = fetch_attention_lane_density(database, table_attr)
        if override not in (None, "default"):
            override_lanes = resolve_attention_override_lanes(database.backend, override)
            if not any(lane in density for lane in override_lanes):
                phase = "context" if table_attr == "_context_attention_data" else "generation"
                raise UnsupportedAttentionBackendError(
                    f"attention_backend={override!r} has no {phase} attention measurements for "
                    f"system {database.system!r}, backend {database.backend!r}, version {database.version!r}; "
                    f"expected at least one stored kernel_source lane from {list(override_lanes)!r}, "
                    f"available lanes: {sorted(density)!r}."
                )
        primary_density = fetch_attention_lane_density(database, table_attr, shared_layer=False)
        return list(_source_tiered_lane_walk_order(density, primary_density, order))
    except UnsupportedAttentionBackendError:
        raise
    except Exception:
        if override not in (None, "default"):
            raise
        logger.warning(
            "attention lane order unresolvable for %s; serializing the default-only order",
            table_attr,
            exc_info=True,
        )
        return ["default"]


# Extrapolation target grids — lifted verbatim from the legacy blocks in
# ``PerfDatabase.__init__`` so behavior stays bit-identical.

# fmt: on


def _cache_key(database: PerfDatabase) -> tuple:
    """Shared cache key — same shape as GEMM's, used by both Attention ops.

    TODO: hoist to ``operations/base.py`` once a third op family (Phase 3
    NCCL / MLA / Mamba) lands and needs the same key shape — preferring
    duplication over premature abstraction with only two callers.
    """
    return (
        database.systems_root,
        database.system,
        database.backend,
        database.version,
        database.enable_shared_layer,
    )


def generation_attn_mode(system_spec: dict, kvcache_quant_mode: common.KVCacheQuantMode) -> common.FMHAQuantMode:
    """Decode-attention FMHA mode implied by the kv-cache dtype.

    fp8 KV implies an fp8-MMA decode kernel only where fp8 tensor cores exist
    (SM >= 89, Ada and newer); on pre-89 hardware — and on specs without
    ``sm_version``, e.g. XPU — the kernel dequantizes KV and issues the MMA on
    the bf16 pipeline. That is how a100's shipped fp8-kv generation data was
    collected in the first place, so gating here keeps that silicon usable
    under the strict per-dtype resolution. The single home for the derivation
    rule (mirrors Rust ``perf_database::attention::generation_attn_mode``).
    """
    from aiconfigurator_core.sdk.system_spec import supports_fp8_mma

    has_fp8_mma = supports_fp8_mma(system_spec)
    if kvcache_quant_mode == common.KVCacheQuantMode.fp8 and has_fp8_mma:
        return common.FMHAQuantMode.fp8
    return common.FMHAQuantMode.bfloat16


def generation_attn_flops(system_spec: dict, kvcache_quant_mode: common.KVCacheQuantMode) -> float:
    """Strictly resolved TC FLOPS for :func:`generation_attn_mode`; used by
    the generation get_sol closures, the eager query-entry checks, and
    ``Attention._correct_sol``.
    """
    return common.get_quant_tc_flops(system_spec, generation_attn_mode(system_spec, kvcache_quant_mode))


class ContextAttention(Operation):
    """
    Context (prefill) attention operation.

    Owns ``_data_cache: {key: LoadedOpData}`` for the context attention CSV —
    raw as-collected rows (no load-time clamp or grid pre-expansion; the
    engine owns interpolation and the SOL floor).
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
        """Idempotent. Fetches the engine's context_attention table view
        (raw rows) and binds ``database._context_attention_data``,
        respecting any pre-set test override."""
        from aiconfigurator_core.sdk.engine_table_view import load_view
        from aiconfigurator_core.sdk.perf_database import PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            cls._data_cache[key] = load_view(database, "_context_attention_data", PerfDataFilename.context_attention)
            cls._record_load()

        # Bind instance attr (respect intentional test pre-overrides).
        if "_context_attention_data" not in database.__dict__:
            database._context_attention_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_context_attention)
    # ------------------------------------------------------------------

    @classmethod
    def _query_context_attention_table(
        cls,
        database: PerfDatabase,
        b: int,
        s: int,
        prefix: int,
        n: int,
        n_kv: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        database_mode: common.DatabaseMode | None = None,
        window_size: int = 0,
        head_size: int = 128,
    ):
        """Query context attention table. Verbatim port of the legacy body."""
        # Strict eager resolution (parity with the Rust engine, which resolves
        # flops with `?` at query entry): reject a missing *_tc_flops entry up
        # front — a SILICON exact hit never invokes the get_sol closure.
        common.get_quant_tc_flops(database.system_spec, fmha_quant_mode)

        def get_sol(
            b: int,
            s: int,
            prefix: int,
            n: int,
            n_kv: int,
            h: int,
            w: int,
            kvcache_quant_mode: common.KVCacheQuantMode,
            fmha_quant_mode: common.FMHAQuantMode,
        ) -> tuple[float, float, float]:
            full_s = s + prefix
            if w > 0 and full_s > w:
                ops = 2 * b * (full_s - prefix) * w * n * h * 2
            else:
                ops = 2 * b * (full_s * full_s - prefix * prefix) * n * h * 2 / 2
            mem_bytes = 2 * b * (
                n * (full_s - prefix) * h + n * (full_s - prefix) * h
            ) + kvcache_quant_mode.value.memory * b * (2 * n_kv * full_s * h)
            sol_math = ops / common.get_quant_tc_flops(database.system_spec, fmha_quant_mode) * 1000
            sol_mem = mem_bytes / database.system_spec["gpu"]["mem_bw"] * 1000
            sol_time = max(sol_math, sol_mem)
            return sol_time, sol_math, sol_mem

        def get_empirical(
            b: int,
            s: int,
            prefix: int,
            n: int,
            n_kv: int,
            head_size: int,
            window_size: int,
            kvcache_quant_mode: common.KVCacheQuantMode,
            fmha_quant_mode: common.FMHAQuantMode,
        ) -> float:
            # latency = SOL / util, util read best-effort from collected data. The query
            # SOL always uses the real window_size (get_sol's windowed branch already
            # captures the reduced O(seq*window) work). The UTIL is borrowed by slice:
            # exact window first, then window=0 (full attention) -- the window axis is
            # collected per-model (scattered: hs64/win128 gpt-oss, hs256/win1024 gemma3,
            # ...), so an uncollected window has no own basis; full-attention util is the
            # right carrier since the kernel efficiency is ~window-independent and SOL
            # absorbs the work difference.
            sol_time = get_sol(b, s, prefix, n, n_kv, head_size, window_size, kvcache_quant_mode, fmha_quant_mode)[0]
            n_kv_lookup = 0 if n == n_kv else n_kv

            def _own_grid(slice_window):
                def _slice():
                    cls.load_data(database)
                    wrapper = database._context_attention_data
                    wrapper.raise_if_not_loaded()
                    return util_empirical.require_data_slice(
                        wrapper,
                        fmha_quant_mode,
                        kvcache_quant_mode,
                        n_kv_lookup,
                        head_size,
                        slice_window,
                    )

                def _sol(c):  # c = (n, full_s, b); samples are full attention (prefix=0)
                    nkv = c[0] if n_kv_lookup == 0 else n_kv_lookup
                    return get_sol(
                        c[2], c[1], 0, c[0], nkv, head_size, slice_window, kvcache_quant_mode, fmha_quant_mode
                    )[0]

                return util_empirical.grid_for(
                    (
                        "ctx_attn",
                        database.system,
                        database.backend,
                        database.version,
                        fmha_quant_mode.name,
                        kvcache_quant_mode.name,
                        n_kv_lookup,
                        head_size,
                        slice_window,
                    ),
                    _slice,
                    _sol,
                    depth=3,
                )

            # Try exact window, then full-attention (window=0) as the util carrier.
            for slice_window in [window_size, 0] if window_size > 0 else [window_size]:
                grid = _own_grid(slice_window)
                if grid is not None and grid.samples:
                    latency, _ = util_empirical.estimate(sol_time, (n, s + prefix, b), grid)
                    return latency
                # Cross-head_size transfer (XSHAPE): this head_size has no data, but
                # num_heads is already an in-grid axis, so only head_size differs. Borrow
                # the nearest collected head_size's util and rescale by the prefill
                # head_size-util ratio (SOL still uses the query's own head_size).
                ref_grid, ref_hs = (
                    _ctx_headsize_ref_grid(
                        database, fmha_quant_mode, kvcache_quant_mode, n_kv_lookup, head_size, slice_window, get_sol
                    )
                    if common.TransferKind.XSHAPE in database.transfer_policy
                    else (None, None)
                )
                if ref_grid is not None:
                    scale = _attn_prefill_hs_ratio(database.backend, head_size) / _attn_prefill_hs_ratio(
                        database.backend, ref_hs
                    )
                    latency, _ = util_empirical.estimate(
                        sol_time, (n, s + prefix, b), ref_grid, util_scale=scale, provenance="xshape"
                    )
                    return latency

            # No own-window, full-attention, or cross-head basis -> raise honestly.
            latency, _ = util_empirical.estimate(sol_time, (n, s + prefix, b), None)
            return latency

        assert n_kv <= n, "n_kv must be less than or equal to n"

        if database_mode is None:
            database_mode = database._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            sol_latency = get_sol(b, s, prefix, n, n_kv, head_size, window_size, kvcache_quant_mode, fmha_quant_mode)[0]
            return PerformanceResult(sol_latency, energy=0.0, source="sol")
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(b, s, prefix, n, n_kv, head_size, window_size, kvcache_quant_mode, fmha_quant_mode)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            emp_latency = get_empirical(
                b, s, prefix, n, n_kv, head_size, window_size, kvcache_quant_mode, fmha_quant_mode
            )
            return PerformanceResult(emp_latency, energy=0.0, source="empirical")
        elif database_mode == common.DatabaseMode.ANALYTICAL:
            import math

            from aiconfigurator_core.sdk.kernelsim.analytical import attention_latency_ms

            if s <= 0:
                return PerformanceResult(0.0, energy=0.0, source="analytical")
            full_s = s + prefix
            kv_length = max(s, min(full_s, window_size) if window_size > 0 else full_s)
            # Compute precision and cache representation are independent in the
            # analytical FA model.  FP8 KV does not imply FP8 FMHA: when the
            # request is FP8-KV + BF16-FMHA, keep BF16 matrix math and pass the
            # physically smaller KV stream separately.
            dtype = "fp8" if fmha_quant_mode.value.compute_dtype == "fp8" else "bf16"
            kv_cache_bytes_per_token = 2 * head_size * kvcache_quant_mode.value.memory
            return PerformanceResult(
                attention_latency_ms(
                    system=database.system,
                    gpu=database.system_spec["gpu"],
                    batch=b,
                    query_length=math.ceil(s),
                    kv_length=math.ceil(kv_length),
                    query_heads=n,
                    kv_heads=n_kv,
                    head_dim=head_size,
                    dtype=dtype,
                    kv_cache_bytes_per_token=kv_cache_bytes_per_token,
                    config=database._analytical_config,
                ),
                energy=0.0,
                source="analytical",
            )

        cls.load_data(database)
        data_wrapper = database._context_attention_data

        def get_silicon():
            data_wrapper.raise_if_not_loaded()
            full_s = s + prefix
            prefix_correction = (full_s * full_s - prefix * prefix) / (full_s * full_s)
            n_kv_lookup = 0 if n == n_kv else n_kv
            # Use the real windowed slice when present -- validation shows it beats a
            # window=0 + SOL-ratio reconstruction (the latter is ~25-77% off vs measured
            # windowed data). When the windowed slice is absent or too sparse to
            # interpolate, perf_interp fails accurately (raises) and HYBRID/EMPIRICAL
            # fall back to get_empirical's window=0 + SOL derivation.
            attention_dict = util_empirical.require_data_slice(
                data_wrapper,
                fmha_quant_mode,
                kvcache_quant_mode,
                n_kv_lookup,
                head_size,
                window_size,
            )
            # Resolve on the raw (n, full_s, batch) grid: sqrt-space blend for the
            # ~seq^2 curvature; past the staircase frontier (large seq x large
            # batch, uncollected) the engine holds the boundary util and lets the
            # SOL carry the growth. Samples are full attention, so the sol_fn is
            # evaluated at prefix=0 with the slice's own kv-head/window setup.
            config = perf_interp.context_attention_config(
                sol_fn=lambda n_v, s_v, b_v: get_sol(
                    b_v,
                    s_v,
                    0,
                    n_v,
                    n_v if n_kv_lookup == 0 else n_kv_lookup,
                    head_size,
                    window_size,
                    kvcache_quant_mode,
                    fmha_quant_mode,
                )[0]
            )
            result = perf_interp.query(config, attention_dict, n, full_s, b)
            latency = perf_interp.get_value(result, "latency") * prefix_correction
            energy = perf_interp.get_value(result, "energy") * prefix_correction
            return database._interp_pr(latency, energy=energy)

        return database._query_silicon_or_hybrid(
            get_silicon=get_silicon,
            get_empirical=lambda: get_empirical(
                b, s, prefix, n, n_kv, head_size, window_size, kvcache_quant_mode, fmha_quant_mode
            ),
            database_mode=database_mode,
            error_msg=(
                f"Failed to query context attention data for {b=}, {s=}, {prefix=}, {n=}, {n_kv=}, "
                f"{head_size=}, {window_size=}, {kvcache_quant_mode=}, {fmha_quant_mode=}"
            ),
        )

    # ------------------------------------------------------------------
    # Op contract: query() + get_weights()
    # ------------------------------------------------------------------


class GenerationAttention(_core.GenerationAttention, OpShellKit):
    """
    Generation (decode) attention operation.

    Owns the SILICON row cache (raw as-collected; the load-time SOL clamp
    and grid expansion retired with #1357 PR-5 — the engine owns both) plus
    the raw-cache alias kept for its historical consumers.
    """

    _data_cache: ClassVar[dict] = {}
    _raw_data_cache: ClassVar[dict] = {}

    def __init__(
        self,
        name: str,
        scale_factor: float,
        n: int,
        n_kv: int,
        kv_cache_dtype: common.KVCacheQuantMode,
        window_size: int = 0,
        head_size: int = 128,
        use_qk_norm: bool = False,
        fmha_quant_mode: common.FMHAQuantMode | None = None,
    ) -> None:
        """Initialize generation attention query parameters."""
        super().__init__(name, scale_factor)
        self._n = n
        self._weights = 0.0
        self._n_kv = n_kv
        self._kv_cache_dtype = kv_cache_dtype
        # Generation silicon tables are keyed only by KV dtype, so legacy
        # callers can leave this unset. Analytical callers pass the configured
        # FMHA mode explicitly to separate compute from cache representation.
        self._fmha_quant_mode = fmha_quant_mode
        self._window_size = window_size
        self._head_size = head_size
        self._use_qk_norm = use_qk_norm

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Fetches the engine's generation_attention table view
        (raw rows) and binds both database views.

        Mirrors ``GEMM.load_data``: loading operates on the
        canonical class-cache value (passed explicitly), then the instance
        attr is bound, respecting any pre-set test override."""
        from aiconfigurator_core.sdk.engine_table_view import load_view
        from aiconfigurator_core.sdk.perf_database import PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            cls._data_cache[key] = load_view(
                database, "_generation_attention_data", PerfDataFilename.generation_attention
            )
            # The raw wrapper stays a plain alias of the table (no load-time
            # grid expansion since PR-5).
            cls._raw_data_cache[key] = cls._data_cache[key]
            cls._record_load()

        # Bind instance attr (respect intentional test pre-overrides).
        if "_generation_attention_data" not in database.__dict__:
            database._generation_attention_data = cls._data_cache[key]
            database._raw_generation_attention_data = cls._raw_data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()
        cls._raw_data_cache.clear()

    # NOTE(#1357 PR-5): the load-time SOL clamp (`_correct_sol`) retired with
    # the Python query math. The loaded table is now the RAW collected data
    # plane (enumeration/charts); the compiled engine applies the same clamp
    # to its own load (see perf_database/attention.rs), so QUERY values stay
    # SOL-floored via the single oracle.

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_generation_attention)
    # ------------------------------------------------------------------

    @classmethod
    def _query_generation_attention_table(
        cls,
        database: PerfDatabase,
        b: int,
        s: int,
        n: int,
        n_kv: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        database_mode: common.DatabaseMode | None = None,
        window_size: int = 0,
        head_size: int = 128,
        fmha_quant_mode: common.FMHAQuantMode | None = None,
    ):
        """Query generation attention table. Verbatim port of legacy body."""
        # Strict eager resolution (parity with the Rust engine, which resolves
        # flops with `?` at query entry): reject a missing *_tc_flops entry up
        # front — a SILICON exact hit never invokes the get_sol closure.
        if database_mode is None:
            database_mode = database._default_database_mode
        if database_mode == common.DatabaseMode.ANALYTICAL and fmha_quant_mode is not None:
            common.get_quant_tc_flops(database.system_spec, fmha_quant_mode)
        else:
            generation_attn_flops(database.system_spec, kvcache_quant_mode)

        def get_sol(
            b: int,
            s: int,
            n: int,
            n_kv: int,
            h: int,
            w: int,
            kvcache_quant_mode: common.KVCacheQuantMode,
        ) -> tuple[float, float, float]:
            if w > 0:
                kv_len = min(s - 1, w)
            else:
                kv_len = s - 1
            ops = 2 * b * n * h * 2 * (kv_len)
            mem_bytes = b * (n * h * 2 + 2 * n_kv * (kv_len) * h * kvcache_quant_mode.value.memory + n * h * 2)

            sol_math = ops / generation_attn_flops(database.system_spec, kvcache_quant_mode) * 1000
            sol_mem = mem_bytes / database.system_spec["gpu"]["mem_bw"] * 1000
            sol_time = max(sol_math, sol_mem)
            return sol_time, sol_math, sol_mem

        def get_empirical(
            b: int,
            s: int,
            n: int,
            n_kv: int,
            h: int,
            w: int,
            kvcache_quant_mode: common.KVCacheQuantMode,
        ) -> float:
            # latency = SOL / util. The query SOL uses the real window w (get_sol caps
            # kv_len at the window); the UTIL is borrowed by slice -- exact window first,
            # then window=0 (full attention). The window axis is collected per-model
            # (scattered), so an uncollected window borrows full-attention util: decode
            # is memory-bound KV read, ~window-independent in efficiency, and SOL absorbs
            # the smaller windowed KV span.
            sol_time = get_sol(b, s, n, n_kv, h, w, kvcache_quant_mode)[0]
            n_kv_lookup = n_kv if n_kv != n else 0

            def _own_grid(slice_window):
                def _slice():
                    cls.load_data(database)
                    wrapper = getattr(database, "_raw_generation_attention_data", None)
                    if wrapper is None:
                        raise PerfDataNotAvailableError("Raw generation attention data is not loaded.")
                    wrapper.raise_if_not_loaded()
                    return util_empirical.require_data_slice(
                        wrapper,
                        kvcache_quant_mode,
                        n_kv_lookup,
                        h,
                        slice_window,
                    )

                def _sol(c):  # c = (n, b, s)
                    nkv = c[0] if n_kv_lookup == 0 else n_kv_lookup
                    return get_sol(c[1], c[2], c[0], nkv, h, slice_window, kvcache_quant_mode)[0]

                return util_empirical.grid_for(
                    (
                        "gen_attn",
                        database.system,
                        database.backend,
                        database.version,
                        kvcache_quant_mode.name,
                        n_kv_lookup,
                        h,
                        slice_window,
                    ),
                    _slice,
                    _sol,
                    depth=3,
                )

            for slice_window in [w, 0] if w > 0 else [w]:
                grid = _own_grid(slice_window)
                if grid is not None and grid.samples:
                    latency, _ = util_empirical.estimate(sol_time, (n, b, s), grid)
                    return latency
                # Cross-head_size transfer (XSHAPE, decode): borrow the nearest collected
                # head_size's util. Decode util is ~head_size-independent (memory-bound
                # KV read), so util_scale stays 1.0 -- no prefill-style correction.
                ref_grid, _ref_hs = (
                    _gen_headsize_ref_grid(database, kvcache_quant_mode, n_kv_lookup, h, slice_window, get_sol)
                    if common.TransferKind.XSHAPE in database.transfer_policy
                    else (None, None)
                )
                if ref_grid is not None:
                    latency, _ = util_empirical.estimate(sol_time, (n, b, s), ref_grid, provenance="xshape")
                    return latency

            latency, _ = util_empirical.estimate(sol_time, (n, b, s), None)
            return latency

        assert n_kv <= n, "n_kv must be less than or equal to n"

        if database_mode == common.DatabaseMode.SOL:
            sol_latency = get_sol(b, s, n, n_kv, head_size, window_size, kvcache_quant_mode)[0]
            return PerformanceResult(sol_latency, energy=0.0, source="sol")
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(b, s, n, n_kv, head_size, window_size, kvcache_quant_mode)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            emp_latency = get_empirical(b, s, n, n_kv, head_size, window_size, kvcache_quant_mode)
            return PerformanceResult(emp_latency, energy=0.0, source="empirical")
        elif database_mode == common.DatabaseMode.ANALYTICAL:
            from aiconfigurator_core.sdk.kernelsim.analytical import attention_latency_ms

            kv_length = max(1, min(s, window_size) if window_size > 0 else s)
            effective_fmha = fmha_quant_mode or generation_attn_mode(database.system_spec, kvcache_quant_mode)
            dtype = "fp8" if effective_fmha.value.compute_dtype == "fp8" else "bf16"
            kv_cache_bytes_per_token = 2 * head_size * kvcache_quant_mode.value.memory
            return PerformanceResult(
                attention_latency_ms(
                    system=database.system,
                    gpu=database.system_spec["gpu"],
                    batch=b,
                    query_length=1,
                    kv_length=kv_length,
                    query_heads=n,
                    kv_heads=n_kv,
                    head_dim=head_size,
                    dtype=dtype,
                    kv_cache_bytes_per_token=kv_cache_bytes_per_token,
                    config=database._analytical_config,
                ),
                energy=0.0,
                source="analytical",
            )

        cls.load_data(database)
        data_wrapper = database._generation_attention_data

        def get_silicon():
            data_wrapper.raise_if_not_loaded()
            n_kv_lookup = n_kv if n_kv != n else 0

            # Use the real windowed slice when present (more accurate than a window=0 +
            # SOL reconstruction); when absent/too sparse, perf_interp raises and
            # HYBRID/EMPIRICAL fall back to get_empirical's window=0 + SOL derivation.
            attention_dict = util_empirical.require_data_slice(
                data_wrapper,
                kvcache_quant_mode,
                n_kv_lookup,
                head_size,
                window_size,
            )
            # Generation is ~linear in seq -> RAW grid resolve on the raw table.
            # The +-10% seq-sample averaging is op-level smoothing (decode s
            # drifts across a request) and is kept as-is.
            config = perf_interp.generation_attention_config(
                sol_fn=lambda n_v, b_v, s_v: get_sol(
                    b_v,
                    s_v,
                    n_v,
                    (n_v if n_kv_lookup == 0 else n_kv_lookup),
                    head_size,
                    window_size,
                    kvcache_quant_mode,
                )[0]
            )
            s_min = max(1, int(s * 0.9))
            s_max = max(s_min, int(s * 1.1))
            sample_cnt = 5
            s_samples = [s_min + (s_max - s_min) * i // (sample_cnt - 1) for i in range(sample_cnt)]

            latency_sum = 0.0
            energy_sum = 0.0
            for s_i in s_samples:
                r = perf_interp.query(config, attention_dict, n, b, s_i)
                latency_sum += perf_interp.get_value(r, "latency")
                energy_sum += perf_interp.get_value(r, "energy")

            latency = latency_sum / sample_cnt
            energy = energy_sum / sample_cnt
            return database._interp_pr(latency, energy=energy)

        return database._query_silicon_or_hybrid(
            get_silicon=get_silicon,
            get_empirical=lambda: get_empirical(b, s, n, n_kv, head_size, window_size, kvcache_quant_mode),
            database_mode=database_mode,
            error_msg=(
                f"Failed to query generation attention data for {b=}, {s=}, {n=}, {n_kv=}, "
                f"{head_size=}, {window_size=}, {kvcache_quant_mode=}"
            ),
        )

    # ------------------------------------------------------------------
    # Op contract: query() + get_weights()
    # ------------------------------------------------------------------


        result = database.query_generation_attention(
            batch_size,
            s,
            self._n,
            self._n_kv,
            self._kv_cache_dtype,
            window_size=self._window_size,
            head_size=self._head_size,
            fmha_quant_mode=self._fmha_quant_mode,
        )
        gen_seq_imbalance_correction_scale = float(
            kwargs.get(
                "gen_seq_imbalance_correction_scale",
                kwargs.get("seq_imbalance_correction_scale", 1.0),
            )
        )
        if gen_seq_imbalance_correction_scale != 1.0:
            result = result * gen_seq_imbalance_correction_scale
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class EncoderAttention(Operation):
    """
    Non-causal encoder attention: full N^2, MHA, no KV cache, optional partial RoPE.

    Used to model bidirectional encoders — ViT (vision), audio encoders, and any
    other omni-modal encoder where the kernel runs full N^2 attention without a
    causal mask and without writing a KV cache. The optional
    ``partial_rotary_factor`` accounts for partial-rotation RoPE variants such as
    Qwen3-VL (factor=0.5, rotating half of head_dim). Defaults to 0.0 (no RoPE),
    matching CLIP / SigLIP / Whisper; set to 0.5 / 1.0 only for RoPE encoders.

    Owns ``_data_cache: {key: LoadedOpData}`` for the encoder attention CSV.
    Schema is simpler than context attention: MHA only (no n_kv), no KV cache
    (no kvcache_quant_mode), no sliding window. No SOL clamp. Grid extrapolation
    resolves on the raw grid via the engine's interpolation.
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
        """Idempotent. Fetches the engine's encoder_attention table view
        (raw rows), binds ``database._encoder_attention_data``.
        """
        from aiconfigurator_core.sdk.engine_table_view import load_view
        from aiconfigurator_core.sdk.perf_database import PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            cls._data_cache[key] = load_view(database, "_encoder_attention_data", PerfDataFilename.encoder_attention)
            cls._record_load()

        # Bind instance attr (respect intentional test pre-overrides).
        if "_encoder_attention_data" not in database.__dict__:
            database._encoder_attention_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_encoder_attention)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Op contract: query() + get_weights()
    # ------------------------------------------------------------------
