# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GEMM operation over the engine table views (gemm, compute_scale, scale_matrix).

GEMM owns its three CSV-backed raw tables (data plane only — per-op
values come from the compiled engine, #1357 PR-5).
``PerfDatabase.query_compute_scale / query_scale_matrix`` are tombstoned;
``query_gemm`` is an engine-routed deprecation shim.

Lazy-load Pattern A: consumers trigger ``load_data`` on cache miss. ``_data_cache`` /
``_compute_scale_cache`` / ``_scale_matrix_cache`` are keyed by
``(systems_root, system, backend, version, enable_shared_layer)`` so the
same op class serves multiple databases in one process. ``systems_root``
is part of the key because test fixtures swap to a fresh ``tmp_path``
between tests and must get distinct cache entries.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

import aiconfigurator_core
import aiconfigurator_core._aiconfigurator_core as _core
from aiconfigurator_core.sdk.operations import util_empirical
from aiconfigurator_core.sdk.operations.base import OpShellKit

if TYPE_CHECKING:
    from aiconfigurator_core.sdk.perf_database import PerfDatabase

logger = logging.getLogger(__name__)


# Per-quant achieved-util LEVEL e(q) for GEMM, keyed by the (memory, compute)
# profile — the GEMM counterpart of moe.py's _MOE_QUANT_UTIL_LEVEL, consumed
# ONLY by the cross-PROFILE relation of the quant-transfer primitive as the
# ratio e(query)/e(ref) (see util_empirical.quant_transfer_grid). A SINGLE
# scalar per profile by design (the per-component split was validated as
# untrustworthy on MoE; the same SOL-attribution argument holds here).
#
# [data] rows: median of util = SOL/measured over the clearly compute-bound
# region (m >= 1024, n >= 2048, k >= 2048) of every collected gemm table on
# b200/h200/h100 x trtllm/vllm/sglang (2026-08 snapshot); the range across
# those six stacks is quoted per row. The bf16/fp8 level RATIO spans
# 1.38-2.00 (~±20% around 1.6) — looser than MoE's ~10% but acceptable for
# the last-resort relation. LOO on the mechanism (predict a collected quant
# from its nearest-profile sibling at shared shapes, m >= 64): nvfp4 <- fp8
# = 22-33% MAPE, comparable to MoE's ~24% xprofile LOO. [inferred] rows
# follow the structure (efficiency drops with weight precision, mildly
# recovers as activation precision drops); levels are relative and tunable —
# only ratios are consumed.
# PROJECTION of the engine's table (PR-6): the Rust
# `operators/gemm.rs::GEMM_QUANT_UTIL_LEVEL` is the single source (with the
# same per-row [data]/[inferred] provenance notes); rebuilding the dict from
# the FFI ends the two-sided sync discipline every new quant used to need.
_GEMM_QUANT_UTIL_LEVEL: dict[tuple[float, float], float] = {
    (memory, compute): level for memory, compute, level in aiconfigurator_core.gemm_quant_util_levels()
}
_GEMM_QUANT_UTIL_DEFAULT = 0.45  # unlisted profile: mid-range relative level


def xprofile_util_level_known(quant_mode) -> bool:
    """Whether the GEMM util-LEVEL table lists this quant's profile.

    The runtime ladder falls back to ``_GEMM_QUANT_UTIL_DEFAULT`` for
    unlisted profiles; the validate gate deliberately does NOT (admitting a
    quant nobody calibrated would hide the missing level line the
    add-a-quant recipe requires), so it asks this instead of reaching into
    the table."""
    return util_empirical.quant_profile(quant_mode) in _GEMM_QUANT_UTIL_LEVEL


class GEMM(_core.GEMM, OpShellKit):
    """
    GEMM operation with power tracking (Rust-backed; see py_ops.rs).

    Owns three CSV-backed tables:
    - ``_data_cache``: gemm latency/energy keyed by ``quant_mode -> m -> n -> k``
    - ``_compute_scale_cache``: compute_scale latency/energy keyed by ``quant_mode -> m -> k``
    - ``_scale_matrix_cache``: scale_matrix latency/energy keyed by ``quant_mode -> m -> k``

    All three are class-level dicts keyed by
    ``(systems_root, system, backend, version, enable_shared_layer)``.
    """

    _data_cache: ClassVar[dict] = {}
    _compute_scale_cache: ClassVar[dict] = {}
    _scale_matrix_cache: ClassVar[dict] = {}

    # ------------------------------------------------------------------
    # Data ownership: load + cache + clear
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        """Cache key uniquely identifying the loaded data set.

        ``systems_root`` is included so test fixtures that swap to a fresh
        ``tmp_path`` between tests get distinct entries (otherwise the
        shared-layer test suite collides). ``enable_shared_layer`` is also
        part of the key because HYBRID unions sibling-row inheritance, so
        a SILICON-only load and a HYBRID load produce different dicts.
        """
        return (
            database.systems_root,
            database.system,
            database.backend,
            database.version,
            database.enable_shared_layer,
        )

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. On cache miss: fetches the three engine table views
        (raw as-collected rows in the retired parsers' exact shape — the
        engine owns parsing, clamping and interpolation) and records the
        load. Always: binds
        ``database._gemm_data``/``_compute_scale_data``/``_scale_matrix_data``
        to the cached wrappers.

        Tests that have already set those instance attributes (e.g.
        ``db._gemm_data = LoadedOpData(...)``) are respected — the binds
        below are gated on ``"_gemm_data" not in database.__dict__`` so
        intentional overrides survive."""
        from aiconfigurator_core.sdk.engine_table_view import load_view
        from aiconfigurator_core.sdk.perf_database import PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache or key not in cls._compute_scale_cache or key not in cls._scale_matrix_cache:
            # Fetch all three into locals first so a failure on the second or
            # third view doesn't leave the cache half-populated (which would
            # let a subsequent ``key in cls._data_cache`` early-out skip past
            # the missing siblings and crash downstream).
            gemm_loaded = load_view(database, "_gemm_data", PerfDataFilename.gemm)
            compute_scale_loaded = load_view(database, "_compute_scale_data", PerfDataFilename.compute_scale)
            scale_matrix_loaded = load_view(database, "_scale_matrix_data", PerfDataFilename.scale_matrix)

            # All three loads succeeded — commit atomically so partially-
            # populated cache state can never be observed.
            cls._data_cache[key] = gemm_loaded
            cls._compute_scale_cache[key] = compute_scale_loaded
            cls._scale_matrix_cache[key] = scale_matrix_loaded

            cls._record_load()

        # Bind instance attrs (respect intentional test pre-overrides).
        if "_gemm_data" not in database.__dict__:
            database._gemm_data = cls._data_cache[key]
        if "_compute_scale_data" not in database.__dict__:
            database._compute_scale_data = cls._compute_scale_cache[key]
        if "_scale_matrix_data" not in database.__dict__:
            database._scale_matrix_data = cls._scale_matrix_cache[key]
        return

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all three GEMM caches plus base-class state."""
        cls._data_cache.clear()
        cls._compute_scale_cache.clear()
        cls._scale_matrix_cache.clear()
        query = cls.__dict__.get("query")
        if query is not None and hasattr(query, "cache_clear"):
            query.cache_clear()

    @classmethod
    def supported_quant_modes(cls, database: PerfDatabase) -> set:
        """Return the quant modes for which loaded GEMM data is available.

        Triggers ``load_data`` on first call so the answer reflects what
        actually loaded for this database."""
        cls.load_data(database)
        gemm_data = cls._data_cache.get(cls._cache_key(database))
        if gemm_data is None or not gemm_data.loaded:
            return set()
        return set(gemm_data.keys())

    # ------------------------------------------------------------------
    # Static helpers (shared with perf_database.py callers)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # SOL correction (formerly in PerfDatabase._correct_data)
    # ------------------------------------------------------------------

    # NOTE(#1357 PR-5): the load-time SOL clamp (`_correct_sol`) retired with
    # the Python query math. The loaded table is now the RAW collected data
    # plane (enumeration/charts); the compiled engine applies the same clamp
    # to its own load (see perf_database/gemm.rs), so QUERY values stay
    # SOL-floored via the single oracle.

    # ------------------------------------------------------------------
    # Table query classmethods (formerly PerfDatabase.query_*)
    # ------------------------------------------------------------------

    @classmethod
    def _query_gemm_table(
        cls,
        database: PerfDatabase,
        m: int,
        n: int,
        k: int,
        quant_mode: common.GEMMQuantMode,
        database_mode: common.DatabaseMode | None = None,
    ):
        """Query GEMM table — preserves PR #721 exact-hit → 1D → 3D fast path."""
        # Strict eager resolution (parity with the Rust engine, which resolves
        # flops with `?` at query entry): reject a missing *_tc_flops entry up
        # front — a SILICON exact hit never invokes the get_sol closure.
        common.get_quant_tc_flops(database.system_spec, quant_mode)

        def get_sol(m_v: int, n_v: int, k_v: int, qm: common.GEMMQuantMode) -> tuple[float, float, float]:
            tc_flops = common.get_quant_tc_flops(database.system_spec, qm)
            sol_math = 2 * m_v * n_v * k_v / tc_flops * 1000
            sol_mem = (
                qm.value.memory * (m_v * n_v + m_v * k_v + n_v * k_v) / database.system_spec["gpu"]["mem_bw"] * 1000
            )
            sol_time = max(sol_math, sol_mem)
            return sol_time, sol_math, sol_mem

        def get_empirical(m_v: int, n_v: int, k_v: int, qm: common.GEMMQuantMode) -> float:
            # SOL / util, where util is read best-effort from this op's own
            # collected data; when the quant has none, the shared
            # quant-transfer primitive borrows a sibling quant's util grid
            # (xquant same-profile / xprofile cross-profile). Raises
            # EmpiricalNotImplementedError only when no relation finds data.
            sol_time = get_sol(m_v, n_v, k_v, qm)[0]
            tqm = cls._normalize_gemm_quant_mode_for_table(qm)

            def _slice():
                cls.load_data(database)
                wrapper = database._gemm_data
                wrapper.raise_if_not_loaded()
                return util_empirical.require_data_slice(wrapper, tqm)  # m -> n -> k -> leaf

            grid = util_empirical.grid_for(
                ("gemm", database.system, database.backend, database.version, tqm.name),
                _slice,
                lambda c: get_sol(c[0], c[1], c[2], qm)[0],
                depth=3,
            )

            util_scale = 1.0
            prov = "empirical"  # own-quant util grid; relations below override
            if grid is None or not grid.samples:
                cls.load_data(database)
                wrapper = database._gemm_data
                policy = database.transfer_policy

                def _collect(q, sol_q, provenance):
                    # GEMM's xshape relation class is structurally EMPTY: the
                    # own-quant grid above is already depth-3 over every
                    # collected (m, n, k) — there is no "other slice of the
                    # same quant" left to borrow, so a same-quant candidate
                    # could only rebuild the identical (empty) sample set.
                    if provenance == "xshape":
                        return []
                    # One candidate per sibling quant: its whole m->n->k
                    # table. GEMM has no categorical slice features, so
                    # features are constant and the pooled xquant selection
                    # degrades to first-in-table (file row) order.
                    return [
                        util_empirical.ReferenceCandidate(
                            features=(1.0,),
                            node=wrapper[q],
                            sol_fn=(lambda c, _sq=sol_q: get_sol(c[0], c[1], c[2], _sq)[0]),
                            provenance=provenance,
                        )
                    ]

                grid, util_scale, ref_prov = util_empirical.quant_transfer_grid(
                    "gemm",
                    (database.system, database.backend, database.version, tqm.name),
                    (1.0,),
                    policy,
                    tqm,
                    wrapper,
                    _collect,
                    _gemm_quant_util_level,
                    depth=3,
                    selection_key=(id(wrapper), policy),
                    # weight-only must borrow bf16, never tie-break into fp8
                    # (rationale on xprofile_quant_order)
                    prefer_same_compute=True,
                )
                if ref_prov:
                    prov = ref_prov

            latency, _ = util_empirical.estimate(
                sol_time, (m_v, n_v, k_v), grid, util_scale=util_scale, provenance=prov
            )
            return latency

        if database_mode is None:
            database_mode = database._default_database_mode

        table_quant_mode = cls._normalize_gemm_quant_mode_for_table(quant_mode)

        if database_mode == common.DatabaseMode.SOL:
            return PerformanceResult(get_sol(m, n, k, quant_mode)[0], energy=0.0, source="sol")
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(m, n, k, quant_mode)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            return PerformanceResult(get_empirical(m, n, k, quant_mode), energy=0.0, source="empirical")
        elif database_mode == common.DatabaseMode.ANALYTICAL:
            from aiconfigurator_core.sdk.kernelsim.analytical import gemm_latency_ms

            return PerformanceResult(
                gemm_latency_ms(m, n, k, quant_mode, database.system_spec["gpu"], database._analytical_config),
                energy=0.0,
                source="analytical",
            )

        # SILICON or HYBRID mode — use database. ``load_data`` is idempotent;
        # it populates the class cache and binds the instance attrs only when
        # the test hasn't already pre-set them.
        cls.load_data(database)
        gemm_data_wrapper = database._gemm_data

        def get_silicon():
            def _to_performance_result(result, *, source: str = "silicon"):
                """Normalize GEMM table entries into a PerformanceResult.

                Interpolated/extrapolated GEMM values are still derived from
                silicon table data; only explicit formula fallbacks are
                tagged as empirical.

                If ``result`` is already a ``PerformanceResult``, return it
                unchanged so upstream attribution (e.g. ``"empirical"`` /
                ``"mixed"`` set by an inner ``_query_silicon_or_hybrid`` /
                ``_interp_pr`` call) is preserved instead of being silently
                overwritten with the ``source`` default."""
                if isinstance(result, PerformanceResult):
                    return result
                if isinstance(result, dict):
                    return PerformanceResult(result["latency"], energy=result.get("energy", 0.0), source=source)
                return PerformanceResult(result, energy=0.0, source=source)

            gemm_data_wrapper.raise_if_not_loaded()
            if table_quant_mode not in gemm_data_wrapper:
                supported = sorted([q.name for q in gemm_data_wrapper])
                raise PerfDataNotAvailableError(
                    "GEMM perf data not available for requested quant mode. "
                    f"system='{database.system}', backend='{database.backend}', version='{database.version}', "
                    f"quant_mode='{quant_mode.name}'. "
                    f"Supported gemm modes: {supported}"
                )

            gemm_data = gemm_data_wrapper[table_quant_mode]

            # Resolve on the raw table: exact hit -> the collected (n, k) site's
            # own m-curve -> nearest-site util transfer -> util-hold beyond the
            # sweep. See sdk/perf_interp/config.py for the design.
            config = perf_interp.gemm_config(sol_fn=lambda m_v, n_v, k_v: get_sol(m_v, n_v, k_v, quant_mode)[0])
            try:
                result = perf_interp.query(config, gemm_data, m, n, k)
            except InterpolationDataNotAvailableError as exc:
                raise PerfDataNotAvailableError(
                    "GEMM perf data not available for requested shape. "
                    f"system='{database.system}', backend='{database.backend}', version='{database.version}', "
                    f"quant_mode='{quant_mode.name}', m={m}, n={n}, k={k}."
                ) from exc
            return _to_performance_result(result)

        return database._query_silicon_or_hybrid(
            get_silicon=get_silicon,
            get_empirical=lambda: get_empirical(m, n, k, quant_mode),
            database_mode=database_mode,
            error_msg=f"Failed to query gemm data for {m=}, {n=}, {k=}, {quant_mode=}",
        )

    @classmethod
    def _query_compute_scale_table(
        cls,
        database: PerfDatabase,
        m: int,
        k: int,
        quant_mode: common.GEMMQuantMode,
        database_mode: common.DatabaseMode | None = None,
    ):
        """Query compute_scale (dynamic minus static quantization) table."""

        def get_sol(m_v: int, k_v: int) -> tuple[float, float, float]:
            sol_mem = 2 * m_v * k_v / database.system_spec["gpu"]["mem_bw"] * 1000.0
            sol_time = sol_mem
            return sol_time, 0, sol_mem

        table_quant_mode = cls._normalize_gemm_quant_mode_for_table(quant_mode)

        def get_empirical(m_v: int, k_v: int) -> float:
            # compute_scale is a non-negative latency delta.  Unlike an actual
            # kernel latency, zero is meaningful and must participate in the
            # nearest-point decision (see _estimate_zero_aware_delta).
            try:
                cls.load_data(database)
                wrapper = database._compute_scale_data
                wrapper.raise_if_not_loaded()
                table = util_empirical.require_data_slice(wrapper, table_quant_mode)
            except PerfDataNotAvailableError as exc:
                raise EmpiricalNotImplementedError(
                    f"No empirical compute_scale data is available for m={m_v}, k={k_v}."
                ) from exc

            return _estimate_zero_aware_delta(
                table,
                (float(m_v), float(k_v)),
                lambda m_ref, k_ref: get_sol(m_ref, k_ref)[0],
                cls._compute_scale_delta_lookup_cache,
            )

        if database_mode is None:
            database_mode = database._default_database_mode

        if database_mode == common.DatabaseMode.SOL:
            return PerformanceResult(get_sol(m, k)[0], energy=0.0, source="sol")
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(m, k)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            return PerformanceResult(get_empirical(m, k), energy=0.0, source="empirical")
        elif database_mode == common.DatabaseMode.ANALYTICAL:
            return PerformanceResult(get_sol(m, k)[0] / 0.8, energy=0.0, source="analytical")

        cls.load_data(database)
        compute_scale_wrapper = database._compute_scale_data

        def get_silicon():
            compute_scale_wrapper.raise_if_not_loaded()
            if table_quant_mode not in compute_scale_wrapper:
                supported = sorted([q.name for q in compute_scale_wrapper])
                from aiconfigurator_core.sdk.perf_database import PerfDataNotAvailableError

                raise PerfDataNotAvailableError(
                    "Compute scale perf data not available for requested quant mode. "
                    f"system='{database.system}', backend='{database.backend}', version='{database.version}', "
                    f"quant_mode='{quant_mode.name}'. "
                    f"Supported modes: {supported}"
                )
            table = compute_scale_wrapper[table_quant_mode]
            # Clamp into the collected range FIRST (preserving the legacy
            # contract), then resolve the interior on the engine (RAW 2-axis).
            m_keys = sorted(table.keys())
            m_c = max(m_keys[0], min(int(m), m_keys[-1]))
            k_min = min(min(row) for row in table.values() if row)
            k_max = max(max(row) for row in table.values() if row)
            k_c = max(k_min, min(int(k), k_max))
            config = perf_interp.OpInterpConfig(
                axes=("m", "k"),
                resolver=perf_interp.Grid(),
                sol_fn=lambda m_v, k_v: get_sol(m_v, k_v)[0],
            )
            result = perf_interp.query(config, table, m_c, k_c)
            interpolated = database._interp_pr(
                perf_interp.get_value(result, "latency"),
                energy=perf_interp.get_value(result, "energy"),
            )
            # compute_scale is a quantization-overhead DELTA: beyond the grid it
            # is deliberately held FLAT at the clamped boundary (legacy contract).
            return interpolated

        return database._query_silicon_or_hybrid(
            get_silicon=get_silicon,
            get_empirical=lambda: get_empirical(m, k),
            database_mode=database_mode,
            error_msg=f"Failed to query compute_scale data for {m=}, {k=}, {quant_mode=}",
        )

    @classmethod
    def _query_scale_matrix_table(
        cls,
        database: PerfDatabase,
        m: int,
        k: int,
        quant_mode: common.GEMMQuantMode,
        database_mode: common.DatabaseMode | None = None,
    ):
        """Query scale_matrix (static quantization) table."""

        def get_sol(m_v: int, k_v: int) -> tuple[float, float, float]:
            sol_mem = 3 * m_v * k_v / database.system_spec["gpu"]["mem_bw"] * 1000.0
            sol_time = sol_mem
            return sol_time, 0, sol_mem

        table_quant_mode = cls._normalize_gemm_quant_mode_for_table(quant_mode)

        def get_empirical(m_v: int, k_v: int) -> float:
            # SOL / util, util read best-effort from collected scale_matrix data
            # (the (m, k) grid for this quant); raises EmpiricalNotImplementedError if none.
            sol_time = get_sol(m_v, k_v)[0]

            def _slice():
                cls.load_data(database)
                wrapper = database._scale_matrix_data
                wrapper.raise_if_not_loaded()
                return util_empirical.require_data_slice(wrapper, table_quant_mode)

            grid = util_empirical.grid_for(
                ("scale_matrix", database.system, database.backend, database.version, table_quant_mode.name),
                _slice,
                lambda c: get_sol(c[0], c[1])[0],
                depth=2,
            )
            latency, _ = util_empirical.estimate(sol_time, (float(m_v), float(k_v)), grid)
            return latency

        if database_mode is None:
            database_mode = database._default_database_mode

        if database_mode == common.DatabaseMode.SOL:
            return PerformanceResult(get_sol(m, k)[0], energy=0.0, source="sol")
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(m, k)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            return PerformanceResult(get_empirical(m, k), energy=0.0, source="empirical")
        elif database_mode == common.DatabaseMode.ANALYTICAL:
            return PerformanceResult(get_sol(m, k)[0] / 0.8, energy=0.0, source="analytical")

        cls.load_data(database)
        scale_matrix_wrapper = database._scale_matrix_data

        def get_silicon():
            scale_matrix_wrapper.raise_if_not_loaded()
            if table_quant_mode not in scale_matrix_wrapper:
                supported = sorted([q.name for q in scale_matrix_wrapper])
                from aiconfigurator_core.sdk.perf_database import PerfDataNotAvailableError

                raise PerfDataNotAvailableError(
                    "Scale matrix perf data not available for requested quant mode. "
                    f"system='{database.system}', backend='{database.backend}', version='{database.version}', "
                    f"quant_mode='{quant_mode.name}'. "
                    f"Supported modes: {supported}"
                )
            table = scale_matrix_wrapper[table_quant_mode]
            # Clamp into the collected range FIRST (preserving the legacy
            # contract), then resolve the interior on the engine (RAW 2-axis).
            m_keys = sorted(table.keys())
            m_c = max(m_keys[0], min(int(m), m_keys[-1]))
            k_min = min(min(row) for row in table.values() if row)
            k_max = max(max(row) for row in table.values() if row)
            k_c = max(k_min, min(int(k), k_max))
            config = perf_interp.OpInterpConfig(
                axes=("m", "k"),
                resolver=perf_interp.Grid(),
                sol_fn=lambda m_v, k_v: get_sol(m_v, k_v)[0],
            )
            result = perf_interp.query(config, table, m_c, k_c)
            interpolated = database._interp_pr(
                perf_interp.get_value(result, "latency"),
                energy=perf_interp.get_value(result, "energy"),
            )
            if m_c == int(m) and k_c == int(k):
                return interpolated
            # Outside the grid, freeze utilization at the clamped boundary:
            # L(q) = L(boundary) * SOL(q)/SOL(boundary) (a real memory kernel,
            # unlike the compute_scale delta above).
            boundary_sol = get_sol(m_c, k_c)[0]
            query_sol = get_sol(int(m), int(k))[0]
            ratio = query_sol / boundary_sol
            return PerformanceResult(
                latency=float(interpolated) * ratio,
                energy=interpolated.energy * ratio,
                source=getattr(interpolated, "source", "silicon"),
            )

        return database._query_silicon_or_hybrid(
            get_silicon=get_silicon,
            get_empirical=lambda: get_empirical(m, k),
            database_mode=database_mode,
            error_msg=f"Failed to query scale_matrix data for {m=}, {k=}, {quant_mode=}",
        )

    # ------------------------------------------------------------------
    # Op contract: query() + get_weights()
    # ------------------------------------------------------------------

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """
        Query GEMM latency with energy data.

        For `fp8_static` quant mode, subtracts compute_scale overhead.
        For GEMMs marked as low-precision input under `fp8_static`, also subtract scale_matrix.

        Returns:
            PerformanceResult: Behaves like float (scaled latency in ms).
                              Energy data accessible via .energy attribute.
                              Power can be derived as energy/latency.
        """
        x = kwargs.get("x")
        x //= self._scale_num_tokens
        x = -(-x // self._seq_split)  # CP: per-rank token count (ceil = busiest rank)
        overwrite_quant_mode = kwargs.get("quant_mode")
        quant_mode = self._quant_mode if overwrite_quant_mode is None else overwrite_quant_mode
        is_fp8_static = quant_mode == common.GEMMQuantMode.fp8_static
        latency_floor = 0.0

        # Query with energy
        result = database.query_gemm(x, self._n, self._k, quant_mode)
        latency = float(result)
        energy = result.energy
        source = getattr(result, "source", "silicon")

        # Static-FP8 GEMM is modeled from the dynamic FP8 base measurement
        # across backends; subtract the separately collected activation-
        # quantization pieces for BF16-input and low-precision-input cases.
        if is_fp8_static:
            compute_scale_result = database.query_compute_scale(x, self._k, quant_mode)
            latency -= float(compute_scale_result)
            energy -= compute_scale_result.energy
            sub_src = getattr(compute_scale_result, "source", "silicon")
            if sub_src != source:
                source = "mixed"
            if self._low_precision_input:
                scale_matrix_result = database.query_scale_matrix(x, self._k, quant_mode)
                latency -= float(scale_matrix_result)
                energy -= scale_matrix_result.energy
                sub_src = getattr(scale_matrix_result, "source", "silicon")
                if sub_src != source:
                    source = "mixed"
            # fp8_static is modeled from dynamic FP8 plus overhead tables, so
            # expose source="estimated" instead of measured silicon.
            source = "estimated"

            # The subtraction leaves a path that still contains the GEMM
            # (GEMM-only for low-precision input, static-quant + GEMM otherwise).
            # Independently interpolated component tables can cross, but that
            # path cannot be faster than the GEMM's own roofline bound. Keep the
            # physical SOL floor instead of turning a negative residual into 0.
            latency_floor = float(
                database.query_gemm(
                    x,
                    self._n,
                    self._k,
                    quant_mode,
                    database_mode=common.DatabaseMode.SOL,
                )
            )

        # Latency has a physical roofline floor. Energy has no analogous SOL
        # model here, so retain the existing conservative non-negative clamp
        # rather than inventing energy when the latency floor fires.
        latency_clamped = max(latency_floor, latency)
        energy_clamped = max(0.0, energy)
        if latency_clamped != latency or energy_clamped != energy:
            logger.warning(
                "GEMM.query applied latency SOL floor / non-negative energy clamp. "
                "op=%s m=%s n=%s k=%s quant_mode=%s post_sub(lat=%.6f, eng=%.6f) floor=%.6f",
                self._name,
                x,
                self._n,
                self._k,
                quant_mode.name,
                latency,
                energy,
                latency_floor,
            )

        latency = latency_clamped
        energy = energy_clamped

        return PerformanceResult(
            latency=latency * self._scale_factor,
            energy=energy * self._scale_factor,
            source=source,
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class ContextKVBProjGEMM(GEMM):
    """SGLang DeepSeek prefill KV projection over fresh and cached tokens."""

    @staticmethod
    def _prefix_tokens(kwargs: dict) -> int:
        prefix = kwargs.get("prefix") or 0
        if isinstance(prefix, (list, tuple)):
            return int(sum(prefix))
        return int(kwargs.get("batch_size", 1)) * int(prefix)

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        corrected = dict(kwargs)
        corrected["x"] = int(corrected.get("x") or 0) + self._prefix_tokens(corrected)
        return super().query(database, **corrected)


# ─────────────────────────────────────────────────────────
# CSV loaders (moved here from perf_database.py so each op family owns its data + parser)
# ─────────────────────────────────────────────────────────


def load_gemm_data(gemm_file):
    """
    Load the gemm data with power support (backward compatible).

    Returns:
        dict: Nested dict structure where leaf values are dicts with
              'latency', 'power', and 'energy' keys.
              For old database formats without power, defaults to power=0.0 and energy=0.0.
    """
    rows = _read_filtered_rows(gemm_file)
    if rows is None:
        logger.debug(f"GEMM data file {gemm_file} not found.")
        return None
    gemm_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug("Legacy database format detected (gemm) - power will default to 0.0")

    for row in rows:
        quant_mode, m, n, k, latency = (
            row["gemm_dtype"],
            row["m"],
            row["n"],
            row["k"],
            row["latency"],
        )
        m = int(m)
        n = int(n)
        k = int(k)
        latency = float(latency)

        # NEW: Read power with backward compatibility
        power = float(row.get("power", 0.0))
        # Note: power_limit is available in row.get("power_limit") if needed for validation

        # NEW: Calculate energy from power and latency
        energy = power * latency  # watt-milliseconds (W·ms)

        quant_mode = common.GEMMQuantMode[quant_mode]

        try:
            # Check for conflict
            gemm_data[quant_mode][m][n][k]
            logger.debug(f"value conflict in gemm data: {quant_mode} {m} {n} {k}")
        except KeyError:
            # Store all three values
            gemm_data[quant_mode][m][n][k] = {
                "latency": latency,
                "power": power,  # Keep for reference
                "energy": energy,  # NEW: precomputed energy
            }

    return gemm_data


def load_compute_scale_data(compute_scale_file):
    """
    Load the compute scale data with power support (backward compatible).

    Returns:
        dict: Nested dict structure {quant_mode: {m: {k: {latency, power, energy}}}}
              For old database formats without power, defaults to power=0.0 and energy=0.0.
    """
    rows = _read_filtered_rows(compute_scale_file)
    if rows is None:
        logger.debug(f"Compute scale data file {compute_scale_file} not found.")
        return None
    compute_scale_data = defaultdict(lambda: defaultdict(lambda: defaultdict()))

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug("Legacy database format detected (compute_scale) - power will default to 0.0")

    for row in rows:
        quant_mode, m, k, latency = (
            row["quant_dtype"],
            row["m"],
            row["k"],
            row["latency"],
        )
        m = int(m)
        k = int(k)
        latency = float(latency)

        # Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # Calculate energy from power and latency
        energy = power * latency  # watt-milliseconds (W·ms)

        quant_mode = common.GEMMQuantMode[quant_mode]

        try:
            # Check for conflict
            compute_scale_data[quant_mode][m][k]
            logger.debug(f"value conflict in compute_scale data: {quant_mode} {m} {k}")
        except KeyError:
            # Store all three values
            compute_scale_data[quant_mode][m][k] = {
                "latency": latency,
                "power": power,
                "energy": energy,
            }

    return compute_scale_data


def load_scale_matrix_data(scale_matrix_file):
    """
    Load the scale matrix data with power support (backward compatible).

    Returns:
        dict: Nested dict structure {quant_mode: {m: {k: {latency, power, energy}}}}
              For old database formats without power, defaults to power=0.0 and energy=0.0.
    """
    rows = _read_filtered_rows(scale_matrix_file)
    if rows is None:
        logger.debug(f"Scale matrix data file {scale_matrix_file} not found.")
        return None
    scale_matrix_data = defaultdict(lambda: defaultdict(lambda: defaultdict()))

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug("Legacy database format detected (scale_matrix) - power will default to 0.0")

    for row in rows:
        quant_mode, m, k, latency = (
            row["quant_dtype"],
            row["m"],
            row["k"],
            row["latency"],
        )
        m = int(m)
        k = int(k)
        latency = float(latency)

        # Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # Calculate energy from power and latency
        energy = power * latency  # watt-milliseconds (W·ms)

        quant_mode = common.GEMMQuantMode[quant_mode]

        try:
            # Check for conflict
            scale_matrix_data[quant_mode][m][k]
            logger.debug(f"value conflict in scale_matrix data: {quant_mode} {m} {k}")
        except KeyError:
            # Store all three values
            scale_matrix_data[quant_mode][m][k] = {
                "latency": latency,
                "power": power,
                "energy": energy,
            }

    return scale_matrix_data
