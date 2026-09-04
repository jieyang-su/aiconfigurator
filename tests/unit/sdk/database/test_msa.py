# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax Sparse Attention (MSA) op: per-op SOL coverage through the query shim.

The cross-op (XOP) DSA-to-MSA utilization transfer this file also used to pin
(policy gating + xop provenance tagging via the ``_dsa_context_util`` seam)
retired to the compiled engine with #1357 PR-5; transfer-ladder behaviour is
anchored by the frozen parity goldens and
the frozen parity goldens."""

import pytest

from aiconfigurator.sdk import common

pytestmark = pytest.mark.unit


def _ctx_msa():
    from aiconfigurator.sdk.operations.msa import ContextMSAModule

    # M3-like per-GPU shape: 8 q / 1 kv heads, head_dim 128, v 128, top-16 blocks * 128.
    return ContextMSAModule(
        "msa",
        1.0,
        num_heads=8,
        num_kv_heads=1,
        hidden_size=4096,
        head_dim=128,
        v_head_dim=128,
        index_n_heads=4,
        index_head_dim=128,
        index_topk=2048,
        block_size=128,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.fp8,
        gemm_quant_mode=common.GEMMQuantMode.fp8_block,
    )


def test_msa_sol_scales_with_workload():
    """SOL mode computes the three-group MSA SOL (gemm + fp8 indexer + sparse attn). Assert it
    RESPONDS to the workload rather than returning a constant: more new tokens (s) add work, and
    a longer cached prefix adds indexer/attention work (full_s > index_topk). Runs on a real
    shipped database: ``op._engine_query`` is the permanent internal single-op plumbing routed
    through the compiled engine's probe, which loads its tables from disk (the synthetic
    fixture is invisible to it)."""
    from aiconfigurator.sdk.perf_database import get_database_view

    db = get_database_view("b200_sxm", "sglang", "0.5.14", database_mode="SOL")
    assert db is not None, "b200_sxm/sglang/0.5.14 data missing"
    op = _ctx_msa()
    small = float(op._engine_query(db, batch_size=8, s=512, prefix=0))
    large = float(op._engine_query(db, batch_size=8, s=2048, prefix=0))
    with_prefix = float(op._engine_query(db, batch_size=8, s=2048, prefix=2048))
    assert 0 < small < large  # scales with new-token count
    assert with_prefix > large  # cached prefix adds indexer work beyond index_topk


def test_rtx_trtllm_rc23_loads_and_m3_is_explicitly_rejected():
    """Dynamo 1.3 pins trtllm 1.3.0rc23; rtx ships rc20-reuse markers so the
    exact-version gate passes (review 4969690316 Spec-3) and every reused
    family serves, while M3 MSA — no rtx table, no DSA xop donor on trtllm —
    fails with a typed empirical error (an explicitly rejected cell), never
    a silent fallback or a version-gate exit."""
    from aiconfigurator.sdk.perf_database import get_database_view
    from aiconfigurator_core.sdk.errors import EmpiricalNotImplementedError

    db = get_database_view("rtx_pro_6000_server", "trtllm", "1.3.0rc23", database_mode="HYBRID")
    assert db is not None, "rc23 reuse markers must make the version root loadable"
    op = _ctx_msa()
    # The EXACT typed contract (review 4972622548 item 3): the miss crosses
    # the engine FFI as EmpiricalNotImplementedError — never a version-gate
    # exit, never a silent fallback, never an untyped error.
    with pytest.raises(EmpiricalNotImplementedError, match=r"(?i)no DSA util|empirical"):
        op._engine_query(db, batch_size=2, s=512, prefix=0)


@pytest.mark.parametrize(
    ("system", "backend", "version"),
    [
        # Withdrawn tables (SGLang v0.5.16 has no CC-8.9 branch) AND no DSA
        # xop donor on this cell — nothing to transfer from.
        ("l40s", "sglang", "0.5.16"),
        # fp8_block tier failed classified on SM120 (DeepGEMM layout.hpp:59)
        # AND no DSA xop donor — the NVFP4 checkpoint's lane is rejected.
        ("rtx_pro_6000_server", "vllm", "0.24.0"),
    ],
)
def test_rejected_msa_cells_raise_typed_errors(system, backend, version):
    """The support matrix's R cells (review 4980441676): a cell with no own
    table and no DSA donor must fail TYPED in both modes — SILICON with
    PerfDataNotAvailableError, HYBRID with EmpiricalNotImplementedError —
    for context and generation alike; never a silent fallback or a value."""
    from aiconfigurator.sdk.operations.msa import ContextMSAModule, GenerationMSAModule
    from aiconfigurator.sdk.perf_database import get_database_view
    from aiconfigurator_core.sdk.errors import (
        EmpiricalNotImplementedError,
        PerfDataNotAvailableError,
    )

        comprehensive_perf_db.set_transfer_policy(None)  # XOP allowed
        with util_empirical.capture_provenance() as tags:
            assert float(_ctx_msa().query(comprehensive_perf_db, **kw)) > 0
        assert len(util_queries) == 1
        assert util_empirical.worst_provenance(tags) == "xop"
    finally:
        comprehensive_perf_db.set_transfer_policy(None)
        comprehensive_perf_db.set_default_database_mode(common.DatabaseMode.SILICON)


def test_msa_analytical_uses_granular_no_table_recipe(mutable_comprehensive_perf_db):
    """ANALYTICAL must not use the historical DSA XOP/module path."""
    from aiconfigurator.sdk.operations.msa import ContextMSAModule
    from aiconfigurator_core.sdk.operations.msa import MsaAnalyticalApproximationWarning

    analytical_op = ContextMSAModule(
        "msa_analytical",
        1.0,
        8,
        1,
        4096,
        128,
        128,
        4,
        128,
        2048,
        128,
        common.KVCacheQuantMode.bfloat16,
        common.FMHAQuantMode.bfloat16,
        common.GEMMQuantMode.bfloat16,
    )
    comprehensive_perf_db = mutable_comprehensive_perf_db
    comprehensive_perf_db.system_spec["gpu"].update(
        {
            "sm_count": 132,
            "clock_hz": 1.8e9,
            "shared_memory_per_sm_bytes": 228 * 1024,
            "l2_capacity_bytes": 50 * 1024 * 1024,
            "l2_bandwidth_bytes_s": 1e15,
            "vector_peak_flops": 1e13,
        }
    )
    comprehensive_perf_db.set_default_database_mode(common.DatabaseMode.ANALYTICAL)
    try:
        with pytest.warns(MsaAnalyticalApproximationWarning):
            result = analytical_op.query(comprehensive_perf_db, batch_size=2, s=4096, prefix=0)
        assert float(result) > 0
        assert result.source == "analytical"
    finally:
        comprehensive_perf_db.set_default_database_mode(common.DatabaseMode.SILICON)
