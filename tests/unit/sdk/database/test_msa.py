# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax Sparse Attention (MSA) op: SOL and the cross-op (XOP) transfer from DSA.

MSA has no own silicon data, so its empirical path is a cross-op transfer gated by the
XOP transfer kind. These tests cover the SOL path and the XOP gate (the headline new op
otherwise had no coverage)."""

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.errors import EmpiricalNotImplementedError

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


def test_msa_sol_scales_with_workload(comprehensive_perf_db):
    """SOL mode computes the three-group MSA SOL (gemm + fp8 indexer + sparse attn). Assert it
    RESPONDS to the workload rather than returning a constant: more new tokens (s) add work, and
    a longer cached prefix adds indexer/attention work (full_s > index_topk)."""
    comprehensive_perf_db.set_default_database_mode(common.DatabaseMode.SOL)
    try:
        op = _ctx_msa()
        small = float(op.query(comprehensive_perf_db, batch_size=8, s=512, prefix=0))
        large = float(op.query(comprehensive_perf_db, batch_size=8, s=2048, prefix=0))
        with_prefix = float(op.query(comprehensive_perf_db, batch_size=8, s=2048, prefix=2048))
        assert 0 < small < large  # scales with new-token count
        assert with_prefix > large  # cached prefix adds indexer work beyond index_topk
    finally:
        comprehensive_perf_db.set_default_database_mode(common.DatabaseMode.SILICON)


def test_msa_xop_gating(comprehensive_perf_db, monkeypatch):
    """The DSA-to-MSA utilization transfer is gated and tagged as XOP."""
    from aiconfigurator.sdk.operations import util_empirical

    util_queries = []

    def dsa_util(_database, **kwargs):
        util_queries.append(kwargs)
        return 0.5

    monkeypatch.setattr("aiconfigurator.sdk.operations.msa._dsa_context_util", dsa_util)
    comprehensive_perf_db.set_default_database_mode(common.DatabaseMode.HYBRID)
    kw = dict(batch_size=8, s=2048, prefix=0)
    try:
        comprehensive_perf_db.set_transfer_policy(["xshape", "xquant"])  # no XOP
        with pytest.raises(EmpiricalNotImplementedError) as exc:
            _ctx_msa().query(comprehensive_perf_db, **kw)
        assert "xop" in str(exc.value).lower()  # gated at the policy, not a data miss
        assert util_queries == []

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
    from aiconfigurator_core.sdk.operations.msa import MsaAnalyticalApproximationWarning

    from aiconfigurator.sdk.operations.msa import ContextMSAModule

    analytical_op = ContextMSAModule(
        "msa_analytical", 1.0, 8, 1, 4096, 128, 128, 4, 128, 2048, 128,
        common.KVCacheQuantMode.bfloat16,
        common.FMHAQuantMode.bfloat16,
        common.GEMMQuantMode.bfloat16,
    )
    comprehensive_perf_db = mutable_comprehensive_perf_db
    comprehensive_perf_db.system_spec["gpu"].update({
        "sm_count": 132,
        "clock_hz": 1.8e9,
        "shared_memory_per_sm_bytes": 228 * 1024,
        "l2_capacity_bytes": 50 * 1024 * 1024,
        "l2_bandwidth_bytes_s": 1e15,
        "vector_peak_flops": 1e13,
    })
    comprehensive_perf_db.set_default_database_mode(common.DatabaseMode.ANALYTICAL)
    try:
        with pytest.warns(MsaAnalyticalApproximationWarning):
            result = analytical_op.query(comprehensive_perf_db, batch_size=2, s=4096, prefix=0)
        assert float(result) > 0
        assert result.source == "analytical"
    finally:
        comprehensive_perf_db.set_default_database_mode(common.DatabaseMode.SILICON)
