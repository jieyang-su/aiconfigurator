from __future__ import annotations

import warnings
from collections import Counter

import pytest

from aiconfigurator_core.sdk import common, config
from aiconfigurator_core.sdk.backends.sglang_backend import SGLANGBackend
from aiconfigurator_core.sdk.config import RuntimeConfig
from aiconfigurator_core.sdk.kernelsim.dsa import DsaIndexModelWarning
from aiconfigurator_core.sdk.models import get_model
from aiconfigurator_core.sdk.operations import (
    GEMM,
    DeepSeekV4KVAllGather,
    DeepSeekV4SparseAttention,
    DSAIndexScore,
    DSATopKSelect,
    FallbackOp,
)
from aiconfigurator_core.sdk.perf_database import get_database_view


def _model(model_id="sgl-project/DeepSeek-V4-Flash-FP8"):
    return get_model(
        model_id,
        config.ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1),
        backend_name="sglang",
    )


def _attention(model, phase):
    source = model.context_ops if phase == "context" else model.generation_ops
    return [op for op in source if op._name == f"{phase}_attention"]


def test_v4_ratio_specific_granular_graphs():
    model = _model()
    for phase in ("context", "generation"):
        wrappers = _attention(model, phase)
        assert all(
            isinstance(op, FallbackOp) and common.DatabaseMode.ANALYTICAL in op._primary_excluded_modes
            for op in wrappers
        )
        assert Counter(op._compress_ratio for op in wrappers) == {0: 1, 4: 1, 128: 1}

        by_ratio = {op._compress_ratio: op._fallback for op in wrappers}
        assert not any(isinstance(op, (DSAIndexScore, DSATopKSelect)) for op in by_ratio[0])
        assert not any(isinstance(op, (DSAIndexScore, DSATopKSelect)) for op in by_ratio[128])
        assert sum(isinstance(op, DSAIndexScore) for op in by_ratio[4]) == 1
        assert sum(isinstance(op, DSATopKSelect) for op in by_ratio[4]) == 1
        assert all(sum(isinstance(op, DeepSeekV4SparseAttention) for op in ops) == 1 for ops in by_ratio.values())
        assert (
            len([op for op in by_ratio[4] if isinstance(op, GEMM)])
            > len([op for op in by_ratio[128] if isinstance(op, GEMM)])
            > len([op for op in by_ratio[0] if isinstance(op, GEMM)])
        )


def test_v4_csa_index_uses_compressed_context():
    score = next(
        op
        for op in _attention(_model(), "context")
        if op._compress_ratio == 4
        for op in op._fallback
        if isinstance(op, DSAIndexScore)
    )
    assert score._context_stride == 4
    assert score._index_topk == 512

    database = get_database_view("h100_sxm", "sglang", "estimate", allow_missing_data=True, database_mode="ANALYTICAL")
    # full context 2048 -> c4 cache 512, so the production select-all path skips scoring.
    assert float(score.query(database, batch_size=1, s=1024, prefix=1024)) == 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DsaIndexModelWarning)
        assert float(score.query(database, batch_size=1, s=4096, prefix=4096)) > 0.0


def test_v4_csa_topk_allows_more_query_rows_than_compressed_candidates():
    topk = next(
        child
        for wrapper in _attention(_model(), "context")
        if wrapper._compress_ratio == 4
        for child in wrapper._fallback
        if isinstance(child, DSATopKSelect)
    )
    assert topk._kernel_recipe == "dsv4"
    assert topk._context_stride == 4
    database = get_database_view("h100_sxm", "sglang", "estimate", allow_missing_data=True, database_mode="ANALYTICAL")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DsaIndexModelWarning)
        assert float(topk.query(database, batch_size=2, s=4096, prefix=0)) > 0.0


def test_v4_context_cp_granular_restores_ratio_specific_all_gathers():
    model = get_model(
        "sgl-project/DeepSeek-V4-Flash-FP8",
        config.ModelConfig(tp_size=1, cp_size=2, moe_tp_size=1, moe_ep_size=2),
        backend_name="sglang",
    )
    by_ratio = {op._compress_ratio: op._fallback for op in _attention(model, "context")}
    assert [op._kind for op in by_ratio[0] if isinstance(op, DeepSeekV4KVAllGather)] == ["window"]
    assert [op._kind for op in by_ratio[4] if isinstance(op, DeepSeekV4KVAllGather)] == ["index", "compressed"]
    assert [op._kind for op in by_ratio[128] if isinstance(op, DeepSeekV4KVAllGather)] == ["window", "compressed"]


def test_v4_analytical_never_queries_attention_module(monkeypatch):
    wrapper = next(op for op in _attention(_model(), "generation") if op._compress_ratio == 4)
    monkeypatch.setattr(wrapper._primary, "query", lambda *_a, **_k: pytest.fail("module queried"))
    database = get_database_view(
        "h100_sxm", "sglang", "estimate", allow_missing_data=True, database_mode=common.DatabaseMode.ANALYTICAL
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DsaIndexModelWarning)
        result = wrapper.query(database, x=1, batch_size=1, s=8192, prefix=0, beam_width=1)
    assert float(result) > 0.0
    assert result.source != "silicon"


def test_v4_sparse_pair_rules_cover_swa_csa_and_hca():
    op = DeepSeekV4SparseAttention
    assert op._causal_limited_pairs(1, 4, 0, 2) == 7
    assert op._compressed_causal_pairs(1, 8, 0, 4, 512) == 6
    assert op._compressed_causal_pairs(1, 8, 2048, 4, 512) == 8 * 512


def test_fa_shape_allows_sparse_equivalent_q_greater_than_k():
    from aiconfigurator_core.sdk.kernelsim.fa import AttentionShape

    shape = AttentionShape(1, 128, 64, 64, 1, 512, "bf16", causal=False)
    assert shape.query_length == 128
    with pytest.raises(ValueError, match="kv_length_total"):
        AttentionShape(1, 128, 64, 64, 1, 512, "bf16", causal=True)


def test_fa_mixed_kv_storage_changes_only_cache_traffic():
    from aiconfigurator_core.sdk.kernelsim.analytical import fa_hardware
    from aiconfigurator_core.sdk.kernelsim.fa import AttentionShape, ModelOptions, estimate_attention
    from aiconfigurator_core.sdk.perf_database import get_database_view

    database = get_database_view(
        "h100_sxm", "sglang", "estimate", allow_missing_data=True, database_mode="ANALYTICAL"
    )
    hardware = fa_hardware(database.system, database.system_spec["gpu"])
    options = ModelOptions(mode="profiled", estimate_level="standard")
    common_shape = dict(
        batch_size=8,
        query_length=1,
        kv_length_total=8192,
        query_heads=64,
        kv_heads=1,
        head_dim=512,
        dtype="bf16",
        value_head_dim=512,
        kv_storage_dim=512,
    )
    bf16_cache = estimate_attention(hardware, AttentionShape(**common_shape), options)
    mixed_cache = estimate_attention(
        hardware,
        AttentionShape(**common_shape, kv_cache_bytes_per_token=584),
        options,
    )

    assert bf16_cache.resources_us["matrix_peak_flops"] == mixed_cache.resources_us["matrix_peak_flops"]
    assert bf16_cache.tiles["automatic_block"] == mixed_cache.tiles["automatic_block"]
    assert bf16_cache.work["kv_cache_bytes_per_token"] == 1024
    assert mixed_cache.work["kv_cache_bytes_per_token"] == 584
    assert mixed_cache.work["mainloop_hbm_bytes"] < bf16_cache.work["mainloop_hbm_bytes"]


def test_v4_sparse_attention_uses_bf16_compute_with_packed_fp8_kv(monkeypatch):
    captured = {}

    def fake_attention_latency_ms(**kwargs):
        captured.update(kwargs)
        return 1.0

    monkeypatch.setattr(
        "aiconfigurator_core.sdk.kernelsim.analytical.attention_latency_ms",
        fake_attention_latency_ms,
    )
    database = get_database_view(
        "h100_sxm", "sglang", "estimate", allow_missing_data=True, database_mode="ANALYTICAL"
    )
    op = DeepSeekV4SparseAttention(
        "dsv4_sparse",
        1.0,
        layout="paged",
        local_heads=64,
        head_dim=512,
        window_size=4096,
        compress_ratio=4,
        index_topk=512,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.fp8,
    )

    result = op.query(database, batch_size=4, s=8192)

    assert float(result) == pytest.approx(1.0)
    assert captured["dtype"] == "bf16"
    assert captured["value_head_dim"] == 512
    assert captured["kv_storage_dim"] == 512
    assert captured["kv_cache_bytes_per_token"] == 584
    assert captured["include_kv_cache_update"] is False


def test_v4_mhc_analytical_is_no_table():
    from aiconfigurator_core.sdk.operations import DeepSeekV4MHCModule

    database = get_database_view(
        "h100_sxm",
        "sglang",
        "estimate",
        allow_missing_data=True,
        database_mode=common.DatabaseMode.ANALYTICAL,
    )
    op = DeepSeekV4MHCModule("mhc", 1, "pre", 4096, 4, 20, common.GEMMQuantMode.bfloat16)
    result = op.query(database, x=32)
    assert float(result) > 0
    assert result.source == "analytical"


def test_v4_fp8_static_analytical_runs_end_to_end_without_module_tables():
    model = get_model(
        "sgl-project/DeepSeek-V4-Flash-FP8",
        config.ModelConfig(
            tp_size=1,
            moe_tp_size=1,
            moe_ep_size=1,
            nextn=1,
            overwrite_num_layers=2,
        ),
        backend_name="sglang",
    )
    database = get_database_view(
        "h100_sxm",
        "sglang",
        "estimate",
        allow_missing_data=True,
        database_mode=common.DatabaseMode.ANALYTICAL,
    )
    runtime = RuntimeConfig(batch_size=1, beam_width=1, isl=128, osl=4, prefix=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DsaIndexModelWarning)
        summary = SGLANGBackend().run_static(model, database, runtime, mode="static", stride=1)
    assert sum(summary.get_context_latency_dict().values()) > 0
    assert sum(summary.get_generation_latency_dict().values()) > 0
    assert "silicon" not in set(summary.get_context_source_dict().values())
    assert "silicon" not in set(summary.get_generation_source_dict().values())
