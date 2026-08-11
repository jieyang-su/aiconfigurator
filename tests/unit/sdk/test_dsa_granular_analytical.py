from __future__ import annotations

import warnings

import pytest

from aiconfigurator_core.sdk import common, config
from aiconfigurator_core.sdk.kernelsim.analytical import dsa_sparse_attention_latency_ms
from aiconfigurator_core.sdk.kernelsim.dsa import DsaIndexModelWarning
from aiconfigurator_core.sdk.models import get_model
from aiconfigurator_core.sdk.operations import (
    GEMM,
    DSAIndexScore,
    DSASparseAttention,
    DSATopKSelect,
    ElementWise,
    FallbackOp,
)
from aiconfigurator_core.sdk.perf_database import get_database_view


def _model(model_id: str):
    model_config = config.ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1)
    return get_model(model_id, model_config, backend_name="sglang")


def _attention(model, phase: str) -> FallbackOp:
    operations = model.context_ops if phase == "context" else model.generation_ops
    return next(op for op in operations if op._name == f"{phase}_attention")


@pytest.mark.parametrize("model_id", ["deepseek-ai/DeepSeek-V3.2", "zai-org/GLM-5.2"])
def test_dsa_models_use_module_primary_with_granular_fallback(model_id):
    model = _model(model_id)
    for phase in ("context", "generation"):
        attention = _attention(model, phase)
        assert isinstance(attention, FallbackOp)
        assert attention._silicon_primary_only
        assert sum(isinstance(op, GEMM) for op in attention._fallback) >= 7
        assert any(isinstance(op, ElementWise) for op in attention._fallback)
        assert sum(isinstance(op, DSAIndexScore) for op in attention._fallback) == 1
        assert sum(isinstance(op, DSATopKSelect) for op in attention._fallback) == 1
        assert sum(isinstance(op, DSASparseAttention) for op in attention._fallback) == 1


def test_glm_shared_index_fraction_and_architecture_dimensions():
    deepseek = _attention(_model("deepseek-ai/DeepSeek-V3.2"), "context")
    glm = _attention(_model("zai-org/GLM-5.2"), "context")

    ds_score = next(op for op in deepseek._fallback if isinstance(op, DSAIndexScore))
    glm_score = next(op for op in glm._fallback if isinstance(op, DSAIndexScore))
    ds_sparse = next(op for op in deepseek._fallback if isinstance(op, DSASparseAttention))
    glm_sparse = next(op for op in glm._fallback if isinstance(op, DSASparseAttention))

    assert ds_score._scale_factor == pytest.approx(deepseek._primary._scale_factor)
    assert glm_score._scale_factor / glm._primary._scale_factor == pytest.approx(21 / 78)
    assert (ds_score._index_heads, ds_score._index_head_dim) == (64, 128)
    assert (glm_score._index_heads, glm_score._index_head_dim) == (32, 128)
    assert (ds_sparse._qk_nope_dim, ds_sparse._output_value_dim) == (128, 128)
    assert (glm_sparse._qk_nope_dim, glm_sparse._output_value_dim) == (192, 256)


def test_analytical_fallback_never_queries_module_primary(monkeypatch):
    attention = _attention(_model("deepseek-ai/DeepSeek-V3.2"), "generation")
    monkeypatch.setattr(attention._primary, "query", lambda *_args, **_kwargs: pytest.fail("module queried"))
    database = get_database_view(
        "h100_sxm",
        "sglang",
        "estimate",
        allow_missing_data=True,
        database_mode="ANALYTICAL",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DsaIndexModelWarning)
        result = attention.query(database, x=1, batch_size=1, s=4096, prefix=0)
    assert float(result) > 0
    # Small recipe ops may retain their existing theoretical source tag; the
    # aggregate is therefore mixed, but must never become a silicon lookup.
    assert result.source != "silicon"


def test_context_select_all_skips_index_score_and_topk():
    database = get_database_view(
        "h100_sxm",
        "sglang",
        "estimate",
        allow_missing_data=True,
        database_mode=common.DatabaseMode.ANALYTICAL,
    )
    model = _model("deepseek-ai/DeepSeek-V3.2")
    fallback = _attention(model, "context")._fallback
    score = next(op for op in fallback if isinstance(op, DSAIndexScore))
    topk = next(op for op in fallback if isinstance(op, DSATopKSelect))
    assert float(score.query(database, batch_size=1, s=1024, prefix=1024)) == 0
    assert float(topk.query(database, batch_size=1, s=1024, prefix=1024)) == 0


def test_sparse_attention_head_quantum_is_explicit_and_changes_only_kernel_work():
    from aiconfigurator_core.sdk.kernelsim.analytical import AnalyticalConfig

    gpu = get_database_view(
        "h100_sxm", "sglang", "estimate", allow_missing_data=True, database_mode="ANALYTICAL"
    ).system_spec["gpu"]
    kwargs = dict(
        gpu=gpu,
        batch=1,
        query_length=8192,
        selected_pairs=8192 * 2048,
        local_heads=16,
        qk_latent_dim=576,
        value_latent_dim=512,
        qk_nope_dim=128,
        output_value_dim=128,
    )
    default = dsa_sparse_attention_latency_ms(**kwargs, config=AnalyticalConfig())
    hopper = dsa_sparse_attention_latency_ms(
        **kwargs, config=AnalyticalConfig(sparse_attention_head_quantum=64)
    )
    blackwell = dsa_sparse_attention_latency_ms(
        **kwargs, config=AnalyticalConfig(sparse_attention_head_quantum=128)
    )
    assert default < hopper < blackwell
