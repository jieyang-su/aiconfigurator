import logging

import pytest

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk import config as sdk_config
from aiconfigurator_core.sdk.config import RuntimeConfig
from aiconfigurator_core.sdk.engine import EngineHandle, build_ops_json
from aiconfigurator_core.sdk.kernelsim.analytical import AnalyticalConfig
from aiconfigurator_core.sdk.models import get_model
from aiconfigurator_core.sdk.performance_result import PerformanceResult
from aiconfigurator_core.sdk.perf_database import get_database_view
from aiconfigurator_core.sdk.rust_engine_step import should_use_rust_engine_step


@pytest.fixture
def analytical_db():
    database = get_database_view(
        "h100_sxm",
        "sglang",
        "current",
        allow_missing_data=True,
        database_mode="ANALYTICAL",
    )
    assert database is not None
    return database


def test_config_validation_and_normalization():
    assert AnalyticalConfig().level == "standard"
    assert AnalyticalConfig().sparse_attention_head_quantum is None
    assert AnalyticalConfig(fp8_gemm_recipe="deepgemm_hopper").fp8_gemm_recipe == "deepgemm-hopper"
    assert AnalyticalConfig(sparse_attention_head_quantum=64).sparse_attention_executed_heads(16) == 64
    assert AnalyticalConfig(sparse_attention_head_quantum=128).sparse_attention_executed_heads(64) == 128
    with pytest.raises(ValueError, match="standard, low, or high"):
        AnalyticalConfig(level="precise")
    with pytest.raises(ValueError, match="communication mode"):
        AnalyticalConfig(communication_mode="guess")
    with pytest.raises(ValueError, match="head quantum"):
        AnalyticalConfig(sparse_attention_head_quantum=32)
    with pytest.raises(ValueError, match="incompatible"):
        AnalyticalConfig(sparse_attention_head_quantum=128).sparse_attention_executed_heads(96)


def test_views_are_isolated_by_analytical_config():
    standard = get_database_view(
        "h100_sxm",
        "sglang",
        "current",
        allow_missing_data=True,
        database_mode="ANALYTICAL",
    )
    high = get_database_view(
        "h100_sxm",
        "sglang",
        "current",
        allow_missing_data=True,
        database_mode="ANALYTICAL",
        analytical_config={"level": "high"},
    )
    assert standard is not high
    assert standard._analytical_config.level == "standard"
    assert high._analytical_config.level == "high"


def _eval_op(database, op, *, is_context, batch_size, s, prefix=0, x=None):
    """Evaluate one Rust-backed operation through the public op-list FFI."""
    name, latency, energy, source = EngineHandle.for_database(
        database, systems_path=database.systems_root
    ).evaluate_ops_json(
        build_ops_json([op]),
        is_context=is_context,
        batch_size=batch_size,
        s=s,
        prefix=prefix,
        x=x,
    )[0]
    return PerformanceResult(latency, energy=energy, source=source)


def test_gemm_attention_and_mla_boundaries(analytical_db):
    from aiconfigurator_core.sdk.operations.attention import ContextAttention
    from aiconfigurator_core.sdk.operations.gemm import GEMM
    from aiconfigurator_core.sdk.operations.mla import ContextMLA, GenerationMLA

    gemm = _eval_op(
        analytical_db,
        GEMM("gemm_query", 1.0, 4096, 4096, common.GEMMQuantMode.fp8),
        is_context=True,
        batch_size=1,
        s=128,
    )
    attention = _eval_op(
        analytical_db,
        ContextAttention(
            "context_attention_query",
            1.0,
            32,
            8,
            common.KVCacheQuantMode.bfloat16,
            common.FMHAQuantMode.bfloat16,
        ),
        is_context=True,
        batch_size=1,
        s=1024,
    )
    assert gemm.source == attention.source == "analytical"
    assert float(gemm) > 0 and float(attention) > 0

    bf16_kv = _eval_op(
        analytical_db,
        GenerationMLA("generation_mla_query", 1.0, 16, common.KVCacheQuantMode.bfloat16),
        is_context=False,
        batch_size=8,
        s=4096,
    )
    fp8_kv = _eval_op(
        analytical_db,
        GenerationMLA("generation_mla_query", 1.0, 16, common.KVCacheQuantMode.fp8),
        is_context=False,
        batch_size=8,
        s=4096,
    )
    assert fp8_kv.source == "analytical"
    assert float(fp8_kv) == pytest.approx(float(bf16_kv))

    from aiconfigurator_core.sdk.operations.mla import ContextMLA

    with pytest.raises(ValueError, match="BF16 compute only"):
        _eval_op(
            analytical_db,
            ContextMLA(
                "context_mla_query",
                1.0,
                16,
                common.KVCacheQuantMode.fp8,
                common.FMHAQuantMode.fp8,
            ),
            is_context=True,
            batch_size=1,
            s=1024,
        )


def test_attention_separates_compute_dtype_from_kv_storage(analytical_db):
    from aiconfigurator_core.sdk.operations.attention import GenerationAttention

    kwargs = dict(b=8, s=8192, n=32, n_kv=8)
    bf16_kv_bf16_math = _eval_op(
        analytical_db,
        GenerationAttention(
            "generation_attention_query",
            1.0,
            kwargs["n"],
            kwargs["n_kv"],
            common.KVCacheQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        ),
        is_context=False,
        batch_size=kwargs["b"],
        s=kwargs["s"],
    )
    fp8_kv_bf16_math = _eval_op(
        analytical_db,
        GenerationAttention(
            "generation_attention_query",
            1.0,
            kwargs["n"],
            kwargs["n_kv"],
            common.KVCacheQuantMode.fp8,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        ),
        is_context=False,
        batch_size=kwargs["b"],
        s=kwargs["s"],
    )
    fp8_kv_fp8_math = _eval_op(
        analytical_db,
        GenerationAttention(
            "generation_attention_query",
            1.0,
            kwargs["n"],
            kwargs["n_kv"],
            common.KVCacheQuantMode.fp8,
            fmha_quant_mode=common.FMHAQuantMode.fp8,
        ),
        is_context=False,
        batch_size=kwargs["b"],
        s=kwargs["s"],
    )

    assert float(fp8_kv_bf16_math) < float(bf16_kv_bf16_math)
    assert float(fp8_kv_bf16_math) != pytest.approx(float(fp8_kv_fp8_math), rel=1e-4)


def test_fp8_kv_bf16_attention_and_mla_need_no_fp8_peak():
    database = get_database_view(
        "_dom_br100_64",
        "sglang",
        "estimate",
        allow_missing_data=True,
        database_mode="ANALYTICAL",
        allow_unlisted_version=True,
    )
    assert database is not None

    from aiconfigurator_core.sdk.operations.attention import ContextAttention, GenerationAttention
    from aiconfigurator_core.sdk.operations.mla import GenerationMLA, MLABmm

    context = _eval_op(
        database,
        ContextAttention(
            "context_attention_query",
            1.0,
            32,
            8,
            common.KVCacheQuantMode.fp8,
            common.FMHAQuantMode.bfloat16,
        ),
        is_context=True,
        batch_size=1,
        s=1024,
    )
    generation = _eval_op(
        database,
        GenerationAttention(
            "generation_attention_query",
            1.0,
            32,
            8,
            common.KVCacheQuantMode.fp8,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        ),
        is_context=False,
        batch_size=8,
        s=8192,
    )
    mla = _eval_op(
        database,
        GenerationMLA("generation_mla_query", 1.0, 16, common.KVCacheQuantMode.fp8),
        is_context=False,
        batch_size=8,
        s=8192,
    )
    bf16_bmm = _eval_op(
        database,
        MLABmm("mla_bmm_query", 1.0, 16, common.GEMMQuantMode.bfloat16),
        is_context=False,
        batch_size=8,
        s=1,
    )
    fp8_kv_bmm = _eval_op(
        database,
        MLABmm("mla_bmm_query", 1.0, 16, common.GEMMQuantMode.fp8),
        is_context=False,
        batch_size=8,
        s=1,
    )

    assert context.source == generation.source == mla.source == bf16_bmm.source == fp8_kv_bmm.source == "analytical"
    assert all(float(result) > 0 for result in (context, generation, mla, bf16_bmm, fp8_kv_bmm))
    assert float(fp8_kv_bmm) == pytest.approx(float(bf16_bmm))


def test_mla_fp8_proxy_preserves_fp8_kv_memory_sizing():
    def build(kv_mode):
        return get_model(
            "moonshotai/Kimi-K3",
            sdk_config.ModelConfig(
                tp_size=16,
                moe_tp_size=1,
                moe_ep_size=16,
                gemm_quant_mode=common.GEMMQuantMode.fp8,
                moe_quant_mode=common.MoEQuantMode.w4a16_mxfp4,
                kvcache_quant_mode=kv_mode,
                fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            ),
            backend_name="sglang",
        )

    bf16 = build(common.KVCacheQuantMode.bfloat16)
    fp8 = build(common.KVCacheQuantMode.fp8)
    seq_len = 4096
    bf16_per_token = bf16.get_kvcache_bytes_per_sequence(seq_len + 1) - bf16.get_kvcache_bytes_per_sequence(seq_len)
    fp8_per_token = fp8.get_kvcache_bytes_per_sequence(seq_len + 1) - fp8.get_kvcache_bytes_per_sequence(seq_len)

    assert fp8_per_token == pytest.approx(bf16_per_token / 2)


def test_rust_request_falls_back_to_python(analytical_db):
    runtime = RuntimeConfig(engine_step_backend="rust")
    assert should_use_rust_engine_step(runtime, analytical_db)


def test_table_free_communication_and_dtype_scaling():
    database = get_database_view(
        "h100_sxm",
        "sglang",
        "current",
        allow_missing_data=True,
        database_mode="EMPIRICAL",
    )
    from aiconfigurator_core.sdk.operations.communication import NCCL

    half = _eval_op(
        database,
        NCCL("nccl_query", 1.0, "all_reduce", 1_000_000, 8, common.CommQuantMode.half),
        is_context=True,
        batch_size=1,
        s=1,
        x=1,
    )
    int8 = _eval_op(
        database,
        NCCL("nccl_query", 1.0, "all_reduce", 1_000_000, 8, common.CommQuantMode.int8),
        is_context=True,
        batch_size=1,
        s=1,
        x=1,
    )
    assert half.source == int8.source == "empirical"
    assert 0 < float(int8) < float(half)


def test_non_sglang_warns_but_remains_usable(caplog):
    with caplog.at_level(logging.WARNING):
        database = get_database_view(
            "h100_sxm",
            "trtllm",
            "current",
            allow_missing_data=True,
            database_mode="ANALYTICAL",
        )
    assert database is not None
    assert "calibrated to SGLang" in caplog.text
    from aiconfigurator_core.sdk.operations.gemm import GEMM

    assert float(
        _eval_op(
            database,
            GEMM("gemm_query", 1.0, 1024, 1024, common.GEMMQuantMode.bfloat16),
            is_context=True,
            batch_size=1,
            s=64,
        )
    ) > 0
