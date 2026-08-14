import logging

import pytest

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk import config as sdk_config
from aiconfigurator_core.sdk.config import RuntimeConfig
from aiconfigurator_core.sdk.kernelsim.analytical import AnalyticalConfig
from aiconfigurator_core.sdk.models import get_model
from aiconfigurator_core.sdk.perf_database import get_database_view
from aiconfigurator_core.sdk.rust_engine_step import should_use_rust_engine_step


@pytest.fixture
def analytical_db():
    database = get_database_view(
        "h100_sxm",
        "sglang",
        "estimate",
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
        "estimate",
        allow_missing_data=True,
        database_mode="ANALYTICAL",
    )
    high = get_database_view(
        "h100_sxm",
        "sglang",
        "estimate",
        allow_missing_data=True,
        database_mode="ANALYTICAL",
        analytical_config={"level": "high"},
    )
    assert standard is not high
    assert standard._analytical_config.level == "standard"
    assert high._analytical_config.level == "high"


def test_gemm_attention_and_mla_boundaries(analytical_db):
    gemm = analytical_db.query_gemm(128, 4096, 4096, common.GEMMQuantMode.fp8)
    attention = analytical_db.query_context_attention(
        1,
        1024,
        0,
        32,
        8,
        common.KVCacheQuantMode.bfloat16,
        common.FMHAQuantMode.bfloat16,
    )
    assert gemm.source == attention.source == "analytical"
    assert float(gemm) > 0 and float(attention) > 0

    bf16_kv = analytical_db.query_generation_mla(8, 4096, 16, common.KVCacheQuantMode.bfloat16)
    fp8_kv = analytical_db.query_generation_mla(8, 4096, 16, common.KVCacheQuantMode.fp8)
    assert fp8_kv.source == "analytical"
    assert float(fp8_kv) == pytest.approx(float(bf16_kv))

    with pytest.raises(ValueError, match="BF16 compute only"):
        analytical_db.query_context_mla(
            1,
            1024,
            0,
            16,
            common.KVCacheQuantMode.fp8,
            common.FMHAQuantMode.fp8,
        )


def test_attention_separates_compute_dtype_from_kv_storage(analytical_db):
    kwargs = dict(b=8, s=8192, n=32, n_kv=8)
    bf16_kv_bf16_math = analytical_db.query_generation_attention(
        **kwargs,
        kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
    )
    fp8_kv_bf16_math = analytical_db.query_generation_attention(
        **kwargs,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
    )
    fp8_kv_fp8_math = analytical_db.query_generation_attention(
        **kwargs,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.fp8,
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
    )
    assert database is not None

    context = database.query_context_attention(
        1,
        1024,
        0,
        32,
        8,
        common.KVCacheQuantMode.fp8,
        common.FMHAQuantMode.bfloat16,
    )
    generation = database.query_generation_attention(
        8,
        8192,
        32,
        8,
        common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
    )
    mla = database.query_generation_mla(8, 8192, 16, common.KVCacheQuantMode.fp8)
    bf16_bmm = database.query_mla_bmm(8, 16, common.GEMMQuantMode.bfloat16)
    fp8_kv_bmm = database.query_mla_bmm(8, 16, common.GEMMQuantMode.fp8)

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
    assert not should_use_rust_engine_step(runtime, analytical_db)


def test_table_free_communication_and_dtype_scaling():
    database = get_database_view(
        "h100_sxm",
        "sglang",
        "estimate",
        allow_missing_data=True,
        database_mode="ANALYTICAL",
    )
    half = database.query_custom_allreduce(common.CommQuantMode.half, 8, 1_000_000)
    fp8 = database.query_custom_allreduce(common.CommQuantMode.fp8, 8, 1_000_000)
    assert half.source == fp8.source == "empirical"
    assert float(half) == pytest.approx(2 * float(fp8))

    wideep = database.query_wideep_deepep_normal(
        1,
        64,
        256,
        8,
        7168,
        20,
        dispatch_dtype=common.CommQuantMode.fp8,
        combine_dtype=common.CommQuantMode.half,
    )
    assert wideep.source == "empirical"
    assert float(wideep) > 0


def test_non_sglang_warns_but_remains_usable(caplog):
    with caplog.at_level(logging.WARNING):
        database = get_database_view(
            "h100_sxm",
            "trtllm",
            "estimate",
            allow_missing_data=True,
            database_mode="ANALYTICAL",
        )
    assert database is not None
    assert "calibrated to SGLang" in caplog.text
    assert float(database.query_gemm(64, 1024, 1024, common.GEMMQuantMode.bfloat16)) > 0
