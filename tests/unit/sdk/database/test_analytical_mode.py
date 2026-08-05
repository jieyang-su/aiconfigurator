import logging

import pytest

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.config import RuntimeConfig
from aiconfigurator_core.sdk.kernelsim.analytical import AnalyticalConfig
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
    assert AnalyticalConfig(fp8_gemm_recipe="deepgemm_hopper").fp8_gemm_recipe == "deepgemm-hopper"
    with pytest.raises(ValueError, match="standard, low, or high"):
        AnalyticalConfig(level="precise")
    with pytest.raises(ValueError, match="communication mode"):
        AnalyticalConfig(communication_mode="guess")


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

    with pytest.raises(ValueError, match="MLA supports BF16 only"):
        analytical_db.query_generation_mla(8, 4096, 16, common.KVCacheQuantMode.fp8)


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
