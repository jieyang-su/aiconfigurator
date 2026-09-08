import pytest

from aiconfigurator_core.sdk.kernelsim.analytical import AnalyticalConfig, msa_sparse_attention_latency_ms
from aiconfigurator_core.sdk.kernelsim.msa import (
    MsaIndexModelWarning,
    MsaIndexShape,
    estimate_msa_index,
)

pytestmark = pytest.mark.unit


def _estimate(shape, level="standard"):
    with pytest.warns(MsaIndexModelWarning):
        return estimate_msa_index(
            shape,
            sm_count=132,
            hbm_bandwidth_bytes_s=3.35e12,
            parameter_level=level,
        )


def test_prefill_and_decode_snapshots():
    prefill = _estimate(MsaIndexShape("prefill", 1, 128, 4096, 4))
    decode = _estimate(MsaIndexShape("decode", 1, 1, 4096, 4))
    assert prefill.latency_us == pytest.approx(73.96053727102512)
    assert decode.latency_us == pytest.approx(4.163601291188894)
    assert prefill.executed_heads == 4
    assert decode.executed_heads == 16


def test_profiles_are_monotonic_and_scope_is_reported():
    shape = MsaIndexShape("decode", 8, 1, 32768, 4)
    low = _estimate(shape, "low")
    standard = _estimate(shape)
    high = _estimate(shape, "high")
    assert low.latency_us < standard.latency_us < high.latency_us
    assert "one H100" in standard.warnings[0]


def test_decode_rejects_multi_query_recipe():
    with pytest.raises(ValueError, match="query_length=1"):
        MsaIndexShape("decode", 1, 2, 4096, 4)


def test_selected_attention_scales_with_real_sparse_pairs():
    gpu = {
        "bfloat16_tc_flops": 0.989e15,
        "vector_peak_flops": 6.0e13,
        "mem_bw": 3.35e12,
    }
    kwargs = dict(
        gpu=gpu,
        batch=1,
        query_length=13_108,
        query_heads=64,
        kv_heads=4,
        head_dim=128,
        value_head_dim=128,
        compute_dtype="bf16",
        kv_cache_bytes_per_element=1,
        block_size=128,
        config=AnalyticalConfig(),
    )
    sparse = msa_sparse_attention_latency_ms(selected_pairs=13_108 * 2_048, **kwargs)
    accidental_dense = msa_sparse_attention_latency_ms(selected_pairs=13_108 * 13_108, **kwargs)
    assert 0 < sparse < accidental_dense
