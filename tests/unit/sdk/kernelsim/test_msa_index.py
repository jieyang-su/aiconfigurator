import pytest

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
