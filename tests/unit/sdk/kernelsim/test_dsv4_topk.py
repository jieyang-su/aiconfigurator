from __future__ import annotations

import pytest

from aiconfigurator_core.sdk.kernelsim.dsv4_topk import (
    Dsv4TopKShape,
    Dsv4TopKWarning,
    estimate_dsv4_topk,
)


def estimate(shape: Dsv4TopKShape, level: str = "standard"):
    with pytest.warns(Dsv4TopKWarning):
        return estimate_dsv4_topk(shape, sm_count=132, parameter_level=level)


def test_v2_long_context_uses_cluster_recipe_without_linear_tail():
    result = estimate(
        Dsv4TopKShape(
            variant="v2",
            batch_size=1,
            fresh_tokens=1,
            prefix_tokens=1_048_574,
        )
    )
    assert result.regime == "v2-cluster"
    assert result.compressed_context == 262_143
    assert result.latency_us == pytest.approx(14.024, rel=0.01)
    assert result.latency_us < 20.0


def test_v1_long_prefill_tracks_causal_rows():
    result = estimate(
        Dsv4TopKShape(
            variant="v1",
            batch_size=1,
            fresh_tokens=8192,
            prefix_tokens=65536,
        )
    )
    assert result.regime == "v1-radix-per-row"
    assert result.active_rows == 8192
    assert result.latency_us == pytest.approx(707.21, rel=0.01)
    assert result.latency_us > 500.0


def test_v1_bypasses_topk_when_all_compressed_candidates_are_selected():
    result = estimate(Dsv4TopKShape(variant="v1", batch_size=8, fresh_tokens=128, prefix_tokens=0))
    assert result.regime == "trivial-bypassed"
    assert result.latency_us == 0.0


def test_parameter_levels_form_engineering_envelope():
    shape = Dsv4TopKShape(variant="v2", batch_size=1, fresh_tokens=1, prefix_tokens=262143)
    low = estimate(shape, "low").latency_us
    standard = estimate(shape).latency_us
    high = estimate(shape, "high").latency_us
    assert low == pytest.approx(standard * 0.75)
    assert high == pytest.approx(standard * 1.30)


def test_non_calibrated_topk_is_explicitly_warned():
    with pytest.warns(Dsv4TopKWarning, match="topk=1024"):
        result = estimate_dsv4_topk(
            Dsv4TopKShape(
                variant="v2",
                batch_size=1,
                fresh_tokens=1,
                prefix_tokens=65535,
                topk=1024,
            ),
            sm_count=132,
        )
    assert any("topk=1024" in warning for warning in result.warnings)
