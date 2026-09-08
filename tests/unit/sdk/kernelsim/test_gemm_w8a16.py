from __future__ import annotations

import math
import warnings

import pytest

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.kernelsim import analytical
from aiconfigurator_core.sdk.kernelsim.gemm import model as gemm_model


@pytest.fixture(autouse=True)
def reset_w8a16_warning() -> None:
    gemm_model._w8a16_warning_emitted = False


def test_w8a16_uses_conservative_fused_proxy() -> None:
    shape = (128, 4096, 4096)
    kwargs = {
        "peak_bf16_flops": 989e12,
        "mem_bandwidth_bytes_s": 3.35e12,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", gemm_model.W8A16TransferModelWarning)
        bf16 = gemm_model.estimate_bf16_gemm(*shape, **kwargs)
        w8a16 = gemm_model.estimate_w8a16_gemm(*shape, **kwargs)

    m, n, k = shape
    assert w8a16.flops == bf16.flops == 2.0 * m * n * k
    assert w8a16.launch_us == 5.0
    assert w8a16.compute_us > bf16.compute_us
    assert w8a16.quant_bytes == 0.0
    assert w8a16.gemm_bytes == 2.0 * m * k + n * k + 4.0 * n + 2.0 * m * n
    assert w8a16.gemm_bytes < bf16.gemm_bytes
    assert w8a16.quant_us == 0.0
    assert w8a16.transition_us > 0.0
    assert w8a16.latency_us == pytest.approx(
        w8a16.launch_us
        + w8a16.gemm_memory_us
        + w8a16.compute_us
        + w8a16.transition_us
    )


def test_w8a16_standard_parameters_match_the_engineering_proxy() -> None:
    params = gemm_model.get_w8a16_parameters()
    assert params.t_launch_us == 5.0
    assert params.eta_mem == 0.70
    assert params.eta_compute == 0.80
    assert params.rho_transition == 1.5
    assert gemm_model.get_w8a16_parameters("precise") == params


def test_w8a16_warning_is_emitted_once() -> None:
    kwargs = {
        "m": 8,
        "n": 16,
        "k": 32,
        "peak_bf16_flops": 1e12,
        "mem_bandwidth_bytes_s": 1e12,
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gemm_model.estimate_w8a16_gemm(**kwargs)
        gemm_model.estimate_w8a16_gemm(**kwargs)

    transfer = [item for item in caught if item.category is gemm_model.W8A16TransferModelWarning]
    assert len(transfer) == 1
    assert "low confidence" in str(transfer[0].message)
    assert "dequantization" in str(transfer[0].message)
    assert "not a Silicon calibration row" in str(transfer[0].message)


def test_w8a16_parameter_levels_are_ordered() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", gemm_model.W8A16TransferModelWarning)
        results = {
            level: gemm_model.estimate_w8a16_gemm(
                128,
                4096,
                4096,
                989e12,
                3.35e12,
                parameter_level=level,
            )
            for level in ("low", "standard", "high")
        }
    assert all(math.isfinite(result.latency_us) and result.latency_us > 0 for result in results.values())
    assert results["low"].latency_us < results["standard"].latency_us < results["high"].latency_us
    assert results["low"].transition_us < results["standard"].transition_us < results["high"].transition_us


def test_analytical_w8a16_requires_only_bf16_peak() -> None:
    gpu = {"bfloat16_tc_flops": 989e12, "mem_bw": 3.35e12}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", gemm_model.W8A16TransferModelWarning)
        latency_ms = analytical.gemm_latency_ms(
            128,
            4096,
            4096,
            common.GEMMQuantMode.int8_wo,
            gpu,
            analytical.AnalyticalConfig(),
        )
        direct = gemm_model.estimate_w8a16_gemm(128, 4096, 4096, 989e12, 3.35e12)
    assert latency_ms == pytest.approx(direct.latency_us / 1000.0)
