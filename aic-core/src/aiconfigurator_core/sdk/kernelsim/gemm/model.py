#!/usr/bin/env python3
"""Final empirical latency models for the three selected AIC GEMM paths.

The module is intentionally self-contained: it has no artifact files and no
third-party dependencies. Hardware rates use FLOP/s and byte/s; latency output
uses microseconds.
"""

from __future__ import annotations

import math
import warnings as python_warnings
from dataclasses import dataclass

EMPIRICAL_MODEL_VERSION = "2026-07-29.aic-gemm-empirical-v3"
W8A16_TRANSFER_MODEL_VERSION = "2026-08-24.w8a16-int8wo-bf16-fused-transition-proxy-v4"
W8A16_TRANSFER_LIMITATION = (
    "W8A16 dense GEMM uses an uncalibrated BF16-compute compatibility proxy. It "
    "retains BF16 activations, outputs, and peak FLOPs while accounting for INT8 "
    "weight and scale traffic in the fused GEMM body. Its launch and utilization "
    "and a fitted-style launch/body transition term are conservatively degraded from "
    "the BF16 model, but it does not model a "
    "separate quantization kernel or materialized BF16 weight traffic: SGLang "
    "performs dequantization inside the kernel. Fused dequantization, packing/tiling, "
    "small-M behavior, backend differences, scale granularity, and non-NVIDIA "
    "hardware are not calibrated. The current SGLang GEMM collector also does not "
    "fully connect its declared int8_wo case. Treat this estimate as low confidence. "
    "This is an ANALYTICAL transfer proxy, not a Silicon calibration row, and must "
    "not be reported as measured W8A16 performance."
)


class W8A16TransferModelWarning(UserWarning):
    """Warning for the deliberately limited W8A16 dense GEMM recipe."""


_w8a16_warning_emitted = False


def _warn_w8a16_transfer_once() -> None:
    global _w8a16_warning_emitted
    if _w8a16_warning_emitted:
        return
    _w8a16_warning_emitted = True
    python_warnings.warn(W8A16_TRANSFER_LIMITATION, W8A16TransferModelWarning, stacklevel=3)


@dataclass(frozen=True)
class Bf16Sum3PParameters:
    t_launch_us: float = 2.0
    eta_mem: float = 0.77
    eta_compute: float = 0.90


@dataclass(frozen=True)
class W8A16GatedTransferParameters:
    """Engineering parameters for the fused, uncalibrated W8A16 proxy."""

    t_launch_us: float = 5.0
    eta_mem: float = 0.70
    eta_compute: float = 0.80
    rho_transition: float = 1.5


@dataclass(frozen=True)
class DeepGemmMax8PParameters:
    t_floor_hopper_us: float = 4.0
    t_floor_blackwell_us: float = 10.0
    eta_mem_hopper: float = 0.86
    eta_mem_blackwell: float = 0.48
    eta_compute: float = 0.65
    eta_quant_hopper: float = 0.68
    eta_quant_blackwell: float = 0.34
    rho_transition: float = 2.1


@dataclass(frozen=True)
class SglangFp8Max5PParameters:
    t_floor_us: float = 5.0
    eta_mem: float = 0.70
    eta_compute: float = 0.62
    eta_quant: float = 0.46
    rho_transition: float = 1.5


@dataclass(frozen=True)
class LatencyBreakdown:
    model: str
    latency_us: float
    launch_us: float
    body_us: float
    transition_us: float
    quant_us: float
    gemm_memory_us: float
    compute_us: float
    roofline_branch: str
    flops: float
    gemm_bytes: float
    quant_bytes: float


BF16_SUM_3P_PRECISE = Bf16Sum3PParameters(
    t_launch_us=2.0777777827803106,
    eta_mem=0.7682798890205326,
    eta_compute=0.8953233499328733,
)
BF16_SUM_3P_STANDARD = Bf16Sum3PParameters()
BF16_SUM_3P_HIGH = Bf16Sum3PParameters(
    t_launch_us=3.0,
    eta_mem=0.60,
    eta_compute=0.70,
)
BF16_SUM_3P_LOW = Bf16Sum3PParameters(
    t_launch_us=1.5,
    eta_mem=0.96,
    eta_compute=1.0,
)

# No pure-W8A16 collector data are available. "precise" therefore deliberately
# aliases the standard engineering parameters instead of implying a fitted model.
W8A16_GATED_TRANSFER_PRECISE = W8A16GatedTransferParameters()
W8A16_GATED_TRANSFER_STANDARD = W8A16GatedTransferParameters()
W8A16_GATED_TRANSFER_HIGH = W8A16GatedTransferParameters(
    t_launch_us=6.5,
    eta_mem=0.58,
    eta_compute=0.65,
    rho_transition=1.9,
)
W8A16_GATED_TRANSFER_LOW = W8A16GatedTransferParameters(
    t_launch_us=4.0,
    eta_mem=0.88,
    eta_compute=0.95,
    rho_transition=1.1,
)

DEEPGEMM_MAX_8P_PRECISE = DeepGemmMax8PParameters(
    t_floor_hopper_us=3.8656665322681274,
    t_floor_blackwell_us=10.378777877324156,
    eta_mem_hopper=0.8581147783419518,
    eta_mem_blackwell=0.48039321452379946,
    eta_compute=0.6486839517063081,
    eta_quant_hopper=0.6765141735130472,
    eta_quant_blackwell=0.3368848516246155,
    rho_transition=2.1264741995206062,
)
DEEPGEMM_MAX_8P_STANDARD = DeepGemmMax8PParameters()
DEEPGEMM_MAX_8P_HIGH = DeepGemmMax8PParameters(
    t_floor_hopper_us=5.0,
    t_floor_blackwell_us=13.0,
    eta_mem_hopper=0.72,
    eta_mem_blackwell=0.40,
    eta_compute=0.54,
    eta_quant_hopper=0.57,
    eta_quant_blackwell=0.28,
    rho_transition=2.6,
)
DEEPGEMM_MAX_8P_LOW = DeepGemmMax8PParameters(
    t_floor_hopper_us=3.0,
    t_floor_blackwell_us=8.0,
    eta_mem_hopper=1.0,
    eta_mem_blackwell=0.60,
    eta_compute=0.81,
    eta_quant_hopper=0.85,
    eta_quant_blackwell=0.43,
    rho_transition=1.6,
)

SGLANG_FP8_MAX_5P_PRECISE = SglangFp8Max5PParameters(
    t_floor_us=4.985881534715493,
    eta_mem=0.7011993788391371,
    eta_compute=0.6165923972877984,
    eta_quant=0.4626841110323739,
    rho_transition=1.5129826018401102,
)
SGLANG_FP8_MAX_5P_STANDARD = SglangFp8Max5PParameters()
SGLANG_FP8_MAX_5P_HIGH = SglangFp8Max5PParameters(
    t_floor_us=6.5,
    eta_mem=0.58,
    eta_compute=0.52,
    eta_quant=0.38,
    rho_transition=1.9,
)
SGLANG_FP8_MAX_5P_LOW = SglangFp8Max5PParameters(
    t_floor_us=4.0,
    eta_mem=0.88,
    eta_compute=0.78,
    eta_quant=0.58,
    rho_transition=1.1,
)

# Backward-compatible names retain the v1 rounded engineering parameters.
BF16_SUM_3P = BF16_SUM_3P_STANDARD
DEEPGEMM_MAX_8P = DEEPGEMM_MAX_8P_STANDARD
SGLANG_FP8_MAX_5P = SGLANG_FP8_MAX_5P_STANDARD
W8A16_GATED_TRANSFER = W8A16_GATED_TRANSFER_STANDARD

_BF16_PARAMETER_LEVELS = {
    "precise": BF16_SUM_3P_PRECISE,
    "standard": BF16_SUM_3P_STANDARD,
    "high": BF16_SUM_3P_HIGH,
    "low": BF16_SUM_3P_LOW,
}
_W8A16_PARAMETER_LEVELS = {
    "precise": W8A16_GATED_TRANSFER_PRECISE,
    "standard": W8A16_GATED_TRANSFER_STANDARD,
    "high": W8A16_GATED_TRANSFER_HIGH,
    "low": W8A16_GATED_TRANSFER_LOW,
}
_DEEPGEMM_PARAMETER_LEVELS = {
    "precise": DEEPGEMM_MAX_8P_PRECISE,
    "standard": DEEPGEMM_MAX_8P_STANDARD,
    "high": DEEPGEMM_MAX_8P_HIGH,
    "low": DEEPGEMM_MAX_8P_LOW,
}
_SGLANG_FP8_PARAMETER_LEVELS = {
    "precise": SGLANG_FP8_MAX_5P_PRECISE,
    "standard": SGLANG_FP8_MAX_5P_STANDARD,
    "high": SGLANG_FP8_MAX_5P_HIGH,
    "low": SGLANG_FP8_MAX_5P_LOW,
}


def _parameter_level(value: str) -> str:
    level = value.strip().lower()
    if level not in {"precise", "standard", "high", "low"}:
        raise ValueError("parameter_level must be precise, standard, high, or low")
    return level


def get_bf16_parameters(level: str = "standard") -> Bf16Sum3PParameters:
    return _BF16_PARAMETER_LEVELS[_parameter_level(level)]


def get_w8a16_parameters(level: str = "standard") -> W8A16GatedTransferParameters:
    return _W8A16_PARAMETER_LEVELS[_parameter_level(level)]


def get_deepgemm_parameters(level: str = "standard") -> DeepGemmMax8PParameters:
    return _DEEPGEMM_PARAMETER_LEVELS[_parameter_level(level)]


def get_sglang_fp8_parameters(level: str = "standard") -> SglangFp8Max5PParameters:
    return _SGLANG_FP8_PARAMETER_LEVELS[_parameter_level(level)]


def _resolve_parameters(level: str, params, getter):
    normalized = _parameter_level(level)
    if params is not None:
        if normalized != "standard":
            raise ValueError("custom params cannot be combined with a non-standard parameter_level")
        return params
    return getter(normalized)


def _positive_shape(m: int, n: int, k: int) -> tuple[int, int, int]:
    m, n, k = int(m), int(n), int(k)
    if min(m, n, k) <= 0:
        raise ValueError("m, n, and k must be positive")
    return m, n, k


def _positive_rate(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite value")
    return value


def _nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a nonnegative finite value")
    return value


def _architecture(value: str) -> str:
    architecture = value.strip().lower()
    if architecture not in {"hopper", "blackwell"}:
        raise ValueError("architecture must be 'hopper' or 'blackwell'")
    return architecture


def _us(seconds: float) -> float:
    return seconds * 1e6


def estimate_bf16_gemm(
    m: int,
    n: int,
    k: int,
    peak_bf16_flops: float,
    mem_bandwidth_bytes_s: float,
    params: Bf16Sum3PParameters | None = None,
    *,
    parameter_level: str = "standard",
) -> LatencyBreakdown:
    """Estimate a BF16 GEMM with the empirical Balanced Sum 3P model."""
    params = _resolve_parameters(parameter_level, params, get_bf16_parameters)
    m, n, k = _positive_shape(m, n, k)
    peak = _positive_rate("peak_bf16_flops", peak_bf16_flops)
    bandwidth = _positive_rate("mem_bandwidth_bytes_s", mem_bandwidth_bytes_s)
    launch_us = _nonnegative("params.t_launch_us", params.t_launch_us)
    eta_mem = _positive_rate("params.eta_mem", params.eta_mem)
    eta_compute = _positive_rate("params.eta_compute", params.eta_compute)

    flops = 2.0 * m * n * k
    gemm_bytes = 2.0 * (m * k + k * n + m * n)
    memory_s = gemm_bytes / bandwidth / eta_mem
    compute_s = flops / peak / eta_compute
    launch_s = launch_us * 1e-6
    body_s = memory_s + compute_s
    return LatencyBreakdown(
        model="bf16_sum_3p_empirical",
        latency_us=_us(launch_s + body_s),
        launch_us=launch_us,
        body_us=_us(body_s),
        transition_us=0.0,
        quant_us=0.0,
        gemm_memory_us=_us(memory_s),
        compute_us=_us(compute_s),
        roofline_branch="sum",
        flops=flops,
        gemm_bytes=gemm_bytes,
        quant_bytes=0.0,
    )


def estimate_w8a16_gemm(
    m: int,
    n: int,
    k: int,
    peak_bf16_flops: float,
    mem_bandwidth_bytes_s: float,
    params: W8A16GatedTransferParameters | None = None,
    *,
    parameter_level: str = "standard",
) -> LatencyBreakdown:
    """Estimate a fused INT8-weight/BF16-activation GEMM compatibility proxy."""
    _warn_w8a16_transfer_once()
    params = _resolve_parameters(parameter_level, params, get_w8a16_parameters)
    m, n, k = _positive_shape(m, n, k)
    peak = _positive_rate("peak_bf16_flops", peak_bf16_flops)
    bandwidth = _positive_rate("mem_bandwidth_bytes_s", mem_bandwidth_bytes_s)
    launch_us = _nonnegative("params.t_launch_us", params.t_launch_us)
    eta_mem = _positive_rate("params.eta_mem", params.eta_mem)
    eta_compute = _positive_rate("params.eta_compute", params.eta_compute)
    rho = _nonnegative("params.rho_transition", params.rho_transition)

    flops = 2.0 * m * n * k
    scale_bytes = 4.0 * n
    gemm_bytes = 2.0 * m * k + n * k + scale_bytes + 2.0 * m * n
    memory_s = gemm_bytes / bandwidth / eta_mem
    compute_s = flops / peak / eta_compute
    body_s = memory_s + compute_s
    launch_s = launch_us * 1e-6
    transition_s = rho * launch_s * body_s / (body_s + launch_s)
    return LatencyBreakdown(
        model="w8a16_int8wo_bf16_fused_transition_proxy_v4",
        latency_us=_us(launch_s + body_s + transition_s),
        launch_us=launch_us,
        body_us=_us(body_s),
        transition_us=_us(transition_s),
        quant_us=0.0,
        gemm_memory_us=_us(memory_s),
        compute_us=_us(compute_s),
        roofline_branch="sum",
        flops=flops,
        gemm_bytes=gemm_bytes,
        quant_bytes=0.0,
    )


def estimate_deepgemm_fp8(
    m: int,
    n: int,
    k: int,
    peak_fp8_flops: float,
    mem_bandwidth_bytes_s: float,
    architecture: str,
    params: DeepGemmMax8PParameters | None = None,
    *,
    parameter_level: str = "standard",
) -> LatencyBreakdown:
    """Estimate activation quantization plus FP8-block DeepGEMM."""
    params = _resolve_parameters(parameter_level, params, get_deepgemm_parameters)
    m, n, k = _positive_shape(m, n, k)
    peak = _positive_rate("peak_fp8_flops", peak_fp8_flops)
    bandwidth = _positive_rate("mem_bandwidth_bytes_s", mem_bandwidth_bytes_s)
    architecture = _architecture(architecture)
    if n < 128 or k < 128:
        raise ValueError("DeepGEMM model requires n and k >= 128")
    eta_compute = _positive_rate("params.eta_compute", params.eta_compute)
    rho = _nonnegative("params.rho_transition", params.rho_transition)

    scale_a_bytes = 4.0 * m * math.ceil(k / 128)
    scale_b_bytes = 4.0 * math.ceil(n / 128) * math.ceil(k / 128)
    flops = 2.0 * m * n * k
    gemm_bytes = m * k + n * k + 2.0 * m * n + scale_a_bytes + scale_b_bytes
    quant_bytes = 3.0 * m * k + scale_a_bytes

    if architecture == "hopper":
        floor_us = params.t_floor_hopper_us
        eta_mem = params.eta_mem_hopper
        eta_quant = params.eta_quant_hopper
    else:
        floor_us = params.t_floor_blackwell_us
        eta_mem = params.eta_mem_blackwell
        eta_quant = params.eta_quant_blackwell
    floor_us = _nonnegative("architecture floor", floor_us)
    eta_mem = _positive_rate("architecture eta_mem", eta_mem)
    eta_quant = _positive_rate("architecture eta_quant", eta_quant)

    memory_s = gemm_bytes / bandwidth / eta_mem
    compute_s = flops / peak / eta_compute
    quant_s = quant_bytes / bandwidth / eta_quant
    gemm_s = max(memory_s, compute_s)
    body_s = quant_s + gemm_s
    floor_s = floor_us * 1e-6
    transition_s = rho * floor_s * body_s / (body_s + floor_s)
    return LatencyBreakdown(
        model="deepgemm_fp8_gated_serial_max_8p_empirical",
        latency_us=_us(floor_s + body_s + transition_s),
        launch_us=floor_us,
        body_us=_us(body_s),
        transition_us=_us(transition_s),
        quant_us=_us(quant_s),
        gemm_memory_us=_us(memory_s),
        compute_us=_us(compute_s),
        roofline_branch="memory" if memory_s >= compute_s else "compute",
        flops=flops,
        gemm_bytes=gemm_bytes,
        quant_bytes=quant_bytes,
    )


def estimate_sglang_fp8(
    m: int,
    n: int,
    k: int,
    peak_fp8_flops: float,
    mem_bandwidth_bytes_s: float,
    params: SglangFp8Max5PParameters | None = None,
    *,
    parameter_level: str = "standard",
) -> LatencyBreakdown:
    """Estimate per-token quantization plus SGLang FP8 scaled-mm."""
    params = _resolve_parameters(parameter_level, params, get_sglang_fp8_parameters)
    m, n, k = _positive_shape(m, n, k)
    peak = _positive_rate("peak_fp8_flops", peak_fp8_flops)
    bandwidth = _positive_rate("mem_bandwidth_bytes_s", mem_bandwidth_bytes_s)
    floor_us = _nonnegative("params.t_floor_us", params.t_floor_us)
    eta_mem = _positive_rate("params.eta_mem", params.eta_mem)
    eta_compute = _positive_rate("params.eta_compute", params.eta_compute)
    eta_quant = _positive_rate("params.eta_quant", params.eta_quant)
    rho = _nonnegative("params.rho_transition", params.rho_transition)

    flops = 2.0 * m * n * k
    gemm_bytes = m * k + n * k + 2.0 * m * n + 4.0 * m + 4.0 * n
    quant_bytes = 3.0 * m * k + 4.0 * m
    memory_s = gemm_bytes / bandwidth / eta_mem
    compute_s = flops / peak / eta_compute
    quant_s = quant_bytes / bandwidth / eta_quant
    gemm_s = max(memory_s, compute_s)
    body_s = quant_s + gemm_s
    floor_s = floor_us * 1e-6
    transition_s = rho * floor_s * body_s / (body_s + floor_s)
    return LatencyBreakdown(
        model="sglang_fp8_gated_serial_max_5p_empirical",
        latency_us=_us(floor_s + body_s + transition_s),
        launch_us=floor_us,
        body_us=_us(body_s),
        transition_us=_us(transition_s),
        quant_us=_us(quant_s),
        gemm_memory_us=_us(memory_s),
        compute_us=_us(compute_s),
        roofline_branch="memory" if memory_s >= compute_s else "compute",
        flops=flops,
        gemm_bytes=gemm_bytes,
        quant_bytes=quant_bytes,
    )


def bf16_gemm_latency_us(*args, **kwargs) -> float:
    return estimate_bf16_gemm(*args, **kwargs).latency_us


def w8a16_gemm_latency_us(*args, **kwargs) -> float:
    return estimate_w8a16_gemm(*args, **kwargs).latency_us


def deepgemm_fp8_latency_us(*args, **kwargs) -> float:
    return estimate_deepgemm_fp8(*args, **kwargs).latency_us


def sglang_fp8_latency_us(*args, **kwargs) -> float:
    return estimate_sglang_fp8(*args, **kwargs).latency_us


__all__ = [
    "BF16_SUM_3P",
    "BF16_SUM_3P_HIGH",
    "BF16_SUM_3P_LOW",
    "BF16_SUM_3P_PRECISE",
    "BF16_SUM_3P_STANDARD",
    "DEEPGEMM_MAX_8P",
    "DEEPGEMM_MAX_8P_HIGH",
    "DEEPGEMM_MAX_8P_LOW",
    "DEEPGEMM_MAX_8P_PRECISE",
    "DEEPGEMM_MAX_8P_STANDARD",
    "EMPIRICAL_MODEL_VERSION",
    "SGLANG_FP8_MAX_5P",
    "SGLANG_FP8_MAX_5P_HIGH",
    "SGLANG_FP8_MAX_5P_LOW",
    "SGLANG_FP8_MAX_5P_PRECISE",
    "SGLANG_FP8_MAX_5P_STANDARD",
    "W8A16_GATED_TRANSFER",
    "W8A16_GATED_TRANSFER_HIGH",
    "W8A16_GATED_TRANSFER_LOW",
    "W8A16_GATED_TRANSFER_PRECISE",
    "W8A16_GATED_TRANSFER_STANDARD",
    "W8A16_TRANSFER_LIMITATION",
    "W8A16_TRANSFER_MODEL_VERSION",
    "Bf16Sum3PParameters",
    "DeepGemmMax8PParameters",
    "LatencyBreakdown",
    "SglangFp8Max5PParameters",
    "W8A16GatedTransferParameters",
    "W8A16TransferModelWarning",
    "bf16_gemm_latency_us",
    "deepgemm_fp8_latency_us",
    "estimate_bf16_gemm",
    "estimate_deepgemm_fp8",
    "estimate_sglang_fp8",
    "estimate_w8a16_gemm",
    "get_bf16_parameters",
    "get_deepgemm_parameters",
    "get_sglang_fp8_parameters",
    "get_w8a16_parameters",
    "sglang_fp8_latency_us",
    "w8a16_gemm_latency_us",
]
