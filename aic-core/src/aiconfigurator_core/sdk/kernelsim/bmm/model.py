#!/usr/bin/env python3
"""Production empirical model for BF16 and compound-FP8 batched matmul.

The analytical work terms accept arbitrary positive B/M/N/K. Parameters were
calibrated on SGLang DeepSeek MLA pre/post BMM, so other matrix geometries are
explicitly reported as extrapolation. Hardware rates use FLOP/s and byte/s;
latency output uses microseconds.
"""

from __future__ import annotations

import math
import warnings as python_warnings
from dataclasses import dataclass, replace

EMPIRICAL_MODEL_VERSION = "2026-07-31.aic-bmm-empirical-v3"
FP8_WARNING_MESSAGE = (
    "FP8 BMM has low predictive confidence: the calibration data and fitted "
    "SGLang quant+bmm_fp8 recipe are substantially slower than BF16 for many "
    "shapes. Do not interpret this estimate as generic FP8 BMM performance."
)
GENERIC_SHAPE_WARNING = (
    "BMM parameters were calibrated only on DeepSeek MLA pre/post matrix "
    "geometries; this B/M/N/K prediction is a formula-level shape extrapolation."
)
COMPUTE_BRANCH_WARNING = (
    "The calibrated BMM dataset was memory-bound; eta_compute is an external "
    "GEMM prior and the compute-bound branch has not been validated by BMM data."
)


class BmmFp8ReliabilityWarning(UserWarning):
    """Warning emitted when using the low-confidence compound FP8 recipe."""


@dataclass(frozen=True)
class BmmParameters:
    t_floor_bf16_us: float = 4.2
    t_floor_fp8_us: float = 7.1
    eta_mem_bf16: float = 0.70
    eta_mem_fp8: float = 0.22
    eta_compute: float = 0.65


@dataclass(frozen=True)
class BmmLatencyBreakdown:
    model: str
    model_version: str
    parameter_level: str
    scope: str
    recipe: str
    op: str
    dtype: str
    warnings: tuple[str, ...]
    latency_us: float
    floor_us: float
    body_us: float
    memory_us: float
    compute_us: float
    roofline_branch: str
    eta_memory: float
    eta_compute: float
    batch: int
    m: int
    n: int
    k: int
    flops: float
    logical_bytes: float
    activation_bytes: float
    token_bytes: float
    weight_bytes: float
    quant_bytes: float
    gemm_bytes: float


# Precise retains the original warm-replay pooled refit for reproducibility.
BMM_PRECISE = BmmParameters(
    t_floor_bf16_us=4.1978933693220215,
    t_floor_fp8_us=7.136533359686535,
    eta_mem_bf16=0.8810094265528273,
    eta_mem_fp8=0.2194306605990924,
    eta_compute=0.65,
)

# BF16 engineering eta values include a 20% conservative full-payment
# correction for the collector's non-evicted L2 warm replay. FP8 is unchanged.
BMM_STANDARD = BmmParameters()
BMM_HIGH = BmmParameters(
    t_floor_bf16_us=5.5,
    t_floor_fp8_us=9.0,
    eta_mem_bf16=0.56,
    eta_mem_fp8=0.18,
    eta_compute=0.52,
)
BMM_LOW = BmmParameters(
    t_floor_bf16_us=3.0,
    t_floor_fp8_us=5.5,
    eta_mem_bf16=0.80,
    eta_mem_fp8=0.28,
    eta_compute=0.81,
)

BMM_FULL_ROOFLINE = BMM_STANDARD

# Backward-compatible names for existing DeepSeek MLA users.
MlaBmmParameters = BmmParameters
MlaBmmLatencyBreakdown = BmmLatencyBreakdown
MLA_BMM_PRECISE = BMM_PRECISE
MLA_BMM_STANDARD = BMM_STANDARD
MLA_BMM_HIGH = BMM_HIGH
MLA_BMM_LOW = BMM_LOW
MLA_BMM_FULL_ROOFLINE = BMM_FULL_ROOFLINE

_PARAMETER_LEVELS = {
    "precise": BMM_PRECISE,
    "standard": BMM_STANDARD,
    "high": BMM_HIGH,
    "low": BMM_LOW,
}


def _parameter_level(value: str) -> str:
    level = value.strip().lower()
    if level not in _PARAMETER_LEVELS:
        raise ValueError("parameter_level must be precise, standard, high, or low")
    return level


def get_bmm_parameters(level: str = "standard") -> BmmParameters:
    return _PARAMETER_LEVELS[_parameter_level(level)]


def get_mla_bmm_parameters(level: str = "standard") -> BmmParameters:
    """Backward-compatible alias for :func:`get_bmm_parameters`."""
    return get_bmm_parameters(level)


def _resolve_parameters(
    level: str,
    params: BmmParameters | None,
) -> tuple[str, BmmParameters]:
    normalized = _parameter_level(level)
    if params is not None:
        if normalized != "standard":
            raise ValueError("custom params cannot be combined with a non-standard parameter_level")
        return "custom", params
    return normalized, get_bmm_parameters(normalized)


def _positive_int(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


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


def normalize_op(value: str) -> str:
    op = value.strip().lower().removeprefix("mla_gen_")
    if op not in {"pre", "post"}:
        raise ValueError("op must be pre, post, mla_gen_pre, or mla_gen_post")
    return op


def normalize_dtype(value: str) -> str:
    dtype = value.strip().lower()
    if dtype in {"bf16", "bfloat16"}:
        return "bf16"
    if dtype in {"fp8", "float8", "float8_e4m3fn"}:
        return "fp8"
    raise ValueError("dtype must identify BF16 or FP8")


def _calibrated_op(n: int, k: int) -> str | None:
    if (n, k) == (512, 128):
        return "pre"
    if (n, k) == (128, 512):
        return "post"
    return None


def bmm_work_terms(
    batch: int,
    m: int,
    n: int,
    k: int,
    dtype: str,
) -> dict[str, float | int | str]:
    """Return logical full-payment work for arbitrary batched matmul geometry."""
    batch = _positive_int("batch", batch)
    m = _positive_int("m", m)
    n = _positive_int("n", n)
    k = _positive_int("k", k)
    dtype = normalize_dtype(dtype)
    calibrated_op = _calibrated_op(n, k)
    flops = 2.0 * batch * m * n * k

    if dtype == "bf16":
        activation_bytes = 2.0 * batch * m * (k + n)
        weight_bytes = 2.0 * batch * k * n
        quant_bytes = 0.0
        gemm_bytes = activation_bytes + weight_bytes
    else:
        scale_a_bytes = 4.0 * batch
        scale_b_bytes = 4.0 * batch * math.ceil(n / 128) * math.ceil(k / 128)
        quant_bytes = 3.0 * batch * m * k + scale_a_bytes
        gemm_bytes = batch * (m * k + k * n + 2.0 * m * n) + scale_a_bytes + scale_b_bytes
        activation_bytes = 4.0 * batch * m * k + 2.0 * batch * m * n + 2.0 * scale_a_bytes
        weight_bytes = batch * k * n + scale_b_bytes

    return {
        "op": calibrated_op or "generic",
        "scope": (
            "calibrated_deepseek_matrix_geometry" if calibrated_op is not None else "generic_shape_extrapolation"
        ),
        "recipe": "torch_bmm" if dtype == "bf16" else "sglang_quant_bmm_fp8",
        "dtype": dtype,
        "batch": batch,
        "m": m,
        "n": n,
        "k": k,
        "flops": flops,
        "logical_bytes": activation_bytes + weight_bytes,
        "activation_bytes": activation_bytes,
        "token_bytes": activation_bytes,
        "weight_bytes": weight_bytes,
        "quant_bytes": quant_bytes,
        "gemm_bytes": gemm_bytes,
    }


def work_terms(
    num_tokens: int,
    num_heads: int,
    op: str,
    dtype: str,
) -> dict[str, float | int | str]:
    """Backward-compatible DeepSeek MLA geometry wrapper."""
    tokens = _positive_int("num_tokens", num_tokens)
    heads = _positive_int("num_heads", num_heads)
    normalized_op = normalize_op(op)
    n, k = (512, 128) if normalized_op == "pre" else (128, 512)
    return bmm_work_terms(heads, tokens, n, k, dtype)


def _selected_parameters(
    params: BmmParameters,
    dtype: str,
) -> tuple[float, float]:
    floor_us = getattr(params, f"t_floor_{dtype}_us")
    eta_memory = getattr(params, f"eta_mem_{dtype}")
    return (
        _nonnegative("selected floor", floor_us),
        _positive_rate("selected eta_memory", eta_memory),
    )


def estimate_bmm(
    batch: int,
    m: int,
    n: int,
    k: int,
    dtype: str,
    peak_flops_s: float,
    mem_bandwidth_bytes_s: float,
    params: BmmParameters | None = None,
    *,
    parameter_level: str = "standard",
) -> BmmLatencyBreakdown:
    """Estimate arbitrary B/M/N/K with the calibrated full-traffic formula."""
    level, params = _resolve_parameters(parameter_level, params)
    peak = _positive_rate("peak_flops_s", peak_flops_s)
    bandwidth = _positive_rate("mem_bandwidth_bytes_s", mem_bandwidth_bytes_s)
    terms = bmm_work_terms(batch, m, n, k, dtype)
    dtype = str(terms["dtype"])
    if dtype == "fp8":
        python_warnings.warn(
            FP8_WARNING_MESSAGE,
            BmmFp8ReliabilityWarning,
            stacklevel=2,
        )
    floor_us, eta_memory = _selected_parameters(params, dtype)
    eta_compute = _positive_rate("params.eta_compute", params.eta_compute)

    memory_s = float(terms["logical_bytes"]) / bandwidth / eta_memory
    compute_s = float(terms["flops"]) / peak / eta_compute
    body_s = max(memory_s, compute_s)
    branch = "memory" if memory_s >= compute_s else "compute"
    warning_messages: list[str] = []
    if dtype == "fp8":
        warning_messages.append(FP8_WARNING_MESSAGE)
    if terms["scope"] == "generic_shape_extrapolation":
        warning_messages.append(GENERIC_SHAPE_WARNING)
    if branch == "compute":
        warning_messages.append(COMPUTE_BRANCH_WARNING)

    return BmmLatencyBreakdown(
        model="sglang_bmm_full_traffic_roofline",
        model_version=EMPIRICAL_MODEL_VERSION,
        parameter_level=level,
        scope=str(terms["scope"]),
        recipe=str(terms["recipe"]),
        op=str(terms["op"]),
        dtype=dtype,
        warnings=tuple(warning_messages),
        latency_us=floor_us + body_s * 1e6,
        floor_us=floor_us,
        body_us=body_s * 1e6,
        memory_us=memory_s * 1e6,
        compute_us=compute_s * 1e6,
        roofline_branch=branch,
        eta_memory=eta_memory,
        eta_compute=eta_compute,
        batch=int(terms["batch"]),
        m=int(terms["m"]),
        n=int(terms["n"]),
        k=int(terms["k"]),
        flops=float(terms["flops"]),
        logical_bytes=float(terms["logical_bytes"]),
        activation_bytes=float(terms["activation_bytes"]),
        token_bytes=float(terms["token_bytes"]),
        weight_bytes=float(terms["weight_bytes"]),
        quant_bytes=float(terms["quant_bytes"]),
        gemm_bytes=float(terms["gemm_bytes"]),
    )


def estimate_mla_bmm(
    num_tokens: int,
    num_heads: int,
    op: str,
    dtype: str,
    peak_flops_s: float,
    mem_bandwidth_bytes_s: float,
    params: BmmParameters | None = None,
    *,
    parameter_level: str = "standard",
) -> BmmLatencyBreakdown:
    """Backward-compatible DeepSeek MLA pre/post BMM wrapper."""
    normalized_op = normalize_op(op)
    n, k = (512, 128) if normalized_op == "pre" else (128, 512)
    result = estimate_bmm(
        num_heads,
        num_tokens,
        n,
        k,
        dtype,
        peak_flops_s,
        mem_bandwidth_bytes_s,
        params,
        parameter_level=parameter_level,
    )
    return replace(result, op=normalized_op)


def bmm_latency_us(*args, **kwargs) -> float:
    return estimate_bmm(*args, **kwargs).latency_us


def bmm_latency_s(*args, **kwargs) -> float:
    return bmm_latency_us(*args, **kwargs) * 1e-6


def mla_bmm_latency_us(*args, **kwargs) -> float:
    return estimate_mla_bmm(*args, **kwargs).latency_us


def mla_bmm_latency_s(*args, **kwargs) -> float:
    return mla_bmm_latency_us(*args, **kwargs) * 1e-6


__all__ = [
    "BMM_FULL_ROOFLINE",
    "BMM_HIGH",
    "BMM_LOW",
    "BMM_PRECISE",
    "BMM_STANDARD",
    "COMPUTE_BRANCH_WARNING",
    "EMPIRICAL_MODEL_VERSION",
    "FP8_WARNING_MESSAGE",
    "GENERIC_SHAPE_WARNING",
    "MLA_BMM_FULL_ROOFLINE",
    "MLA_BMM_HIGH",
    "MLA_BMM_LOW",
    "MLA_BMM_PRECISE",
    "MLA_BMM_STANDARD",
    "BmmFp8ReliabilityWarning",
    "BmmLatencyBreakdown",
    "BmmParameters",
    "MlaBmmLatencyBreakdown",
    "MlaBmmParameters",
    "bmm_latency_s",
    "bmm_latency_us",
    "bmm_work_terms",
    "estimate_bmm",
    "estimate_mla_bmm",
    "get_bmm_parameters",
    "get_mla_bmm_parameters",
    "mla_bmm_latency_s",
    "mla_bmm_latency_us",
    "normalize_dtype",
    "normalize_op",
    "work_terms",
]
