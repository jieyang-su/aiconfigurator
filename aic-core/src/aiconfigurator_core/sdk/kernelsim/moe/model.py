#!/usr/bin/env python3
"""Hardware-independent Sum-3P model for balanced SGLang MoE latency.

This module is self-contained. Hardware rates use FLOP/s and byte/s; latency
outputs use microseconds. ``peak_flops_s`` must be the direct peak for the
selected recipe, never a value inferred from the BF16 peak.
"""

from __future__ import annotations

import math
import warnings as python_warnings
from dataclasses import dataclass

EMPIRICAL_MODEL_VERSION = "2026-07-30.aic-sglang-moe-full-traffic-sum3-v2"
W4A16_TRANSFER_MODEL_VERSION = "2026-08-12.w4a16-mxfp4-bf16-transfer-v1"
W8A16_TRANSFER_MODEL_VERSION = "2026-08-13.w8a16-int8wo-bf16-transfer-v1"
W4A16_TRANSFER_LIMITATION = (
    "W4A16 MXFP4 MoE uses a provisional BF16-Triton parameter transfer with only "
    "the logical weight and scale traffic adjusted. It is not a W4A16 kernel fit: "
    "unpack/dequantization, backend launch and tiling, small-token behavior, and "
    "EP>1 execution are uncalibrated; the Cutlass quant label is also only a "
    "temporary proxy through this SGLang model. Treat the estimate as low confidence."
)
W8A16_TRANSFER_LIMITATION = (
    "W8A16 INT8-WO MoE uses an uncalibrated BF16-Triton parameter transfer with "
    "only INT8 expert-weight and FP32 per-output-channel scale traffic adjusted. "
    "Fused dequantization, backend packing/tiling, small-token behavior, non-NVIDIA "
    "hardware, and EP>1 execution are uncalibrated. SGLang exposes the underlying "
    "use_int8_w8a16 helper, but the current top-level collector does not persist an "
    "int8_wo case. Treat the estimate as low confidence. This is an ANALYTICAL "
    "transfer proxy, not a Silicon calibration row, and must not be reported as "
    "measured W8A16 performance."
)


class W4A16TransferModelWarning(UserWarning):
    """Warning for the deliberately limited W4A16 transfer recipe."""


class W8A16TransferModelWarning(UserWarning):
    """Warning for the deliberately limited W8A16 transfer recipe."""


_w4a16_warning_emitted = False
_w8a16_warning_emitted = False


@dataclass(frozen=True)
class RecipeParameters:
    t_launch_us: float
    eta_compute: float
    eta_mem: float


@dataclass(frozen=True)
class MoeParameters:
    bf16_triton: RecipeParameters
    fp8_block_triton: RecipeParameters
    nvfp4_cutedsl: RecipeParameters


@dataclass(frozen=True)
class MoeLatencyBreakdown:
    model: str
    model_version: str
    recipe: str
    parameter_level: str
    scope: str
    required_peak_field: str
    collector_boundary: str
    latency_us: float
    launch_us: float
    body_us: float
    compute_us: float
    memory_us: float
    eta_compute: float
    eta_memory: float
    local_inter_size: float
    local_assignments: float
    local_experts: float
    active_experts: float
    flops: float
    logical_bytes: float
    routing_bytes: float
    input_quant_bytes: float
    gemm_io_bytes: float
    weight_value_bytes: float
    weight_scale_bytes: float
    activation_quant_bytes: float
    combine_bytes: float
    control_bytes: float


_RECIPE_METADATA = {
    "bf16_triton": {
        "required_peak_field": "bfloat16_tc_flops",
        "collector_boundary": "routing_and_triton_fused_moe",
    },
    "w8a16_int8wo_bf16_transfer": {
        "required_peak_field": "bfloat16_tc_flops",
        "collector_boundary": "routing_and_sglang_moe_w8a16_transfer_proxy",
    },
    "fp8_block_triton": {
        "required_peak_field": "fp8_tc_flops",
        "collector_boundary": "routing_and_triton_fused_moe_with_block_quant",
    },
    "w4a16_mxfp4_bf16_transfer": {
        "required_peak_field": "bfloat16_tc_flops",
        "collector_boundary": "routing_and_sglang_moe_w4a16_transfer_proxy",
    },
    "nvfp4_cutedsl": {
        "required_peak_field": "fp4_tc_flops",
        "collector_boundary": "predispatched_cutedsl_quant_gemm_act_quant_gemm",
    },
}

_RECIPE_PARAMETER_SOURCE = {
    "w8a16_int8wo_bf16_transfer": "bf16_triton",
    "w4a16_mxfp4_bf16_transfer": "bf16_triton",
}


MOE_PRECISE = MoeParameters(
    bf16_triton=RecipeParameters(49.20565631866641, 0.27423837764286596, 0.8254748004958514),
    fp8_block_triton=RecipeParameters(45.87778996859051, 0.7260468956413237, 0.7022557946361518),
    nvfp4_cutedsl=RecipeParameters(42.49834487842186, 0.5596081765288708, 0.7578316517757806),
)
MOE_STANDARD = MoeParameters(
    bf16_triton=RecipeParameters(49.0, 0.27, 0.83),
    fp8_block_triton=RecipeParameters(46.0, 0.73, 0.70),
    nvfp4_cutedsl=RecipeParameters(42.5, 0.56, 0.76),
)
MOE_HIGH = MoeParameters(
    bf16_triton=RecipeParameters(75.0, 0.17, 0.55),
    fp8_block_triton=RecipeParameters(75.0, 0.47, 0.43),
    nvfp4_cutedsl=RecipeParameters(75.0, 0.34, 0.45),
)
MOE_LOW = MoeParameters(
    bf16_triton=RecipeParameters(30.0, 0.42, 1.0),
    fp8_block_triton=RecipeParameters(25.0, 1.0, 1.0),
    nvfp4_cutedsl=RecipeParameters(20.0, 0.90, 1.0),
)

MOE_FULL_TRAFFIC_SUM3 = MOE_STANDARD

_PARAMETER_LEVELS = {
    "precise": MOE_PRECISE,
    "standard": MOE_STANDARD,
    "high": MOE_HIGH,
    "low": MOE_LOW,
}


def _positive_int(name: str, value: int) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _positive_rate(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite value")
    return result


def _ceil_div(value: float, divisor: int) -> int:
    return math.ceil(value / divisor)


def _nvfp4_scale_bytes(rows: float, cols: float) -> float:
    rounded_rows = _ceil_div(rows, 128) * 128
    scale_cols = _ceil_div(cols, 16)
    rounded_scale_cols = _ceil_div(scale_cols, 4) * 4
    return float(rounded_rows * rounded_scale_cols)


def _mxfp4_scale_bytes(rows: float, cols: float) -> float:
    """Logical one-byte E8M0 scale traffic for each 32-value MXFP4 block."""
    return float(rows * _ceil_div(cols, 32))


def _warn_w4a16_transfer_once() -> None:
    global _w4a16_warning_emitted
    if _w4a16_warning_emitted:
        return
    _w4a16_warning_emitted = True
    python_warnings.warn(W4A16_TRANSFER_LIMITATION, W4A16TransferModelWarning, stacklevel=3)


def _warn_w8a16_transfer_once() -> None:
    global _w8a16_warning_emitted
    if _w8a16_warning_emitted:
        return
    _w8a16_warning_emitted = True
    python_warnings.warn(W8A16_TRANSFER_LIMITATION, W8A16TransferModelWarning, stacklevel=3)


def required_peak_field(recipe: str) -> str:
    if recipe not in _RECIPE_METADATA:
        raise KeyError(f"unknown recipe {recipe!r}; choose from {sorted(_RECIPE_METADATA)}")
    return _RECIPE_METADATA[recipe]["required_peak_field"]


def get_moe_parameters(level: str = "standard") -> MoeParameters:
    normalized = level.strip().lower()
    if normalized not in _PARAMETER_LEVELS:
        raise ValueError("parameter_level must be precise, standard, high, or low")
    return _PARAMETER_LEVELS[normalized]


def _resolve_parameters(
    recipe: str,
    level: str,
    params: MoeParameters | None,
) -> tuple[str, RecipeParameters]:
    normalized = level.strip().lower()
    if params is not None:
        if normalized != "standard":
            raise ValueError("custom params cannot be combined with a non-standard parameter_level")
        selected = params
        normalized = "custom"
    else:
        selected = get_moe_parameters(normalized)
    parameter_source = _RECIPE_PARAMETER_SOURCE.get(recipe, recipe)
    return normalized, getattr(selected, parameter_source)


def work_terms(
    recipe: str,
    num_tokens: int,
    hidden_size: int,
    inter_size: int,
    topk: int,
    num_experts: int,
    moe_tp_size: int = 1,
    moe_ep_size: int = 1,
) -> dict[str, float | str]:
    """Return GEMM FLOPs and stable no-cache semantic tensor traffic.

    The byte total is a full-payment logical HBM demand. It includes every
    stable producer/consumer tensor boundary inside the collector timing scope,
    but excludes implementation-private padding, scratch, allocator traffic,
    and cache effects.
    """
    if recipe not in _RECIPE_METADATA:
        raise KeyError(f"unknown recipe {recipe!r}; choose from {sorted(_RECIPE_METADATA)}")
    tokens = _positive_int("num_tokens", num_tokens)
    hidden = _positive_int("hidden_size", hidden_size)
    intermediate = _positive_int("inter_size", inter_size)
    routed = _positive_int("topk", topk)
    experts = _positive_int("num_experts", num_experts)
    tp = _positive_int("moe_tp_size", moe_tp_size)
    ep = _positive_int("moe_ep_size", moe_ep_size)
    if intermediate % tp:
        raise ValueError("inter_size must be divisible by moe_tp_size")
    if experts % ep:
        raise ValueError("num_experts must be divisible by moe_ep_size")

    t = float(tokens)
    h = float(hidden)
    j = float(intermediate // tp)
    assignments = t * routed / ep
    local_experts = float(experts // ep)
    active = min(local_experts, assignments)
    terms: dict[str, float | str] = {
        "recipe": recipe,
        "collector_boundary": _RECIPE_METADATA[recipe]["collector_boundary"],
        "required_peak_field": required_peak_field(recipe),
        "local_inter_size": j,
        "local_assignments": assignments,
        "local_experts": local_experts,
        "active_experts": active,
        "flops_gemm1": 4.0 * assignments * h * j,
        "flops_gemm2": 2.0 * assignments * h * j,
    }

    if recipe in {
        "bf16_triton",
        "fp8_block_triton",
        "w8a16_int8wo_bf16_transfer",
        "w4a16_mxfp4_bf16_transfer",
    }:
        terms["bytes_routing_logits"] = 6.0 * t * experts
        terms["bytes_routing_topk"] = 32.0 * assignments
        terms["bytes_combine"] = 0.0 if routed == 1 else 2.0 * assignments * h + 2.0 * t * h

    if recipe == "bf16_triton":
        terms.update(
            {
                "bytes_input_quant": 0.0,
                "bytes_gemm1_input": 2.0 * assignments * h,
                "bytes_gemm1_weight_values": 4.0 * active * h * j,
                "bytes_gemm1_weight_scale": 0.0,
                "bytes_gemm1_output": 4.0 * assignments * j,
                "bytes_activation_quant": 6.0 * assignments * j,
                "bytes_gemm2_input": 2.0 * assignments * j,
                "bytes_gemm2_weight_values": 2.0 * active * h * j,
                "bytes_gemm2_weight_scale": 0.0,
                "bytes_gemm2_output": 2.0 * assignments * h,
                "bytes_control": 0.0,
            }
        )
    elif recipe == "w8a16_int8wo_bf16_transfer":
        # Match SGLang's use_int8_w8a16 helper: BF16 activations and outputs,
        # INT8 fused gate/up and down weights, and FP32 output-channel scales.
        # The collector allocates w1_scale as [E, 2 * fused_inter] and w2_scale
        # as [hidden, E]; only active rank-local experts are charged here.
        terms.update(
            {
                "bytes_input_quant": 0.0,
                "bytes_gemm1_input": 2.0 * assignments * h,
                "bytes_gemm1_weight_values": 2.0 * active * h * j,
                "bytes_gemm1_weight_scale": 16.0 * active * j,
                "bytes_gemm1_output": 4.0 * assignments * j,
                "bytes_activation_quant": 6.0 * assignments * j,
                "bytes_gemm2_input": 2.0 * assignments * j,
                "bytes_gemm2_weight_values": active * h * j,
                "bytes_gemm2_weight_scale": 4.0 * active * h,
                "bytes_gemm2_output": 2.0 * assignments * h,
                "bytes_control": 0.0,
            }
        )
    elif recipe == "w4a16_mxfp4_bf16_transfer":
        # Scheme 1 deliberately preserves the BF16 Triton execution model and
        # all BF16 activation/intermediate traffic. Only packed weight values
        # and their logical per-32-value MXFP4 scales differ from BF16.
        terms.update(
            {
                "bytes_input_quant": 0.0,
                "bytes_gemm1_input": 2.0 * assignments * h,
                "bytes_gemm1_weight_values": active * h * j,
                "bytes_gemm1_weight_scale": active * _mxfp4_scale_bytes(2.0 * j, h),
                "bytes_gemm1_output": 4.0 * assignments * j,
                "bytes_activation_quant": 6.0 * assignments * j,
                "bytes_gemm2_input": 2.0 * assignments * j,
                "bytes_gemm2_weight_values": 0.5 * active * h * j,
                "bytes_gemm2_weight_scale": active * _mxfp4_scale_bytes(h, j),
                "bytes_gemm2_output": 2.0 * assignments * h,
                "bytes_control": 0.0,
            }
        )
    elif recipe == "fp8_block_triton":
        h_blocks = _ceil_div(hidden, 128)
        j_blocks = _ceil_div(j, 128)
        terms.update(
            {
                "bytes_input_quant": 3.0 * t * h + 4.0 * t * h_blocks,
                "bytes_gemm1_input": assignments * h + 4.0 * assignments * h_blocks,
                "bytes_gemm1_weight_values": 2.0 * active * h * j,
                "bytes_gemm1_weight_scale": 4.0 * active * _ceil_div(2.0 * j, 128) * h_blocks,
                "bytes_gemm1_output": 4.0 * assignments * j,
                "bytes_activation_quant": 9.0 * assignments * j + 4.0 * assignments * j_blocks,
                "bytes_gemm2_input": assignments * j + 4.0 * assignments * j_blocks,
                "bytes_gemm2_weight_values": active * h * j,
                "bytes_gemm2_weight_scale": 4.0 * active * _ceil_div(hidden, 128) * j_blocks,
                "bytes_gemm2_output": 2.0 * assignments * h,
                "bytes_control": 0.0,
            }
        )
    else:
        h_scale_cols = _ceil_div(hidden, 16)
        j_scale_cols = _ceil_div(j, 16)
        terms.update(
            {
                "bytes_routing_logits": 0.0,
                "bytes_routing_topk": 0.0,
                "bytes_input_quant": 2.5 * assignments * h + assignments * h_scale_cols,
                "bytes_gemm1_input": 0.5 * assignments * h + assignments * h_scale_cols,
                "bytes_gemm1_weight_values": active * h * j,
                "bytes_gemm1_weight_scale": active * _nvfp4_scale_bytes(2.0 * j, h),
                "bytes_gemm1_output": 4.0 * assignments * j,
                "bytes_activation_quant": 4.5 * assignments * j + assignments * j_scale_cols,
                "bytes_gemm2_input": 0.5 * assignments * j + assignments * j_scale_cols,
                "bytes_gemm2_weight_values": 0.5 * active * h * j,
                "bytes_gemm2_weight_scale": active * _nvfp4_scale_bytes(h, j),
                "bytes_gemm2_output": 2.0 * assignments * h,
                "bytes_combine": 0.0,
                "bytes_control": 16.0 * local_experts + 16.0 * active,
            }
        )

    terms["flops_total"] = float(terms["flops_gemm1"]) + float(terms["flops_gemm2"])
    terms["bytes_total"] = sum(float(value) for key, value in terms.items() if key.startswith("bytes_"))
    return terms


def estimate_sglang_moe(
    recipe: str,
    num_tokens: int,
    hidden_size: int,
    inter_size: int,
    topk: int,
    num_experts: int,
    moe_tp_size: int,
    moe_ep_size: int,
    peak_flops_s: float,
    mem_bandwidth_bytes_s: float,
    params: MoeParameters | None = None,
    *,
    parameter_level: str = "standard",
) -> MoeLatencyBreakdown:
    """Estimate balanced rank-local MoE latency with the Sum-3P model."""
    if recipe not in _RECIPE_METADATA:
        raise KeyError(f"unknown recipe {recipe!r}; choose from {sorted(_RECIPE_METADATA)}")
    peak = _positive_rate("peak_flops_s", peak_flops_s)
    bandwidth = _positive_rate("mem_bandwidth_bytes_s", mem_bandwidth_bytes_s)
    if recipe == "w4a16_mxfp4_bf16_transfer":
        _warn_w4a16_transfer_once()
    elif recipe == "w8a16_int8wo_bf16_transfer":
        _warn_w8a16_transfer_once()
    level, selected = _resolve_parameters(recipe, parameter_level, params)
    terms = work_terms(
        recipe,
        num_tokens,
        hidden_size,
        inter_size,
        topk,
        num_experts,
        moe_tp_size,
        moe_ep_size,
    )
    eta_compute = _positive_rate("eta_compute", selected.eta_compute)
    eta_memory = _positive_rate("eta_mem", selected.eta_mem)
    launch_us = _positive_rate("t_launch_us", selected.t_launch_us)
    compute_us = float(terms["flops_total"]) / peak / eta_compute * 1e6
    memory_us = float(terms["bytes_total"]) / bandwidth / eta_memory * 1e6
    body_us = compute_us + memory_us

    routing_bytes = float(terms["bytes_routing_logits"]) + float(terms["bytes_routing_topk"])
    gemm_io_bytes = sum(
        float(terms[key])
        for key in (
            "bytes_gemm1_input",
            "bytes_gemm1_output",
            "bytes_gemm2_input",
            "bytes_gemm2_output",
        )
    )
    weight_value_bytes = float(terms["bytes_gemm1_weight_values"]) + float(terms["bytes_gemm2_weight_values"])
    weight_scale_bytes = float(terms["bytes_gemm1_weight_scale"]) + float(terms["bytes_gemm2_weight_scale"])
    if recipe == "w4a16_mxfp4_bf16_transfer":
        # Preserve the established W4A16 scope label for downstream reports.
        scope = "provisional BF16 parameter transfer; EP=1 assumption"
        if moe_ep_size != 1:
            scope += "; ideal uniform EP extrapolation, uncalibrated"
    elif recipe == "w8a16_int8wo_bf16_transfer":
        scope = "provisional w8a16_int8wo BF16 parameter transfer; EP=1 assumption"
        if moe_ep_size != 1:
            scope += "; ideal uniform EP extrapolation, uncalibrated"
    else:
        scope = "measured EP=1" if moe_ep_size == 1 else "ideal uniform EP extrapolation"
    return MoeLatencyBreakdown(
        model="sglang_moe_aggregate_sum3_full_semantic_traffic",
        model_version=(
            W4A16_TRANSFER_MODEL_VERSION
            if recipe == "w4a16_mxfp4_bf16_transfer"
            else W8A16_TRANSFER_MODEL_VERSION
            if recipe == "w8a16_int8wo_bf16_transfer"
            else EMPIRICAL_MODEL_VERSION
        ),
        recipe=recipe,
        parameter_level=level,
        scope=scope,
        required_peak_field=required_peak_field(recipe),
        collector_boundary=str(terms["collector_boundary"]),
        latency_us=launch_us + body_us,
        launch_us=launch_us,
        body_us=body_us,
        compute_us=compute_us,
        memory_us=memory_us,
        eta_compute=eta_compute,
        eta_memory=eta_memory,
        local_inter_size=float(terms["local_inter_size"]),
        local_assignments=float(terms["local_assignments"]),
        local_experts=float(terms["local_experts"]),
        active_experts=float(terms["active_experts"]),
        flops=float(terms["flops_total"]),
        logical_bytes=float(terms["bytes_total"]),
        routing_bytes=routing_bytes,
        input_quant_bytes=float(terms["bytes_input_quant"]),
        gemm_io_bytes=gemm_io_bytes,
        weight_value_bytes=weight_value_bytes,
        weight_scale_bytes=weight_scale_bytes,
        activation_quant_bytes=float(terms["bytes_activation_quant"]),
        combine_bytes=float(terms["bytes_combine"]),
        control_bytes=float(terms["bytes_control"]),
    )


def sglang_moe_latency_us(*args, **kwargs) -> float:
    return estimate_sglang_moe(*args, **kwargs).latency_us


def sglang_moe_latency_s(*args, **kwargs) -> float:
    return sglang_moe_latency_us(*args, **kwargs) * 1e-6


__all__ = [
    "EMPIRICAL_MODEL_VERSION",
    "MOE_FULL_TRAFFIC_SUM3",
    "MOE_HIGH",
    "MOE_LOW",
    "MOE_PRECISE",
    "MOE_STANDARD",
    "W4A16_TRANSFER_LIMITATION",
    "W4A16_TRANSFER_MODEL_VERSION",
    "W8A16_TRANSFER_LIMITATION",
    "W8A16_TRANSFER_MODEL_VERSION",
    "MoeLatencyBreakdown",
    "MoeParameters",
    "RecipeParameters",
    "W4A16TransferModelWarning",
    "W8A16TransferModelWarning",
    "estimate_sglang_moe",
    "get_moe_parameters",
    "required_peak_field",
    "sglang_moe_latency_s",
    "sglang_moe_latency_us",
    "work_terms",
]
