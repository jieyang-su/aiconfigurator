"""Provisional KernelSim models for DSA Index MQA and TopK kernels.

The recipes were calibrated only on one H100 SXM with SGLang 0.5.12. They are
archived for sparse-attention architecture studies, not as mature cross-GPU
kernel models. Latency inputs use FLOP/s, byte/s and Hz; outputs use microseconds.
"""

from __future__ import annotations

import math
import warnings as python_warnings
from dataclasses import dataclass

DSA_INDEX_MODEL_VERSION = "2026-08-06.aic-dsa-index-provisional-v1"
LIMITED_SCOPE_MESSAGE = (
    "The DSA Index MQA/TopK KernelSim model is provisional: it was calibrated "
    "from a temporary single-H100 SGLang 0.5.12 collection, covers only the "
    "recorded DeepGEMM FP8 MQA and FP32-score TopK semantics, and has not been "
    "validated across GPUs, backends, or sparse-attention architectures."
)


class DsaIndexModelWarning(UserWarning):
    """Warning emitted for the deliberately limited provisional model."""


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive_rate(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite value")
    return value


def _level(value: str) -> str:
    level = value.strip().lower()
    if level not in {"standard", "high", "low"}:
        raise ValueError("parameter_level must be standard, high, or low")
    return level


def _layout(value: str) -> str:
    layout = value.strip().lower()
    if layout not in {"ragged", "paged"}:
        raise ValueError("layout must be ragged or paged")
    return layout


@dataclass(frozen=True)
class IndexMqaShape:
    layout: str
    batch_size: int
    query_length: int
    context_length: int
    index_heads: int
    head_dim: int = 128
    page_size: int = 64
    score_chunk_slots: int = 8_000_000
    dtype: str = "fp8"

    def __post_init__(self) -> None:
        object.__setattr__(self, "layout", _layout(self.layout))
        for name in (
            "batch_size", "query_length", "context_length", "index_heads",
            "head_dim", "page_size", "score_chunk_slots",
        ):
            object.__setattr__(self, name, _positive_int(name, getattr(self, name)))
        if self.query_length > self.context_length:
            raise ValueError("query_length must not exceed context_length")
        dtype = self.dtype.strip().lower()
        if dtype not in {"fp8", "float8", "float8_e4m3fn"}:
            raise ValueError("Index MQA currently supports only the calibrated FP8 kernel")
        object.__setattr__(self, "dtype", "fp8")
        if self.layout == "paged" and self.query_length not in {1, 2}:
            raise ValueError("the calibrated SGLang 0.5.12 paged MQA kernel supports next_n=1 or 2")


@dataclass(frozen=True)
class IndexMqaParameters:
    eta_compute: float
    eta_memory: float
    paged_floor_us_h32: float
    paged_floor_us_h64: float
    ragged_floor_us_h32: float
    ragged_floor_us_h64: float
    ragged_control_cycles_h32: float
    ragged_control_cycles_h64: float


_MQA_PROFILES = {
    "standard": IndexMqaParameters(0.85, 0.95, 25.0, 37.0, 22.0, 27.0, 330.0, 245.0),
    "high": IndexMqaParameters(0.65, 0.75, 32.0, 46.0, 28.0, 34.0, 430.0, 320.0),
    "low": IndexMqaParameters(0.98, 1.00, 20.0, 29.0, 17.0, 21.0, 245.0, 180.0),
}


def get_index_mqa_parameters(level: str = "standard") -> IndexMqaParameters:
    return _MQA_PROFILES[_level(level)]


def _head_parameter(heads: int, h32: float, h64: float) -> tuple[float, bool]:
    if heads == 32:
        return h32, False
    if heads == 64:
        return h64, False
    # Formula-level interpolation/extrapolation is explicit in the result.
    return h32 + (heads - 32) * (h64 - h32) / 32.0, True


def _ragged_geometry(shape: IndexMqaShape, sm_count: int) -> tuple[int, int, float, int]:
    rows = shape.batch_size * shape.query_length
    total_k = shape.batch_size * shape.context_length
    chunk_rows = max(1, shape.score_chunk_slots // total_k)
    chunk_rows = max(16, chunk_rows // 16 * 16) if rows > 16 else rows
    chunk_rows = min(rows, chunk_rows)
    critical_blocks = 0.0
    for chunk_begin in range(0, rows, chunk_rows):
        chunk_end = min(rows, chunk_begin + chunk_rows)
        blocks: list[int] = []
        for block_begin in range(chunk_begin, chunk_end, 16):
            block_end = min(chunk_end, block_begin + 16)
            starts = []
            ends = []
            for row in range(block_begin, block_end):
                request = row // shape.query_length
                local_query = row % shape.query_length
                start = request * shape.context_length
                end = start + shape.context_length - shape.query_length + 1 + local_query
                starts.append(start)
                ends.append(end)
            blocks.append(math.ceil((max(ends) - min(starts)) / 64))
        critical_blocks += max(max(blocks), sum(blocks) / sm_count)
    valid_pairs = shape.batch_size * (
        shape.query_length * (shape.context_length - shape.query_length + 1)
        + shape.query_length * (shape.query_length - 1) // 2
    )
    return chunk_rows, math.ceil(rows / chunk_rows), critical_blocks, valid_pairs


@dataclass(frozen=True)
class IndexMqaEstimate:
    latency_us: float
    parameter_level: str
    layout: str
    scope: str
    warnings: tuple[str, ...]
    floor_us: float
    compute_us: float
    memory_us: float
    resource_us: float
    control_us: float
    roofline_branch: str
    valid_pairs: int
    score_slots: int
    chunk_rows: int
    chunk_count: int
    critical_kv_blocks: float
    flops: float
    modeled_bytes: float
    eta_compute: float
    eta_memory: float


def estimate_index_mqa(
    shape: IndexMqaShape,
    *,
    sm_count: int,
    clock_hz: float,
    fp8_peak_flops_s: float,
    hbm_bandwidth_bytes_s: float,
    parameter_level: str = "standard",
    params: IndexMqaParameters | None = None,
) -> IndexMqaEstimate:
    """Estimate the DeepGEMM-style FP8 index-score kernel boundary."""
    if not isinstance(shape, IndexMqaShape):
        raise TypeError("shape must be an IndexMqaShape")
    level = _level(parameter_level)
    if params is not None and level != "standard":
        raise ValueError("custom params cannot be combined with a non-standard parameter_level")
    params = params or get_index_mqa_parameters(level)
    level = "custom" if params is not _MQA_PROFILES.get(level) else level
    sm_count = _positive_int("sm_count", sm_count)
    clock = _positive_rate("clock_hz", clock_hz)
    peak = _positive_rate("fp8_peak_flops_s", fp8_peak_flops_s)
    bandwidth = _positive_rate("hbm_bandwidth_bytes_s", hbm_bandwidth_bytes_s)
    python_warnings.warn(LIMITED_SCOPE_MESSAGE, DsaIndexModelWarning, stacklevel=2)

    floor32 = params.ragged_floor_us_h32 if shape.layout == "ragged" else params.paged_floor_us_h32
    floor64 = params.ragged_floor_us_h64 if shape.layout == "ragged" else params.paged_floor_us_h64
    floor, extrapolated = _head_parameter(shape.index_heads, floor32, floor64)
    warning_messages = [LIMITED_SCOPE_MESSAGE]
    if extrapolated:
        warning_messages.append("index_heads is outside the calibrated 32/64-head recipes")

    if shape.layout == "ragged":
        chunk_rows, chunks, critical, valid_pairs = _ragged_geometry(shape, sm_count)
        flops = critical * 2.0 * 16 * 64 * shape.index_heads * shape.head_dim
        modeled_bytes = critical * (64 * (shape.head_dim + 4) + 16 * 64 * 4)
        score_slots = shape.batch_size * shape.query_length * shape.batch_size * shape.context_length
        control32, control64 = params.ragged_control_cycles_h32, params.ragged_control_cycles_h64
        control_cycles, _ = _head_parameter(shape.index_heads, control32, control64)
        control_us = critical * control_cycles / clock * 1e6
        floor_us = chunks * floor
    else:
        chunk_rows = shape.batch_size * shape.query_length
        chunks = 1
        critical = 0.0
        valid_pairs = shape.batch_size * (
            shape.query_length * shape.context_length
            + shape.query_length * (shape.query_length - 1) // 2
        )
        aligned = math.ceil((shape.context_length + shape.query_length - 1) / shape.page_size) * shape.page_size
        score_slots = shape.batch_size * shape.query_length * aligned
        flops = 2.0 * valid_pairs * shape.index_heads * shape.head_dim
        q_rows = shape.batch_size * shape.query_length
        modeled_bytes = (
            q_rows * shape.index_heads * shape.head_dim
            + q_rows * shape.index_heads * 4
            + shape.batch_size * shape.context_length * (shape.head_dim + 4)
            + score_slots * 4
        )
        control_us = 0.0
        floor_us = floor

    compute_us = flops / peak / _positive_rate("eta_compute", params.eta_compute) * 1e6
    memory_us = modeled_bytes / bandwidth / _positive_rate("eta_memory", params.eta_memory) * 1e6
    resource_us = max(compute_us, memory_us)
    branch = "compute" if compute_us >= memory_us else "memory"
    return IndexMqaEstimate(
        latency_us=floor_us + resource_us + control_us,
        parameter_level=level,
        layout=shape.layout,
        scope="single_h100_sglang_0.5.12_provisional",
        warnings=tuple(warning_messages),
        floor_us=floor_us,
        compute_us=compute_us,
        memory_us=memory_us,
        resource_us=resource_us,
        control_us=control_us,
        roofline_branch=branch,
        valid_pairs=valid_pairs,
        score_slots=score_slots,
        chunk_rows=chunk_rows,
        chunk_count=chunks,
        critical_kv_blocks=critical,
        flops=flops,
        modeled_bytes=modeled_bytes,
        eta_compute=params.eta_compute,
        eta_memory=params.eta_memory,
    )


@dataclass(frozen=True)
class IndexTopKShape:
    layout: str
    batch_size: int
    query_length: int
    context_length: int
    topk: int = 2048
    variant: str = "fused"
    score_distribution: str = "standard"

    def __post_init__(self) -> None:
        object.__setattr__(self, "layout", _layout(self.layout))
        for name in ("batch_size", "query_length", "context_length", "topk"):
            object.__setattr__(self, name, _positive_int(name, getattr(self, name)))
        if self.query_length > self.context_length:
            raise ValueError("query_length must not exceed context_length")
        variant = self.variant.strip().lower()
        if variant == "fused":
            variant = f"{self.layout}_fused"
        if variant not in {"plain", "paged_fused", "ragged_fused"}:
            raise ValueError("variant must be plain or fused for the selected layout")
        if variant != "plain" and not variant.startswith(self.layout):
            raise ValueError("fused variant must match layout")
        object.__setattr__(self, "variant", variant)
        distribution = self.score_distribution.strip().lower()
        if distribution not in {"standard", "flat", "top_last"}:
            raise ValueError("score_distribution must be standard, flat, or top_last")
        object.__setattr__(self, "score_distribution", distribution)


@dataclass(frozen=True)
class IndexTopKParameters:
    floor_scale: float
    pass_scale: float
    saturation_scale: float


_TOPK_PROFILES = {
    "standard": IndexTopKParameters(1.00, 1.00, 1.00),
    "high": IndexTopKParameters(1.25, 1.30, 1.20),
    "low": IndexTopKParameters(0.80, 0.75, 0.80),
}

# Rounded H100 extended-sweep centers: floor_us, effective passes, row saturation.
_TOPK_RECIPES = {
    ("paged", "plain", "flat"): (20.7, 3.33, 104.0),
    ("paged", "plain", "top_last"): (18.9, 1.90, 89.0),
    ("paged", "paged_fused", "flat"): (21.3, 3.36, 103.0),
    ("paged", "paged_fused", "top_last"): (19.4, 1.95, 87.0),
    ("ragged", "plain", "flat"): (24.8, 0.95, 83.0),
    ("ragged", "plain", "top_last"): (16.2, 0.49, 78.0),
    ("ragged", "ragged_fused", "flat"): (24.9, 0.96, 82.0),
    ("ragged", "ragged_fused", "top_last"): (16.3, 0.50, 77.0),
}


def get_index_topk_parameters(level: str = "standard") -> IndexTopKParameters:
    return _TOPK_PROFILES[_level(level)]


@dataclass(frozen=True)
class IndexTopKEstimate:
    latency_us: float
    parameter_level: str
    layout: str
    variant: str
    score_distribution: str
    scope: str
    warnings: tuple[str, ...]
    floor_us: float
    scan_us: float
    score_slots: int
    output_indices: int
    query_rows: int
    row_parallel_efficiency: float
    effective_pass_factor: float
    row_saturation: float


def estimate_index_topk(
    shape: IndexTopKShape,
    *,
    hbm_bandwidth_bytes_s: float,
    parameter_level: str = "standard",
    params: IndexTopKParameters | None = None,
) -> IndexTopKEstimate:
    """Estimate FP32-score TopK/index-transform latency."""
    if not isinstance(shape, IndexTopKShape):
        raise TypeError("shape must be an IndexTopKShape")
    level = _level(parameter_level)
    if params is not None and level != "standard":
        raise ValueError("custom params cannot be combined with a non-standard parameter_level")
    params = params or get_index_topk_parameters(level)
    level = "custom" if params is not _TOPK_PROFILES.get(level) else level
    bandwidth = _positive_rate("hbm_bandwidth_bytes_s", hbm_bandwidth_bytes_s)
    python_warnings.warn(LIMITED_SCOPE_MESSAGE, DsaIndexModelWarning, stacklevel=2)

    def recipe(distribution: str) -> tuple[float, float, float]:
        return _TOPK_RECIPES[(shape.layout, shape.variant, distribution)]

    if shape.score_distribution == "standard":
        flat = recipe("flat")
        top_last = recipe("top_last")
        base = tuple((a + b) / 2 for a, b in zip(flat, top_last, strict=True))
    else:
        base = recipe(shape.score_distribution)
    floor_us = base[0] * _positive_rate("floor_scale", params.floor_scale)
    passes = base[1] * _positive_rate("pass_scale", params.pass_scale)
    saturation = base[2] * _positive_rate("saturation_scale", params.saturation_scale)
    rows = shape.batch_size * shape.query_length
    width = shape.context_length if shape.layout == "paged" else shape.batch_size * shape.context_length
    score_slots = rows * width
    output_indices = rows * min(shape.topk, width)
    raw_scan_us = (score_slots * 4 + output_indices * 4) / bandwidth * 1e6
    row_efficiency = min(1.0, rows / saturation)
    scan_us = passes * raw_scan_us / row_efficiency
    return IndexTopKEstimate(
        latency_us=floor_us + scan_us,
        parameter_level=level,
        layout=shape.layout,
        variant=shape.variant,
        score_distribution=shape.score_distribution,
        scope="single_h100_sglang_0.5.12_provisional",
        warnings=(LIMITED_SCOPE_MESSAGE,),
        floor_us=floor_us,
        scan_us=scan_us,
        score_slots=score_slots,
        output_indices=output_indices,
        query_rows=rows,
        row_parallel_efficiency=row_efficiency,
        effective_pass_factor=passes,
        row_saturation=saturation,
    )


__all__ = [
    "DSA_INDEX_MODEL_VERSION",
    "LIMITED_SCOPE_MESSAGE",
    "DsaIndexModelWarning",
    "IndexMqaEstimate",
    "IndexMqaParameters",
    "IndexMqaShape",
    "IndexTopKEstimate",
    "IndexTopKParameters",
    "IndexTopKShape",
    "estimate_index_mqa",
    "estimate_index_topk",
    "get_index_mqa_parameters",
    "get_index_topk_parameters",
]
