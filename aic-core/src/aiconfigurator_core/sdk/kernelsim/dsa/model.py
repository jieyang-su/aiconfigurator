"""KernelSim models for DSA Index MQA and TopK/index-transform kernels.

The production recipes in this module are graph-replay fits from one
H100 SXM running SGLang 0.5.12.  Index MQA is modeled as a task-service kernel;
production-natural TopK uses the v3 short/long query-wave model, while the
synthetic flat/top-last diagnostics retain the v2 kernel-regime formula.

This is an engineering model for sparse-attention architecture studies, not a
cross-GPU silicon model.  Every estimate carries the calibration limitation in
its warnings field and emits :class:`DsaIndexModelWarning`.
"""

from __future__ import annotations

import math
import warnings as python_warnings
from dataclasses import dataclass

DSA_INDEX_MODEL_VERSION = "2026-08-13.aic-dsa-index-v4"
CALIBRATED_TOPK = 2048
CALIBRATED_HEADS = (32, 64)
CALIBRATED_HEAD_DIM = 128
CALIBRATED_PAGE_SIZE = 64
CALIBRATED_SCORE_CHUNK_SLOTS = 8_000_000
CALIBRATED_TAIL_THRESHOLD = 32_768
CALIBRATED_ROW_SATURATION = 166.11336514650998
CALIBRATED_NATURAL_MAX_CONTEXT = 524_288
CALIBRATED_NATURAL_MAX_ROWS = 8_192

LIMITED_SCOPE_MESSAGE = (
    "The DSA Index MQA/TopK KernelSim model is provisional: it was calibrated "
    "from CUDA-graph-replay measurements on one H100 SXM with SGLang 0.5.12. "
    "Index MQA covers the recorded FP8 DeepGEMM-style paged/ragged recipes; "
    "TopK natural covers the FP32-score fused transform through 524K context "
    "and 8192 query rows; flat/top-last remain v2 diagnostics. It is not validated "
    "across GPUs, backends, dtypes, layouts, or sparse-attention variants."
)

BF16_PROXY_MESSAGE = (
    "BF16 DSA Index MQA is an uncalibrated proxy, not a measured SGLang/DeepGEMM "
    "kernel. It preserves the H100 FP8 launch/task-service fit and scales only "
    "the task term by the BF16-to-FP8 theoretical resource ratio. Backend, "
    "scheduler, tile, and cross-GPU transfer effects are not modeled."
)


class DsaIndexModelWarning(UserWarning):
    """Warning emitted for the deliberately limited DSA model."""


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
    head_dim: int = CALIBRATED_HEAD_DIM
    page_size: int = CALIBRATED_PAGE_SIZE
    score_chunk_slots: int = CALIBRATED_SCORE_CHUNK_SLOTS
    dtype: str = "fp8"

    def __post_init__(self) -> None:
        object.__setattr__(self, "layout", _layout(self.layout))
        for name in (
            "batch_size",
            "query_length",
            "context_length",
            "index_heads",
            "head_dim",
            "page_size",
            "score_chunk_slots",
        ):
            object.__setattr__(self, name, _positive_int(name, getattr(self, name)))
        if self.query_length > self.context_length:
            raise ValueError("query_length must not exceed context_length")
        dtype = self.dtype.strip().lower()
        if dtype in {"fp8", "float8", "float8_e4m3fn"}:
            dtype = "fp8"
        elif dtype in {"bf16", "bfloat16"}:
            dtype = "bf16"
        else:
            raise ValueError("Index MQA supports the calibrated FP8 kernel or the uncalibrated BF16 proxy")
        object.__setattr__(self, "dtype", dtype)
        if self.layout == "paged" and self.query_length not in {1, 2}:
            raise ValueError("the calibrated SGLang 0.5.12 paged MQA kernel supports next_n=1 or 2")


@dataclass(frozen=True)
class IndexMqaParameters:
    """Engineering parameters for the task-service MQA recipe.

    ``task_cycles_*`` are equivalent service-cycle coefficients for the
    measured scheduler atom.  They are not a CUDA thread-cycle count and are
    not portable hardware constants.
    """

    floor_paged_us: float
    floor_ragged_us: float
    task_cycles_paged: float
    task_cycles_ragged: float
    latency_scale: float = 1.0


_MQA_STANDARD = IndexMqaParameters(
    floor_paged_us=3.6963528878913916,
    floor_ragged_us=4.9474693848765945,
    task_cycles_paged=3.752455488695439,
    task_cycles_ragged=414.75532225389344,
)
_MQA_PROFILES = {
    "standard": _MQA_STANDARD,
    # Engineering envelopes, not confidence intervals.  The scale is kept
    # explicit so callers can distinguish conservative planning from a refit.
    "high": IndexMqaParameters(
        _MQA_STANDARD.floor_paged_us,
        _MQA_STANDARD.floor_ragged_us,
        _MQA_STANDARD.task_cycles_paged,
        _MQA_STANDARD.task_cycles_ragged,
        latency_scale=1.20,
    ),
    "low": IndexMqaParameters(
        _MQA_STANDARD.floor_paged_us,
        _MQA_STANDARD.floor_ragged_us,
        _MQA_STANDARD.task_cycles_paged,
        _MQA_STANDARD.task_cycles_ragged,
        latency_scale=0.80,
    ),
}


def get_index_mqa_parameters(level: str = "standard") -> IndexMqaParameters:
    return _MQA_PROFILES[_level(level)]


def _head_interpolate(heads: int, h32: float, h64: float) -> tuple[float, bool]:
    if heads in CALIBRATED_HEADS:
        return (h32 if heads == 32 else h64), False
    return h32 + (heads - 32) * (h64 - h32) / 32.0, True


def _scope_warnings(*, heads: int, head_dim: int, page_size: int, chunk_slots: int) -> list[str]:
    messages = [LIMITED_SCOPE_MESSAGE]
    if heads not in CALIBRATED_HEADS:
        messages.append("index_heads is outside the calibrated 32/64-head recipes")
    if head_dim != CALIBRATED_HEAD_DIM:
        messages.append("index head_dim differs from the calibrated 128-dim recipe")
    if page_size != CALIBRATED_PAGE_SIZE:
        messages.append("page_size differs from the calibrated 64-token recipe")
    if chunk_slots != CALIBRATED_SCORE_CHUNK_SLOTS:
        messages.append("score chunk size differs from the calibrated 8M-slot recipe")
    return messages


def _ragged_service_atoms(shape: IndexMqaShape, sm_count: int) -> tuple[int, int, float, int]:
    """Return chunk rows, chunk count, critical service atoms and valid pairs."""
    rows = shape.batch_size * shape.query_length
    total_k = shape.batch_size * shape.context_length
    chunk_rows = max(1, shape.score_chunk_slots // total_k)
    # Match the collector exactly: a workload of <= one Q tile is kept as one
    # tile, even when the score-slot heuristic would produce a smaller number.
    chunk_rows = rows if rows <= 16 else max(16, (chunk_rows // 16) * 16)
    chunk_rows = min(rows, chunk_rows)
    chunk_count = math.ceil(rows / chunk_rows)
    critical_atoms = 0.0
    block_size = shape.page_size
    for chunk_begin in range(0, rows, chunk_rows):
        chunk_end = min(rows, chunk_begin + chunk_rows)
        blocks: list[float] = []
        for tile_begin in range(chunk_begin, chunk_end, 16):
            tile_end = min(chunk_end, tile_begin + 16)
            starts: list[int] = []
            ends: list[int] = []
            for row in range(tile_begin, tile_end):
                request = row // shape.query_length
                local_query = row % shape.query_length
                start = request * shape.context_length
                end = start + shape.context_length - shape.query_length + 1 + local_query
                starts.append(start)
                ends.append(end)
            span_blocks = math.ceil((max(ends) - min(starts)) / block_size)
            # Cross-request ragged tiles are not equivalent to one contiguous
            # request.  This cap is the H100 scheduler abstraction identified
            # by the graph-replay data, not a universal architectural constant.
            request_cap = max(1.0, 128.0 / shape.index_heads)
            one_request_blocks = math.ceil(shape.context_length / block_size)
            blocks.append(min(float(span_blocks), one_request_blocks * request_cap))
        critical_atoms += max(max(blocks), sum(blocks) / sm_count)
    valid_pairs = shape.batch_size * (
        shape.query_length * (shape.context_length - shape.query_length + 1)
        + shape.query_length * (shape.query_length - 1) // 2
    )
    return chunk_rows, chunk_count, critical_atoms, valid_pairs


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
    eta_compute: float | None = None
    eta_memory: float | None = None
    service_atoms: float = 0.0
    reference_compute_us: float = 0.0
    reference_memory_us: float = 0.0
    baseline_control_us: float = 0.0
    resource_scale: float = 1.0


def estimate_index_mqa(
    shape: IndexMqaShape,
    *,
    sm_count: int,
    clock_hz: float,
    fp8_peak_flops_s: float | None,
    hbm_bandwidth_bytes_s: float,
    bf16_peak_flops_s: float | None = None,
    parameter_level: str = "standard",
    params: IndexMqaParameters | None = None,
) -> IndexMqaEstimate:
    """Estimate the FP8 index-score kernel or an explicit BF16 proxy.

    The calibrated FP8 path remains task-service rather than roofline. The
    BF16 proxy preserves its launch floor and scales only its task term by the
    ratio between BF16 and FP8 theoretical resource lower bounds.
    """
    if not isinstance(shape, IndexMqaShape):
        raise TypeError("shape must be an IndexMqaShape")
    level = _level(parameter_level)
    if params is not None and level != "standard":
        raise ValueError("custom params cannot be combined with a non-standard parameter_level")
    params = params or get_index_mqa_parameters(level)
    if params is not _MQA_PROFILES.get(level):
        level = "custom"
    sm_count = _positive_int("sm_count", sm_count)
    clock = _positive_rate("clock_hz", clock_hz)
    bandwidth = _positive_rate("hbm_bandwidth_bytes_s", hbm_bandwidth_bytes_s)
    warning_messages = _scope_warnings(
        heads=shape.index_heads,
        head_dim=shape.head_dim,
        page_size=shape.page_size,
        chunk_slots=shape.score_chunk_slots,
    )
    python_warnings.warn(LIMITED_SCOPE_MESSAGE, DsaIndexModelWarning, stacklevel=2)
    if shape.dtype == "fp8":
        peak = _positive_rate("fp8_peak_flops_s", fp8_peak_flops_s)
        bf16_peak = None
    else:
        if bf16_peak_flops_s is None:
            raise ValueError("bf16_peak_flops_s is required for the BF16 Index MQA proxy")
        bf16_peak = _positive_rate("bf16_peak_flops_s", bf16_peak_flops_s)
        if fp8_peak_flops_s is None:
            peak = 2.0 * bf16_peak
            warning_messages.append(
                "hardware has no FP8 peak; the BF16 proxy assumes a counterfactual "
                "FP8 reference peak equal to 2x BF16 peak"
            )
        else:
            peak = _positive_rate("fp8_peak_flops_s", fp8_peak_flops_s)
        warning_messages.append(BF16_PROXY_MESSAGE)
        python_warnings.warn(BF16_PROXY_MESSAGE, DsaIndexModelWarning, stacklevel=2)

    paged = shape.layout == "paged"
    if paged:
        chunk_rows, chunks = shape.batch_size * shape.query_length, 1
        service_atoms = shape.batch_size * math.ceil(shape.context_length / shape.page_size)
        critical_blocks = 0.0
        valid_pairs = shape.batch_size * (
            shape.query_length * shape.context_length + shape.query_length * (shape.query_length - 1) // 2
        )
        aligned = math.ceil((shape.context_length + shape.query_length - 1) / shape.page_size) * shape.page_size
        score_slots = shape.batch_size * shape.query_length * aligned
        floor = params.floor_paged_us
        cycles = params.task_cycles_paged
    else:
        chunk_rows, chunks, service_atoms, valid_pairs = _ragged_service_atoms(shape, sm_count)
        critical_blocks = service_atoms
        score_slots = shape.batch_size * shape.query_length * shape.batch_size * shape.context_length
        floor = params.floor_ragged_us
        cycles = params.task_cycles_ragged

    # Diagnostic work uses logical/critical tile geometry.  It is intentionally
    # excluded from the selected prediction because the candidate comparison
    # showed roofline terms had weak identification for this kernel.
    flops = float(valid_pairs * 2 * shape.index_heads * shape.head_dim)
    fp8_modeled_bytes = float(
        shape.batch_size * shape.query_length * shape.index_heads * shape.head_dim
        + shape.batch_size * shape.query_length * shape.index_heads * 4
        + shape.batch_size * shape.context_length * (shape.head_dim + 4)
        + (score_slots * 4)
    )
    bf16_modeled_bytes = float(
        shape.batch_size * shape.query_length * shape.index_heads * shape.head_dim * 2
        + shape.batch_size * shape.query_length * shape.index_heads * 4
        + shape.batch_size * shape.context_length * shape.head_dim * 2
        + (score_slots * 4)
    )
    reference_compute_us = flops / peak * 1e6
    reference_memory_us = fp8_modeled_bytes / bandwidth * 1e6
    if shape.dtype == "bf16":
        assert bf16_peak is not None
        compute_us = flops / bf16_peak * 1e6
        memory_us = bf16_modeled_bytes / bandwidth * 1e6
        reference_resource_us = max(reference_compute_us, reference_memory_us)
        resource_scale = max(1.0, max(compute_us, memory_us) / reference_resource_us)
        modeled_bytes = bf16_modeled_bytes
    else:
        compute_us = reference_compute_us
        memory_us = reference_memory_us
        resource_scale = 1.0
        modeled_bytes = fp8_modeled_bytes
    baseline_task_us = service_atoms * cycles / clock * 1e6
    floor_us = floor * (chunks if not paged else 1) * params.latency_scale
    baseline_task_us *= params.latency_scale
    task_us = baseline_task_us * resource_scale
    if shape.dtype == "bf16":
        warning_messages.append(
            "FLOPs/bytes determine only the BF16-to-FP8 task scaling ratio; "
            "the launch floor and task-service geometry remain the FP8 calibration"
        )
    else:
        warning_messages.append("FLOPs/bytes are diagnostic lower bounds; task-service terms determine latency")
    return IndexMqaEstimate(
        latency_us=floor_us + task_us,
        parameter_level=level,
        layout=shape.layout,
        scope=(
            "uncalibrated_bf16_resource_scaled_proxy"
            if shape.dtype == "bf16"
            else "single_h100_sglang_0.5.12_graph_replay_v2"
        ),
        warnings=tuple(warning_messages),
        floor_us=floor_us,
        compute_us=compute_us,
        memory_us=memory_us,
        resource_us=0.0,
        control_us=task_us,
        roofline_branch=("task_service_bf16_proxy" if shape.dtype == "bf16" else "task_service_layout"),
        valid_pairs=valid_pairs,
        score_slots=score_slots,
        chunk_rows=chunk_rows,
        chunk_count=chunks,
        critical_kv_blocks=critical_blocks,
        flops=flops,
        modeled_bytes=modeled_bytes,
        service_atoms=service_atoms,
        reference_compute_us=reference_compute_us,
        reference_memory_us=reference_memory_us,
        baseline_control_us=baseline_task_us,
        resource_scale=resource_scale,
    )


@dataclass(frozen=True)
class IndexTopKShape:
    layout: str
    batch_size: int
    query_length: int
    context_length: int
    topk: int = CALIBRATED_TOPK
    variant: str = "fused"
    score_distribution: str = "standard"

    def __post_init__(self) -> None:
        object.__setattr__(self, "layout", _layout(self.layout))
        for name in ("batch_size", "query_length", "context_length", "topk"):
            object.__setattr__(self, name, _positive_int(name, getattr(self, name)))
        variant = self.variant.strip().lower()
        if variant == "fused":
            variant = f"{self.layout}_fused"
        if variant not in {"plain", "paged_fused", "ragged_fused"}:
            raise ValueError("variant must be plain or fused for the selected layout")
        if variant != "plain" and not variant.startswith(self.layout):
            raise ValueError("fused variant must match layout")
        object.__setattr__(self, "variant", variant)
        distribution = self.score_distribution.strip().lower()
        if distribution not in {"standard", "natural", "flat", "top_last"}:
            raise ValueError("score_distribution must be standard, natural, flat, or top_last")
        object.__setattr__(self, "score_distribution", distribution)


@dataclass(frozen=True)
class IndexTopKParameters:
    short_floor_paged_us: float
    short_floor_ragged_us: float
    long_floor_flat_us: float
    long_floor_top_last_us: float
    long_floor_natural_us: float
    log_flat_ms: float
    log_top_last_ms: float
    log_natural_ms: float
    tail_flat_ms: float
    tail_top_last_ms: float
    tail_natural_ms: float
    row_flat_ms: float
    row_top_last_ms: float
    row_saturation: float
    natural_short_floor_paged_us: float
    natural_short_floor_ragged_us: float
    natural_long_floor_paged_us: float
    natural_long_floor_ragged_us: float
    natural_short_inverse_efficiency_paged: float
    natural_short_inverse_efficiency_ragged: float
    natural_long_inverse_efficiency_paged: float
    natural_long_inverse_efficiency_ragged: float
    natural_tail_inverse_efficiency_paged: float
    natural_tail_inverse_efficiency_ragged: float
    natural_short_query_tile: int = 256
    natural_long_query_tile: int = 128
    natural_tail_start_waves: int = 4
    latency_scale: float = 1.0
    tail_threshold: int = CALIBRATED_TAIL_THRESHOLD


_TOPK_STANDARD = IndexTopKParameters(
    short_floor_paged_us=2.630201842240251,
    short_floor_ragged_us=4.027809283865894,
    long_floor_flat_us=6.322895955808065,
    long_floor_top_last_us=1.9900552171303654,
    long_floor_natural_us=4.327762776306002,
    log_flat_ms=0.002132959183954102,
    log_top_last_ms=0.0016183315932891823,
    log_natural_ms=0.0020487719364600423,
    tail_flat_ms=0.017215360559564713,
    tail_top_last_ms=0.0052298997198841875,
    tail_natural_ms=0.013992504314375502,
    row_flat_ms=0.009409707852940298,
    row_top_last_ms=0.005859134754325899,
    row_saturation=CALIBRATED_ROW_SATURATION,
    natural_short_floor_paged_us=3.223162275792814,
    natural_short_floor_ragged_us=2.1849641735464873,
    natural_long_floor_paged_us=6.885035995748586,
    natural_long_floor_ragged_us=7.00554041693735,
    natural_short_inverse_efficiency_paged=2.0177496158220974e-06,
    natural_short_inverse_efficiency_ragged=1.2788940003724325,
    natural_long_inverse_efficiency_paged=2.404003100179183,
    natural_long_inverse_efficiency_ragged=2.9601479014257794,
    natural_tail_inverse_efficiency_paged=0.0,
    natural_tail_inverse_efficiency_ragged=2.620612297666048,
)
_TOPK_PROFILES = {
    "standard": _TOPK_STANDARD,
    "high": IndexTopKParameters(
        **{
            **_TOPK_STANDARD.__dict__,
            "latency_scale": 1.35,
        }
    ),
    "low": IndexTopKParameters(
        **{
            **_TOPK_STANDARD.__dict__,
            "latency_scale": 0.75,
        }
    ),
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
    long_path: bool = False
    log_us: float = 0.0
    tail_us: float = 0.0
    row_penalty_us: float = 0.0
    threshold: int = CALIBRATED_TOPK
    tail_threshold: int = CALIBRATED_TAIL_THRESHOLD
    query_tile: int = 0
    waves: int = 0
    tail_waves: int = 0
    executed_score_bytes: float = 0.0
    model_recipe: str = "v2_kernel_regime"


def estimate_index_topk(
    shape: IndexTopKShape,
    *,
    hbm_bandwidth_bytes_s: float,
    parameter_level: str = "standard",
    params: IndexTopKParameters | None = None,
) -> IndexTopKEstimate:
    """Estimate the fused FP32-score TopK/index-transform kernel.

    HBM bandwidth is validated and used for diagnostic byte bounds only.  The
    selected v2 latency model is dominated by the measured kernel regime,
    because the bytes-only candidate had 53.45% OOF MAPE.
    """
    if not isinstance(shape, IndexTopKShape):
        raise TypeError("shape must be an IndexTopKShape")
    level = _level(parameter_level)
    if params is not None and level != "standard":
        raise ValueError("custom params cannot be combined with a non-standard parameter_level")
    params = params or get_index_topk_parameters(level)
    if params is not _TOPK_PROFILES.get(level):
        level = "custom"
    bandwidth = _positive_rate("hbm_bandwidth_bytes_s", hbm_bandwidth_bytes_s)
    python_warnings.warn(LIMITED_SCOPE_MESSAGE, DsaIndexModelWarning, stacklevel=2)
    warnings = [LIMITED_SCOPE_MESSAGE]
    if shape.topk != CALIBRATED_TOPK:
        warnings.append(
            f"topk={shape.topk} differs from calibrated K={CALIBRATED_TOPK}; "
            "reuse of fitted coefficients is low-confidence and needs refit"
        )
    if shape.variant == "plain":
        warnings.append("plain TopK variant was not the production fused calibration boundary")
    if shape.score_distribution == "top_last":
        warnings.append("top_last score distribution has high validation error and is diagnostic only")
    distribution = "natural" if shape.score_distribution == "standard" else shape.score_distribution
    rows = shape.batch_size * shape.query_length
    width = shape.context_length if shape.layout == "paged" else shape.batch_size * shape.context_length
    score_slots = rows * width
    output_indices = rows * min(shape.topk, width)
    short = shape.context_length <= shape.topk

    if distribution == "natural":
        if shape.context_length > CALIBRATED_NATURAL_MAX_CONTEXT:
            warnings.append(f"natural TopK context exceeds the calibrated {CALIBRATED_NATURAL_MAX_CONTEXT}-token range")
        if rows > CALIBRATED_NATURAL_MAX_ROWS:
            warnings.append(f"natural TopK query rows exceed the calibrated {CALIBRATED_NATURAL_MAX_ROWS}-row range")
        if shape.layout == "paged" and shape.query_length != 1:
            warnings.append("natural paged TopK was calibrated only for the production Q=1 recipe")

        branch = "short" if short else "long"
        tile = params.natural_short_query_tile if short else params.natural_long_query_tile
        waves = math.ceil(rows / tile)
        tail_waves = 0 if short else max(0, waves - params.natural_tail_start_waves)
        floor_us = getattr(params, f"natural_{branch}_floor_{shape.layout}_us")
        inverse_efficiency = getattr(params, f"natural_{branch}_inverse_efficiency_{shape.layout}")
        executed_score_bytes = float(waves * tile * shape.context_length * 4)
        main_scan_us = executed_score_bytes / bandwidth * inverse_efficiency * 1e6
        tail_inverse_efficiency = getattr(params, f"natural_tail_inverse_efficiency_{shape.layout}")
        tail_score_bytes = float(tail_waves * tile * shape.context_length * 4)
        tail_us = tail_score_bytes / bandwidth * tail_inverse_efficiency * 1e6
        scale = params.latency_scale
        floor_us *= scale
        main_scan_us *= scale
        tail_us *= scale
        scan_us = main_scan_us + tail_us
        raw_scan_us = (score_slots * 4 + output_indices * 4) / bandwidth * 1e6
        return IndexTopKEstimate(
            latency_us=floor_us + scan_us,
            parameter_level=level,
            layout=shape.layout,
            variant=shape.variant,
            score_distribution=shape.score_distribution,
            scope="single_h100_sglang_0.5.12_graph_replay_natural_v3",
            warnings=tuple(warnings),
            floor_us=floor_us,
            scan_us=scan_us,
            score_slots=score_slots,
            output_indices=output_indices,
            query_rows=rows,
            row_parallel_efficiency=min(1.0, rows / tile),
            effective_pass_factor=(scan_us / raw_scan_us if raw_scan_us > 0 else 0.0),
            row_saturation=float(tile),
            long_path=not short,
            tail_us=tail_us,
            threshold=shape.topk,
            tail_threshold=params.natural_tail_start_waves,
            query_tile=tile,
            waves=waves,
            tail_waves=tail_waves,
            executed_score_bytes=executed_score_bytes,
            model_recipe="natural_wave_v3",
        )

    if shape.context_length > 65_536:
        warnings.append("diagnostic distribution context exceeds the calibrated 65536-token range")
    short_floor = params.short_floor_paged_us if shape.layout == "paged" else params.short_floor_ragged_us
    if short:
        floor_us = short_floor * params.latency_scale
        scan_us = 0.0
        row_efficiency = min(1.0, rows / params.row_saturation)
        return IndexTopKEstimate(
            latency_us=floor_us,
            parameter_level=level,
            layout=shape.layout,
            variant=shape.variant,
            score_distribution=shape.score_distribution,
            scope="single_h100_sglang_0.5.12_graph_replay_v2",
            warnings=tuple(warnings),
            floor_us=floor_us,
            scan_us=scan_us,
            score_slots=score_slots,
            output_indices=output_indices,
            query_rows=rows,
            row_parallel_efficiency=row_efficiency,
            effective_pass_factor=0.0,
            row_saturation=params.row_saturation,
            threshold=shape.topk,
            tail_threshold=params.tail_threshold,
        )

    prefix = {
        "flat": "flat",
        "top_last": "top_last",
        "natural": "natural",
    }[distribution]
    long_floor = getattr(params, f"long_floor_{prefix}_us")
    log_coeff = getattr(params, f"log_{prefix}_ms")
    tail_coeff = getattr(params, f"tail_{prefix}_ms")
    row_coeff = {
        "flat": params.row_flat_ms,
        "top_last": params.row_top_last_ms,
        # The natural dataset did not contain enough high-row shapes to
        # identify an independent row coefficient.  Keeping it at zero is
        # deliberate; borrowing another distribution's fit would manufacture
        # unsupported calibration.
        "natural": 0.0,
    }[distribution]
    x_log = math.log2(shape.context_length / shape.topk)
    x_tail = max(0.0, (shape.context_length - params.tail_threshold) / params.tail_threshold)
    row_excess = max(0.0, rows / params.row_saturation - 1.0)
    log_us = log_coeff * x_log * 1000.0
    tail_us = tail_coeff * x_tail * 1000.0
    row_penalty_us = row_coeff * row_excess * 1000.0
    floor_us = short_floor * params.latency_scale
    scan_us = (long_floor + log_us + tail_us + row_penalty_us) * params.latency_scale
    raw_scan_us = (score_slots * 4 + output_indices * 4) / bandwidth * 1e6
    return IndexTopKEstimate(
        latency_us=floor_us + scan_us,
        parameter_level=level,
        layout=shape.layout,
        variant=shape.variant,
        score_distribution=shape.score_distribution,
        scope="single_h100_sglang_0.5.12_graph_replay_v2",
        warnings=tuple(warnings),
        floor_us=floor_us,
        scan_us=scan_us,
        score_slots=score_slots,
        output_indices=output_indices,
        query_rows=rows,
        row_parallel_efficiency=min(1.0, rows / params.row_saturation),
        effective_pass_factor=(scan_us / raw_scan_us if raw_scan_us > 0 else 0.0),
        row_saturation=params.row_saturation,
        long_path=True,
        log_us=log_us * params.latency_scale,
        tail_us=tail_us * params.latency_scale,
        row_penalty_us=row_penalty_us * params.latency_scale,
        threshold=shape.topk,
        tail_threshold=params.tail_threshold,
    )


__all__ = [
    "BF16_PROXY_MESSAGE",
    "CALIBRATED_TOPK",
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
