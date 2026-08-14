"""Standalone FA2/FA3 analytical and generic-profile latency model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Any

from .profiles import PROFILE_VERSION, ReferenceProfile, get_reference_profile
from .schema import AttentionShape, HardwareSpec, ModelOptions, dtype_bytes

MODEL_VERSION = "2026-08-13.aic-fa-roofline-v4"


@dataclass(frozen=True)
class EstimateResult:
    model_version: str
    profile_version: str | None
    mode: str
    algorithm: str
    latency_us: float
    bottleneck: str
    shape: dict[str, Any]
    hardware: dict[str, Any]
    options: dict[str, Any]
    tiles: dict[str, Any]
    work: dict[str, Any]
    resources_us: dict[str, Any]
    scheduling: dict[str, Any]
    reference_profile: dict[str, Any] | None
    assumptions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["assumptions"] = list(self.assumptions)
        return payload


def _power_of_two_floor(value: int) -> int:
    if value < 1:
        return 1
    return 1 << (value.bit_length() - 1)


def _block_sizes(
    hardware: HardwareSpec, shape: AttentionShape, options: ModelOptions
) -> tuple[int, int, dict[str, Any]]:
    words = hardware.shared_memory_per_sm_bytes // dtype_bytes(shape.dtype)
    resident_width = shape.head_dim + shape.kv_storage_dim + shape.value_head_dim
    capacity_bound = max(1, words // resident_width)
    automatic = min(128, _power_of_two_floor(capacity_bound))
    br = automatic if options.br is None else options.br
    bc = automatic if options.bc is None else options.bc
    return (
        br,
        bc,
        {
            "shared_memory_words": words,
            "resident_width_elements": resident_width,
            "four_buffer_capacity_bound": capacity_bound,
            "automatic_block": automatic,
            "br_source": "capacity_heuristic" if options.br is None else "explicit",
            "bc_source": "capacity_heuristic" if options.bc is None else "explicit",
        },
    )


def _tile_work(shape: AttentionShape, br: int, bc: int) -> dict[str, int]:
    history = shape.history_length
    score_elements = 0
    exact_valid_scores = 0
    kv_tokens_loaded = 0
    row_tile_updates = 0
    active_kv_tiles = 0

    for query_start in range(0, shape.query_length, br):
        rows = min(br, shape.query_length - query_start)
        if shape.causal:
            exact_columns = min(shape.kv_length_total, history + query_start + rows)
            exact_valid_scores += sum(
                min(shape.kv_length_total, history + row + 1) for row in range(query_start, query_start + rows)
            )
        else:
            exact_columns = shape.kv_length_total
            exact_valid_scores += rows * shape.kv_length_total
        key_tiles = ceil(exact_columns / bc)
        loaded_columns = min(shape.kv_length_total, key_tiles * bc)
        score_elements += rows * loaded_columns
        kv_tokens_loaded += loaded_columns
        row_tile_updates += rows * key_tiles
        active_kv_tiles += key_tiles

    return {
        "score_elements_per_query_head": score_elements,
        "exact_valid_scores_per_query_head": exact_valid_scores,
        "kv_tokens_loaded_per_query_head": kv_tokens_loaded,
        "row_tile_updates_per_query_head": row_tile_updates,
        "active_kv_tiles_per_query_head": active_kv_tiles,
    }


def _decode_splits(
    hardware: HardwareSpec,
    shape: AttentionShape,
    options: ModelOptions,
    query_tiles: int,
    bc: int,
) -> tuple[int, dict[str, Any]]:
    total_kv_tiles = ceil(shape.kv_length_total / bc)
    max_by_work = max(1, total_kv_tiles // options.min_kv_tiles_per_split)
    split_limit = min(options.max_decode_splits, total_kv_tiles, max_by_work)
    base_ctas = shape.batch_size * shape.query_heads * query_tiles
    splits_for_one_wave = ceil(hardware.sm_count / base_ctas)

    if isinstance(options.decode_splits, int):
        selected = min(options.decode_splits, split_limit)
        source = "explicit_capped" if selected != options.decode_splits else "explicit"
    elif shape.query_length > options.decode_query_threshold or shape.history_length == 0:
        selected = 1
        source = "disabled_outside_decode_domain"
    else:
        selected = max(1, min(splits_for_one_wave, split_limit))
        source = "one_sm_wave_heuristic"
    return selected, {
        "source": source,
        "total_kv_tiles": total_kv_tiles,
        "base_ctas": base_ctas,
        "splits_for_one_wave": splits_for_one_wave,
        "max_by_minimum_work": max_by_work,
        "split_limit": split_limit,
        "minimum_kv_tiles_per_split": options.min_kv_tiles_per_split,
    }


def estimate_attention(
    hardware: HardwareSpec,
    shape: AttentionShape,
    options: ModelOptions | None = None,
) -> EstimateResult:
    """Estimate one attention operation and return a fully inspectable result."""
    options = options or ModelOptions()
    algorithm = options.algorithm
    matrix_peak = hardware.matrix_peak(shape.dtype)
    input_bytes = dtype_bytes(shape.dtype)
    output_bytes = dtype_bytes(shape.output_dtype)
    kv_cache_bytes_per_token = (
        shape.kv_storage_dim * input_bytes
        if shape.kv_cache_bytes_per_token is None
        else shape.kv_cache_bytes_per_token
    )
    br, bc, block_metadata = _block_sizes(hardware, shape, options)
    tile_work = _tile_work(shape, br, bc)
    query_tiles = ceil(shape.query_length / br)
    kv_tiles = ceil(shape.kv_length_total / bc)
    kv_splits, split_metadata = _decode_splits(hardware, shape, options, query_tiles, bc)

    base_ctas = shape.batch_size * shape.query_heads * query_tiles
    cta_count = base_ctas * kv_splits
    if options.account_for_parallelism:
        parallel_efficiency = min(1.0, cta_count / hardware.sm_count)
        reduction_parallel_efficiency = min(1.0, base_ctas / hardware.sm_count)
    else:
        parallel_efficiency = reduction_parallel_efficiency = 1.0
    parallel_efficiency = max(parallel_efficiency, 1.0 / hardware.sm_count)
    reduction_parallel_efficiency = max(reduction_parallel_efficiency, 1.0 / hardware.sm_count)

    score_elements = shape.batch_size * shape.query_heads * tile_work["score_elements_per_query_head"]
    exact_scores = shape.batch_size * shape.query_heads * tile_work["exact_valid_scores_per_query_head"]
    qk_flops = 2 * score_elements * shape.head_dim
    pv_flops = 2 * score_elements * shape.value_head_dim
    matrix_flops = qk_flops + pv_flops

    query_rows = shape.batch_size * shape.query_heads * shape.query_length
    row_updates = shape.batch_size * shape.query_heads * tile_work["row_tile_updates_per_query_head"]
    exp_cost = hardware.exp_flop_equivalent
    q_scale_flops = kv_splits * query_rows * shape.head_dim
    score_softmax_flops = score_elements * (exp_cost + 3)
    online_state_flops = row_updates * (exp_cost + 4 + shape.value_head_dim)
    final_normalize_flops = kv_splits * query_rows * (exp_cost + 2 + shape.value_head_dim)
    if kv_splits > 1:
        split_reduction_flops = query_rows * (
            2 * (kv_splits - 1) + kv_splits * (exp_cost + 1) + exp_cost + 1 + 2 * kv_splits * shape.value_head_dim
        )
    else:
        split_reduction_flops = 0.0
    mainloop_vector_flops = score_softmax_flops + online_state_flops
    split_boundary_flops = q_scale_flops + final_normalize_flops
    boundary_vector_flops = split_boundary_flops + split_reduction_flops
    vector_flops = mainloop_vector_flops + boundary_vector_flops

    query_elements = shape.batch_size * shape.query_heads * shape.query_length * shape.head_dim
    output_elements = shape.batch_size * shape.query_heads * shape.query_length * shape.value_head_dim
    kv_l2_heads = shape.kv_heads if options.assume_gqa_l2_reuse else shape.query_heads
    kv_hbm_heads = shape.kv_heads if options.assume_gqa_hbm_reuse else shape.query_heads
    kv_loaded = tile_work["kv_tokens_loaded_per_query_head"]
    kv_hbm_loaded = shape.kv_length_total if options.assume_query_tile_l2_reuse else kv_loaded
    kv_l2_tokens = shape.batch_size * kv_l2_heads * kv_loaded
    kv_hbm_tokens = shape.batch_size * kv_hbm_heads * kv_hbm_loaded
    lse_bytes = query_rows * 4 if options.store_lse else 0
    q_bytes = kv_splits * query_elements * input_bytes
    kv_hbm_bytes = kv_hbm_tokens * kv_cache_bytes_per_token
    output_hbm_bytes = output_elements * output_bytes
    partial_bytes = kv_splits * query_rows * (shape.value_head_dim + 1) * 4 if kv_splits > 1 else 0
    if kv_splits > 1:
        mainloop_hbm_bytes = q_bytes + kv_hbm_bytes + partial_bytes
        reduction_hbm_bytes = partial_bytes + output_hbm_bytes + lse_bytes
        mainloop_l2_bytes = (
            kv_splits * query_elements * input_bytes
            + kv_l2_tokens * kv_cache_bytes_per_token
            + partial_bytes
        )
        reduction_l2_bytes = partial_bytes + output_hbm_bytes + lse_bytes
    else:
        mainloop_hbm_bytes = q_bytes + kv_hbm_bytes + output_hbm_bytes + lse_bytes
        reduction_hbm_bytes = 0
        mainloop_l2_bytes = (
            query_elements * input_bytes
            + kv_l2_tokens * kv_cache_bytes_per_token
            + output_hbm_bytes
            + lse_bytes
        )
        reduction_l2_bytes = 0

    live_kv_tokens = shape.batch_size * shape.kv_heads * shape.query_length
    live_kv_cache_update_bytes = 2 * live_kv_tokens * kv_cache_bytes_per_token if options.include_kv_cache_update else 0
    hbm_bytes = mainloop_hbm_bytes + reduction_hbm_bytes
    l2_bytes = mainloop_l2_bytes + reduction_l2_bytes

    matrix_effective_peak = matrix_peak * parallel_efficiency
    vector_effective_peak = hardware.vector_peak_flops * parallel_efficiency
    hbm_effective_bandwidth = hardware.hbm_bandwidth_bytes_s * parallel_efficiency
    l2_effective_bandwidth = hardware.l2_bandwidth_bytes_s * parallel_efficiency
    reduction_hbm_bandwidth = hardware.hbm_bandwidth_bytes_s * reduction_parallel_efficiency
    reduction_l2_bandwidth = hardware.l2_bandwidth_bytes_s * reduction_parallel_efficiency

    raw_hbm_s = (
        mainloop_hbm_bytes / hbm_effective_bandwidth
        + reduction_hbm_bytes / reduction_hbm_bandwidth
        + live_kv_cache_update_bytes / hbm_effective_bandwidth
    )
    raw_l2_s = mainloop_l2_bytes / l2_effective_bandwidth + reduction_l2_bytes / reduction_l2_bandwidth
    matrix_s = matrix_flops / matrix_effective_peak
    mainloop_vector_s = mainloop_vector_flops / vector_effective_peak
    boundary_vector_s = split_boundary_flops / vector_effective_peak + split_reduction_flops / (
        hardware.vector_peak_flops * reduction_parallel_efficiency
    )
    overlap = options.overlap_fraction
    if overlap is None:
        overlap = 1.0 if algorithm == "fa3" else 0.0
    compute_s = matrix_s + mainloop_vector_s - overlap * min(matrix_s, mainloop_vector_s) + boundary_vector_s

    raw_components = {"hbm": raw_hbm_s, "l2": raw_l2_s, "compute": compute_s}
    raw_bottleneck = max(raw_components, key=raw_components.get)
    raw_resource_s = raw_components[raw_bottleneck]
    decode_service_active = shape.history_length > 0 and shape.query_length <= options.decode_query_threshold
    n_task = shape.batch_size * shape.kv_heads * query_tiles * kv_splits if decode_service_active else 0
    fractional_waves = n_task / hardware.sm_count

    reference: ReferenceProfile | None = None
    profile_dict: dict[str, Any] | None = None
    effective_task_cycles = 0.0
    fixed_overhead_s = 0.0
    task_service_s = 0.0
    adjusted_components = dict(raw_components)

    if options.mode == "profiled":
        reference = get_reference_profile(options.estimate_level)
        adjusted_components = {
            "hbm": raw_hbm_s / reference.hbm_efficiency,
            "l2": raw_l2_s / reference.l2_efficiency,
            "compute": compute_s / reference.compute_efficiency,
        }
        fixed_overhead_s = reference.fixed_overhead_us * 1e-6
        if decode_service_active:
            effective_task_cycles = reference.kv_task_cycles
            task_service_s = fractional_waves * effective_task_cycles / hardware.clock_hz
        profile_dict = reference.to_dict()

    bottleneck = max(adjusted_components, key=adjusted_components.get)
    resource_s = adjusted_components[bottleneck]
    latency_s = fixed_overhead_s + resource_s + task_service_s

    total_flops = matrix_flops + vector_flops
    total_counted_hbm_bytes = hbm_bytes + live_kv_cache_update_bytes
    work = {
        "matrix_flops": matrix_flops,
        "qk_flops": qk_flops,
        "pv_flops": pv_flops,
        "vector_equivalent_flops": vector_flops,
        "mainloop_vector_equivalent_flops": mainloop_vector_flops,
        "boundary_vector_equivalent_flops": boundary_vector_flops,
        "split_reduction_equivalent_flops": split_reduction_flops,
        "total_equivalent_flops": total_flops,
        "executed_score_elements": score_elements,
        "exact_valid_score_elements": exact_scores,
        "tile_padding_score_elements": score_elements - exact_scores,
        "mainloop_hbm_bytes": mainloop_hbm_bytes,
        "reduction_hbm_bytes": reduction_hbm_bytes,
        "live_kv_cache_update_bytes": live_kv_cache_update_bytes,
        "kv_cache_bytes_per_token": kv_cache_bytes_per_token,
        "total_hbm_bytes": total_counted_hbm_bytes,
        "mainloop_l2_requested_bytes": mainloop_l2_bytes,
        "reduction_l2_requested_bytes": reduction_l2_bytes,
        "total_l2_requested_bytes": l2_bytes,
        "partial_state_bytes_per_direction": partial_bytes,
        "arithmetic_intensity_flop_per_hbm_byte": (
            total_flops / total_counted_hbm_bytes if total_counted_hbm_bytes else None
        ),
    }
    resources_us = {
        "raw_hbm_us": raw_hbm_s * 1e6,
        "raw_l2_us": raw_l2_s * 1e6,
        "raw_matrix_us": matrix_s * 1e6,
        "raw_mainloop_vector_us": mainloop_vector_s * 1e6,
        "raw_boundary_vector_us": boundary_vector_s * 1e6,
        "raw_compute_combined_us": compute_s * 1e6,
        "raw_resource_roof_us": raw_resource_s * 1e6,
        "raw_bottleneck": raw_bottleneck,
        "adjusted_hbm_us": adjusted_components["hbm"] * 1e6,
        "adjusted_l2_us": adjusted_components["l2"] * 1e6,
        "adjusted_compute_us": adjusted_components["compute"] * 1e6,
        "adjusted_resource_roof_us": resource_s * 1e6,
        "fixed_overhead_us": fixed_overhead_s * 1e6,
        "task_service_us": task_service_s * 1e6,
        "overlap_fraction": overlap,
        "matrix_peak_flops": matrix_peak,
        "vector_peak_flops": hardware.vector_peak_flops,
    }
    scheduling = {
        "base_cta_count_hq_mapping": base_ctas,
        "cta_count_hq_mapping_with_splits": cta_count,
        "parallel_efficiency": parallel_efficiency,
        "reduction_parallel_efficiency": reduction_parallel_efficiency,
        "decode_service_active": decode_service_active,
        "n_task_b_hkv_query_tiles_splits": n_task,
        "fractional_task_waves": fractional_waves,
        "effective_task_cycles": effective_task_cycles,
        "split_resolution": split_metadata,
    }
    tiles = {
        "br": br,
        "bc": bc,
        "query_tiles": query_tiles,
        "kv_tiles": kv_tiles,
        "kv_splits": kv_splits,
        **block_metadata,
        **tile_work,
    }
    hardware_summary = {
        "sm_count": hardware.sm_count,
        "clock_hz": hardware.clock_hz,
        "hbm_bandwidth_bytes_s": hardware.hbm_bandwidth_bytes_s,
        "l2_bandwidth_bytes_s": hardware.l2_bandwidth_bytes_s,
        "shared_memory_per_sm_bytes": hardware.shared_memory_per_sm_bytes,
        "l2_capacity_bytes": hardware.l2_capacity_bytes,
        "matrix_peak_flops": dict(hardware.matrix_peak_flops),
        "vector_peak_flops": hardware.vector_peak_flops,
        "exp_flop_equivalent": hardware.exp_flop_equivalent,
    }
    assumptions = (
        "Q-outer/KV-inner tiled online-softmax dataflow.",
        "A separate KV-cache byte width changes HBM/L2 traffic only; conversion into the compute dtype is assumed "
        "fused and has no explicit penalty.",
        "One base logical CTA per batch, query head and query tile; occupancy is linearized by CTA/SM count.",
        "GQA HBM/L2 reuse follows the selected ideal reuse switches, not a cache simulation.",
        "L2 capacity is reported but no residency, associativity or hit-rate model is applied.",
        "FA3 overlap is represented by a scalar overlap fraction, not a WGMMA/TMA pipeline simulation.",
        "Decode N_task is an equivalent KV-head-group service count, not an observed CUDA grid.",
    )
    return EstimateResult(
        model_version=MODEL_VERSION,
        profile_version=PROFILE_VERSION if reference else None,
        mode=options.mode,
        algorithm=algorithm,
        latency_us=latency_s * 1e6,
        bottleneck=bottleneck,
        shape=shape.to_dict(),
        hardware=hardware_summary,
        options=options.to_dict(),
        tiles=tiles,
        work=work,
        resources_us=resources_us,
        scheduling=scheduling,
        reference_profile=profile_dict,
        assumptions=assumptions,
    )


__all__ = ["MODEL_VERSION", "EstimateResult", "estimate_attention"]
