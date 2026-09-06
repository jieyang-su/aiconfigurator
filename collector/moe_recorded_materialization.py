#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Materialize simulation-facing Recorded WideEP MoE rows from AIC sources.

This module intentionally accepts only collector/operator data.  Server/profile
truth is not an input here; truth remains a compare/report concern.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import median, pstdev

from moe_hybrid_policy import apply_profile_free_hybrid_latency


RECORDED_DISTRIBUTIONS = {"recorded", "recorded_no_eplb", "recorded_eplb"}
WIDEEP_MOE_OUTPUT_FIELDS = [
    "framework",
    "version",
    "device",
    "op_name",
    "kernel_source",
    "moe_dtype",
    "num_tokens",
    "hidden_size",
    "inter_size",
    "topk",
    "num_experts",
    "moe_tp_size",
    "moe_ep_size",
    "distribution",
    "workload_source",
    "measurement_scope",
    "latency",
    "origin_latency",
    "power",
    "gemm_path",
    "kernel_regime",
    "primary_latency_source",
    "latency_policy_scope",
    "materialization_role",
    "materialization_source_family",
]
WIDEEP_MOE_DIAGNOSTIC_FIELDS = [
    "aic_rank_rawmax_over_mean",
    "aic_rank_p90_over_mean",
    "aic_sync_tail_over_mean",
    "aic_workload_rank_imbalance",
    "aic_workload_rank_assignments_max",
    "aic_workload_expert_m_max",
    "aic_ep8_lowlat_tail_risk_hint",
    "aic_rank_steady_mean_min",
    "aic_rank_steady_mean_median",
    "aic_rank_steady_mean_max",
    "aic_rank_steady_mean_std",
    "aic_rank_steady_mean_cv",
    "aic_rank_steady_p90_min",
    "aic_rank_steady_p90_median",
    "aic_rank_steady_p90_max",
    "aic_rank_steady_p90_std",
    "aic_rank_steady_p90_cv",
    "aic_rank_envelope_ms",
    "aic_rank_envelope_over_mean",
    "aic_rank_bimodal_gap_ms",
    "aic_rank_bimodal_gap_over_mean",
    "aic_stage_replay_p90_over_mean",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fields_with_origin_latency(rows: list[dict[str, str]]) -> list[str]:
    available = set(rows[0].keys()) if rows else set()
    available.update(
        {
            "origin_latency",
            "latency_policy_scope",
            "materialization_role",
            "materialization_source_family",
            *WIDEEP_MOE_DIAGNOSTIC_FIELDS,
        }
    )
    output_fields = [*WIDEEP_MOE_OUTPUT_FIELDS, *WIDEEP_MOE_DIAGNOSTIC_FIELDS]
    return [field for field in output_fields if field in available or field == "origin_latency"]


def _as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return 0.0
    try:
        parsed = float(value)
    except ValueError:
        return 0.0
    if not math.isfinite(parsed):
        return 0.0
    return parsed


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _format_diag(value: float) -> str:
    if not math.isfinite(value):
        return "0"
    return f"{value:.12g}"


def _json_numeric_values(row: dict[str, str], key: str) -> list[float]:
    value = row.get(key, "")
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return []
    if isinstance(parsed, dict):
        iterable = parsed.values()
    elif isinstance(parsed, list):
        iterable = parsed
    else:
        return []
    values: list[float] = []
    for item in iterable:
        try:
            number = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return values


def _json_numeric_value(row: dict[str, str], key: str, item_key: str) -> float:
    value = row.get(key, "")
    if not value:
        return 0.0
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return 0.0
    if not isinstance(parsed, dict):
        return 0.0
    try:
        number = float(parsed.get(item_key, 0.0))
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return number


def _add_vector_stats(output: dict[str, str], prefix: str, values: list[float]) -> None:
    if not values:
        for suffix in ("min", "median", "max", "std", "cv"):
            output[f"{prefix}_{suffix}"] = "0"
        return
    mean = sum(values) / len(values)
    std = pstdev(values) if len(values) > 1 else 0.0
    output[f"{prefix}_min"] = _format_diag(min(values))
    output[f"{prefix}_median"] = _format_diag(median(values))
    output[f"{prefix}_max"] = _format_diag(max(values))
    output[f"{prefix}_std"] = _format_diag(std)
    output[f"{prefix}_cv"] = _format_diag(_ratio(std, mean))


def _largest_adjacent_gap(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    ordered = sorted(values)
    return max(right - left for left, right in zip(ordered, ordered[1:]))


def _add_wideep_diagnostics(output: dict[str, str], source: dict[str, str], *, phase: str) -> None:
    rank_mean = _as_float(source, "rank_mean_latency")
    rank_raw_max = _as_float(source, "latency_raw_max") or _as_float(source, "latency_max")
    rank_p90 = _as_float(source, "rank_p90_latency")
    sync_tail = _as_float(source, "rank_sync_tail_mean")
    workload_imbalance = _as_float(source, "workload_rank_imbalance_max_over_mean")
    workload_rank_max = _as_float(source, "workload_rank_assignments_max")
    workload_expert_m_max = _as_float(source, "workload_expert_m_max")

    output["aic_rank_rawmax_over_mean"] = _format_diag(_ratio(rank_raw_max, rank_mean))
    output["aic_rank_p90_over_mean"] = _format_diag(_ratio(rank_p90, rank_mean))
    output["aic_sync_tail_over_mean"] = _format_diag(_ratio(sync_tail, rank_mean))
    output["aic_workload_rank_imbalance"] = _format_diag(workload_imbalance)
    output["aic_workload_rank_assignments_max"] = _format_diag(workload_rank_max)
    output["aic_workload_expert_m_max"] = _format_diag(workload_expert_m_max)

    ep_size = int(float(source.get("moe_ep_size", "0") or 0))
    token = int(float(source.get("num_tokens", "0") or 0))
    kernel_regime = str(source.get("kernel_regime", "")).lower()
    risk_hint = 0.0
    if phase == "generation" and ep_size >= 8 and "low_latency" in kernel_regime:
        risk_hint = max(
            0.0,
            _ratio(rank_raw_max, rank_mean) - 1.0,
            _ratio(rank_p90, rank_mean) - 1.0,
            _ratio(sync_tail, rank_mean),
            workload_imbalance - 1.0,
        )
        if token >= 512:
            risk_hint *= 1.10
    output["aic_ep8_lowlat_tail_risk_hint"] = _format_diag(risk_hint)

    steady_mean_values = _json_numeric_values(source, "rank_steady_mean_ms_json")
    steady_p90_values = _json_numeric_values(source, "rank_steady_p90_ms_json")
    _add_vector_stats(output, "aic_rank_steady_mean", steady_mean_values)
    _add_vector_stats(output, "aic_rank_steady_p90", steady_p90_values)

    rank_envelope = max(steady_mean_values) - min(steady_mean_values) if steady_mean_values else 0.0
    rank_vector_mean = sum(steady_mean_values) / len(steady_mean_values) if steady_mean_values else 0.0
    bimodal_gap = _largest_adjacent_gap(steady_mean_values)
    output["aic_rank_envelope_ms"] = _format_diag(rank_envelope)
    output["aic_rank_envelope_over_mean"] = _format_diag(_ratio(rank_envelope, rank_vector_mean))
    output["aic_rank_bimodal_gap_ms"] = _format_diag(bimodal_gap)
    output["aic_rank_bimodal_gap_over_mean"] = _format_diag(_ratio(bimodal_gap, rank_vector_mean))

    stage_mean = _json_numeric_value(source, "stage_mean_ms_json", "cuda_graph_replay")
    if stage_mean <= 0.0:
        stage_mean = _as_float(source, "stage_kernel_sum_mean")
    stage_p90_values = _json_numeric_values(source, "stage_p90_ms_json")
    replay_p90 = max(stage_p90_values) if stage_p90_values else 0.0
    output["aic_stage_replay_p90_over_mean"] = _format_diag(_ratio(replay_p90, stage_mean))


ShapeKey = tuple[str, str]
RowKey = tuple[str, str, int]


def _shape_key(row: dict[str, str]) -> ShapeKey:
    return (str(row.get("moe_tp_size", "")), str(row.get("moe_ep_size", "")))


def _indexed_rows(rows: list[dict[str, str]], *, distribution: str) -> dict[RowKey, dict[str, str]]:
    return {
        (*_shape_key(row), int(float(row["num_tokens"]))): row
        for row in rows
        if row.get("distribution") == distribution
    }


def _ep_sizes_with_tokens(
    rows_by_shape_token: dict[RowKey, dict[str, str]],
    tokens: tuple[int, ...],
) -> list[ShapeKey]:
    shapes = {(tp_size, ep_size) for tp_size, ep_size, _ in rows_by_shape_token}
    return sorted(
        [
            shape
            for shape in shapes
            if all((*shape, token) in rows_by_shape_token for token in tokens)
        ],
        key=lambda value: (int(float(value[0])), int(float(value[1]))),
    )


def _all_shapes(rows_by_shape_token: dict[RowKey, dict[str, str]]) -> list[ShapeKey]:
    return sorted(
        {(tp_size, ep_size) for tp_size, ep_size, _ in rows_by_shape_token},
        key=lambda value: (int(float(value[0])), int(float(value[1]))),
    )


def _tokens_for_shape(rows_by_shape_token: dict[RowKey, dict[str, str]], shape: ShapeKey) -> list[int]:
    return sorted(token for tp_size, ep_size, token in rows_by_shape_token if (tp_size, ep_size) == shape)


def _materialized_row(row: dict[str, str], *, phase: str, role: str) -> dict[str, str]:
    output = apply_profile_free_hybrid_latency(row, phase=phase)
    output["latency_policy_scope"] = "profile_free_hybrid_recorded"
    output["materialization_role"] = role
    output["materialization_source_family"] = "collector_recorded_materialization"
    _add_wideep_diagnostics(output, row, phase=phase)
    return output


def context_recorded_rows(
    *,
    sparse_rows: list[dict[str, str]],
    dense_rows: list[dict[str, str]],
    distribution: str,
) -> list[dict[str, str]]:
    sparse = _indexed_rows(sparse_rows, distribution=distribution)
    dense = _indexed_rows(dense_rows, distribution=distribution)

    output: list[dict[str, str]] = []
    for shape in _all_shapes(sparse):
        for token in _tokens_for_shape(sparse, shape):
            output.append(
                _materialized_row(
                    sparse[(*shape, token)],
                    phase="context",
                    role="context_sparse",
                )
            )

    for shape in _all_shapes(dense):
        for token in _tokens_for_shape(dense, shape):
            if (*shape, token) in sparse:
                continue
            output.append(
                _materialized_row(
                    dense[(*shape, token)],
                    phase="context",
                    role="context_dense",
                )
            )
    return output


def _append_generation_token_if_present(
    output: list[dict[str, str]],
    rows_by_shape_token: dict[RowKey, dict[str, str]],
    *,
    shape: ShapeKey,
    token: int,
    role: str,
) -> None:
    row = rows_by_shape_token.get((*shape, token))
    if row is None:
        return
    output.append(
        _materialized_row(
            row,
            phase="generation",
            role=role,
        )
    )


def _generation_ep_sizes(
    small: dict[RowKey, dict[str, str]],
    main: dict[RowKey, dict[str, str]],
) -> list[ShapeKey]:
    shapes = (
        {(tp_size, ep_size) for tp_size, ep_size, _ in small}
        | {(tp_size, ep_size) for tp_size, ep_size, _ in main}
    )
    return sorted(
        [
            shape
            for shape in shapes
            if (*shape, 8) in small
        ],
        key=lambda value: (int(float(value[0])), int(float(value[1]))),
    )


def generation_recorded_rows(
    *,
    small_rows: list[dict[str, str]],
    main_rows: list[dict[str, str]],
    distribution: str,
) -> list[dict[str, str]]:
    small = _indexed_rows(small_rows, distribution=distribution)
    main = _indexed_rows(main_rows, distribution=distribution)

    output: list[dict[str, str]] = []
    for shape in _generation_ep_sizes(small, main):
        for token in _tokens_for_shape(small, shape):
            _append_generation_token_if_present(
                output,
                small,
                shape=shape,
                token=token,
                role="generation_small",
            )
        for token in _tokens_for_shape(main, shape):
            if (*shape, token) in small:
                continue
            _append_generation_token_if_present(
                output,
                main,
                shape=shape,
                token=token,
                role="generation_main",
            )

    return output


def materialize_recorded_wideep_tables(
    *,
    base_context_rows: list[dict[str, str]],
    base_generation_rows: list[dict[str, str]],
    context_sparse_rows: list[dict[str, str]],
    context_dense_noeplb_rows: list[dict[str, str]],
    context_dense_eplb_rows: list[dict[str, str]],
    generation_small_noeplb_rows: list[dict[str, str]],
    generation_small_eplb_rows: list[dict[str, str]],
    generation_main_noeplb_rows: list[dict[str, str]],
    generation_main_eplb_rows: list[dict[str, str]],
) -> tuple[list[str], list[dict[str, str]], list[str], list[dict[str, str]]]:
    fields_context = _fields_with_origin_latency(base_context_rows)
    fields_generation = _fields_with_origin_latency(base_generation_rows)
    for fields in (fields_context, fields_generation):
        for key in (
            "latency_policy_scope",
            "materialization_role",
            "materialization_source_family",
        ):
            if key not in fields:
                fields.append(key)

    context_rows = [
        row
        for row in base_context_rows
        if row.get("distribution") not in RECORDED_DISTRIBUTIONS
    ]
    context_rows.extend(
        {key: row.get(key, "") for key in fields_context}
        for row in context_recorded_rows(
            sparse_rows=context_sparse_rows,
            dense_rows=context_dense_noeplb_rows,
            distribution="recorded_no_eplb",
        )
    )
    context_rows.extend(
        {key: row.get(key, "") for key in fields_context}
        for row in context_recorded_rows(
            sparse_rows=context_sparse_rows,
            dense_rows=context_dense_eplb_rows,
            distribution="recorded_eplb",
        )
    )
    context_rows.sort(
        key=lambda row: (
            int(float(row["moe_ep_size"])),
            int(float(row["num_tokens"])),
            row["distribution"],
            row.get("kernel_source", ""),
        )
    )

    generation_rows = [
        row
        for row in base_generation_rows
        if row.get("distribution") not in RECORDED_DISTRIBUTIONS
    ]
    generation_rows.extend(
        {key: row.get(key, "") for key in fields_generation}
        for row in generation_recorded_rows(
            small_rows=generation_small_noeplb_rows,
            main_rows=generation_main_noeplb_rows,
            distribution="recorded_no_eplb",
        )
    )
    generation_rows.extend(
        {key: row.get(key, "") for key in fields_generation}
        for row in generation_recorded_rows(
            small_rows=generation_small_eplb_rows,
            main_rows=generation_main_eplb_rows,
            distribution="recorded_eplb",
        )
    )
    generation_rows.sort(
        key=lambda row: (
            int(float(row["moe_ep_size"])),
            int(float(row["num_tokens"])),
            row["distribution"],
            row.get("kernel_source", ""),
        )
    )
    return fields_context, context_rows, fields_generation, generation_rows


def materialize_recorded_wideep_tables_from_paths(
    *,
    base_validation_source: Path,
    context_sparse: Path,
    context_dense_noeplb: Path,
    context_dense_eplb: Path,
    generation_small_noeplb: Path,
    generation_small_eplb: Path,
    generation_main_noeplb: Path,
    generation_main_eplb: Path,
    output_dir: Path,
) -> None:
    fields_context, context_rows, fields_generation, generation_rows = materialize_recorded_wideep_tables(
        base_context_rows=read_csv_rows(base_validation_source / "wideep_context_moe_perf.txt"),
        base_generation_rows=read_csv_rows(base_validation_source / "wideep_generation_moe_perf.txt"),
        context_sparse_rows=read_csv_rows(context_sparse),
        context_dense_noeplb_rows=read_csv_rows(context_dense_noeplb),
        context_dense_eplb_rows=read_csv_rows(context_dense_eplb),
        generation_small_noeplb_rows=read_csv_rows(generation_small_noeplb),
        generation_small_eplb_rows=read_csv_rows(generation_small_eplb),
        generation_main_noeplb_rows=read_csv_rows(generation_main_noeplb),
        generation_main_eplb_rows=read_csv_rows(generation_main_eplb),
    )
    write_csv_rows(output_dir / "wideep_context_moe_perf.txt", [
        {key: row.get(key, "") for key in fields_context} for row in context_rows
    ])
    write_csv_rows(output_dir / "wideep_generation_moe_perf.txt", [
        {key: row.get(key, "") for key in fields_generation} for row in generation_rows
    ])
