#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Materialize simulation-facing Recorded WideEP MoE rows from AIC sources.

This module intentionally accepts only collector/operator data.  Server/profile
truth is not an input here; truth remains a compare/report concern.
"""

from __future__ import annotations

import csv
from pathlib import Path

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
        }
    )
    return [field for field in WIDEEP_MOE_OUTPUT_FIELDS if field in available or field == "origin_latency"]


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
