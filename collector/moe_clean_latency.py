#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Clean-latency source selection for DeepSeek-V3 MoE collector outputs.

This module is collector-side logic: it reads only AIC collector outputs and
AIC latency source bundles, then writes final MoE latency tables whose
``latency`` column is the clean critical-path latency.  Server/profile truth is
not an input here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from collector.moe_recorded_materializer import (
    ORDINARY_COMMON_FILTER,
    WIDEEP_CONTEXT_COMMON_FILTER,
    WIDEEP_GENERATION_COMMON_FILTER,
    default_betas,
    load_ordinary_sources,
    load_wideep_context_sources,
    load_wideep_generation_sources,
    materialize_table,
    passes_filter,
)


csv.field_size_limit(sys.maxsize)

WIDEEP_SOURCE_FIELDS = (
    "latency",
    "latency_max",
    "latency_raw_max",
    "latency_stage_sum_rankmax",
    "latency_stage_sum_rankmax_max",
    "rank_mean_latency",
    "rank_p90_latency",
    "rank_critical_mean_latency",
    "rank_sync_tail_mean",
    "kernel_regime",
    "gemm_path",
)

WIDEEP_SOURCE_FILES = (
    "context_sparse.txt",
    "context_dense_noeplb.txt",
    "context_dense_eplb.txt",
    "generation_small_noeplb.txt",
    "generation_small_eplb.txt",
    "generation_main_noeplb.txt",
    "generation_main_eplb.txt",
)


def build_clean_latency_tables(
    *,
    source_dir: Path,
    candidate_dir: Path,
    write_origin_dir: bool = False,
) -> None:
    """Build clean-latency MoE tables from AIC-only collector sources."""

    source_dir = source_dir.resolve()
    candidate_dir = candidate_dir.resolve()
    _truth_guard(source_dir, role="source_dir")
    _truth_guard(candidate_dir, role="candidate_dir")
    input_manifest = _validate_inputs(source_dir)
    _copy_source_skeleton(source_dir, candidate_dir)
    _materialize_recorded_with_unified_materializer(source_dir, candidate_dir, input_manifest)
    origin_dir = candidate_dir.with_name(candidate_dir.name + "_origin_latency")
    if write_origin_dir:
        _write_origin_dir(candidate_dir, origin_dir)
    _write_candidate_manifest(
        candidate_dir,
        source_dir=source_dir,
        origin_dir=origin_dir,
        report_dir=candidate_dir,
        input_manifest=input_manifest,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _as_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _truth_guard(path: Path, *, role: str) -> None:
    markers = ("profile_validation", "rank_aggregate", "dense_refresh", "sharegpt", "longbench", "truth")
    text = str(path).lower()
    matched = [marker for marker in markers if marker in text]
    if matched:
        raise ValueError(f"{role} must be AIC-only data, got {path} matching {matched}")


def _copy_source_skeleton(source_dir: Path, candidate_dir: Path) -> None:
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    candidate_dir.mkdir(parents=True)
    for name in (
        "collection_summary_sglang.json",
        "collector.log",
        "collector_errors.log",
        "moe_token_distribution_perf.txt",
    ):
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, candidate_dir / name)
    for dirname in (
        "raw_collector_source",
        "recorded_materialized_source",
        "ordinary_moe_materialized_source",
        "aic_latency_source_bundle",
    ):
        src = source_dir / dirname
        if src.exists():
            shutil.copytree(src, candidate_dir / dirname)


def _validate_inputs(source_dir: Path) -> dict[str, object]:
    manifest: dict[str, object] = {
        "source_dir": str(source_dir),
        "collection_summary_total_errors": None,
        "ordinary_source_available": False,
        "wideep_context_available": False,
        "wideep_generation_available": False,
        "wideep_sources": {},
    }
    summary = source_dir / "collection_summary_sglang.json"
    if summary.exists():
        data = json.loads(summary.read_text())
        errors = int(data.get("total_errors", 0) or 0)
        manifest["collection_summary_total_errors"] = errors
        if errors:
            raise RuntimeError(f"{summary} reports total_errors={errors}")

    bundle = source_dir / "aic_latency_source_bundle" / "recorded_materialization_inputs"
    ordinary_raw = source_dir / "raw_collector_source" / "moe_perf.txt"
    ordinary_table = source_dir / "moe_perf.txt"
    if ordinary_raw.exists() and ordinary_table.exists():
        manifest["ordinary_source_available"] = True
        manifest["ordinary_raw_rows"] = len(_read_csv(ordinary_raw))
    else:
        manifest["ordinary_raw_rows"] = None

    context_required = {"context_sparse.txt", "context_dense_noeplb.txt", "context_dense_eplb.txt"}
    generation_required = set(WIDEEP_SOURCE_FILES) - context_required
    context_table = source_dir / "wideep_context_moe_perf.txt"
    generation_table = source_dir / "wideep_generation_moe_perf.txt"
    context_raw = source_dir / "raw_collector_source" / "wideep_context_moe_perf.txt"
    generation_raw = source_dir / "raw_collector_source" / "wideep_generation_moe_perf.txt"
    context_missing = sorted(name for name in context_required if not (bundle / name).exists())
    generation_missing = sorted(name for name in generation_required if not (bundle / name).exists())
    if context_table.exists() and context_raw.exists():
        manifest["wideep_context_available"] = True
    if generation_table.exists() and generation_raw.exists():
        manifest["wideep_generation_available"] = True
    if not (
        manifest["ordinary_source_available"]
        or manifest["wideep_context_available"]
        or manifest["wideep_generation_available"]
    ):
        missing = {
            "ordinary": [str(ordinary_table), str(ordinary_raw)],
            "wideep_context": context_missing or [str(context_table)],
            "wideep_generation": generation_missing or [str(generation_table)],
        }
        raise FileNotFoundError(f"missing clean-latency source files: {missing}")

    active_wideep_files = []
    if manifest["wideep_context_available"] and not context_missing:
        active_wideep_files.extend(sorted(context_required))
    if manifest["wideep_generation_available"] and not generation_missing:
        active_wideep_files.extend(sorted(generation_required))
    for name in active_wideep_files:
        rows = _read_csv(bundle / name)
        recorded_rows = [row for row in rows if row.get("distribution", "").startswith("recorded")]
        if not rows:
            raise RuntimeError(f"{bundle / name} is empty")
        if not recorded_rows:
            raise RuntimeError(f"{bundle / name} contains no recorded rows")
        workload_fields = [field for field in rows[0] if field.startswith("workload_")]
        if not workload_fields:
            raise RuntimeError(f"{bundle / name} contains no workload_* fields")
        required_fields = (*WIDEEP_SOURCE_FIELDS, *workload_fields)
        missing_by_field = {
            field: sum(1 for row in recorded_rows if row.get(field, "") == "")
            for field in required_fields
        }
        missing_by_field = {field: count for field, count in missing_by_field.items() if count}
        if missing_by_field:
            raise RuntimeError(f"{bundle / name} has empty required recorded fields: {missing_by_field}")
        source_manifest = {
            "rows": len(rows),
            "recorded_rows": len(recorded_rows),
            "workload_fields": workload_fields,
            "required_fields_checked": list(required_fields),
        }
        manifest["wideep_sources"][name] = source_manifest
    return manifest


def _write_candidate_manifest(
    candidate_dir: Path,
    *,
    source_dir: Path,
    origin_dir: Path,
    report_dir: Path,
    input_manifest: dict[str, object],
) -> None:
    row_counts = {}
    for filename in ("moe_perf.txt", "wideep_context_moe_perf.txt", "wideep_generation_moe_perf.txt"):
        if not (source_dir / filename).exists() or not (candidate_dir / filename).exists():
            row_counts[filename] = {
                "source_rows": None,
                "candidate_rows": None,
                "missing_aic_critical_path_latency": None,
                "row_count_matches_source": None,
                "skipped": True,
            }
            continue
        source_rows = _read_csv(source_dir / filename)
        candidate_rows = _read_csv(candidate_dir / filename)
        missing_clean = sum(1 for row in candidate_rows if row.get("aic_critical_path_latency", "") == "")
        expected_mask = _expected_clean_mask(filename, pd.DataFrame(candidate_rows))
        expected_rows = int(expected_mask.sum())
        expected_missing = int(
            pd.Series(
                [row.get("aic_critical_path_latency", "") == "" for row in candidate_rows],
                index=expected_mask.index,
            )[expected_mask].sum()
        )
        row_counts[filename] = {
            "source_rows": len(source_rows),
            "candidate_rows": len(candidate_rows),
            "total_missing_aic_critical_path_latency": missing_clean,
            "expected_clean_recorded_rows": expected_rows,
            "missing_expected_clean_recorded_rows": expected_missing,
            "inactive_or_passthrough_rows": len(candidate_rows) - expected_rows,
            "row_count_matches_source": len(source_rows) == len(candidate_rows),
        }
    manifest = {
        "source_dir": str(source_dir),
        "candidate_dir": str(candidate_dir),
        "origin_latency_dir": str(origin_dir),
        "current_invocation_report_dir": str(report_dir),
        "truth_usage": "profile_root is used only by compare report commands; candidate latency generation reads AIC-only source_dir files.",
        "input_validation": input_manifest,
        "candidate_row_counts": row_counts,
        "candidate_columns": [
            "origin_latency",
            "aic_critical_path_latency",
            "aic_latency_source",
            "aic_latency_policy",
            "latency",
        ],
    }
    (candidate_dir / "clean_latency_candidate_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def _expected_clean_mask(filename: str, rows: pd.DataFrame) -> pd.Series:
    """Rows that this candidate is expected to overwrite with validated clean latency."""

    if rows.empty:
        return pd.Series(False, index=rows.index)
    if filename == "moe_perf.txt":
        required = set(ORDINARY_COMMON_FILTER) | {"phase", "distribution"}
        if not required.issubset(rows.columns):
            return pd.Series(False, index=rows.index)
        return (
            passes_filter(rows, ORDINARY_COMMON_FILTER)
            & rows["phase"].isin(["context", "generation"])
            & rows["distribution"].isin(["recorded_eplb", "recorded_no_eplb"])
        )
    if filename == "wideep_context_moe_perf.txt":
        required = set(WIDEEP_CONTEXT_COMMON_FILTER) | {"distribution"}
        if not required.issubset(rows.columns):
            return pd.Series(False, index=rows.index)
        return (
            passes_filter(rows, WIDEEP_CONTEXT_COMMON_FILTER)
            & rows["distribution"].isin(["recorded_eplb", "recorded_no_eplb"])
        )
    if filename == "wideep_generation_moe_perf.txt":
        required = set(WIDEEP_GENERATION_COMMON_FILTER) | {"distribution"}
        if not required.issubset(rows.columns):
            return pd.Series(False, index=rows.index)
        return (
            passes_filter(rows, WIDEEP_GENERATION_COMMON_FILTER)
            & rows["distribution"].isin(["recorded_eplb", "recorded_no_eplb"])
        )
    return pd.Series(False, index=rows.index)



def _materialized_table_source(source_dir: Path, filename: str) -> Path:
    preferred = {
        "moe_perf.txt": source_dir / "ordinary_moe_materialized_source" / "moe_perf.txt",
        "wideep_context_moe_perf.txt": source_dir
        / "recorded_materialized_source"
        / "wideep_context_moe_perf.txt",
        "wideep_generation_moe_perf.txt": source_dir
        / "recorded_materialized_source"
        / "wideep_generation_moe_perf.txt",
    }
    candidate = preferred.get(filename, source_dir / filename)
    return candidate if candidate.exists() else source_dir / filename


def _semantic_key_columns(filename: str) -> list[str]:
    if filename == "moe_perf.txt":
        return ["op_name", "phase", "moe_ep_size", "distribution", "num_tokens"]
    return ["op_name", "moe_ep_size", "distribution", "num_tokens"]


def _latency_layer_rows(path: Path, *, filename: str, layer: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = pd.read_csv(path)
    if rows.empty:
        return pd.DataFrame()
    if filename == "moe_perf.txt":
        mask = _expected_clean_mask(filename, rows)
    elif filename == "wideep_context_moe_perf.txt":
        mask = _expected_clean_mask(filename, rows)
    elif filename == "wideep_generation_moe_perf.txt":
        mask = _expected_clean_mask(filename, rows)
    else:
        mask = pd.Series(False, index=rows.index)
    rows = rows[mask].copy()
    if rows.empty:
        return pd.DataFrame()
    key_cols = _semantic_key_columns(filename)
    available = [col for col in key_cols if col in rows.columns]
    if len(available) != len(key_cols) or "latency" not in rows.columns:
        return pd.DataFrame()
    out = rows[key_cols + ["latency"]].copy()
    out["table"] = filename
    out["layer"] = layer
    out = out.rename(columns={"latency": f"{layer}_latency_ms"})
    duplicate_counts = out.groupby(key_cols, dropna=False).size().rename(f"{layer}_duplicate_rows")
    out = out.sort_values(key_cols + [f"{layer}_latency_ms"]).groupby(
        key_cols,
        dropna=False,
        as_index=False,
    ).agg(
        **{
            f"{layer}_latency_ms": (f"{layer}_latency_ms", "median"),
        }
    )
    out = out.merge(duplicate_counts.reset_index(), on=key_cols, how="left")
    out["table"] = filename
    return out


def _write_latency_layer_audit(source_dir: Path, candidate_dir: Path) -> None:
    parts = []
    for filename in ("moe_perf.txt", "wideep_context_moe_perf.txt", "wideep_generation_moe_perf.txt"):
        raw = _latency_layer_rows(
            source_dir / "raw_collector_source" / filename,
            filename=filename,
            layer="raw",
        )
        materialized = _latency_layer_rows(
            _materialized_table_source(source_dir, filename),
            filename=filename,
            layer="materialized",
        )
        final = _latency_layer_rows(candidate_dir / filename, filename=filename, layer="final")
        if raw.empty and materialized.empty and final.empty:
            continue
        key_cols = _semantic_key_columns(filename)
        merged = None
        for frame in (raw, materialized, final):
            if frame.empty:
                continue
            payload_cols = [
                col
                for col in frame.columns
                if col not in key_cols and col != "table"
            ]
            frame = frame[key_cols + payload_cols].copy()
            merged = frame if merged is None else merged.merge(frame, on=key_cols, how="outer")
        if merged is None:
            continue
        merged.insert(0, "table", filename)
        for left, right, out in (
            ("raw_latency_ms", "materialized_latency_ms", "materialized_vs_raw_pct"),
            ("materialized_latency_ms", "final_latency_ms", "final_vs_materialized_pct"),
            ("raw_latency_ms", "final_latency_ms", "final_vs_raw_pct"),
        ):
            if left in merged.columns and right in merged.columns:
                lhs = pd.to_numeric(merged[left], errors="coerce")
                rhs = pd.to_numeric(merged[right], errors="coerce")
                merged[out] = (rhs - lhs) / lhs * 100.0
        parts.append(merged)
    if parts:
        pd.concat(parts, ignore_index=True).to_csv(
            candidate_dir / "clean_latency_layer_audit.csv",
            index=False,
        )
    else:
        pd.DataFrame().to_csv(candidate_dir / "clean_latency_layer_audit.csv", index=False)


def _materialize_recorded_with_unified_materializer(
    source_dir: Path,
    candidate_dir: Path,
    input_manifest: dict[str, object],
) -> None:
    betas = default_betas()
    source_parts: list[pd.DataFrame] = []

    if input_manifest["ordinary_source_available"]:
        ordinary_sources = load_ordinary_sources(
            "collector",
            _materialized_table_source(source_dir, "moe_perf.txt"),
            source_dir / "raw_collector_source" / "moe_perf.txt",
        )
        source_parts.append(ordinary_sources)
        materialize_table(
            pd.read_csv(_materialized_table_source(source_dir, "moe_perf.txt")),
            ordinary_sources,
            ["ordinary_context", "ordinary_generation"],
            betas,
            candidate_dir / "moe_perf.txt",
            candidate_dir / "moe_perf_invalid_rows.csv",
            candidate_dir / "moe_perf_duplicate_source_keys.csv",
        )

    if input_manifest["wideep_context_available"]:
        wc_sources = load_wideep_context_sources(
            "collector",
            _materialized_table_source(source_dir, "wideep_context_moe_perf.txt"),
            source_dir / "raw_collector_source" / "wideep_context_moe_perf.txt",
        )
        source_parts.append(wc_sources)
        materialize_table(
            pd.read_csv(_materialized_table_source(source_dir, "wideep_context_moe_perf.txt")),
            wc_sources,
            ["wideep_context"],
            betas,
            candidate_dir / "wideep_context_moe_perf.txt",
            candidate_dir / "wideep_context_moe_perf_invalid_rows.csv",
            candidate_dir / "wideep_context_moe_perf_duplicate_source_keys.csv",
        )

    if input_manifest["wideep_generation_available"]:
        wg_sources = load_wideep_generation_sources(
            "collector",
            _materialized_table_source(source_dir, "wideep_generation_moe_perf.txt"),
            source_dir / "raw_collector_source" / "wideep_generation_moe_perf.txt",
        )
        source_parts.append(wg_sources)
        materialize_table(
            pd.read_csv(_materialized_table_source(source_dir, "wideep_generation_moe_perf.txt")),
            wg_sources,
            ["wideep_generation"],
            betas,
            candidate_dir / "wideep_generation_moe_perf.txt",
            candidate_dir / "wideep_generation_moe_perf_invalid_rows.csv",
            candidate_dir / "wideep_generation_moe_perf_duplicate_source_keys.csv",
        )

    if source_parts:
        pd.concat(source_parts, ignore_index=True).to_csv(
            candidate_dir / "recorded_moe_materializer_source_join.csv",
            index=False,
        )
    _write_latency_layer_audit(source_dir, candidate_dir)



def _write_origin_dir(candidate_dir: Path, origin_dir: Path) -> None:
    _copy_source_skeleton(candidate_dir, origin_dir)
    for filename in ("moe_perf.txt", "wideep_context_moe_perf.txt", "wideep_generation_moe_perf.txt"):
        if not (candidate_dir / filename).exists():
            continue
        rows = _read_csv(candidate_dir / filename)
        fieldnames = list(rows[0].keys())
        out = []
        for row in rows:
            row = dict(row)
            origin = row.get("origin_latency") or row.get("latency", "")
            if origin:
                row["latency"] = origin
            out.append(row)
        _write_csv(origin_dir / filename, out, fieldnames)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="AIC collector output directory.")
    parser.add_argument("--candidate-dir", type=Path, required=True, help="Output directory for clean latency tables.")
    parser.add_argument(
        "--write-origin-dir",
        action="store_true",
        help="Also write a sibling *_origin_latency directory with original latency restored.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_clean_latency_tables(
        source_dir=args.source_dir,
        candidate_dir=args.candidate_dir,
        write_origin_dir=args.write_origin_dir,
    )
    print(f"Wrote clean latency candidate: {args.candidate_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
