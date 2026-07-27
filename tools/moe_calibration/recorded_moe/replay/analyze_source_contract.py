#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit recorded MoE source contracts from a manifest.

This tool is offline-only.  It verifies that recorded source tables can be
loaded by the collector materializer, optionally compares final materialized
latency against frozen truth, and writes source/schema diagnostics.  It does
not write collector perf tables and never uses truth to materialize latency.

Manifest sketch:

source_inputs:
  - platform: h20
    family_group: ordinary
    final: collector/.../ordinary_moe_materialized_source/moe_perf.txt
    raw: collector/.../raw_collector_source/moe_perf.txt
    snapshot: baseline
  - platform: h20
    family_group: wideep_context
    final: collector/.../recorded_materialized_source/wideep_context_moe_perf.txt
    raw: collector/.../raw_collector_source/wideep_context_moe_perf.txt
truth_points: results/.../four_modes_latest_truth_summary.csv
output_dir: results/recorded_moe_source_contract
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from collector.moe_recorded_materializer import (  # noqa: E402
    MODEL_SPECS,
    default_betas,
    load_ordinary_sources,
    load_wideep_context_sources,
    load_wideep_generation_sources,
    predict,
)


KEY = ["platform", "family", "ep", "eplb", "token"]
LOADERS = {
    "ordinary": load_ordinary_sources,
    "wideep_context": load_wideep_context_sources,
    "wideep_generation": load_wideep_generation_sources,
}


def _read_manifest(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    if yaml is None:
        raise RuntimeError("PyYAML is required for YAML manifests")
    return yaml.safe_load(text)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _required_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column not in df.columns]


def _load_source(spec: dict[str, Any]) -> pd.DataFrame:
    family_group = str(spec["family_group"])
    if family_group not in LOADERS:
        raise ValueError(f"unknown family_group={family_group!r}")
    platform = str(spec["platform"])
    final_path = _resolve(spec["final"])
    raw_path = _resolve(spec["raw"])
    source = LOADERS[family_group](platform, final_path, raw_path)
    source["source_label"] = spec.get("label", f"{platform}:{family_group}")
    source["source_snapshot"] = spec.get("snapshot", "")
    source["source_final_path"] = str(final_path)
    source["source_raw_path"] = str(raw_path)
    return source


def _attach_materializer_prediction(source: pd.DataFrame) -> pd.DataFrame:
    betas = default_betas()
    parts: list[pd.DataFrame] = []
    for family in sorted(set(source["family"]) & set(MODEL_SPECS)):
        pred, _ = predict(source, family, betas[family])
        valid = pred.dropna()
        if valid.empty:
            continue
        part = source.loc[valid.index].copy()
        part["existing_pred_us"] = valid.to_numpy()
        part["materializer_version"] = MODEL_SPECS[family]["version"]
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _schema_summary(source: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (platform, family, snapshot), group in source.groupby(
        ["platform", "family", "source_snapshot"], dropna=False
    ):
        raw_cols = [col for col in group.columns if col.startswith("raw_")]
        rows.append(
            {
                "platform": platform,
                "family": family,
                "snapshot": snapshot,
                "rows": len(group),
                "columns": len(group.columns),
                "raw_columns": len(raw_cols),
                "has_current_latency_ms": "current_latency_ms" in group.columns,
                "has_raw_latency_us": "raw_latency_us" in group.columns,
                "has_existing_pred_us": "existing_pred_us" in group.columns,
            }
        )
    return pd.DataFrame(rows)


def _source_diff(source: pd.DataFrame) -> pd.DataFrame:
    snapshots = sorted(s for s in source["source_snapshot"].dropna().unique() if s)
    if len(snapshots) < 2:
        return pd.DataFrame()
    left_name, right_name = snapshots[0], snapshots[-1]
    cols = sorted(
        col
        for col in source.columns
        if col.startswith("raw_") and pd.api.types.is_numeric_dtype(source[col])
    )
    if not cols:
        return pd.DataFrame()
    left = source[source["source_snapshot"].eq(left_name)][KEY + cols]
    right = source[source["source_snapshot"].eq(right_name)][KEY + cols]
    merged = left.merge(right, on=KEY, suffixes=(f"_{left_name}", f"_{right_name}"))
    for col in cols:
        old = pd.to_numeric(merged[f"{col}_{left_name}"], errors="coerce")
        new = pd.to_numeric(merged[f"{col}_{right_name}"], errors="coerce")
        merged[f"{col}_delta_pct"] = (new - old) / old.replace(0, np.nan) * 100.0
    return merged


def _summarize_error(points: pd.DataFrame, value_col: str, ape_col: str) -> pd.DataFrame:
    rows = []
    for (platform, family), group in points.groupby(["platform", "family"], dropna=False):
        if group.empty:
            continue
        idx = group[ape_col].idxmax()
        worst = group.loc[idx]
        rows.append(
            {
                "platform": platform,
                "family": family,
                "points": len(group),
                "mape_pct": float(group[ape_col].mean()),
                "median_ape_pct": float(group[ape_col].median()),
                "max_ape_pct": float(worst[ape_col]),
                "max_ep": int(worst["ep"]),
                "max_eplb": worst["eplb"],
                "max_token": int(worst["token"]),
                "truth_us": float(worst["truth_median_us"]),
                value_col: float(worst[value_col]),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    manifest = _read_manifest(args.manifest)
    output_dir = args.output_dir or _resolve(manifest["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = pd.concat(
        [_load_source(spec) for spec in manifest["source_inputs"]],
        ignore_index=True,
    )
    predicted = _attach_materializer_prediction(sources)
    predicted.to_csv(output_dir / "recorded_source_with_materializer_prediction.csv", index=False)
    _schema_summary(predicted if not predicted.empty else sources).to_csv(
        output_dir / "source_schema_summary.csv",
        index=False,
    )
    diff = _source_diff(predicted if not predicted.empty else sources)
    if not diff.empty:
        diff.to_csv(output_dir / "source_snapshot_diff.csv", index=False)

    missing_key_cols = _required_columns(sources, KEY)
    if missing_key_cols:
        raise RuntimeError(f"source tables missing key columns: {missing_key_cols}")

    truth_path = manifest.get("truth_points")
    if truth_path:
        truth = pd.read_csv(_resolve(truth_path))[KEY + ["truth_median_us", "stability"]]
        points = predicted.merge(truth, on=KEY, how="inner")
        points["existing_ape_pct"] = (
            (points["existing_pred_us"] - points["truth_median_us"]).abs()
            / points["truth_median_us"]
            * 100.0
        )
        points.to_csv(output_dir / "materializer_vs_truth_points.csv", index=False)
        _summarize_error(points, "existing_pred_us", "existing_ape_pct").to_csv(
            output_dir / "materializer_vs_truth_summary.csv",
            index=False,
        )

    manifest_out = {
        "manifest": str(args.manifest),
        "source_inputs": len(manifest["source_inputs"]),
        "source_rows": int(len(sources)),
        "predicted_rows": int(len(predicted)),
        "output_dir": str(output_dir),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest_out, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
