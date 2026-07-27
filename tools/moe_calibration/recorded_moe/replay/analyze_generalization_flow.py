#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replay recorded MoE generalization gates from a manifest.

This tool compares one or more candidate latency sets against frozen truth.
Recorded candidates can be loaded from compact AIC collector directories.
Historical baselines such as balanced, power_law, or uniform can be supplied as
already-standardized point CSVs.  The tool does not modify collector outputs.

Manifest sketch:

truth_points: results/.../four_modes_latest_truth_summary.csv
output_dir: results/recorded_moe_generalization_replay
recorded_clean_inputs:
  - platform: h20
    path: collector/moe+moe_token_distribution+wideep_moe_...
  - platform: new_hw
    path: collector/moe+moe_token_distribution+wideep_moe_...
baseline_points:
  - name: balanced
    path: results/balanced_points.csv
    latency_col: candidate_latency_us
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.moe_calibration.recorded_moe.gate.evaluate_clean_latency import (  # noqa: E402
    KEY,
    _target_rows,
)


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


def _load_truth(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)[KEY + ["truth_median_us", "stability"]].copy()


def _recorded_candidates(inputs: list[dict[str, Any]]) -> pd.DataFrame:
    parts = []
    for spec in inputs:
        platform = str(spec["platform"])
        clean_dir = _resolve(spec["path"])
        part = _target_rows(platform, clean_dir)
        part["candidate_set"] = spec.get("name", "recorded")
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _baseline_candidates(inputs: list[dict[str, Any]]) -> pd.DataFrame:
    parts = []
    for spec in inputs:
        path = _resolve(spec["path"])
        latency_col = spec.get("latency_col", "candidate_latency_us")
        df = pd.read_csv(path)
        missing = [col for col in KEY + [latency_col] if col not in df.columns]
        if missing:
            raise RuntimeError(f"{path} missing required columns: {missing}")
        part = df[KEY + [latency_col]].copy()
        part = part.rename(columns={latency_col: "candidate_latency_us"})
        part["candidate_set"] = spec["name"]
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _summarize(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groupings = [
        ["candidate_set"],
        ["candidate_set", "platform"],
        ["candidate_set", "platform", "family"],
    ]
    for cols in groupings:
        for keys, group in points.groupby(cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            idx = group["abs_error_pct"].idxmax()
            worst = group.loc[idx]
            rows.append(
                {
                    **dict(zip(cols, keys)),
                    "points": len(group),
                    "mape_pct": float(group["abs_error_pct"].mean()),
                    "median_ape_pct": float(group["abs_error_pct"].median()),
                    "p90_ape_pct": float(group["abs_error_pct"].quantile(0.9)),
                    "max_ape_pct": float(worst["abs_error_pct"]),
                    "max_family": worst["family"],
                    "max_phase": worst["phase"],
                    "max_ep": int(worst["ep"]),
                    "max_eplb": worst["eplb"],
                    "max_token": int(worst["token"]),
                    "truth_us": float(worst["truth_median_us"]),
                    "pred_us": float(worst["candidate_latency_us"]),
                    "signed_pct": float(worst["error_pct"]),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-error-pct", type=float, default=25.0)
    args = parser.parse_args()

    manifest = _read_manifest(args.manifest)
    output_dir = args.output_dir or _resolve(manifest["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    truth = _load_truth(_resolve(manifest["truth_points"]))
    candidates = pd.concat(
        [
            _recorded_candidates(manifest.get("recorded_clean_inputs", [])),
            _baseline_candidates(manifest.get("baseline_points", [])),
        ],
        ignore_index=True,
    )
    if candidates.empty:
        raise RuntimeError("manifest did not provide any recorded or baseline candidates")

    points = truth.merge(candidates, on=KEY, how="left")
    missing = points[points["candidate_latency_us"].isna()].copy()
    valid = points.dropna(subset=["candidate_latency_us"]).copy()
    valid["error_pct"] = (
        (valid["candidate_latency_us"] - valid["truth_median_us"])
        / valid["truth_median_us"]
        * 100.0
    )
    valid["abs_error_pct"] = valid["error_pct"].abs()
    summary = _summarize(valid)
    over_threshold = valid[valid["abs_error_pct"] > args.max_error_pct].copy()

    valid.to_csv(output_dir / "generalization_replay_points.csv", index=False)
    summary.to_csv(output_dir / "generalization_replay_summary.csv", index=False)
    missing.to_csv(output_dir / "generalization_replay_missing_candidates.csv", index=False)
    over_threshold.to_csv(output_dir / "generalization_replay_over_threshold.csv", index=False)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "truth_points": manifest["truth_points"],
                "candidate_rows": int(len(candidates)),
                "valid_points": int(len(valid)),
                "missing_candidates": int(len(missing)),
                "rows_over_threshold": int(len(over_threshold)),
                "max_error_pct": float(args.max_error_pct),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(summary.sort_values(["candidate_set", "mape_pct"]).to_string(index=False))
    print(f"\nrows_over_{args.max_error_pct:g}pct={len(over_threshold)}")
    print(f"Wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
