#!/usr/bin/env python3
"""Evaluate recorded MoE clean-latency outputs against frozen truth.

This verifier is final-output-only: it reads the already-written collector
``*.txt`` latency tables and never recomputes predictions through the offline
tools materializer.  That keeps the gate aligned with the collector runtime
path being validated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from collector.moe_recorded_materializer import (  # noqa: E402
    ORDINARY_COMMON_FILTER,
    WIDEEP_CONTEXT_COMMON_FILTER,
    WIDEEP_GENERATION_COMMON_FILTER,
    passes_filter,
)


KEY = ["platform", "family", "phase", "ep", "eplb", "token"]
RECORDED_DISTS = ["recorded_eplb", "recorded_no_eplb"]


def _eplb(distribution: pd.Series) -> pd.Series:
    return distribution.map({"recorded_eplb": "on", "recorded_no_eplb": "off"})


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _parse_clean_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"clean input must be platform=path, got {value!r}"
        )
    platform, path = value.split("=", 1)
    platform = platform.strip()
    if not platform:
        raise argparse.ArgumentTypeError(f"missing platform in {value!r}")
    return platform, Path(path)


def _target_rows(platform: str, clean_dir: Path) -> pd.DataFrame:
    parts = []
    ordinary = clean_dir / "moe_perf.txt"
    if ordinary.exists():
        parts.extend(
            [
                _ordinary_rows(platform, ordinary, "ordinary_context", "context"),
                _ordinary_rows(platform, ordinary, "ordinary_generation", "generation"),
            ]
        )
    wideep_context = clean_dir / "wideep_context_moe_perf.txt"
    if wideep_context.exists():
        parts.append(_wideep_rows(platform, wideep_context, "wideep_context", "context"))
    wideep_generation = clean_dir / "wideep_generation_moe_perf.txt"
    if wideep_generation.exists():
        parts.append(
            _wideep_rows(platform, wideep_generation, "wideep_generation", "generation")
        )
    if not parts:
        raise FileNotFoundError(f"no recorded MoE compact tables found in {clean_dir}")
    out = pd.concat(parts, ignore_index=True)
    return out


def _ordinary_rows(platform: str, path: Path, family: str, phase: str) -> pd.DataFrame:
    df = pd.read_csv(_require(path))
    mask = (
        passes_filter(df, ORDINARY_COMMON_FILTER)
        & df["phase"].eq(phase)
        & df["distribution"].isin(RECORDED_DISTS)
    )
    return _standardize(df.loc[mask], platform=platform, family=family, phase=phase)


def _wideep_rows(platform: str, path: Path, family: str, phase: str) -> pd.DataFrame:
    df = pd.read_csv(_require(path))
    common = WIDEEP_CONTEXT_COMMON_FILTER if family == "wideep_context" else WIDEEP_GENERATION_COMMON_FILTER
    mask = passes_filter(df, common) & df["distribution"].isin(RECORDED_DISTS)
    return _standardize(df.loc[mask], platform=platform, family=family, phase=phase)


def _standardize(df: pd.DataFrame, *, platform: str, family: str, phase: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "platform": platform,
            "family": family,
            "phase": phase,
            "ep": pd.to_numeric(df["moe_ep_size"], errors="coerce").astype("Int64"),
            "eplb": _eplb(df["distribution"]),
            "token": pd.to_numeric(df["num_tokens"], errors="coerce").astype("Int64"),
            "candidate_latency_us": pd.to_numeric(df["latency"], errors="coerce") * 1000.0,
            "aic_latency_source": df.get("aic_latency_source", ""),
            "aic_latency_policy": df.get("aic_latency_policy", ""),
            "kernel_source": df.get("kernel_source", ""),
            "aic_critical_path_latency": df.get("aic_critical_path_latency", ""),
        }
    )
    return out.dropna(subset=["ep", "eplb", "token", "candidate_latency_us"])


def _duplicates(candidates: pd.DataFrame) -> pd.DataFrame:
    counts = candidates.groupby(KEY, dropna=False).size().reset_index(name="rows")
    dup_keys = counts[counts["rows"] > 1]
    if dup_keys.empty:
        return pd.DataFrame(columns=[*KEY, "rows"])
    return candidates.merge(dup_keys[KEY], on=KEY, how="inner").sort_values(KEY)


def _summarize(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_cols in (
        [],
        ["platform"],
        ["family"],
        ["platform", "family"],
        ["platform", "family", "phase"],
    ):
        grouped = [((), points)] if not group_cols else points.groupby(group_cols, dropna=False)
        for keys, group in grouped:
            if group.empty:
                continue
            if not isinstance(keys, tuple):
                keys = (keys,)
            idx = group["abs_error_pct"].idxmax()
            worst = group.loc[idx]
            rows.append(
                {
                    "scope": "all" if not group_cols else "+".join(group_cols),
                    **dict(zip(group_cols, keys)),
                    "points": int(len(group)),
                    "mape_pct": float(group["abs_error_pct"].mean()),
                    "median_ape_pct": float(group["abs_error_pct"].median()),
                    "p90_ape_pct": float(group["abs_error_pct"].quantile(0.9)),
                    "max_ape_pct": float(worst["abs_error_pct"]),
                    "max_platform": worst["platform"],
                    "max_family": worst["family"],
                    "max_phase": worst["phase"],
                    "max_ep": int(worst["ep"]),
                    "max_eplb": worst["eplb"],
                    "max_token": int(worst["token"]),
                    "truth_us": float(worst["truth_median_us"]),
                    "pred_us": float(worst["candidate_latency_us"]),
                    "signed_pct": float(worst["error_pct"]),
                    "aic_latency_policy": worst.get("aic_latency_policy", ""),
                    "aic_latency_source": worst.get("aic_latency_source", ""),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth-points", type=Path, required=True)
    parser.add_argument(
        "--clean",
        action="append",
        type=_parse_clean_spec,
        required=True,
        metavar="PLATFORM=DIR",
        help="Recorded compact AIC output directory for a platform. Repeatable.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-error-pct", type=float, default=20.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    truth = pd.read_csv(_require(args.truth_points))[KEY + ["truth_median_us", "stability"]].copy()
    candidates = pd.concat(
        [_target_rows(platform, clean_dir) for platform, clean_dir in args.clean],
        ignore_index=True,
    )
    duplicates = _duplicates(candidates)
    if not duplicates.empty:
        duplicates.to_csv(args.output_dir / "collector_clean_latency_duplicate_candidate_rows.csv", index=False)
        candidates = candidates.sort_values(KEY).drop_duplicates(KEY, keep="first")

    points = truth.merge(candidates, on=KEY, how="left")
    missing = points[points["candidate_latency_us"].isna()].copy()
    valid = points.dropna(subset=["candidate_latency_us"]).copy()
    valid["error_pct"] = (
        (valid["candidate_latency_us"] - valid["truth_median_us"]) / valid["truth_median_us"] * 100.0
    )
    valid["abs_error_pct"] = valid["error_pct"].abs()
    over_threshold = valid[valid["abs_error_pct"] > args.max_error_pct].copy()
    extra = candidates.merge(truth[KEY], on=KEY, how="left", indicator=True)
    extra = extra[extra["_merge"].eq("left_only")].drop(columns=["_merge"])

    summary = _summarize(valid)
    valid.to_csv(args.output_dir / "collector_clean_latency_gate_points.csv", index=False)
    missing.to_csv(args.output_dir / "collector_clean_latency_missing_truth_points.csv", index=False)
    extra.to_csv(args.output_dir / "collector_clean_latency_extra_aic_points.csv", index=False)
    over_threshold.to_csv(args.output_dir / "collector_clean_latency_over_threshold.csv", index=False)
    summary.to_csv(args.output_dir / "collector_clean_latency_gate_summary.csv", index=False)

    manifest = {
        "truth_points": str(args.truth_points),
        "clean_inputs": {platform: str(clean_dir) for platform, clean_dir in args.clean},
        "output_dir": str(args.output_dir),
        "candidate_rows": int(len(candidates)),
        "valid_points": int(len(valid)),
        "missing_truth_points": int(len(missing)),
        "extra_aic_points": int(len(extra)),
        "duplicate_candidate_rows": int(len(duplicates)),
        "max_error_pct_threshold": float(args.max_error_pct),
        "rows_over_threshold": int(len(over_threshold)),
        "final_output_only": True,
        "materializer_import": "collector.moe_recorded_materializer filters only; no prediction recompute",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(summary[summary["scope"].eq("platform+family")].sort_values(["platform", "family"]).to_string(index=False))
    print("\nOverall:")
    print(summary[summary["scope"].eq("all")].to_string(index=False))
    print(f"\nmissing_truth_points={len(missing)} extra_aic_points={len(extra)} duplicate_candidate_rows={len(duplicates)}")
    print(f"rows_over_{args.max_error_pct:g}pct={len(over_threshold)}")
    try:
        rel_out = args.output_dir.relative_to(REPO_ROOT)
    except ValueError:
        rel_out = args.output_dir
    print(f"Wrote {rel_out}")
    return 1 if len(duplicates) or len(over_threshold) else 0


if __name__ == "__main__":
    raise SystemExit(main())
