#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Select diagnostic WideEP generation semantic truth candidates.

This is a truth-parser validation helper.  It does not read AIC data and does
not update frozen truth files.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import re
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _infer_ep(path: Path) -> int:
    for part in path.parts:
        if part.startswith("ep") and ("_off_" in part or "_on_" in part):
            return int(part.split("_", 1)[0].removeprefix("ep"))
        if part.startswith("ep") and part[2:].isdigit():
            return int(part[2:])
        match = re.fullmatch(r"ep(?P<ep>\d+)(?:[_-].*)?", part)
        if match:
            return int(match.group("ep"))
    raise RuntimeError(f"Could not infer EP size from {path}")


def _f(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value else float("nan")


def _values(value: str) -> list[float]:
    return [float(item) for item in value.split(";") if item.strip()]


def _spread_pct(values: list[float]) -> float:
    if not values:
        return float("inf")
    mean = sum(values) / len(values)
    if mean == 0.0:
        return 0.0 if max(values) == 0.0 and min(values) == 0.0 else float("inf")
    return (max(values) - min(values)) / mean * 100.0


def _best_low_cluster(
    values: list[float],
    *,
    min_size: int,
    spread_threshold: float,
) -> tuple[tuple[int, ...], float, float] | None:
    best: tuple[tuple[int, ...], float, float] | None = None
    for size in range(len(values), min_size - 1, -1):
        for indexes in itertools.combinations(range(len(values)), size):
            cluster_values = [values[index] for index in indexes]
            spread = _spread_pct(cluster_values)
            if spread > spread_threshold:
                continue
            mean = sum(cluster_values) / len(cluster_values)
            candidate = (indexes, mean, spread)
            if best is None or len(indexes) > len(best[0]) or (
                len(indexes) == len(best[0]) and mean < best[1]
            ):
                best = candidate
        if best is not None and len(best[0]) == size:
            return best
    return best


def _collect(root: Path) -> dict[tuple[int, str, int], dict[str, Any]]:
    rows: dict[tuple[int, str, int], dict[str, Any]] = {}
    for path in sorted(root.rglob("wideep_generation_trace_window_stability_summary.csv")):
        ep = _infer_ep(path)
        for row in _read_csv(path):
            eplb = row["eplb"].replace("eplb_", "")
            token = int(row["tokens"])
            key = (ep, eplb, token)
            candidate = {
                "moe_ep_size": ep,
                "eplb_norm": eplb,
                "num_tokens": token,
                "sessions": int(row["sessions"]),
                "annotation_us": _f(row, "annotation_us_rank_max_us_median"),
                "annotation_spread_pct": _f(row, "annotation_us_rank_max_us_spread_pct"),
                "window_strict_compute_us": _f(row, "window_compute_kernel_union_us_rank_max_us_median"),
                "window_strict_compute_values": row["window_compute_kernel_union_us_rank_max_us_values"],
                "window_strict_compute_spread_pct": _f(
                    row, "window_compute_kernel_union_us_rank_max_us_spread_pct"
                ),
                "window_semantic_compute_us": _f(
                    row, "window_semantic_compute_kernel_union_us_rank_max_us_median"
                ),
                "window_semantic_compute_values": row[
                    "window_semantic_compute_kernel_union_us_rank_max_us_values"
                ],
                "window_semantic_compute_spread_pct": _f(
                    row, "window_semantic_compute_kernel_union_us_rank_max_us_spread_pct"
                ),
                "external_id_compute_us": _f(row, "kernel_external_id_compute_union_us_rank_max_us_median"),
                "external_id_compute_values": row["kernel_external_id_compute_union_us_rank_max_us_values"],
                "external_id_compute_spread_pct": _f(
                    row, "kernel_external_id_compute_union_us_rank_max_us_spread_pct"
                ),
                "all_kernel_us": _f(row, "window_all_kernel_union_us_rank_max_us_median"),
                "source_file": str(path),
            }
            old = rows.get(key)
            if old is None or candidate["sessions"] > old["sessions"]:
                rows[key] = candidate
    return rows


def _choose(row: dict[str, Any], compute_spread_threshold: float, external_spread_threshold: float) -> None:
    strict_ok = (
        row["window_strict_compute_us"] > 0.0
        and row["window_strict_compute_spread_pct"] <= compute_spread_threshold
    )
    external_bounded = row["external_id_compute_us"] <= row["annotation_us"] * 1.05
    external_ok = (
        row["external_id_compute_us"] > 0.0
        and row["external_id_compute_spread_pct"] <= external_spread_threshold
        and external_bounded
    )
    strict_values = _values(row["window_strict_compute_values"])
    if strict_ok:
        row["selected_truth_us"] = row["window_strict_compute_us"]
        row["selected_source"] = "strict_window_compute"
        row["selected_note"] = "primary_strict_compute_3session_stable"
    elif row["sessions"] >= 5:
        cluster = _best_low_cluster(
            strict_values,
            min_size=3,
            spread_threshold=compute_spread_threshold,
        )
        if cluster is not None:
            indexes, mean, spread = cluster
            row["selected_truth_us"] = mean
            row["selected_source"] = "strict_compute_stable_cluster"
            row["selected_note"] = (
                f"stable_low_cluster_after_refill:indexes={indexes};spread_pct={spread:.3f}"
            )
        elif external_ok:
            row["selected_truth_us"] = row["external_id_compute_us"]
            row["selected_source"] = "external_id_compute"
            row["selected_note"] = "profile_window_visibility_fallback"
        else:
            row["selected_truth_us"] = ""
            row["selected_source"] = "profile_window_unstable"
            row["selected_note"] = "no_stable_strict_compute_cluster_after_refill"
    elif _best_low_cluster(strict_values, min_size=2, spread_threshold=compute_spread_threshold):
        cluster = _best_low_cluster(strict_values, min_size=2, spread_threshold=compute_spread_threshold)
        assert cluster is not None
        indexes, mean, spread = cluster
        row["selected_truth_us"] = ""
        row["selected_source"] = "needs_refill_2of3_cluster"
        row["selected_note"] = f"candidate_mean_us={mean:.6f};indexes={indexes};spread_pct={spread:.3f}"
    elif external_ok:
        row["selected_truth_us"] = row["external_id_compute_us"]
        row["selected_source"] = "external_id_compute"
        row["selected_note"] = "profile_window_visibility_fallback"
    else:
        row["selected_truth_us"] = ""
        row["selected_source"] = "profile_window_unstable"
        row["selected_note"] = "needs_refill_or_trace_audit"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-root", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--semantic-spread-threshold", type=float, default=10.0)
    parser.add_argument("--compute-spread-threshold", type=float, default=None)
    parser.add_argument("--external-spread-threshold", type=float, default=5.0)
    args = parser.parse_args()

    merged: dict[tuple[int, str, int], dict[str, Any]] = {}
    for root in args.profile_root:
        for key, row in _collect(root).items():
            old = merged.get(key)
            if old is None or row["sessions"] > old["sessions"]:
                merged[key] = row

    rows = [merged[key] for key in sorted(merged)]
    compute_spread_threshold = (
        args.semantic_spread_threshold if args.compute_spread_threshold is None else args.compute_spread_threshold
    )
    for row in rows:
        _choose(row, compute_spread_threshold, args.external_spread_threshold)
    _write_csv(args.out, rows)
    print(f"rows={len(rows)}")
    print(f"out={args.out}")


if __name__ == "__main__":
    main()
