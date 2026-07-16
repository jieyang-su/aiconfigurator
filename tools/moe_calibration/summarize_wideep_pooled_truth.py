#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build pooled-median WideEP truth summaries from multi-session profiles.

This summarizes the raw effective rows across sessions directly:

* generation: normally 3 sessions * 5 repetitions = 15 values per point
* context: normally 3 sessions * 3 samples = 9 values per point

It intentionally does not rewrite the historical parsed CSVs.  The output is a
candidate frozen-truth view plus diagnostics for session-median vs pooled-median
differences and rank-tail spread.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


csv.field_size_limit(sys.maxsize)

DATASETS = ("sharegpt", "longbench")
EPS = (2, 4, 8)
EPLBS = ("off", "on")
METRICS = ("rank_min_us", "rank_mean_us", "rank_max_us")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def median(values: list[float]) -> float:
    return statistics.median(sorted(values))


def quartiles(values: list[float]) -> tuple[float, float]:
    if len(values) < 4:
        return min(values), max(values)
    q = statistics.quantiles(sorted(values), n=4, method="inclusive")
    return q[0], q[2]


def phase_layout(phase: str) -> tuple[str, str]:
    if phase == "generation":
        return "wideep_generation_low_latency", "generation_dense_refresh_eplb_{eplb}_rank_aggregate.csv"
    if phase == "context":
        return "wideep_context", "context_dense_refresh_eplb_{eplb}_rank_aggregate.csv"
    raise ValueError(f"unsupported phase: {phase}")


def collect_phase_rows(root: Path, phase: str) -> list[dict[str, object]]:
    family, filename_template = phase_layout(phase)
    rows: list[dict[str, object]] = []

    for dataset in DATASETS:
        for ep in EPS:
            ep_dir = root / dataset / family / f"ep{ep}"
            if not ep_dir.exists():
                continue
            for eplb in EPLBS:
                values_by_token: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
                for session_dir in sorted(path for path in ep_dir.glob("session*") if path.is_dir()):
                    path = session_dir / "parsed" / filename_template.format(eplb=eplb)
                    for row in read_csv(path):
                        token = int(float(row["num_tokens"]))
                        for metric in METRICS:
                            values_by_token[token][metric].append(float(row[metric]))

                for token, values_by_metric in sorted(values_by_token.items()):
                    rank_max_values = values_by_metric["rank_max_us"]
                    rank_mean_values = values_by_metric["rank_mean_us"]
                    rank_min_values = values_by_metric["rank_min_us"]
                    if not rank_max_values:
                        continue
                    rank_max_median = median(rank_max_values)
                    q1, q3 = quartiles(rank_max_values)
                    rank_max_spread = (
                        (max(rank_max_values) - min(rank_max_values)) / rank_max_median * 100.0
                        if rank_max_median
                        else 0.0
                    )
                    rank_max_iqr = (q3 - q1) / rank_max_median * 100.0 if rank_max_median else 0.0
                    slow_count = sum(value > rank_max_median * 1.25 for value in rank_max_values)
                    rows.append(
                        {
                            "phase": phase,
                            "dataset": dataset,
                            "moe_ep_size": ep,
                            "eplb": eplb,
                            "num_tokens": token,
                            "n": len(rank_max_values),
                            "rank_max_pooled_median_us": f"{rank_max_median:.3f}",
                            "rank_max_min_us": f"{min(rank_max_values):.3f}",
                            "rank_max_max_us": f"{max(rank_max_values):.3f}",
                            "rank_max_spread_pct": f"{rank_max_spread:.2f}",
                            "rank_max_iqr_pct": f"{rank_max_iqr:.2f}",
                            "rank_max_slow_gt_1p25x_median": slow_count,
                            "rank_mean_pooled_median_us": f"{median(rank_mean_values):.3f}",
                            "rank_min_pooled_median_us": f"{median(rank_min_values):.3f}",
                        }
                    )

    return rows


def compare_session_vs_pooled(root: Path, pooled_generation_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    session_summary_path = root / "wideep_generation_session_median_summary.csv"
    session_rows = read_csv(session_summary_path)
    if not session_rows:
        return []

    pooled_by_key = {
        (
            str(row["dataset"]),
            str(row["moe_ep_size"]),
            str(row["eplb"]),
            str(row["num_tokens"]),
        ): row
        for row in pooled_generation_rows
    }
    out: list[dict[str, object]] = []
    for row in session_rows:
        key = (row["dataset"], row["moe_ep_size"], row["eplb"], row["num_tokens"])
        pooled = pooled_by_key.get(key)
        if not pooled:
            continue
        old_value = float(row["rank_max_final_median_us"])
        pooled_value = float(pooled["rank_max_pooled_median_us"])
        delta_pct = (pooled_value - old_value) / old_value * 100.0 if old_value else 0.0
        out.append(
            {
                "dataset": row["dataset"],
                "moe_ep_size": row["moe_ep_size"],
                "eplb": row["eplb"],
                "num_tokens": row["num_tokens"],
                "stability": row["stability"],
                "rank_max_spread_pct": row["rank_max_spread_pct"],
                "rank_max_session_medians_us": row["rank_max_session_medians_us"],
                "session_median_final_us": f"{old_value:.3f}",
                "pooled_median_us": f"{pooled_value:.3f}",
                "delta_pct": f"{delta_pct:.2f}",
                "pooled_iqr_pct": pooled["rank_max_iqr_pct"],
                "pooled_slow_count": pooled["rank_max_slow_gt_1p25x_median"],
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Full multi-session truth root.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to ROOT/pooled_median_truth_candidate.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or (args.root / "pooled_median_truth_candidate")
    output_dir.mkdir(parents=True, exist_ok=True)

    fields = [
        "phase",
        "dataset",
        "moe_ep_size",
        "eplb",
        "num_tokens",
        "n",
        "rank_max_pooled_median_us",
        "rank_max_min_us",
        "rank_max_max_us",
        "rank_max_spread_pct",
        "rank_max_iqr_pct",
        "rank_max_slow_gt_1p25x_median",
        "rank_mean_pooled_median_us",
        "rank_min_pooled_median_us",
    ]

    context_rows = collect_phase_rows(args.root, "context")
    generation_rows = collect_phase_rows(args.root, "generation")
    write_csv(output_dir / "wideep_context_pooled_median_summary.csv", context_rows, fields)
    write_csv(output_dir / "wideep_generation_pooled_median_summary.csv", generation_rows, fields)

    compare_rows = compare_session_vs_pooled(args.root, generation_rows)
    if compare_rows:
        write_csv(
            output_dir / "wideep_generation_session_vs_pooled_median_compare.csv",
            compare_rows,
            [
                "dataset",
                "moe_ep_size",
                "eplb",
                "num_tokens",
                "stability",
                "rank_max_spread_pct",
                "rank_max_session_medians_us",
                "session_median_final_us",
                "pooled_median_us",
                "delta_pct",
                "pooled_iqr_pct",
                "pooled_slow_count",
            ],
        )

    print(f"Wrote {output_dir / 'wideep_context_pooled_median_summary.csv'}")
    print(f"Wrote {output_dir / 'wideep_generation_pooled_median_summary.csv'}")
    if compare_rows:
        print(f"Wrote {output_dir / 'wideep_generation_session_vs_pooled_median_compare.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
