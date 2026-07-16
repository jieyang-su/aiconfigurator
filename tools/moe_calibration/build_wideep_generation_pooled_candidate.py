#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build an offline WideEP generation candidate from pooled H20 truth.

The candidate intentionally keeps context unchanged and applies only a thin
scale to generation recorded rows.  Scales are fitted on ShareGPT exact table
hits against a selected pooled median truth metric.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path


csv.field_size_limit(sys.maxsize)

SHAPE = {
    "moe_dtype": "fp8_block",
    "hidden_size": "7168",
    "inter_size": "2048",
    "topk": "8",
    "num_experts": "256",
}

RECORDED_TO_EPLB = {
    "recorded_no_eplb": "off",
    "recorded_eplb": "on",
}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def shape_matches(row: dict[str, str]) -> bool:
    return all(str(row.get(column, "")) == expected for column, expected in SHAPE.items())


def load_sharegpt_truth(path: Path, *, truth_metric: str) -> dict[tuple[int, str, int], float]:
    column = {
        "rank_max": "rank_max_pooled_median_us",
        "rank_mean": "rank_mean_pooled_median_us",
        "rank_min": "rank_min_pooled_median_us",
    }[truth_metric]
    truth: dict[tuple[int, str, int], float] = {}
    for row in read_csv(path)[1]:
        if row["dataset"] != "sharegpt":
            continue
        ep = int(row["moe_ep_size"])
        eplb = row["eplb"]
        token = int(row["num_tokens"])
        truth[(ep, eplb, token)] = float(row[column]) / 1000.0
    return truth


def fit_scales(rows: list[dict[str, str]], truth: dict[tuple[int, str, int], float]) -> dict[tuple[int, str], float]:
    ratios: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        if not shape_matches(row):
            continue
        distribution = row.get("distribution", "")
        if distribution not in RECORDED_TO_EPLB:
            continue
        ep = int(float(row["moe_ep_size"]))
        token = int(float(row["num_tokens"]))
        eplb = RECORDED_TO_EPLB[distribution]
        truth_latency = truth.get((ep, eplb, token))
        if truth_latency is None:
            continue
        current_latency = float(row["latency"])
        if current_latency <= 0:
            continue
        ratios[(ep, eplb)].append(truth_latency / current_latency)

    return {key: statistics.median(values) for key, values in ratios.items() if values}


def token_bucket(token: int, *, scheme: str) -> str:
    if token <= 8:
        return "tiny_le8"
    if token <= 128:
        return "small_32_128"
    if token <= 288:
        return "mid_288"
    if scheme == "split_tail":
        if token <= 512:
            return "tail_512"
        return "tail_ge896"
    return "tail_ge512"


def fit_bucket_scales(
    rows: list[dict[str, str]],
    truth: dict[tuple[int, str, int], float],
    base_scales: dict[tuple[int, str], float],
    *,
    bucket_scheme: str,
) -> dict[tuple[int, str, str], float]:
    ratios: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if not shape_matches(row):
            continue
        distribution = row.get("distribution", "")
        if distribution not in RECORDED_TO_EPLB:
            continue
        ep = int(float(row["moe_ep_size"]))
        token = int(float(row["num_tokens"]))
        eplb = RECORDED_TO_EPLB[distribution]
        truth_latency = truth.get((ep, eplb, token))
        base_scale = base_scales.get((ep, eplb))
        if truth_latency is None or base_scale is None:
            continue
        scaled_latency = float(row["latency"]) * base_scale
        if scaled_latency <= 0:
            continue
        ratios[(ep, eplb, token_bucket(token, scheme=bucket_scheme))].append(truth_latency / scaled_latency)
    return {key: statistics.median(values) for key, values in ratios.items() if values}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--truth-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--token-bucket-scale",
        action="store_true",
        help="Also fit a coarse EP/EPLB/token-bucket residual scale on ShareGPT exact hits.",
    )
    parser.add_argument(
        "--truth-metric",
        choices=["rank_max", "rank_mean", "rank_min"],
        default="rank_max",
        help="Pooled truth metric used to fit generation scales.",
    )
    parser.add_argument(
        "--bucket-scheme",
        choices=["coarse", "split_tail"],
        default="coarse",
        help="Token bucket scheme used with --token-bucket-scale.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for filename in ("moe_perf.txt", "wideep_context_moe_perf.txt"):
        source = args.data_dir / filename
        if source.exists():
            shutil.copy2(source, args.output_dir / filename)

    generation_path = args.data_dir / "wideep_generation_moe_perf.txt"
    fields, rows = read_csv(generation_path)
    truth = load_sharegpt_truth(
        args.truth_dir / "wideep_generation_pooled_median_summary.csv",
        truth_metric=args.truth_metric,
    )
    scales = fit_scales(rows, truth)
    bucket_scales = (
        fit_bucket_scales(rows, truth, scales, bucket_scheme=args.bucket_scheme)
        if args.token_bucket_scale
        else {}
    )

    candidate_rows: list[dict[str, str]] = []
    scale_report_rows: list[dict[str, str]] = []
    for (ep, eplb), scale in sorted(scales.items()):
        scale_report_rows.append(
            {
                "moe_ep_size": str(ep),
                "eplb": eplb,
                "scale": f"{scale:.8f}",
            }
        )
    bucket_report_rows: list[dict[str, str]] = []
    for (ep, eplb, bucket), scale in sorted(bucket_scales.items()):
        bucket_report_rows.append(
            {
                "moe_ep_size": str(ep),
                "eplb": eplb,
                "token_bucket": bucket,
                "scale": f"{scale:.8f}",
            }
        )

    for row in rows:
        out = dict(row)
        distribution = row.get("distribution", "")
        if shape_matches(row) and distribution in RECORDED_TO_EPLB:
            ep = int(float(row["moe_ep_size"]))
            token = int(float(row["num_tokens"]))
            eplb = RECORDED_TO_EPLB[distribution]
            scale = scales.get((ep, eplb))
            if scale is not None:
                bucket_scale = bucket_scales.get((ep, eplb, token_bucket(token, scheme=args.bucket_scheme)), 1.0)
                out["latency"] = f"{float(row['latency']) * scale * bucket_scale:.12g}"
        candidate_rows.append(out)

    write_csv(args.output_dir / "wideep_generation_moe_perf.txt", fields, candidate_rows)
    write_csv(args.output_dir / "wideep_generation_pooled_scale_report.csv", ["moe_ep_size", "eplb", "scale"], scale_report_rows)
    if bucket_report_rows:
        write_csv(
            args.output_dir / "wideep_generation_pooled_bucket_scale_report.csv",
            ["moe_ep_size", "eplb", "token_bucket", "scale"],
            bucket_report_rows,
        )

    print(f"Wrote {args.output_dir}")
    print(f"Wrote {args.output_dir / 'wideep_generation_pooled_scale_report.csv'}")
    if bucket_report_rows:
        print(f"Wrote {args.output_dir / 'wideep_generation_pooled_bucket_scale_report.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
