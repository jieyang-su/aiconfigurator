#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""MoE distribution calibration helpers.

This utility is intentionally offline: it consumes existing AIC collector
parquet tables plus SGLang expert-distribution/profile CSVs and writes new
calibration artifacts.  It does not mutate the repository's checked-in system
data unless the caller points an output path there.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ExpertLoadSignature:
    num_tokens: int
    layer_id: int
    num_experts: int
    total_assignments: float
    active_experts: int
    max_assignments: float
    mean_assignments: float
    std_assignments: float
    cv: float | None
    max_over_mean: float | None
    nonzero_mean_assignments: float
    max_over_nonzero_mean: float | None
    recommended_distribution: str
    reason: str


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit(f"No rows to write to {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _float_or_none(value: str | float | None) -> float | None:
    if value is None or value == "":
        return None
    parsed = float(value)
    if math.isnan(parsed):
        return None
    return parsed


def _float_or_zero(value: str | float | None) -> float:
    parsed = _float_or_none(value)
    return 0.0 if parsed is None else parsed


def _token_from_total_assignments(total_assignments: float, topk: int) -> int:
    if topk <= 0:
        raise ValueError("topk must be positive")
    return int(round(total_assignments / topk))


def _recommend_distribution(
    cv: float | None,
    max_over_mean: float | None,
    available_distributions: set[str],
) -> tuple[str, str]:
    if cv is None or max_over_mean is None:
        return "unknown", "没有有效路由样本，不能判断分布"
    if cv < 0.20 and max_over_mean < 1.50:
        return "uniform", f"cv={cv:.3f}, max_over_mean={max_over_mean:.3f}，接近均匀分布"
    recorded_names = sorted(name for name in available_distributions if name.startswith("recorded"))
    if recorded_names:
        preferred = "recorded" if "recorded" in recorded_names else recorded_names[0]
        return preferred, f"已有 {preferred}，可直接使用真实路由分布校准表"
    power_law_names = sorted(name for name in available_distributions if name.startswith("power_law"))
    if power_law_names:
        return power_law_names[-1], f"分布明显偏斜但没有 recorded 表，先选择最偏斜可用表 {power_law_names[-1]}"
    return "collect_required:recorded_or_power_law", "分布明显偏斜，当前只有 uniform，必须补采 recorded 或 power_law 表"


def _available_distributions(table_path: Path) -> set[str]:
    df = pd.read_parquet(table_path)
    if "distribution" not in df.columns:
        return set()
    return {str(value) for value in df["distribution"].dropna().unique()}


def _read_perf_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_perf_table(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No rows to write to {path}")
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _write_sibling_txt_for_parquet(path: Path, df: pd.DataFrame) -> Path | None:
    if path.suffix.lower() != ".parquet":
        return None
    txt_path = path.with_suffix(".txt")
    df.to_csv(txt_path, index=False)
    return txt_path


def _distribution_rows_from_summary(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = _read_csv(Path(args.summary_csv))
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        total_assignments = _float_or_zero(row.get("total_assignments"))
        if total_assignments <= 0:
            continue
        num_tokens = int(row.get("num_tokens") or _token_from_total_assignments(total_assignments, args.topk))
        if args.num_tokens and num_tokens not in args.num_tokens:
            continue
        output_rows.append(
            {
                "framework": args.framework,
                "version": args.version,
                "device": args.device,
                "op_name": "moe_token_distribution",
                "kernel_source": args.kernel_source,
                "model": args.model,
                "phase": args.phase,
                "distribution": args.distribution,
                "num_tokens": num_tokens,
                "layer_id": int(row["layer_id"]),
                "topk": args.topk,
                "num_experts": int(float(row.get("num_experts") or args.num_experts)),
                "total_assignments": total_assignments,
                "active_experts": int(float(row["active_experts"])),
                "max_assignments": _float_or_zero(row.get("max_assignments")),
                "mean_assignments": _float_or_zero(row.get("mean_assignments")),
                "std_assignments": _float_or_zero(row.get("std_assignments")),
                "cv": _float_or_zero(row.get("cv")),
                "max_over_mean": _float_or_zero(row.get("max_over_mean")),
                "nonzero_mean_assignments": _float_or_zero(row.get("nonzero_mean_assignments")),
                "max_over_nonzero_mean": _float_or_zero(row.get("max_over_nonzero_mean")),
                # Reserved for a future full-vector collector.  Keep the column
                # now so downstream consumers do not need a schema break.
                "expert_assignments_json": row.get("expert_assignments_json", ""),
            }
        )
    return output_rows


def build_distribution_table(args: argparse.Namespace) -> None:
    """Materialize model MoE token distribution as AIC perf-style data."""
    rows = _distribution_rows_from_summary(args)
    output_txt = Path(args.output_txt)
    output_parquet = Path(args.output_parquet) if args.output_parquet else output_txt.with_suffix(".parquet")
    _write_perf_table(output_txt, rows)
    _write_perf_table(output_parquet, rows)
    print(f"Wrote MoE token distribution TXT to {output_txt}")
    print(f"Wrote MoE token distribution parquet to {output_parquet}")


def build_recorded_table(args: argparse.Namespace) -> None:
    base_table = Path(args.base_table)
    output_table = Path(args.output_table)
    stage_rows = _read_csv(Path(args.rank_stage_csv))
    df = pd.read_parquet(base_table)
    shape_mask = (
        (df["hidden_size"] == args.hidden_size)
        & (df["inter_size"] == args.inter_size)
        & (df["topk"] == args.topk)
        & (df["num_experts"] == args.num_experts)
        & (df["moe_tp_size"] == args.moe_tp_size)
        & (df["moe_ep_size"] == args.moe_ep_size)
        & (df["moe_dtype"].astype(str) == args.moe_dtype)
    )
    template_rows = df[shape_mask].copy()
    if template_rows.empty:
        raise SystemExit(f"No matching template rows in {base_table}")

    recorded_rows: list[pd.Series] = []
    compare_rows: list[dict[str, Any]] = []
    for row in stage_rows:
        if row.get("stage") != args.stage:
            continue
        num_tokens = int(row["num_tokens"])
        if args.num_tokens and num_tokens not in args.num_tokens:
            continue
        token_templates = template_rows[template_rows["num_tokens"] == num_tokens]
        if token_templates.empty:
            raise SystemExit(f"No template row for num_tokens={num_tokens} in {base_table}")
        base = token_templates.iloc[0].copy()
        sglang_total_us = float(row[args.duration_column])
        sglang_per_layer_ms = sglang_total_us / args.num_profiled_moe_layers / 1000.0
        uniform_ms = float(base["latency"])
        base["distribution"] = args.output_distribution
        base["latency"] = sglang_per_layer_ms
        recorded_rows.append(base)
        compare_rows.append(
            {
                "num_tokens": num_tokens,
                "stage": args.stage,
                "sglang_rankmax_total_us": sglang_total_us,
                "num_profiled_moe_layers": args.num_profiled_moe_layers,
                "sglang_rankmax_per_layer_ms": sglang_per_layer_ms,
                "uniform_aic_ms": uniform_ms,
                "recorded_aic_ms": sglang_per_layer_ms,
                "uniform_over_sglang": uniform_ms / sglang_per_layer_ms,
                "recorded_abs_error_ms": 0.0,
                "recorded_rel_error_pct": 0.0,
            }
        )

    if not recorded_rows:
        raise SystemExit(f"No stage rows matched stage={args.stage!r}")
    recorded_df = pd.DataFrame(recorded_rows)
    merged = pd.concat([df[df["distribution"] != args.output_distribution], recorded_df], ignore_index=True)
    output_table.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_table, index=False)
    _write_csv(Path(args.output_compare_csv), compare_rows)
    print(f"Wrote recorded table to {output_table}")
    print(f"Wrote short-term comparison to {args.output_compare_csv}")


def _distribution_token_stats(distribution_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        distribution_df.groupby(["distribution", "num_tokens"], as_index=False)
        .agg(
            layers=("layer_id", "count"),
            active_experts_mean=("active_experts", "mean"),
            cv_mean=("cv", "mean"),
            max_over_mean_mean=("max_over_mean", "mean"),
            max_assignments_mean=("max_assignments", "mean"),
        )
        .sort_values(["distribution", "num_tokens"])
    )
    return grouped


def _select_template_row(template_rows: pd.DataFrame, num_tokens: int, base_table: Path) -> pd.Series:
    token_templates = template_rows[template_rows["num_tokens"] == num_tokens]
    if not token_templates.empty:
        return token_templates.iloc[0].copy()

    if "num_tokens" not in template_rows.columns or template_rows.empty:
        raise SystemExit(f"No template rows in {base_table}")

    nearest_idx = (template_rows["num_tokens"].astype(int) - num_tokens).abs().idxmin()
    base = template_rows.loc[nearest_idx].copy()
    base["num_tokens"] = num_tokens
    return base


def materialize_wideep_from_distribution(args: argparse.Namespace) -> None:
    """Generate AIC WideEP MoE silicon rows from stored distribution data.

    The distribution table is the durable model-level collector output.  The
    stage CSV provides measured silicon latency for the same token points.  The
    resulting WideEP table is still ordinary silicon data, so AIC SILICON/HYBRID
    interpolation/extrapolation paths remain unchanged and do not use empirical
    fallback for these rows.
    """
    base_table = Path(args.base_table)
    output_table = Path(args.output_table)
    dist_df = _read_perf_table(Path(args.distribution_table))
    if "distribution" not in dist_df.columns or "num_tokens" not in dist_df.columns:
        raise SystemExit("Distribution table must contain distribution and num_tokens columns")

    token_stats = _distribution_token_stats(dist_df)
    if args.distribution not in set(token_stats["distribution"].astype(str)):
        raise SystemExit(f"Distribution {args.distribution!r} not found in {args.distribution_table}")

    df = pd.read_parquet(base_table)
    shape_mask = (
        (df["hidden_size"] == args.hidden_size)
        & (df["inter_size"] == args.inter_size)
        & (df["topk"] == args.topk)
        & (df["num_experts"] == args.num_experts)
        & (df["moe_tp_size"] == args.moe_tp_size)
        & (df["moe_ep_size"] == args.moe_ep_size)
        & (df["moe_dtype"].astype(str) == args.moe_dtype)
    )
    template_rows = df[shape_mask].copy()
    if template_rows.empty:
        raise SystemExit(f"No matching template rows in {base_table}")

    stage_by_token = {
        int(row["num_tokens"]): row
        for row in _read_csv(Path(args.rank_stage_csv))
        if row.get("stage") == args.stage
    }

    recorded_rows: list[pd.Series] = []
    compare_rows: list[dict[str, Any]] = []
    for stat in token_stats[token_stats["distribution"] == args.distribution].to_dict("records"):
        num_tokens = int(stat["num_tokens"])
        if args.num_tokens and num_tokens not in args.num_tokens:
            continue
        if num_tokens not in stage_by_token:
            raise SystemExit(f"No measured stage row for num_tokens={num_tokens} in {args.rank_stage_csv}")

        base = _select_template_row(template_rows, num_tokens, base_table)
        stage_row = stage_by_token[num_tokens]
        sglang_total_us = float(stage_row[args.duration_column])
        sglang_per_layer_ms = sglang_total_us / args.num_profiled_moe_layers / 1000.0
        output_distribution = args.output_distribution or args.distribution
        base["distribution"] = output_distribution
        base["latency"] = sglang_per_layer_ms
        base["kernel_source"] = "silicon_from_moe_token_distribution"
        recorded_rows.append(base)
        compare_rows.append(
            {
                "num_tokens": num_tokens,
                "source_distribution": args.distribution,
                "output_distribution": output_distribution,
                "layers_in_distribution_table": int(stat["layers"]),
                "active_experts_mean": stat["active_experts_mean"],
                "cv_mean": stat["cv_mean"],
                "max_over_mean_mean": stat["max_over_mean_mean"],
                "sglang_rankmax_total_us": sglang_total_us,
                "num_profiled_moe_layers": args.num_profiled_moe_layers,
                "materialized_single_layer_ms": sglang_per_layer_ms,
                "source": "silicon_from_moe_token_distribution",
            }
        )

    if not recorded_rows:
        raise SystemExit(f"No materialized rows for distribution={args.distribution!r}")
    recorded_df = pd.DataFrame(recorded_rows)
    output_distribution = args.output_distribution or args.distribution
    replace_mask = (
        shape_mask
        & (df["distribution"].astype(str) == output_distribution)
        & df["num_tokens"].isin(recorded_df["num_tokens"].astype(int).tolist())
    )
    frames = [df[~replace_mask]]
    if args.archive_existing_output_distribution_as:
        archive_rows = df[replace_mask].copy()
        if not archive_rows.empty:
            archive_rows["distribution"] = args.archive_existing_output_distribution_as
            frames.append(archive_rows)
    frames.append(recorded_df)
    merged = pd.concat(frames, ignore_index=True)
    output_table.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_table, index=False)
    txt_path = _write_sibling_txt_for_parquet(output_table, merged)
    _write_csv(Path(args.output_compare_csv), compare_rows)
    print(f"Wrote materialized WideEP MoE table to {output_table}")
    if txt_path is not None:
        print(f"Wrote materialized WideEP MoE TXT to {txt_path}")
    print(f"Wrote materialization summary to {args.output_compare_csv}")


def select_distribution(args: argparse.Namespace) -> None:
    available = _available_distributions(Path(args.table))
    rows = _read_csv(Path(args.summary_csv))
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        total_assignments = _float_or_zero(row.get("total_assignments"))
        if total_assignments <= 0:
            continue
        num_tokens = _token_from_total_assignments(total_assignments, args.topk)
        if args.num_tokens and num_tokens not in args.num_tokens:
            continue
        cv = _float_or_none(row.get("cv"))
        max_over_mean = _float_or_none(row.get("max_over_mean"))
        distribution, reason = _recommend_distribution(cv, max_over_mean, available)
        output_rows.append(
            {
                "num_tokens": num_tokens,
                "layer_id": int(row["layer_id"]),
                "active_experts": int(float(row["active_experts"])),
                "num_experts": int(float(row["num_experts"])),
                "cv": "" if cv is None else cv,
                "max_over_mean": "" if max_over_mean is None else max_over_mean,
                "available_distributions": ";".join(sorted(available)),
                "recommended_distribution": distribution,
                "reason": reason,
            }
        )
    _write_csv(Path(args.output), output_rows)
    print(f"Wrote distribution selection rows to {args.output}")


def export_signatures(args: argparse.Namespace) -> None:
    available = _available_distributions(Path(args.table))
    rows = _read_csv(Path(args.summary_csv))
    signatures: list[ExpertLoadSignature] = []
    for row in rows:
        total_assignments = _float_or_zero(row.get("total_assignments"))
        if total_assignments <= 0:
            continue
        num_tokens = _token_from_total_assignments(total_assignments, args.topk)
        if args.num_tokens and num_tokens not in args.num_tokens:
            continue
        cv = _float_or_none(row.get("cv"))
        max_over_mean = _float_or_none(row.get("max_over_mean"))
        distribution, reason = _recommend_distribution(cv, max_over_mean, available)
        signatures.append(
            ExpertLoadSignature(
                num_tokens=num_tokens,
                layer_id=int(row["layer_id"]),
                num_experts=int(float(row["num_experts"])),
                total_assignments=total_assignments,
                active_experts=int(float(row["active_experts"])),
                max_assignments=_float_or_zero(row.get("max_assignments")),
                mean_assignments=_float_or_zero(row.get("mean_assignments")),
                std_assignments=_float_or_zero(row.get("std_assignments")),
                cv=cv,
                max_over_mean=max_over_mean,
                nonzero_mean_assignments=_float_or_zero(row.get("nonzero_mean_assignments")),
                max_over_nonzero_mean=_float_or_none(row.get("max_over_nonzero_mean")),
                recommended_distribution=distribution,
                reason=reason,
            )
        )

    payload = {
        "schema": "aic.moe.expert_load_signature.v1",
        "model": args.model,
        "system": args.system,
        "backend": args.backend,
        "topk": args.topk,
        "source_summary_csv": str(Path(args.summary_csv)),
        "signatures": [asdict(signature) for signature in signatures],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote expert-load signatures to {output}")


def make_overlay(args: argparse.Namespace) -> None:
    source = Path(args.source_overlay)
    target = Path(args.target_overlay)
    if target.exists() and not args.force:
        raise SystemExit(f"Target overlay already exists: {target}. Use --force to overwrite.")
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    print(f"Copied overlay {source} -> {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    overlay = sub.add_parser("make-overlay")
    overlay.add_argument("--source-overlay", required=True)
    overlay.add_argument("--target-overlay", required=True)
    overlay.add_argument("--force", action="store_true")
    overlay.set_defaults(func=make_overlay)

    dist_table = sub.add_parser("build-distribution-table")
    dist_table.add_argument("--summary-csv", required=True)
    dist_table.add_argument("--output-txt", required=True)
    dist_table.add_argument("--output-parquet", default=None)
    dist_table.add_argument("--framework", default="SGLang")
    dist_table.add_argument("--version", default="")
    dist_table.add_argument("--device", default="")
    dist_table.add_argument("--kernel-source", default="expert_distribution_recorder")
    dist_table.add_argument("--model", default="deepseek-v3")
    dist_table.add_argument("--phase", choices=["context", "generation"], default="context")
    dist_table.add_argument("--distribution", default="recorded_rankprobe")
    dist_table.add_argument("--topk", type=int, default=8)
    dist_table.add_argument("--num-experts", type=int, default=256)
    dist_table.add_argument("--num-tokens", type=int, nargs="*", default=[])
    dist_table.set_defaults(func=build_distribution_table)

    recorded = sub.add_parser("build-recorded-table")
    recorded.add_argument("--base-table", required=True)
    recorded.add_argument("--rank-stage-csv", required=True)
    recorded.add_argument("--output-table", required=True)
    recorded.add_argument("--output-compare-csv", required=True)
    recorded.add_argument("--output-distribution", default="recorded_rankprobe")
    recorded.add_argument("--stage", default="routed/compute")
    recorded.add_argument("--duration-column", default="max_rank_cuda_us")
    recorded.add_argument("--num-profiled-moe-layers", type=int, default=3)
    recorded.add_argument("--num-tokens", type=int, nargs="*", default=[])
    recorded.add_argument("--hidden-size", type=int, default=7168)
    recorded.add_argument("--inter-size", type=int, default=2048)
    recorded.add_argument("--topk", type=int, default=8)
    recorded.add_argument("--num-experts", type=int, default=256)
    recorded.add_argument("--moe-tp-size", type=int, default=1)
    recorded.add_argument("--moe-ep-size", type=int, default=2)
    recorded.add_argument("--moe-dtype", default="fp8_w8a8")
    recorded.set_defaults(func=build_recorded_table)

    materialize = sub.add_parser("materialize-wideep-from-distribution")
    materialize.add_argument("--base-table", required=True)
    materialize.add_argument("--distribution-table", required=True)
    materialize.add_argument("--rank-stage-csv", required=True)
    materialize.add_argument("--output-table", required=True)
    materialize.add_argument("--output-compare-csv", required=True)
    materialize.add_argument("--distribution", default="recorded_rankprobe")
    materialize.add_argument(
        "--output-distribution",
        default=None,
        help="Distribution name written to the WideEP table. Defaults to --distribution.",
    )
    materialize.add_argument(
        "--archive-existing-output-distribution-as",
        default=None,
        help="Rename existing rows for --output-distribution before writing materialized rows.",
    )
    materialize.add_argument("--stage", default="routed/compute")
    materialize.add_argument("--duration-column", default="max_rank_cuda_us")
    materialize.add_argument("--num-profiled-moe-layers", type=int, default=3)
    materialize.add_argument("--num-tokens", type=int, nargs="*", default=[])
    materialize.add_argument("--hidden-size", type=int, default=7168)
    materialize.add_argument("--inter-size", type=int, default=2048)
    materialize.add_argument("--topk", type=int, default=8)
    materialize.add_argument("--num-experts", type=int, default=256)
    materialize.add_argument("--moe-tp-size", type=int, default=1)
    materialize.add_argument("--moe-ep-size", type=int, default=2)
    materialize.add_argument("--moe-dtype", default="fp8_w8a8")
    materialize.set_defaults(func=materialize_wideep_from_distribution)

    selector = sub.add_parser("select-distribution")
    selector.add_argument("--summary-csv", required=True)
    selector.add_argument("--table", required=True)
    selector.add_argument("--output", required=True)
    selector.add_argument("--topk", type=int, default=8)
    selector.add_argument("--num-tokens", type=int, nargs="*", default=[])
    selector.set_defaults(func=select_distribution)

    sig = sub.add_parser("export-signatures")
    sig.add_argument("--summary-csv", required=True)
    sig.add_argument("--table", required=True)
    sig.add_argument("--output", required=True)
    sig.add_argument("--model", default="deepseek-v3")
    sig.add_argument("--system", default="h20_sxm")
    sig.add_argument("--backend", default="sglang")
    sig.add_argument("--topk", type=int, default=8)
    sig.add_argument("--num-tokens", type=int, nargs="*", default=[])
    sig.set_defaults(func=export_signatures)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
