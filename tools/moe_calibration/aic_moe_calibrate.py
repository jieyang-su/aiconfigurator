#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Utilities for DeepSeekV3 MoE AIC-vs-SGLang calibration.

This script intentionally stays small and explicit. It does four jobs used by
the runbook:

1. Parse SGLang torch-profiler chrome traces containing ``aic_moe/...`` events.
2. Query AIC ``PerfDatabase.query_moe`` for matching DeepSeekV3 MoE shapes.
3. Query AIC DeepEP dispatch/combine tables for matching DeepSeekV3 MoE shapes.
4. Compare real trace aggregates against AIC predictions.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


TRACE_STAGE_RE = re.compile(r"^aic_moe/layer_(?P<layer_id>\d+)/(?P<stage>.+)$")
AIC_OP_STAGE_RE = re.compile(r"^aic_moe/shared/(?P<stage>.+)$")
NTOK_RE = re.compile(r"(?:^|[_-])(?:ntok|numtok|num_tokens)[_-]?(?P<ntok>\d+)(?:$|[_-])")
ISL_RE = re.compile(r"(?:^|[_-])isl[_-]?(?P<isl>\d+)(?:$|[_-])")
INPUT_RE = re.compile(r"(?:^|[_-])input(?P<input>\d+)(?:$|[_-])")


def _open_trace(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _infer_num_tokens(path: Path) -> int | None:
    text = str(path)
    for regex, group in ((NTOK_RE, "ntok"), (ISL_RE, "isl"), (INPUT_RE, "input")):
        match = regex.search(text)
        if match:
            return int(match.group(group))
    return None


def _infer_phase(path: Path) -> str:
    text = str(path).lower()
    if "decode" in text or "generation" in text:
        return "generation"
    return "context"


def _iter_trace_files(trace_root: Path) -> Iterable[Path]:
    yield from trace_root.rglob("*.trace.json")
    yield from trace_root.rglob("*.trace.json.gz")


def parse_trace(args: argparse.Namespace) -> None:
    trace_root = Path(args.trace_root)
    rows: list[dict[str, str | int | float]] = []
    missing_num_token_traces: set[str] = set()

    for trace_path in sorted(_iter_trace_files(trace_root)):
        run_name = trace_path.parent.name
        num_tokens = args.num_tokens or _infer_num_tokens(trace_path) or ""
        phase = args.phase or _infer_phase(trace_path)
        with _open_trace(trace_path) as f:
            payload = json.load(f)

        events = payload.get("traceEvents", [])
        kernels: list[tuple[float, float, str]] = []
        kernel_duration_by_external_id: dict[object, float] = {}
        for event in events:
            if event.get("ph") != "X" or event.get("cat") != "kernel":
                continue
            kernel_name = str(event.get("name", ""))
            kernel_start = float(event.get("ts", 0.0))
            kernel_end = kernel_start + float(event.get("dur", 0.0))
            kernels.append((kernel_start, kernel_end, kernel_name))
            external_id = (event.get("args") or {}).get("External id")
            if external_id is None:
                continue
            kernel_duration_by_external_id[external_id] = kernel_duration_by_external_id.get(external_id, 0.0) + float(
                event.get("dur", 0.0)
            )

        cpu_ops: list[tuple[float, object]] = []
        for event in events:
            if event.get("ph") != "X" or event.get("cat") != "cpu_op":
                continue
            external_id = (event.get("args") or {}).get("External id")
            if external_id is None:
                continue
            cpu_ops.append((float(event.get("ts", 0.0)), external_id))

        trace_rows: list[dict[str, str | int | float]] = []
        for event in events:
            if event.get("ph") != "X":
                continue
            # PyTorch emits the same record_function range both as the CPU
            # annotation and as GPU stream annotations.  Keep the CPU range as
            # the source of truth to avoid double-counting stages.
            if event.get("cat") != "user_annotation":
                continue
            name = event.get("name", "")
            match = TRACE_STAGE_RE.match(name)
            op_match = AIC_OP_STAGE_RE.match(name)
            if not match and not op_match:
                continue
            start_ts = float(event.get("ts", 0.0))
            end_ts = start_ts + float(event.get("dur", 0.0))
            external_ids = {external_id for ts, external_id in cpu_ops if start_ts <= ts <= end_ts}
            overlapping_kernels = [
                (kernel_start, kernel_end, kernel_name)
                for kernel_start, kernel_end, kernel_name in kernels
                if kernel_start < end_ts and kernel_end > start_ts
            ]
            trace_rows.append(
                {
                    "run": run_name,
                    "trace": str(trace_path),
                    "phase": phase,
                    "num_tokens": num_tokens,
                    "layer_id": int(match.group("layer_id")) if match else -1,
                    "stage": match.group("stage") if match else op_match.group("stage"),
                    "duration_us": float(event.get("dur", 0.0)),
                    "cuda_duration_us": sum(kernel_duration_by_external_id.get(external_id, 0.0) for external_id in external_ids),
                    "cuda_window_duration_us": sum(kernel_end - kernel_start for kernel_start, kernel_end, _ in overlapping_kernels),
                    "deepep_window_duration_us": sum(
                        kernel_end - kernel_start
                        for kernel_start, kernel_end, kernel_name in overlapping_kernels
                        if "deep_ep::" in kernel_name
                    ),
                    "cpu_op_count": len(external_ids),
                    "ts": float(event.get("ts", 0.0)),
                    "pid": event.get("pid", ""),
                    "tid": event.get("tid", ""),
                    "event_category": event.get("cat", ""),
                }
            )
        if trace_rows and num_tokens == "":
            missing_num_token_traces.add(str(trace_path))
        rows.extend(trace_rows)

    if not rows:
        raise SystemExit(f"No aic_moe events found under {trace_root}")
    if missing_num_token_traces:
        sample = "\n".join(sorted(missing_num_token_traces)[:5])
        raise SystemExit(
            "Could not infer num_tokens for one or more traces. "
            "Use --num-tokens for a single-shape parse or include ntok_/isl_/input in profile paths.\n"
            f"Sample traces:\n{sample}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run",
        "trace",
        "phase",
        "num_tokens",
        "layer_id",
        "stage",
        "duration_us",
        "cuda_duration_us",
        "cuda_window_duration_us",
        "deepep_window_duration_us",
        "cpu_op_count",
        "ts",
        "pid",
        "tid",
        "event_category",
    ]
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} trace events to {output}")


def _pr_value(result) -> tuple[float, float, str]:
    latency_ms = float(result)
    energy = float(getattr(result, "energy", 0.0))
    source = str(getattr(result, "source", "unknown"))
    return latency_ms, energy, source


def _make_database(args: argparse.Namespace):
    from aiconfigurator.sdk import common
    from aiconfigurator.sdk.perf_database import PerfDatabase

    database = PerfDatabase(
        system=args.system,
        backend=args.backend,
        version=args.backend_version,
        systems_root=args.systems_root,
        database_mode=args.database_mode,
    )
    database.set_default_database_mode(common.DatabaseMode[args.database_mode])
    return database


def query_aic(args: argparse.Namespace) -> None:
    from aiconfigurator.sdk import common

    database = _make_database(args)

    quant_mode = common.MoEQuantMode[args.quant_mode]
    rows: list[dict[str, str | int | float | bool]] = []
    for num_tokens in args.num_tokens:
        for distribution in args.distribution:
            result = database.query_moe(
                num_tokens=num_tokens,
                hidden_size=args.hidden_size,
                inter_size=args.inter_size,
                topk=args.topk,
                num_experts=args.num_experts,
                moe_tp_size=args.moe_tp_size,
                moe_ep_size=args.moe_ep_size,
                quant_mode=quant_mode,
                workload_distribution=distribution,
                is_context=args.phase == "context",
                moe_backend=args.moe_backend,
                is_gated=not args.non_gated,
                enable_eplb=args.enable_eplb,
            )
            latency_ms, energy, source = _pr_value(result)
            rows.append(
                {
                    "num_tokens": num_tokens,
                    "phase": args.phase,
                    "system": args.system,
                    "backend": args.backend,
                    "backend_version": args.backend_version,
                    "database_mode": args.database_mode,
                    "quant_mode": args.quant_mode,
                    "distribution": distribution,
                    "hidden_size": args.hidden_size,
                    "inter_size": args.inter_size,
                    "topk": args.topk,
                    "num_experts": args.num_experts,
                    "moe_tp_size": args.moe_tp_size,
                    "moe_ep_size": args.moe_ep_size,
                    "moe_backend": args.moe_backend or "",
                    "enable_eplb": args.enable_eplb,
                    "aic_latency_ms": latency_ms,
                    "aic_latency_us": latency_ms * 1000.0,
                    "aic_energy": energy,
                    "aic_source": source,
                }
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} AIC predictions to {output}")


def query_deepep_dispatch(args: argparse.Namespace) -> None:
    database = _make_database(args)

    rows: list[dict[str, str | int | float]] = []
    for num_tokens in args.num_tokens:
        if args.deepep_mode == "normal":
            result = database.query_wideep_deepep_normal(
                node_num=args.node_num,
                num_tokens=num_tokens,
                num_experts=args.num_experts,
                topk=args.topk,
                hidden_size=args.hidden_size,
                sms=args.sms,
            )
        else:
            result = database.query_wideep_deepep_ll(
                node_num=args.node_num,
                num_tokens=num_tokens,
                num_experts=args.num_experts,
                topk=args.topk,
                hidden_size=args.hidden_size,
            )
        latency_ms, energy, source = _pr_value(result)
        rows.append(
            {
                "num_tokens": num_tokens,
                "phase": args.phase,
                "system": args.system,
                "backend": args.backend,
                "backend_version": args.backend_version,
                "database_mode": args.database_mode,
                "distribution": args.distribution,
                "operation": f"deepep_{args.deepep_mode}_dispatch_combine",
                "node_num": args.node_num,
                "sms": args.sms if args.deepep_mode == "normal" else "",
                "hidden_size": args.hidden_size,
                "topk": args.topk,
                "num_experts": args.num_experts,
                "aic_latency_ms": latency_ms,
                "aic_latency_us": latency_ms * 1000.0,
                "aic_energy": energy,
                "aic_source": source,
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} DeepEP dispatch predictions to {output}")


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _row_phase(row: dict[str, str]) -> str:
    return row.get("phase") or "context"


def _row_duration(row: dict[str, str], duration_column: str) -> float:
    if duration_column not in row:
        raise SystemExit(
            f"Trace CSV does not contain duration column {duration_column!r}. "
            "Use duration_us for CPU record_function ranges or a CSV with cuda_duration_us."
        )
    return float(row[duration_column])


def _stage_values(
    rows: list[dict[str, str]],
    stage_expr: str,
    duration_column: str = "duration_us",
) -> dict[str, list[float]]:
    stages = [stage.strip() for stage in stage_expr.split("+") if stage.strip()]
    if not stages:
        raise SystemExit(f"Invalid stage expression: {stage_expr!r}")

    if len(stages) == 1:
        groups: dict[str, list[float]] = {}
        stage = stages[0]
        for row in rows:
            if row["stage"] != stage:
                continue
            num_tokens = row.get("num_tokens") or ""
            if not num_tokens:
                continue
            groups.setdefault(num_tokens, []).append(_row_duration(row, duration_column))
        return groups

    partial: dict[
        tuple[str, str, str, str, str, str],
        dict[str, list[tuple[float, float]]],
    ] = {}
    for row in rows:
        stage = row["stage"]
        if stage not in stages:
            continue
        num_tokens = row.get("num_tokens") or ""
        if not num_tokens:
            continue
        key = (
            num_tokens,
            row.get("run", ""),
            row.get("trace", ""),
            row.get("layer_id", ""),
            row.get("pid", ""),
            row.get("tid", ""),
        )
        bucket = partial.setdefault(key, {})
        bucket.setdefault(stage, []).append(
            (float(row.get("ts") or 0.0), _row_duration(row, duration_column))
        )

    groups: dict[str, list[float]] = {}
    required = set(stages)
    for (
        num_tokens,
        _run,
        _trace,
        _layer_id,
        _pid,
        _tid,
    ), stage_values in partial.items():
        if not required.issubset(stage_values):
            continue
        sorted_stage_values = {
            stage: [duration for _ts, duration in sorted(values)]
            for stage, values in stage_values.items()
        }
        sample_count = min(len(sorted_stage_values[stage]) for stage in stages)
        for sample_index in range(sample_count):
            groups.setdefault(num_tokens, []).append(
                sum(sorted_stage_values[stage][sample_index] for stage in stages)
            )
    return groups


def self_test(args: argparse.Namespace) -> None:
    def row(pid: str, tid: str, stage: str, duration_us: float, ts: float):
        return {
            "num_tokens": "128",
            "run": "r",
            "trace": "t",
            "layer_id": "3",
            "pid": pid,
            "tid": tid,
            "stage": stage,
            "duration_us": str(duration_us),
            "ts": str(ts),
        }

    rows = [
        # Intentionally shuffled timestamps and interleaved pid/tid groups.
        row("1", "a", "routed/compute", 20, 50),
        row("2", "b", "topk", 3, 10),
        row("1", "a", "topk", 2, 40),
        row("1", "a", "topk", 1, 20),
        row("2", "b", "routed/compute", 30, 15),
        row("1", "a", "routed/compute", 10, 30),
        row("1", "a", "topk", 4, 60),
    ]
    composite = sorted(_stage_values(rows, "topk+routed/compute")["128"])
    if composite != [11.0, 22.0, 33.0]:
        raise SystemExit(f"Composite stage self-test failed: {composite}")
    single_stage = sorted(_stage_values(rows, "topk")["128"])
    if single_stage != [1.0, 2.0, 3.0, 4.0]:
        raise SystemExit(f"Single-stage self-test failed: {single_stage}")
    print("aic_moe_calibrate self-test passed")

def summarize_trace(args: argparse.Namespace) -> None:
    rows = _load_csv(Path(args.trace_csv))
    stage_exprs = args.stage or sorted({row["stage"] for row in rows})
    groups: dict[tuple[str, str, str], list[float]] = {}
    phases = sorted({_row_phase(row) for row in rows})
    for phase in phases:
        phase_rows = [row for row in rows if _row_phase(row) == phase]
        for stage_expr in stage_exprs:
            for num_tokens, vals in _stage_values(phase_rows, stage_expr, args.duration_column).items():
                groups[(phase, num_tokens, stage_expr)] = vals

    output_rows = []
    for (phase, num_tokens, stage), vals in sorted(
        groups.items(), key=lambda item: (item[0][0], int(item[0][1]), item[0][2])
    ):
        vals_sorted = sorted(vals)
        p90 = vals_sorted[int(0.9 * (len(vals_sorted) - 1))]
        output_rows.append(
            {
                "phase": phase,
                "num_tokens": num_tokens,
                "stage": stage,
                "samples": len(vals),
                "mean_us": statistics.mean(vals),
                "p50_us": statistics.median(vals),
                "p90_us": p90,
                "min_us": min(vals),
                "max_us": max(vals),
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output_rows:
        raise SystemExit("No summary rows. Check --trace-csv and optional --stage filters.")
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} trace summary rows to {output}")


def validate_trace(args: argparse.Namespace) -> None:
    rows = _load_csv(Path(args.trace_csv))
    required_stages = [stage.strip() for stage in args.required_stage if stage.strip()]
    optional_stages = [stage.strip() for stage in args.optional_stage if stage.strip()]
    all_stages = required_stages + optional_stages
    all_stage_set = set(all_stages)
    if not required_stages:
        raise SystemExit("At least one --required-stage is required.")

    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        phase = _row_phase(row)
        num_tokens = row.get("num_tokens") or ""
        stage = row.get("stage") or ""
        if not num_tokens or stage not in all_stage_set:
            continue
        grouped.setdefault((phase, num_tokens, stage), []).append(row)

    phase_token_values = sorted(
        {
            (_row_phase(row), row.get("num_tokens") or "")
            for row in rows
            if row.get("num_tokens")
        },
        key=lambda value: (value[0], int(value[1])),
    )
    output_rows = []
    has_error = False
    for phase, num_tokens in phase_token_values:
        for stage in all_stages:
            stage_rows = grouped.get((phase, num_tokens, stage), [])
            samples = len(stage_rows)
            layer_ids = sorted(
                {int(row["layer_id"]) for row in stage_rows if row.get("layer_id") != ""},
            )
            observed_layers = len(layer_ids)
            status = "ok"
            if stage in required_stages and samples == 0:
                status = "missing"
                has_error = True
            elif (
                stage in required_stages
                and args.expected_layers is not None
                and observed_layers < args.expected_layers
            ):
                status = "incomplete_layers"
                has_error = True
            elif stage in optional_stages and samples == 0:
                status = "optional_missing"
            elif (
                stage in optional_stages
                and args.expected_layers is not None
                and 0 < observed_layers < args.expected_layers
            ):
                status = "optional_incomplete_layers"
            output_rows.append(
                {
                    "phase": phase,
                    "num_tokens": num_tokens,
                    "stage": stage,
                    "required": stage in required_stages,
                    "samples": samples,
                    "observed_layers": observed_layers,
                    "expected_layers": args.expected_layers or "",
                    "layer_ids": " ".join(str(layer_id) for layer_id in layer_ids),
                    "status": status,
                }
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output_rows:
        raise SystemExit("No validation rows. Check --trace-csv and num_tokens inference.")
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    if has_error and args.fail_on_missing:
        raise SystemExit(f"Missing or incomplete required stages; wrote validation rows to {output}")
    print(f"Wrote {len(output_rows)} trace validation rows to {output}")


def breakdown_trace(args: argparse.Namespace) -> None:
    rows = _load_csv(Path(args.trace_csv))
    components = [stage.strip() for stage in args.component if stage.strip()]
    if not components:
        raise SystemExit("At least one --component stage is required.")

    modules = [row for row in rows if row["stage"] == args.module_stage]
    if not modules:
        raise SystemExit(f"No {args.module_stage!r} rows found in {args.trace_csv}")

    rows_by_scope: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            _row_phase(row),
            row.get("trace", ""),
            row.get("layer_id", ""),
            row.get("num_tokens", ""),
        )
        rows_by_scope.setdefault(key, []).append(row)

    output_rows = []
    component_set = set(components)
    for module in modules:
        key = (
            _row_phase(module),
            module.get("trace", ""),
            module.get("layer_id", ""),
            module.get("num_tokens", ""),
        )
        module_ts = float(module["ts"])
        module_duration_us = float(module["duration_us"])
        module_end = module_ts + module_duration_us

        component_totals = {component: 0.0 for component in components}
        for row in rows_by_scope.get(key, []):
            stage = row["stage"]
            if stage not in component_set:
                continue
            row_ts = float(row["ts"])
            row_end = row_ts + float(row["duration_us"])
            if module_ts <= row_ts and row_end <= module_end + args.time_epsilon_us:
                component_totals[stage] += float(row["duration_us"])

        component_sum_us = sum(component_totals.values())
        residual_us = module_duration_us - component_sum_us
        missing_components = [
            component for component, duration in component_totals.items() if duration == 0.0
        ]
        output_row = {
            "run": module.get("run", ""),
            "trace": module.get("trace", ""),
            "phase": _row_phase(module),
            "num_tokens": module.get("num_tokens", ""),
            "layer_id": module.get("layer_id", ""),
            "module_stage": args.module_stage,
            "module_duration_us": module_duration_us,
            "component_sum_us": component_sum_us,
            "residual_us": residual_us,
            "residual_pct": (
                residual_us / module_duration_us * 100.0 if module_duration_us else ""
            ),
            "missing_components": "+".join(missing_components),
        }
        for component in components:
            output_row[f"component_{component.replace('/', '_')}_us"] = (
                component_totals[component]
            )
        output_rows.append(output_row)

    output_rows.sort(
        key=lambda row: (
            str(row["phase"]),
            int(row["num_tokens"]) if row["num_tokens"] else -1,
            str(row["trace"]),
            int(row["layer_id"]) if row["layer_id"] else -1,
        )
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} module breakdown rows to {output}")

    if args.summary_output:
        summary_groups: dict[tuple[str, str], list[dict[str, str | float]]] = {}
        for row in output_rows:
            summary_groups.setdefault((str(row["phase"]), str(row["num_tokens"])), []).append(row)

        summary_rows = []
        for (phase, num_tokens), group in sorted(
            summary_groups.items(),
            key=lambda item: (item[0][0], int(item[0][1]) if item[0][1] else -1),
        ):
            residuals = [float(row["residual_us"]) for row in group]
            module_values = [float(row["module_duration_us"]) for row in group]
            component_sum_values = [float(row["component_sum_us"]) for row in group]
            module_mean_us = statistics.mean(module_values)
            residual_mean_us = statistics.mean(residuals)
            summary_row = {
                "phase": phase,
                "num_tokens": num_tokens,
                "samples": len(group),
                "module_mean_us": module_mean_us,
                "component_sum_mean_us": statistics.mean(component_sum_values),
                "residual_mean_us": residual_mean_us,
                "residual_p50_us": statistics.median(residuals),
                "residual_min_us": min(residuals),
                "residual_max_us": max(residuals),
                "residual_mean_pct": (
                    residual_mean_us / module_mean_us * 100.0
                    if module_mean_us
                    else ""
                ),
            }
            for component in components:
                key = f"component_{component.replace('/', '_')}_us"
                summary_row[f"{key}_mean"] = statistics.mean(
                    float(row[key]) for row in group
                )
            summary_rows.append(summary_row)

        summary_output = Path(args.summary_output)
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        with summary_output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Wrote {len(summary_rows)} module breakdown summary rows to {summary_output}")


def summarize_expert_distribution(args: argparse.Namespace) -> None:
    import torch

    rows = []
    for path in sorted(Path(args.input_dir).glob(args.pattern)):
        payload = torch.load(path, map_location="cpu")
        count = None
        count_kind = ""
        if isinstance(payload, dict) and "logical_count" in payload:
            count = payload["logical_count"]
            count_kind = "logical"
        elif isinstance(payload, dict) and "records" in payload:
            counts = [
                record.get("global_physical_count")
                for record in payload["records"]
                if isinstance(record, dict) and record.get("global_physical_count") is not None
            ]
            if counts:
                count = torch.stack([item.cpu() for item in counts], dim=0)
                physical_to_logical_map = payload.get("last_physical_to_logical_map")
                if physical_to_logical_map is not None:
                    physical_to_logical_map = physical_to_logical_map.cpu().to(torch.int64)
                    num_logical_experts = int(physical_to_logical_map.max().item()) + 1
                    logical_count = torch.zeros(
                        (*count.shape[:2], num_logical_experts),
                        dtype=count.dtype,
                    )
                    logical_count.scatter_add_(
                        dim=2,
                        index=physical_to_logical_map.unsqueeze(0).expand(count.shape[0], -1, -1),
                        src=count,
                    )
                    count = logical_count
                    count_kind = "physical_to_logical"
                else:
                    count_kind = "physical"

        if count is None:
            continue

        count = count.cpu()
        if count.ndim == 3:
            count = count.sum(dim=0)
        if count.ndim != 2:
            raise SystemExit(f"Unsupported count shape in {path}: {tuple(count.shape)}")

        for logical_layer_index, layer_count in enumerate(count):
            layer_id = logical_layer_index + args.first_moe_layer_id
            values = layer_count.to(torch.float64)
            total = float(values.sum().item())
            active = values[values > 0]
            mean = float(values.mean().item()) if values.numel() else 0.0
            std = float(values.std(unbiased=False).item()) if values.numel() else 0.0
            max_value, max_index = torch.max(values, dim=0)
            nonzero_mean = float(active.mean().item()) if active.numel() else 0.0
            rows.append(
                {
                    "file": str(path),
                    "count_kind": count_kind,
                    "logical_layer_index": logical_layer_index,
                    "layer_id": layer_id,
                    "num_experts": int(values.numel()),
                    "total_assignments": total,
                    "active_experts": int(active.numel()),
                    "max_expert_id": int(max_index.item()),
                    "max_assignments": float(max_value.item()),
                    "mean_assignments": mean,
                    "std_assignments": std,
                    "cv": std / mean if mean else "",
                    "max_over_mean": float(max_value.item()) / mean if mean else "",
                    "nonzero_mean_assignments": nonzero_mean,
                    "max_over_nonzero_mean": (
                        float(max_value.item()) / nonzero_mean if nonzero_mean else ""
                    ),
                }
            )

    if not rows:
        raise SystemExit(f"No supported expert distribution records found in {args.input_dir}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} expert distribution summary rows to {output}")


def _format_float(value: str | float, digits: int = 2) -> str:
    if value == "":
        return ""
    return f"{float(value):.{digits}f}"


def _markdown_table(rows: list[dict[str, str]], columns: list[str], limit: int) -> str:
    if not rows:
        return "_No rows._\n"
    shown = rows[:limit]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in shown:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def make_report(args: argparse.Namespace) -> None:
    lines = [
        "# DeepSeekV3 MoE Calibration Report",
        "",
        "This report is generated from SGLang trace-derived CSVs and AIC query results.",
        "",
    ]

    compare_paths = [Path(path) for path in (args.compare_csv or [])]
    compare_paths = [path for path in compare_paths if path.exists()]
    if compare_paths:
        lines += ["## AIC vs SGLang", ""]
        for compare_path in compare_paths:
            rows = _load_csv(compare_path)
            for row in rows:
                row["real_mean_us"] = _format_float(row.get("real_mean_us", ""))
                row["aic_latency_us"] = _format_float(row.get("aic_latency_us", ""))
                row["abs_error_us"] = _format_float(row.get("abs_error_us", ""))
                row["rel_error_pct"] = _format_float(row.get("rel_error_pct", ""))
            rows.sort(
                key=lambda row: abs(float(row.get("rel_error_pct") or 0.0)),
                reverse=True,
            )
            lines += [
                f"### {compare_path.stem}",
                "",
                _markdown_table(
                    rows,
                    [
                        "phase",
                        "num_tokens",
                        "distribution",
                        "operation",
                        "real_stage",
                        "real_samples",
                        "real_mean_us",
                        "aic_latency_us",
                        "abs_error_us",
                        "rel_error_pct",
                        "aic_source",
                        "quant_mode",
                        "moe_backend",
                    ],
                    args.limit,
                ),
                "",
            ]

    if args.breakdown_summary_csv and Path(args.breakdown_summary_csv).exists():
        rows = _load_csv(Path(args.breakdown_summary_csv))
        for row in rows:
            row["module_mean_us"] = _format_float(row.get("module_mean_us", ""))
            row["component_sum_mean_us"] = _format_float(
                row.get("component_sum_mean_us", "")
            )
            row["residual_mean_us"] = _format_float(row.get("residual_mean_us", ""))
            row["residual_mean_pct"] = _format_float(row.get("residual_mean_pct", ""))
        rows.sort(
            key=lambda row: abs(float(row.get("residual_mean_pct") or 0.0)),
            reverse=True,
        )
        lines += [
            "## Module Closure",
            "",
            _markdown_table(
                rows,
                [
                    "phase",
                    "num_tokens",
                    "samples",
                    "module_mean_us",
                    "component_sum_mean_us",
                    "residual_mean_us",
                    "residual_mean_pct",
                ],
                args.limit,
            ),
            "",
        ]

    if args.expert_distribution_csv and Path(args.expert_distribution_csv).exists():
        rows = _load_csv(Path(args.expert_distribution_csv))
        for row in rows:
            row["cv"] = _format_float(row.get("cv", ""))
            row["max_over_mean"] = _format_float(row.get("max_over_mean", ""))
            row["total_assignments"] = _format_float(row.get("total_assignments", ""))
        rows.sort(key=lambda row: float(row.get("max_over_mean") or 0.0), reverse=True)
        lines += [
            "## Expert Distribution",
            "",
            _markdown_table(
                rows,
                [
                    "layer_id",
                    "logical_layer_index",
                    "count_kind",
                    "total_assignments",
                    "active_experts",
                    "max_expert_id",
                    "cv",
                    "max_over_mean",
                ],
                args.limit,
            ),
            "",
        ]

    if args.stage_map_csv and Path(args.stage_map_csv).exists():
        rows = _load_csv(Path(args.stage_map_csv))
        lines += [
            "## Stage Alignment",
            "",
            _markdown_table(
                rows,
                [
                    "sglang_stage",
                    "aic_target",
                    "primary_compare_csv",
                    "notes",
                ],
                1000,
            ),
            "",
        ]

    lines += [
        "## Reading Notes",
        "",
        "- Use `topk+routed/compute` for the strict ordinary AIC `moe_perf` collector boundary (`select_experts + fused_moe`).",
        "- Use `collector/moe` as a SGLang routed-wrapper sanity check; it can include dispatcher/combine wrapper overhead around the fused kernel.",
        "- Use `routed/compute` for WideEP compute-only calibration.",
        "- Use `routed/dispatch+routed/combine` or the TBO segmented equivalent for DeepEP communication calibration.",
        "- Composite real stages are grouped by trace/layer/pid/tid, timestamp-sorted, and paired by occurrence, so repeated forwards stay as separate samples.",
        "- Large `cv` or `max_over_mean` means the real route distribution is skewed; avoid attributing that error directly to kernel latency.",
        "- Large module residual means the module-level gap is likely in router/shared/output/all-reduce, overlap, or an untracked stage.",
        "",
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote calibration report to {output}")


def compare(args: argparse.Namespace) -> None:
    trace_rows = [
        row
        for row in _load_csv(Path(args.trace_csv))
        if (not args.phase or _row_phase(row) == args.phase)
    ]
    aic_rows = [
        row
        for row in _load_csv(Path(args.aic_csv))
        if (not args.phase or _row_phase(row) == args.phase)
    ]

    groups = {
        (num_tokens, args.distribution): vals
        for num_tokens, vals in _stage_values(trace_rows, args.real_stage, args.duration_column).items()
    }

    aic_by_key = {
        (row["num_tokens"], row["distribution"]): row
        for row in aic_rows
    }

    output_rows = []
    for key, vals in sorted(groups.items(), key=lambda item: (int(item[0][0]), item[0][1])):
        aic = aic_by_key.get(key)
        if not aic:
            continue
        real_mean_us = statistics.mean(vals)
        real_p50_us = statistics.median(vals)
        aic_us = float(aic["aic_latency_us"])
        output_rows.append(
            {
                "num_tokens": key[0],
                "distribution": key[1],
                "real_stage": args.real_stage,
                "duration_column": args.duration_column,
                "real_samples": len(vals),
                "real_mean_us": real_mean_us,
                "real_p50_us": real_p50_us,
                "aic_latency_us": aic_us,
                "abs_error_us": aic_us - real_mean_us,
                "rel_error_pct": (aic_us - real_mean_us) / real_mean_us * 100.0 if real_mean_us else "",
                "aic_source": aic.get("aic_source", ""),
                "operation": aic.get("operation", ""),
                "phase": aic.get("phase", args.phase or ""),
                "backend_version": aic.get("backend_version", ""),
                "quant_mode": aic.get("quant_mode", ""),
                "moe_tp_size": aic.get("moe_tp_size", ""),
                "moe_ep_size": aic.get("moe_ep_size", ""),
                "moe_backend": aic.get("moe_backend", ""),
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output_rows:
        trace_keys = ", ".join(
            f"{num_tokens}/{distribution}" for num_tokens, distribution in sorted(groups)
        ) or "none"
        aic_keys = ", ".join(
            f"{num_tokens}/{distribution}" for num_tokens, distribution in sorted(aic_by_key)
        ) or "none"
        raise SystemExit(
            "No comparable rows. Check --real-stage, trace num_tokens inference, "
            f"AIC distribution, and phase. trace_keys={trace_keys}; aic_keys={aic_keys}"
        )
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} comparison rows to {output}")


def add_shape_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--inter-size", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--moe-tp-size", type=int, default=1)
    parser.add_argument("--moe-ep-size", type=int, default=1)
    parser.add_argument("--quant-mode", default="bfloat16")
    parser.add_argument("--distribution", nargs="+", default=["balanced"])
    parser.add_argument("--phase", choices=["context", "generation"], default="context")
    parser.add_argument("--moe-backend", default=None, help="Use deepep_moe for SGLang WideEP tables.")
    parser.add_argument("--non-gated", action="store_true")
    parser.add_argument("--enable-eplb", action="store_true")


def add_database_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--systems-root", default=str(REPO_ROOT / "src" / "aiconfigurator" / "systems"))
    parser.add_argument("--system", default="h100_sxm")
    parser.add_argument("--backend", default="sglang")
    parser.add_argument("--backend-version", default="0.5.9")
    parser.add_argument("--database-mode", choices=["SILICON", "HYBRID", "SOL", "EMPIRICAL"], default="SILICON")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    p = subparsers.add_parser("self-test", help="Run parser-only self checks.")
    p.set_defaults(func=self_test)

    p = subparsers.add_parser("parse-trace", help="Parse aic_moe events from SGLang chrome traces.")
    p.add_argument("--trace-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--num-tokens", type=int, default=None, help="Override num_tokens for all parsed events.")
    p.add_argument(
        "--phase",
        choices=["context", "generation"],
        default=None,
        help="Override phase for all parsed events. By default inferred from trace path.",
    )
    p.set_defaults(func=parse_trace)

    p = subparsers.add_parser("query-aic", help="Query AIC PerfDatabase.query_moe.")
    add_database_args(p)
    p.add_argument("--num-tokens", nargs="+", type=int, required=True)
    p.add_argument("--output", required=True)
    add_shape_args(p)
    p.set_defaults(func=query_aic)

    p = subparsers.add_parser(
        "query-deepep-dispatch",
        help="Query AIC DeepEP dispatch+combine tables for SGLang WideEP communication.",
    )
    add_database_args(p)
    p.add_argument("--num-tokens", nargs="+", type=int, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--hidden-size", type=int, default=7168)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--num-experts", type=int, default=256)
    p.add_argument("--node-num", type=int, default=1)
    p.add_argument("--sms", type=int, default=20)
    p.add_argument("--deepep-mode", choices=["normal", "ll"], default="normal")
    p.add_argument("--phase", choices=["context", "generation"], default="context")
    p.add_argument("--distribution", default="balanced")
    p.set_defaults(func=query_deepep_dispatch)

    p = subparsers.add_parser("summarize-trace", help="Summarize parsed trace events by token and stage.")
    p.add_argument("--trace-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--stage", nargs="*", default=None)
    p.add_argument(
        "--duration-column",
        default="duration_us",
        help="Trace CSV duration column to aggregate, e.g. duration_us or cuda_duration_us.",
    )
    p.set_defaults(func=summarize_trace)

    p = subparsers.add_parser("validate-trace", help="Validate required aic_moe stages exist.")
    p.add_argument("--trace-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--required-stage",
        nargs="+",
        default=[
            "module",
            "router",
            "topk",
            "shared_experts",
            "routed_experts",
            "output_postprocess",
        ],
    )
    p.add_argument(
        "--optional-stage",
        nargs="*",
        default=[
            "collector/moe",
            "routed/dispatch",
            "routed/compute",
            "routed/combine",
            "routed/dispatch_a",
            "routed/dispatch_b",
            "routed/combine_a",
            "routed/combine_b",
            "routed/all_reduce",
            "output_all_reduce",
        ],
    )
    p.add_argument("--fail-on-missing", action="store_true")
    p.add_argument(
        "--expected-layers",
        type=int,
        default=None,
        help="Expected number of MoE layers per token/stage. Marks required stages incomplete if fewer layers are observed.",
    )
    p.set_defaults(func=validate_trace)

    p = subparsers.add_parser(
        "breakdown-trace",
        help="Build per-module closure rows from parsed trace events.",
    )
    p.add_argument("--trace-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--module-stage", default="module")
    p.add_argument(
        "--component",
        nargs="+",
        default=[
            "router",
            "collector/moe",
            "shared_experts",
            "output_postprocess",
            "output_all_reduce",
        ],
    )
    p.add_argument("--time-epsilon-us", type=float, default=1.0)
    p.add_argument("--summary-output", default=None)
    p.set_defaults(func=breakdown_trace)

    p = subparsers.add_parser(
        "summarize-expert-distribution",
        help="Summarize SGLang expert_distribution_recorder .pt files.",
    )
    p.add_argument("--input-dir", required=True)
    p.add_argument("--pattern", default="expert_distribution_recorder_*.pt")
    p.add_argument("--output", required=True)
    p.add_argument(
        "--first-moe-layer-id",
        type=int,
        default=0,
        help="Offset logical MoE layer indexes to model layer ids, e.g. 3 for DeepSeekV3.",
    )
    p.set_defaults(func=summarize_expert_distribution)

    p = subparsers.add_parser("make-report", help="Generate a Markdown calibration summary.")
    p.add_argument("--compare-csv", nargs="*", default=None)
    p.add_argument("--breakdown-summary-csv", default=None)
    p.add_argument("--expert-distribution-csv", default=None)
    p.add_argument("--stage-map-csv", default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=12)
    p.set_defaults(func=make_report)

    p = subparsers.add_parser("compare", help="Compare parsed trace CSV against AIC query CSV.")
    p.add_argument("--trace-csv", required=True)
    p.add_argument("--aic-csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--real-stage", default="topk+routed/compute")
    p.add_argument("--distribution", default="balanced")
    p.add_argument("--phase", choices=["context", "generation"], default="context")
    p.add_argument(
        "--duration-column",
        default="duration_us",
        help="Trace CSV duration column to compare, e.g. duration_us or cuda_duration_us.",
    )
    p.set_defaults(func=compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
