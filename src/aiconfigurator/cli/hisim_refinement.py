# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import pandas as pd

from aiconfigurator.sdk.pareto_analysis import get_pareto_front

logger = logging.getLogger(__name__)


DEFAULT_REFINEMENT_CONFIG: dict[str, Any] = {
    "enabled": False,
    "backend": "hisim",
    "serving_modes": ["agg"],
    "capacity_mode": "auto",
    "merge_analytical_results": True,
    "max_workers": 1,
    "cache": {
        "enabled": False,
        "dir": None,
    },
    "progress": {
        "enabled": False,
        "poll_interval_s": 2,
        "show_worker_stages": False,
    },
    "candidate_top_k": 20,
    "frontier_buffer_pct": 0.10,
    "sla_metric": "mean",
    "workload": {
        "dataset": "random_ids",
        "seed": 42,
        "num_prompts_min": 512,
        "duration_s": 60,
        "input_len": None,
        "output_len": None,
        "prefix_hit_rate": None,
    },
    "load_search": {
        "max_iters": 8,
        "rps_tolerance_pct": 5,
        "initial_upper_scale": 1.5,
        "max_upper_scale": 4.0,
        "num_operating_points": 5,
        "min_operating_rps_scale": 0.25,
        "operating_rps_scales": None,
        "target_concurrency_points": [1],
        "target_concurrency_refine_iters": 1,
    },
    "hisim": {
        "repo_path": None,
        "output_dir": None,
        "platform_accelerator_name": None,
        "predictor_device_name": None,
        "database_path": None,
        "database_mode": "SILICON",
        "prefill_scale_factor": 1,
        "decode_scale_factor": 1,
        "predictor_backend_name": None,
        "predictor_backend_version": None,
        "xgb_model_path": None,
        "max_running_requests": None,
        "max_total_tokens": None,
        "timeout_s": 3600,
    },
}


class CandidateEvaluator(Protocol):
    def evaluate(self, candidate: pd.Series, request_rate: float) -> dict[str, Any]:
        """Evaluate one candidate at one request rate and return hisim metrics."""


@dataclass
class LoadSearchResult:
    best_rps: float
    metrics: dict[str, Any]
    history: list[dict[str, Any]] = field(default_factory=list)
    operating_points: list[dict[str, Any]] = field(default_factory=list)


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(base.get(key), dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def merge_refinement_config(
    root_config: dict[str, Any] | bool | None,
    exp_config: dict[str, Any] | bool | None = None,
) -> dict[str, Any]:
    """Merge default, root-level, and per-experiment refinement settings."""

    merged = copy.deepcopy(DEFAULT_REFINEMENT_CONFIG)
    for config in (root_config, exp_config):
        if config is None:
            continue
        if isinstance(config, bool):
            merged["enabled"] = config
            continue
        if not isinstance(config, dict):
            raise TypeError(f"refinement config must be a mapping or bool, got {type(config)!r}")
        _deep_merge(merged, config)
    return merged


def should_refine_task(task_config: Any) -> bool:
    refinement = getattr(task_config, "refinement", None) or {}
    if not refinement.get("enabled", False):
        return False
    if refinement.get("backend") != "hisim":
        return False
    serving_modes = refinement.get("serving_modes") or ["agg"]
    return task_config.serving_mode in serving_modes


def _get_runtime_value(task_config: Any, name: str, default: Any = None) -> Any:
    runtime_config = task_config.config.runtime_config
    return getattr(runtime_config, name, default)


def _get_workload_config(refinement: dict[str, Any], task_config: Any) -> dict[str, Any]:
    workload = copy.deepcopy(refinement.get("workload", {}))
    isl = int(_get_runtime_value(task_config, "isl", 1))
    osl = int(_get_runtime_value(task_config, "osl", 1))
    prefix = int(_get_runtime_value(task_config, "prefix", 0) or 0)
    workload["input_len"] = int(workload.get("input_len") or isl)
    workload["output_len"] = int(workload.get("output_len") or osl)
    if workload.get("prefix_hit_rate") is None:
        workload["prefix_hit_rate"] = max(0.0, min(1.0, prefix / isl)) if isl > 0 else 0.0
    return workload


def _ranking_col(df: pd.DataFrame) -> str:
    return "tokens/s/gpu_cluster" if "tokens/s/gpu_cluster" in df.columns else "tokens/s/gpu"


def _capacity_mode(refinement: dict[str, Any] | None) -> str:
    mode = str((refinement or {}).get("capacity_mode", "auto")).lower()
    return "candidate_bs" if mode in {"candidate_bs", "candidate-bs", "bs"} else "auto"


def _candidate_dedupe_cols(selected: pd.DataFrame, capacity_mode: str) -> list[str]:
    if capacity_mode == "candidate_bs":
        preferred = ("system", "parallel", "bs", "ctx_tokens")
    else:
        preferred = (
            "system",
            "num_total_gpus",
            "parallel",
            "ctx_tokens",
            "gemm",
            "moe",
            "kvcache",
            "fmha",
            "comm",
            "backend",
            "version",
        )
    return [col for col in preferred if col in selected.columns]


def select_refinement_candidates(
    pareto_df: pd.DataFrame,
    *,
    candidate_top_k: int = 20,
    frontier_buffer_pct: float = 0.10,
    x_axis_col: str = "tokens/s/user",
    capacity_mode: str = "auto",
) -> pd.DataFrame:
    """Select analytical frontier/buffer points plus top candidates on both Pareto axes."""

    if pareto_df is None or pareto_df.empty:
        return pd.DataFrame()

    df = pareto_df.copy()
    y_col = _ranking_col(df)
    if y_col not in df.columns or x_axis_col not in df.columns:
        return df.head(candidate_top_k).copy()

    top_k = max(int(candidate_top_k or 0), 0)
    maximize_x = x_axis_col != "request_latency"

    top_y = (
        df.sort_values(
            [y_col, x_axis_col],
            ascending=[False, not maximize_x],
            kind="mergesort",
        ).head(top_k)
        if top_k > 0
        else df.iloc[0:0]
    )
    top_x = (
        df.sort_values(
            [x_axis_col, y_col],
            ascending=[not maximize_x, False],
            kind="mergesort",
        ).head(top_k)
        if top_k > 0
        else df.iloc[0:0]
    )

    frontier = get_pareto_front(
        df,
        x_axis_col,
        y_col,
        maximize_x=maximize_x,
        maximize_y=True,
    )
    buffer_pct = max(float(frontier_buffer_pct or 0), 0.0)
    in_buffer: list[int] = []

    if not frontier.empty:
        frontier_numeric = frontier[[x_axis_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
        for idx, row in df.iterrows():
            try:
                x_val = float(row[x_axis_col])
                y_val = float(row[y_col])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(x_val) or not math.isfinite(y_val):
                continue
            if maximize_x:
                comparable = frontier_numeric[frontier_numeric[x_axis_col] >= x_val]
            else:
                comparable = frontier_numeric[frontier_numeric[x_axis_col] <= x_val]
            if comparable.empty:
                nearest_idx = (frontier_numeric[x_axis_col] - x_val).abs().idxmin()
                comparable = frontier_numeric.loc[[nearest_idx]]
            frontier_y = float(comparable[y_col].max())
            if y_val >= frontier_y * (1.0 - buffer_pct):
                in_buffer.append(idx)

    selected = pd.concat([frontier, df.loc[in_buffer], top_y, top_x], ignore_index=False)
    capacity_mode = _capacity_mode({"capacity_mode": capacity_mode})
    if capacity_mode == "auto" and not selected.empty:
        selected = selected.sort_values(
            [y_col, x_axis_col],
            ascending=[False, not maximize_x],
            kind="mergesort",
        )
    dedupe_cols = _candidate_dedupe_cols(selected, capacity_mode)
    if dedupe_cols:
        selected = selected.drop_duplicates(subset=dedupe_cols, keep="first")
    else:
        selected = selected.drop_duplicates(keep="first")
    return selected.reset_index(drop=True)


def metrics_meet_sla(metrics: dict[str, Any], task_config: Any, sla_metric: str = "mean") -> bool:
    prefix = "p99" if str(sla_metric).lower() == "p99" else "mean"
    runtime_config = task_config.config.runtime_config

    checks: list[tuple[str, float | None]] = [
        (f"{prefix}_ttft_ms", getattr(runtime_config, "ttft", None)),
        (f"{prefix}_tpot_ms", getattr(runtime_config, "tpot", None)),
    ]
    request_latency_target = getattr(runtime_config, "request_latency", None)
    if request_latency_target is not None and request_latency_target > 0:
        checks.append((f"{prefix}_e2e_latency_ms", request_latency_target))

    for metric_name, target in checks:
        if target is None or target <= 0:
            continue
        value = metrics.get(metric_name)
        if value is None:
            return False
        try:
            if float(value) > float(target):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _float_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        values = [value]
    else:
        values = list(value)
    result = []
    for item in values:
        try:
            number = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and number > 0:
            result.append(number)
    return result


def _candidate_e2e_latency_s(candidate: pd.Series) -> float:
    request_latency_ms = float(candidate.get("request_latency", 0.0) or 0.0)
    if request_latency_ms > 0:
        return request_latency_ms / 1000.0

    ttft_ms = float(candidate.get("ttft", 0.0) or 0.0)
    tpot_ms = float(candidate.get("tpot", 0.0) or 0.0)
    osl = max(int(candidate.get("osl", 1) or 1), 1)
    latency_ms = ttft_ms + max(osl - 1, 0) * tpot_ms
    return max(latency_ms / 1000.0, 1e-6)


def _metrics_e2e_latency_s(metrics: dict[str, Any]) -> float:
    e2e_ms = float(metrics.get("mean_e2e_latency_ms", 0.0) or 0.0)
    if e2e_ms > 0:
        return e2e_ms / 1000.0
    throughput = float(metrics.get("request_throughput", 0.0) or 0.0)
    concurrency = float(metrics.get("concurrency", 0.0) or 0.0)
    if throughput > 0 and concurrency > 0:
        return concurrency / throughput
    return 1e-6


def _metrics_progress_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "request_throughput",
        "output_throughput",
        "mean_ttft_ms",
        "mean_tpot_ms",
        "mean_e2e_latency_ms",
        "concurrency",
    )
    summary: dict[str, Any] = {}
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            try:
                summary[key] = round(float(value), 6)
            except (TypeError, ValueError):
                summary[key] = value
    return summary


def search_best_rps(
    evaluator: CandidateEvaluator,
    candidate: pd.Series,
    task_config: Any,
    refinement: dict[str, Any],
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> LoadSearchResult:
    search_cfg = refinement.get("load_search", {})
    max_iters = int(search_cfg.get("max_iters", 8))
    tolerance_pct = float(search_cfg.get("rps_tolerance_pct", 5))
    initial_upper_scale = float(search_cfg.get("initial_upper_scale", 1.5))
    max_upper_scale = float(search_cfg.get("max_upper_scale", 4.0))
    num_operating_points = max(int(search_cfg.get("num_operating_points", 5) or 1), 1)
    min_operating_rps_scale = max(
        0.0, min(1.0, float(search_cfg.get("min_operating_rps_scale", 0.25) or 0.0))
    )
    operating_rps_scales = [
        max(0.0, min(1.0, scale))
        for scale in _float_list(search_cfg.get("operating_rps_scales"))
    ]
    target_concurrency_points = _float_list(search_cfg.get("target_concurrency_points", []))
    target_concurrency_refine_iters = max(
        int(search_cfg.get("target_concurrency_refine_iters", 1) or 0), 0
    )
    sla_metric = refinement.get("sla_metric", "mean")

    aic_seq_s = float(candidate.get("seq/s", candidate.get("request_rate", 1.0)) or 1.0)
    aic_seq_s = max(aic_seq_s, 1e-6)
    low = 0.0
    high = max(aic_seq_s * initial_upper_scale, 1e-6)
    max_high = max(aic_seq_s * max_upper_scale, high)
    best_rps = 0.0
    best_metrics: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    evaluated: dict[float, tuple[bool, dict[str, Any]]] = {}

    def rps_key(rps: float) -> float:
        return round(float(rps), 9)

    def emit(phase: str, step: str, **fields: Any) -> None:
        if progress_callback is None:
            return
        progress_callback({"phase": phase, "step": step, **_json_safe(fields)})

    def evaluate(
        rps: float,
        *,
        phase: str,
        step: str,
        **fields: Any,
    ) -> tuple[bool, dict[str, Any]]:
        key = rps_key(rps)
        if key in evaluated:
            ok, metrics = evaluated[key]
            emit(
                phase,
                f"{step}_cached",
                request_rate=rps,
                ok=ok,
                metrics=_metrics_progress_summary(metrics),
                **fields,
            )
            return ok, metrics
        emit(phase, f"{step}_start", request_rate=rps, **fields)
        try:
            metrics = evaluator.evaluate(candidate, rps)
        except Exception as exc:
            emit(phase, f"{step}_failed", request_rate=rps, error=repr(exc), **fields)
            raise
        ok = metrics_meet_sla(metrics, task_config, sla_metric=sla_metric)
        history.append({"request_rate": rps, "ok": ok, "metrics": metrics})
        evaluated[key] = (ok, metrics)
        emit(
            phase,
            f"{step}_done",
            request_rate=rps,
            ok=ok,
            metrics=_metrics_progress_summary(metrics),
            **fields,
        )
        return ok, metrics

    emit(
        "max_rps_search",
        "start",
        aic_seq_s=aic_seq_s,
        initial_upper_rps=high,
        initial_upper_scale=initial_upper_scale,
        max_upper_rps=max_high,
        max_upper_scale=max_upper_scale,
        max_iters=max_iters,
        rps_tolerance_pct=tolerance_pct,
    )
    ok, metrics = evaluate(
        high,
        phase="max_rps_search",
        step="initial_upper",
        upper_scale=high / aic_seq_s,
    )
    if ok:
        low = high
        best_rps = high
        best_metrics = metrics
        expand_iter = 0
        while low < max_high:
            expand_iter += 1
            next_high = min(max(high * 2, high + 1e-6), max_high)
            if next_high <= high:
                break
            high = next_high
            ok, metrics = evaluate(
                high,
                phase="max_rps_search",
                step="expand_upper",
                iteration=expand_iter,
                upper_scale=high / aic_seq_s,
                low=low,
                max_high=max_high,
            )
            if ok:
                low = high
                best_rps = high
                best_metrics = metrics
            else:
                break

    for iteration in range(max_iters):
        if high <= 0:
            break
        if low > 0 and (high - low) / low * 100 <= tolerance_pct:
            break
        mid = (low + high) / 2
        if mid <= 0:
            break
        ok, metrics = evaluate(
            mid,
            phase="max_rps_search",
            step="binary",
            iteration=iteration + 1,
            low=low,
            high=high,
        )
        if ok:
            low = mid
            best_rps = mid
            best_metrics = metrics
        else:
            high = mid

    if best_metrics is None:
        # Keep the lowest tested point as diagnostics when nothing meets SLA.
        best_entry = min(history, key=lambda item: item["request_rate"])
        best_rps = float(best_entry["request_rate"])
        best_metrics = dict(best_entry["metrics"])
    emit(
        "max_rps_search",
        "done",
        best_rps=best_rps,
        low=low,
        high=high,
        evaluated_points=len(history),
        metrics=_metrics_progress_summary(best_metrics or {}),
    )

    operating_points: list[dict[str, Any]] = []

    def add_operating_point(
        rps: float,
        *,
        point_type: str,
        target_concurrency: float | None = None,
        rps_scale: float | None = None,
    ) -> None:
        if rps <= 0:
            return
        capped_rps = min(rps, best_rps)
        ok, metrics = evaluate(
            capped_rps,
            phase="rps_scale_search",
            step="operating_point",
            point_type=point_type,
            target_rps=rps,
            capped_rps=capped_rps,
            rps_scale=rps_scale,
            target_concurrency=target_concurrency,
        )
        if ok:
            operating_points.append(
                {
                    "request_rate": capped_rps,
                    "ok": ok,
                    "metrics": metrics,
                    "point_type": point_type,
                    "target_concurrency": target_concurrency,
                }
            )

    if best_rps > 0 and best_metrics is not None:
        if operating_rps_scales:
            target_rps_values = [(best_rps * scale, scale) for scale in operating_rps_scales]
            if all(abs(value - best_rps) > 1e-9 for value, _scale in target_rps_values):
                target_rps_values.append((best_rps, 1.0))
        elif num_operating_points == 1:
            target_rps_values = [(best_rps, 1.0)]
        else:
            target_rps_values = [
                (
                    best_rps
                    * (
                        min_operating_rps_scale
                        + (1.0 - min_operating_rps_scale) * i / (num_operating_points - 1)
                    ),
                    min_operating_rps_scale
                    + (1.0 - min_operating_rps_scale) * i / (num_operating_points - 1),
                )
                for i in range(num_operating_points)
            ]
        for target_rps, rps_scale in target_rps_values:
            add_operating_point(target_rps, point_type="rps_scale", rps_scale=rps_scale)

        seed_latency_s = _candidate_e2e_latency_s(candidate)
        for target_concurrency in target_concurrency_points:
            target_rps = min(best_rps, target_concurrency / seed_latency_s)
            last_rps = target_rps
            emit(
                "target_concurrency_search",
                "target_start",
                target_concurrency=target_concurrency,
                seed_latency_s=seed_latency_s,
                initial_rps=target_rps,
                best_rps=best_rps,
            )
            for iteration in range(target_concurrency_refine_iters + 1):
                ok, metrics = evaluate(
                    last_rps,
                    phase="target_concurrency_search",
                    step="target_iteration",
                    target_concurrency=target_concurrency,
                    iteration=iteration,
                    best_rps=best_rps,
                )
                if ok:
                    operating_points.append(
                        {
                            "request_rate": last_rps,
                            "ok": ok,
                            "metrics": metrics,
                            "point_type": "target_concurrency",
                            "target_concurrency": target_concurrency,
                        }
                    )
                if iteration >= target_concurrency_refine_iters:
                    break
                measured_latency_s = _metrics_e2e_latency_s(metrics)
                next_rps = min(best_rps, target_concurrency / measured_latency_s)
                converged = last_rps > 0 and abs(next_rps - last_rps) / last_rps <= 0.05
                emit(
                    "target_concurrency_search",
                    "target_update",
                    target_concurrency=target_concurrency,
                    iteration=iteration,
                    measured_latency_s=measured_latency_s,
                    previous_rps=last_rps,
                    next_rps=next_rps,
                    converged=converged,
                )
                if converged:
                    break
                last_rps = max(next_rps, 1e-9)
            emit(
                "target_concurrency_search",
                "target_done",
                target_concurrency=target_concurrency,
                last_rps=last_rps,
            )

    if not operating_points:
        operating_points.append(
            {
                "request_rate": best_rps,
                "ok": False,
                "metrics": best_metrics,
                "point_type": "fallback",
                "target_concurrency": None,
            }
        )

    operating_points = sorted(
        {
            rps_key(point["request_rate"]): point
            for point in operating_points
        }.values(),
        key=lambda point: float(point["request_rate"]),
    )

    return LoadSearchResult(
        best_rps=best_rps,
        metrics=best_metrics,
        history=history,
        operating_points=operating_points,
    )


_AIC_COLUMNS_TO_PRESERVE = (
    "ttft",
    "tpot",
    "request_latency",
    "seq/s",
    "seq/s/gpu",
    "tokens/s",
    "tokens/s/gpu",
    "tokens/s/user",
    "tokens/s/gpu_cluster",
    "request_rate",
    "concurrency",
)


def map_hisim_metrics_to_aic_row(
    candidate: pd.Series,
    metrics: dict[str, Any],
    *,
    best_rps: float,
    operating_rps: float | None = None,
    sla_ok: bool | None = None,
    operating_point_type: str | None = None,
    target_concurrency: float | None = None,
    capacity_mode: str = "auto",
) -> dict[str, Any]:
    row = candidate.to_dict()
    for col in _AIC_COLUMNS_TO_PRESERVE:
        if col in row:
            row[f"aic_{col}"] = row[col]
    row.pop("tokens/s/gpu_cluster", None)
    if "bs" in row:
        row["aic_seed_bs"] = row["bs"]
    if "global_bs" in row:
        row["aic_seed_global_bs"] = row["global_bs"]

    num_total_gpus = float(row.get("num_total_gpus") or 0)
    if num_total_gpus <= 0:
        num_total_gpus = float(row.get("tp", 1) or 1) * float(row.get("pp", 1) or 1) * float(row.get("dp", 1) or 1)

    mean_ttft = float(metrics.get("mean_ttft_ms", 0.0) or 0.0)
    mean_tpot = float(metrics.get("mean_tpot_ms", 0.0) or 0.0)
    mean_e2e = float(metrics.get("mean_e2e_latency_ms", 0.0) or 0.0)
    request_throughput = float(metrics.get("request_throughput", best_rps) or 0.0)
    output_throughput = float(metrics.get("output_throughput", 0.0) or 0.0)

    row["ttft"] = mean_ttft
    row["tpot"] = mean_tpot
    row["request_latency"] = mean_e2e
    row["seq/s"] = request_throughput
    row["seq/s/gpu"] = request_throughput / num_total_gpus if num_total_gpus else 0.0
    row["tokens/s"] = output_throughput
    row["tokens/s/gpu"] = output_throughput / num_total_gpus if num_total_gpus else 0.0
    row["tokens/s/user"] = 1000.0 / mean_tpot if mean_tpot > 0 else 0.0
    row["request_rate"] = request_throughput
    row["concurrency"] = float(metrics.get("concurrency", request_throughput * mean_e2e / 1000.0) or 0.0)

    row["hisim_capacity_mode"] = _capacity_mode({"capacity_mode": capacity_mode})
    row["hisim_best_rps"] = best_rps
    row["hisim_operating_rps"] = best_rps if operating_rps is None else operating_rps
    if operating_point_type is not None:
        row["hisim_operating_point_type"] = operating_point_type
    if target_concurrency is not None:
        row["hisim_target_concurrency"] = target_concurrency
    if sla_ok is not None:
        row["hisim_sla_ok"] = bool(sla_ok)
    row["hisim_max_running_requests"] = metrics.get("hisim_max_running_requests")
    row["hisim_max_total_tokens"] = metrics.get("hisim_max_total_tokens")
    row["hisim_resolved_max_running_requests"] = metrics.get(
        "hisim_resolved_max_running_requests"
    )
    row["hisim_resolved_max_total_tokens"] = metrics.get(
        "hisim_resolved_max_total_tokens"
    )
    row["hisim_p99_ttft_ms"] = metrics.get("p99_ttft_ms")
    row["hisim_p99_tpot_ms"] = metrics.get("p99_tpot_ms")
    row["hisim_p99_e2e_latency_ms"] = metrics.get("p99_e2e_latency_ms")
    row["hisim_mean_queue_ms"] = metrics.get("mean_queue_ms")
    row["hisim_prefix_cache_reused_ratio"] = metrics.get("prefix_cache_reused_ratio")
    row["hisim_completed"] = metrics.get("completed")
    row["hisim_duration"] = metrics.get("duration")
    return row


def _progress_config(refinement: dict[str, Any]) -> dict[str, Any]:
    progress = refinement.get("progress", {}) or {}
    return {
        "enabled": bool(progress.get("enabled", False)),
        "poll_interval_s": max(float(progress.get("poll_interval_s", 2) or 2), 0.1),
        "show_worker_stages": bool(progress.get("show_worker_stages", False)),
    }


def _candidate_label(candidate: pd.Series | dict[str, Any]) -> str:
    getter = candidate.get
    parts = []
    for key in ("system", "system_name", "hardware"):
        value = getter(key, None)
        if value is not None:
            parts.append(str(value))
            break
    parallel = getter("parallel", None)
    if parallel is not None:
        parts.append(str(parallel))
    for key, label in (("num_total_gpus", "gpus"), ("bs", "bs"), ("ctx_tokens", "ctx")):
        value = getter(key, None)
        if value is not None:
            parts.append(f"{label}={_format_progress_value(value)}")
    return " ".join(parts) if parts else "candidate"


def _format_progress_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return str(int(value))
        return f"{value:.6g}"
    return str(value)


def _format_progress_fields(fields: dict[str, Any]) -> str:
    if not fields:
        return ""
    flattened: list[str] = []
    fields = dict(fields)
    metrics = fields.pop("metrics", None)
    aliases = {
        "aic_seq_s": "aic_rps",
        "aic_tokens_s_gpu": "aic_tok/gpu",
        "aic_tokens_s_user": "aic_tok/user",
        "request_rate": "rps",
        "target_rps": "target_rps",
        "capped_rps": "capped_rps",
        "rps_scale": "scale",
        "upper_scale": "scale",
        "initial_upper_scale": "init_scale",
        "max_upper_scale": "max_scale",
        "initial_upper_rps": "init_rps",
        "max_upper_rps": "max_rps",
        "target_concurrency": "tc",
        "seed_latency_s": "seed_lat_s",
        "measured_latency_s": "lat_s",
        "previous_rps": "prev_rps",
        "next_rps": "next_rps",
        "iteration": "iter",
        "evaluated_points": "evals",
        "rps_tolerance_pct": "tol_pct",
        "point_type": "type",
        "best_rps": "best_rps",
    }
    preferred_order = (
        "request_rate",
        "rps_scale",
        "upper_scale",
        "target_concurrency",
        "iteration",
        "ok",
        "best_rps",
        "target_rps",
        "capped_rps",
        "low",
        "high",
        "previous_rps",
        "next_rps",
        "converged",
        "evaluated_points",
        "error",
    )
    ordered_keys = [key for key in preferred_order if key in fields]
    ordered_keys.extend(key for key in fields if key not in ordered_keys)
    for key in ordered_keys:
        value = fields[key]
        if value is None:
            continue
        flattened.append(f"{aliases.get(key, key)}={_format_progress_value(value)}")
    if isinstance(metrics, dict):
        metric_aliases = {
            "request_throughput": "req/s",
            "output_throughput": "tok/s",
            "mean_ttft_ms": "ttft",
            "mean_tpot_ms": "tpot",
            "mean_e2e_latency_ms": "e2e",
            "concurrency": "conc",
        }
        for key, value in metrics.items():
            flattened.append(f"{metric_aliases.get(key, key)}={_format_progress_value(value)}")
    return " ".join(flattened)


class _RefinementProgressReporter:
    def __init__(self, *, enabled: bool, total: int, exp_name: str, max_active: int = 1):
        self.enabled = enabled
        self.total = total
        self.completed = 0
        self.exp_name = exp_name
        self.max_active = max(max_active, 1)
        self._started_at = time.monotonic()
        self._slot_by_candidate: dict[int, int] = {}
        self._candidate_by_slot: dict[int, int] = {}
        self._status_by_candidate: dict[int, str] = {}
        self._free_slots = list(range(self.max_active))
        self._lock = threading.Lock()
        self._last_rendered_lines = 0
        self._last_logged_completed = 0
        self._tty = (
            enabled
            and sys.stderr.isatty()
            and os.environ.get("TERM", "dumb").lower() != "dumb"
        )
        if not enabled:
            return
        if self._tty:
            with self._lock:
                self._render_locked()

    def start(
        self,
        *,
        candidate_index: int,
        label: str,
        aic_seq_s: float,
        aic_tokens_s_gpu: float,
        aic_tokens_s_user: float,
    ) -> None:
        if not self.enabled:
            return
        self.stage(
            candidate_index=candidate_index,
            phase="candidate",
            step="start",
            label=label,
            aic_seq_s=aic_seq_s,
            aic_tokens_s_gpu=aic_tokens_s_gpu,
            aic_tokens_s_user=aic_tokens_s_user,
        )

    def stage(self, *, candidate_index: int, phase: str, step: str, **fields: Any) -> None:
        if not self.enabled:
            return
        text = self._stage_text(candidate_index, phase, step, fields)
        with self._lock:
            slot = self._ensure_slot(candidate_index)
            if slot is not None:
                self._candidate_by_slot[slot] = candidate_index
            self._status_by_candidate[candidate_index] = text
            self._render_locked()

    def update(self, *, candidate_index: int, ok: bool, best_rps: float | None = None) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.completed += 1
            status = "done" if ok else "failed"
            suffix = f" best_rps={best_rps:.6g}" if best_rps is not None else ""
            self._status_by_candidate[candidate_index] = f"cand {candidate_index} | {status}{suffix}"
            self._render_locked()
            self._release_slot(candidate_index)
            if not self._tty and (
                self.completed == self.total
                or self.completed - self._last_logged_completed >= max(self.total // 10, 1)
            ):
                self._last_logged_completed = self.completed
                logger.info(
                    "hisim refinement progress: %d/%d candidates complete "
                    "(candidate=%s ok=%s best_rps=%s)",
                    self.completed,
                    self.total,
                    candidate_index,
                    ok,
                    f"{best_rps:.6g}" if best_rps is not None else "n/a",
                )

    def close(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._render_locked(final=True)

    def _ensure_slot(self, candidate_index: int) -> int | None:
        if candidate_index in self._slot_by_candidate:
            return self._slot_by_candidate[candidate_index]
        if self._free_slots:
            slot = self._free_slots.pop(0)
            self._slot_by_candidate[candidate_index] = slot
            return slot
        return None

    def _release_slot(self, candidate_index: int) -> None:
        slot = self._slot_by_candidate.pop(candidate_index, None)
        if slot is None:
            return
        self._candidate_by_slot.pop(slot, None)
        self._status_by_candidate.pop(candidate_index, None)
        self._free_slots.append(slot)
        self._free_slots.sort()

    def _render_locked(self, *, final: bool = False) -> None:
        if not self._tty:
            return
        columns = max(shutil.get_terminal_size((120, 20)).columns, 40)
        lines = [self._truncate(self._total_line(), columns)]
        for slot in range(self.max_active):
            candidate_index = self._candidate_by_slot.get(slot)
            if candidate_index is None:
                line = f"slot {slot + 1}: idle"
            else:
                line = f"slot {slot + 1}: {self._status_by_candidate.get(candidate_index, f'cand {candidate_index} | running')}"
            lines.append(self._truncate(line, columns))

        if self._last_rendered_lines:
            sys.stderr.write(f"\x1b[{self._last_rendered_lines}A")
        for line in lines:
            sys.stderr.write(f"\x1b[2K{line}\n")
        self._last_rendered_lines = len(lines)
        if final:
            sys.stderr.write("\n")
        sys.stderr.flush()

    def _total_line(self) -> str:
        ratio = self.completed / self.total if self.total else 1.0
        width = 24
        filled = max(0, min(width, int(round(ratio * width))))
        bar = "#" * filled + "-" * (width - filled)
        elapsed = self._format_elapsed(time.monotonic() - self._started_at)
        active = len(self._candidate_by_slot)
        pct = ratio * 100
        return (
            f"hisim refine: [{bar}] {self.completed}/{self.total} "
            f"({pct:.1f}%) active={active} elapsed={elapsed} exp={self.exp_name}"
        )

    @staticmethod
    def _format_elapsed(elapsed_s: float) -> str:
        elapsed = max(int(elapsed_s), 0)
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    @staticmethod
    def _truncate(text: str, columns: int) -> str:
        text = text.replace("\n", " ")
        if len(text) <= columns:
            return text
        if columns <= 3:
            return text[:columns]
        return f"{text[: columns - 3]}..."

    @staticmethod
    def _stage_text(candidate_index: int, phase: str, step: str, fields: dict[str, Any]) -> str:
        fields = dict(fields)
        label = fields.pop("label", None)
        phase = {
            "max_rps_search": "max_rps",
            "rps_scale_search": "rps_scale",
            "target_concurrency_search": "target_conc",
        }.get(phase, phase)
        step = (
            step.replace("operating_point", "point")
            .replace("target_iteration", "iter")
            .replace("initial_upper", "initial")
            .replace("expand_upper", "expand")
        )
        prefix = f"cand {candidate_index} | {phase}/{step}"
        if label:
            prefix = f"cand {candidate_index} | {label} | {phase}/{step}"
        formatted = _format_progress_fields(fields)
        return f"{prefix} | {formatted}" if formatted else prefix


class SubprocessHisimEvaluator:
    def __init__(self, refinement: dict[str, Any], task_config: Any, exp_name: str):
        self.refinement = refinement
        self.task_config = task_config
        self.exp_name = exp_name
        self.hisim_cfg = refinement.get("hisim", {})
        self.base_output_dir = Path(
            self.hisim_cfg.get("output_dir")
            or Path(tempfile.gettempdir()) / "aic_hisim_refinement"
        )
        cache_cfg = refinement.get("cache", {}) or {}
        self.cache_enabled = bool(cache_cfg.get("enabled", False))
        self.cache_dir = Path(
            cache_cfg.get("dir") or self.base_output_dir / "_cache"
        )
        self._cache_lock = threading.Lock()
        self.progress_cfg = _progress_config(refinement)
        self._progress_lock = threading.Lock()

    def evaluate(self, candidate: pd.Series, request_rate: float) -> dict[str, Any]:
        cache_path = self._cache_path(candidate, request_rate)
        if cache_path is not None:
            cached = self._read_cache(cache_path)
            if cached is not None:
                logger.debug(
                    "hisim cache hit: %s rps=%.6g label=%s",
                    cache_path,
                    request_rate,
                    _candidate_label(candidate),
                )
                return cached

        metrics = self._evaluate_uncached(candidate, request_rate)
        if cache_path is not None:
            self._write_cache(cache_path, metrics)
        return metrics

    def _evaluate_uncached(self, candidate: pd.Series, request_rate: float) -> dict[str, Any]:
        output_dir = self.base_output_dir / self._safe_name(self.exp_name) / str(uuid.uuid4())
        output_dir.mkdir(parents=True, exist_ok=True)
        job_path = output_dir / "job.json"
        result_path = output_dir / "result.json"
        progress_path = output_dir / "progress.jsonl"

        job = {
            "exp_name": self.exp_name,
            "task": self._task_payload(),
            "candidate": _json_safe(candidate.to_dict()),
            "request_rate": request_rate,
            "refinement": _json_safe(self.refinement),
            "output_dir": str(output_dir),
            "result_path": str(result_path),
            "progress_path": str(progress_path),
        }
        job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")

        env = os.environ.copy()
        pythonpath = self._pythonpath_entries()
        if pythonpath:
            env["PYTHONPATH"] = os.pathsep.join([*pythonpath, env.get("PYTHONPATH", "")]).strip(os.pathsep)
        env["HISIM_OUTPUT_DIR"] = str(output_dir)
        env["HISIM_SIMULATION_MODE"] = "OFFLINE"

        cmd = [sys.executable, "-m", "aiconfigurator.cli.hisim_refinement_worker", str(job_path)]
        if self.progress_cfg["enabled"]:
            logger.debug(
                "hisim eval start: rps=%.6g label=%s output_dir=%s",
                request_rate,
                _candidate_label(candidate),
                output_dir,
            )

        timeout_s = float(self.hisim_cfg.get("timeout_s", 3600))
        completed = self._run_worker_process(
            cmd,
            env=env,
            timeout_s=timeout_s,
            progress_path=progress_path,
            request_rate=request_rate,
            label=_candidate_label(candidate),
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "hisim refinement worker failed with code "
                f"{completed.returncode}: stdout={completed.stdout[-2000:]!r} stderr={completed.stderr[-4000:]!r}"
            )
        if not result_path.exists():
            raise RuntimeError(f"hisim refinement worker did not write result: {result_path}")
        metrics = json.loads(result_path.read_text(encoding="utf-8"))["metrics"]
        if self.progress_cfg["enabled"]:
            logger.debug(
                "hisim eval done: rps=%.6g throughput=%.6g ttft_ms=%.3f tpot_ms=%.3f label=%s",
                request_rate,
                float(metrics.get("request_throughput", 0.0) or 0.0),
                float(metrics.get("mean_ttft_ms", 0.0) or 0.0),
                float(metrics.get("mean_tpot_ms", 0.0) or 0.0),
                _candidate_label(candidate),
            )
        return metrics

    def _run_worker_process(
        self,
        cmd: list[str],
        *,
        env: dict[str, str],
        timeout_s: float,
        progress_path: Path,
        request_rate: float,
        label: str,
    ) -> Any:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []
        readers = [
            threading.Thread(
                target=self._read_stream,
                args=(process.stdout, stdout_chunks),
                daemon=True,
            ),
            threading.Thread(
                target=self._read_stream,
                args=(process.stderr, stderr_chunks),
                daemon=True,
            ),
        ]
        for reader in readers:
            reader.start()

        start_time = time.monotonic()
        progress_offset = 0
        while process.poll() is None:
            if self.progress_cfg["enabled"] and self.progress_cfg["show_worker_stages"]:
                progress_offset = self._log_progress_events(
                    progress_path,
                    progress_offset,
                    request_rate=request_rate,
                    label=label,
                )
            if time.monotonic() - start_time > timeout_s:
                process.kill()
                try:
                    process.wait(timeout=5)
                except Exception:
                    pass
                for reader in readers:
                    reader.join(timeout=1)
                raise subprocess.TimeoutExpired(cmd, timeout_s)
            time.sleep(float(self.progress_cfg["poll_interval_s"]))

        for reader in readers:
            reader.join(timeout=1)
        if self.progress_cfg["enabled"] and self.progress_cfg["show_worker_stages"]:
            self._log_progress_events(
                progress_path,
                progress_offset,
                request_rate=request_rate,
                label=label,
            )

        return subprocess.CompletedProcess(
            cmd,
            process.returncode,
            stdout="".join(stdout_chunks),
            stderr="".join(stderr_chunks),
        )

    @staticmethod
    def _read_stream(stream: Any, chunks: list[str]) -> None:
        if stream is None:
            return
        try:
            for line in stream:
                chunks.append(line)
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def _log_progress_events(
        self,
        progress_path: Path,
        offset: int,
        *,
        request_rate: float,
        label: str,
    ) -> int:
        if not progress_path.exists():
            return offset
        try:
            with progress_path.open("r", encoding="utf-8") as fh:
                fh.seek(offset)
                lines = fh.readlines()
                offset = fh.tell()
        except Exception:
            return offset

        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            stage = event.get("stage", "unknown")
            elapsed_s = event.get("elapsed_s")
            extras = {
                key: value
                for key, value in event.items()
                if key not in {"stage", "time", "elapsed_s"}
            }
            with self._progress_lock:
                logger.info(
                    "hisim stage: stage=%s elapsed=%s rps=%.6g label=%s%s",
                    stage,
                    f"{float(elapsed_s):.1f}s" if elapsed_s is not None else "n/a",
                    request_rate,
                    label,
                    f" details={extras}" if extras else "",
                )
        return offset

    def _cache_path(self, candidate: pd.Series, request_rate: float) -> Path | None:
        if not self.cache_enabled:
            return None
        payload = {
            "schema_version": 1,
            "request_rate": round(float(request_rate), 9),
            "task": self._task_payload(),
            "candidate": _json_safe(candidate.to_dict()),
            "refinement": self._cacheable_refinement(),
        }
        encoded = json.dumps(
            _json_safe(payload),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        key = hashlib.sha256(encoded).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _cacheable_refinement(self) -> dict[str, Any]:
        refinement = copy.deepcopy(self.refinement)
        refinement.pop("max_workers", None)
        refinement.pop("progress", None)
        cache_cfg = refinement.get("cache")
        if isinstance(cache_cfg, dict):
            cache_cfg.pop("dir", None)
        hisim_cfg = refinement.get("hisim")
        if isinstance(hisim_cfg, dict):
            hisim_cfg.pop("output_dir", None)
            hisim_cfg.pop("timeout_s", None)
        return _json_safe(refinement)

    def _read_cache(self, cache_path: Path) -> dict[str, Any] | None:
        with self._cache_lock:
            if not cache_path.exists():
                return None
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                metrics = payload.get("metrics")
                return metrics if isinstance(metrics, dict) else None
            except Exception:
                logger.warning("Ignoring unreadable hisim refinement cache entry: %s", cache_path)
                return None

    def _write_cache(self, cache_path: Path, metrics: dict[str, Any]) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = cache_path.with_name(f"{cache_path.stem}.{uuid.uuid4().hex}.tmp")
        payload = {"metrics": _json_safe(metrics)}
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with self._cache_lock:
            temp_path.replace(cache_path)

    def _task_payload(self) -> dict[str, Any]:
        runtime = self.task_config.config.runtime_config
        return {
            "model_path": self.task_config.model_path,
            "backend_name": self.task_config.backend_name,
            "backend_version": self.task_config.backend_version,
            "serving_mode": self.task_config.serving_mode,
            "enable_chunked_prefill": bool(getattr(self.task_config, "enable_chunked_prefill", False)),
            "isl": int(runtime.isl),
            "osl": int(runtime.osl),
            "prefix": int(getattr(runtime, "prefix", 0) or 0),
            "ttft": float(getattr(runtime, "ttft", 0) or 0),
            "tpot": float(getattr(runtime, "tpot", 0) or 0),
            "request_latency": getattr(runtime, "request_latency", None),
            "prefer_nccl_for_custom_allreduce": getattr(
                self.task_config,
                "prefer_nccl_for_custom_allreduce",
                None,
            ),
            "disable_hybrid_shared_layer": getattr(
                self.task_config,
                "disable_hybrid_shared_layer",
                None,
            ),
            "workload": _json_safe(_get_workload_config(self.refinement, self.task_config)),
        }

    def _pythonpath_entries(self) -> list[str]:
        entries = []
        repo_path = self.hisim_cfg.get("repo_path")
        if repo_path:
            repo = Path(repo_path)
            entries.extend([str(repo / "src"), str(repo)])
        entries.append(str(Path(__file__).resolve().parents[2]))
        return entries

    @staticmethod
    def _safe_name(value: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def refine_task_result(
    *,
    exp_name: str,
    task_config: Any,
    task_result: dict[str, pd.DataFrame | None],
    evaluator: CandidateEvaluator | None = None,
) -> dict[str, pd.DataFrame | None]:
    refinement = getattr(task_config, "refinement", None) or {}
    if not should_refine_task(task_config):
        return task_result

    analytical_df = task_result.get("pareto_df")
    if analytical_df is None or analytical_df.empty:
        return task_result

    runtime_config = task_config.config.runtime_config
    capacity_mode = _capacity_mode(refinement)
    x_axis_col = "request_latency" if getattr(runtime_config, "request_latency", None) else "tokens/s/user"
    candidates = select_refinement_candidates(
        analytical_df,
        candidate_top_k=int(refinement.get("candidate_top_k", 20)),
        frontier_buffer_pct=float(refinement.get("frontier_buffer_pct", 0.10)),
        x_axis_col=x_axis_col,
        capacity_mode=capacity_mode,
    )
    if candidates.empty:
        logger.warning("Experiment %s refinement selected no candidates; keeping analytical results.", exp_name)
        return task_result

    max_workers = max(int(refinement.get("max_workers", 1) or 1), 1)
    logger.info(
        "Experiment %s: refining %d analytical candidates with hisim "
        "(max_workers=%d).",
        exp_name,
        len(candidates),
        max_workers,
    )
    evaluator = evaluator or SubprocessHisimEvaluator(refinement, task_config, exp_name)
    refined_rows: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    progress_cfg = _progress_config(refinement)
    progress_reporter = _RefinementProgressReporter(
        enabled=progress_cfg["enabled"],
        total=len(candidates),
        exp_name=exp_name,
        max_active=max_workers,
    )

    def refine_candidate(idx: Any, candidate: pd.Series) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
        candidate_index = int(idx)
        candidate_label = _candidate_label(candidate)
        if progress_cfg["enabled"]:
            progress_reporter.start(
                candidate_index=candidate_index,
                label=candidate_label,
                aic_seq_s=float(candidate.get("seq/s", 0.0) or 0.0),
                aic_tokens_s_gpu=float(candidate.get("tokens/s/gpu", 0.0) or 0.0),
                aic_tokens_s_user=float(candidate.get("tokens/s/user", 0.0) or 0.0),
            )

        def search_progress(event: dict[str, Any]) -> None:
            if not progress_cfg["enabled"]:
                return
            phase = event.pop("phase", "unknown")
            step = event.pop("step", "unknown")
            progress_reporter.stage(
                candidate_index=candidate_index,
                phase=phase,
                step=step,
                label=candidate_label,
                **event,
            )

        search_result = search_best_rps(
            evaluator,
            candidate,
            task_config,
            refinement,
            progress_callback=search_progress,
        )
        rows = [
            map_hisim_metrics_to_aic_row(
                candidate,
                point["metrics"],
                best_rps=search_result.best_rps,
                operating_rps=float(point["request_rate"]),
                sla_ok=bool(point.get("ok", False)),
                operating_point_type=point.get("point_type"),
                target_concurrency=point.get("target_concurrency"),
                capacity_mode=capacity_mode,
            )
            for point in search_result.operating_points
        ]
        return int(idx), rows, {
            "candidate_index": int(idx),
            "history": search_result.history,
            "best_rps": search_result.best_rps,
        }

    indexed_candidates = [
        (idx, candidate.copy(deep=True))
        for idx, candidate in candidates.iterrows()
    ]

    completed_results = []
    try:
        if max_workers == 1:
            for idx, candidate in indexed_candidates:
                try:
                    result = refine_candidate(idx, candidate)
                    completed_results.append(result)
                    progress_reporter.update(
                        candidate_index=int(idx),
                        ok=True,
                        best_rps=float(result[2].get("best_rps", 0.0)),
                    )
                except Exception as exc:
                    progress_reporter.update(candidate_index=int(idx), ok=False)
                    logger.exception("Experiment %s: hisim refinement failed for candidate %s: %s", exp_name, idx, exc)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(refine_candidate, idx, candidate): idx
                    for idx, candidate in indexed_candidates
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        result = future.result()
                        completed_results.append(result)
                        progress_reporter.update(
                            candidate_index=int(idx),
                            ok=True,
                            best_rps=float(result[2].get("best_rps", 0.0)),
                        )
                    except Exception as exc:
                        progress_reporter.update(candidate_index=int(idx), ok=False)
                        logger.exception(
                            "Experiment %s: hisim refinement failed for candidate %s: %s",
                            exp_name,
                            idx,
                            exc,
                        )
    finally:
        progress_reporter.close()

    for _idx, rows, history in sorted(completed_results, key=lambda item: item[0]):
        try:
            refined_rows.extend(rows)
            histories.append(history)
        except Exception as exc:
            logger.exception("Experiment %s: failed to collect hisim result: %s", exp_name, exc)

    if not refined_rows:
        logger.warning("Experiment %s refinement produced no successful rows; keeping analytical results.", exp_name)
        return task_result

    refined_df = pd.DataFrame(refined_rows)
    refined_df["refinement_source"] = "hisim_refined"
    refined_df = refined_df.sort_values(
        _ranking_col(refined_df),
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)
    analytical_with_source = analytical_df.copy()
    analytical_with_source["refinement_source"] = "aic_analytical"
    if refinement.get("merge_analytical_results", True):
        result_df = pd.concat(
            [analytical_with_source, refined_df],
            ignore_index=True,
            sort=False,
        )
        result_df = result_df.sort_values(
            _ranking_col(result_df),
            ascending=False,
            kind="mergesort",
        ).reset_index(drop=True)
    else:
        result_df = refined_df

    new_result = dict(task_result)
    new_result["analytical_pareto_df"] = analytical_with_source
    new_result["hisim_refined_df"] = refined_df
    new_result["refinement_candidates_df"] = candidates
    new_result["refinement_history"] = pd.DataFrame({"history": [json.dumps(histories)]})
    new_result["pareto_df"] = result_df
    return new_result


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    if not isinstance(value, (dict, list, tuple)):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
    return value
