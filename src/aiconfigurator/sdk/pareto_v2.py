# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lossless two-objective Pareto search for disaggregated inference.

This module intentionally lives beside the legacy sweep.  It reuses worker
simulation and result construction, but replaces heuristic Top-K retention
with dominance-preserving worker envelopes and a final exact frontier.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.errors import NoFeasibleConfigError
from aiconfigurator.sdk.perf_database import PerfDatabase
from aiconfigurator.sdk.speculative import SpeculativeDecodingProfile
from aiconfigurator.sdk.sweep import (
    _AUTOSCALE_TTFT_CORRECTION_FACTOR,
    _DEFAULT_DECODE_BATCH_SCHEDULE,
    _RATE_MATCH_DECODE_DEGRADATION,
    _RATE_MATCH_PREFILL_DEGRADATION,
    _get_disagg_worker_candidates,
    _rate_match_dict,
)

logger = logging.getLogger(__name__)


def _frontier_mask_2d(x: np.ndarray, y: np.ndarray, *, maximize_x: bool, maximize_y: bool) -> np.ndarray:
    """Return an exact two-dimensional non-dominated mask.

    Equal objective points are all retained so callers can inspect equivalent
    deployment configurations.  NaN/inf rows are rejected.
    """

    valid = np.isfinite(x) & np.isfinite(y)
    mask = np.zeros(len(x), dtype=bool)
    indices = np.flatnonzero(valid)
    if not len(indices):
        return mask

    xv = x[indices] if maximize_x else -x[indices]
    yv = y[indices] if maximize_y else -y[indices]
    order = np.lexsort((-yv, -xv))
    best_y = -np.inf
    previous_x = None
    previous_y = None
    previous_kept = False
    for pos in order:
        current_x = xv[pos]
        current_y = yv[pos]
        # Preserve equivalent configurations only when the first copy was
        # itself non-dominated. Otherwise a second copy of a dominated point
        # would be reintroduced, making the frontier operation non-idempotent.
        previous_kept = current_y > best_y or (
            previous_kept and current_x == previous_x and current_y == previous_y
        )
        if previous_kept:
            mask[indices[pos]] = True
            best_y = max(best_y, current_y)
        previous_x = current_x
        previous_y = current_y
    return mask


def _pareto_2d(df: pd.DataFrame, x_col: str, y_col: str, *, maximize_x: bool, maximize_y: bool) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    mask = _frontier_mask_2d(
        df[x_col].to_numpy(dtype=float),
        df[y_col].to_numpy(dtype=float),
        maximize_x=maximize_x,
        maximize_y=maximize_y,
    )
    return df.loc[mask].sort_values(x_col, ascending=not maximize_x).reset_index(drop=True)


def _worker_envelope(df: pd.DataFrame, *, role: str, require_same_tp: bool) -> pd.DataFrame:
    """Remove only workers dominated for every compatible deployment.

    Workers with different GPU cost or TP compatibility are never compared.
    Prefill minimizes corrected TTFT and maximizes throughput. Decode minimizes
    TPOT and maximizes throughput.
    """

    if df.empty:
        return df.copy()
    group_cols = ["num_total_gpus"] + (["tp"] if require_same_tp else [])
    x_col = "ttft" if role == "prefill" else "tpot"
    parts: list[pd.DataFrame] = []
    for _, group in df.groupby(group_cols, dropna=False, sort=False):
        parts.append(_pareto_2d(group, x_col, "seq/s", maximize_x=False, maximize_y=True))
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()


def _worker_templates(
    prefill_gpus: int,
    decode_gpus: int,
    prefill_num_workers: Iterable[int],
    decode_num_workers: Iterable[int],
    num_gpu_set: set[int],
    max_prefill_gpus: int | None,
    max_decode_gpus: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_values: list[int] = []
    d_values: list[int] = []
    totals: list[int] = []
    for d_num in decode_num_workers:
        d_total = decode_gpus * d_num
        if max_decode_gpus is not None and d_total > max_decode_gpus:
            continue
        for p_num in prefill_num_workers:
            p_total = prefill_gpus * p_num
            if max_prefill_gpus is not None and p_total > max_prefill_gpus:
                continue
            total = p_total + d_total
            if num_gpu_set and total not in num_gpu_set:
                continue
            p_values.append(p_num)
            d_values.append(d_num)
            totals.append(total)
    return np.asarray(p_values), np.asarray(d_values), np.asarray(totals)


def _best_worker_match(
    p_rate: float,
    d_rate: float,
    templates: tuple[np.ndarray, np.ndarray, np.ndarray],
    p_degradation: float,
    d_degradation: float,
) -> tuple[int, int] | None:
    p_num, d_num, total = templates
    if not len(total):
        return None
    efficiency = np.minimum(p_rate * p_num * p_degradation, d_rate * d_num * d_degradation) / total
    best = int(np.argmax(efficiency))
    return int(p_num[best]), int(d_num[best])


def _max_tpot(runtime_config: config.RuntimeConfig) -> float:
    values = runtime_config.tpot if isinstance(runtime_config.tpot, list) else [runtime_config.tpot]
    return float(max(values))


def _attach_source_provenance(candidate: dict, prefill: dict, decode: dict) -> dict:
    """Keep worker operation sources aligned with a rate-matched deployment."""
    candidate["_per_ops_source"] = {
        "prefill": prefill.get("_per_ops_source"),
        "decode": decode.get("_per_ops_source"),
    }
    return candidate


def _validate_frontier(
    frontier: pd.DataFrame,
    x_col: str,
    *,
    maximize_x: bool,
    num_gpu_set: set[int],
) -> None:
    """Reject malformed Pareto output before it reaches result files or plots."""
    if frontier.empty:
        raise RuntimeError("Pareto v2 produced an empty frontier")

    expected = _pareto_2d(frontier, x_col, "tokens/s/gpu", maximize_x=maximize_x, maximize_y=True)
    if len(expected) != len(frontier):
        raise RuntimeError("Pareto v2 produced dominated or non-finite objective points")

    ordered = frontier.sort_values(x_col, ascending=not maximize_x)
    efficiency = ordered["tokens/s/gpu"].to_numpy(dtype=float)
    if np.any(np.diff(efficiency) < 0):
        raise RuntimeError("Pareto v2 frontier is not monotonic in objective space")

    if num_gpu_set:
        used_gpus = set(frontier["num_total_gpus"].astype(int))
        if not used_gpus <= num_gpu_set:
            unexpected = sorted(used_gpus - num_gpu_set)
            raise RuntimeError(f"Pareto v2 returned deployments outside num_gpu_list: {unexpected}")


def sweep_disagg_pareto_v2(
    *,
    model_path: str,
    runtime_config: config.RuntimeConfig,
    prefill_database: PerfDatabase,
    prefill_backend_name: str,
    prefill_model_config: config.ModelConfig,
    prefill_parallel_config_list: list[tuple[int, int, int, int, int, int]] | list[list[int]],
    prefill_latency_correction: float,
    decode_database: PerfDatabase,
    decode_backend_name: str,
    decode_model_config: config.ModelConfig,
    decode_parallel_config_list: list[tuple[int, int, int, int, int, int]] | list[list[int]],
    decode_latency_correction: float,
    prefill_max_num_tokens: int = 16384,
    decode_max_num_tokens: int = 512,
    prefill_num_worker_list: list[int] | None = None,
    decode_num_worker_list: list[int] | None = None,
    num_gpu_list: list[int] | None = None,
    max_prefill_gpus: int | None = None,
    max_decode_gpus: int | None = None,
    require_same_tp: bool = False,
    autoscale: bool = False,
    target_tpot: float | None = None,
    rate_matching_prefill_degradation: float | None = None,
    rate_matching_decode_degradation: float | None = None,
    autoscale_ttft_correction_factor: float | None = None,
    predictor: Any = None,
    speculative_profile: SpeculativeDecodingProfile | None = None,
    free_gpu_memory_fraction: float | None = None,
) -> pd.DataFrame:
    """Return the exact feasible throughput frontier for PD disaggregation.

    ``autoscale`` is deliberately unsupported: autoscaling has a different
    objective and remains on the legacy dedicated path.
    """

    if autoscale:
        raise ValueError("Pareto v2 does not support autoscale; use pareto_algorithm='v1'")
    p_workers = prefill_num_worker_list or []
    d_workers = decode_num_worker_list or []
    if not p_workers or not d_workers:
        raise ValueError("Pareto v2 requires non-empty prefill and decode worker lists")
    if max_prefill_gpus is not None and max_prefill_gpus <= 0:
        raise ValueError("max_prefill_gpus must be > 0")
    if max_decode_gpus is not None and max_decode_gpus <= 0:
        raise ValueError("max_decode_gpus must be > 0")

    p_deg = (
        _RATE_MATCH_PREFILL_DEGRADATION
        if rate_matching_prefill_degradation is None
        else rate_matching_prefill_degradation
    )
    d_deg = (
        _RATE_MATCH_DECODE_DEGRADATION if rate_matching_decode_degradation is None else rate_matching_decode_degradation
    )
    ttft_corr = (
        _AUTOSCALE_TTFT_CORRECTION_FACTOR
        if autoscale_ttft_correction_factor is None
        else autoscale_ttft_correction_factor
    )
    num_gpu_set = set(num_gpu_list or [])

    decode_max_num_tokens = max(1, decode_max_num_tokens)
    if decode_max_num_tokens > max(_DEFAULT_DECODE_BATCH_SCHEDULE):
        decode_batches = _DEFAULT_DECODE_BATCH_SCHEDULE + [decode_max_num_tokens]
    else:
        decode_batches = [b for b in _DEFAULT_DECODE_BATCH_SCHEDULE if b <= decode_max_num_tokens]
    prefill_max_num_tokens = max(prefill_max_num_tokens, runtime_config.isl)
    prefill_batches = range(1, prefill_max_num_tokens // runtime_config.isl + 1)

    p_raw = _get_disagg_worker_candidates(
        model_path=model_path,
        model_config=prefill_model_config,
        parallel_config_list=prefill_parallel_config_list,
        b_list=prefill_batches,
        runtime_config=runtime_config,
        role="prefill",
        database=prefill_database,
        backend_name=prefill_backend_name,
        latency_correction=prefill_latency_correction,
        predictor=predictor,
        speculative_profile=speculative_profile,
        free_gpu_memory_fraction=free_gpu_memory_fraction,
    )
    d_raw = _get_disagg_worker_candidates(
        model_path=model_path,
        model_config=decode_model_config,
        parallel_config_list=decode_parallel_config_list,
        b_list=decode_batches,
        runtime_config=runtime_config,
        role="decode",
        database=decode_database,
        backend_name=decode_backend_name,
        latency_correction=decode_latency_correction,
        predictor=predictor,
        speculative_profile=speculative_profile,
        free_gpu_memory_fraction=free_gpu_memory_fraction,
    )

    p_feasible = p_raw.assign(_corrected_ttft=p_raw["ttft"] * ttft_corr)
    p_feasible = p_feasible[p_feasible["_corrected_ttft"] < runtime_config.ttft].copy()
    d_feasible = d_raw[d_raw["tpot"] < (target_tpot if target_tpot is not None else _max_tpot(runtime_config))].copy()
    if p_feasible.empty or d_feasible.empty:
        raise NoFeasibleConfigError("Pareto v2 found no worker candidates satisfying TTFT/TPOT constraints")

    # Do not prune individual workers before rate matching. Discrete worker
    # templates can make an apparently dominated worker useful at a different
    # P/D replica ratio, so the only lossy reduction is in final objective space.
    p_envelope = p_feasible.drop(columns=["_corrected_ttft"])
    d_envelope = d_feasible

    template_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    rows: list[dict] = []
    pairs_visited = 0
    request_latency_mode = runtime_config.request_latency is not None and runtime_config.request_latency > 0
    p_records = p_envelope.to_dict("records")
    d_records = d_envelope.to_dict("records")

    for d_row in d_records:
        best_choice: tuple[dict, tuple[int, int]] | None = None
        best_efficiency = -np.inf
        for p_row in p_records:
            if require_same_tp and p_row["tp"] != d_row["tp"]:
                continue
            pairs_visited += 1
            key = (int(p_row["num_total_gpus"]), int(d_row["num_total_gpus"]))
            templates = template_cache.get(key)
            if templates is None:
                templates = _worker_templates(
                    key[0], key[1], p_workers, d_workers, num_gpu_set, max_prefill_gpus, max_decode_gpus
                )
                template_cache[key] = templates
            match = _best_worker_match(float(p_row["seq/s"]), float(d_row["seq/s"]), templates, p_deg, d_deg)
            if match is None:
                continue
            if request_latency_mode:
                candidate = _attach_source_provenance(
                    _rate_match_dict(p_row, match[0], d_row, match[1], p_deg, d_deg), p_row, d_row
                )
                if candidate["request_latency"] > float(runtime_config.request_latency):
                    continue
                rows.append(candidate)
            else:
                throughput = min(
                    float(p_row["seq/s"]) * match[0] * p_deg,
                    float(d_row["seq/s"]) * match[1] * d_deg,
                )
                total_gpus = (
                    int(p_row["num_total_gpus"]) * match[0]
                    + int(d_row["num_total_gpus"]) * match[1]
                )
                efficiency = throughput * runtime_config.osl / total_gpus
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_choice = (p_row, match)
        if not request_latency_mode and best_choice is not None:
            p_row, match = best_choice
            rows.append(
                _attach_source_provenance(
                    _rate_match_dict(p_row, match[0], d_row, match[1], p_deg, d_deg), p_row, d_row
                )
            )

    if not rows:
        raise NoFeasibleConfigError("Pareto v2 found no rate-matched deployment satisfying all constraints")
    candidates = pd.DataFrame(rows, columns=[*common.ColumnsDisagg, "_per_ops_source"])
    if request_latency_mode:
        frontier = _pareto_2d(candidates, "request_latency", "tokens/s/gpu", maximize_x=False, maximize_y=True)
        frontier = _pareto_2d(
            frontier.round(3), "request_latency", "tokens/s/gpu", maximize_x=False, maximize_y=True
        )
    else:
        frontier = _pareto_2d(candidates, "tokens/s/user", "tokens/s/gpu", maximize_x=True, maximize_y=True)
        frontier = _pareto_2d(
            frontier.round(3), "tokens/s/user", "tokens/s/gpu", maximize_x=True, maximize_y=True
        )
    _validate_frontier(
        frontier,
        "request_latency" if request_latency_mode else "tokens/s/user",
        maximize_x=not request_latency_mode,
        num_gpu_set=num_gpu_set,
    )
    frontier.attrs["pareto_v2_diagnostics"] = {
        "raw_prefill_workers": len(p_raw),
        "feasible_prefill_workers": len(p_feasible),
        "envelope_prefill_workers": len(p_envelope),
        "raw_decode_workers": len(d_raw),
        "feasible_decode_workers": len(d_feasible),
        "envelope_decode_workers": len(d_envelope),
        "pd_pairs_visited": pairs_visited,
        "worker_template_groups": len(template_cache),
        "matched_candidates": len(candidates),
        "frontier_points": len(frontier),
    }
    logger.info("Pareto v2 diagnostics: %s", frontier.attrs["pareto_v2_diagnostics"])
    return frontier
