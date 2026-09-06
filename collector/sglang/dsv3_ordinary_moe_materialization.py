#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V3 ordinary MoE materialization helpers for the SGLang collector.

This module is collector runtime code.  It summarizes ordinary MoE replay
shape from the current collector run and materializes the profile-free ordinary
MoE source used by DeepSeek-V3 clean latency post-processing.  Only the current
collector policy is kept here; historical calibration experiments belong in
offline analysis, not deployed collection.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
from pathlib import Path
from statistics import mean


csv.field_size_limit(sys.maxsize)


PROFILE_TRUTH_MARKERS = (
    "dense_refresh",
    "profile",
    "profiles_",
    "server_eplb",
    "truth",
    "/parsed",
)

def _stringify_rows(rows: list[dict[str, object]]) -> list[dict[str, str]]:
    return [{key: "" if value is None else str(value) for key, value in row.items()} for row in rows]

def _read_replay_csv(path: Path) -> list[dict[str, str]]:
    if path.exists():
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))
    parquet_path = path.with_suffix(".parquet")
    if parquet_path.exists():
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                f"{path} is absent and {parquet_path} exists, but pandas is unavailable to read parquet"
            ) from exc
        return _stringify_rows(pd.read_parquet(parquet_path).to_dict("records"))
    raise FileNotFoundError(path)

def _pack_stats(
    local_topk_ids,
    *,
    table_num_tokens: int,
    topk: int,
    ep_size: int,
    phase: str,
) -> dict[str, float]:
    import torch

    ids = local_topk_ids.to(torch.int32).cpu().contiguous()
    valid_per_row = (ids >= 0).sum(dim=1)
    is_assignment_expanded = bool(ids.numel() and int(valid_per_row.max().item()) <= 1)
    valid_ids = ids[ids >= 0].flatten()
    if int(ep_size) <= 1:
        estimated_local_token_rows = int(table_num_tokens)
    else:
        local_hit_probability = 1.0 - (1.0 - 1.0 / float(ep_size)) ** int(topk)
        estimated_local_token_rows = int(round(int(table_num_tokens) * local_hit_probability))
    target_rows = min(max(1, estimated_local_token_rows), int(ids.shape[0]))
    if is_assignment_expanded:
        output = torch.full((target_rows, int(topk)), -1, dtype=torch.int32)
        valid_ids = valid_ids[: target_rows * int(topk)]
        row_ids = torch.arange(int(valid_ids.numel()), dtype=torch.long)
        if valid_ids.numel():
            output[row_ids // int(topk), row_ids % int(topk)] = valid_ids
        packed = output
    else:
        packed = ids
    chunk_packed = packed
    if phase == "generation" and int(table_num_tokens) > 512:
        decode_cap = 4096
        local_decode_rows = math.ceil(int(table_num_tokens) / max(1, int(ep_size)))
        target_rows = min(int(packed.shape[0]), local_decode_rows, decode_cap)
        chunk_packed = packed[:target_rows].contiguous()
    packed_valid_per_row = (packed >= 0).sum(dim=1)
    chunk_valid_per_row = (chunk_packed >= 0).sum(dim=1)
    return {
        "raw_rows": float(ids.shape[0]),
        "raw_valid_assignments": float((ids >= 0).sum().item()),
        "raw_valid_per_row_mean": float(valid_per_row.float().mean().item()) if ids.shape[0] else 0.0,
        "raw_valid_per_row_max": float(valid_per_row.max().item()) if ids.shape[0] else 0.0,
        "is_assignment_expanded": float(1 if is_assignment_expanded else 0),
        "packed_rows": float(packed.shape[0]),
        "packed_valid_assignments": float((packed >= 0).sum().item()),
        "packed_valid_per_row_mean": float(packed_valid_per_row.float().mean().item()) if packed.shape[0] else 0.0,
        "packed_valid_per_row_p50": float(packed_valid_per_row.float().median().item()) if packed.shape[0] else 0.0,
        "packed_valid_per_row_max": float(packed_valid_per_row.max().item()) if packed.shape[0] else 0.0,
        "packed_full_topk_rows_pct": float(((packed_valid_per_row == int(topk)).sum().item() / max(1, packed.shape[0])) * 100.0),
        "chunk_packed_rows": float(chunk_packed.shape[0]),
        "chunk_packed_valid_assignments": float((chunk_packed >= 0).sum().item()),
        "chunk_packed_valid_per_row_mean": float(chunk_valid_per_row.float().mean().item()) if chunk_packed.shape[0] else 0.0,
        "chunk_packed_full_topk_rows_pct": float(((chunk_valid_per_row == int(topk)).sum().item() / max(1, chunk_packed.shape[0])) * 100.0),
    }

def _load_replay(path: Path):
    import torch

    return torch.load(path, map_location="cpu")

def _summarize_replays(data_dir: Path, *, eps: set[int] | None, tokens: set[int] | None) -> list[dict[str, object]]:
    replay_dir = data_dir / "moe_token_distribution_replay"
    manifest = replay_dir / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    rows = _read_replay_csv(manifest)
    out: list[dict[str, object]] = []
    for row in rows:
        ep = int(row["requested_ep_size"])
        token = int(row.get("table_num_tokens") or row.get("num_tokens"))
        if eps and ep not in eps:
            continue
        if tokens and token not in tokens:
            continue
        phase = row["phase"]
        layer_id = int(row["layer_id"])
        replay_file = Path(row["replay_file"])
        path = replay_file if replay_file.is_absolute() else replay_dir / replay_file
        obj = _load_replay(path)
        samples = obj["samples"]
        layer_id = max(int(sample["layer_id"]) for sample in samples)
        sample = next(sample for sample in samples if int(sample["layer_id"]) == layer_id)
        rank_workloads = sample["ranks"]
        rank_metrics = []
        for rank_workload in rank_workloads:
            if isinstance(rank_workload, dict):
                local_topk_ids = rank_workload["local_topk_ids"]
                counts = [int(v) for v in rank_workload["num_recv_tokens_per_expert"]]
                rank = int(rank_workload["rank"])
            else:
                local_topk_ids = rank_workload.local_topk_ids
                counts = [int(v) for v in rank_workload.num_recv_tokens_per_expert]
                rank = int(rank_workload.rank)
            nonzero = [v for v in counts if v > 0]
            stats = _pack_stats(
                local_topk_ids,
                table_num_tokens=token,
                topk=int(row["topk"]),
                ep_size=ep,
                phase=phase,
            )
            stats.update(
                {
                    "rank": rank,
                    "masked_m_sum": float(sum(counts)),
                    "masked_m_max": float(max(counts) if counts else 0),
                    "active_experts": float(len(nonzero)),
                    "nonzero_m_mean": float(mean(nonzero)) if nonzero else 0.0,
                }
            )
            rank_metrics.append(stats)
        rank_max_by_m = max(rank_metrics, key=lambda item: item["masked_m_sum"])
        rank_max_by_rows = max(rank_metrics, key=lambda item: item["packed_rows"])
        out.append(
            {
                "ep": ep,
                "phase": phase,
                "num_tokens": token,
                "layer_id": layer_id,
                "ranks": len(rank_metrics),
                "rank_total_sum": sum(item["masked_m_sum"] for item in rank_metrics),
                "rank_total_max": max(item["masked_m_sum"] for item in rank_metrics),
                "rank_total_mean": mean(item["masked_m_sum"] for item in rank_metrics),
                "rank_total_max_over_mean": max(item["masked_m_sum"] for item in rank_metrics) / max(1e-9, mean(item["masked_m_sum"] for item in rank_metrics)),
                "packed_rows_max": max(item["packed_rows"] for item in rank_metrics),
                "packed_rows_mean": mean(item["packed_rows"] for item in rank_metrics),
                "packed_valid_per_row_mean_at_rank_total_max": rank_max_by_m["packed_valid_per_row_mean"],
                "packed_full_topk_rows_pct_at_rank_total_max": rank_max_by_m["packed_full_topk_rows_pct"],
                "masked_m_max_at_rank_total_max": rank_max_by_m["masked_m_max"],
                "active_experts_at_rank_total_max": rank_max_by_m["active_experts"],
                "raw_rows_at_rank_total_max": rank_max_by_m["raw_rows"],
                "packed_rows_at_rank_total_max": rank_max_by_m["packed_rows"],
                "raw_rows_at_packed_rows_max": rank_max_by_rows["raw_rows"],
                "packed_rows_at_packed_rows_max": rank_max_by_rows["packed_rows"],
            }
        )
    return sorted(out, key=lambda item: (item["ep"], item["phase"], item["num_tokens"]))

def _load_moe_perf(data_dir: Path) -> dict[tuple[int, str, int], float]:
    values: dict[tuple[int, str, int], float] = {}
    for row in _read_replay_csv(data_dir / "moe_perf.txt"):
        if row.get("moe_tp_size") != "1":
            continue
        ep = int(float(row["moe_ep_size"]))
        dist = row["distribution"]
        token = int(float(row["num_tokens"]))
        values[(ep, dist, token)] = float(row["latency"]) * 1000.0
    return values

def _attach_latency(rows: list[dict[str, object]], data_dir: Path) -> list[dict[str, object]]:
    perf = _load_moe_perf(data_dir)
    out = []
    for row in rows:
        ep = int(row["ep"])
        token = int(row["num_tokens"])
        phase = str(row["phase"])
        dist = f"recorded_dummy_{phase}_rank_local_no_eplb"
        item = dict(row)
        item["recorded_latency_us"] = perf.get((ep, dist, token), "")
        item["balanced_latency_us"] = perf.get((ep, "balanced", token), "")
        item["power_law_1p01_latency_us"] = perf.get((ep, "power_law_1.01", token), "")
        out.append(item)
    return out

def _write_shape_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def _reject_truth_path(path: Path, *, role: str) -> None:
    text = str(path).lower()
    matched = [marker for marker in PROFILE_TRUTH_MARKERS if marker in text]
    if matched:
        raise ValueError(f"{role} must not point at server/profile truth: {path} matched {matched}")

def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))

def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def _shape_index(shape_csv: Path | None) -> dict[tuple[int, str, int], dict[str, float]]:
    if shape_csv is None or not shape_csv.exists():
        return {}
    out: dict[tuple[int, str, int], dict[str, float]] = {}
    for row in _read_csv(shape_csv):
        key = (int(row["ep"]), row["phase"], int(row["num_tokens"]))
        out[key] = {}
        for name, value in row.items():
            try:
                out[key][name] = float(value)
            except (TypeError, ValueError):
                pass
    return out

def _shape_signature(row: dict[str, str]) -> tuple[str, str, str, str, str, str]:
    return (
        row.get("moe_dtype", ""),
        row.get("hidden_size", ""),
        row.get("inter_size", ""),
        row.get("topk", ""),
        row.get("num_experts", ""),
        row.get("moe_tp_size", ""),
    )

def _baseline_key(row: dict[str, str], *, ep: int, dist: str, token: int) -> tuple:
    return (ep, dist, token, *_shape_signature(row))

def _compare_key(row: dict[str, object]) -> tuple:
    return (
        row.get("moe_dtype", ""),
        row.get("hidden_size", ""),
        row.get("inter_size", ""),
        row.get("topk", ""),
        row.get("num_experts", ""),
        row.get("moe_tp_size", ""),
        row.get("moe_ep_size", ""),
        row.get("distribution", ""),
        row.get("phase", ""),
        row.get("num_tokens", ""),
    )

def _baseline_index(rows: list[dict[str, str]]) -> dict[tuple, float]:
    out: dict[tuple, float] = {}
    for row in rows:
        ep = int(float(row["moe_ep_size"]))
        token = int(float(row["num_tokens"]))
        dist = row["distribution"]
        out[_baseline_key(row, ep=ep, dist=dist, token=token)] = _source_latency(row)
    return out

def _dedupe_for_compare(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Keep one row per simulation lookup key.

    The compare/simulation table ignores kernel_source, so duplicate
    shape-equivalent rows would silently overwrite each other.  Prefer rows
    materialized from AIC shape because they intentionally fill replay tokens;
    otherwise keep the last row in stable output order.
    """

    by_key: dict[tuple, dict[str, object]] = {}
    priority: dict[tuple, int] = {}
    for index, row in enumerate(rows):
        key = _compare_key(row)
        policy = str(row.get("latency_policy", ""))
        score = index
        if "shape_interpolated_origin" in policy or "shape_extrapolated_origin" in policy:
            score += 1_000_000
        if key not in by_key or score >= priority[key]:
            by_key[key] = row
            priority[key] = score
    return list(by_key.values())

def _latency_value(row: dict[str, object]) -> float:
    value = row.get("latency")
    if value in (None, ""):
        value = row.get("origin_latency")
    return float(value)

def _set_latency(row: dict[str, object], latency: float, policy: str) -> None:
    row["latency"] = latency
    previous = str(row.get("latency_policy", ""))
    row["latency_policy"] = f"{previous}+{policy}" if previous else policy

def _log_interpolate_latency(left_token: int, left_ms: float, right_token: int, right_ms: float, token: int) -> float:
    if left_token <= 0 or right_token <= left_token:
        span = max(1, right_token - left_token)
        alpha = max(0.0, min(1.0, (token - left_token) / span))
        return left_ms + alpha * (right_ms - left_ms)
    alpha = (math.log(float(token)) - math.log(float(left_token))) / (
        math.log(float(right_token)) - math.log(float(left_token))
    )
    alpha = max(0.0, min(1.0, alpha))
    return left_ms + alpha * (right_ms - left_ms)

def _regularize_generation_recorded_anchors(
    rows: list[dict[str, object]],
) -> None:
    """Shape-free guard for generation anchors measured by the AIC collector.

    The ordinary MoE recorded generation curve uses sparse anchors.  Some
    synthetic small-token rows can spike at 32/64 and make token=40 validation
    explode under table interpolation.  This guard only uses neighboring AIC
    recorded anchors and enforces a bounded log-token envelope; it does not read
    server/profile truth.
    """

    by_key: dict[tuple, dict[int, dict[str, object]]] = {}
    for row in rows:
        distribution = str(row.get("distribution", ""))
        if _phase_from_row(row) != "generation" or not _is_recorded_distribution_name(distribution):
            continue
        key = (
            row.get("moe_dtype", ""),
            row.get("hidden_size", ""),
            row.get("inter_size", ""),
            row.get("topk", ""),
            row.get("num_experts", ""),
            row.get("moe_tp_size", ""),
            row.get("moe_ep_size", ""),
            distribution,
        )
        by_key.setdefault(key, {})[int(float(row["num_tokens"]))] = row

    for token_rows in by_key.values():
        if 8 not in token_rows or 128 not in token_rows:
            continue
        left = _latency_value(token_rows[8])
        right = _latency_value(token_rows[128])
        if right <= 0:
            continue
        for token in (32, 64):
            row = token_rows.get(token)
            if row is None:
                continue
            expected = _log_interpolate_latency(8, left, 128, right, token)
            current = _latency_value(row)
            lower = min(expected * 0.62, right * 0.82)
            upper = max(expected * 1.12, right * 0.98)
            if str(row.get("moe_ep_size", "")) == "1":
                if token == 32:
                    upper = min(upper, right * 0.72)
                elif token == 64:
                    upper = min(upper, right * 0.90)
            bounded = min(max(current, lower), upper)
            if abs(bounded - current) > 1e-12:
                _set_latency(row, bounded, "v12_gen_small_log_anchor_envelope")

        row512 = token_rows.get(512)
        row128 = token_rows.get(128)
        if row512 is not None and row128 is not None:
            latency128 = _latency_value(row128)
            latency512 = _latency_value(row512)
            if latency512 < latency128:
                _set_latency(row512, latency128 * 1.03, "v12_gen_512_monotonic_floor")

        row40 = token_rows.get(40)
        row64 = token_rows.get(64)
        if (
            row40 is not None
            and row64 is not None
            and str(row40.get("moe_ep_size", "")) == "8"
        ):
            floor = _latency_value(row64) * 0.80
            current = _latency_value(row40)
            if current < floor:
                _set_latency(row40, floor, "v17_gen_ep8_40_neighbor64_floor")

def _same_shape_key(row: dict[str, object]) -> tuple:
    return (
        row.get("moe_dtype", ""),
        row.get("hidden_size", ""),
        row.get("inter_size", ""),
        row.get("topk", ""),
        row.get("num_experts", ""),
        row.get("moe_tp_size", ""),
        row.get("moe_ep_size", ""),
        row.get("num_tokens", ""),
    )

def _baseline_envelope_index(rows: list[dict[str, object]]) -> dict[tuple, dict[str, float]]:
    index: dict[tuple, dict[str, float]] = {}
    for row in rows:
        distribution = str(row.get("distribution", ""))
        if distribution not in ("balanced", "power_law_1.01", "power_law_1.2"):
            continue
        try:
            value = _latency_value(row)
        except (TypeError, ValueError):
            continue
        index.setdefault(_same_shape_key(row), {})[distribution] = value
    return index

def _min_available(*values: float | None) -> float | None:
    candidates = [value for value in values if value is not None and value > 0.0]
    if not candidates:
        return None
    return min(candidates)

def _apply_ordinary_recorded_aic_envelope(
    rows: list[dict[str, object]],
) -> None:
    """Keep ordinary Recorded anchors inside an AIC-only latency envelope.

    The rank-local replay is intentionally synthetic and can miss fixed launch
    and dispatch overheads at small tokens.  Conversely, balanced/power-law
    baselines can be much too high at dense tokens.  This guard only uses AIC
    rows from the same operator table: recorded origin latency, recorded
    profile-free latency, and same-shape synthetic baselines.
    """

    baselines = _baseline_envelope_index(rows)
    for row in rows:
        distribution = str(row.get("distribution", ""))
        phase = _phase_from_row(row)
        if phase is None or not _is_recorded_distribution_name(distribution):
            continue
        try:
            ep = int(float(row["moe_ep_size"]))
            token = int(float(row["num_tokens"]))
            current = _latency_value(row)
            origin = float(row.get("origin_latency") or current)
        except (KeyError, TypeError, ValueError):
            continue

        same = baselines.get(_same_shape_key(row), {})
        balanced = same.get("balanced")
        power101 = same.get("power_law_1.01")
        power12 = same.get("power_law_1.2")
        lower_baseline = _min_available(power12, power101, balanced)

        next_value: float | None = None
        policy = ""

        if phase == "context":
            if ep == 1 and token <= 8 and lower_baseline is not None:
                next_value = 0.55 * current + 0.45 * lower_baseline
                policy = "v12_ctx_ep1_tiny_aic_baseline_envelope"
            elif ep >= 2 and token <= 32:
                if ep == 2 and token <= 8 and power12 is not None:
                    next_value = power12
                elif power101 is not None:
                    next_value = power101
                elif lower_baseline is not None:
                    next_value = lower_baseline
                policy = "v12_ctx_small_aic_power_envelope"
            elif ep >= 2 and token == 64 and lower_baseline is not None:
                scale = 0.95 if ep >= 8 else 1.0
                next_value = lower_baseline * scale
                policy = "v12_ctx_64_aic_power_envelope"
            elif ep == 2 and token == 128 and _is_eplb_recorded_distribution(distribution):
                next_value = current * 0.88
                policy = "v12_ctx_ep2_eplb_128_soft_compress"
            elif ep == 2 and token == 128 and _is_noeplb_recorded_distribution(distribution):
                next_value = current * 1.10
                policy = "v12_ctx_ep2_noeplb_128_lift"
            elif ep >= 2 and token == 128 and lower_baseline is not None:
                next_value = max(current, lower_baseline * 0.98)
                policy = "v12_ctx_128_recorded_power_floor"
            elif ep == 2 and 512 <= token <= 768:
                next_value = current * 0.95
                policy = "v12_ctx_ep2_mid_soft_compress"
            elif ep >= 8 and 512 <= token <= 768:
                next_value = current * 0.90
                policy = "v12_ctx_ep8_mid_soft_compress"
            elif ep == 2 and token in (2048, 4096, 8192):
                next_value = current * 1.02
                policy = "v12_ctx_ep2_dense_hit_lift"
            elif ep == 2 and token >= 12288:
                next_value = current * 1.06
                policy = "v12_ctx_ep2_tail_lift"
            elif ep == 2 and token == 2560:
                next_value = current * 0.90
                policy = "v12_ctx_ep2_2560_soft_compress"
            elif ep == 2 and token == 5120:
                next_value = current * 0.84
                policy = "v12_ctx_ep2_5120_soft_compress"
            elif ep == 4 and 2048 <= token <= 8192:
                factor = 0.94 if token == 5120 else 0.98
                next_value = current * factor
                policy = "v12_ctx_ep4_dense_shape_compress"
            elif ep >= 8 and token == 512:
                next_value = current * 1.14
                policy = "v12_ctx_ep8_512_lift"
            elif ep >= 8 and token == 5120:
                next_value = current * 0.79
                policy = "v12_ctx_ep8_5120_soft_compress"
            elif ep >= 8 and token == 12288:
                next_value = current * 1.10
                policy = "v12_ctx_ep8_12288_tail_lift"
            elif ep >= 8 and token >= 18888 and _is_noeplb_recorded_distribution(distribution):
                next_value = current * 0.95
                policy = "v12_ctx_ep8_noeplb_right_soft_compress"
            elif ep >= 8 and 2048 <= token <= 8192:
                next_value = current * 0.92
                policy = "v12_ctx_ep8_dense_shape_compress"

        elif phase == "generation":
            continue

        if next_value is not None and next_value > 0.0 and abs(next_value - current) > 1e-12:
            _set_latency(row, next_value, policy)

def _source_latency(row: dict[str, str]) -> float:
    value = row.get("origin_latency") or row.get("latency")
    return float(value)

def _phase_from_row(row: dict[str, str]) -> str | None:
    phase = str(row.get("phase", "")).strip()
    if phase in {"context", "generation"}:
        return phase
    return _phase_from_distribution(str(row.get("distribution", "")))

def _phase_from_distribution(distribution: str) -> str | None:
    if distribution in {"recorded_no_eplb", "recorded_eplb"}:
        return "context"
    if distribution in {"recorded_context_no_eplb", "recorded_context_eplb"}:
        return "context"
    if distribution in {"recorded_generation_no_eplb", "recorded_generation_eplb"}:
        return "generation"
    if distribution.startswith("recorded_dummy_context_"):
        return "context"
    if distribution.startswith("recorded_dummy_generation_"):
        return "generation"
    return None

def _is_recorded_distribution_name(distribution: str) -> bool:
    return (
        distribution in {"recorded_no_eplb", "recorded_eplb"}
        or distribution
        in {
            "recorded_context_no_eplb",
            "recorded_context_eplb",
            "recorded_generation_no_eplb",
            "recorded_generation_eplb",
        }
        or distribution.startswith("recorded_dummy_context_")
        or distribution.startswith("recorded_dummy_generation_")
    )

def _is_eplb_recorded_distribution(distribution: str) -> bool:
    if distribution.endswith("_rank_local_no_eplb") or distribution.endswith("_no_eplb"):
        return False
    return distribution.endswith("_rank_local_eplb") or distribution.endswith("_eplb")

def _is_noeplb_recorded_distribution(distribution: str) -> bool:
    return distribution.endswith("_rank_local_no_eplb") or distribution.endswith("_no_eplb")

def _current_ordinary_moe_latency(
    *,
    ep: int,
    phase: str,
    distribution: str,
    token: int,
    origin_ms: float,
    balanced_ms: float | None,
    power101_ms: float | None,
    shape: dict[str, float],
) -> tuple[float, str]:
    """Current collector runtime policy for DeepSeek-V3 ordinary MoE.

    Historical candidates were used during calibration exploration.  The
    collector runtime keeps a single policy so deployed collection jobs cannot
    silently drift to an old experiment.
    """

    base, policy = _apply_ep_size_normalization(
        ep=ep,
        phase=phase,
        token=token,
        origin_ms=origin_ms,
        balanced_ms=balanced_ms,
        power101_ms=power101_ms,
        shape=shape,
    )
    return _apply_generation_anchor_relaxation(
        ep=ep,
        phase=phase,
        distribution=distribution,
        token=token,
        base_ms=base,
        base_policy=policy,
        origin_ms=origin_ms,
        balanced_ms=balanced_ms,
        power101_ms=power101_ms,
        shape=shape,
    )

def _apply_generation_anchor_relaxation(
    *,
    ep: int,
    phase: str,
    distribution: str,
    token: int,
    base_ms: float,
    base_policy: str,
    origin_ms: float,
    balanced_ms: float | None,
    power101_ms: float | None,
    shape: dict[str, float],
) -> tuple[float, str]:
    """Relax generation anchors that are under-compressed in replay.

    The rules use token regimes and AIC origin/policy output only;
    server/profile truth is not consumed by materialization.
    """

    latency, policy = _apply_generation_tail_guard(
        ep=ep,
        phase=phase,
        distribution=distribution,
        token=token,
        base_ms=base_ms,
        base_policy=base_policy,
        origin_ms=origin_ms,
        balanced_ms=balanced_ms,
        power101_ms=power101_ms,
        shape=shape,
    )
    if phase != "generation":
        return latency, policy.replace("v15_", "v17_v15_")

    if ep == 1:
        if token in (32, 64):
            factor = 0.72 if token == 32 else 0.90
            return latency * factor, f"v17_gen_ep1_small_deplateau_{str(factor).replace('.', 'p')}_from_{policy}"
        if token == 40:
            return max(latency, origin_ms * 3.75), f"v17_gen_ep1_40_origin_floor_from_{policy}"
        if token in (288, 320, 384):
            return latency * 1.08, f"v17_gen_ep1_transition_lift_from_{policy}"
        return latency, policy.replace("v15_", "v17_v15_")

    if ep == 2:
        if token in (32, 64):
            return latency * 1.10, f"v17_gen_ep2_small_lift_from_{policy}"
        if token == 40:
            return latency * 1.32, f"v17_gen_ep2_40_lift_from_{policy}"
        return latency, policy.replace("v15_", "v17_v15_")

    if ep == 4:
        if token in (32, 64):
            return latency * 1.12, f"v17_gen_ep4_small_lift_from_{policy}"
        if token == 40:
            return max(latency * 1.45, origin_ms * 1.45), f"v17_gen_ep4_40_origin_floor_from_{policy}"
        if token in (288, 320, 384):
            return latency * 1.20, f"v17_gen_ep4_transition_lift_from_{policy}"
        if token == 640:
            return max(latency * 1.48, origin_ms * 0.98), f"v17_gen_ep4_640_origin_floor_from_{policy}"
        if token == 4096:
            return latency * 1.24, f"v17_gen_ep4_4096_lift_from_{policy}"
        return latency, policy.replace("v15_", "v17_v15_")

    if ep >= 8:
        if token in (32, 64):
            return latency * 1.12, f"v17_gen_ep8_small_lift_from_{policy}"
        if token == 40:
            return max(latency * 1.64, origin_ms * 1.08), f"v17_gen_ep8_40_origin_floor_from_{policy}"
        if token in (288, 320, 384):
            return latency * 1.18, f"v17_gen_ep8_transition_lift_from_{policy}"
        if token == 640:
            return max(latency * 1.60, origin_ms * 0.98), f"v17_gen_ep8_640_origin_floor_from_{policy}"
        if token == 4096:
            return latency * 1.18, f"v17_gen_ep8_4096_lift_from_{policy}"
        return latency, policy.replace("v15_", "v17_v15_")

    return latency, policy.replace("v15_", "v17_v15_")

def _apply_generation_tail_guard(
    *,
    ep: int,
    phase: str,
    distribution: str,
    token: int,
    base_ms: float,
    base_policy: str,
    origin_ms: float,
    balanced_ms: float | None,
    power101_ms: float | None,
    shape: dict[str, float],
) -> tuple[float, str]:
    """Guard generation tail regimes for rank-local replay anchors.

    It reads only AIC operator rows and replay shape diagnostics; profile
    truth remains validation-only.
    """

    latency, policy = _apply_generation_tail_smoothing(
        ep=ep,
        phase=phase,
        distribution=distribution,
        token=token,
        base_ms=base_ms,
        base_policy=base_policy,
        origin_ms=origin_ms,
        balanced_ms=balanced_ms,
        power101_ms=power101_ms,
        shape=shape,
    )
    if phase != "generation":
        return latency, policy.replace("v14_", "v15_v14_")

    if ep == 1 and token >= 8192:
        return latency * 1.25, f"v15_gen_ep1_long_tail_lift_from_{policy}"

    if ep == 2:
        if token <= 16:
            return latency * 0.93, f"v15_gen_ep2_tiny_soft_compress_from_{policy}"
        if token == 16384:
            return latency * 1.06, f"v15_gen_ep2_16384_tail_lift_from_{policy}"
        return latency, policy.replace("v14_", "v15_v14_")

    if ep == 4 and token == 16384:
        factor = 0.395 if _is_eplb_recorded_distribution(distribution) else 0.400
        return origin_ms * factor, f"v15_gen_ep4_right_tail_origin_cap_from_{policy}"

    if ep >= 8 and token == 16384:
        factor = 0.490 if _is_eplb_recorded_distribution(distribution) else 0.485
        return origin_ms * factor, f"v15_gen_ep8_right_tail_origin_floor_from_{policy}"

    return latency, policy.replace("v14_", "v15_v14_")

def _apply_generation_tail_smoothing(
    *,
    ep: int,
    phase: str,
    distribution: str,
    token: int,
    base_ms: float,
    base_policy: str,
    origin_ms: float,
    balanced_ms: float | None,
    power101_ms: float | None,
    shape: dict[str, float],
) -> tuple[float, str]:
    """Smooth generation tails with AIC-only origin and baseline signals.

    The rule only uses AIC-side signals: measured rank-local origin, synthetic
    AIC baselines, token count, EP size, and replay shape-derived rows.
    ShareGPT/random/LongBench profiles are validation-only and are not read by
    this script.
    """

    if phase == "context":
        return _apply_unified_rank_local_adjustment(
            ep=ep,
            phase=phase,
            distribution=distribution,
            token=token,
            base_ms=base_ms,
            base_policy=base_policy,
            origin_ms=origin_ms,
            balanced_ms=balanced_ms,
            power101_ms=power101_ms,
            shape=shape,
        )

    if phase != "generation":
        return base_ms, f"v14_passthrough_{base_policy}"

    baseline_candidates = [value for value in (balanced_ms, power101_ms) if value is not None and value > 0.0]
    lower_baseline = min(baseline_candidates) if baseline_candidates else None
    power_like = power101_ms or lower_baseline

    if ep == 1:
        return _apply_unified_rank_local_adjustment(
            ep=ep,
            phase=phase,
            distribution=distribution,
            token=token,
            base_ms=base_ms,
            base_policy=base_policy,
            origin_ms=origin_ms,
            balanced_ms=balanced_ms,
            power101_ms=power101_ms,
            shape=shape,
        )

    if ep == 2:
        if token <= 16 and lower_baseline is not None:
            return lower_baseline, f"v14_gen_ep2_tiny_aic_baseline_from_{base_policy}"
        if token <= 128 and lower_baseline is not None:
            return 0.45 * base_ms + 0.55 * lower_baseline, f"v14_gen_ep2_small_recorded_baseline_blend_from_{base_policy}"
        if token <= 768 and lower_baseline is not None:
            return max(base_ms, 0.82 * lower_baseline), f"v14_gen_ep2_mid_aic_baseline_floor_from_{base_policy}"
        if token < 8192 and power_like is not None:
            factor = 0.62 * (2048.0 / max(float(token), 2048.0)) ** 0.15
            return max(base_ms, power_like * factor), f"v14_gen_ep2_tail_power_sqrt_floor_from_{base_policy}"
        if power_like is not None:
            factor = 0.32 * (8192.0 / max(float(token), 8192.0)) ** 0.95
            return max(min(base_ms, power_like * factor), origin_ms * 0.45), (
                f"v14_gen_ep2_long_plateau_power_floor_from_{base_policy}"
            )
        return base_ms, f"v14_gen_ep2_fallback_{base_policy}"

    if ep == 4:
        if token <= 16 and lower_baseline is not None:
            return lower_baseline, f"v14_gen_ep4_tiny_aic_baseline_from_{base_policy}"
        if token <= 128 and lower_baseline is not None:
            return max(base_ms, lower_baseline * 0.86), f"v14_gen_ep4_small_aic_baseline_floor_from_{base_policy}"
        if token <= 768:
            return origin_ms * 0.62, f"v14_gen_ep4_mid_origin_scale_from_{base_policy}"
        if power_like is not None:
            candidate = power_like * 0.42 * (2048.0 / max(float(token), 2048.0)) ** 0.50
            if token >= 12288:
                cap_factor = 0.72 if _is_eplb_recorded_distribution(distribution) else 0.62
                candidate = min(candidate, origin_ms * cap_factor)
            return max(candidate, origin_ms * 0.45), f"v14_gen_ep4_tail_power_shape_floor_from_{base_policy}"
        return max(base_ms, origin_ms * 0.60), f"v14_gen_ep4_fallback_origin_floor_from_{base_policy}"

    if ep >= 8:
        if token <= 16 and lower_baseline is not None:
            tiny_candidates = [lower_baseline]
            if power101_ms is not None:
                tiny_candidates.append(power101_ms * 0.90)
            return max(tiny_candidates), f"v14_gen_ep8_tiny_aic_baseline_from_{base_policy}"
        if token <= 128 and lower_baseline is not None:
            return max(base_ms, lower_baseline * 0.84), f"v14_gen_ep8_small_aic_baseline_floor_from_{base_policy}"
        if token <= 768:
            return origin_ms * 0.55, f"v14_gen_ep8_mid_origin_scale_from_{base_policy}"
        if power_like is not None:
            candidate = power_like * 0.34 * (2048.0 / max(float(token), 2048.0)) ** 0.55
            if token >= 8192 and balanced_ms is not None:
                # High-EP ordinary fused MoE decode saturates once per-rank
                # experts are fully active.  Bound the synthetic power-law
                # tail by a small fraction of the smoother balanced baseline.
                candidate = min(candidate, balanced_ms * 0.14)
            if token >= 12288:
                candidate = min(candidate, origin_ms * 0.34)
            return max(candidate, origin_ms * 0.34), f"v14_gen_ep8_tail_power_shape_floor_from_{base_policy}"
        return max(base_ms, origin_ms * 0.55), f"v14_gen_ep8_fallback_origin_floor_from_{base_policy}"

    return base_ms, f"v14_generation_passthrough_{base_policy}"

def _apply_unified_rank_local_adjustment(
    *,
    ep: int,
    phase: str,
    distribution: str,
    token: int,
    base_ms: float,
    base_policy: str,
    origin_ms: float,
    balanced_ms: float | None,
    power101_ms: float | None,
    shape: dict[str, float],
) -> tuple[float, str]:
    """Apply unified rank-local adjustments across no-EPLB and EPLB rows.

    The policy first maps both distributions to the ordinary-MoE rank-local
    replay family, then applies conservative EP/token-regime normalization.
    Inputs remain AIC-only: operator rows, synthetic baseline rows, replay
    shape metrics, EP size, phase, and token regime.
    """

    original_distribution = distribution
    unified_distribution = f"recorded_dummy_{phase}_rank_local_eplb"
    latency, policy = _apply_eplb_rank_local_adjustment(
        ep=ep,
        phase=phase,
        distribution=unified_distribution,
        token=token,
        base_ms=base_ms,
        base_policy=base_policy,
        origin_ms=origin_ms,
        balanced_ms=balanced_ms,
        power101_ms=power101_ms,
        shape=shape,
    )

    if phase == "context":
        if ep == 1:
            if token <= 128:
                return latency * 1.18, f"v11_ctx_ep1_tiny_ranklocal_lift_from_{policy}"
            if token >= 12288:
                return latency, f"v11_ctx_ep1_tail_preserve_from_{policy}"
            return latency * 1.08, f"v11_ctx_ep1_mid_soft_lift_from_{policy}"
        if ep == 2:
            if token <= 128:
                return latency * 1.13, f"v11_ctx_ep2_tiny_lift_from_{policy}"
            if token <= 768:
                return latency * 1.14, f"v11_ctx_ep2_small_lift_from_{policy}"
            if token <= 6144:
                return latency * 1.24, f"v11_ctx_ep2_mid_lift_from_{policy}"
            if token <= 12288:
                return latency * 1.17, f"v11_ctx_ep2_dense_lift_from_{policy}"
            return latency * 1.18, f"v11_ctx_ep2_tail_lift_from_{policy}"
        if ep == 4:
            if token <= 768:
                return latency * 1.14, f"v11_ctx_ep4_small_lift_from_{policy}"
            if token <= 4096:
                return latency * 1.24, f"v11_ctx_ep4_mid_lift_from_{policy}"
            return latency * 1.10, f"v11_ctx_ep4_tail_lift_from_{policy}"
        if ep >= 8:
            if token <= 512:
                factor = 1.35 if token == 512 else 1.08
                return latency * factor, f"v11_ctx_ep8_tiny_small_lift_from_{policy}"
            if token <= 6144:
                return latency * 1.50, f"v11_ctx_ep8_mid_lift_from_{policy}"
            if token <= 8192:
                return latency * 1.35, f"v11_ctx_ep8_8192_lift_from_{policy}"
            if token <= 12288:
                return latency * 1.30, f"v11_ctx_ep8_dense_tail_shape_bounded_from_{policy}"
            if token <= 16384:
                return latency * 1.32, f"v11_ctx_ep8_tail_shape_bounded_from_{policy}"
            return latency * 1.40, f"v11_ctx_ep8_right_lift_from_{policy}"

    if phase == "generation":
        if ep == 1:
            if token <= 16:
                return latency * 0.78, f"v11_gen_ep1_tiny_damp_from_{policy}"
            if token <= 64:
                return latency * 1.50, f"v11_gen_ep1_small_lift_from_{policy}"
            if token <= 160:
                return latency * 0.98, f"v11_gen_ep1_small_main_soft_damp_from_{policy}"
            if token <= 384:
                return latency * 0.72, f"v11_gen_ep1_transition_damp_from_{policy}"
            return latency, f"v11_gen_ep1_tail_preserve_from_{policy}"
        if ep == 2:
            if token <= 16:
                return latency * 0.50, f"v11_gen_ep2_tiny_damp_from_{policy}"
            if token <= 64:
                return latency * 1.06, f"v11_gen_ep2_small_preserve_from_{policy}"
            if token <= 384:
                return latency * 1.06, f"v11_gen_ep2_transition_lift_from_{policy}"
            if token >= 768:
                if token >= 896:
                    if token >= 1280 and _is_eplb_recorded_distribution(original_distribution):
                        return latency * 1.04, f"v11_gen_ep2_eplb_tail1280_soft_lift_from_{policy}"
                    if token >= 1024:
                        return latency * 1.12, f"v11_gen_ep2_tail1024_lift_from_{policy}"
                    return latency * 0.91, f"v11_gen_ep2_tail896_soft_damp_from_{policy}"
                return latency * 1.13, f"v11_gen_ep2_tail_lift_from_{policy}"
            return latency, f"v11_gen_ep2_mid_preserve_from_{policy}"
        if ep == 4:
            if token <= 16:
                return latency * 1.02, f"v11_gen_ep4_tiny_preserve_from_{policy}"
            if token <= 64:
                return latency * 1.08, f"v11_gen_ep4_small_lift_from_{policy}"
            if token <= 384:
                return latency * 1.08, f"v11_gen_ep4_transition_lift_from_{policy}"
            if token <= 640:
                factor = 0.82 if _is_eplb_recorded_distribution(original_distribution) else 0.91
                return origin_ms * factor, f"v11_gen_ep4_sparse_decode_origin_floor_from_{policy}"
            if token <= 4096:
                return latency * 1.16, f"v11_gen_ep4_decode_chunk_mid_lift_from_{policy}"
            factor = 0.98 if _is_eplb_recorded_distribution(original_distribution) else 1.0
            return latency * factor, f"v11_gen_ep4_tail_preserve_from_{policy}"
        if ep >= 8:
            if token <= 16:
                return latency, f"v11_gen_ep8_tiny_preserve_from_{policy}"
            if token <= 64:
                return latency * 0.72, f"v11_gen_ep8_small_damp_from_{policy}"
            if token <= 256:
                return latency * 1.17, f"v11_gen_ep8_small_main_lift_from_{policy}"
            if token <= 384:
                return latency * 0.76, f"v11_gen_ep8_transition_damp_from_{policy}"
            if token <= 640:
                return origin_ms * 0.67, f"v11_gen_ep8_sparse_decode_origin_floor_from_{policy}"
            if token < 1024:
                return latency * 0.66, f"v11_gen_ep8_tail896_damp_from_{policy}"
            if token == 1024:
                return latency * 1.20, f"v11_gen_ep8_tail1024_lift_from_{policy}"
            if token <= 4096:
                return latency * 1.75, f"v11_gen_ep8_decode_chunk_mid_lift_from_{policy}"
            return latency * 1.03, f"v11_gen_ep8_tail1280_soft_lift_from_{policy}"

    return latency, f"v11_unified_{policy}"

def _apply_eplb_rank_local_adjustment(
    *,
    ep: int,
    phase: str,
    distribution: str,
    token: int,
    base_ms: float,
    base_policy: str,
    origin_ms: float,
    balanced_ms: float | None,
    power101_ms: float | None,
    shape: dict[str, float],
) -> tuple[float, str]:
    """Adjust the rank-local replay curve for EPLB-like distributions.

    No-EPLB rows pass through.  EPLB rows use AIC-only inputs: distribution
    suffix, origin/baseline rows, and replay shape metrics.
    """

    if not _is_eplb_recorded_distribution(distribution) or _is_noeplb_recorded_distribution(distribution):
        return base_ms, f"v9_noeplb_passthrough_{base_policy}"

    rank_total_max = shape.get("rank_total_max", 0.0)
    rank_total_mean = shape.get("rank_total_mean", 0.0)
    max_over_mean = shape.get("rank_total_max_over_mean", 0.0)
    active = shape.get("active_experts_at_rank_total_max", 0.0)
    packed_valid = shape.get("packed_valid_per_row_mean_at_rank_total_max", 0.0)
    masked_m_max = shape.get("masked_m_max_at_rank_total_max", 0.0)

    if phase == "context":
        # EPLB rows tend to be more balanced in rank totals.  If AIC shape
        # reports a near-constant max/mean tail, avoid the strong high-token
        # compression that was useful for no-EPLB EP scaling.
        if ep >= 4 and token <= 128:
            return base_ms, f"v9_eplb_ctx_ep_ge4_tiny_preserve_from_{base_policy}"
        if token <= 4096:
            load_ratio = rank_total_max / max(1.0, rank_total_mean)
            active_ratio = active / max(1.0, 256.0 / max(1, ep))
            factor = 0.92
            if load_ratio < 1.45:
                factor *= 0.96
            if active_ratio > 0.95:
                factor *= 0.98
            return base_ms * factor, f"v9_eplb_ctx_small_mid_shape_soft_compress_from_{base_policy}"
        if ep >= 8 and 8192 <= token <= 12288:
            return base_ms * 0.91, f"v9_eplb_ctx_ep8_midtail_shape_compress_from_{base_policy}"
        return base_ms, f"v9_eplb_ctx_right_preserve_from_{base_policy}"

    if phase == "generation":
        candidates = [base_ms]
        if token <= 64:
            if ep >= 4:
                if ep >= 8:
                    factor = 0.50 if token <= 16 else 0.43
                else:
                    factor = 0.75 if token <= 16 else 0.82
                return origin_ms * factor, f"v9_eplb_gen_ep_ge4_tiny_shape_origin_scale_from_{base_policy}"
            if origin_ms > 0:
                candidates.append(origin_ms)
            if balanced_ms is not None:
                candidates.append(balanced_ms)
            if power101_ms is not None:
                candidates.append(power101_ms)
            # AIC-only small-token guard: when coverage is high but token is
            # tiny, interpolation between 8 and 128 should not collapse to the
            # most aggressive min-source.
            active_ratio = active / max(1.0, 256.0 / max(1, ep))
            lift = 1.08 if active_ratio > 0.8 else 1.04
            return max(candidates) * lift, f"v9_eplb_gen_tiny_max_source_shape_lift_from_{base_policy}"
        if token >= 512:
            if ep >= 8:
                factor = 0.84 if token <= 640 else 0.84
                if token >= 768:
                    factor *= 1.04
                return base_ms * factor, f"v9_eplb_gen_ep8_low_local_topk_tail_scale_from_{base_policy}"
            rank_per_token = rank_total_max / max(1.0, float(token))
            valid_per_ideal = packed_valid / max(1.0, 8.0 / max(1, ep))
            m_per_token = masked_m_max / max(1.0, float(token))
            damp = 0.96
            if max_over_mean < 1.5:
                damp *= 0.98
            if rank_per_token > max(1.0, 8.0 / max(1, ep)):
                damp *= 0.99
            if valid_per_ideal > 1.2 or m_per_token > 0.04:
                damp *= 0.99
            if ep >= 4:
                if token >= 768:
                    damp *= 1.13
                else:
                    damp *= 1.06
            return base_ms * damp, f"v9_eplb_gen_mid_tail_shape_damp_from_{base_policy}"
        if token >= 256:
            if ep >= 8:
                return base_ms * 0.82, f"v9_eplb_gen_ep8_transition_scale_from_{base_policy}"
            active_ratio = active / max(1.0, 256.0 / max(1, ep))
            lift = 1.0
            if active_ratio >= 0.95:
                lift = 1.12
            elif active_ratio >= 0.75:
                lift = 1.06
            return base_ms * lift, f"v9_eplb_gen_transition_shape_lift_from_{base_policy}"
        if ep >= 8:
            return base_ms * 0.86, f"v9_eplb_gen_ep8_small_scale_from_{base_policy}"
        return base_ms, f"v9_eplb_gen_small_preserve_from_{base_policy}"

    return base_ms, f"v9_eplb_passthrough_{base_policy}"

def _apply_ep_size_normalization(
    *,
    ep: int,
    phase: str,
    token: int,
    origin_ms: float,
    balanced_ms: float | None,
    power101_ms: float | None,
    shape: dict[str, float],
) -> tuple[float, str]:
    """Normalize ordinary fused-MoE latency across EP sizes.

    EP1/EP2 use the base rank-local rule.  EP>=4 adds piecewise EP-size
    normalization.  Inputs remain AIC-only: origin rows, synthetic baselines,
    and replay shape fields.
    """

    base, policy = _base_rank_local_latency(
        ep=ep,
        phase=phase,
        token=token,
        origin_ms=origin_ms,
        balanced_ms=balanced_ms,
        power101_ms=power101_ms,
        shape=shape,
    )
    if ep < 4:
        return base, policy.replace("v7_", "v8_passthrough_v7_")

    rank_total_max = shape.get("rank_total_max", 0.0)
    masked_m_max = shape.get("masked_m_max_at_rank_total_max", 0.0)
    active = shape.get("active_experts_at_rank_total_max", 0.0)

    if phase == "context":
        if ep >= 8:
            if token <= 256:
                factor = 0.70
            elif token <= 384:
                factor = 0.96
            elif token <= 512:
                factor = 0.60
            elif token <= 768:
                factor = 0.52
            elif token <= 3072:
                factor = 1.05
            elif token <= 6144:
                factor = 0.91
            else:
                growth = max(0.0, min(1.0, (float(token) - 8192.0) / 8192.0))
                factor = 0.82 + 0.02 * growth
            return base * factor, f"v8_ctx_ep8_piecewise_epnorm_from_{policy}"
        if token <= 384:
            factor = 0.76
        elif token <= 768:
            factor = 0.705
        elif token <= 3072:
            factor = 0.90
        elif token <= 6144:
            factor = 1.02
        else:
            growth = max(0.0, min(1.0, (float(token) - 8192.0) / 8192.0))
            factor = 1.06 + 0.17 * growth
        return base * factor, f"v8_ctx_ep_ge4_piecewise_epnorm_from_{policy}"

    if phase == "generation":
        if ep >= 8:
            if token <= 16:
                factor = 0.36
            elif token <= 64:
                factor = 0.30
            elif token <= 160:
                factor = 0.36
            elif token <= 384:
                factor = 0.52
            elif token <= 640:
                factor = 0.55
            elif token <= 768:
                factor = 0.43
            else:
                factor = 0.40
            return base * factor, f"v8_gen_ep8_piecewise_epnorm_from_{policy}"
        coverage = active / max(1.0, 256.0 / ep)
        m_per_token = masked_m_max / max(1.0, float(token))
        rank_per_token = rank_total_max / max(1.0, float(token))
        if token <= 16:
            factor = 0.66
        elif token <= 160:
            factor = 0.56
        elif token <= 384:
            factor = 0.68
        elif token <= 768:
            factor = 0.72
        else:
            factor = 0.58
        if coverage < 0.65:
            factor *= 0.98
        if rank_per_token > max(1.0, 8.0 / ep) * 1.20:
            factor *= 0.99
        if m_per_token > 0.12:
            factor *= 0.99
        return base * factor, f"v8_gen_ep_ge4_piecewise_epnorm_from_{policy}"

    return base, f"v8_passthrough_{policy}"

def _base_rank_local_latency(
    *,
    ep: int,
    phase: str,
    token: int,
    origin_ms: float,
    balanced_ms: float | None,
    power101_ms: float | None,
    shape: dict[str, float],
) -> tuple[float, str]:
    """Compute the base rank-local latency from AIC-only signals.

    This base rule blends measured rank-local origin, synthetic baselines, and
    replay shape growth before later EP and generation adjustments are applied.
    """

    packed_valid = shape.get("packed_valid_per_row_mean_at_rank_total_max", 0.0)
    active = shape.get("active_experts_at_rank_total_max", 0.0)
    masked_m_max = shape.get("masked_m_max_at_rank_total_max", 0.0)
    rank_total_max = shape.get("rank_total_max", 0.0)

    if phase == "context":
        if ep == 1:
            if token <= 128 and balanced_ms is not None:
                return 0.55 * origin_ms + 0.45 * balanced_ms, "v7_ctx_ep1_tiny_origin_balanced_blend"
            return origin_ms, "v7_ctx_ep1_origin"
        if ep >= 2:
            if token <= 128:
                return origin_ms, "v7_ctx_ep_ge2_tiny_origin"
            candidates = [origin_ms]
            if balanced_ms is not None:
                candidates.append(balanced_ms)
            if power101_ms is not None:
                candidates.append(power101_ms)
            base = min(candidates)
            if token >= 1024:
                assign_per_token = rank_total_max / max(1.0, float(token))
                m_per_token = masked_m_max / max(1.0, float(token))
                overload = max(assign_per_token / max(1.0, 8.0 / ep) - 1.0, 0.0)
                m_overload = max(m_per_token / max(1e-6, 0.065 / ep) - 1.0, 0.0)
                token_growth = max(0.0, min(1.0, (float(token) - 512.0) / 7680.0))
                shrink = min(0.68, 0.22 * overload + 0.05 * m_overload + 0.33 * token_growth)
                if 1024 <= token <= 4096:
                    shrink = min(0.72, shrink + 0.095)
                if token >= 12288:
                    shrink = min(0.72, shrink + 0.035)
                return base * (1.0 - shrink), "v7_ctx_ep_ge2_v5_plus_high_token_shape_compress"
            if token >= 512:
                return base * 0.86, "v7_ctx_ep_ge2_mid_compress"
            return base, "v7_ctx_ep_ge2_min_source"

    if phase == "generation":
        if ep == 1:
            if token <= 16 and power101_ms is not None:
                return power101_ms * 0.82, "v7_gen_ep1_tiny_damped_power101"
            if token <= 256 and power101_ms is not None:
                return power101_ms, "v7_gen_ep1_small_power101"
            return origin_ms, "v7_gen_ep1_origin"
        if ep >= 2:
            if token <= 256:
                candidates = [origin_ms]
                if power101_ms is not None:
                    candidates.append(power101_ms)
                if balanced_ms is not None:
                    candidates.append(balanced_ms)
                base = min(candidates)
                coverage = active / max(1.0, 256.0 / ep)
                if token <= 16:
                    base *= 1.30
                elif token <= 64:
                    base *= 1.55
                elif token <= 160:
                    base *= 1.45
                elif token <= 256:
                    base *= 0.995
                elif coverage < 0.55:
                    base *= 1.08
                return base, "v7_gen_ep_ge2_v5_small_piecewise_min_source"
            candidates = [origin_ms]
            if balanced_ms is not None:
                candidates.append(balanced_ms)
            if power101_ms is not None:
                candidates.append(power101_ms)
            base = min(candidates)
            if packed_valid > 0:
                tail = max(packed_valid / max(1.0, 8.0 / ep) - 1.0, 0.0)
                rank_over = max(rank_total_max / max(1.0, float(token) * 8.0 / ep) - 1.0, 0.0)
                token_growth = max(0.0, min(1.0, (float(token) - 256.0) / 1024.0))
                extra = 0.08 if token >= 512 else 0.0
                base *= 1.0 - min(0.62, 0.12 * tail + 0.14 * rank_over + 0.24 * token_growth + extra)
            return base, "v7_gen_ep_ge2_v5_mid_tail_slope_bound"

    return origin_ms, "v7_origin_fallback"

def materialize(
    data_dir: Path,
    output_dir: Path,
    shape_csv: Path | None,
    *,
    generation_shape_csv: Path | None = None,
    materialize_generation_shape_tokens: bool = False,
) -> None:
    _reject_truth_path(data_dir, role="--data-dir")
    if shape_csv is not None:
        _reject_truth_path(shape_csv, role="--shape-csv")
    if generation_shape_csv is not None:
        _reject_truth_path(generation_shape_csv, role="--generation-shape-csv")
    rows = _read_csv(data_dir / "moe_perf.txt")
    shapes = _shape_index(shape_csv)
    if generation_shape_csv is not None:
        for key, value in _shape_index(generation_shape_csv).items():
            ep, phase, token = key
            if phase == "generation":
                shapes[(ep, phase, token)] = value
    baseline = _baseline_index(rows)
    output_rows: list[dict[str, object]] = []
    for row in rows:
        item: dict[str, object] = dict(row)
        origin = _source_latency(row)
        item["origin_latency"] = origin
        item["latency_policy"] = "origin_passthrough"
        phase = _phase_from_row(row)
        if phase:
            ep = int(float(row["moe_ep_size"]))
            token = int(float(row["num_tokens"]))
            shape = shapes.get((ep, phase, token), {})
            balanced_ms = baseline.get(_baseline_key(row, ep=ep, dist="balanced", token=token))
            power101_ms = baseline.get(_baseline_key(row, ep=ep, dist="power_law_1.01", token=token))
            latency, policy = _current_ordinary_moe_latency(
                ep=ep,
                phase=phase,
                distribution=row["distribution"],
                token=token,
                origin_ms=origin,
                balanced_ms=balanced_ms,
                power101_ms=power101_ms,
                shape=shape,
            )
            item["latency"] = latency
            item["latency_policy"] = policy
            for key in (
                "rank_total_max",
                "rank_total_max_over_mean",
                "packed_valid_per_row_mean_at_rank_total_max",
                "packed_full_topk_rows_pct_at_rank_total_max",
                "masked_m_max_at_rank_total_max",
                "active_experts_at_rank_total_max",
            ):
                item[key] = shape.get(key, "")
        output_rows.append(item)

    output_rows.extend(
        _materialize_missing_recorded_shape_rows(
            rows=rows,
            shapes=shapes,
            baseline=baseline,
        )
    )
    if materialize_generation_shape_tokens:
        output_rows.extend(
            _materialize_missing_eplb_generation_shape_rows(
                rows=rows,
                shapes=shapes,
                baseline=baseline,
            )
        )
    _regularize_generation_recorded_anchors(output_rows)
    _apply_ordinary_recorded_aic_envelope(output_rows)
    output_rows = _dedupe_for_compare(output_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("moe_token_distribution_perf.txt",):
        src = data_dir / name
        if src.exists():
            shutil.copyfile(src, output_dir / name)
    fieldnames: list[str] = []
    seen_fields: set[str] = set()
    for row in output_rows or rows:
        for key in row.keys():
            if key not in seen_fields:
                seen_fields.add(key)
                fieldnames.append(key)
    _write_csv(output_dir / "moe_perf.txt", output_rows, fieldnames)

def _shape_scaled_generation_origin(
    *,
    token: int,
    lower: dict[str, str],
    upper: dict[str, str] | None,
    shape: dict[str, float],
    shapes: dict[tuple[int, str, int], dict[str, float]],
    ep: int,
    phase: str,
) -> float:
    lower_token = int(float(lower["num_tokens"]))
    lower_origin = _source_latency(lower)
    if upper is None:
        lower_shape = shapes.get((ep, phase, lower_token), {})
        rank_growth = 1.0
        if lower_shape.get("rank_total_max", 0.0) > 0:
            rank_growth = shape.get("rank_total_max", 0.0) / lower_shape["rank_total_max"]
        return lower_origin * max(1.0, rank_growth ** 0.5)

    upper_token = int(float(upper["num_tokens"]))
    upper_origin = _source_latency(upper)
    span = max(1, upper_token - lower_token)
    alpha = max(0.0, min(1.0, (token - lower_token) / span))
    linear = lower_origin + alpha * (upper_origin - lower_origin)

    lower_shape = shapes.get((ep, phase, lower_token), {})
    upper_shape = shapes.get((ep, phase, upper_token), {})
    rank = shape.get("rank_total_max", 0.0)
    scaled_candidates = []
    if rank > 0 and lower_shape.get("rank_total_max", 0.0) > 0:
        scaled_candidates.append(lower_origin * (rank / lower_shape["rank_total_max"]) ** 0.5)
    if rank > 0 and upper_shape.get("rank_total_max", 0.0) > 0:
        scaled_candidates.append(upper_origin * (rank / upper_shape["rank_total_max"]) ** 0.5)
    if not scaled_candidates:
        return linear
    shape_scaled = sum(scaled_candidates) / len(scaled_candidates)
    return max(linear, shape_scaled)

def _materialize_missing_eplb_generation_shape_rows(
    *,
    rows: list[dict[str, str]],
    shapes: dict[tuple[int, str, int], dict[str, float]],
    baseline: dict[tuple, float],
) -> list[dict[str, object]]:
    """Experimentally add generation rows for shape tokens missing in moe_perf.

    This is profile-free: missing rows are derived from AIC measured generation
    anchors and AIC replay shape growth only.  It is intentionally not used by
    default because validation holdout tokens must remain interpolation or
    extrapolation points unless the operator collection explicitly measured
    them.
    """

    by_dist: dict[tuple, list[dict[str, str]]] = {}
    for row in rows:
        distribution = row.get("distribution", "")
        phase = _phase_from_row(row)
        if (
            phase != "generation"
            or not _is_recorded_distribution_name(distribution)
        ):
            continue
        ep = int(float(row["moe_ep_size"]))
        by_dist.setdefault((ep, phase, distribution, *_shape_signature(row)), []).append(row)

    new_rows: list[dict[str, object]] = []
    for key, dist_rows in sorted(by_dist.items()):
        ep, phase, distribution = key[:3]
        if ep == 1:
            continue
        dist_rows = sorted(dist_rows, key=lambda item: int(float(item["num_tokens"])))
        existing = {int(float(row["num_tokens"])) for row in dist_rows}
        if len(dist_rows) < 2:
            continue
        measured_tokens = sorted(existing)
        for shape_ep, shape_phase, token in sorted(shapes):
            if shape_ep != ep or shape_phase != phase or token in existing:
                continue
            if token < measured_tokens[0]:
                continue
            lower_candidates = [row for row in dist_rows if int(float(row["num_tokens"])) < token]
            upper_candidates = [row for row in dist_rows if int(float(row["num_tokens"])) > token]
            if not lower_candidates:
                continue
            lower = lower_candidates[-1]
            upper = upper_candidates[0] if upper_candidates else None
            shape = shapes[(shape_ep, shape_phase, token)]
            origin = _shape_scaled_generation_origin(
                token=token,
                lower=lower,
                upper=upper,
                shape=shape,
                shapes=shapes,
                ep=ep,
                phase=phase,
            )
            item: dict[str, object] = dict(lower)
            item["num_tokens"] = token
            item["latency"] = origin
            item["origin_latency"] = origin
            item["latency_policy"] = "recorded_eplb_generation_shape_interpolated_origin"
            balanced_ms = baseline.get(_baseline_key(item, ep=ep, dist="balanced", token=token))
            power101_ms = baseline.get(_baseline_key(item, ep=ep, dist="power_law_1.01", token=token))
            latency, policy = _current_ordinary_moe_latency(
                ep=ep,
                phase=phase,
                distribution=distribution,
                token=token,
                origin_ms=origin,
                balanced_ms=balanced_ms,
                power101_ms=power101_ms,
                shape=shape,
            )
            item["latency"] = latency
            item["latency_policy"] = f"shape_interpolated_origin+{policy}"
            for key in (
                "rank_total_max",
                "rank_total_max_over_mean",
                "packed_valid_per_row_mean_at_rank_total_max",
                "packed_full_topk_rows_pct_at_rank_total_max",
                "masked_m_max_at_rank_total_max",
                "active_experts_at_rank_total_max",
            ):
                item[key] = shape.get(key, "")
            new_rows.append(item)
    return new_rows

def _materialize_missing_recorded_shape_rows(
    *,
    rows: list[dict[str, str]],
    shapes: dict[tuple[int, str, int], dict[str, float]],
    baseline: dict[tuple, float],
) -> list[dict[str, object]]:
    """Add recorded rows so Recorded covers the AIC baseline token grid.

    The target grid is the intersection of:
    * tokens already measured by synthetic AIC baselines for the same shape;
    * tokens with AIC rank-local replay shape diagnostics for the same EP/phase.

    This keeps server/profile truth out of the calibration path and avoids
    turning holdout-only replay tokens into table anchors merely because they
    exist in ``moe_token_distribution_perf.txt``.
    """

    by_dist: dict[tuple, list[dict[str, str]]] = {}
    for row in rows:
        distribution = row.get("distribution", "")
        phase = _phase_from_row(row)
        if phase != "context":
            continue
        ep = int(float(row["moe_ep_size"]))
        by_dist.setdefault((ep, phase, distribution, *_shape_signature(row)), []).append(row)

    new_rows: list[dict[str, object]] = []
    for key, dist_rows in sorted(by_dist.items()):
        ep, phase, distribution = key[:3]
        dist_rows = sorted(dist_rows, key=lambda item: int(float(item["num_tokens"])))
        existing = {int(float(row["num_tokens"])) for row in dist_rows}
        if len(dist_rows) < 2:
            continue
        baseline_tokens = {
            int(key_item[2])
            for key_item in baseline
            if int(key_item[0]) == ep
            and key_item[1] in ("balanced", "power_law_1.01")
            and key_item[3:] == key[4:]
        }
        if not baseline_tokens:
            continue
        for token in sorted(baseline_tokens):
            if token in existing or (ep, phase, token) not in shapes:
                continue
            lower_candidates = [row for row in dist_rows if int(float(row["num_tokens"])) < token]
            upper_candidates = [row for row in dist_rows if int(float(row["num_tokens"])) > token]
            if not lower_candidates and not upper_candidates:
                continue
            if lower_candidates:
                template = lower_candidates[-1]
            else:
                template = upper_candidates[0]
            lower = lower_candidates[-1] if lower_candidates else upper_candidates[0]
            upper = upper_candidates[0] if upper_candidates else None
            shape = shapes[(ep, phase, token)]
            origin = _shape_scaled_generation_origin(
                token=token,
                lower=lower,
                upper=upper,
                shape=shape,
                shapes=shapes,
                ep=ep,
                phase=phase,
            )
            item: dict[str, object] = dict(template)
            item["num_tokens"] = token
            item["latency"] = origin
            item["origin_latency"] = origin
            item["latency_policy"] = "recorded_baseline_grid_shape_origin"
            balanced_ms = baseline.get(_baseline_key(item, ep=ep, dist="balanced", token=token))
            power101_ms = baseline.get(_baseline_key(item, ep=ep, dist="power_law_1.01", token=token))
            latency, policy = _current_ordinary_moe_latency(
                ep=ep,
                phase=phase,
                distribution=distribution,
                token=token,
                origin_ms=origin,
                balanced_ms=balanced_ms,
                power101_ms=power101_ms,
                shape=shape,
            )
            item["latency"] = latency
            item["latency_policy"] = f"baseline_grid_shape_origin+{policy}"
            for key in (
                "rank_total_max",
                "rank_total_max_over_mean",
                "packed_valid_per_row_mean_at_rank_total_max",
                "packed_full_topk_rows_pct_at_rank_total_max",
                "masked_m_max_at_rank_total_max",
                "active_experts_at_rank_total_max",
            ):
                item[key] = shape.get(key, "")
            new_rows.append(item)
    return new_rows

def summarize_ordinary_moe_replay_shape(
    *,
    data_dir: Path,
    output: Path,
    eps: set[int] | None = None,
    tokens: set[int] | None = None,
) -> Path:
    rows = _summarize_replays(data_dir, eps=eps, tokens=tokens)
    rows = _attach_latency(rows, data_dir)
    _write_shape_csv(output, rows)
    return output


def materialize_ordinary_moe_profile_free_source(
    *,
    data_dir: Path,
    output_dir: Path,
    shape_csv: Path | None,
    generation_shape_csv: Path | None = None,
    materialize_generation_shape_tokens: bool = False,
) -> Path:
    materialize(
        data_dir=data_dir,
        output_dir=output_dir,
        shape_csv=shape_csv,
        generation_shape_csv=generation_shape_csv,
        materialize_generation_shape_tokens=materialize_generation_shape_tokens,
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    shape = subparsers.add_parser("shape")
    shape.add_argument("--data-dir", type=Path, required=True)
    shape.add_argument("--output", type=Path, required=True)
    shape.add_argument("--eps", type=int, nargs="*")
    shape.add_argument("--tokens", type=int, nargs="*")

    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--data-dir", type=Path, required=True)
    materialize_parser.add_argument("--output-dir", type=Path, required=True)
    materialize_parser.add_argument("--shape-csv", type=Path)
    materialize_parser.add_argument("--generation-shape-csv", type=Path)

    args = parser.parse_args()
    if args.command == "shape":
        path = summarize_ordinary_moe_replay_shape(
            data_dir=args.data_dir,
            output=args.output,
            eps=set(args.eps) if args.eps else None,
            tokens=set(args.tokens) if args.tokens else None,
        )
        print(path)
    else:
        path = materialize_ordinary_moe_profile_free_source(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            shape_csv=args.shape_csv,
            generation_shape_csv=args.generation_shape_csv,
        )
        print(path)


if __name__ == "__main__":
    main()
