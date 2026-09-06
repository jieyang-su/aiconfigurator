# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inline source health guard for recorded MoE microbenchmarks.

The guard runs inside one collector invocation.  It selects one complete source
row from repeated measurements and writes an audit trail; it never uses truth
data and never materializes final latency itself.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Callable, Iterable


GUARD_NAME = "recorded_inline_source_health_guard_v1"
AUDIT_FILENAME = "recorded_source_health_guard_audit.csv"


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def enabled_for(*, family: str, phase: str, token: int, recorded: bool = True) -> bool:
    if not recorded:
        return False
    if not _bool_env("COLLECTOR_RECORDED_SOURCE_HEALTH_GUARD", False):
        return False
    phase = phase.lower()
    if phase == "context":
        max_tokens = _int_env("COLLECTOR_RECORDED_SOURCE_HEALTH_CONTEXT_MAX_TOKENS", 32)
    elif phase == "generation":
        max_tokens = _int_env("COLLECTOR_RECORDED_SOURCE_HEALTH_GENERATION_MAX_TOKENS", 32)
    else:
        return False
    if int(token) > max_tokens:
        return False
    family_env = os.environ.get("COLLECTOR_RECORDED_SOURCE_HEALTH_FAMILIES", "")
    if family_env.strip():
        allowed = {item.strip() for item in family_env.replace(",", " ").split() if item.strip()}
        return family in allowed
    return family in {
        "ordinary_context",
        "ordinary_generation",
        "wideep_context",
        "wideep_generation",
    }


def _latency(item: dict, field: str) -> float:
    value = float(item.get(field, float("nan")))
    return value if math.isfinite(value) and value > 0 else float("nan")


def _spread(values: list[float]) -> float:
    valid = [value for value in values if math.isfinite(value) and value > 0]
    if not valid:
        return float("inf")
    low = min(valid)
    high = max(valid)
    return high / low if low > 0 else float("inf")


def _spread_status(spread: float, stable: float, weak: float) -> str:
    if spread <= stable:
        return "stable"
    if spread <= weak:
        return "weak_stable"
    return "unstable"


def _median_index(values: list[float]) -> int:
    ordered = sorted(enumerate(values), key=lambda entry: entry[1])
    return ordered[len(ordered) // 2][0]


def _lower_cluster_index(values: list[float], *, gap_ratio: float) -> tuple[int, str]:
    ordered = sorted(enumerate(values), key=lambda entry: entry[1])
    valid = [(idx, value) for idx, value in ordered if math.isfinite(value) and value > 0]
    if len(valid) < 3:
        return _median_index(values), "median"
    gaps = []
    for pos in range(len(valid) - 1):
        left = valid[pos][1]
        right = valid[pos + 1][1]
        gaps.append((right / left if left > 0 else float("inf"), pos))
    gap, pos = max(gaps, key=lambda entry: entry[0])
    if gap < gap_ratio:
        return _median_index(values), "median"
    lower = valid[: pos + 1]
    if len(lower) < 2:
        return _median_index(values), "median_no_lower_cluster"
    return lower[len(lower) // 2][0], "lower_cluster"


def _audit_path(output_path: str | os.PathLike[str] | None) -> Path | None:
    if not _bool_env("COLLECTOR_RECORDED_SOURCE_HEALTH_AUDIT", False):
        return None
    if output_path:
        return Path(output_path) / AUDIT_FILENAME
    current = os.environ.get("COLLECTOR_CURRENT_OUTPUT_DIR")
    if current:
        return Path(current) / AUDIT_FILENAME
    return None


def _write_audit(path: Path | None, rows: Iterable[dict]) -> None:
    if path is None:
        return
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "guard",
        "family",
        "phase",
        "ep",
        "eplb",
        "token",
        "attempt",
        "sessions",
        "values_ms_json",
        "selected_session",
        "selected_value_ms",
        "spread_ratio",
        "spread_status",
        "local_shape_status",
        "action",
        "reason",
        "source_latency_field",
        "kernel_source",
    ]
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def select_recorded_source(
    *,
    family: str,
    phase: str,
    ep: int,
    eplb: str,
    token: int,
    measure_once: Callable[[], dict],
    reset_before_attempt: Callable[[], None] | None = None,
    warmup_once: Callable[[], object] | None = None,
    output_path: str | os.PathLike[str] | None = None,
    rank_print: Callable[[str], None] | None = None,
    latency_field: str = "latency",
    kernel_source: str = "",
) -> dict:
    if not enabled_for(family=family, phase=phase, token=token, recorded=True):
        return measure_once()

    sessions = max(1, _int_env("COLLECTOR_RECORDED_SOURCE_HEALTH_SESSIONS", 3))
    max_sessions = max(sessions, _int_env("COLLECTOR_RECORDED_SOURCE_HEALTH_MAX_SESSIONS", 5))
    max_retries = max(0, _int_env("COLLECTOR_RECORDED_SOURCE_HEALTH_MAX_RETRIES", 2))
    stable_spread = _float_env("COLLECTOR_RECORDED_SOURCE_HEALTH_SPREAD", 1.08)
    weak_spread = _float_env("COLLECTOR_RECORDED_SOURCE_HEALTH_MAX_SPREAD", 1.20)
    lower_cluster_gap = _float_env("COLLECTOR_RECORDED_SOURCE_HEALTH_LOWER_CLUSTER_GAP", 1.12)
    warmups = max(0, _int_env("COLLECTOR_RECORDED_SOURCE_HEALTH_WARMUPS", 1))

    all_items: list[dict] = []
    audit_rows: list[dict] = []
    selected_item: dict | None = None
    selected_reason = ""
    selected_attempt = 0
    selected_spread = float("inf")
    selected_status = "unstable"
    selected_session = 0
    selected_values: list[float] = []

    for attempt in range(max_retries + 1):
        if reset_before_attempt is not None:
            reset_before_attempt()
        if warmup_once is not None:
            for _ in range(warmups):
                warmup_once()
        measured = [measure_once() for _ in range(sessions)]
        values = [_latency(item, latency_field) for item in measured]
        spread = _spread(values)
        if spread > stable_spread and len(measured) < max_sessions:
            while len(measured) < max_sessions:
                measured.append(measure_once())
            values = [_latency(item, latency_field) for item in measured]
            spread = _spread(values)

        status = _spread_status(spread, stable_spread, weak_spread)
        if status == "unstable":
            chosen_idx, reason = _lower_cluster_index(values, gap_ratio=lower_cluster_gap)
        else:
            chosen_idx, reason = _median_index(values), "median"
        chosen = dict(measured[chosen_idx])

        action = "accept" if status in {"stable", "weak_stable"} else "retry"
        audit_rows.append(
            {
                "guard": GUARD_NAME,
                "family": family,
                "phase": phase,
                "ep": int(ep),
                "eplb": eplb,
                "token": int(token),
                "attempt": attempt + 1,
                "sessions": len(measured),
                "values_ms_json": json.dumps(values),
                "selected_session": chosen_idx + 1,
                "selected_value_ms": values[chosen_idx],
                "spread_ratio": spread,
                "spread_status": status,
                "local_shape_status": "not_evaluated_v1",
                "action": action,
                "reason": reason,
                "source_latency_field": latency_field,
                "kernel_source": kernel_source,
            }
        )
        all_items.extend(measured)
        if rank_print is not None:
            rank_print(
                "Recorded source health guard: "
                f"family={family}, phase={phase}, ep={ep}, eplb={eplb}, token={token}, "
                f"attempt={attempt + 1}, sessions={len(measured)}, spread={spread:.4f}, "
                f"status={status}, selected_session={chosen_idx + 1}, "
                f"values_ms={[round(value, 6) for value in values]}"
            )

        selected_item = chosen
        selected_reason = reason
        selected_attempt = attempt + 1
        selected_spread = spread
        selected_status = status
        selected_session = chosen_idx + 1
        selected_values = values
        if status in {"stable", "weak_stable"}:
            break

    if selected_item is None:
        values = [_latency(item, latency_field) for item in all_items]
        chosen_idx, selected_reason = _lower_cluster_index(values, gap_ratio=lower_cluster_gap)
        selected_item = dict(all_items[chosen_idx])
        selected_attempt = -1
        selected_session = chosen_idx + 1
        selected_values = values
        selected_spread = _spread(values)
        selected_status = _spread_status(selected_spread, stable_spread, weak_spread)

    selected_item.update(
        {
            "source_health_guard": GUARD_NAME,
            "source_health_status": selected_status,
            "source_health_attempt": selected_attempt,
            "source_health_sessions": len(selected_values),
            "source_health_selected_session": selected_session,
            "source_health_spread_ratio": selected_spread,
            "source_health_values_ms_json": json.dumps(selected_values),
            "source_health_reason": selected_reason,
        }
    )
    if selected_status == "unstable":
        audit_rows[-1]["action"] = "fallback_unstable"
    _write_audit(_audit_path(output_path), audit_rows)
    return selected_item
