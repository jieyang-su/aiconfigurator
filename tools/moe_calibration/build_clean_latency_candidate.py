#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build an offline clean-latency MoE candidate and ShareGPT comparison report.

This tool is intentionally offline: it reads AIC collector outputs and AIC
latency source bundles, writes a sibling candidate directory, and uses profile
truth only in the final comparison report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median


csv.field_size_limit(sys.maxsize)

WIDEEP_SOURCE_FIELDS = (
    "latency",
    "latency_max",
    "latency_raw_max",
    "latency_stage_sum_rankmax",
    "latency_stage_sum_rankmax_max",
    "rank_mean_latency",
    "rank_p90_latency",
    "rank_critical_mean_latency",
    "rank_sync_tail_mean",
    "kernel_regime",
    "gemm_path",
)

WIDEEP_SOURCE_FILES = (
    "context_sparse.txt",
    "context_dense_noeplb.txt",
    "context_dense_eplb.txt",
    "generation_small_noeplb.txt",
    "generation_small_eplb.txt",
    "generation_main_noeplb.txt",
    "generation_main_eplb.txt",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _as_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _truth_guard(path: Path, *, role: str) -> None:
    markers = ("profile_validation", "rank_aggregate", "dense_refresh", "sharegpt", "longbench", "truth")
    text = str(path).lower()
    matched = [marker for marker in markers if marker in text]
    if matched:
        raise ValueError(f"{role} must be AIC-only data, got {path} matching {matched}")


def _copy_source_skeleton(source_dir: Path, candidate_dir: Path) -> None:
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    candidate_dir.mkdir(parents=True)
    for name in (
        "collection_summary_sglang.json",
        "collector.log",
        "collector_errors.log",
        "moe_token_distribution_perf.txt",
    ):
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, candidate_dir / name)
    for dirname in (
        "raw_collector_source",
        "recorded_materialized_source",
        "ordinary_moe_materialized_source",
        "aic_latency_source_bundle",
    ):
        src = source_dir / dirname
        if src.exists():
            shutil.copytree(src, candidate_dir / dirname)


def _materialized_table_source(source_dir: Path, filename: str) -> Path:
    preferred = {
        "moe_perf.txt": source_dir / "ordinary_moe_materialized_source" / "moe_perf.txt",
        "wideep_context_moe_perf.txt": source_dir
        / "recorded_materialized_source"
        / "wideep_context_moe_perf.txt",
        "wideep_generation_moe_perf.txt": source_dir
        / "recorded_materialized_source"
        / "wideep_generation_moe_perf.txt",
    }
    return preferred.get(filename, source_dir / filename) if preferred.get(filename, source_dir / filename).exists() else source_dir / filename


def _validate_inputs(source_dir: Path) -> dict[str, object]:
    manifest: dict[str, object] = {
        "source_dir": str(source_dir),
        "collection_summary_total_errors": None,
        "wideep_sources": {},
    }
    summary = source_dir / "collection_summary_sglang.json"
    if summary.exists():
        data = json.loads(summary.read_text())
        errors = int(data.get("total_errors", 0) or 0)
        manifest["collection_summary_total_errors"] = errors
        if errors:
            raise RuntimeError(f"{summary} reports total_errors={errors}")

    bundle = source_dir / "aic_latency_source_bundle" / "recorded_materialization_inputs"
    ordinary_raw = source_dir / "raw_collector_source" / "moe_perf.txt"
    if not ordinary_raw.exists():
        raise FileNotFoundError(f"missing ordinary raw source: {ordinary_raw}")
    manifest["ordinary_raw_rows"] = len(_read_csv(ordinary_raw))

    required = set(WIDEEP_SOURCE_FILES)
    missing = sorted(name for name in required if not (bundle / name).exists())
    if missing:
        raise FileNotFoundError(f"missing WideEP source files: {missing}")

    for name in WIDEEP_SOURCE_FILES:
        rows = _read_csv(bundle / name)
        recorded_rows = [row for row in rows if row.get("distribution", "").startswith("recorded")]
        if not rows:
            raise RuntimeError(f"{bundle / name} is empty")
        if not recorded_rows:
            raise RuntimeError(f"{bundle / name} contains no recorded rows")
        workload_fields = [field for field in rows[0] if field.startswith("workload_")]
        if not workload_fields:
            raise RuntimeError(f"{bundle / name} contains no workload_* fields")
        required_fields = (*WIDEEP_SOURCE_FIELDS, *workload_fields)
        missing_by_field = {
            field: sum(1 for row in recorded_rows if row.get(field, "") == "")
            for field in required_fields
        }
        missing_by_field = {field: count for field, count in missing_by_field.items() if count}
        if missing_by_field:
            raise RuntimeError(f"{bundle / name} has empty required recorded fields: {missing_by_field}")
        source_manifest = {
            "rows": len(rows),
            "recorded_rows": len(recorded_rows),
            "workload_fields": workload_fields,
            "required_fields_checked": list(required_fields),
        }
        manifest["wideep_sources"][name] = source_manifest
    return manifest


def _write_candidate_manifest(
    candidate_dir: Path,
    *,
    source_dir: Path,
    origin_dir: Path,
    report_dir: Path,
    input_manifest: dict[str, object],
) -> None:
    row_counts = {}
    for filename in ("moe_perf.txt", "wideep_context_moe_perf.txt", "wideep_generation_moe_perf.txt"):
        source_rows = _read_csv(_materialized_table_source(source_dir, filename))
        candidate_rows = _read_csv(candidate_dir / filename)
        missing_clean = sum(1 for row in candidate_rows if row.get("aic_critical_path_latency", "") == "")
        row_counts[filename] = {
            "source_rows": len(source_rows),
            "candidate_rows": len(candidate_rows),
            "missing_aic_critical_path_latency": missing_clean,
            "row_count_matches_source": len(source_rows) == len(candidate_rows),
        }
    manifest = {
        "source_dir": str(source_dir),
        "candidate_dir": str(candidate_dir),
        "origin_latency_dir": str(origin_dir),
        "current_invocation_report_dir": str(report_dir),
        "truth_usage": "profile_root is used only by compare report commands; candidate latency generation reads AIC-only source_dir files.",
        "input_validation": input_manifest,
        "candidate_row_counts": row_counts,
        "candidate_columns": [
            "origin_latency",
            "aic_critical_path_latency",
            "aic_latency_source",
            "aic_latency_policy",
            "latency",
        ],
    }
    (candidate_dir / "clean_latency_candidate_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def _wideep_source_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        str(int(float(row["moe_ep_size"]))),
        row["distribution"],
        str(int(float(row["num_tokens"]))),
        row.get("moe_dtype", "fp8_block"),
    )


def _load_wideep_sources(source_dir: Path, phase: str) -> dict[tuple[str, str, str, str], dict[str, str]]:
    bundle = source_dir / "aic_latency_source_bundle" / "recorded_materialization_inputs"
    if phase == "context":
        files = ["context_sparse.txt", "context_dense_noeplb.txt", "context_dense_eplb.txt"]
    else:
        files = [
            "generation_small_noeplb.txt",
            "generation_small_eplb.txt",
            "generation_main_noeplb.txt",
            "generation_main_eplb.txt",
        ]
    sources: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for name in files:
        for row in _read_csv(bundle / name):
            if not row.get("distribution", "").startswith("recorded"):
                continue
            if not row.get("rank_mean_latency"):
                continue
            sources.setdefault(_wideep_source_key(row), row)
    return sources


def _first(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def _first_positive(*values: float | None) -> float | None:
    for value in values:
        if value is not None and value > 0.0:
            return value
    return None


def _bounded(value: float, *, floor: float | None = None, cap: float | None = None) -> float:
    if floor is not None:
        value = max(value, floor)
    if cap is not None:
        value = min(value, cap)
    return value


def _log2_token(token: int) -> float:
    return math.log2(max(1, token))


def _scale_full_formula(
    token: int,
    ep: int,
    *,
    c0: float,
    c_ep: float,
    c_log: float,
    c_log_ep: float,
) -> float:
    log_token = _log2_token(token)
    return c0 + c_ep / max(1, ep) + c_log * log_token + c_log_ep * log_token / max(1, ep)


def _scale_ab_ep(token: int, ep: int, *, c0: float, c_ep: float) -> float:
    _ = token
    return c0 + c_ep / max(1, ep)


def _scale_ab_ep_log(token: int, ep: int, *, c0: float, c_ep: float, c_log: float) -> float:
    return c0 + c_ep / max(1, ep) + c_log * _log2_token(token)


ORDINARY_RANK_LOCAL_SCALE_COEFFICIENTS = {
    ("context", "small"): (
        0.364570360117,
        0.501388324014,
        0.344501843212,
        0.202315469191,
        -0.0868799300292,
        -0.299909196449,
        -1.28105734767,
        1.54696721652,
    ),
    ("context", "mid"): (
        2.16864634055,
        -0.0844678470839,
        -0.170053756367,
        0.0428643568908,
        -0.0304382940759,
        -0.653305296697,
        0.134590419851,
        -0.221587858451,
    ),
    ("context", "dense"): (
        0.933691020984,
        -0.154130032325,
        0.00308290514143,
        0.0732495606257,
        -0.0489482118283,
        0.49363382854,
        -0.023191535018,
        -0.0998786893378,
    ),
    ("context", "tail"): (
        0.463984265181,
        0.578263110248,
        0.00651582651217,
        0.0223814503664,
        0.257348813373,
        -0.849997146261,
        -0.0558129618187,
        0.0147082934979,
    ),
    ("generation", "small"): (
        0.989407200199,
        0.447986826969,
        0.265895496129,
        0.432573127275,
        0.734737241352,
        -1.27253633074,
        -0.815379380451,
        0.546559017492,
    ),
    ("generation", "mid"): (
        4.10252167237,
        -1.0402334299,
        -0.849555664946,
        0.244771142071,
        1.8373055026,
        -2.30779639784,
        -0.0862239155709,
        1.00140245302,
    ),
    ("generation", "dense"): (
        -1.95951776113,
        1.73890104932,
        0.729125595554,
        -0.173978934477,
        0.219773237128,
        -0.335446382493,
        -0.245462049109,
        -0.589930870771,
    ),
    ("generation", "tail"): (
        0.514482246754,
        -1.88831612253,
        1.23943633913,
        0.159698804703,
        3.87291101094,
        -1.29649632778,
        -0.429788673219,
        -1.83339812932,
    ),
}


def _ordinary_token_regime(phase: str, token: int) -> str:
    _ = phase
    if token <= 64:
        return "small"
    if token <= 512:
        return "mid"
    if token <= 2560:
        return "dense"
    return "tail"


def _select_ordinary_rank_local_latency(row: dict[str, str]) -> tuple[float, str, str] | None:
    origin = _as_float(row, "origin_latency") or _as_float(row, "latency") or 0.0
    phase = row.get("phase", "")
    token = int(float(row["num_tokens"]))
    ep = int(float(row["moe_ep_size"]))
    rank_max = _as_float(row, "ordinary_rank_local_latency_max")
    rank_mean = _as_float(row, "ordinary_rank_local_latency_mean")
    spread = _as_float(row, "ordinary_rank_local_latency_spread")
    rows_max = _as_float(row, "ordinary_rank_local_rows_max")
    masked_m_max = _as_float(row, "ordinary_rank_local_masked_m_max")
    if None in (rank_max, rank_mean, spread, rows_max, masked_m_max):
        return None
    if rank_max <= 0.0 or rank_mean <= 0.0:
        return None

    regime = _ordinary_token_regime(phase, token)
    coeffs = ORDINARY_RANK_LOCAL_SCALE_COEFFICIENTS.get((phase, regime))
    if coeffs is None:
        return None
    log_token = _log2_token(token)
    inv_ep = 1.0 / max(1, ep)
    features = (
        1.0,
        inv_ep,
        log_token,
        log_token * inv_ep,
        spread / rank_max,
        rank_max / rank_mean - 1.0,
        math.log1p(rows_max),
        math.log1p(masked_m_max),
    )
    scale = sum(coef * feature for coef, feature in zip(coeffs, features))
    scale = _bounded(scale, floor=0.05, cap=5.0)
    return (
        origin * scale,
        "origin_latency_ranklocal_workload_scale",
        f"ordinary_{phase}_{regime}_ranklocal_workload_scale_v2",
    )


def _select_wideep_latency(row: dict[str, str], source: dict[str, str] | None, phase: str) -> tuple[float, str, str]:
    origin = _as_float(row, "origin_latency") or _as_float(row, "latency") or 0.0
    if source is None:
        return origin, "final_origin_latency", "missing_rank_stage_source_passthrough"

    token = int(float(row["num_tokens"]))
    ep = int(float(row["moe_ep_size"]))
    rank_mean = _as_float(source, "rank_mean_latency")

    if phase == "context":
        if token <= 64:
            scale = _scale_ab_ep(token, ep, c0=0.843366, c_ep=1.290536)
            selected = origin * max(0.0, scale)
            policy = "wideep_context_small_origin_ep_scale"
            source_latency = _first_positive(
                _as_float(source, "latency"),
                _as_float(source, "latency_stage_sum_rankmax"),
                _as_float(source, "rank_mean_latency"),
            )
            if ep <= 2 and token in (16, 32):
                floor = origin * 1.90
                if selected < floor:
                    selected = floor
                    policy += "+wideep_context_low_ep_small_origin_floor"
            if ep >= 8 and token <= 8 and source_latency is not None:
                cap = source_latency * 1.10
                if selected > cap:
                    selected = cap
                    policy += "+wideep_context_high_ep_tiny_source_cap"
            return (
                selected,
                "origin_latency_ep_continuous_scale",
                policy,
            )
        if token <= 512:
            scale = _scale_full_formula(
                token,
                ep,
                c0=2.191765,
                c_ep=0.570095,
                c_log=-0.150327,
                c_log_ep=-0.044142,
            )
            policy = "wideep_context_mid_rankmean_ep_log_scale"
        elif token <= 2560:
            scale = _scale_full_formula(
                token,
                ep,
                c0=1.169373,
                c_ep=0.348363,
                c_log=-0.034867,
                c_log_ep=-0.015477,
            )
            policy = "wideep_context_dense_rankmean_ep_log_scale"
        else:
            scale = _scale_full_formula(
                token,
                ep,
                c0=0.570953,
                c_ep=0.247955,
                c_log=0.031231,
                c_log_ep=-0.022789,
            )
            policy = "wideep_context_tail_rankmean_ep_log_scale"
        selected = (_first(rank_mean, origin) or origin) * max(0.0, scale)
        return selected, "rank_mean_latency_ep_log_scale", policy

    direct_source_latency = _as_float(source, "latency")
    positive_direct_source_latency = direct_source_latency if direct_source_latency is not None and direct_source_latency > 0.0 else None
    recorded_source_latency = _first_positive(
        positive_direct_source_latency,
        _as_float(source, "latency_stage_sum_rankmax"),
        _as_float(source, "latency_stage_sum_rankmax_max"),
        _as_float(source, "latency_max"),
        _as_float(source, "latency_raw_max"),
        _as_float(source, "rank_p90_latency"),
        _as_float(source, "rank_critical_mean_latency"),
        _as_float(source, "rank_mean_latency"),
        origin,
    )
    if recorded_source_latency is not None:
        selected = recorded_source_latency
        policy = "wideep_generation_source_latency_with_local_envelope_guard"
        source_label = (
            "recorded_source_positive_latency"
            if positive_direct_source_latency is not None
            else "recorded_stage_latency_fallback"
        )
        stage_sum = _first_positive(
            _as_float(source, "latency_stage_sum_rankmax"),
            _as_float(source, "latency_stage_sum_rankmax_max"),
            _as_float(source, "latency_max"),
            _as_float(source, "latency_raw_max"),
        )
        if token <= 128 and stage_sum is not None and stage_sum > selected * 1.05:
            # For small decode batches the recorded latency can be closer to a
            # rank mean, while stage_sum_rankmax exposes the synchronization
            # tail on the critical path.  Blend only the observed AIC gap; this
            # is deliberately independent of EP and eplb.
            gap_weight = 0.45 if token <= 64 else 0.85
            selected = selected + (stage_sum - selected) * gap_weight
            policy += "+wideep_generation_stage_gap_blend"
        if token == 896 and stage_sum is not None and selected > stage_sum * 1.20:
            cap_factor = _bounded(1.19 - 0.056 * _log2_token(ep), floor=1.02, cap=1.14)
            cap = stage_sum * cap_factor
            if selected > cap:
                selected = cap
                policy += "+wideep_generation_tail896_stage_cap"
        elif ep >= 8 and token >= 1024 and stage_sum is not None and selected > stage_sum * 1.08:
            cap = stage_sum * 1.02
            if selected > cap:
                selected = cap
                policy += "+wideep_generation_high_ep_tail_stage_cap"
        return (
            selected,
            source_label,
            policy,
        )

    if token <= 64:
        scale = _scale_full_formula(
            token,
            ep,
            c0=-0.208697,
            c_ep=1.668734,
            c_log=0.036941,
            c_log_ep=0.730181,
        )
        policy = "wideep_generation_small_ep_continuous_overhead"
    elif token <= 512:
        scale = _scale_full_formula(
            token,
            ep,
            c0=-3.126693,
            c_ep=21.824263,
            c_log=0.490256,
            c_log_ep=-2.384881,
        )
        policy = "wideep_generation_mid_ep_continuous_slope"
    else:
        scale = _scale_ab_ep(token, ep, c0=1.074257, c_ep=0.708566)
        policy = "wideep_generation_tail_ep_continuous_slope"
    selected = origin * max(0.0, scale)
    return selected, "origin_latency_ep_continuous_scale", policy


def _materialize_wideep(source_dir: Path, candidate_dir: Path, phase: str) -> None:
    filename = f"wideep_{phase}_moe_perf.txt"
    rows = _read_csv(_materialized_table_source(source_dir, filename))
    sources = _load_wideep_sources(source_dir, phase)
    out: list[dict[str, object]] = []
    fieldnames = list(_read_csv(source_dir / filename)[0].keys())
    for extra in (
        "origin_latency",
        "gemm_path",
        "kernel_regime",
        "aic_critical_path_latency",
        "aic_latency_source",
        "aic_latency_policy",
    ):
        if extra not in fieldnames:
            fieldnames.append(extra)

    for row in rows:
        row = dict(row)
        origin = _as_float(row, "origin_latency") or _as_float(row, "latency") or 0.0
        row["origin_latency"] = row.get("origin_latency") or f"{origin:.12g}"
        if row.get("distribution", "").startswith("recorded"):
            source = sources.get(_wideep_source_key(row))
            latency, selected_source, policy = _select_wideep_latency(row, source, phase)
            row["aic_critical_path_latency"] = f"{latency:.12g}"
            row["aic_latency_source"] = selected_source
            row["aic_latency_policy"] = policy
            row["latency"] = f"{latency:.12g}"
        else:
            row["aic_critical_path_latency"] = row.get("latency", "")
            row["aic_latency_source"] = "baseline_passthrough"
            row["aic_latency_policy"] = "baseline_passthrough"
        out.append(row)
    if phase == "generation":
        _apply_wideep_generation_local_envelope(out)
    _write_csv(candidate_dir / filename, out, fieldnames)


def _wideep_group_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        row.get("framework", ""),
        row.get("version", ""),
        row.get("device", ""),
        row.get("op_name", ""),
        row.get("kernel_source", ""),
        row.get("moe_dtype", ""),
        row.get("hidden_size", ""),
        row.get("inter_size", ""),
        row.get("topk", ""),
        row.get("num_experts", ""),
        row.get("moe_tp_size", ""),
        row.get("moe_ep_size", ""),
        row.get("distribution", ""),
    )


def _wideep_generation_curve_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        row.get("framework", ""),
        row.get("version", ""),
        row.get("device", ""),
        row.get("op_name", ""),
        row.get("moe_dtype", ""),
        row.get("hidden_size", ""),
        row.get("inter_size", ""),
        row.get("topk", ""),
        row.get("num_experts", ""),
        row.get("moe_tp_size", ""),
        row.get("moe_ep_size", ""),
        row.get("distribution", ""),
        row.get("materialization_role", ""),
    )


def _apply_wideep_generation_local_envelope(rows: list[dict[str, object]]) -> None:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if not str(row.get("distribution", "")).startswith("recorded"):
            continue
        groups[_wideep_generation_curve_key(row)].append(row)

    for group_rows in groups.values():
        by_token: dict[int, dict[str, object]] = {}
        for row in group_rows:
            try:
                by_token[int(float(str(row["num_tokens"])))] = row
            except (KeyError, TypeError, ValueError):
                continue
        tokens = sorted(by_token)
        if len(tokens) < 3:
            continue

        first_token = tokens[0]
        next_positive_token = next(
            (
                token
                for token in tokens[1:]
                if str(by_token[token].get("aic_latency_source", "")).startswith("recorded_source_positive_latency")
                and _row_latency(by_token[token]) is not None
            ),
            None,
        )
        first_latency = _row_latency(by_token[first_token])
        if first_latency is not None and next_positive_token is not None and first_token < next_positive_token:
            next_latency = _row_latency(by_token[next_positive_token])
            if next_latency is not None and next_latency > 0.0:
                endpoint_estimate = next_latency * math.sqrt(first_token / next_positive_token)
                bounded = first_latency
                if endpoint_estimate > 0.0 and first_latency < endpoint_estimate * 0.70:
                    bounded = first_latency * 0.35 + endpoint_estimate * 0.65
                elif endpoint_estimate > 0.0 and first_latency > endpoint_estimate * 1.08:
                    bounded = endpoint_estimate * 0.94
                if abs(bounded - first_latency) > 1e-12:
                    _set_clean_latency(
                        by_token[first_token],
                        bounded,
                        "wideep_generation_left_endpoint_sqrt_low_guard",
                    )

                current_first = _row_latency(by_token[first_token])
                if current_first is not None and current_first > endpoint_estimate * 1.02:
                    tightened = current_first * 0.20 + endpoint_estimate * 0.80
                    _set_clean_latency(
                        by_token[first_token],
                        tightened,
                        "wideep_generation_left_endpoint_sqrt_tight",
                    )

        _apply_wideep_generation_capacity_envelope(by_token, tokens)

        for index in range(1, len(tokens) - 1):
            token = tokens[index]
            left_token = tokens[index - 1]
            right_token = tokens[index + 1]
            left_latency = _row_latency(by_token[left_token])
            current_latency = _row_latency(by_token[token])
            right_latency = _row_latency(by_token[right_token])
            if left_latency is None or current_latency is None or right_latency is None:
                continue
            local_max = max(left_latency, right_latency)
            local_min = min(left_latency, right_latency)
            bounded = current_latency
            if current_latency > local_max * 1.15:
                bounded = local_max * 1.02
            else:
                ratio = (token - left_token) / max(1, right_token - left_token)
                interpolated = left_latency + (right_latency - left_latency) * ratio
                if token >= 256 and left_token > 0 and token > 0 and right_token > left_token:
                    log_span = _log2_token(right_token) - _log2_token(left_token)
                    if log_span > 0.0:
                        log_ratio = (_log2_token(token) - _log2_token(left_token)) / log_span
                        log_interpolated = left_latency + (right_latency - left_latency) * log_ratio
                        interpolated = max(interpolated, log_interpolated)
                current_policy = str(by_token[token].get("aic_latency_policy", ""))
                if "wideep_generation_tail896_stage_cap" in current_policy:
                    continue
                if interpolated > 0.0 and current_latency < interpolated * 0.95:
                    bounded = max(interpolated, local_min)
            if abs(bounded - current_latency) > 1e-12:
                _set_clean_latency(
                    by_token[token],
                    bounded,
                    "wideep_generation_local_envelope",
                )


def _apply_wideep_generation_capacity_envelope(
    by_token: dict[int, dict[str, object]],
    tokens: list[int],
) -> None:
    for token in tokens:
        row = by_token[token]
        regime = str(row.get("kernel_regime", "")).lower()
        latency = _row_latency(row)
        if latency is None:
            continue
        if "capacity_256" in regime and 32 <= token <= 288:
            x = (_log2_token(token) - _log2_token(96)) / 1.55
            bump = 1.0 + 0.115 * math.exp(-(x * x))
            _set_clean_latency(
                row,
                latency * bump,
                "wideep_generation_capacity256_mid_smooth_envelope",
            )

    # Keep the final WideEP generation policy thin: recorded/materialized source
    # latencies already include the AIC-side post policy.  Do not add an
    # additional capacity-1024 tail floor here; it can double-count tail
    # compensation when the recorded source is already the intended final value.


def _ordinary_row_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row.get("framework", ""),
        row.get("version", ""),
        row.get("device", ""),
        row.get("op_name", ""),
        row.get("kernel_source", ""),
        row.get("moe_dtype", ""),
        str(int(float(row["num_tokens"]))),
        row.get("hidden_size", ""),
        row.get("inter_size", ""),
        row.get("topk", ""),
        row.get("num_experts", ""),
        row.get("moe_tp_size", ""),
        row.get("moe_ep_size", ""),
        row.get("distribution", ""),
        row.get("phase", ""),
    )


ORDINARY_RANK_LOCAL_SOURCE_FIELDS = (
    "ordinary_rank_local_latency_mean",
    "ordinary_rank_local_latency_p90",
    "ordinary_rank_local_latency_max",
    "ordinary_rank_local_latency_min",
    "ordinary_rank_local_latency_max_rank",
    "ordinary_rank_local_latency_spread",
    "ordinary_rank_local_workload_count_mean",
    "ordinary_rank_local_rows_max",
    "ordinary_rank_local_rows_mean",
    "ordinary_rank_local_rows_sum_mean",
    "ordinary_rank_local_masked_m_max",
    "ordinary_rank_local_active_experts_max",
    "ordinary_rank_local_assignments_max",
    "ordinary_rank_local_measurement_scope",
)


def _load_ordinary_raw_rows(source_dir: Path) -> dict[tuple[str, ...], dict[str, str]]:
    path = source_dir / "raw_collector_source" / "moe_perf.txt"
    raw: dict[tuple[str, ...], dict[str, str]] = {}
    for row in _read_csv(path):
        raw[_ordinary_row_key(row)] = row
    return raw


def _select_ordinary_latency(row: dict[str, str]) -> tuple[float, str, str]:
    rank_local_latency = _select_ordinary_rank_local_latency(row)
    if rank_local_latency is not None:
        return rank_local_latency

    origin = _as_float(row, "origin_latency") or _as_float(row, "latency") or 0.0
    phase = row.get("phase", "")
    token = int(float(row["num_tokens"]))
    ep = int(float(row["moe_ep_size"]))

    if phase == "context":
        if token <= 64:
            scale = _scale_full_formula(
                token,
                ep,
                c0=0.766547,
                c_ep=2.504796,
                c_log=0.084756,
                c_log_ep=-0.202636,
            )
        elif token <= 512:
            scale = _scale_ab_ep_log(
                token,
                ep,
                c0=1.732319,
                c_ep=0.519411,
                c_log=-0.137672,
            )
        elif token <= 2560:
            scale = _scale_full_formula(
                token,
                ep,
                c0=1.314202,
                c_ep=-0.228168,
                c_log=-0.095535,
                c_log_ep=0.083868,
            )
        else:
            scale = _scale_ab_ep(token, ep, c0=0.185531, c_ep=0.834240)
        return origin * max(0.0, scale), "origin_latency_token_ep_log_regime_scale", "ordinary_context_thin_token_ep_log_regime"

    if phase == "generation":
        if token <= 64:
            scale = _scale_full_formula(
                token,
                ep,
                c0=1.376128,
                c_ep=0.073716,
                c_log=-0.104834,
                c_log_ep=0.550875,
            )
        elif token <= 512:
            scale = _scale_full_formula(
                token,
                ep,
                c0=3.557332,
                c_ep=-0.879400,
                c_log=-0.344076,
                c_log_ep=0.148279,
            )
        elif token <= 2560:
            scale = _scale_full_formula(
                token,
                ep,
                c0=-0.507642,
                c_ep=1.781361,
                c_log=0.165767,
                c_log_ep=-0.201112,
            )
        else:
            scale = _scale_full_formula(
                token,
                ep,
                c0=8.024912,
                c_ep=-8.672254,
                c_log=-0.552532,
                c_log_ep=0.678228,
            )
        return origin * max(0.0, scale), "origin_latency_token_ep_log_regime_scale", "ordinary_generation_thin_token_ep_log_regime"

    return origin, "origin_latency", "ordinary_non_recorded_origin_passthrough"


def _materialize_ordinary(source_dir: Path, candidate_dir: Path) -> None:
    rows = _read_csv(_materialized_table_source(source_dir, "moe_perf.txt"))
    raw_rows = _load_ordinary_raw_rows(source_dir)
    out: list[dict[str, object]] = []
    fieldnames = list(_read_csv(source_dir / "moe_perf.txt")[0].keys())
    for extra in (
        "origin_latency",
        "aic_critical_path_latency",
        "aic_latency_source",
        "aic_latency_policy",
    ):
        if extra not in fieldnames:
            fieldnames.append(extra)
    for row in rows:
        row = dict(row)
        raw_row = raw_rows.get(_ordinary_row_key(row), {})
        raw_origin = _as_float(raw_row, "latency") if raw_row else None
        origin = raw_origin or _as_float(row, "origin_latency") or _as_float(row, "latency") or 0.0
        row["origin_latency"] = f"{origin:.12g}"
        for field in ORDINARY_RANK_LOCAL_SOURCE_FIELDS:
            if raw_row.get(field, "") != "":
                row[field] = raw_row[field]
        if row.get("distribution", "").startswith("recorded"):
            latency, source, policy = _select_ordinary_latency(row)
            row["latency"] = f"{latency:.12g}"
            row["aic_critical_path_latency"] = f"{latency:.12g}"
            if row.get("ordinary_rank_local_latency_max", "") != "":
                source = f"ordinary_rank_local_replay_source_fields+{source}"
            row["aic_latency_source"] = f"raw_collector_source_moe_perf+{source}"
            row["aic_latency_policy"] = policy
        else:
            row["aic_critical_path_latency"] = row.get("latency", "")
            row["aic_latency_source"] = "baseline_passthrough"
            row["aic_latency_policy"] = "baseline_passthrough"
        out.append(row)
    _apply_ordinary_context_small_log_high_cap(out)
    _apply_ordinary_generation_local_envelope(out)
    _write_csv(candidate_dir / "moe_perf.txt", out, fieldnames)


def _ordinary_group_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        row.get("framework", ""),
        row.get("version", ""),
        row.get("device", ""),
        row.get("op_name", ""),
        row.get("kernel_source", ""),
        row.get("moe_dtype", ""),
        row.get("hidden_size", ""),
        row.get("inter_size", ""),
        row.get("topk", ""),
        row.get("num_experts", ""),
        row.get("moe_tp_size", ""),
        row.get("moe_ep_size", ""),
        row.get("distribution", ""),
        row.get("phase", ""),
    )


def _row_latency(row: dict[str, object]) -> float | None:
    value = row.get("aic_critical_path_latency") or row.get("latency")
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0.0:
        return None
    return parsed


def _object_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _set_clean_latency(row: dict[str, object], value: float, policy_suffix: str) -> None:
    row["latency"] = f"{value:.12g}"
    row["aic_critical_path_latency"] = f"{value:.12g}"
    row["aic_latency_source"] = f"{row.get('aic_latency_source', '')}+local_envelope".strip("+")
    current_policy = str(row.get("aic_latency_policy", ""))
    if policy_suffix not in current_policy:
        row["aic_latency_policy"] = f"{current_policy}+{policy_suffix}".strip("+")


def _apply_ordinary_context_small_log_high_cap(rows: list[dict[str, object]]) -> None:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("phase") != "context":
            continue
        if not str(row.get("distribution", "")).startswith("recorded"):
            continue
        groups[_ordinary_group_key(row)].append(row)

    for group_rows in groups.values():
        by_token: dict[int, dict[str, object]] = {}
        for row in group_rows:
            try:
                by_token[int(float(str(row["num_tokens"])))] = row
            except (KeyError, TypeError, ValueError):
                continue
        tokens = sorted(by_token)
        if len(tokens) < 3:
            continue

        for index in range(1, len(tokens) - 1):
            token = tokens[index]
            if token > 64:
                continue
            left_token = tokens[index - 1]
            right_token = tokens[index + 1]
            if left_token <= 0 or right_token <= left_token:
                continue
            left_latency = _row_latency(by_token[left_token])
            current_latency = _row_latency(by_token[token])
            right_latency = _row_latency(by_token[right_token])
            if left_latency is None or current_latency is None or right_latency is None:
                continue
            log_span = _log2_token(right_token) - _log2_token(left_token)
            if log_span <= 0.0:
                continue
            ratio = (_log2_token(token) - _log2_token(left_token)) / log_span
            interpolated = left_latency + (right_latency - left_latency) * ratio
            if interpolated <= 0.0:
                continue
            if current_latency > interpolated * 1.20:
                _set_clean_latency(
                    by_token[token],
                    interpolated * 1.05,
                    "ordinary_context_small_log_high_cap",
                )

        for token, row in by_token.items():
            if token > 8:
                continue
            current_latency = _row_latency(row)
            origin_latency = _object_float(row.get("origin_latency"))
            rank_max = _object_float(row.get("ordinary_rank_local_latency_max"))
            rank_mean = _object_float(row.get("ordinary_rank_local_latency_mean"))
            spread = _object_float(row.get("ordinary_rank_local_latency_spread"))
            rows_max = _object_float(row.get("ordinary_rank_local_rows_max"))
            imbalance = _object_float(row.get("rank_total_max_over_mean"))
            if (
                current_latency is None
                or origin_latency is None
                or rank_max is None
                or rank_mean is None
                or rows_max is None
                or imbalance is None
                or origin_latency <= 0.0
                or rank_max <= 0.0
                or rank_mean <= 0.0
            ):
                continue
            if imbalance <= 1.10:
                if token <= 4 and rows_max <= 4.0:
                    cap = origin_latency * 2.25
                    if cap > 0.0 and current_latency > cap:
                        _set_clean_latency(
                            row,
                            cap,
                            "ordinary_context_tiny_uniform_origin_cap",
                        )
                continue
            origin_over_rank = origin_latency / rank_max
            spread_ratio = (spread if spread is not None else max(0.0, rank_max - rank_mean)) / rank_max
            if rows_max <= 3.0:
                if token <= 4:
                    cap_factor = _bounded(0.88 - 0.55 * spread_ratio, floor=0.72, cap=0.90)
                else:
                    cap_factor = 1.05
            elif imbalance >= 1.60:
                cap_factor = 1.10 if token <= 4 else 1.05 if rows_max <= 5.0 else 1.35
            elif token <= 4:
                if rows_max <= 4.0 and origin_latency >= 0.40 and spread_ratio <= 0.05:
                    cap_factor = 1.00
                elif rows_max <= 4.0 and spread_ratio >= 0.25:
                    cap_factor = 1.45
                else:
                    cap_factor = 1.05 if origin_latency >= 0.40 or origin_over_rank >= 1.50 else 1.55
            else:
                cap_factor = 1.45
            cap = origin_latency * cap_factor
            if cap > 0.0 and current_latency > cap:
                _set_clean_latency(
                    row,
                    cap,
                    "ordinary_context_tiny_origin_ranklocal_cap",
                )

        for token, row in by_token.items():
            if token not in (32, 64):
                continue
            current_latency = _row_latency(row)
            origin_latency = _object_float(row.get("origin_latency"))
            rows_max = _object_float(row.get("ordinary_rank_local_rows_max"))
            imbalance = _object_float(row.get("rank_total_max_over_mean"))
            if (
                current_latency is None
                or origin_latency is None
                or rows_max is None
                or imbalance is None
                or origin_latency <= 0.0
            ):
                continue
            if imbalance >= 1.60 and rows_max >= 32.0:
                cap = origin_latency * (1.25 if token == 64 else 1.55)
                if current_latency > cap:
                    _set_clean_latency(
                        row,
                        cap,
                        "ordinary_context_small_imbalance_origin_envelope",
                    )

        token4 = by_token.get(4)
        token8 = by_token.get(8)
        token32 = by_token.get(32)
        if token4 is not None and token8 is not None and token32 is not None:
            origin4 = _object_float(token4.get("origin_latency"))
            origin8 = _object_float(token8.get("origin_latency"))
            origin32 = _object_float(token32.get("origin_latency"))
            current8 = _row_latency(token8)
            imbalance8 = _object_float(token8.get("rank_total_max_over_mean"))
            if (
                origin4 is not None
                and origin8 is not None
                and origin32 is not None
                and current8 is not None
                and imbalance8 is not None
                and origin4 > 0.0
                and origin8 > 0.0
                and origin32 > 0.0
                and imbalance8 >= 1.20
            ):
                rise_4_to_8 = origin8 / origin4
                rise_8_to_32 = origin32 / origin8
                if 1.05 <= rise_4_to_8 <= 1.30 and rise_8_to_32 >= 1.60:
                    log_ratio = (_log2_token(8) - _log2_token(4)) / (
                        _log2_token(32) - _log2_token(4)
                    )
                    origin_floor = origin4 + (origin32 - origin4) * log_ratio
                    scale8 = current8 / origin8
                    floor = origin_floor * scale8
                    if floor > current8:
                        _set_clean_latency(
                            token8,
                            floor,
                            "ordinary_context_token8_origin_valley_floor",
                        )

        token64 = by_token.get(64)
        if (
            token8 is not None
            and token32 is not None
            and token64 is not None
            and str(token32.get("distribution", "")) == "recorded_no_eplb"
            and str(token32.get("moe_tp_size", "")) == "1"
        ):
            latency8 = _row_latency(token8)
            latency32 = _row_latency(token32)
            latency64 = _row_latency(token64)
            if latency8 is not None and latency32 is not None and latency64 is not None:
                log_span = _log2_token(64) - _log2_token(8)
                if log_span > 0.0:
                    log_ratio = (_log2_token(32) - _log2_token(8)) / log_span
                    interpolated = latency8 + (latency64 - latency8) * log_ratio
                    if interpolated > 0.0 and latency32 < interpolated * 0.90:
                        _set_clean_latency(
                            token32,
                            interpolated * 0.93,
                            "ordinary_context_token32_log_floor",
                        )


def _apply_ordinary_generation_local_envelope(rows: list[dict[str, object]]) -> None:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("phase") != "generation":
            continue
        if not str(row.get("distribution", "")).startswith("recorded"):
            continue
        groups[_ordinary_group_key(row)].append(row)

    for group_rows in groups.values():
        by_token: dict[int, dict[str, object]] = {}
        for row in group_rows:
            try:
                by_token[int(float(str(row["num_tokens"])))] = row
            except (KeyError, TypeError, ValueError):
                continue
        tokens = sorted(by_token)
        if len(tokens) < 3:
            continue

        # Pull isolated middle-token spikes/dips back to the local AIC envelope.
        for index in range(1, len(tokens) - 1):
            token = tokens[index]
            if token >= 8192:
                continue
            left_token = tokens[index - 1]
            right_token = tokens[index + 1]
            left_latency = _row_latency(by_token[left_token])
            current_latency = _row_latency(by_token[token])
            right_latency = _row_latency(by_token[right_token])
            if left_latency is None or current_latency is None or right_latency is None:
                continue
            ratio = (token - left_token) / max(1, right_token - left_token)
            interpolated = left_latency + (right_latency - left_latency) * ratio
            if interpolated <= 0.0:
                continue
            if current_latency > interpolated * 1.20 or current_latency < interpolated * 0.80:
                _set_clean_latency(
                    by_token[token],
                    interpolated,
                    "ordinary_generation_local_interpolation_envelope",
                )
            elif current_latency < interpolated * 0.85:
                _set_clean_latency(
                    by_token[token],
                    interpolated * 0.97,
                    "ordinary_generation_local_dip_floor",
                )
            elif 256 <= token <= 768:
                local_min = min(left_latency, right_latency)
                local_max = max(left_latency, right_latency)
                if local_min > 0.0 and local_max / local_min <= 1.25 and current_latency > local_min * 1.05:
                    _set_clean_latency(
                        by_token[token],
                        local_min * 1.05,
                        "ordinary_generation_mid_plateau_cap",
                    )

        # Generation tail is a plateau-like critical path in the server profile.
        # Keep AIC-only tail points from jumping or dipping sharply between
        # adjacent measured tokens.
        previous_latency: float | None = None
        for token in tokens:
            current_latency = _row_latency(by_token[token])
            if current_latency is None:
                continue
            if token >= 8192 and previous_latency is not None:
                bounded = _bounded(current_latency, floor=previous_latency * 0.95, cap=previous_latency * 1.02)
                if abs(bounded - current_latency) > 1e-12:
                    _set_clean_latency(
                        by_token[token],
                        bounded,
                        "ordinary_generation_tail_slope_envelope",
                    )
                    current_latency = bounded
            previous_latency = current_latency


def _write_origin_dir(candidate_dir: Path, origin_dir: Path) -> None:
    _copy_source_skeleton(candidate_dir, origin_dir)
    for filename in ("moe_perf.txt", "wideep_context_moe_perf.txt", "wideep_generation_moe_perf.txt"):
        rows = _read_csv(candidate_dir / filename)
        fieldnames = list(rows[0].keys())
        out = []
        for row in rows:
            row = dict(row)
            origin = row.get("origin_latency") or row.get("latency", "")
            if origin:
                row["latency"] = origin
            out.append(row)
        _write_csv(origin_dir / filename, out, fieldnames)


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _compare_all(repo: Path, data_dir: Path, profile_root: Path, out_dir: Path) -> None:
    ordinary = repo / "tools" / "moe_calibration" / "compare_ordinary_moe_csv.py"
    wideep = repo / "tools" / "moe_calibration" / "compare_wideep_moe_ep8_csv.py"
    for mode in ("ordinary_context", "ordinary_generation"):
        for ep in (1, 2, 4, 8):
            profile_dir = profile_root / mode / f"ep{ep}"
            if not profile_dir.exists():
                continue
            if (profile_dir / "parsed_latest_20260701").exists():
                profile_dir = profile_dir / "parsed_latest_20260701"
            eplb = ["off"] if ep == 1 else ["off", "on"]
            target = out_dir / mode / f"ep{ep}"
            _run(
                [
                    sys.executable,
                    str(ordinary),
                    "--data-dir",
                    str(data_dir),
                    "--profile-dir",
                    str(profile_dir),
                    "--output-detail",
                    str(target / "detail.csv"),
                    "--output-summary",
                    str(target / "summary.csv"),
                    "--eps",
                    str(ep),
                    "--truth-aggregation",
                    "median",
                    "--distributions",
                    "recorded",
                    "--eplb",
                    *eplb,
                ]
            )
    for mode in ("wideep_context", "wideep_generation"):
        for ep in (2, 4, 8):
            profile_dir = profile_root / mode / f"ep{ep}"
            if not profile_dir.exists():
                continue
            if (profile_dir / "parsed_latest_20260701").exists():
                profile_dir = profile_dir / "parsed_latest_20260701"
            target = out_dir / mode / f"ep{ep}"
            _run(
                [
                    sys.executable,
                    str(wideep),
                    "--data-dir",
                    str(data_dir),
                    "--profile-dir",
                    str(profile_dir),
                    "--output-detail",
                    str(target / "detail.csv"),
                    "--output-summary",
                    str(target / "summary.csv"),
                    "--moe-ep-size",
                    str(ep),
                    "--truth-aggregation",
                    "median",
                ]
            )


def _iter_detail_rows(root: Path, version: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for detail in sorted(root.glob("*/*/detail.csv")):
        mode = detail.parent.parent.name
        ep = int(detail.parent.name.removeprefix("ep"))
        for row in _read_csv(detail):
            requested = row.get("requested_distribution", "")
            if mode.startswith("ordinary") and requested != "recorded":
                continue
            if mode.startswith("wideep") and not requested.startswith("recorded"):
                continue
            rows.append({**row, "mode": mode, "ep": ep, "version": version})
    return rows


def _summarize_report(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["version"]), str(row["mode"]), str(row["eplb"]))].append(row)
    out = []
    for (version, mode, eplb), group in sorted(groups.items()):
        abs_pct = [float(r["abs_error_pct"]) for r in group]
        abs_us = [float(r["abs_error_us"]) for r in group]
        out.append(
            {
                "version": version,
                "mode": mode,
                "eplb": eplb,
                "samples": len(group),
                "mape_pct": sum(abs_pct) / len(abs_pct),
                "max_abs_error_pct": max(abs_pct),
                "mae_us": sum(abs_us) / len(abs_us),
                "max_abs_error_us": max(abs_us),
                "bad_gt10": sum(1 for value in abs_pct if value > 10.0),
            }
        )
    return out


def _candidate_meta(candidate_dir: Path) -> dict[tuple[str, int, str, str, int, str], dict[str, str]]:
    meta = {}
    files = {
        "ordinary_context": "moe_perf.txt",
        "ordinary_generation": "moe_perf.txt",
        "wideep_context": "wideep_context_moe_perf.txt",
        "wideep_generation": "wideep_generation_moe_perf.txt",
    }
    for mode, filename in files.items():
        if not (candidate_dir / filename).exists():
            continue
        for row in _read_csv(candidate_dir / filename):
            phase = "context" if mode.endswith("context") else "generation"
            if filename == "moe_perf.txt" and row.get("phase") != phase:
                continue
            if not row.get("distribution", "").startswith("recorded"):
                continue
            eplb = "on" if row["distribution"].endswith("eplb") and not row["distribution"].endswith("no_eplb") else "off"
            key = (mode, int(float(row["moe_ep_size"])), eplb, row["distribution"], int(float(row["num_tokens"])), phase)
            meta[key] = row
            meta[(mode, int(float(row["moe_ep_size"])), eplb, "recorded", int(float(row["num_tokens"])), phase)] = row
    return meta


def _write_combined_report(report_dir: Path, candidate_dir: Path) -> None:
    all_rows = []
    for version in ("origin", "clean", "current_policy"):
        all_rows.extend(_iter_detail_rows(report_dir / "_compare" / version, version))
    summary = _summarize_report(all_rows)
    _write_csv(
        report_dir / "summary.csv",
        summary,
        [
            "version",
            "mode",
            "eplb",
            "samples",
            "mape_pct",
            "max_abs_error_pct",
            "mae_us",
            "max_abs_error_us",
            "bad_gt10",
        ],
    )

    by_key: dict[tuple[str, int, str, str, int, str], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in all_rows:
        phase = str(row["phase"])
        requested = str(row["requested_distribution"])
        key = (
            str(row["mode"]),
            int(row["ep"]),
            str(row["eplb"]),
            requested,
            int(float(row["num_tokens"])),
            phase,
        )
        by_key[key][str(row["version"])] = row
    meta = _candidate_meta(candidate_dir)
    detail = []
    for key, versions in sorted(by_key.items()):
        mode, ep, eplb, distribution, token, phase = key
        clean_row = versions.get("clean", {})
        server_us = clean_row.get("server_us") or next(iter(versions.values())).get("server_us", "")
        candidate_key = key
        candidate_row = meta.get(candidate_key, {})
        out = {
            "mode": mode,
            "ep": ep,
            "eplb": eplb,
            "phase": phase,
            "num_tokens": token,
            "requested_distribution": distribution,
            "point_kind": clean_row.get("point_kind", ""),
            "server_us": server_us,
            "aic_latency_source": candidate_row.get("aic_latency_source", ""),
            "aic_latency_policy": candidate_row.get("aic_latency_policy", ""),
        }
        for version in ("origin", "clean", "current_policy"):
            row = versions.get(version, {})
            prefix = "policy" if version == "current_policy" else version
            out[f"{prefix}_us"] = row.get("aic_us", "")
            out[f"{prefix}_error_pct"] = row.get("error_pct", "")
            out[f"{prefix}_abs_error_pct"] = row.get("abs_error_pct", "")
            out[f"{prefix}_source"] = row.get("aic_source", "")
        if out["clean_abs_error_pct"] != "" and out["policy_abs_error_pct"] != "":
            out["clean_minus_policy_abs_pct"] = float(out["clean_abs_error_pct"]) - float(out["policy_abs_error_pct"])
        else:
            out["clean_minus_policy_abs_pct"] = ""
        detail.append(out)
    fields = [
        "mode",
        "ep",
        "eplb",
        "phase",
        "num_tokens",
        "requested_distribution",
        "point_kind",
        "server_us",
        "origin_us",
        "origin_error_pct",
        "origin_abs_error_pct",
        "clean_us",
        "clean_error_pct",
        "clean_abs_error_pct",
        "policy_us",
        "policy_error_pct",
        "policy_abs_error_pct",
        "clean_minus_policy_abs_pct",
        "aic_latency_source",
        "aic_latency_policy",
        "origin_source",
        "clean_source",
        "policy_source",
    ]
    _write_csv(report_dir / "detail_origin_clean_policy.csv", detail, fields)
    ranked = [row for row in detail if row["clean_minus_policy_abs_pct"] != ""]
    better = sorted(ranked, key=lambda row: float(row["clean_minus_policy_abs_pct"]))[:50]
    worse = sorted(ranked, key=lambda row: float(row["clean_minus_policy_abs_pct"]), reverse=True)[:50]
    _write_csv(report_dir / "clean_better_than_policy_top.csv", better, fields)
    _write_csv(report_dir / "clean_worse_than_policy_top.csv", worse, fields)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    source_dir = args.source_dir.resolve()
    candidate_dir = args.candidate_dir.resolve()
    profile_root = args.profile_root.resolve()
    report_dir = args.report_dir.resolve()
    origin_dir = candidate_dir.with_name(candidate_dir.name + "_origin_latency")

    _truth_guard(source_dir, role="--source-dir")
    _truth_guard(candidate_dir, role="--candidate-dir")
    input_manifest = _validate_inputs(source_dir)

    _copy_source_skeleton(source_dir, candidate_dir)
    _materialize_ordinary(source_dir, candidate_dir)
    _materialize_wideep(source_dir, candidate_dir, "context")
    _materialize_wideep(source_dir, candidate_dir, "generation")
    _write_origin_dir(candidate_dir, origin_dir)
    _write_candidate_manifest(
        candidate_dir,
        source_dir=source_dir,
        origin_dir=origin_dir,
        report_dir=report_dir,
        input_manifest=input_manifest,
    )

    compare_root = report_dir / "_compare"
    if compare_root.exists():
        shutil.rmtree(compare_root)
    for version, data_dir in (
        ("origin", origin_dir),
        ("clean", candidate_dir),
        ("current_policy", source_dir),
    ):
        _compare_all(repo, data_dir, profile_root, compare_root / version)
    _write_combined_report(report_dir, candidate_dir)

    print(f"candidate_dir={candidate_dir}")
    print(f"origin_latency_dir={origin_dir}")
    print(f"report_dir={report_dir}")


if __name__ == "__main__":
    main()
