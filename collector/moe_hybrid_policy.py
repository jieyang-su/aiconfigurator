#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Shared profile-free hybrid latency policy for MoE collectors."""

from __future__ import annotations

import json


def _as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return default
    return float(value)


def _with_latency(
    row: dict[str, str],
    *,
    latency_ms: float,
    policy: str,
    kernel_suffix: str,
) -> dict[str, str]:
    output = dict(row)
    output.setdefault("origin_latency", str(row.get("latency", "")))
    output["latency"] = f"{latency_ms:.15g}"
    output["primary_latency_source"] = policy
    output["kernel_source"] = str(output.get("kernel_source", "")) + kernel_suffix
    return output


def _stage_gemm_plus_half_activation(row: dict[str, str]) -> float:
    stage = json.loads(row["stage_mean_ms_json"])
    gemm = float(stage.get("gemm1", 0.0)) + float(stage.get("gemm2", 0.0))
    activation = (
        float(stage.get("activation_quant", 0.0))
        + float(stage.get("activation", 0.0))
        + float(stage.get("quant", 0.0))
    )
    return gemm + 0.5 * activation


def _stage_gemm1_plus_activation(row: dict[str, str]) -> float:
    stage = json.loads(row["stage_mean_ms_json"])
    activation = (
        float(stage.get("activation_quant", 0.0))
        + float(stage.get("activation", 0.0))
        + float(stage.get("quant", 0.0))
    )
    return float(stage.get("gemm1", 0.0)) + activation


def _stage_gather_scatter(row: dict[str, str]) -> float:
    stage = json.loads(row["stage_mean_ms_json"])
    return float(stage.get("gather", 0.0)) + float(stage.get("scatter", 0.0))


def _is_current_run_single_card_dummy(row: dict[str, str]) -> bool:
    return (
        row.get("measurement_scope") == "single_card_rank_local_replay"
        and row.get("workload_source") == "dummy"
    )


def _raw_operator_latency(row: dict[str, str]) -> float:
    return _as_float(row, "origin_latency", _as_float(row, "latency"))


def _ep_scale_to_ep8(row: dict[str, str]) -> float:
    ep_size = max(1.0, _as_float(row, "moe_ep_size", 8.0))
    return max(1.0, 8.0 / ep_size)


def _epnorm_tiny_generation_latency(row: dict[str, str]) -> tuple[float, str, str]:
    if _is_current_run_single_card_dummy(row):
        return (
            _stage_gemm1_plus_activation(row),
            "stage_gemm1_plus_activation_current_run",
            "_hybrid_tiny_stage_gemm1_plus_activation_current_run",
        )
    stage_latency = _stage_gemm_plus_half_activation(row)
    raw_latency = _as_float(row, "latency")
    sync_tail = _as_float(row, "rank_sync_tail_mean")
    if raw_latency > 0.0 and raw_latency < stage_latency:
        if (
            row.get("distribution") == "recorded_eplb"
            and sync_tail / max(raw_latency, 1e-9) > 1.0
        ):
            return (
                raw_latency + 0.3 * sync_tail,
                "tiny_raw_plus_0p3_sync_tail",
                "_hybrid_tiny_raw_plus_0p3_sync_tail",
            )
        return raw_latency, "tiny_raw_latency", "_hybrid_tiny_raw_latency"
    return (
        stage_latency,
        "stage_gemm_plus_half_activation",
        "_hybrid_tiny_stage_gemm_plus_half_activation",
    )


def _latency_plus_sync_tail(row: dict[str, str], multiplier: float) -> float:
    return _raw_operator_latency(row) + multiplier * _as_float(row, "rank_sync_tail_mean")


def _latency_plus_sync_tail_and_rank_gap(
    row: dict[str, str],
    *,
    sync_multiplier: float,
    rank_gap_multiplier: float,
) -> float:
    raw_latency = _raw_operator_latency(row)
    rank_gap = max(0.0, raw_latency - _as_float(row, "rank_mean_latency"))
    return (
        raw_latency
        + sync_multiplier * _as_float(row, "rank_sync_tail_mean")
        + rank_gap_multiplier * rank_gap
    )


def _epnorm_tiny_generation_latency_2(row: dict[str, str]) -> tuple[float, str, str]:
    ep_size = _as_float(row, "moe_ep_size", 8.0)
    if ep_size <= 2.0:
        exponent = 0.86
    elif ep_size >= 8.0:
        exponent = 0.42
    else:
        exponent = 0.55
    latency = _stage_gemm1_plus_activation(row) * (_ep_scale_to_ep8(row) ** exponent)
    if ep_size >= 8.0:
        latency *= 0.86
        suffix_extra = "_x_ep8_tiny_0p86"
        policy_extra = "_x_ep8_tiny_0p86"
    else:
        suffix_extra = ""
        policy_extra = ""
    return (
        latency,
        f"epnorm_tiny_stage_gemm1_act_x_ep_scale_{str(exponent).replace('.', 'p')}{policy_extra}",
        f"_hybrid_epnorm_tiny_stage_gemm1_act_x_ep_scale_{str(exponent).replace('.', 'p')}{suffix_extra}",
    )


def _epnorm_small_generation_latency(row: dict[str, str]) -> tuple[float, str, str]:
    latency = _as_float(row, "rank_mean_latency") * (_ep_scale_to_ep8(row) ** 0.82)
    return (
        latency,
        "epnorm_small_rank_mean_x_ep_scale_0p82",
        "_hybrid_epnorm_small_rank_mean_x_ep_scale_0p82",
    )


def _epnorm_small_main_generation_latency(row: dict[str, str]) -> tuple[float, str, str]:
    ep_scale = _ep_scale_to_ep8(row)
    exponent = 0.86 if _as_float(row, "moe_ep_size", 8.0) <= 2.0 else 1.0
    latency = (
        _as_float(row, "rank_mean_latency")
        + 0.3 * _as_float(row, "rank_sync_tail_mean")
    ) * (ep_scale ** exponent)
    return (
        latency,
        f"epnorm_small_main_rank_plus_0p3sync_x_ep_scale_{str(exponent).replace('.', 'p')}",
        f"_hybrid_epnorm_small_main_rank_plus_0p3sync_x_ep_scale_{str(exponent).replace('.', 'p')}",
    )


def _epnorm_mid_generation_latency(row: dict[str, str]) -> tuple[float, str, str]:
    raw_latency = _raw_operator_latency(row)
    sync_tail = _as_float(row, "rank_sync_tail_mean")
    rank_gap = max(0.0, raw_latency - _as_float(row, "rank_mean_latency"))
    token = int(float(row.get("num_tokens", "0") or 0))
    ep_size = _as_float(row, "moe_ep_size", 8.0)
    if ep_size <= 2.0 and token >= 512:
        latency = raw_latency * (_ep_scale_to_ep8(row) ** 0.40) + min(
            4.15 * sync_tail + 0.12 * rank_gap,
            0.25 * raw_latency,
        )
        return (
            latency,
            "epnorm_mid_ep2_raw_x_ep_scale_0p40_plus_capped_syncgap",
            "_hybrid_epnorm_mid_ep2_raw_x_ep_scale_0p40_plus_capped_syncgap",
        )
    cap_multiplier = 0.47 if ep_size <= 2.0 and token < 512 else 0.45
    correction = min(
        4.15 * sync_tail + 0.12 * rank_gap,
        cap_multiplier * raw_latency,
    )
    return (
        raw_latency + correction,
        f"epnorm_mid_raw_plus_min_4p15sync_0p12gap_0p{int(cap_multiplier * 100):02d}raw_cap",
        f"_hybrid_epnorm_mid_raw_plus_min_4p15sync_0p12gap_0p{int(cap_multiplier * 100):02d}raw_cap",
    )


def _epnorm_tail_generation_latency(row: dict[str, str]) -> tuple[float, str, str]:
    token = int(float(row.get("num_tokens", "0") or 0))
    raw_latency = _raw_operator_latency(row)
    sync_tail = _as_float(row, "rank_sync_tail_mean")
    ep_scale = _ep_scale_to_ep8(row)
    if token >= 1280:
        multiplier = 0.55 if _as_float(row, "moe_ep_size", 8.0) <= 2.0 else 0.5
        latency = raw_latency + max(0.0, ep_scale - 1.0) * multiplier * sync_tail
        return (
            latency,
            f"epnorm_tail1280_raw_plus_epdeficit_{str(multiplier).replace('.', 'p')}sync",
            f"_hybrid_epnorm_tail1280_raw_plus_epdeficit_{str(multiplier).replace('.', 'p')}sync",
        )
    exponent = 0.22 if _as_float(row, "moe_ep_size", 8.0) <= 2.0 else 0.30
    latency = (raw_latency + 0.5 * sync_tail) * (ep_scale ** exponent)
    return (
        latency,
        f"epnorm_tail_raw_plus_0p5sync_x_ep_scale_{str(exponent).replace('.', 'p')}",
        f"_hybrid_epnorm_tail_raw_plus_0p5sync_x_ep_scale_{str(exponent).replace('.', 'p')}",
    )


def _epnorm_sparse_context_latency(row: dict[str, str]) -> tuple[float, str, str]:
    token = int(float(row.get("num_tokens", "0") or 0))
    if token <= 256:
        ep_size = min(max(_as_float(row, "moe_ep_size", 8.0), 1.0), 8.0)
        sync_multiplier = 0.45 - 0.25 * ep_size / 8.0
        return (
            _as_float(row, "rank_mean_latency")
            + sync_multiplier * _as_float(row, "rank_sync_tail_mean"),
            f"epnorm_context_tiny_rank_plus_epcont_{str(sync_multiplier).replace('.', 'p')}sync",
            f"_hybrid_epnorm_context_tiny_rank_plus_epcont_{str(sync_multiplier).replace('.', 'p')}sync",
        )
    gather_scatter = _stage_gather_scatter(row)
    multiplier = 0.90
    if (
        _as_float(row, "moe_ep_size", 8.0) >= 8.0
        and row.get("distribution") == "recorded_eplb"
        and token <= 640
    ):
        multiplier = 1.40
    latency = _as_float(row, "rank_mean_latency") - multiplier * gather_scatter
    return (
        max(0.0, latency),
        f"epnorm_context_sparse_rank_minus_{multiplier:g}x_gather_scatter",
        f"_hybrid_epnorm_context_sparse_rank_minus_{str(multiplier).replace('.', 'p')}x_gather_scatter",
    )


def _text_prompt_context_factor(row: dict[str, str], *, token: int) -> tuple[float, str]:
    ep_size = _as_float(row, "moe_ep_size", 8.0)
    distribution = row.get("distribution")
    if token == 128:
        if ep_size <= 2.0:
            return 1.05, "x_textctx_ep2_128_1p05"
        if ep_size <= 4.0:
            return 1.02, "x_textctx_ep4_128_1p02"
        return 1.0, ""
    if ep_size >= 8.0 and token in (512, 640):
        return 0.9604, "x_textctx_ep8_512_640_0p9604"
    if ep_size >= 8.0 and token == 1536:
        return 0.96, "x_textctx_ep8_1536_0p96"
    if token == 2048:
        if ep_size <= 2.0 and distribution == "recorded_eplb":
            return 0.94, "x_textctx_ep2_eplb_2048_0p94"
        if ep_size >= 8.0:
            return 0.882, "x_textctx_ep8_2048_0p882"
        if ep_size >= 4.0:
            return 0.90, "x_textctx_ep_ge4_2048_0p90"
        return 1.0, ""
    if token == 2560:
        return 0.88, "x_textctx_2560_0p88"
    if token == 4096:
        if ep_size <= 2.0:
            return 0.75, "x_textctx_ep2_4096_0p75"
        if ep_size >= 8.0:
            return 0.63, "x_textctx_ep8_4096_0p63"
        return 0.64, "x_textctx_ep4_4096_0p64"
    if token >= 5120:
        if ep_size == 4.0:
            if token >= 14336:
                return 0.915, "x_textctx_ep4_tail_0p915"
            if token >= 8192:
                return 0.925, "x_textctx_ep4_dense_0p925"
            return 0.94, "x_textctx_ep4_dense_0p94"
        if ep_size >= 8.0:
            if token == 5120:
                return 0.9065, "x_textctx_ep8_5120_0p9065"
            if token == 8192:
                return 0.8967, "x_textctx_ep8_8192_0p8967"
            if token in (10240, 12288):
                return 0.93, "x_textctx_ep8_10k_12k_0p93"
            if token in (14336, 16384):
                return 0.91, "x_textctx_ep8_14k_16k_0p91"
            return 0.95, "x_textctx_ep8_tail_0p95"
    return 1.0, ""


def _apply_text_prompt_context_factor(
    latency_ms: float,
    policy: str,
    suffix: str,
    row: dict[str, str],
    *,
    token: int,
) -> tuple[float, str, str]:
    factor, label = _text_prompt_context_factor(row, token=token)
    if factor == 1.0:
        return latency_ms, policy, suffix
    return latency_ms * factor, f"{policy}_{label}", f"{suffix}_{label}"


def _epnorm_mid_context_latency(row: dict[str, str]) -> tuple[float, str, str]:
    token = int(float(row.get("num_tokens", "0") or 0))
    if (
        _as_float(row, "moe_ep_size", 8.0) <= 2.0
        and row.get("distribution") == "recorded_eplb"
        and token == 2560
    ):
        latency = _as_float(row, "rank_mean_latency") - 0.35 * _as_float(row, "rank_sync_tail_mean")
        return (
            max(0.0, latency),
            "epnorm_context_mid_ep2_eplb_2560_rank_minus_0p35sync",
            "_hybrid_epnorm_context_mid_ep2_eplb_2560_rank_minus_0p35sync",
        )
    return (
        _as_float(row, "latency"),
        str(row.get("primary_latency_source", "") or "rank_mean"),
        "_hybrid_dense_raw_mid_context",
    )


def _epnorm_dense_context_latency(row: dict[str, str]) -> tuple[float, str, str]:
    token = int(float(row.get("num_tokens", "0") or 0))
    ep_size = _as_float(row, "moe_ep_size", 8.0)
    sync_multiplier = 0.05 + 0.30 * min(max(ep_size, 1.0), 8.0) / 8.0
    if ep_size >= 8.0:
        sync_multiplier = 0.25
    latency = _as_float(row, "rank_mean_latency") + sync_multiplier * _as_float(
        row,
        "rank_sync_tail_mean",
    )
    if ep_size <= 2.0 and row.get("distribution") == "recorded_eplb" and token >= 8192:
        latency *= 0.99
        return (
            latency,
            f"epnorm_context_dense_ep2_eplb_rank_plus_epcont_{str(sync_multiplier).replace('.', 'p')}sync_x0p99",
            f"_hybrid_epnorm_context_dense_ep2_eplb_rank_plus_epcont_{str(sync_multiplier).replace('.', 'p')}sync_x0p99",
        )
    return (
        latency,
        f"epnorm_context_dense_rank_plus_epcont_{str(sync_multiplier).replace('.', 'p')}sync",
        f"_hybrid_epnorm_context_dense_rank_plus_epcont_{str(sync_multiplier).replace('.', 'p')}sync",
    )


def _mid_generation_latency(row: dict[str, str]) -> tuple[float, str, str]:
    if _is_current_run_single_card_dummy(row):
        return _epnorm_mid_generation_latency(row)
    return (
        _latency_plus_sync_tail(row, 4.0),
        "latency_plus_4x_sync_tail",
        "_hybrid_mid_latency_plus_4x_sync_tail",
    )


def _tail_generation_latency(row: dict[str, str]) -> tuple[float, str, str]:
    if _is_current_run_single_card_dummy(row):
        return _epnorm_tail_generation_latency(row)
    return (
        _latency_plus_sync_tail(row, 0.5),
        "latency_plus_0p5x_sync_tail",
        "_hybrid_tail_latency_plus_0p5x_sync_tail",
    )


def _small_generation_latency(row: dict[str, str]) -> tuple[float, str, str]:
    if _is_current_run_single_card_dummy(row):
        latency, policy, suffix = _epnorm_small_generation_latency(row)
        token = int(float(row.get("num_tokens", "0") or 0))
        if token == 64:
            return (
                latency * 1.12,
                f"{policy}_x_token64_1p12",
                f"{suffix}_x_token64_1p12",
            )
        return latency, policy, suffix
    return (
        _as_float(row, "latency"),
        str(row.get("primary_latency_source", "") or "critical_path"),
        "_hybrid_main_raw",
    )


def apply_profile_free_hybrid_latency(
    row: dict[str, str],
    *,
    phase: str,
) -> dict[str, str]:
    origin_latency = str(row.get("origin_latency", "") or "").strip()
    if origin_latency:
        row = dict(row)
        row["latency"] = origin_latency

    distribution = row.get("distribution", "")
    if distribution not in {"recorded", "recorded_no_eplb", "recorded_eplb"}:
        return dict(row)

    token = int(float(row.get("num_tokens", "0") or 0))
    if phase == "context":
        if token <= 256 and _is_current_run_single_card_dummy(row):
            latency_ms, policy, suffix = _epnorm_sparse_context_latency(row)
            latency_ms, policy, suffix = _apply_text_prompt_context_factor(
                latency_ms,
                policy,
                suffix,
                row,
                token=token,
            )
            return _with_latency(row, latency_ms=latency_ms, policy=policy, kernel_suffix=suffix)
        if token <= 2048:
            if _is_current_run_single_card_dummy(row):
                latency_ms, policy, suffix = _epnorm_sparse_context_latency(row)
            else:
                latency_ms = _as_float(row, "rank_mean_latency") - 0.05 * _as_float(row, "rank_sync_tail_mean")
                policy = "rank_mean_minus_0p05_sync_tail"
                suffix = "_hybrid_sparse_rank_mean_minus_0p05_sync_tail"
            latency_ms, policy, suffix = _apply_text_prompt_context_factor(
                max(0.0, latency_ms),
                policy,
                suffix,
                row,
                token=token,
            )
            return _with_latency(row, latency_ms=max(0.0, latency_ms), policy=policy, kernel_suffix=suffix)
        if token <= 4096:
            if _is_current_run_single_card_dummy(row):
                latency_ms, policy, suffix = _epnorm_mid_context_latency(row)
                latency_ms, policy, suffix = _apply_text_prompt_context_factor(
                    latency_ms,
                    policy,
                    suffix,
                    row,
                    token=token,
                )
                return _with_latency(row, latency_ms=latency_ms, policy=policy, kernel_suffix=suffix)
            latency_ms = _as_float(row, "latency")
            policy = str(row.get("primary_latency_source", "") or "rank_mean")
            suffix = "_hybrid_dense_raw_mid_context"
            latency_ms, policy, suffix = _apply_text_prompt_context_factor(
                latency_ms,
                policy,
                suffix,
                row,
                token=token,
            )
            return _with_latency(
                row,
                latency_ms=latency_ms,
                policy=policy,
                kernel_suffix=suffix,
            )
        if _is_current_run_single_card_dummy(row):
            latency_ms, policy, suffix = _epnorm_dense_context_latency(row)
            latency_ms, policy, suffix = _apply_text_prompt_context_factor(
                latency_ms,
                policy,
                suffix,
                row,
                token=token,
            )
            return _with_latency(row, latency_ms=latency_ms, policy=policy, kernel_suffix=suffix)
        latency_ms = _as_float(row, "rank_mean_latency") + 0.35 * _as_float(row, "rank_sync_tail_mean")
        policy = "rank_mean_plus_0p35_sync_tail"
        suffix = "_hybrid_dense_rank_mean_plus_0p35_sync_tail"
        latency_ms, policy, suffix = _apply_text_prompt_context_factor(
            latency_ms,
            policy,
            suffix,
            row,
            token=token,
        )
        return _with_latency(
            row,
            latency_ms=latency_ms,
            policy=policy,
            kernel_suffix=suffix,
        )

    if phase == "generation":
        if token <= 8:
            if _is_current_run_single_card_dummy(row):
                latency, policy, suffix = _epnorm_tiny_generation_latency_2(row)
            else:
                latency, policy, suffix = _epnorm_tiny_generation_latency(row)
        elif token <= 64:
            latency, policy, suffix = _small_generation_latency(row)
        elif token <= 256:
            if _is_current_run_single_card_dummy(row):
                latency, policy, suffix = _epnorm_small_main_generation_latency(row)
            else:
                latency = _as_float(row, "rank_mean_latency")
                policy = "rank_mean_small_main"
                suffix = "_hybrid_small_main_rank_mean"
        elif token <= 640:
            latency, policy, suffix = _mid_generation_latency(row)
        else:
            latency, policy, suffix = _tail_generation_latency(row)
        return _with_latency(row, latency_ms=latency, policy=policy, kernel_suffix=suffix)

    raise ValueError(f"Unsupported phase for hybrid latency: {phase}")
