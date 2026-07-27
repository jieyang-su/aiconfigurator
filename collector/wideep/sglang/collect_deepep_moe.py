# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang WideEP DeepEP MoE collector.

Runs distributed DeepEP dispatch/combine and MoE-compute benchmarks through a
minimal SGLang engine setup. The module owns process-group initialization,
rank-local model runner construction, DeepEP buffer sizing, warmup/measurement,
and perf-row aggregation for WideEP MoE cases.
"""

import functools
import csv
import json
import logging
import os
import random
import sys
import tempfile
import time
from collections import defaultdict
from contextlib import AbstractContextManager
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    DeepEPLLDispatchOutput,
    DeepEPNormalDispatchOutput,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    configure_logger,
    get_bool_env_var,
    set_gpu_proc_affinity,
    suppress_other_loggers,
)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
COLLECTOR_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
if COLLECTOR_ROOT not in sys.path:
    sys.path.append(COLLECTOR_ROOT)

try:
    from helper import (
        _resolve_local_model_path,
        log_perf,
        power_law_deepep_decode,
        power_law_deepep_prefill,
        resolve_subprocess_visible_device,
    )
except ModuleNotFoundError:
    sys.path.append(COLLECTOR_ROOT)
    from helper import (
        _resolve_local_model_path,
        log_perf,
        power_law_deepep_decode,
        power_law_deepep_prefill,
        resolve_subprocess_visible_device,
    )
from importlib.metadata import version as get_version
from math import ceil as _ceil

try:
    from collector.wideep.sglang.rank_local_moe_replay import select_replay_workloads
except ModuleNotFoundError:
    if THIS_DIR not in sys.path:
        sys.path.append(THIS_DIR)
    from rank_local_moe_replay import select_replay_workloads

try:
    from collector.recorded_source_health_guard import select_recorded_source
except ModuleNotFoundError:
    if COLLECTOR_ROOT not in sys.path:
        sys.path.append(COLLECTOR_ROOT)
    from recorded_source_health_guard import select_recorded_source

MOE_SUBPROCESS_TIMEOUT_SEC = 1800
MOE_PROGRESS_LOG_INTERVAL_SEC = 60
DEFAULT_MOE_MEM_FRACTION_STATIC = 0.3
DEFAULT_MOE_TEST_LAYER = 3
DEFAULT_MOE_COLLECTION_LAYERS = DEFAULT_MOE_TEST_LAYER + 1
DEFAULT_DSV3_CONTEXT_LOGGED_TOKENS = (
    128,
    192,
    256,
    384,
    512,
    640,
    768,
    1024,
    1536,
    2048,
    2304,
    2560,
    3072,
    4096,
    5120,
    6144,
    8192,
    10240,
    12288,
    14336,
    16384,
    18888,
)
DEFAULT_REPLAY_RANDOM_SEED = 20260620
SOURCE_STABILITY_GUARD_NAME = "wideep_context_tiny_token_source_stability_v1"


@dataclass(frozen=True)
class ReplayMeasurementPolicy:
    """Shared rank-local replay policy for context and generation."""

    name: str
    measurement_boundary: str
    stage_measurement_boundary: str
    rank_order: str
    outlier_mad_multiplier: float
    outlier_min_relative_headroom: float
    multistream_probe: bool
    multistream_rounds: int
    primary_latency_source: str


def _replay_measurement_policy() -> ReplayMeasurementPolicy:
    return ReplayMeasurementPolicy(
        name="rank_local_replay_v2",
        measurement_boundary="run_moe_core_cuda_critical_path",
        stage_measurement_boundary="instrumented_function_cuda_event_spans",
        rank_order="deterministic_random_per_round",
        outlier_mad_multiplier=float(
            os.environ.get(
                "COLLECTOR_WIDEEP_MOE_REPLAY_OUTLIER_MAD_MULTIPLIER",
                "8.0",
            )
        ),
        outlier_min_relative_headroom=float(
            os.environ.get(
                "COLLECTOR_WIDEEP_MOE_REPLAY_OUTLIER_MIN_REL_HEADROOM",
                "0.5",
            )
        ),
        multistream_probe=get_bool_env_var(
            "COLLECTOR_WIDEEP_MOE_REPLAY_MULTISTREAM_PROBE",
            "true",
        ),
        multistream_rounds=max(
            1,
            int(
                os.environ.get(
                    "COLLECTOR_WIDEEP_MOE_REPLAY_MULTISTREAM_ROUNDS",
                    "5",
                )
            ),
        ),
        primary_latency_source="critical_path",
    )


def _env_int_list(name: str) -> list[int] | None:
    raw_value = os.environ.get(name)
    if not raw_value:
        return None
    values = []
    for item in raw_value.replace(",", " ").split():
        values.append(int(item))
    return values


def _env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return default
    return int(raw_value)


def _env_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return default
    return float(raw_value)


def _env_float_list(name: str) -> list[float] | None:
    raw_value = os.environ.get(name)
    if not raw_value:
        return None
    values = []
    for item in raw_value.replace(",", " ").split():
        values.append(float(item))
    return values


def _env_str_set(name: str) -> set[str] | None:
    raw_value = os.environ.get(name)
    if not raw_value:
        return None
    return {item.strip() for item in raw_value.replace(",", " ").split() if item.strip()}


def _visible_device_count() -> int:
    raw = os.environ.get("COLLECTOR_MOE_DISTRIBUTION_VISIBLE_DEVICES")
    if not raw:
        raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw:
        return max(1, len([item for item in raw.split(",") if item.strip()]))
    count = torch.cuda.device_count()
    return max(1, int(count or 1))


def _default_ep_sizes_for_visible_devices(total_experts: int) -> list[int]:
    visible = _visible_device_count()
    candidates = [1, 2, 4, 8]
    sizes = [
        ep_size
        for ep_size in candidates
        if ep_size <= visible and total_experts % ep_size == 0
    ]
    return sizes or [1]


def _get_recorded_distribution_name() -> str:
    return os.environ.get("COLLECTOR_WIDEEP_MOE_RECORDED_DISTRIBUTION", "recorded")


def _get_rank_local_replay_dir(output_path: str | None = None) -> str | None:
    path = os.environ.get("COLLECTOR_WIDEEP_MOE_RANK_LOCAL_REPLAY_DIR")
    if not path and output_path:
        candidate = os.path.join(output_path, "moe_token_distribution_replay")
        if os.path.isdir(candidate):
            path = candidate
    if not path:
        current_output_dir = os.environ.get("COLLECTOR_CURRENT_OUTPUT_DIR")
        if current_output_dir:
            candidate = os.path.join(
                current_output_dir,
                "moe_token_distribution_replay",
            )
            if os.path.isdir(candidate):
                path = candidate
    if not path:
        return None
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"COLLECTOR_WIDEEP_MOE_RANK_LOCAL_REPLAY_DIR is not a directory: {path}"
        )
    return path


_CURRENT_WIDEEP_ENABLE_EPLB_OVERRIDE: bool | None = None


def _rank_local_replay_enable_eplb() -> bool:
    if _CURRENT_WIDEEP_ENABLE_EPLB_OVERRIDE is not None:
        return _CURRENT_WIDEEP_ENABLE_EPLB_OVERRIDE
    return get_bool_env_var("COLLECTOR_WIDEEP_MOE_ENABLE_EPLB")


def _wideep_moe_use_cuda_graph_for_phase(phase: str) -> bool:
    if phase == "context":
        return get_bool_env_var(
            "COLLECTOR_WIDEEP_MOE_CONTEXT_USE_CUDA_GRAPH",
            "false",
        )
    if phase == "generation":
        return get_bool_env_var(
            "COLLECTOR_WIDEEP_MOE_GENERATION_USE_CUDA_GRAPH",
            "true",
        )
    return get_bool_env_var("COLLECTOR_WIDEEP_MOE_USE_CUDA_GRAPH", "true")


def _rank_local_replay_source() -> str | None:
    value = os.environ.get("COLLECTOR_WIDEEP_MOE_REPLAY_SOURCE")
    return value.strip().lower() if value else "dummy"


def _recorded_output_distribution() -> str:
    return (
        "recorded_eplb"
        if _rank_local_replay_enable_eplb()
        else "recorded_no_eplb"
    )


def _wideep_eplb_modes_for_cases() -> list[bool]:
    raw = os.environ.get("COLLECTOR_WIDEEP_MOE_ENABLE_EPLB")
    if raw is None:
        return [False, True]
    return [raw.strip().lower() not in ("0", "false", "no", "off")]


def _recorded_latency_value(stats: dict, policy: str) -> float:
    if policy == "critical_path":
        return float(stats["latency"])
    if policy == "rank_mean":
        return max(0.0, float(stats["rank_mean_latency"]))
    if policy == "rank_p90":
        return max(0.0, float(stats["rank_p90_latency"]))
    if policy == "rank_critical_mean":
        return max(0.0, float(stats["rank_critical_mean_latency"]))
    if policy == "rank_mean_minus_sync_tail":
        return max(
            0.0,
            float(stats["rank_mean_latency"])
            - float(stats.get("rank_sync_tail_mean") or 0.0),
        )
    if policy == "stage_kernel_sum_mean":
        return max(0.0, float(stats["stage_kernel_sum_mean"]))
    if policy == "stage_sum_rankmax":
        return max(0.0, float(stats["latency_stage_sum_rankmax"]))
    raise ValueError(f"Unsupported Recorded latency policy: {policy}")


def _recorded_latency_policy(phase: str, num_tokens_log: int) -> str:
    """Choose the profile-free primary latency source for Recorded replay rows."""

    override = os.environ.get("COLLECTOR_WIDEEP_MOE_RECORDED_LATENCY_POLICY")
    if override:
        return override.strip()
    phase_override = os.environ.get(
        f"COLLECTOR_WIDEEP_MOE_RECORDED_{phase.upper()}_LATENCY_POLICY"
    )
    if phase_override:
        return phase_override.strip()
    if phase == "context":
        # Context prefill replay measures one single-card rank-local slice.
        # The rank mean is a better standalone compute estimate than the
        # worst-rank critical path when the caller later models EP-wide overlap.
        return "rank_mean"
    if phase == "generation" and int(num_tokens_log) <= int(
        os.environ.get("COLLECTOR_WIDEEP_MOE_RECORDED_SMALL_DECODE_MAX", "64")
    ):
        # Tiny decode batches are dominated by rank synchronization slack in
        # the single-card replay harness.  Remove that slack for the primary
        # operator latency while keeping the raw fields in the row.  Keep the
        # default broad enough to cover the sparse 8/16/32/40/64 decode region;
        # callers can still narrow it with the env override when needed.
        return "rank_mean_minus_sync_tail"
    return "critical_path"


def _apply_recorded_latency_policy(
    stats: dict,
    *,
    phase: str,
    num_tokens_log: int,
) -> dict:
    policy = _recorded_latency_policy(phase, num_tokens_log)
    output = dict(stats)
    output.setdefault("origin_latency", float(stats["latency"]))
    output["latency"] = _recorded_latency_value(output, policy)
    output["primary_latency_source"] = policy
    return output


def _apply_profile_free_hybrid_recorded_row(
    row: dict,
    *,
    phase: str,
) -> dict:
    try:
        from moe_hybrid_policy import apply_profile_free_hybrid_latency
    except ModuleNotFoundError:
        if COLLECTOR_ROOT not in sys.path:
            sys.path.append(COLLECTOR_ROOT)
        from moe_hybrid_policy import apply_profile_free_hybrid_latency
    merged = apply_profile_free_hybrid_latency(row, phase=phase)
    merged["latency_policy_scope"] = "profile_free_hybrid_recorded"
    return merged


def _source_stability_guard_enabled(*, phase: str, num_tokens_log: int) -> bool:
    if phase != "context":
        return False
    if not get_bool_env_var("COLLECTOR_WIDEEP_MOE_SOURCE_STABILITY_GUARD", "false"):
        return False
    max_tokens = _env_int("COLLECTOR_WIDEEP_MOE_SOURCE_STABILITY_MAX_TOKENS", 32)
    return int(num_tokens_log) <= max_tokens


def _select_stable_recorded_source(
    *,
    measure_once,
    phase: str,
    num_tokens_log: int,
    rank_print,
) -> dict:
    """Repeat high-risk recorded source probes and select a stable median row.

    The selector keeps the row internally consistent by returning one complete
    measured item, instead of mixing latency and feature columns across runs.
    """

    if not _source_stability_guard_enabled(
        phase=phase,
        num_tokens_log=num_tokens_log,
    ):
        return measure_once()

    initial_sessions = max(
        1,
        _env_int("COLLECTOR_WIDEEP_MOE_SOURCE_STABILITY_SESSIONS", 3),
    )
    max_sessions = max(
        initial_sessions,
        _env_int("COLLECTOR_WIDEEP_MOE_SOURCE_STABILITY_MAX_SESSIONS", 5),
    )
    stable_spread = _env_float(
        "COLLECTOR_WIDEEP_MOE_SOURCE_STABILITY_SPREAD",
        1.08,
    )
    max_spread = _env_float(
        "COLLECTOR_WIDEEP_MOE_SOURCE_STABILITY_MAX_SPREAD",
        1.20,
    )

    measured: list[dict] = []

    def _spread() -> float:
        values = [float(item["latency"]) for item in measured]
        min_value = min(values)
        max_value = max(values)
        if min_value <= 0:
            return float("inf") if max_value > 0 else 1.0
        return max_value / min_value

    while len(measured) < initial_sessions:
        measured.append(measure_once())

    spread = _spread()
    if spread > stable_spread and len(measured) < max_sessions:
        while len(measured) < max_sessions:
            measured.append(measure_once())
        spread = _spread()

    ordered = sorted(
        enumerate(measured, start=1),
        key=lambda entry: float(entry[1]["latency"]),
    )
    selected_session, selected_item = ordered[len(ordered) // 2]
    selected = dict(selected_item)
    if spread <= stable_spread:
        status = "stable"
    elif spread <= max_spread:
        status = "weak_stable"
    else:
        status = "unstable"
    latency_values = [float(item["latency"]) for item in measured]
    selected.update(
        {
            "source_stability_guard": SOURCE_STABILITY_GUARD_NAME,
            "source_stability_status": status,
            "source_stability_sessions": len(measured),
            "source_stability_selected_session": selected_session,
            "source_stability_spread_ratio": spread,
            "source_stability_values_ms_json": json.dumps(latency_values),
        }
    )
    rank_print(
        "Recorded source stability guard: "
        f"phase={phase}, token={num_tokens_log}, sessions={len(measured)}, "
        f"spread={spread:.4f}, status={status}, "
        f"selected_session={selected_session}, "
        f"values_ms={[round(value, 6) for value in latency_values]}"
    )
    return selected


def _rank_local_replay_samples(
    *,
    phase: str,
    table_num_tokens: int,
    layer_id: int,
    ep_size: int,
    num_experts: int,
    output_path: str | None = None,
):
    replay_dir = _get_rank_local_replay_dir(output_path)
    if replay_dir is None:
        return None
    return select_replay_workloads(
        replay_dir=replay_dir,
        phase=phase,
        table_num_tokens=table_num_tokens,
        layer_id=layer_id,
        ep_size=ep_size,
        num_experts=num_experts,
        enable_eplb=_rank_local_replay_enable_eplb(),
        workload_source=_rank_local_replay_source(),
    )


def _resolve_recorded_distribution_file(output_path: str | None = None) -> str:
    for env_name in (
        "COLLECTOR_WIDEEP_MOE_RECORDED_DISTRIBUTION_FILE",
        "COLLECTOR_MOE_TOKEN_DISTRIBUTION_FILE",
    ):
        raw_value = os.environ.get(env_name)
        if raw_value:
            if not os.path.exists(raw_value):
                raise FileNotFoundError(f"{env_name} points to missing file: {raw_value}")
            return raw_value

    candidates = []
    if output_path:
        candidates.append(os.path.join(output_path, "moe_token_distribution_perf.txt"))
    collector_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates.append(os.path.join(collector_dir, "moe_token_distribution_perf.txt"))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "recorded MoE distribution requested, but no moe_token_distribution_perf.txt was found. "
        "Set COLLECTOR_WIDEEP_MOE_RECORDED_DISTRIBUTION_FILE or run moe_token_distribution first "
        "in the same collector output directory."
    )


def _has_recorded_distribution_file(output_path: str | None = None) -> bool:
    if _get_rank_local_replay_dir(output_path) is not None:
        return True
    try:
        _resolve_recorded_distribution_file(output_path)
    except FileNotFoundError:
        return False
    return True


@functools.cache
def _load_recorded_distribution_rows(
    path: str,
    distribution: str,
    phase: str,
) -> tuple[dict, ...]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            if row.get("distribution") != distribution:
                continue
            if row.get("phase", "context") != phase:
                continue
            rows.append(row)
    if not rows:
        raise ValueError(
            f"No distribution={distribution!r} phase={phase!r} rows found in {path}"
        )
    return tuple(rows)


def _parse_recorded_expert_counts(row: dict, total_experts: int) -> list[int]:
    raw_counts = row.get("expert_assignments_json") or ""
    if not raw_counts:
        raise ValueError(
            "recorded MoE distribution row has empty expert_assignments_json. "
            "Re-run moe_token_distribution with the updated collector so the exact expert counts are saved."
        )
    counts = json.loads(raw_counts)
    if len(counts) != total_experts:
        raise ValueError(
            f"recorded expert count length mismatch: got {len(counts)}, expected {total_experts}"
        )
    return [max(0, int(round(float(value)))) for value in counts]


def _select_recorded_expert_counts(
    *,
    output_path: str | None,
    distribution: str,
    num_tokens: int,
    topk: int,
    total_experts: int,
    preferred_layer_id: int | None,
    preferred_recorder_ep_size: int | None,
    phase: str = "context",
) -> list[int]:
    path = _resolve_recorded_distribution_file(output_path)
    rows = _load_recorded_distribution_rows(path, distribution, phase)
    candidates = []
    for row in rows:
        try:
            if int(float(row.get("num_tokens", 0))) != int(num_tokens):
                continue
            if int(float(row.get("topk", topk))) != int(topk):
                continue
            if int(float(row.get("num_experts", total_experts))) != int(total_experts):
                continue
        except ValueError:
            continue
        candidates.append(row)

    if not candidates:
        raise ValueError(
            f"No recorded MoE distribution row for num_tokens={num_tokens}, topk={topk}, "
            f"num_experts={total_experts} in {path}"
        )

    def score(row: dict) -> tuple[int, int, int, float, int]:
        try:
            layer_id = int(float(row.get("layer_id", -1)))
        except ValueError:
            layer_id = -1
        try:
            recorder_ep_size = int(float(row.get("recorder_ep_size", -1)))
        except ValueError:
            recorder_ep_size = -1
        try:
            total_assignments = float(row.get("total_assignments", 0.0) or 0.0)
        except ValueError:
            total_assignments = 0.0
        layer_match = int(preferred_layer_id is not None and layer_id == preferred_layer_id)
        ep_match = int(
            preferred_recorder_ep_size is not None
            and recorder_ep_size == int(preferred_recorder_ep_size)
        )
        nonzero = int(total_assignments > 0)
        return (ep_match, nonzero, layer_match, total_assignments, layer_id)

    selected = max(candidates, key=score)
    counts = _parse_recorded_expert_counts(selected, total_experts)
    actual_total = sum(counts)
    if actual_total <= 0:
        raise ValueError(
            f"Recorded MoE distribution row for num_tokens={num_tokens} has zero assignments; "
            "check that the selected layer is a routed MoE layer."
        )
    return counts


def _recorded_distribution_has_case(
    *,
    output_path: str | None,
    phase: str = "generation",
    num_tokens: int,
    topk: int,
    total_experts: int,
    preferred_recorder_ep_size: int | None,
) -> bool:
    replay_dir = _get_rank_local_replay_dir(output_path)
    if replay_dir is not None:
        try:
            select_replay_workloads(
                replay_dir=replay_dir,
                phase=phase,
                table_num_tokens=num_tokens,
                layer_id=_get_moe_test_layer(),
                ep_size=preferred_recorder_ep_size or 1,
                num_experts=total_experts,
                enable_eplb=_rank_local_replay_enable_eplb(),
                workload_source=_rank_local_replay_source(),
            )
        except (FileNotFoundError, ValueError):
            return False
        return True
    try:
        _select_recorded_expert_counts(
            output_path=output_path,
            distribution=_get_recorded_distribution_name(),
            num_tokens=num_tokens,
            topk=topk,
            total_experts=total_experts,
            preferred_layer_id=None,
            preferred_recorder_ep_size=preferred_recorder_ep_size,
            phase=phase,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return False
    return True


def _rescale_counts_to_total(counts: list[int], target_total: int) -> list[int]:
    actual_total = sum(counts)
    if actual_total <= 0:
        raise ValueError("Cannot rescale recorded MoE distribution with zero total assignments")
    if actual_total == target_total:
        return counts

    scaled = [count * target_total / actual_total for count in counts]
    floors = [int(value) for value in scaled]
    remainder = target_total - sum(floors)
    if remainder > 0:
        fractional_order = sorted(
            range(len(scaled)),
            key=lambda idx: (scaled[idx] - floors[idx], counts[idx]),
            reverse=True,
        )
        for idx in fractional_order[:remainder]:
            floors[idx] += 1
    elif remainder < 0:
        removable_order = sorted(range(len(floors)), key=lambda idx: floors[idx], reverse=True)
        for idx in removable_order[: -remainder]:
            if floors[idx] > 0:
                floors[idx] -= 1

    final_total = sum(floors)
    if final_total != target_total:
        raise ValueError(
            f"Failed to rescale recorded MoE distribution: got total={final_total}, expected={target_total}"
        )
    return floors


def _build_recorded_prefill_sample(
    *,
    num_tokens: int,
    topk: int,
    num_local_experts: int,
    total_experts: int,
    output_path: str | None,
    preferred_layer_id: int | None,
    preferred_recorder_ep_size: int | None,
    device,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    distribution = _get_recorded_distribution_name()
    counts = _select_recorded_expert_counts(
        output_path=output_path,
        distribution=distribution,
        num_tokens=num_tokens,
        topk=topk,
        total_experts=total_experts,
        preferred_layer_id=preferred_layer_id,
        preferred_recorder_ep_size=preferred_recorder_ep_size,
        phase="context",
    )
    counts = _rescale_counts_to_total(counts, num_tokens * topk)
    local_counts = [0] * num_local_experts
    for expert_id, count in enumerate(counts):
        local_counts[expert_id % num_local_experts] += count

    probabilities = torch.tensor(local_counts, device=device, dtype=torch.float32)
    if torch.count_nonzero(probabilities).item() < topk:
        raise ValueError(
            f"Recorded local distribution has fewer active experts than topk: "
            f"active={torch.count_nonzero(probabilities).item()}, topk={topk}"
        )
    probabilities = probabilities / probabilities.sum()

    seed = int(os.environ.get("COLLECTOR_WIDEEP_MOE_RECORDED_ROUTING_SEED", "20260615"))
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + num_tokens + topk + num_local_experts)
    topk_rows = [
        torch.multinomial(probabilities, num_samples=topk, replacement=False, generator=generator)
        for _ in range(num_tokens)
    ]
    topk_idx = torch.stack(topk_rows).to(torch.int32)
    topk_weights = torch.full((num_tokens, topk), 1.0 / topk, device=device, dtype=torch.float32)
    sampled_counts = torch.bincount(topk_idx.reshape(-1), minlength=num_local_experts).to(torch.int32).tolist()
    return topk_idx.contiguous(), topk_weights.contiguous(), sampled_counts


def _pad_prefill_dispatch_contract(
    *,
    hidden_states_fp8: torch.Tensor,
    scale_tensor: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_recv_tokens_per_expert: list[int],
    device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Pad synthetic prefill dispatch inputs to match DeepEP normal-scatter capacity.

    DeepEP normal scatter expects `num_recv_tokens_per_expert` to describe the
    expert capacity, not just the real routed rows. When capacity is padded
    beyond the real routing rows, we must append matching dummy rows so the
    hidden/topk tensors stay consistent with the capacity contract.
    """
    block_e = int(os.environ.get("COLLECTOR_WIDEEP_MOE_NORMAL_REPLAY_BLOCK_E", "128"))
    padded_counts = [
        ((int(count) + block_e - 1) // block_e) * block_e if int(count) > 0 else 0
        for count in num_recv_tokens_per_expert
    ]
    actual_counts = torch.bincount(
        topk_ids[topk_ids >= 0].to(torch.int64),
        minlength=len(padded_counts),
    ).tolist()
    pad_rows = []
    for expert_id, capacity in enumerate(padded_counts):
        deficit = int(capacity) - int(actual_counts[expert_id])
        if deficit <= 0:
            continue
        pad_row = torch.full(
            (deficit, topk_ids.shape[1]),
            -1,
            dtype=torch.int32,
            device=device,
        )
        pad_row[:, 0] = int(expert_id)
        pad_rows.append(pad_row)
    if pad_rows:
        padding_topk_ids = torch.cat(pad_rows, dim=0)
        topk_ids = torch.cat([topk_ids, padding_topk_ids], dim=0)
        padding_hidden = torch.randn(
            padding_topk_ids.shape[0],
            hidden_states_fp8.shape[1],
            dtype=torch.bfloat16,
            device=device,
        ).to(torch.float8_e4m3fn)
        hidden_states_fp8 = torch.cat([hidden_states_fp8, padding_hidden], dim=0)
        padding_scale = _make_scale_tensor(
            padding_topk_ids.shape[0],
            hidden_states_fp8.shape[1],
            device,
        )
        scale_tensor = torch.cat([scale_tensor, padding_scale], dim=0)
        padding_topk_weights = torch.zeros(
            padding_topk_ids.shape,
            device=device,
            dtype=torch.float32,
        )
        padding_topk_weights[:, 0] = 1.0
        topk_weights = torch.cat([topk_weights, padding_topk_weights], dim=0)
    return (
        hidden_states_fp8.contiguous(),
        scale_tensor.contiguous(),
        topk_ids.contiguous(),
        topk_weights.contiguous(),
        padded_counts,
    )


def _build_recorded_decode_masked_m(
    *,
    num_tokens: int,
    topk: int,
    num_local_experts: int,
    total_experts: int,
    output_path: str | None,
    preferred_layer_id: int | None,
    preferred_recorder_ep_size: int | None,
    device,
) -> torch.Tensor:
    counts = _select_recorded_expert_counts(
        output_path=output_path,
        distribution=_get_recorded_distribution_name(),
        num_tokens=num_tokens,
        topk=topk,
        total_experts=total_experts,
        preferred_layer_id=preferred_layer_id,
        preferred_recorder_ep_size=preferred_recorder_ep_size,
        phase="generation",
    )
    counts = _rescale_counts_to_total(counts, num_tokens * topk)
    local_counts = [0] * num_local_experts
    for expert_id, count in enumerate(counts):
        local_counts[expert_id % num_local_experts] += count
    return torch.tensor(local_counts, device=device, dtype=torch.int32)


def _get_wideep_moe_max_local_assignments() -> int:
    raw_value = os.environ.get("COLLECTOR_WIDEEP_MOE_MAX_LOCAL_ASSIGNMENTS")
    if raw_value is None:
        return 512
    value = int(raw_value)
    if value < 0:
        raise ValueError(
            "COLLECTOR_WIDEEP_MOE_MAX_LOCAL_ASSIGNMENTS must be >= 0 "
            f"or unset, got {raw_value!r}"
        )
    return value


def _get_moe_mem_fraction_static() -> float:
    """Return the WideEP MoE mem_fraction_static override from env."""
    raw_value = os.environ.get("COLLECTOR_WIDEEP_MOE_MEM_FRACTION_STATIC")
    if raw_value is None:
        return DEFAULT_MOE_MEM_FRACTION_STATIC

    try:
        mem_fraction_static = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            "COLLECTOR_WIDEEP_MOE_MEM_FRACTION_STATIC must be a float, "
            f"got {raw_value!r}"
        ) from exc

    if not 0 < mem_fraction_static < 1:
        raise ValueError(
            "COLLECTOR_WIDEEP_MOE_MEM_FRACTION_STATIC must be between 0 and 1, "
            f"got {mem_fraction_static}"
        )

    return mem_fraction_static


def _get_requested_moe_num_layers() -> int:
    raw_value = os.environ.get("SGLANG_TEST_NUM_LAYERS")
    if raw_value is None:
        return DEFAULT_MOE_COLLECTION_LAYERS
    try:
        num_layers = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"SGLANG_TEST_NUM_LAYERS must be an integer, got {raw_value!r}") from exc
    if num_layers < 1:
        raise ValueError(f"SGLANG_TEST_NUM_LAYERS must be >= 1, got {num_layers}")
    return num_layers


def _get_moe_test_layer(num_layers: int | None = None) -> int:
    raw_value = os.environ.get("COLLECTOR_WIDEEP_MOE_TEST_LAYER")
    if raw_value is not None:
        test_layer = int(raw_value)
    else:
        test_layer = DEFAULT_MOE_TEST_LAYER
    num_layers = _get_requested_moe_num_layers() if num_layers is None else num_layers
    return min(test_layer, num_layers - 1)


def _force_first_layer_moe_for_short_dummy_config(config: dict, num_layers: int) -> None:
    """Keep 1-layer dummy collection on a real MoE layer for DeepSeek-style models."""
    if num_layers > DEFAULT_MOE_TEST_LAYER:
        return
    if "first_k_dense_replace" in config:
        config["first_k_dense_replace"] = 0
    text_config = config.get("text_config")
    if isinstance(text_config, dict) and "first_k_dense_replace" in text_config:
        text_config["first_k_dense_replace"] = 0


def _is_scale_ue8m0() -> bool:
    """Check if deep_gemm uses ue8m0 scale format (Blackwell GPUs)."""
    try:
        from sglang.srt.layers.deep_gemm_wrapper import configurer as deep_gemm_wrapper

        return getattr(deep_gemm_wrapper, "DEEPGEMM_SCALE_UE8M0", False)
    except ImportError:
        return False


def _make_scale_tensor(num_tokens: int, hidden_size: int, device) -> torch.Tensor:
    """Create a scale tensor matching the format expected by run_moe_core.

    On Blackwell (ue8m0), scales are packed as int32 with 4 scales per element.
    On older GPUs, scales are float32 with one per 128-element block.
    """
    scale_dim = hidden_size // 128
    if _is_scale_ue8m0():
        return torch.ones(
            num_tokens,
            _ceil(scale_dim / 4),
            device=device,
            dtype=torch.int32,
        )
    return torch.ones(num_tokens, scale_dim, device=device, dtype=torch.float32)


def _selected_moe_model_id() -> str:
    # collect.py sets COLLECTOR_MODEL_PATH from --model-path / case_plan.model_path.
    return (
        os.environ.get("COLLECTOR_MODEL_PATH")
        or os.environ.get("MOE_MODEL_PATH")
        or os.environ.get("DEEPSEEK_MODEL_PATH")
        or "deepseek-ai/DeepSeek-V3"
    )


@functools.cache
def _resolve_moe_model_path(model_id: str) -> str:
    return _resolve_local_model_path(model_id)


def _get_moe_model_path() -> str:
    """Resolve MoE model path lazily so that ``collect.py``'s registry import
    of this module does not trigger tempdir / JSON I/O on every invocation.

    Cached per model id for the lifetime of the process; subprocess workers each get a
    fresh interpreter and re-resolve on first call (which converges on the
    same deterministic tempdir built by ``helper._resolve_local_model_path``).
    """
    return _resolve_moe_model_path(_selected_moe_model_id())


def _get_config_value(config, *names: str):
    """Read a config attribute from top-level config or nested text_config."""
    for name in names:
        value = config.get(name) if isinstance(config, dict) else getattr(config, name, None)
        if value is not None:
            return value

    text_config = config.get("text_config") if isinstance(config, dict) else getattr(config, "text_config", None)
    if text_config is None:
        return None

    for name in names:
        value = text_config.get(name) if isinstance(text_config, dict) else getattr(text_config, name, None)
        if value is not None:
            return value
    return None


def _strip_moe_collection_quantization(config: dict) -> None:
    """Optionally remove checkpoint quant metadata for older SGLang runners.

    Current SGLang DeepEP fp8 MoE keeps the DeepGEMM path behind Fp8Config.
    Stripping quantization metadata makes DeepEPMoE fall through to deprecated
    forward_deepgemm_* assertions, so preserve it unless explicitly requested.
    """
    if not get_bool_env_var("COLLECTOR_WIDEEP_MOE_STRIP_QUANTIZATION"):
        return
    config.pop("quantization_config", None)
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        text_config.pop("quantization_config", None)


def _set_moe_collection_override(config: dict, num_experts: int) -> None:
    num_layers = _get_requested_moe_num_layers()
    override = {
        "num_hidden_layers": num_layers,
        "n_routed_experts": num_experts,
        "num_experts": num_experts,
        "num_local_experts": num_experts,
    }
    config.update(override)
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        text_config.update(override)
    _force_first_layer_moe_for_short_dummy_config(config, num_layers)


def _resolve_sglang_model_path(model_ref: str) -> str:
    """Build an SGLang-friendly local config directory for MoE collection."""
    model_ref = _resolve_local_model_path(model_ref)
    config_path = os.path.join(model_ref, "config.json") if os.path.isdir(model_ref) else None
    if not config_path or not os.path.exists(config_path):
        return model_ref

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    if config.get("model_type") in ("deepseek_v32", "glm_moe_dsa"):
        config["architectures"] = ["DeepseekV3ForCausalLM"]
        config["model_type"] = "deepseek_v3"

    config.pop("auto_map", None)
    _strip_moe_collection_quantization(config)

    safe_name = model_ref.replace("/", "_").replace("\\", "_").replace(":", "_")
    tmp_dir = os.path.join(tempfile.gettempdir(), f"aic_sglang_moe_config_{safe_name}_{os.getpid()}")
    os.makedirs(tmp_dir, exist_ok=True)
    with open(os.path.join(tmp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f)
    return tmp_dir


def _resolve_reduced_moe_model_path(model_path: str, num_experts: int) -> str:
    config_path = os.path.join(model_path, "config.json") if os.path.isdir(model_path) else None
    if not config_path or not os.path.exists(config_path):
        return model_path

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    _strip_moe_collection_quantization(config)
    _set_moe_collection_override(config, num_experts)

    safe_name = os.path.basename(os.path.normpath(model_path)).replace("\\", "_").replace(":", "_")
    tmp_dir = os.path.join(tempfile.gettempdir(), f"{safe_name}_experts_{num_experts}_{os.getpid()}")
    os.makedirs(tmp_dir, exist_ok=True)
    with open(os.path.join(tmp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f)
    return tmp_dir


def _get_total_experts_for_selected_model(default: int = 256) -> int:
    model_path = _resolve_sglang_model_path(_selected_moe_model_id())
    config_path = os.path.join(model_path, "config.json") if os.path.isdir(model_path) else None
    if not config_path or not os.path.exists(config_path):
        return default
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    return int(_get_config_value(config, "n_routed_experts", "num_experts", "num_local_experts") or default)


def get_moe_prefill_test_cases(
    rank,
    output_path: str | None = None,
    topk: int | None = None,
    total_experts: int | None = None,
):
    """Get test cases for MoE prefill phase including distribution and alpha.

    Returns a list of dicts with keys: 'num_tokens', 'distributed', 'power_law_alpha'.
    For uniform distribution, 'power_law_alpha' is None.
    """
    test_cases = []
    requested_logged_tokens = _env_int_list("COLLECTOR_WIDEEP_MOE_PREFILL_TOKENS")
    using_default_tokens = not requested_logged_tokens
    if requested_logged_tokens:
        num_tokens = sorted({max(1, token // rank) for token in requested_logged_tokens})
    elif os.environ.get("COLLECTOR_SMOKE") == "1":
        num_tokens = sorted({max(1, 128 // rank)})
    else:
        default_logged_tokens = DEFAULT_DSV3_CONTEXT_LOGGED_TOKENS
        num_tokens = sorted(
            {max(1, token // rank) for token in default_logged_tokens}
        )
    requested_distributions = _env_str_set("COLLECTOR_WIDEEP_MOE_DISTRIBUTIONS")
    if requested_distributions is None:
        requested_distributions = {"uniform"}
        if "deepseek" in _selected_moe_model_id().lower():
            requested_distributions.add("power_law")
        if _has_recorded_distribution_file(output_path):
            requested_distributions.add("recorded")
    default_alphas = (
        [0.6, 1.01]
        if "deepseek" in _selected_moe_model_id().lower()
        else [0.6, 0.8, 1.01, 1.02, 1.2]
    )
    power_law_alphas = (
        _env_float_list("COLLECTOR_WIDEEP_MOE_POWER_LAW_ALPHAS")
        or default_alphas
    )
    recorded_first = get_bool_env_var(
        "COLLECTOR_WIDEEP_MOE_PREFILL_RECORDED_FIRST",
        "true",
    )

    for num_token in sorted(num_tokens):
        logged_tokens = num_token * rank
        if num_token * rank > 256 * 2048:
            continue
        include_recorded = (
            "recorded" in requested_distributions
            and _has_recorded_distribution_file(output_path)
        )
        recorded_case_available = include_recorded and (
            topk is None
            or total_experts is None
            or _recorded_distribution_has_case(
                output_path=output_path,
                phase="context",
                num_tokens=logged_tokens,
                topk=topk,
                total_experts=total_experts,
                preferred_recorder_ep_size=rank,
            )
        )
        recorded_case = {
            "num_tokens": num_token,
            "distributed": "recorded",
            "power_law_alpha": None,
        }
        if recorded_first and recorded_case_available:
            test_cases.append(recorded_case)
        if num_token * 8 < 128:
            if not recorded_first and recorded_case_available:
                test_cases.append(recorded_case)
            continue
        # Uniform
        uniform_default_safe = logged_tokens in DEFAULT_DSV3_CONTEXT_LOGGED_TOKENS
        if "uniform" in requested_distributions and (
            not using_default_tokens or uniform_default_safe
        ):
            test_cases.append({"num_tokens": num_token, "distributed": "uniform", "power_law_alpha": None})
        # Power-law variants
        power_law_default_safe = logged_tokens in DEFAULT_DSV3_CONTEXT_LOGGED_TOKENS
        if "power_law" in requested_distributions and (
            not using_default_tokens or power_law_default_safe
        ):
            for alpha in power_law_alphas:
                test_cases.append(
                    {
                        "num_tokens": num_token,
                        "distributed": "power_law",
                        "power_law_alpha": alpha,
                    }
                )
        if not recorded_first and recorded_case_available:
            test_cases.append(recorded_case)

    return test_cases


def get_moe_decode_test_cases(
    output_path: str | None = None,
    simulated_ep_size: int | None = None,
    topk: int | None = None,
    total_experts: int | None = None,
):
    """Get test cases for MoE decode phase including distribution and alpha.

    Returns a list of dicts with keys: 'num_tokens', 'distributed', 'power_law_alpha'.
    For uniform distribution, 'power_law_alpha' is None.
    """
    # Per-rank decode tokens.  For EP8 this logs the global grid
    # 8,16,32,64,96,128,144,160,192,224,256,320,384,512,640,768,1024.
    # The maximum remains 128 tokens/rank, the validated DeepEP LL capacity.
    batch_sizes = [
        1,
        2,
        4,
        5,
        8,
        12,
        16,
        18,
        20,
        24,
        28,
        32,
        36,
        40,
        48,
        64,
        80,
        96,
        112,
        128,
        160,
    ]
    requested_logged_tokens = _env_int_list("COLLECTOR_WIDEEP_MOE_DECODE_TOKENS")
    if requested_logged_tokens:
        if simulated_ep_size is None:
            raise ValueError(
                "COLLECTOR_WIDEEP_MOE_DECODE_TOKENS requires simulated_ep_size"
            )
        invalid = [
            value
            for value in requested_logged_tokens
            if value % simulated_ep_size
        ]
        if invalid:
            raise ValueError(
                "COLLECTOR_WIDEEP_MOE_DECODE_TOKENS are global table tokens "
                f"and must be divisible by EP={simulated_ep_size}: {invalid}"
            )
        batch_sizes = sorted(
            {value // simulated_ep_size for value in requested_logged_tokens}
        )
    requested_distributions = _env_str_set("COLLECTOR_WIDEEP_MOE_DISTRIBUTIONS")
    if requested_distributions is None:
        requested_distributions = {"uniform"}
        if "deepseek" in _selected_moe_model_id().lower():
            requested_distributions.add("power_law")
        if _has_recorded_distribution_file(output_path):
            requested_distributions.add("recorded")
    default_alphas = (
        [0.6, 1.01]
        if "deepseek" in _selected_moe_model_id().lower()
        else [0.6, 0.8, 1.01, 1.02, 1.2]
    )
    power_law_alphas = (
        _env_float_list("COLLECTOR_WIDEEP_MOE_POWER_LAW_ALPHAS")
        or default_alphas
    )
    test_cases = []
    # Uniform cases
    if "uniform" in requested_distributions:
        for bs in batch_sizes:
            test_cases.append(
                {
                    "num_tokens": bs,
                    "distributed": "uniform",
                    "power_law_alpha": None,
                }
            )
    # Power-law cases
    if "power_law" in requested_distributions:
        for bs in batch_sizes:
            for alpha in power_law_alphas:
                test_cases.append(
                    {
                        "num_tokens": bs,
                        "distributed": "power_law",
                        "power_law_alpha": alpha,
                    }
                )
    include_recorded = (
        "recorded" in requested_distributions
        and _has_recorded_distribution_file(output_path)
    )
    if include_recorded and simulated_ep_size is not None and topk is not None and total_experts is not None:
        for bs in batch_sizes:
            if _recorded_distribution_has_case(
                output_path=output_path,
                num_tokens=bs * simulated_ep_size,
                topk=topk,
                total_experts=total_experts,
                preferred_recorder_ep_size=simulated_ep_size,
            ):
                test_cases.append(
                    {
                        "num_tokens": bs,
                        "distributed": "recorded",
                        "power_law_alpha": None,
                    }
                )
    return test_cases


def load_model_with_dummy_weights(server_args, port_args, tp_rank):
    """Load model with dummy weights and limited layers for MoE testing"""
    suppress_other_loggers()
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    if server_args.load_format == "dummy":
        existing_override = {}
        if server_args.json_model_override_args:
            existing_override = json.loads(server_args.json_model_override_args)

        num_layers = _get_requested_moe_num_layers()
        existing_override["num_hidden_layers"] = num_layers
        if num_layers <= DEFAULT_MOE_TEST_LAYER:
            existing_override["first_k_dense_replace"] = 0
        server_args.json_model_override_args = json.dumps(existing_override)

    model_config = ModelConfig.from_server_args(server_args)
    rank_print(f"Loading model with {model_config.num_hidden_layers} layers")
    rank_print(f"Will test MoE module from layer {_get_moe_test_layer(model_config.num_hidden_layers)}")

    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=tp_rank,
        tp_rank=tp_rank,
        tp_size=server_args.tp_size,
        pp_rank=0,
        pp_size=1,
        moe_ep_rank=tp_rank,
        moe_ep_size=server_args.ep_size,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
    )

    rank_print("Model loaded successfully.")

    if server_args.tp_size > 1:
        dist.barrier()

    return model_runner


def _make_replay_normal_dispatch_output(workload, hidden_size: int, device):
    layout = workload.dispatch_layout or {}
    hidden_layout = layout.get("hidden_states")
    scale_layout = layout.get("hidden_states_scale")
    if hidden_layout:
        hidden_states_fp8 = _make_tensor_from_recorded_layout(
            hidden_layout,
            device=device,
            random=True,
        )
        if hidden_states_fp8.shape[0] != workload.num_recv_tokens:
            raise ValueError(
                "Recorded normal dispatch layout token count does not match "
                f"workload: layout={hidden_states_fp8.shape[0]}, "
                f"workload={workload.num_recv_tokens}"
            )
    else:
        hidden_states = torch.randn(
            workload.num_recv_tokens,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        if hidden_size % 128:
            hidden_states = torch.nn.functional.pad(
                hidden_states,
                (0, 128 - hidden_size % 128),
            )
        hidden_states_fp8 = hidden_states.to(torch.float8_e4m3fn)
    if scale_layout:
        scale = _make_tensor_from_recorded_layout(
            scale_layout,
            device=device,
            fill_value=1,
        )
    else:
        scale = _make_scale_tensor(
            hidden_states_fp8.shape[0],
            hidden_states_fp8.shape[1],
            device,
        )
    topk_ids = workload.local_topk_ids.to(device=device, dtype=torch.int32)
    num_recv_tokens_per_expert = [
        int(count) for count in workload.num_recv_tokens_per_expert
    ]
    # Keep recorded replay tensors in one contract by default: hidden/topk rows
    # describe the real rank-local receive tokens, and expert counts describe the
    # same real assignments.  Padding counts without padding the token buffers
    # can make DeepEP normal scatter read past the replayed hidden states on
    # sparse router-shaped context workloads.
    if get_bool_env_var("COLLECTOR_WIDEEP_MOE_NORMAL_REPLAY_PAD_COUNTS", "true"):
        block_e = int(
            os.environ.get("COLLECTOR_WIDEEP_MOE_NORMAL_REPLAY_BLOCK_E", "128")
        )
        padding_mode = os.environ.get(
            "COLLECTOR_WIDEEP_MOE_NORMAL_REPLAY_PADDING_MODE",
            "per_expert",
        ).strip().lower()
        if padding_mode == "per_expert":
            num_recv_tokens_per_expert = [
                ((count + block_e - 1) // block_e) * block_e if count > 0 else 0
                for count in num_recv_tokens_per_expert
            ]
        elif padding_mode == "rank_total":
            total = sum(num_recv_tokens_per_expert)
            remainder = total % block_e
            if total > 0 and remainder:
                pad = block_e - remainder
                target = max(
                    range(len(num_recv_tokens_per_expert)),
                    key=lambda idx: num_recv_tokens_per_expert[idx],
                )
                num_recv_tokens_per_expert[target] += pad
        else:
            raise ValueError(
                "COLLECTOR_WIDEEP_MOE_NORMAL_REPLAY_PADDING_MODE must be "
                f"'rank_total' or 'per_expert', got {padding_mode!r}"
            )
    actual_counts = torch.bincount(
        topk_ids[topk_ids >= 0].to(torch.int64),
        minlength=len(num_recv_tokens_per_expert),
    ).tolist()
    pad_rows = []
    for expert_id, capacity in enumerate(num_recv_tokens_per_expert):
        deficit = int(capacity) - int(actual_counts[expert_id])
        if deficit <= 0:
            continue
        pad_row = torch.full(
            (deficit, topk_ids.shape[1]),
            -1,
            dtype=torch.int32,
            device=device,
        )
        pad_row[:, 0] = int(expert_id)
        pad_rows.append(pad_row)
    if pad_rows:
        padding_topk_ids = torch.cat(pad_rows, dim=0)
        topk_ids = torch.cat([topk_ids, padding_topk_ids], dim=0)
        padding_hidden = torch.randn(
            padding_topk_ids.shape[0],
            hidden_states_fp8.shape[1],
            dtype=torch.bfloat16,
            device=device,
        ).to(torch.float8_e4m3fn)
        hidden_states_fp8 = torch.cat([hidden_states_fp8, padding_hidden], dim=0)
        padding_scale = _make_scale_tensor(
            padding_topk_ids.shape[0],
            hidden_states_fp8.shape[1],
            device,
        )
        scale = torch.cat([scale, padding_scale], dim=0)
    topk_weights = torch.where(
        topk_ids >= 0,
        torch.full_like(
            topk_ids,
            1.0 / max(1, topk_ids.shape[1]),
            dtype=torch.float32,
        ),
        torch.zeros_like(topk_ids, dtype=torch.float32),
    )
    return DeepEPNormalDispatchOutput(
        hidden_states=hidden_states_fp8,
        hidden_states_scale=scale,
        topk_ids=topk_ids.contiguous(),
        topk_weights=topk_weights.contiguous(),
        num_recv_tokens_per_expert=num_recv_tokens_per_expert,
    )


def _torch_dtype_from_name(name: str):
    dtype = getattr(torch, str(name).removeprefix("torch."), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported recorded tensor dtype: {name}")
    return dtype


def _make_tensor_from_recorded_layout(
    layout,
    *,
    device,
    random: bool = False,
    fill_value=0,
):
    shape = tuple(int(value) for value in layout["shape"])
    stride = tuple(int(value) for value in layout["stride"])
    dtype = _torch_dtype_from_name(layout["dtype"])
    tensor = torch.empty_strided(
        shape,
        stride,
        dtype=dtype,
        device=device,
    )
    if random:
        source = torch.randn(shape, dtype=torch.bfloat16, device=device)
        tensor.copy_(source)
    else:
        tensor.fill_(fill_value)
    return tensor


def _percentile(values, q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


class _ReplayStageRecorder(AbstractContextManager):
    """Record the same CUDA-kernel stages for normal and low-latency replay."""

    def __init__(self):
        self._patches = []
        self._events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self._kernel_wrappers: list[str] = []
        self._gemm_index = 0

    def _patch(self, owner, name: str, stage_name):
        original = getattr(owner, name)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            label = stage_name() if callable(stage_name) else stage_name
            self._kernel_wrappers.append(name)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = original(*args, **kwargs)
            end.record()
            self._events.append((label, start, end))
            return result

        setattr(owner, name, wrapped)
        self._patches.append((owner, name, original))

    def _next_gemm(self):
        self._gemm_index += 1
        return "gemm1" if self._gemm_index % 2 else "gemm2"

    def __enter__(self):
        from sglang.srt.layers import deep_gemm_wrapper
        from sglang.srt.layers.moe.ep_moe import kernels as ep_kernels
        from sglang.srt.layers.moe.moe_runner import deep_gemm as deep_gemm_runner
        from sglang.srt.layers.quantization import fp8_kernel

        for name in (
            "grouped_gemm_nt_f8f8bf16_contig",
            "grouped_gemm_nt_f8f8bf16_masked",
        ):
            self._patch(deep_gemm_wrapper, name, self._next_gemm)
        self._patch(ep_kernels, "ep_scatter", "scatter")
        self._patch(ep_kernels, "ep_gather", "gather")
        self._patch(ep_kernels, "tma_align_input_scale", "scale_align")
        self._patch(
            deep_gemm_wrapper,
            "get_mn_major_tma_aligned_tensor",
            "scale_align",
        )
        self._patch(
            ep_kernels,
            "silu_and_mul_masked_post_quant_fwd",
            "activation_quant",
        )
        self._patch(deep_gemm_runner, "silu_and_mul", "activation")
        self._patch(
            fp8_kernel,
            "sglang_per_token_group_quant_fp8",
            "quant",
        )
        self._patch(
            fp8_kernel,
            "sglang_per_token_group_quant_8bit",
            "activation_quant",
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for owner, name, original in reversed(self._patches):
            setattr(owner, name, original)
        return False

    def durations_ms(self) -> dict[str, float]:
        durations: dict[str, float] = defaultdict(float)
        for name, start, end in self._events:
            durations[name] += start.elapsed_time(end)
        return dict(durations)

    def kernel_wrappers(self) -> list[str]:
        return list(self._kernel_wrappers)


def _layout_fingerprint(layout) -> dict[str, object]:
    if not layout:
        return {
            "shape": "",
            "stride": "",
            "dtype": "",
            "is_contiguous": "",
        }
    return {
        "shape": "x".join(str(int(value)) for value in layout.get("shape", ())),
        "stride": "x".join(str(int(value)) for value in layout.get("stride", ())),
        "dtype": str(layout.get("dtype", "")),
        "is_contiguous": layout.get("is_contiguous", ""),
    }


def _runtime_tensor_layout(value) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "dtype": str(value.dtype).removeprefix("torch."),
        "is_contiguous": value.is_contiguous(),
    }


def _dispatch_output_layout(dispatch_output) -> dict[str, object]:
    return {
        "hidden_states": _runtime_tensor_layout(
            getattr(dispatch_output, "hidden_states", None)
        ),
        "hidden_states_scale": _runtime_tensor_layout(
            getattr(dispatch_output, "hidden_states_scale", None)
        ),
        "topk_ids": _runtime_tensor_layout(
            getattr(dispatch_output, "topk_ids", None)
        ),
        "topk_weights": _runtime_tensor_layout(
            getattr(dispatch_output, "topk_weights", None)
        ),
        "masked_m": _runtime_tensor_layout(
            getattr(dispatch_output, "masked_m", None)
        ),
        "expected_m": int(getattr(dispatch_output, "expected_m", 0) or 0),
    }


def _replay_kernel_template_fingerprint(
    *,
    phase: str,
    replay_samples,
    capacity: int,
    kernel_regime: str,
    observed_kernel_wrappers: list[str],
    observed_dispatch_layouts: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    workloads = [workload for sample in replay_samples for workload in sample]
    layouts = (
        observed_dispatch_layouts
        if observed_dispatch_layouts
        else [workload.dispatch_layout or {} for workload in workloads]
    )
    hidden_layouts = [
        _layout_fingerprint(layout.get("hidden_states")) for layout in layouts
    ]
    scale_layouts = [
        _layout_fingerprint(layout.get("hidden_states_scale")) for layout in layouts
    ]
    topk_layouts = [
        _layout_fingerprint(layout.get("topk_ids")) for layout in layouts
    ]
    expected_m = [int(workload.expected_m) for workload in workloads]
    masked_m_max = [
        int(workload.masked_m.max().item()) if workload.masked_m.numel() else 0
        for workload in workloads
    ]
    wrappers = sorted(set(observed_kernel_wrappers))
    has_masked = any("masked" in wrapper for wrapper in wrappers)
    has_contig = any("contig" in wrapper for wrapper in wrappers)
    quant_scale_path = (
        "recorded_fp8_hidden_scale_layout"
        if any(layout["shape"] for layout in scale_layouts)
        else "synthetic_fp8_hidden_scale"
    )
    fingerprint = {
        "phase": phase,
        "kernel_regime": kernel_regime,
        "replay_capacity": capacity,
        "gemm_wrappers": wrappers,
        "gemm_path": (
            "mixed_masked_contiguous"
            if has_masked and has_contig
            else "masked"
            if has_masked
            else "contiguous"
            if has_contig
            else "not_observed"
        ),
        "expected_m_min": min(expected_m, default=0),
        "expected_m_max": max(expected_m, default=0),
        "masked_m_max": max(masked_m_max, default=0),
        "hidden_layouts": hidden_layouts,
        "scale_layouts": scale_layouts,
        "topk_layouts": topk_layouts,
        "quant_scale_path": quant_scale_path,
    }
    return {
        "kernel_template_fingerprint_json": json.dumps(
            fingerprint,
            sort_keys=True,
        ),
        "gemm_wrapper_json": json.dumps(wrappers),
        "gemm_path": fingerprint["gemm_path"],
        "quant_scale_path": quant_scale_path,
        "template_expected_m_min": fingerprint["expected_m_min"],
        "template_expected_m_max": fingerprint["expected_m_max"],
        "template_masked_m_max": fingerprint["masked_m_max"],
    }


def _replay_workload_features(replay_samples) -> dict[str, float | int]:
    workloads = [workload for sample in replay_samples for workload in sample]
    rank_totals = np.asarray(
        [
            sum(workload.num_recv_tokens_per_expert)
            for workload in workloads
        ],
        dtype=np.float64,
    )
    expert_m = np.asarray(
        [
            count
            for workload in workloads
            for count in workload.num_recv_tokens_per_expert
            if count > 0
        ],
        dtype=np.float64,
    )
    active_experts = [
        sum(count > 0 for count in workload.num_recv_tokens_per_expert)
        for workload in workloads
    ]
    expected_m = [workload.expected_m for workload in workloads]
    return {
        "replay_samples": len(replay_samples),
        "replay_ranks": max((len(sample) for sample in replay_samples), default=0),
        "workload_total_assignments_mean": float(rank_totals.mean()) if rank_totals.size else 0.0,
        "workload_rank_assignments_max": float(rank_totals.max()) if rank_totals.size else 0.0,
        "workload_rank_imbalance_max_over_mean": (
            float(rank_totals.max() / rank_totals.mean())
            if rank_totals.size and rank_totals.mean()
            else 0.0
        ),
        "workload_active_experts_mean": float(np.mean(active_experts)) if active_experts else 0.0,
        "workload_active_experts_max": max(active_experts, default=0),
        "workload_expert_m_mean": float(expert_m.mean()) if expert_m.size else 0.0,
        "workload_expert_m_p50": _percentile(expert_m, 50) if expert_m.size else 0.0,
        "workload_expert_m_p90": _percentile(expert_m, 90) if expert_m.size else 0.0,
        "workload_expert_m_max": float(expert_m.max()) if expert_m.size else 0.0,
        "workload_expected_m_mean": float(np.mean(expected_m)) if expected_m else 0.0,
        "workload_expected_m_max": max(expected_m, default=0),
    }


def _summarize_replay_timings(
    *,
    cold_rank_latencies: list[float],
    steady_round_rank_latencies: list[list[float]],
    steady_round_rank_critical_latencies: list[list[float]],
    rank_latency_history: dict[int, list[float]],
    rank_critical_latency_history: dict[int, list[float]],
    stage_duration_history: dict[str, list[float]],
    multistream_round_latencies: list[float],
    multistream_probe_error: str | None,
    workload_features: dict[str, float | int],
    capacity: int,
    kernel_regime: str,
    policy: ReplayMeasurementPolicy | None = None,
) -> dict[str, object]:
    policy = policy or _replay_measurement_policy()
    raw_stage_round_max = [
        max(values) for values in steady_round_rank_latencies if values
    ]
    raw_critical_round_max = [
        max(values)
        for values in steady_round_rank_critical_latencies
        if values
    ]
    raw_critical_round_mean = [
        float(np.mean(values))
        for values in steady_round_rank_critical_latencies
        if values
    ]
    if not raw_critical_round_max:
        raise ValueError("Recorded replay produced no steady-state timing rounds")

    # DeepGEMM can occasionally defer one template compile until a measured
    # invocation even after explicit warmups. Such samples are hundreds of
    # milliseconds while the steady kernel is sub-ms or low-ms. Keep them
    # observable, but exclude them from every metric labelled steady-state.
    median_round_max = float(np.median(raw_critical_round_max))
    mad_round_max = float(
        np.median(
            np.abs(np.asarray(raw_critical_round_max) - median_round_max)
        )
    )
    # Keep genuine rank-tail variation while rejecting delayed compilation,
    # clock-transition and scheduler spikes that are not representative of a
    # steady kernel. The unfiltered maximum is retained separately.
    steady_threshold = median_round_max + max(
        policy.outlier_mad_multiplier * mad_round_max,
        policy.outlier_min_relative_headroom * median_round_max,
    )
    kept_indices = [
        index
        for index, value in enumerate(raw_critical_round_max)
        if value <= steady_threshold
    ]
    if not kept_indices:
        kept_indices = list(range(len(raw_critical_round_max)))
    critical_round_max = [
        raw_critical_round_max[index] for index in kept_indices
    ]
    critical_round_mean = [
        raw_critical_round_mean[index] for index in kept_indices
    ]
    stage_round_max = [
        raw_stage_round_max[index]
        for index in kept_indices
        if index < len(raw_stage_round_max)
    ]

    def _steady_values(values: list[float]) -> list[float]:
        if not values:
            return values
        median = float(np.median(values))
        mad = float(np.median(np.abs(np.asarray(values) - median)))
        threshold = median + max(
            policy.outlier_mad_multiplier * mad,
            policy.outlier_min_relative_headroom * median,
        )
        kept = [value for value in values if value <= threshold]
        return kept or values

    rank_means = {
        rank: float(np.mean(_steady_values(values)))
        for rank, values in rank_latency_history.items()
        if values
    }
    rank_p90 = {
        rank: _percentile(_steady_values(values), 90)
        for rank, values in rank_latency_history.items()
        if values
    }
    rank_critical_means = {
        rank: float(np.mean(_steady_values(values)))
        for rank, values in rank_critical_latency_history.items()
        if values
    }
    stage_means = {
        stage: float(np.mean(_steady_values(values)))
        for stage, values in stage_duration_history.items()
        if values
    }
    stage_p90 = {
        stage: _percentile(_steady_values(values), 90)
        for stage, values in stage_duration_history.items()
        if values
    }
    return {
        # ``run_moe_core`` elapsed time is the authoritative standalone
        # compute latency. Function-level stage events are observability only:
        # they can omit kernels and include launch gaps, so they must not be
        # substituted for either a CUPTI kernel sum or the critical path.
        "latency": float(np.mean(critical_round_max)),
        "latency_p90": _percentile(critical_round_max, 90),
        "latency_max": float(max(critical_round_max)),
        "latency_raw_max": float(max(raw_critical_round_max)),
        "latency_stage_sum_rankmax": float(np.mean(stage_round_max)),
        "latency_stage_sum_rankmax_max": float(max(stage_round_max)),
        "cold_latency": float(max(cold_rank_latencies)) if cold_rank_latencies else 0.0,
        "rank_mean_latency": float(np.mean(list(rank_means.values()))),
        "rank_p90_latency": float(max(rank_p90.values())),
        "rank_critical_mean_latency": float(
            np.mean(list(rank_critical_means.values()))
        ),
        "rank_sync_tail_mean": float(
            np.mean(
                np.asarray(critical_round_max)
                - np.asarray(critical_round_mean)
            )
        ),
        "rank_steady_mean_ms_json": json.dumps(rank_means, sort_keys=True),
        "rank_steady_p90_ms_json": json.dumps(rank_p90, sort_keys=True),
        "stage_mean_ms_json": json.dumps(stage_means, sort_keys=True),
        "stage_p90_ms_json": json.dumps(stage_p90, sort_keys=True),
        "stage_kernel_sum_mean": float(sum(stage_means.values())),
        "multistream_latency": (
            float(np.mean(multistream_round_latencies))
            if multistream_round_latencies
            else 0.0
        ),
        "multistream_latency_p90": (
            _percentile(multistream_round_latencies, 90)
            if multistream_round_latencies
            else 0.0
        ),
        "multistream_measurement_rounds": len(multistream_round_latencies),
        "multistream_measurement_boundary": (
            "common_start_rank_stream_completion_max"
            if multistream_round_latencies
            else "probe_failed"
            if multistream_probe_error
            else "disabled"
        ),
        "multistream_probe_error": multistream_probe_error or "",
        "replay_policy": policy.name,
        "primary_latency_source": policy.primary_latency_source,
        "measurement_rounds": len(raw_critical_round_max),
        "measurement_steady_rounds": len(critical_round_max),
        "measurement_outliers_discarded": (
            len(raw_critical_round_max) - len(critical_round_max)
        ),
        "measurement_outlier_threshold_ms": steady_threshold,
        "measurement_boundary": policy.measurement_boundary,
        "stage_measurement_boundary": policy.stage_measurement_boundary,
        "measurement_rank_order": policy.rank_order,
        "replay_capacity": capacity,
        "kernel_regime": kernel_regime,
        **workload_features,
    }


def _measure_rank_local_replay(
    *,
    replay_samples,
    make_dispatch_output,
    run_moe_core,
    device,
    num_warmup: int,
    num_iterations: int,
    random_seed: int,
    capacity: int,
    kernel_regime: str,
    kernel_template_fingerprint,
    use_cuda_graph: bool = False,
) -> dict[str, object]:
    accelerator = torch.get_device_module(device)
    policy = _replay_measurement_policy()
    rng = random.Random(random_seed)
    cold_rank_latencies: list[float] = []
    steady_round_rank_latencies: list[list[float]] = []
    steady_round_rank_critical_latencies: list[list[float]] = []
    rank_latency_history: dict[int, list[float]] = {}
    rank_critical_latency_history: dict[int, list[float]] = {}
    stage_duration_history: dict[str, list[float]] = defaultdict(list)
    multistream_round_latencies: list[float] = []
    multistream_probe_error: str | None = None
    observed_kernel_wrappers: list[str] = []
    observed_dispatch_layouts: list[dict[str, object]] = []

    def _timed_call(workload):
        dispatch_output = make_dispatch_output(workload)
        observed_dispatch_layouts.append(_dispatch_output_layout(dispatch_output))
        accelerator.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        if use_cuda_graph:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run_moe_core(dispatch_output)
            accelerator.synchronize()
            start_event.record()
            graph.replay()
            end_event.record()
            end_event.synchronize()
            critical_latency = start_event.elapsed_time(end_event)
            stage_durations = {"cuda_graph_replay": critical_latency}
            kernel_sum_latency = critical_latency
        else:
            with _ReplayStageRecorder() as stage_recorder:
                start_event.record()
                run_moe_core(dispatch_output)
                end_event.record()
                end_event.synchronize()
            critical_latency = start_event.elapsed_time(end_event)
            stage_durations = stage_recorder.durations_ms()
            observed_kernel_wrappers.extend(stage_recorder.kernel_wrappers())
            kernel_sum_latency = sum(stage_durations.values()) or critical_latency
        return critical_latency, kernel_sum_latency, stage_durations

    # Record the first invocation separately. It is useful for observability
    # but never contributes to the steady-state latency installed in AIC.
    for sample in replay_samples:
        order = list(sample)
        rng.shuffle(order)
        active_order = [
            workload
            for workload in order
            if workload.num_recv_tokens > 0
            or sum(workload.num_recv_tokens_per_expert) > 0
        ]
        for workload in active_order:
            critical, kernel_sum, _ = _timed_call(workload)
            cold_rank_latencies.append(critical)

    # JIT/cache warmup is deliberately untimed and discarded.
    for sample in replay_samples:
        for _ in range(num_warmup):
            order = list(sample)
            rng.shuffle(order)
            active_order = [
                workload
                for workload in order
                if workload.num_recv_tokens > 0
                or sum(workload.num_recv_tokens_per_expert) > 0
            ]
            for workload in active_order:
                run_moe_core(make_dispatch_output(workload))
                # DeepGEMM may finish template compilation lazily. A single
                # synchronize after the whole warmup batch can let that work
                # spill into the first measured rank. Force every rank/template
                # warmup to finish before steady-state rounds begin.
                accelerator.synchronize()
    accelerator.synchronize()

    for sample in replay_samples:
        for _ in range(num_iterations):
            order = list(sample)
            rng.shuffle(order)
            active_order = [
                workload
                for workload in order
                if workload.num_recv_tokens > 0
                or sum(workload.num_recv_tokens_per_expert) > 0
            ]
            rank_latencies = []
            rank_critical_latencies = []
            for workload in active_order:
                critical, kernel_sum, stage_durations = _timed_call(workload)
                rank_latencies.append(kernel_sum)
                rank_critical_latencies.append(critical)
                rank_latency_history.setdefault(workload.rank, []).append(
                    kernel_sum
                )
                rank_critical_latency_history.setdefault(
                    workload.rank, []
                ).append(critical)
                for stage, duration in stage_durations.items():
                    stage_duration_history[stage].append(duration)
            steady_round_rank_latencies.append(rank_latencies)
            steady_round_rank_critical_latencies.append(rank_critical_latencies)

    if policy.multistream_probe:
        try:
            for sample in replay_samples:
                active_sample = [
                    workload
                    for workload in sample
                    if workload.num_recv_tokens > 0
                    or sum(workload.num_recv_tokens_per_expert) > 0
                ]
                if not active_sample:
                    continue
                streams = [torch.cuda.Stream() for _ in active_sample]
                for _ in range(policy.multistream_rounds):
                    # Build every rank's dispatch input on the default stream, then
                    # release all rank streams from one ready event. This applies
                    # exactly the same concurrency model to normal and low-latency
                    # replay while retaining sequential rank timing as the primary
                    # standalone AIC value.
                    dispatch_outputs = [
                        make_dispatch_output(workload)
                        for workload in active_sample
                    ]
                    ready = torch.cuda.Event(enable_timing=True)
                    ready.record()
                    end_events = []
                    for stream, dispatch_output in zip(streams, dispatch_outputs):
                        stream.wait_event(ready)
                        with torch.cuda.stream(stream):
                            run_moe_core(dispatch_output)
                            end = torch.cuda.Event(enable_timing=True)
                            end.record()
                            end_events.append(end)
                    for end in end_events:
                        end.synchronize()
                    multistream_round_latencies.append(
                        max(ready.elapsed_time(end) for end in end_events)
                    )
        except Exception as exc:
            multistream_round_latencies.clear()
            multistream_probe_error = str(exc)
            try:
                accelerator.synchronize()
            except Exception:
                pass

    return _summarize_replay_timings(
        policy=policy,
        cold_rank_latencies=cold_rank_latencies,
        steady_round_rank_latencies=steady_round_rank_latencies,
        steady_round_rank_critical_latencies=(
            steady_round_rank_critical_latencies
        ),
        rank_latency_history=rank_latency_history,
        rank_critical_latency_history=rank_critical_latency_history,
        stage_duration_history=stage_duration_history,
        multistream_round_latencies=multistream_round_latencies,
        multistream_probe_error=multistream_probe_error,
        workload_features=_replay_workload_features(replay_samples),
        capacity=capacity,
        kernel_regime=kernel_regime,
    ) | kernel_template_fingerprint(
        observed_kernel_wrappers,
        observed_dispatch_layouts,
    )


def _benchmark_rank_local_prefill_replay(
    *,
    moe_layer,
    replay_samples,
    hidden_size: int,
    device,
    num_warmup: int,
    num_iterations: int,
) -> dict[str, object]:
    seed = int(
        os.environ.get(
            "COLLECTOR_WIDEEP_MOE_REPLAY_RANDOM_SEED",
            str(DEFAULT_REPLAY_RANDOM_SEED),
        )
    )
    return _measure_rank_local_replay(
        replay_samples=replay_samples,
        make_dispatch_output=lambda workload: _make_replay_normal_dispatch_output(
            workload,
            hidden_size,
            device,
        ),
        run_moe_core=moe_layer.experts.run_moe_core,
        device=device,
        num_warmup=max(1, num_warmup),
        num_iterations=max(
            1,
            int(
                os.environ.get(
                    "COLLECTOR_WIDEEP_MOE_REPLAY_ROUNDS",
                    str(max(20, num_iterations)),
                )
            ),
        ),
        random_seed=seed,
        capacity=0,
        kernel_regime="normal_contiguous",
        kernel_template_fingerprint=lambda observed_wrappers, observed_layouts: _replay_kernel_template_fingerprint(
            phase="context",
            replay_samples=replay_samples,
            capacity=0,
            kernel_regime="normal_contiguous",
            observed_kernel_wrappers=observed_wrappers,
            observed_dispatch_layouts=observed_layouts,
        ),
        use_cuda_graph=_wideep_moe_use_cuda_graph_for_phase("context"),
    )


def _make_replay_ll_dispatch_output(workload, hidden_size: int, capacity: int, device):
    num_local_experts = len(workload.num_recv_tokens_per_expert)
    masked_m = workload.masked_m.to(device=device, dtype=torch.int32)
    if int(masked_m.max().item()) > capacity:
        raise ValueError(
            f"Replay masked_m exceeds decode capacity: "
            f"max={int(masked_m.max().item())}, capacity={capacity}"
        )
    layout = workload.dispatch_layout or {}
    hidden_layout = layout.get("hidden_states")
    scale_layout = layout.get("hidden_states_scale")
    if hidden_layout:
        hidden_states_fp8 = _make_tensor_from_recorded_layout(
            hidden_layout,
            device=device,
            random=True,
        )
        expected_shape = (num_local_experts, capacity)
        if tuple(hidden_states_fp8.shape[:2]) != expected_shape:
            raise ValueError(
                "Recorded low-latency dispatch layout does not match replay "
                f"experts/capacity: layout={tuple(hidden_states_fp8.shape)}, "
                f"expected_prefix={expected_shape}"
            )
    else:
        hidden_states = torch.randn(
            num_local_experts,
            capacity,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        if hidden_size % 128:
            hidden_states = torch.nn.functional.pad(
                hidden_states,
                (0, 128 - hidden_size % 128),
            )
        hidden_states_fp8 = hidden_states.to(torch.float8_e4m3fn)
    if scale_layout:
        scale = _make_tensor_from_recorded_layout(
            scale_layout,
            device=device,
            fill_value=1,
        )
    else:
        scale = torch.ones(
            num_local_experts,
            capacity,
            hidden_states_fp8.shape[-1] // 128,
            device=device,
            dtype=torch.float32,
        )
    return DeepEPLLDispatchOutput(
        hidden_states=hidden_states_fp8,
        hidden_states_scale=scale,
        topk_ids=torch.empty(0, device=device, dtype=torch.int32),
        topk_weights=torch.empty(0, device=device, dtype=torch.float32),
        masked_m=masked_m,
        expected_m=workload.expected_m,
    )


def _benchmark_rank_local_decode_replay(
    *,
    moe_layer,
    replay_samples,
    hidden_size: int,
    device,
    num_warmup: int,
    num_iterations: int,
) -> dict[str, object]:
    active_samples = [
        [
            workload
            for workload in sample
            if workload.expected_m > 0
            and int(workload.masked_m.sum().item()) > 0
        ]
        for sample in replay_samples
    ]
    active_samples = [sample for sample in active_samples if sample]
    if not active_samples:
        raise ValueError("Rank-local decode replay contains no active rank")
    max_masked_m = max(
        int(workload.masked_m.max().item())
        for sample in active_samples
        for workload in sample
    )
    ep_size = max(len(sample) for sample in replay_samples)
    # Match DeepEP LL's dispatch buffer layout. The runtime allocates
    # max-dispatch-tokens-per-rank for every source EP rank, so the per-expert
    # M capacity is 128 * EP even when the observed masked_m is sparse.
    # Shrinking replay to ceil(max(masked_m), 128) changes DeepGEMM template
    # selection and materially underestimates the real server boundary.
    recorded_capacities = {
        int(workload.dispatch_layout["hidden_states"]["shape"][1])
        for sample in active_samples
        for workload in sample
        if workload.dispatch_layout
        and workload.dispatch_layout.get("hidden_states")
    }
    if len(recorded_capacities) > 1:
        raise ValueError(
            f"Inconsistent recorded low-latency capacities: "
            f"{sorted(recorded_capacities)}"
        )
    capacity = (
        recorded_capacities.pop()
        if recorded_capacities
        else max(
            128 * ep_size,
            _ceil(max_masked_m / 128) * 128,
        )
    )
    seed = int(
        os.environ.get(
            "COLLECTOR_WIDEEP_MOE_REPLAY_RANDOM_SEED",
            str(DEFAULT_REPLAY_RANDOM_SEED),
        )
    )
    return _measure_rank_local_replay(
        replay_samples=active_samples,
        make_dispatch_output=lambda workload: _make_replay_ll_dispatch_output(
            workload,
            hidden_size,
            capacity,
            device,
        ),
        run_moe_core=moe_layer.experts.run_moe_core,
        device=device,
        num_warmup=max(1, num_warmup),
        num_iterations=max(
            1,
            int(
                os.environ.get(
                    "COLLECTOR_WIDEEP_MOE_REPLAY_ROUNDS",
                    str(max(20, num_iterations)),
                )
            ),
        ),
        random_seed=seed,
        capacity=capacity,
        kernel_regime=f"low_latency_masked_capacity_{capacity}",
        kernel_template_fingerprint=lambda observed_wrappers, observed_layouts: _replay_kernel_template_fingerprint(
            phase="generation",
            replay_samples=active_samples,
            capacity=capacity,
            kernel_regime=f"low_latency_masked_capacity_{capacity}",
            observed_kernel_wrappers=observed_wrappers,
            observed_dispatch_layouts=observed_layouts,
        ),
        use_cuda_graph=_wideep_moe_use_cuda_graph_for_phase("generation"),
    )


def benchmark_moe_layer_prefill(
    model_runner,
    server_args,
    port_args,
    num_warmup,
    num_iterations,
    test_layer,
    rank_print,
    device,
    tp_rank,
    prefill_test_cases,
    moe_layer,
    num_local_experts,
    simulated_ep_size,
    output_path,
    model_hidden_size,
    model_inter_size,
    model_total_experts,
):
    """Benchmark MoE layer in prefill phase

    Args:
        num_local_experts: Number of experts on this GPU (= model's n_routed_experts or num_experts)
        simulated_ep_size: The EP size being simulated (= total_experts / num_local_experts)
        model_hidden_size: Model's hidden_size from config
        model_inter_size: Model's moe_intermediate_size from config
        model_total_experts: Total number of experts in the model (256 for DeepSeek-V3, 128 for Qwen3)
    """

    logged_count = 0
    recorded_expected_count = 0
    recorded_logged_count = 0
    max_local_assignments = _get_wideep_moe_max_local_assignments()
    for case in prefill_test_cases:
        try:
            # Backward compatible: old format was just an int
            if isinstance(case, dict):
                num_token = case["num_tokens"]
                distributed = case.get("distributed", "uniform")
                power_law_alpha = case.get("power_law_alpha", 0.8) if distributed == "power_law" else None
            else:
                num_token = int(case)
                distributed = "uniform"
                power_law_alpha = None

            num_tokens_log = num_token * simulated_ep_size
            replay_samples = (
                _rank_local_replay_samples(
                    phase="context",
                    table_num_tokens=num_tokens_log,
                    layer_id=test_layer,
                    ep_size=simulated_ep_size,
                    num_experts=model_total_experts,
                    output_path=output_path,
                )
                if distributed == "recorded"
                and _get_rank_local_replay_dir(output_path)
                else None
            )
            if distributed == "recorded":
                recorded_expected_count += 1
            if replay_samples is not None:
                collector_dir = os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                )
                perf_filename = (
                    os.path.join(collector_dir, "wideep_context_moe_perf.txt")
                    if output_path is None
                    else os.path.join(output_path, "wideep_context_moe_perf.txt")
                )

                def _measure_recorded_prefill_item_once() -> dict:
                    replay_stats = _benchmark_rank_local_prefill_replay(
                        moe_layer=moe_layer,
                        replay_samples=replay_samples,
                        hidden_size=model_hidden_size,
                        device=device,
                        num_warmup=num_warmup,
                        num_iterations=num_iterations,
                    )
                    replay_stats = _apply_recorded_latency_policy(
                        replay_stats,
                        phase="context",
                        num_tokens_log=num_tokens_log,
                    )
                    rank_print(
                        "Rank-local recorded replay (Prefill): "
                        f"samples={len(replay_samples)}, ranks={simulated_ep_size}, "
                        f"steady mean/p90/max={replay_stats['latency']:.3f}/"
                        f"{replay_stats['latency_p90']:.3f}/"
                        f"{replay_stats['latency_max']:.3f}ms, "
                        f"cold={replay_stats['cold_latency']:.3f}ms"
                    )
                    item = {
                        "moe_dtype": "fp8_block",
                        "num_tokens": num_tokens_log,
                        "hidden_size": model_hidden_size,
                        "inter_size": model_inter_size,
                        "topk": moe_layer.topk.topk_config.top_k,
                        "num_experts": model_total_experts,
                        "moe_tp_size": 1,
                        "moe_ep_size": simulated_ep_size,
                        "distribution": _recorded_output_distribution(),
                        "workload_source": _rank_local_replay_source(),
                        "measurement_scope": "single_card_rank_local_replay",
                        **replay_stats,
                    }
                    return _apply_profile_free_hybrid_recorded_row(
                        item,
                        phase="context",
                    )

                def _reset_recorded_prefill_guard_attempt() -> None:
                    torch.get_device_module(device).synchronize()
                    model_runner.req_to_token_pool.clear()
                    model_runner.token_to_kv_pool_allocator.clear()
                    torch.cuda.empty_cache()

                item = select_recorded_source(
                    family="wideep_context",
                    phase="context",
                    ep=simulated_ep_size,
                    eplb="on" if _rank_local_replay_enable_eplb() else "off",
                    token=num_tokens_log,
                    measure_once=_measure_recorded_prefill_item_once,
                    reset_before_attempt=_reset_recorded_prefill_guard_attempt,
                    warmup_once=_measure_recorded_prefill_item_once,
                    output_path=output_path,
                    rank_print=rank_print,
                    kernel_source="deepepmoe_rank_local_replay",
                )

                if tp_rank == 0:
                    log_perf(
                        item_list=[item],
                        framework="SGLang",
                        version=get_version("sglang"),
                        device_name=torch.cuda.get_device_name(server_args.device),
                        op_name="moe_context",
                        kernel_source="deepepmoe_rank_local_replay",
                        perf_filename=perf_filename,
                    )
                    logged_count += 1
                    recorded_logged_count += 1
                torch.cuda.empty_cache()
                continue

            model_runner.req_to_token_pool.clear()
            model_runner.token_to_kv_pool_allocator.clear()

            # Fake dispatch outputs with random data
            hidden_states_per_token_iter = torch.randn(
                int(num_token * simulated_ep_size),
                model_runner.model.config.hidden_size,
                dtype=torch.bfloat16,
                device=device,
            )

            if hidden_states_per_token_iter.shape[1] % 128 != 0:
                pad_size = 128 - (hidden_states_per_token_iter.shape[1] % 128)
                hidden_states_per_token_iter = torch.nn.functional.pad(hidden_states_per_token_iter, (0, pad_size))

            num_tokens_iter = hidden_states_per_token_iter.shape[0]
            topk = moe_layer.topk.topk_config.top_k
            topk_idx_iter = torch.full((num_tokens_iter, topk), -1, device=device, dtype=torch.int32)
            topk_weights_iter = torch.zeros((num_tokens_iter, topk), device=device, dtype=torch.float32)

            if distributed == "uniform":
                tokens_per_local_expert = int(num_tokens_iter * topk // num_local_experts)
                rank_print(f"tokens_per_local_expert: {tokens_per_local_expert}")
                if tokens_per_local_expert <= 0:
                    continue
                num_recv = [tokens_per_local_expert] * num_local_experts
                expert_indices = torch.arange(
                    num_tokens_iter * topk,
                    device=device,
                    dtype=torch.int32,
                )
                topk_idx_iter = (expert_indices % num_local_experts).reshape(num_tokens_iter, topk)
                topk_weights_iter.fill_(1.0 / topk)

            elif distributed == "power_law":
                # Use power_law_deepep_prefill to generate router logits for local experts
                # Generate multiple samples to avoid outliers from a single sampling
                power_law_samples = []
                for _ in range(5):
                    topk_idx_sample, topk_weights_sample, num_recv_tensor = power_law_deepep_prefill(
                        num_tokens_iter,
                        num_local_experts * simulated_ep_size,
                        topk,
                        simulated_ep_size,
                        power_law_alpha if power_law_alpha is not None else 0.8,
                    )
                    topk_idx_sample = topk_idx_sample.to(device).contiguous()
                    topk_weights_sample = topk_weights_sample.to(device).contiguous()
                    topk_weights_sample = torch.nan_to_num(topk_weights_sample, nan=0.0, posinf=0.0, neginf=0.0)
                    # ``power_law_deepep_prefill`` already maps rank-0 experts
                    # to local IDs and masks assignments owned by other EP
                    # ranks with -1.  Applying remainder here turns every -1
                    # into the final local expert and folds the entire global
                    # workload onto one rank, producing impossible loads.
                    valid_local_ids = topk_idx_sample[topk_idx_sample >= 0]
                    actual_num_recv = torch.bincount(
                        valid_local_ids,
                        minlength=num_local_experts,
                    ).to(torch.int32)
                    # DeepGEMM's normal DeepEP scatter path requires each
                    # expert capacity, and therefore their total, to be
                    # padded to BLOCK_E=128.  The helper returns exactly that
                    # padded capacity; the routing matrix still contains only
                    # the real assignments.
                    padded_num_recv = num_recv_tensor.to(
                        device=actual_num_recv.device,
                        dtype=torch.int32,
                    )
                    if torch.any(padded_num_recv < actual_num_recv):
                        raise ValueError(
                            "Power-law padded expert capacity is smaller than "
                            "the actual local routing load"
                        )
                    num_recv = padded_num_recv.tolist()
                    power_law_samples.append((topk_idx_sample, topk_weights_sample, num_recv))

            elif distributed == "recorded":
                topk_idx_iter, topk_weights_iter, num_recv = _build_recorded_prefill_sample(
                    num_tokens=num_tokens_iter,
                    topk=topk,
                    num_local_experts=num_local_experts,
                    total_experts=model_total_experts,
                    output_path=output_path,
                    preferred_layer_id=test_layer,
                    preferred_recorder_ep_size=simulated_ep_size,
                    device=device,
                )
                active_local_experts = sum(1 for count in num_recv if count > 0)
                rank_print(
                    "recorded distribution: "
                    f"global_tokens={num_tokens_iter}, active_local_experts={active_local_experts}, "
                    f"max_local_assignments={max(num_recv) if num_recv else 0}"
                )

            else:
                raise ValueError(f"Unsupported distributed mode: {distributed}")

            # For single-sample distributions, create a single-element list for unified processing.
            if distributed in ("uniform", "recorded"):
                # Safety clamp for weights
                topk_weights_iter = torch.nan_to_num(topk_weights_iter, nan=0.0, posinf=0.0, neginf=0.0)
                power_law_samples = [(topk_idx_iter, topk_weights_iter, num_recv)]

            if max_local_assignments:
                safe_samples = []
                for topk_idx_sample, topk_weights_sample, num_recv_sample in power_law_samples:
                    sample_max = max(num_recv_sample) if num_recv_sample else 0
                    if sample_max > max_local_assignments:
                        rank_print(
                            "Skipping prefill case because local expert load exceeds guard: "
                            f"global_tokens={num_tokens_iter}, distribution={distributed}, "
                            f"max_local_assignments={sample_max}, "
                            f"limit={max_local_assignments}"
                        )
                        continue
                    safe_samples.append((topk_idx_sample, topk_weights_sample, num_recv_sample))
                power_law_samples = safe_samples
                if not power_law_samples:
                    continue

            # Warmup
            for _ in range(num_warmup):
                for topk_idx_sample, topk_weights_sample, num_recv_sample in power_law_samples:
                    hidden_states_fp8_tensor_iter = hidden_states_per_token_iter.to(torch.float8_e4m3fn)
                    scale_tensor_iter = _make_scale_tensor(
                        hidden_states_per_token_iter.shape[0],
                        hidden_states_per_token_iter.shape[1],
                        hidden_states_per_token_iter.device,
                    )
                    if distributed in ("uniform", "recorded"):
                        (
                            hidden_states_fp8_tensor_iter,
                            scale_tensor_iter,
                            topk_idx_sample_run,
                            topk_weights_sample_run,
                            num_recv_sample_run,
                        ) = _pad_prefill_dispatch_contract(
                            hidden_states_fp8=hidden_states_fp8_tensor_iter,
                            scale_tensor=scale_tensor_iter,
                            topk_ids=topk_idx_sample.clone(),
                            topk_weights=topk_weights_sample.clone(),
                            num_recv_tokens_per_expert=num_recv_sample,
                            device=device,
                        )
                    else:
                        topk_idx_sample_run = topk_idx_sample.clone()
                        topk_weights_sample_run = topk_weights_sample.clone()
                        num_recv_sample_run = num_recv_sample
                    dispatch_output = DeepEPNormalDispatchOutput(
                        hidden_states=hidden_states_fp8_tensor_iter,
                        hidden_states_scale=scale_tensor_iter,
                        topk_ids=topk_idx_sample_run,
                        topk_weights=topk_weights_sample_run,
                        num_recv_tokens_per_expert=num_recv_sample_run,
                    )
                    _ = moe_layer.experts.run_moe_core(dispatch_output)

            torch.get_device_module(device).synchronize()
            torch.cuda.empty_cache()

            gemm_latencies = []

            for i in range(num_iterations):
                for topk_idx_sample, topk_weights_sample, num_recv_sample in power_law_samples:
                    hidden_states_fp8_tensor_iter = hidden_states_per_token_iter.to(torch.float8_e4m3fn)
                    scale_tensor_iter = _make_scale_tensor(
                        hidden_states_per_token_iter.shape[0],
                        hidden_states_per_token_iter.shape[1],
                        hidden_states_per_token_iter.device,
                    )
                    if distributed in ("uniform", "recorded"):
                        (
                            hidden_states_fp8_tensor_iter,
                            scale_tensor_iter,
                            topk_idx_sample_run,
                            topk_weights_sample_run,
                            num_recv_sample_run,
                        ) = _pad_prefill_dispatch_contract(
                            hidden_states_fp8=hidden_states_fp8_tensor_iter,
                            scale_tensor=scale_tensor_iter,
                            topk_ids=topk_idx_sample.clone(),
                            topk_weights=topk_weights_sample.clone(),
                            num_recv_tokens_per_expert=num_recv_sample,
                            device=device,
                        )
                    else:
                        topk_idx_sample_run = topk_idx_sample.clone()
                        topk_weights_sample_run = topk_weights_sample.clone()
                        num_recv_sample_run = num_recv_sample
                    dispatch_output = DeepEPNormalDispatchOutput(
                        hidden_states=hidden_states_fp8_tensor_iter,
                        hidden_states_scale=scale_tensor_iter,
                        topk_ids=topk_idx_sample_run,
                        topk_weights=topk_weights_sample_run,
                        num_recv_tokens_per_expert=num_recv_sample_run,
                    )
                    torch.get_device_module(device).synchronize()
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()

                    _ = moe_layer.experts.run_moe_core(dispatch_output)

                    end_event.record()
                    end_event.synchronize()
                    latency_ms = start_event.elapsed_time(end_event)
                    if i > 2:
                        gemm_latencies.append(latency_ms)

            torch.cuda.empty_cache()

            avg_latency_ms = np.mean(gemm_latencies)

            if tp_rank == 0:
                rank_print("DeepEP MoE GEMM Results (Prefill):")
                rank_print(f"  Average latency: {avg_latency_ms:.3f}ms")
            if tp_rank == 0:
                try:
                    moe_tp_size = 1
                    moe_ep_size = simulated_ep_size
                    device_name = torch.cuda.get_device_name(server_args.device)
                    version = get_version("sglang")
                    # Save to collector/ directory to match non-wideep behavior
                    collector_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    perf_filename = (
                        os.path.join(collector_dir, "wideep_context_moe_perf.txt")
                        if output_path is None
                        else os.path.join(output_path, "wideep_context_moe_perf.txt")
                    )
                    distribution_str = f"power_law_{power_law_alpha}" if distributed == "power_law" else distributed
                    log_perf(
                        item_list=[
                            {
                                "moe_dtype": "fp8_block",
                                "num_tokens": num_tokens_log,
                                "hidden_size": model_hidden_size,
                                "inter_size": model_inter_size,
                                "topk": topk,
                                "num_experts": model_total_experts,
                                "moe_tp_size": moe_tp_size,
                                "moe_ep_size": moe_ep_size,
                                "distribution": distribution_str,
                                "latency": avg_latency_ms,
                            }
                        ],
                        framework="SGLang",
                        version=version,
                        device_name=device_name,
                        op_name="moe_context",
                        kernel_source="deepepmoe",
                        perf_filename=perf_filename,
                    )
                    logged_count += 1
                except Exception as e:
                    rank_print(f"  Warning: failed to log prefill MoE metrics: {e}")
            del (
                hidden_states_per_token_iter,
                hidden_states_fp8_tensor_iter,
                scale_tensor_iter,
                topk_idx_iter,
                topk_weights_iter,
                num_recv,
                dispatch_output,
            )
            torch.cuda.empty_cache()

        except Exception as e:
            rank_print(f"Prefill case failed: {e}, skipping...")
            import traceback

            rank_print(traceback.format_exc())
            # Check if this is a CUDA error - if so, the context is corrupted and we should exit
            if "CUDA error" in str(e) or "illegal memory access" in str(e).lower():
                rank_print("CUDA error detected, exiting prefill benchmark early to avoid cascading failures")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                break
            try:
                torch.cuda.empty_cache()
            except Exception:
                # If empty_cache fails, CUDA context is corrupted
                rank_print("CUDA context corrupted, exiting prefill benchmark early")
                break
            continue
    return {
        "total": logged_count,
        "recorded_expected": recorded_expected_count,
        "recorded_logged": recorded_logged_count,
    }


def benchmark_moe_layer_decode(
    model_runner,
    server_args,
    port_args,
    num_warmup,
    num_iterations,
    test_layer,
    rank_print,
    device,
    tp_rank,
    decode_test_cases,
    moe_layer,
    num_local_experts,
    simulated_ep_size,
    output_path=None,
    model_hidden_size=7168,
    model_inter_size=2048,
    model_total_experts=256,
):
    """Benchmark MoE layer in decode phase

    Args:
        num_local_experts: Number of experts on this GPU (= model's n_routed_experts or num_experts)
        simulated_ep_size: The EP size being simulated (= total_experts / num_local_experts)
        model_hidden_size: Model's hidden_size from config
        model_inter_size: Model's moe_intermediate_size from config
        model_total_experts: Total number of experts in the model (256 for DeepSeek-V3, 128 for Qwen3)
    """
    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()
    top_k = moe_layer.topk.topk_config.top_k

    logged_count = 0
    for case in decode_test_cases:
        try:
            num_token = case["num_tokens"]
            distributed = case["distributed"]
            power_law_alpha = case.get("power_law_alpha", 0.8) if distributed == "power_law" else None
            num_tokens_log = num_token * simulated_ep_size
            replay_samples = (
                _rank_local_replay_samples(
                    phase="generation",
                    table_num_tokens=num_tokens_log,
                    layer_id=test_layer,
                    ep_size=simulated_ep_size,
                    num_experts=model_total_experts,
                    output_path=output_path,
                )
                if distributed == "recorded"
                and _get_rank_local_replay_dir(output_path)
                else None
            )
            if replay_samples is not None:
                def _measure_recorded_decode_item_once() -> dict:
                    replay_stats = _benchmark_rank_local_decode_replay(
                        moe_layer=moe_layer,
                        replay_samples=replay_samples,
                        hidden_size=model_hidden_size,
                        device=device,
                        num_warmup=num_warmup,
                        num_iterations=num_iterations,
                    )
                    replay_stats = _apply_recorded_latency_policy(
                        replay_stats,
                        phase="generation",
                        num_tokens_log=num_tokens_log,
                    )
                    rank_print(
                        "Rank-local recorded replay (Decode): "
                        f"samples={len(replay_samples)}, ranks={simulated_ep_size}, "
                        f"steady mean/p90/max={replay_stats['latency']:.3f}/"
                        f"{replay_stats['latency_p90']:.3f}/"
                        f"{replay_stats['latency_max']:.3f}ms, "
                        f"cold={replay_stats['cold_latency']:.3f}ms"
                    )
                    item = {
                        "moe_dtype": "fp8_block",
                        "num_tokens": num_tokens_log,
                        "hidden_size": model_hidden_size,
                        "inter_size": model_inter_size,
                        "topk": top_k,
                        "num_experts": model_total_experts,
                        "moe_tp_size": 1,
                        "moe_ep_size": simulated_ep_size,
                        "distribution": _recorded_output_distribution(),
                        "workload_source": _rank_local_replay_source(),
                        "measurement_scope": "single_card_rank_local_replay",
                        **replay_stats,
                    }
                    return _apply_profile_free_hybrid_recorded_row(
                        item,
                        phase="generation",
                    )

                def _reset_recorded_decode_guard_attempt() -> None:
                    torch.get_device_module(device).synchronize()
                    model_runner.req_to_token_pool.clear()
                    model_runner.token_to_kv_pool_allocator.clear()
                    torch.cuda.empty_cache()

                item = select_recorded_source(
                    family="wideep_generation",
                    phase="generation",
                    ep=simulated_ep_size,
                    eplb="on" if _rank_local_replay_enable_eplb() else "off",
                    token=num_tokens_log,
                    measure_once=_measure_recorded_decode_item_once,
                    reset_before_attempt=_reset_recorded_decode_guard_attempt,
                    warmup_once=_measure_recorded_decode_item_once,
                    output_path=output_path,
                    rank_print=rank_print,
                    kernel_source="deepepmoe_rank_local_replay",
                )
                if tp_rank == 0:
                    collector_dir = os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))
                    )
                    perf_filename = (
                        os.path.join(collector_dir, "wideep_generation_moe_perf.txt")
                        if output_path is None
                        else os.path.join(output_path, "wideep_generation_moe_perf.txt")
                    )
                    log_perf(
                        item_list=[item],
                        framework="SGLang",
                        version=get_version("sglang"),
                        device_name=torch.cuda.get_device_name(server_args.device),
                        op_name="moe_generation",
                        kernel_source="deepepmoe_rank_local_replay",
                        perf_filename=perf_filename,
                    )
                    logged_count += 1
                torch.cuda.empty_cache()
                continue

            num_max_dispatch_tokens_per_rank = 128

            if num_token > num_max_dispatch_tokens_per_rank:
                print(
                    f"num_token {num_token} > num_max_dispatch_tokens_per_rank "
                    f"{num_max_dispatch_tokens_per_rank}, skipping"
                )
                continue

            hidden_size = model_runner.model.config.hidden_size

            if hidden_size % 128 != 0:
                pad_size = 128 - (hidden_size % 128)
                hidden_size += pad_size

            hidden_states = torch.randn(
                num_local_experts,
                num_max_dispatch_tokens_per_rank * simulated_ep_size,
                hidden_size,
                dtype=torch.bfloat16,
                device="cuda",
            )

            scale_hidden_size = hidden_size // 128
            scale_tensor = torch.ones(
                num_local_experts,
                num_max_dispatch_tokens_per_rank * simulated_ep_size,
                scale_hidden_size,
                device=hidden_states.device,
                dtype=torch.float32,
            )
            hidden_states_fp8_tensor = hidden_states.to(torch.float8_e4m3fn)

            masked_m = torch.zeros(num_local_experts, device=device, dtype=torch.int32)

            # support three distributed modes: power_law, uniform, and recorded
            if distributed == "power_law":
                masked_m_list = [
                    power_law_deepep_decode(
                        num_token * simulated_ep_size,
                        num_local_experts * simulated_ep_size,
                        top_k,
                        simulated_ep_size,
                        power_law_alpha,
                    )
                    .to(masked_m.dtype)
                    .to(torch.device(device))
                    for _ in range(5)
                ]
            elif distributed == "uniform":
                # Total experts = model_total_experts, simulated_ep_size = model_total_experts / num_local_experts
                base_tokens_per_expert = int(num_token * top_k) * simulated_ep_size // model_total_experts
                if base_tokens_per_expert == 0:
                    # Each expert that receives tokens gets exactly 1 token
                    # Number of experts with tokens on this card = total_calls / simulated_ep_size
                    # = (num_token * top_k * num_rank) / num_rank = num_token * top_k
                    masked_m[: int(num_token * top_k)] = 1
                else:
                    masked_m[:] = base_tokens_per_expert
                masked_m_list = [masked_m]
            elif distributed == "recorded":
                masked_m = _build_recorded_decode_masked_m(
                    num_tokens=num_token * simulated_ep_size,
                    topk=top_k,
                    num_local_experts=num_local_experts,
                    total_experts=model_total_experts,
                    output_path=output_path,
                    preferred_layer_id=test_layer,
                    preferred_recorder_ep_size=simulated_ep_size,
                    device=device,
                )
                active_local_experts = int((masked_m > 0).sum().item())
                rank_print(
                    "recorded decode distribution: "
                    f"global_tokens={num_token * simulated_ep_size}, "
                    f"active_local_experts={active_local_experts}, "
                    f"max_local_assignments={int(masked_m.max().item()) if masked_m.numel() else 0}"
                )
                masked_m_list = [masked_m]
            else:
                raise ValueError(f"Unsupported distributed mode: {distributed}")
            max_masked_m = int(torch.stack([mm.max() for mm in masked_m_list]).max().item())
            if max_masked_m > hidden_states.shape[1]:
                print(
                    f"  Skipping: max(masked_m_list) {max_masked_m} > hidden_states.shape[1] {hidden_states.shape[1]}"
                )
                continue
            scale_tensor = torch.ones(
                num_local_experts,
                num_max_dispatch_tokens_per_rank * simulated_ep_size,
                scale_hidden_size,
                device=hidden_states.device,
                dtype=torch.float32,
            )
            hidden_states_fp8_tensor = hidden_states.to(torch.float8_e4m3fn)

            topk_idx_empty = torch.empty(0, device=device, dtype=torch.int32)
            topk_weights_empty = torch.empty(0, device=device, dtype=torch.float32)

            torch.get_device_module(device).synchronize()
            torch.cuda.empty_cache()

            for _ in range(num_warmup):
                dispatch_output_list = []
                for masked_m in masked_m_list:
                    hidden_states_fp8_tensor_copy = hidden_states_fp8_tensor.clone()
                    scale_tensor_copy = scale_tensor.clone()

                    output = DeepEPLLDispatchOutput(
                        hidden_states=hidden_states_fp8_tensor_copy,
                        hidden_states_scale=scale_tensor_copy,
                        topk_ids=topk_idx_empty,
                        topk_weights=topk_weights_empty,
                        masked_m=masked_m,
                        expected_m=int(torch.ceil(masked_m.float().mean()).item()),
                    )
                    dispatch_output_list.append(output)

                for dispatch_output in dispatch_output_list:
                    _ = moe_layer.experts.run_moe_core(dispatch_output)

            torch.get_device_module(device).synchronize()
            torch.cuda.empty_cache()

            # Use benchmark_with_power for timing
            from helper import benchmark_with_power

            # Pre-compute expected_m values outside of kernel_func to avoid .item() during CUDA graph capture
            expected_m_list = [int(torch.ceil(masked_m_item.float().mean()).item()) for masked_m_item in masked_m_list]

            # Pre-clone masked_m tensors (they won't be disposed by run_moe_core)
            masked_m_clones = [m.clone() for m in masked_m_list]

            # Pre-create enough tensor copies to avoid clone() inside
            # kernel_func. run_moe_core disposes hidden_states and
            # hidden_states_scale via dispose_tensor(). benchmark_with_power
            # calls kernel_func three times before graph capture and once
            # during capture; graph replay does not consume new Python tensor
            # objects. Keep one extra call of headroom instead of the old
            # fixed 20-call pool, which consumed ~22.6 GiB for EP8 power-law.
            num_masked_m = len(masked_m_list)
            num_kernel_calls = 5
            num_tensor_sets = num_kernel_calls * num_masked_m

            bytes_per_tensor_set = (
                num_local_experts
                * num_max_dispatch_tokens_per_rank
                * simulated_ep_size
                * hidden_size
                + num_local_experts
                * num_max_dispatch_tokens_per_rank
                * simulated_ep_size
                * scale_hidden_size
                * 4
            )
            required_pool_bytes = bytes_per_tensor_set * num_tensor_sets
            free_bytes, _ = torch.cuda.mem_get_info(device)
            reserve_bytes = int(
                float(
                    os.environ.get(
                        "COLLECTOR_WIDEEP_MOE_DECODE_RESERVE_GIB",
                        "1.0",
                    )
                )
                * 2**30
            )
            if reserve_bytes < 0:
                raise ValueError(
                    "COLLECTOR_WIDEEP_MOE_DECODE_RESERVE_GIB must be >= 0"
                )
            if required_pool_bytes > free_bytes:
                rank_print(
                    "Skipping decode case because it cannot fit in currently "
                    "available GPU memory: tensor pool "
                    f"requires {required_pool_bytes / 2**30:.2f} GiB, "
                    f"but only {free_bytes / 2**30:.2f} GiB is free"
                )
                torch.cuda.empty_cache()
                continue
            remaining_bytes = free_bytes - required_pool_bytes
            if remaining_bytes < reserve_bytes:
                rank_print(
                    "Decode tensor-pool warning: allocation leaves only "
                    f"{remaining_bytes / 2**30:.2f} GiB free, below the "
                    f"requested runtime reserve of {reserve_bytes / 2**30:.2f} "
                    "GiB; trying the case anyway"
                )
            rank_print(
                "Decode tensor-pool guard: "
                f"allocating {required_pool_bytes / 2**30:.2f} GiB "
                f"from {free_bytes / 2**30:.2f} GiB free; allocation will "
                "be attempted and the case is skipped only on a real OOM"
            )

            hidden_states_copies = []
            scale_copies = []
            try:
                for _ in range(num_tensor_sets):
                    hidden_states_copies.append(
                        torch.randn(
                            num_local_experts,
                            num_max_dispatch_tokens_per_rank
                            * simulated_ep_size,
                            hidden_size,
                            dtype=torch.bfloat16,
                            device=device,
                        ).to(torch.float8_e4m3fn)
                    )
                    scale_copies.append(
                        torch.ones(
                            num_local_experts,
                            num_max_dispatch_tokens_per_rank
                            * simulated_ep_size,
                            scale_hidden_size,
                            device=device,
                            dtype=torch.float32,
                        )
                    )
            except torch.OutOfMemoryError:
                del hidden_states_copies
                del scale_copies
                torch.cuda.empty_cache()
                rank_print(
                    "Skipping decode case after a real CUDA OOM while "
                    f"allocating the {required_pool_bytes / 2**30:.2f} GiB "
                    "tensor pool"
                )
                continue

            # Use a mutable container to track tensor index across all run_moe_core calls
            tensor_idx = [0]

            def kernel_func():
                for masked_m_clone, expected_m_val in zip(masked_m_clones, expected_m_list, strict=True):
                    idx = tensor_idx[0] % num_tensor_sets
                    tensor_idx[0] += 1
                    dispatch_output = DeepEPLLDispatchOutput(
                        hidden_states=hidden_states_copies[idx],
                        hidden_states_scale=scale_copies[idx],
                        topk_ids=torch.empty(0, device=device, dtype=torch.int32),
                        topk_weights=torch.empty(0, device=device, dtype=torch.float32),
                        masked_m=masked_m_clone,
                        expected_m=expected_m_val,
                    )
                    _ = moe_layer.experts.run_moe_core(dispatch_output)

            with benchmark_with_power(
                device=device,
                kernel_func=kernel_func,
                num_warmups=3,
                num_runs=num_iterations,
                repeat_n=1,
            ) as results:
                pass

            avg_latency_ms = results["latency_ms"] / len(masked_m_list)
            power_stats = results["power_stats"]

            if tp_rank == 0:
                rank_print("DeepEP MoE GEMM Results (Decode) - CUDA Graph Enabled:")
                rank_print(f"  Average latency: {avg_latency_ms:.3f}ms")
            if tp_rank == 0:
                try:
                    moe_tp_size = 1
                    moe_ep_size = simulated_ep_size
                    device_name = torch.cuda.get_device_name(server_args.device)
                    version = get_version("sglang")
                    distribution_str = f"power_law_{power_law_alpha}" if distributed == "power_law" else distributed
                    # Save to collector/ directory to match non-wideep behavior
                    collector_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    perf_filename = (
                        os.path.join(collector_dir, "wideep_generation_moe_perf.txt")
                        if output_path is None
                        else os.path.join(output_path, "wideep_generation_moe_perf.txt")
                    )
                    log_perf(
                        item_list=[
                            {
                                "moe_dtype": "fp8_block",
                                "num_tokens": num_tokens_log,
                                "hidden_size": model_hidden_size,
                                "inter_size": model_inter_size,
                                "topk": top_k,
                                "num_experts": model_total_experts,
                                "moe_tp_size": moe_tp_size,
                                "moe_ep_size": moe_ep_size,
                                "distribution": distribution_str,
                                "latency": avg_latency_ms,
                            }
                        ],
                        framework="SGLang",
                        version=version,
                        device_name=device_name,
                        op_name="moe_generation",
                        kernel_source="deepepmoe",
                        perf_filename=perf_filename,
                        power_stats=power_stats,
                    )
                    logged_count += 1
                except Exception as e:
                    rank_print(f"  Warning: failed to log decode MoE metrics: {e}")
            del hidden_states, hidden_states_fp8_tensor, scale_tensor, dispatch_output_list
            torch.cuda.empty_cache()

        except Exception as e:
            rank_print(f"Decode case failed: {e}, skipping...")
            import traceback

            rank_print(traceback.format_exc())
            # Check if this is a CUDA error - if so, the context is corrupted and we should exit
            if "CUDA error" in str(e) or "illegal memory access" in str(e).lower():
                rank_print("CUDA error detected, exiting decode benchmark early to avoid cascading failures")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                break
            try:
                torch.cuda.empty_cache()
            except Exception:
                # If empty_cache fails, CUDA context is corrupted
                rank_print("CUDA context corrupted, exiting decode benchmark early")
                break
            continue
    return logged_count


def run_moe(
    server_args,
    port_args,
    num_warmup,
    num_iterations,
    test_layer,
    num_experts,
    tp_rank,
    output_path=None,
):
    """Run the complete MoE benchmark"""

    if get_bool_env_var("SGLANG_SET_CPU_AFFINITY"):
        set_gpu_proc_affinity(server_args.tp_size, server_args.nnodes, tp_rank)

    configure_logger(server_args, prefix=f" TP{tp_rank}")

    # Initialize MoE config in subprocess (required for DeepEP + DeepGEMM backend)
    _set_envs_and_config(server_args)
    initialize_moe_config(server_args)

    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    rank_print(f"\n{'=' * 60}")
    rank_print(f"Testing MoE Layer {test_layer}")
    rank_print(f"{'=' * 60}")

    try:
        rank_print(f"\n{'=' * 50}")
        rank_print(f"Testing with {num_experts} experts")
        rank_print(f"{'=' * 50}")

        # Get ORIGINAL model config BEFORE applying override
        # This is needed to get the true total_experts count
        original_json_override = server_args.json_model_override_args
        original_model_config = ModelConfig.from_server_args(server_args)
        original_hf_config = original_model_config.hf_config
        model_hidden_size = _get_config_value(original_hf_config, "hidden_size")
        if model_hidden_size is None:
            raise AttributeError(
                f"Could not find hidden size on hf_config "
                f"({type(original_hf_config).__name__}); tried hidden_size "
                "at top level and under text_config."
            )
        # Per-expert MLP intermediate size. The HF field name varies:
        #   moe_intermediate_size  - DeepSeek-V3, Qwen3-MoE, GPT-OSS
        #   intermediate_size      - MiniMax-M2 (Mixtral-style; no separate dense MLP)
        # Probe in priority order; raise if neither is present rather than silently
        # falling back to a wrong default (2048 was DeepSeek-shaped).
        model_inter_size = _get_config_value(original_hf_config, "moe_intermediate_size", "intermediate_size")
        if model_inter_size is None:
            raise AttributeError(
                f"Could not find MoE intermediate size on hf_config "
                f"({type(original_hf_config).__name__}); tried "
                "moe_intermediate_size / intermediate_size at top level and under text_config."
            )
        # Total expert count. The HF field name varies:
        #   n_routed_experts   - DeepSeek-V3
        #   num_experts        - Qwen3-MoE, GPT-OSS
        #   num_local_experts  - MiniMax-M2 (Mixtral-style)
        model_total_experts = _get_config_value(
            original_hf_config, "n_routed_experts", "num_experts", "num_local_experts"
        )
        if model_total_experts is None:
            raise AttributeError(
                f"Could not find expert count on hf_config "
                f"({type(original_hf_config).__name__}); tried "
                "n_routed_experts / num_experts / num_local_experts at top level and under text_config."
            )
        rank_print(
            f"Original model config: hidden_size={model_hidden_size}, "
            f"inter_size={model_inter_size}, total_experts={model_total_experts}"
        )

        # Now apply override to load model with reduced experts.
        # The HF expert-count field varies across model families; override
        # all known names so this works for DeepSeek / Qwen / MiniMax.
        original_model_path = server_args.model_path
        server_args.model_path = _resolve_reduced_moe_model_path(original_model_path, num_experts)
        num_layers = _get_requested_moe_num_layers()
        override_args = {
            "num_hidden_layers": num_layers,
            "n_routed_experts": num_experts,  # DeepSeek-V3
            "num_experts": num_experts,  # Qwen3-MoE, GPT-OSS
            "num_local_experts": num_experts,  # MiniMax-M2 (Mixtral-style)
        }
        if num_layers <= DEFAULT_MOE_TEST_LAYER:
            override_args["first_k_dense_replace"] = 0
        server_args.json_model_override_args = json.dumps(override_args)

        try:
            model_runner = load_model_with_dummy_weights(server_args, port_args, tp_rank)
        finally:
            server_args.model_path = original_model_path
            server_args.json_model_override_args = original_json_override

        # MoE submodule attribute name differs across sglang models:
        #   .mlp                 - DeepSeek-V2/V3, Qwen2/3-MoE, GPT-OSS
        #   .block_sparse_moe    - MiniMax-M2 (HF Mixtral-style), Mixtral
        model_layers = getattr(getattr(model_runner.model, "model", None), "layers", None)
        if model_layers is None and hasattr(model_runner.model, "language_model"):
            model_layers = getattr(getattr(model_runner.model.language_model, "model", None), "layers", None)
        if model_layers is None:
            raise AttributeError(
                f"Could not find decoder layers on {type(model_runner.model).__name__}; "
                "tried .model.layers and .language_model.model.layers."
            )
        decoder_layer = model_layers[test_layer]
        moe_layer = None
        for attr in ("mlp", "block_sparse_moe"):
            candidate = getattr(decoder_layer, attr, None)
            # Require an `experts` submodule whose `run_moe_core` is callable: this
            # is what the benchmark actually invokes below, and it filters out
            # non-MoE MLP layers (e.g. DeepSeek's leading dense layers).
            experts = getattr(candidate, "experts", None) if candidate is not None else None
            if experts is not None and callable(getattr(experts, "run_moe_core", None)):
                moe_layer = candidate
                break
        if moe_layer is None:
            raise AttributeError(
                f"Could not find MoE submodule on {type(decoder_layer).__name__}; "
                "tried .mlp / .block_sparse_moe. "
                "Add the attribute name used by this model to the probe list."
            )
        # Supports DeepSeek-V3 and Qwen3 MoE
        if hasattr(moe_layer, "config") and hasattr(moe_layer.config, "n_routed_experts"):
            # DeepSeek-V3 style
            actual_num_experts = moe_layer.config.n_routed_experts
        elif hasattr(moe_layer, "experts") and hasattr(moe_layer.experts, "num_experts"):
            # Qwen3 MoE style - from experts submodule
            actual_num_experts = moe_layer.experts.num_experts
        elif hasattr(moe_layer, "num_experts"):
            # Direct attribute (deepep mode)
            actual_num_experts = moe_layer.num_experts
        else:
            # Fall back to hf_config; probe the same three field names as the
            # pre-load probe at the top of this function, since the loaded
            # config has had all three overridden to the simulated count.
            hf_config = model_runner.model_config.hf_config
            actual_num_experts = _get_config_value(hf_config, "n_routed_experts", "num_experts", "num_local_experts")
            if actual_num_experts is None:
                raise AttributeError(
                    f"Could not determine expert count from {type(moe_layer).__name__} "
                    "or hf_config; tried .config.n_routed_experts / "
                    ".experts.num_experts / .num_experts on the MoE layer and "
                    "n_routed_experts / num_experts / num_local_experts on hf_config "
                    "at top level and under text_config."
                )

        rank_print(f"Loaded model with {actual_num_experts} local experts (simulating {model_total_experts} total)")

        # Calculate simulated EP size: total_experts / num_local_experts
        num_local_experts = actual_num_experts  # With ep_size=1, all experts are local
        simulated_ep_size = model_total_experts // num_local_experts
        rank_print(
            f"Simulating EP size: {simulated_ep_size} "
            f"(num_local_experts={num_local_experts}, total_experts={model_total_experts})"
        )

        if get_bool_env_var("COLLECTOR_WIDEEP_MOE_SKIP_PREFILL"):
            prefill_test_cases = []
            prefill_stats = {
                "total": 0,
                "recorded_expected": 0,
                "recorded_logged": 0,
            }
            rank_print("Skipping prefill configurations because COLLECTOR_WIDEEP_MOE_SKIP_PREFILL=1")
        else:
            prefill_test_cases = get_moe_prefill_test_cases(
                simulated_ep_size,
                output_path,
                topk=moe_layer.topk.topk_config.top_k,
                total_experts=model_total_experts,
            )
            rank_print(f"Testing {len(prefill_test_cases)} prefill configurations...")

            # Use deepep_mode="normal" for prefill
            server_args.deepep_mode = "normal"
            prefill_stats = benchmark_moe_layer_prefill(
                model_runner,
                server_args,
                port_args,
                num_warmup,
                num_iterations,
                test_layer,
                rank_print,
                server_args.device,
                tp_rank,
                prefill_test_cases,
                moe_layer,
                num_local_experts,
                simulated_ep_size,
                output_path,
                model_hidden_size=model_hidden_size,
                model_inter_size=model_inter_size,
                model_total_experts=model_total_experts,
            )

        if get_bool_env_var("COLLECTOR_WIDEEP_MOE_SKIP_DECODE"):
            decode_test_cases = []
            decode_logged_count = 0
            rank_print("Skipping decode configurations because COLLECTOR_WIDEEP_MOE_SKIP_DECODE=1")
        else:
            decode_test_cases = get_moe_decode_test_cases(
                output_path=output_path,
                simulated_ep_size=simulated_ep_size,
                topk=moe_layer.topk.topk_config.top_k,
                total_experts=model_total_experts,
            )
            rank_print(f"Testing {len(decode_test_cases)} decode configurations...")
            # Use deepep_mode="low_latency" for decode
            server_args.deepep_mode = "low_latency"
            decode_logged_count = benchmark_moe_layer_decode(
                model_runner,
                server_args,
                port_args,
                num_warmup,
                num_iterations,
                test_layer,
                rank_print,
                server_args.device,
                tp_rank,
                decode_test_cases,
                moe_layer,
                num_local_experts,
                simulated_ep_size,
                output_path=output_path,
                model_hidden_size=model_hidden_size,
                model_inter_size=model_inter_size,
                model_total_experts=model_total_experts,
            )
        if prefill_test_cases and int(prefill_stats["total"]) == 0:
            raise RuntimeError(
                "WideEP MoE prefill produced no perf rows; "
                "all prefill cases failed or were skipped"
            )
        if (
            int(prefill_stats["recorded_expected"]) > 0
            and int(prefill_stats["recorded_logged"]) == 0
        ):
            raise RuntimeError(
                "WideEP MoE prefill Recorded replay produced no perf rows; "
                "context Recorded rows are required when recorded replay cases exist"
            )
        if decode_test_cases and decode_logged_count == 0:
            raise RuntimeError(
                "WideEP MoE decode produced no perf rows; "
                "all decode cases failed or were skipped"
            )

        del model_runner, moe_layer
        torch.cuda.empty_cache()

    except Exception as e:
        rank_print(f"Error during MoE benchmark: {e}")
        import traceback

        rank_print(f"Traceback: {traceback.format_exc()}")
        raise

    torch.cuda.empty_cache()

    rank_print(f"\n{'=' * 60}")
    rank_print("BENCHMARK COMPLETED SUCCESSFULLY")
    rank_print(f"{'=' * 60}")


# ============================================================================
# Functions for collect.py framework (trtllm style: direct params, not index)
# ============================================================================


def get_wideep_moe_test_cases(total_experts=None):
    """Returns list of [num_experts] for MOE collection.

    Each num_experts value simulates a different EP size based on model's total experts.
    Defaults to production DeepSeek-style DeepEP coverage, EP=2/4/8.
    EP=1 is covered by the ordinary local MoE collector.

    For DeepSeek-V3 (256 experts):
    - num_experts=128 → EP=2
    - num_experts=64 → EP=4
    - num_experts=32 → EP=8
    - num_experts=16 → EP=16
    - num_experts=8 → EP=32
    - num_experts=4 → EP=64
    - num_experts=2 → EP=128
    - num_experts=1 → EP=256

    For Qwen3-235B (128 experts):
    - num_experts=64 → EP=2
    - num_experts=32 → EP=4
    - num_experts=16 → EP=8
    - num_experts=8 → EP=16
    - num_experts=4 → EP=32
    - num_experts=2 → EP=64
    - num_experts=1 → EP=128

    Formula: simulated_ep_size = total_experts / num_experts

    Args:
        total_experts: Total number of experts in the model (256 for DeepSeek-V3, 128 for Qwen3)
    """
    if total_experts is None:
        total_experts = _get_total_experts_for_selected_model()

    requested_ep_sizes = _env_int_list("COLLECTOR_WIDEEP_MOE_EP_SIZES")
    eplb_modes = _wideep_eplb_modes_for_cases()
    def valid_eplb_modes_for_ep(ep_size: int) -> list[bool]:
        # WideEP/DeepEP EPLB is only meaningful for EP>1.  Keep EP1 no-EPLB
        # available for compatibility, but skip EP1+EPLB explicitly.
        return [enable_eplb for enable_eplb in eplb_modes if not (int(ep_size) == 1 and enable_eplb)]

    if requested_ep_sizes:
        test_cases = []
        for ep_size in requested_ep_sizes:
            if ep_size <= 0:
                raise ValueError(f"COLLECTOR_WIDEEP_MOE_EP_SIZES entries must be positive, got {ep_size}")
            if total_experts % ep_size != 0:
                raise ValueError(
                    f"Cannot simulate EP={ep_size} for total_experts={total_experts}: "
                    "total experts must be divisible by EP size"
                )
            for enable_eplb in valid_eplb_modes_for_ep(ep_size):
                test_cases.append([total_experts // ep_size, enable_eplb])
        return test_cases

    default_ep_sizes = _default_ep_sizes_for_visible_devices(total_experts)
    return [
        [total_experts // ep_size, enable_eplb]
        for ep_size in default_ep_sizes
        if total_experts % ep_size == 0
        for enable_eplb in valid_eplb_modes_for_ep(ep_size)
    ]


def run_moe_benchmark(num_experts, enable_eplb=None, gpu_id=0, output_path=None):
    """Run MOE benchmark - called in subprocess with CUDA_VISIBLE_DEVICES set.

    This function contains all the initialization logic that must happen
    after CUDA_VISIBLE_DEVICES is set.

    Supports both DeepSeek-V3 and Qwen3 MoE models.
    """
    global _CURRENT_WIDEEP_ENABLE_EPLB_OVERRIDE
    if isinstance(enable_eplb, str):
        enable_eplb = enable_eplb.strip().lower() not in ("0", "false", "no", "off")
    _CURRENT_WIDEEP_ENABLE_EPLB_OVERRIDE = enable_eplb
    # In subprocess, always use cuda:0 since CUDA_VISIBLE_DEVICES isolates the GPU
    torch.cuda.set_device("cuda:0")

    original_model_path = _get_moe_model_path()
    model_path = _resolve_sglang_model_path(original_model_path)
    mem_fraction_static = _get_moe_mem_fraction_static()

    server_port = 30000 + gpu_id * 100
    server_args = ServerArgs(
        model_path=model_path,
        dtype="auto",
        device="cuda",
        load_format="dummy",
        tp_size=1,
        trust_remote_code=True,
        mem_fraction_static=mem_fraction_static,
        moe_a2a_backend="deepep",
        moe_runner_backend="deep_gemm",
        deepep_mode="auto",
        ep_size=1,
        node_rank=0,
        host="localhost",
        port=server_port,
        cuda_graph_max_bs=4,
        disable_cuda_graph=True,
    )

    logging.basicConfig(level=getattr(logging, server_args.log_level.upper()), format="%(message)s")
    _set_envs_and_config(server_args)

    # PortArgs.init_new() must be called in subprocess for proper isolation
    port_args = PortArgs.init_new(server_args)

    # Get total experts from model config to calculate simulated EP size
    model_config = ModelConfig.from_server_args(server_args)
    hf_config = model_config.hf_config
    total_experts = _get_config_value(hf_config, "n_routed_experts", "num_experts", "num_local_experts") or 256

    simulated_ep_size = total_experts // num_experts * server_args.ep_size
    print(f"\n{'=' * 60}")
    print(f"Original model path: {original_model_path}")
    print(f"Resolved model config path: {model_path}")
    print(
        f"MOE Benchmark: num_experts={num_experts}, EP_size={simulated_ep_size}, "
        f"total_experts={total_experts}, GPU={gpu_id}"
    )
    print(f"Using mem_fraction_static={server_args.mem_fraction_static}")
    print(f"{'=' * 60}")

    # Run the actual benchmark
    run_moe(
        server_args,
        port_args,
        3,
        10,
        _get_moe_test_layer(_get_requested_moe_num_layers()),
        num_experts,
        0,
        output_path,
    )

    torch.cuda.empty_cache()
    print(f"Completed num_experts={num_experts} (EP size {simulated_ep_size})")


def _run_moe_subprocess(num_experts, enable_eplb, gpu_id, output_path=None):
    """Helper to run MOE in subprocess with CUDA_VISIBLE_DEVICES isolation."""
    import subprocess
    import sys

    env = os.environ.copy()
    visible_device = resolve_subprocess_visible_device(gpu_id)
    env["CUDA_VISIBLE_DEVICES"] = visible_device
    max_retries = int(os.environ.get("COLLECTOR_WIDEEP_MOE_SUBPROCESS_RETRIES", "2"))

    code = f'''
import sys
sys.path.insert(0, "{THIS_DIR}")
sys.path.insert(0, "{COLLECTOR_ROOT}")
from collect_deepep_moe import run_moe_benchmark
run_moe_benchmark({num_experts}, {enable_eplb!r}, {gpu_id}, {output_path!r})
'''

    last_returncode = 0
    last_output = ""
    for attempt in range(max_retries + 1):
        proc = subprocess.Popen(
            [sys.executable, "-c", code],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=THIS_DIR,
        )

        print(
            "Starting MOE subprocess: "
            f"num_experts={num_experts}, enable_eplb={enable_eplb}, gpu_id={gpu_id}, "
            f"CUDA_VISIBLE_DEVICES={visible_device}, timeout={MOE_SUBPROCESS_TIMEOUT_SEC}s, "
            f"attempt={attempt + 1}/{max_retries + 1}"
        )

        start_time = time.monotonic()
        stdout = b""
        while True:
            elapsed = time.monotonic() - start_time
            remaining = MOE_SUBPROCESS_TIMEOUT_SEC - elapsed
            if remaining <= 0:
                proc.kill()
                stdout, _ = proc.communicate()
                print(
                    "MOE subprocess timed out "
                    f"after {int(elapsed)}s for num_experts={num_experts}, "
                    f"enable_eplb={enable_eplb}, gpu_id={gpu_id}, CUDA_VISIBLE_DEVICES={visible_device}"
                )
                break

            try:
                stdout, _ = proc.communicate(timeout=min(MOE_PROGRESS_LOG_INTERVAL_SEC, remaining))
                break
            except subprocess.TimeoutExpired:
                print(
                    "MOE subprocess still running: "
                    f"num_experts={num_experts}, enable_eplb={enable_eplb}, gpu_id={gpu_id}, "
                    f"CUDA_VISIBLE_DEVICES={visible_device}, "
                    f"elapsed={int(time.monotonic() - start_time)}s/{MOE_SUBPROCESS_TIMEOUT_SEC}s"
                )

        last_returncode = proc.returncode
        last_output = stdout.decode("utf-8", errors="replace") if stdout else ""
        if last_output:
            print(last_output)

        if proc.returncode == 0:
            return

        retryable = (
            "illegal memory access" in last_output.lower()
            or "DGException" in last_output
            or proc.returncode in (-6, -11)
        )
        if attempt < max_retries and retryable:
            print(
                "MOE subprocess failed with retryable CUDA/DeepGEMM error; "
                f"retrying after GPU isolation reset ({attempt + 1}/{max_retries})"
            )
            time.sleep(5.0)
            continue
        break

    raise RuntimeError(f"MOE subprocess failed with exit code {last_returncode}")


def run_wideep_moe(num_experts, enable_eplb=None, *, perf_filename, device="cuda:0"):
    """Run wideep DeepEP MOE benchmark.

    Compatible with collect.py framework - uses subprocess for GPU isolation.
    Supports both DeepSeek-V3 (256 experts) and Qwen3 (128 experts) models.
    """
    device_str = str(device) if not isinstance(device, str) else device
    gpu_id = int(device_str.split(":")[-1]) if ":" in device_str else 0

    print("\n" + "=" * 60)
    print(f"MOE: num_experts={num_experts}, enable_eplb={enable_eplb}, GPU={gpu_id}")
    print("=" * 60)

    # collect.py resolves perf_filename into the active run directory. Use that
    # directory so WideEP's split outputs land next to the other collector
    # artifacts instead of the caller's current working directory.
    output_path = os.path.dirname(os.path.abspath(str(perf_filename))) or os.getcwd()
    _run_moe_subprocess(num_experts, enable_eplb, gpu_id, output_path)


if __name__ == "__main__":
    import argparse

    from registry_types import PerfFile

    parser = argparse.ArgumentParser(description="SGLang Wideep DeepEP MOE Benchmark")
    parser.add_argument("--output-path", default=None, help="Output directory for perf files")
    args = parser.parse_args()

    print(f"Model path: {_get_moe_model_path()}")

    # Run all MOE test cases
    perf_filename = PerfFile.WIDEEP_MOE
    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
        perf_filename = os.path.join(args.output_path, str(PerfFile.WIDEEP_MOE))
    for test_case in get_wideep_moe_test_cases():
        run_wideep_moe(*test_case, perf_filename=perf_filename)

    print("\n" + "=" * 60)
    print("SCRIPT COMPLETED SUCCESSFULLY")
    print("=" * 60)
