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
import sys
import tempfile
import time

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

MOE_SUBPROCESS_TIMEOUT_SEC = 1800
MOE_PROGRESS_LOG_INTERVAL_SEC = 60
DEFAULT_MOE_MEM_FRACTION_STATIC = 0.3
DEFAULT_MOE_TEST_LAYER = 3
DEFAULT_MOE_COLLECTION_LAYERS = DEFAULT_MOE_TEST_LAYER + 1


def _env_int_list(name: str) -> list[int] | None:
    raw_value = os.environ.get(name)
    if not raw_value:
        return None
    values = []
    for item in raw_value.replace(",", " ").split():
        values.append(int(item))
    return values


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


def _get_recorded_distribution_name() -> str:
    return os.environ.get("COLLECTOR_WIDEEP_MOE_RECORDED_DISTRIBUTION", "recorded")


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
    try:
        _resolve_recorded_distribution_file(output_path)
    except FileNotFoundError:
        return False
    return True


@functools.cache
def _load_recorded_distribution_rows(path: str, distribution: str) -> tuple[dict, ...]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            if row.get("distribution") != distribution:
                continue
            if row.get("phase", "context") != "context":
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"No distribution={distribution!r} context rows found in {path}")
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
) -> list[int]:
    path = _resolve_recorded_distribution_file(output_path)
    rows = _load_recorded_distribution_rows(path, distribution)
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
    num_tokens: int,
    topk: int,
    total_experts: int,
    preferred_recorder_ep_size: int | None,
) -> bool:
    try:
        _select_recorded_expert_counts(
            output_path=output_path,
            distribution=_get_recorded_distribution_name(),
            num_tokens=num_tokens,
            topk=topk,
            total_experts=total_experts,
            preferred_layer_id=None,
            preferred_recorder_ep_size=preferred_recorder_ep_size,
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


def get_moe_prefill_test_cases(rank, output_path: str | None = None):
    """Get test cases for MoE prefill phase including distribution and alpha.

    Returns a list of dicts with keys: 'num_tokens', 'distributed', 'power_law_alpha'.
    For uniform distribution, 'power_law_alpha' is None.
    """
    test_cases = []
    requested_logged_tokens = _env_int_list("COLLECTOR_WIDEEP_MOE_PREFILL_TOKENS")
    if requested_logged_tokens:
        num_tokens = sorted({max(1, token // rank) for token in requested_logged_tokens})
    elif os.environ.get("COLLECTOR_SMOKE") == "1":
        num_tokens = sorted({max(1, 128 // rank)})
    else:
        # Default to production calibration points expressed as global/logged
        # token counts, then convert to this simulated EP rank's local count.
        num_tokens = sorted({max(1, token // rank) for token in (128, 512, 2048, 4096)})
    requested_distributions = _env_str_set("COLLECTOR_WIDEEP_MOE_DISTRIBUTIONS")
    if requested_distributions is None:
        requested_distributions = (
            {"uniform", "recorded"} if _has_recorded_distribution_file(output_path) else {"uniform"}
        )
    power_law_alphas = _env_float_list("COLLECTOR_WIDEEP_MOE_POWER_LAW_ALPHAS") or [0.6, 0.8, 1.01, 1.02, 1.2]

    for num_token in sorted(num_tokens):
        if num_token * 8 < 128:
            continue
        if num_token * rank > 256 * 2048:
            continue
        # Uniform
        if "uniform" in requested_distributions:
            test_cases.append({"num_tokens": num_token, "distributed": "uniform", "power_law_alpha": None})
        # Power-law variants
        if "power_law" in requested_distributions:
            for alpha in power_law_alphas:
                test_cases.append(
                    {
                        "num_tokens": num_token,
                        "distributed": "power_law",
                        "power_law_alpha": alpha,
                    }
                )
        include_recorded = (
            "recorded" in requested_distributions
            and _has_recorded_distribution_file(output_path)
        )
        if include_recorded:
            test_cases.append({"num_tokens": num_token, "distributed": "recorded", "power_law_alpha": None})

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
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    requested_logged_tokens = _env_int_list("COLLECTOR_WIDEEP_MOE_DECODE_TOKENS")
    if requested_logged_tokens:
        batch_sizes = sorted(set(requested_logged_tokens))
    requested_distributions = _env_str_set("COLLECTOR_WIDEEP_MOE_DISTRIBUTIONS")
    if requested_distributions is None:
        requested_distributions = (
            {"uniform", "recorded"} if _has_recorded_distribution_file(output_path) else {"uniform"}
        )
    power_law_alphas = _env_float_list("COLLECTOR_WIDEEP_MOE_POWER_LAW_ALPHAS") or [0.6, 0.8, 1.01, 1.02, 1.2]
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
                    topk_idx_sample = topk_idx_sample.remainder(num_local_experts)
                    num_recv = torch.bincount(
                        topk_idx_sample.reshape(-1),
                        minlength=num_local_experts,
                    ).to(torch.int32).tolist()
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
                    dispatch_output = DeepEPNormalDispatchOutput(
                        hidden_states=hidden_states_fp8_tensor_iter,
                        hidden_states_scale=scale_tensor_iter,
                        topk_ids=topk_idx_sample.clone(),
                        topk_weights=topk_weights_sample.clone(),
                        num_recv_tokens_per_expert=num_recv_sample,
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
                    dispatch_output = DeepEPNormalDispatchOutput(
                        hidden_states=hidden_states_fp8_tensor_iter,
                        hidden_states_scale=scale_tensor_iter,
                        topk_ids=topk_idx_sample.clone(),
                        topk_weights=topk_weights_sample.clone(),
                        num_recv_tokens_per_expert=num_recv_sample,
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
                    num_tokens_log = num_token * simulated_ep_size
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
    return logged_count


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

            # Pre-create enough tensor copies to avoid clone() inside kernel_func
            # run_moe_core disposes hidden_states and hidden_states_scale via dispose_tensor()
            # Estimate: kernel_func called ~4 times (warmup 3 + capture 1) in graph mode
            # Each call iterates len(masked_m_list) times (max 5 for power_law)
            # Total: 4 * 5 = 20 tensor sets needed, use 50 for safety
            num_masked_m = len(masked_m_list)
            num_kernel_calls = 20  # Conservative estimate for kernel_func invocations
            num_tensor_sets = num_kernel_calls * num_masked_m

            hidden_states_copies = []
            scale_copies = []
            for _ in range(num_tensor_sets):
                hidden_states_copies.append(
                    torch.randn(
                        num_local_experts,
                        num_max_dispatch_tokens_per_rank * simulated_ep_size,
                        hidden_size,
                        dtype=torch.bfloat16,
                        device=device,
                    ).to(torch.float8_e4m3fn)
                )
                scale_copies.append(
                    torch.ones(
                        num_local_experts,
                        num_max_dispatch_tokens_per_rank * simulated_ep_size,
                        scale_hidden_size,
                        device=device,
                        dtype=torch.float32,
                    )
                )

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
                    num_tokens_log = num_token * simulated_ep_size
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

        prefill_test_cases = get_moe_prefill_test_cases(simulated_ep_size, output_path)
        rank_print(f"Testing {len(prefill_test_cases)} prefill configurations...")

        # Use deepep_mode="normal" for prefill
        server_args.deepep_mode = "normal"
        prefill_logged_count = benchmark_moe_layer_prefill(
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
        if prefill_test_cases and prefill_logged_count == 0:
            raise RuntimeError(
                "WideEP MoE prefill produced no perf rows; "
                "all prefill cases failed or were skipped"
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
            test_cases.append([total_experts // ep_size])
        return test_cases

    # EP=1 is ordinary local MoE rather than an expert-parallel DeepEP case.
    # On current H20/SGLang, forcing EP=1 through DeepEP can corrupt the CUDA
    # context for larger prefill points. Keep it opt-in via
    # COLLECTOR_WIDEEP_MOE_EP_SIZES=1 for debugging, but do not run it by default.
    default_ep_sizes = [2, 4, 8]
    return [[total_experts // ep_size] for ep_size in default_ep_sizes if total_experts % ep_size == 0]


def run_moe_benchmark(num_experts, gpu_id, output_path=None):
    """Run MOE benchmark - called in subprocess with CUDA_VISIBLE_DEVICES set.

    This function contains all the initialization logic that must happen
    after CUDA_VISIBLE_DEVICES is set.

    Supports both DeepSeek-V3 and Qwen3 MoE models.
    """
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


def _run_moe_subprocess(num_experts, gpu_id, output_path=None):
    """Helper to run MOE in subprocess with CUDA_VISIBLE_DEVICES isolation."""
    import subprocess
    import sys

    env = os.environ.copy()
    visible_device = resolve_subprocess_visible_device(gpu_id)
    env["CUDA_VISIBLE_DEVICES"] = visible_device

    code = f'''
import sys
sys.path.insert(0, "{THIS_DIR}")
sys.path.insert(0, "{COLLECTOR_ROOT}")
from collect_deepep_moe import run_moe_benchmark
run_moe_benchmark({num_experts}, {gpu_id}, {output_path!r})
'''

    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=THIS_DIR,
    )

    print(
        "Starting MOE subprocess: "
        f"num_experts={num_experts}, gpu_id={gpu_id}, "
        f"CUDA_VISIBLE_DEVICES={visible_device}, timeout={MOE_SUBPROCESS_TIMEOUT_SEC}s"
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
                f"gpu_id={gpu_id}, CUDA_VISIBLE_DEVICES={visible_device}"
            )
            break

        try:
            stdout, _ = proc.communicate(timeout=min(MOE_PROGRESS_LOG_INTERVAL_SEC, remaining))
            break
        except subprocess.TimeoutExpired:
            print(
                "MOE subprocess still running: "
                f"num_experts={num_experts}, gpu_id={gpu_id}, "
                f"CUDA_VISIBLE_DEVICES={visible_device}, "
                f"elapsed={int(time.monotonic() - start_time)}s/{MOE_SUBPROCESS_TIMEOUT_SEC}s"
            )

    if stdout:
        print(stdout.decode("utf-8", errors="replace"))

    if proc.returncode != 0:
        raise RuntimeError(f"MOE subprocess failed with exit code {proc.returncode}")


def run_wideep_moe(num_experts, *, perf_filename, device="cuda:0"):
    """Run wideep DeepEP MOE benchmark.

    Compatible with collect.py framework - uses subprocess for GPU isolation.
    Supports both DeepSeek-V3 (256 experts) and Qwen3 (128 experts) models.
    """
    device_str = str(device) if not isinstance(device, str) else device
    gpu_id = int(device_str.split(":")[-1]) if ":" in device_str else 0

    print("\n" + "=" * 60)
    print(f"MOE: num_experts={num_experts}, GPU={gpu_id}")
    print("=" * 60)

    # collect.py resolves perf_filename into the active run directory. Use that
    # directory so WideEP's split outputs land next to the other collector
    # artifacts instead of the caller's current working directory.
    output_path = os.path.dirname(os.path.abspath(str(perf_filename))) or os.getcwd()
    _run_moe_subprocess(num_experts, gpu_id, output_path)


if __name__ == "__main__":
    import argparse

    from registry_types import PerfFile

    parser = argparse.ArgumentParser(description="SGLang Wideep DeepEP MOE Benchmark")
    parser.add_argument("--output-path", default=None, help="Output directory for perf files")
    args = parser.parse_args()

    print(f"Model path: {_get_moe_model_path()}")

    # Run all MOE test cases
    for test_case in get_wideep_moe_test_cases():
        run_wideep_moe(*test_case, perf_filename=PerfFile.WIDEEP_MOE)

    print("\n" + "=" * 60)
    print("SCRIPT COMPLETED SUCCESSFULLY")
    print("=" * 60)
