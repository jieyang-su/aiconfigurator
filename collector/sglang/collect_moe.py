# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang MoE collector.

Benchmarks SGLang fused MoE kernels across BF16, FP8 block, NVFP4, and INT4
paths when supported. Shared MoE model/sweep cases come from YAML; this module
owns SGLang kernel compatibility, server-args mocking, routing-logit synthesis,
rank-local workload construction, quantized weight setup, and perf logging.
"""

import inspect
import itertools
import csv
import json
import math
import os
from pathlib import Path
from typing import TypedDict
from unittest.mock import MagicMock

import pkg_resources

# Mock global server args before importing MOE modules (required by SGLang 0.5.5+)
# The fused_moe_triton_config module now requires get_global_server_args() to be set
import sglang.srt.server_args as _server_args_module
import torch

if _server_args_module._global_server_args is None:
    _mock_server_args = MagicMock()
    _mock_server_args.enable_deterministic_inference = False
    _mock_server_args.enable_fused_moe_sum_all_reduce = (
        False  # sglang >=0.5.10; prevents fused all-reduce in single-GPU benchmarks
    )
    _mock_server_args.kt_weight_path = None
    _mock_server_args.flashinfer_mxfp4_moe_precision = "default"
    _server_args_module._global_server_args = _mock_server_args

import sglang.srt.layers.moe.fused_moe_triton.layer as _moe_layer_mod
import sglang.srt.layers.moe.token_dispatcher.standard as _std_dispatch_mod
import sglang.srt.layers.moe.utils as _moe_utils

try:
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
        get_config_dtype_str,
        get_default_config,
        get_moe_configs,
    )
except ImportError:
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_config import (
        get_config_dtype_str,
        get_default_config,
        get_moe_configs,
    )
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.topk import BypassedTopKOutput, StandardTopKOutput, TopKConfig, select_experts
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.utils import is_hip

from collector.wideep.sglang.rank_local_moe_replay import (
    MANIFEST_FILENAME as RANK_LOCAL_REPLAY_MANIFEST,
    select_replay_workloads,
)

try:
    import sglang.srt.layers.quantization.mxfp4 as _mxfp4_mod
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.quantization.mxfp4 import Mxfp4Config

    _HAS_SGLANG_MXFP4 = True
except ImportError:
    _HAS_SGLANG_MXFP4 = False

# sglang >=0.5.10: fused_experts_impl uses @torch.compile on moe_sum_reduce_torch_compile
# for tokens_in_chunk <= 32 and topk > 2.  torch.compile's JIT compilation can hang
# during CUDA graph capture or in headless benchmark contexts.  Replace the compiled
# function with an eager equivalent so benchmarks don't stall.
try:
    try:
        import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe as _fmoe_mod
    except ImportError:
        import sglang.srt.layers.moe.fused_moe_triton.fused_moe as _fmoe_mod

    def _eager_moe_sum_reduce(x, out, routed_scaling_factor):
        torch.sum(x, dim=1, out=out)
        out.mul_(routed_scaling_factor)

    if hasattr(_fmoe_mod, "moe_sum_reduce_torch_compile"):
        _fmoe_mod.moe_sum_reduce_torch_compile = _eager_moe_sum_reduce
except Exception:
    pass

try:
    from sglang.srt.layers.moe.flashinfer_cutedsl_moe import (
        flashinfer_cutedsl_moe_masked,
    )

    HAS_FLASHINFER_CUTE = True
except ImportError:
    HAS_FLASHINFER_CUTE = False

try:
    from sglang.jit_kernel.nvfp4 import scaled_fp4_quant as _scaled_fp4_quant

    _HAS_SCALED_FP4_QUANT = True
except ImportError:
    _HAS_SCALED_FP4_QUANT = False

# Marlin int4 MoE kernel (W4A16) — much faster than the Triton GPTQ/AWQ path.
_HAS_MARLIN_MOE = False
try:
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
    from sglang.srt.layers.quantization.gptq import gptq_marlin_moe_repack
    from sglang.srt.layers.quantization.marlin_utils import marlin_moe_permute_scales

    _HAS_MARLIN_MOE = True
except ImportError:
    pass

try:
    from case_generator import get_common_moe_test_cases, moe_model_allows_quantization

    from helper import (
        balanced_logits,
        benchmark_with_power,
        build_rank0_local_workload,
        get_sm_version,
        log_perf,
        power_law_logits_v3,
    )
except ModuleNotFoundError:
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from case_generator import get_common_moe_test_cases, moe_model_allows_quantization

    from helper import (
        balanced_logits,
        benchmark_with_power,
        build_rank0_local_workload,
        get_sm_version,
        log_perf,
        power_law_logits_v3,
    )


_is_hip = is_hip()
_MOE_RUNNER_CONFIG_PARAMS = set(inspect.signature(MoeRunnerConfig).parameters)
_NON_GATED_MOE_MODEL_PATTERNS = ("Nemotron-3", "nemotron-ultra", "Nemotron-H")
_SM120_NEMOTRON_NVFP4_MODELS = {
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
}


def _env_str_set(name: str) -> set[str] | None:
    raw_value = os.environ.get(name)
    if not raw_value:
        return None
    return {item.strip() for item in raw_value.replace(",", " ").split() if item.strip()}


def _env_int_set(name: str) -> set[int] | None:
    raw_value = os.environ.get(name)
    if not raw_value:
        return None
    return {int(item.strip()) for item in raw_value.replace(",", " ").split() if item.strip()}


def _rank_local_replay_phases() -> list[str]:
    raw_phases = _env_str_set("COLLECTOR_MOE_RANK_LOCAL_REPLAY_PHASES")
    if not raw_phases:
        return ["context"]
    phases = [phase for phase in ("context", "generation") if phase in raw_phases]
    invalid = raw_phases.difference(phases)
    if invalid:
        raise ValueError(
            "COLLECTOR_MOE_RANK_LOCAL_REPLAY_PHASES only supports "
            f"context/generation, got {sorted(invalid)}"
        )
    return phases


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _use_cuda_graph_for_phase(phase: str) -> bool:
    if phase == "context":
        return _env_bool("COLLECTOR_MOE_CONTEXT_USE_CUDA_GRAPH", False)
    if phase == "generation":
        return _env_bool("COLLECTOR_MOE_GENERATION_USE_CUDA_GRAPH", True)
    return _env_bool("COLLECTOR_MOE_USE_CUDA_GRAPH", True)


def _get_recorded_distribution_name() -> str:
    return os.environ.get("COLLECTOR_MOE_RECORDED_DISTRIBUTION", "recorded")


def _public_recorded_distribution(enable_eplb: bool) -> str:
    return "recorded_eplb" if enable_eplb else "recorded_no_eplb"


def _get_rank_local_replay_dir() -> Path | None:
    explicit = os.environ.get("COLLECTOR_MOE_RANK_LOCAL_REPLAY_DIR")
    if explicit:
        path = Path(explicit)
    else:
        output_dir = os.environ.get("COLLECTOR_CURRENT_OUTPUT_DIR")
        if not output_dir:
            return None
        path = Path(output_dir) / "moe_token_distribution_replay"
    if not (path / RANK_LOCAL_REPLAY_MANIFEST).is_file():
        return None
    return path


def _rank_local_replay_distributions(
    *,
    num_tokens: int,
    ep_size: int,
    num_experts: int,
) -> list[str]:
    replay_dir = _get_rank_local_replay_dir()
    if replay_dir is None:
        return []
    phases = _rank_local_replay_phases()
    with (replay_dir / RANK_LOCAL_REPLAY_MANIFEST).open(
        newline="",
        encoding="utf-8",
    ) as f:
        rows = list(csv.DictReader(f))
    specs = {
        (
            row.get("phase", "context"),
            row.get("workload_source", "runtime"),
            str(row.get("enable_eplb", "")).lower() in ("1", "true"),
        )
        for row in rows
        if row.get("phase") in phases
        and int(row.get("num_tokens", -1)) == int(num_tokens)
        and int(row.get("requested_ep_size", -1)) == int(ep_size)
        and int(row.get("num_experts", -1)) == int(num_experts)
    }
    distributions = []
    for phase, source, enable_eplb in sorted(specs):
        suffix = "rank_local_eplb" if enable_eplb else "rank_local_no_eplb"
        if phases == ["context"]:
            distributions.append(f"recorded_{source}_{suffix}")
        else:
            distributions.append(f"recorded_{source}_{phase}_{suffix}")
    return distributions


def _rank_local_replay_tokens(
    *,
    ep_size: int,
    num_experts: int,
) -> set[int]:
    replay_dir = _get_rank_local_replay_dir()
    if replay_dir is None:
        return set()
    phases = _rank_local_replay_phases()
    with (replay_dir / RANK_LOCAL_REPLAY_MANIFEST).open(
        newline="",
        encoding="utf-8",
    ) as f:
        return {
            int(row.get("num_tokens", -1))
            for row in csv.DictReader(f)
            if row.get("phase") in phases
            and int(row.get("requested_ep_size", -1)) == int(ep_size)
            and int(row.get("num_experts", -1)) == int(num_experts)
        }


def _parse_rank_local_distribution(distributed: str) -> tuple[str, str, bool] | None:
    prefix = "recorded_"
    if not distributed.startswith(prefix):
        return None
    value = distributed[len(prefix) :]
    phase = "context"
    if value.endswith("_rank_local_no_eplb"):
        source = value.removesuffix("_rank_local_no_eplb")
        for candidate_phase in ("context", "generation"):
            marker = f"_{candidate_phase}"
            if source.endswith(marker):
                return source.removesuffix(marker), candidate_phase, False
        return source, phase, False
    if value.endswith("_rank_local_eplb"):
        source = value.removesuffix("_rank_local_eplb")
        for candidate_phase in ("context", "generation"):
            marker = f"_{candidate_phase}"
            if source.endswith(marker):
                return source.removesuffix(marker), candidate_phase, True
        return source, phase, True

    # Backward compatibility for rows produced before the descriptive naming
    # convention was introduced.
    marker = "_rank_local_eplb"
    if marker in value:
        source, raw_eplb = value.rsplit(marker, 1)
        if raw_eplb in ("0", "1"):
            return source, phase, raw_eplb == "1"
    return None


def _public_distribution_for_output(
    distributed: str,
    power_law_alpha: float | None,
) -> tuple[str, str]:
    replay_spec = _parse_rank_local_distribution(distributed)
    if replay_spec is None:
        return (
            "power_law_" + str(power_law_alpha) if distributed == "power_law" else distributed,
            "",
        )
    _, phase, enable_eplb = replay_spec
    return _public_recorded_distribution(enable_eplb), phase


def _resolve_recorded_distribution_file() -> str:
    for env_name in (
        "COLLECTOR_MOE_RECORDED_DISTRIBUTION_FILE",
        "COLLECTOR_MOE_TOKEN_DISTRIBUTION_FILE",
    ):
        raw_value = os.environ.get(env_name)
        if raw_value:
            if not os.path.exists(raw_value):
                raise FileNotFoundError(f"{env_name} points to missing file: {raw_value}")
            return raw_value

    output_dir = os.environ.get("COLLECTOR_CURRENT_OUTPUT_DIR")
    if output_dir:
        candidate = os.path.join(output_dir, "moe_token_distribution_perf.txt")
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "recorded MoE distribution requested, but no moe_token_distribution_perf.txt was found. "
        "Set COLLECTOR_MOE_RECORDED_DISTRIBUTION_FILE or run moe_token_distribution first "
        "in the same collector output directory."
    )


def _has_recorded_distribution_file() -> bool:
    try:
        _resolve_recorded_distribution_file()
    except FileNotFoundError:
        return False
    return True


def _load_recorded_distribution_rows(path: str, distribution: str) -> tuple[dict, ...]:
    cache_key = (path, distribution)
    cache = getattr(_load_recorded_distribution_rows, "_cache", {})
    if cache_key in cache:
        return cache[cache_key]
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
    result = tuple(rows)
    cache[cache_key] = result
    _load_recorded_distribution_rows._cache = cache
    return result


def _recorded_distribution_has_case(num_tokens: int, topk: int, num_experts: int) -> bool:
    try:
        _select_recorded_expert_counts(
            distribution=_get_recorded_distribution_name(),
            num_tokens=num_tokens,
            topk=topk,
            num_experts=num_experts,
            preferred_layer_id=None,
            preferred_recorder_ep_size=None,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return False
    return True


def _parse_recorded_expert_counts(row: dict, num_experts: int) -> list[float]:
    raw_counts = row.get("expert_assignments_json") or ""
    if not raw_counts:
        raise ValueError(
            "recorded MoE distribution row has empty expert_assignments_json. "
            "Re-run moe_token_distribution with the updated collector so the exact expert counts are saved."
        )
    counts = json.loads(raw_counts)
    if len(counts) != num_experts:
        raise ValueError(f"recorded expert count length mismatch: got {len(counts)}, expected {num_experts}")
    return [max(0.0, float(value)) for value in counts]


def _select_recorded_expert_counts(
    *,
    distribution: str,
    num_tokens: int,
    topk: int,
    num_experts: int,
    preferred_layer_id: int | None,
    preferred_recorder_ep_size: int | None,
) -> list[float]:
    path = _resolve_recorded_distribution_file()
    candidates = []
    for row in _load_recorded_distribution_rows(path, distribution):
        try:
            if int(float(row.get("num_tokens", 0))) != int(num_tokens):
                continue
            if int(float(row.get("topk", topk))) != int(topk):
                continue
            if int(float(row.get("num_experts", num_experts))) != int(num_experts):
                continue
        except ValueError:
            continue
        candidates.append(row)
    if not candidates:
        raise ValueError(
            f"No recorded MoE distribution row for num_tokens={num_tokens}, topk={topk}, "
            f"num_experts={num_experts}"
        )

    def score(row: dict) -> tuple[int, int, float, int]:
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
        return (ep_match, layer_match, total_assignments, layer_id)

    selected = max(candidates, key=score)
    counts = _parse_recorded_expert_counts(selected, num_experts)
    if sum(counts) <= 0:
        raise ValueError(f"Recorded MoE distribution row for num_tokens={num_tokens} has zero assignments")
    return counts


def recorded_logits_v3(
    num_tokens: int,
    num_experts: int,
    topk: int,
    ep: int,
    return_rank0_info: bool = False,
):
    import torch.nn.functional as F

    counts = _select_recorded_expert_counts(
        distribution=_get_recorded_distribution_name(),
        num_tokens=num_tokens,
        topk=topk,
        num_experts=num_experts,
        preferred_layer_id=None,
        preferred_recorder_ep_size=ep,
    )
    probabilities = torch.tensor(counts, dtype=torch.float32, device="cpu")
    if torch.count_nonzero(probabilities).item() < topk:
        raise ValueError(
            f"Recorded distribution has fewer active experts than topk: "
            f"active={torch.count_nonzero(probabilities).item()}, topk={topk}"
        )
    probabilities = probabilities / probabilities.sum()

    seed = int(os.environ.get("COLLECTOR_MOE_RECORDED_ROUTING_SEED", "20260615"))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + num_tokens + topk + num_experts + ep)
    selected_experts = torch.stack(
        [
            torch.multinomial(probabilities, num_samples=topk, replacement=False, generator=generator)
            for _ in range(num_tokens)
        ]
    )
    expert_map = F.one_hot(selected_experts.long(), num_classes=num_experts).sum(1)
    router_logits = F.softmax(expert_map.bfloat16(), dim=1)

    if return_rank0_info:
        experts_per_rank = num_experts // ep
        rank0_selections_mask = selected_experts < experts_per_rank
        rank0_token_mask = rank0_selections_mask.any(dim=1)
        rank0_logits = router_logits[rank0_token_mask]
        rank0_selected_slots = selected_experts[rank0_token_mask]
        rank0_info = {
            "rank0_token_mask": rank0_token_mask,
            "rank0_logits": rank0_logits,
            "rank0_selected_slots": rank0_selected_slots,
            "rank0_num_tokens": int(rank0_logits.shape[0]),
            "slots_per_rank": experts_per_rank,
            "rank0_total_selections": int(rank0_selections_mask.sum().item()),
        }
        return router_logits, rank0_info
    return router_logits


def _make_moe_runner_config(swiglu_limit: float | None = None) -> MoeRunnerConfig:
    kwargs = {}
    if "swiglu_limit" in _MOE_RUNNER_CONFIG_PARAMS:
        kwargs["swiglu_limit"] = swiglu_limit
    elif "gemm1_clamp_limit" in _MOE_RUNNER_CONFIG_PARAMS:
        kwargs["gemm1_clamp_limit"] = swiglu_limit
    return MoeRunnerConfig(**kwargs)


def _uses_relu2_moe_activation(model_name: str) -> bool:
    return any(pattern in model_name for pattern in _NON_GATED_MOE_MODEL_PATTERNS)


def get_moe_test_cases():
    # fp8_block MOE requires SM90+ due to shared memory requirements
    # L40S (SM89) has 100KB shared memory, fp8_block kernel needs ~144KB
    sm_version = get_sm_version()
    if sm_version < 90:
        moe_list = ["bfloat16", "int4_wo"]
    elif sm_version < 100:
        moe_list = ["bfloat16", "fp8_block", "int4_wo"]
    elif sm_version in (100, 103):
        moe_list = [
            "bfloat16",
            "fp8_block",
            "nvfp4",
            "int4_wo",
            "w4a16_mxfp4",
            "w4a8_mxfp4_mxfp8",
        ]
    else:
        # SGLang 0.5.10 routes many nvfp4 MoE cases through FlashInfer paths
        # that were not validated on SM120. Add back only live-smoked Nemotron
        # NVFP4 model cases below instead of enabling the mode globally.
        moe_list = ["bfloat16", "fp8_block", "int4_wo"]

    test_cases = []
    recorded_requested = _env_str_set("COLLECTOR_MOE_DISTRIBUTIONS") or set()
    requested_moe_types = _env_str_set("COLLECTOR_MOE_TYPES")
    requested_tp_sizes = _env_int_set("COLLECTOR_MOE_TP_SIZES")
    requested_ep_sizes = _env_int_set("COLLECTOR_MOE_EP_SIZES")
    requested_tokens = _env_int_set("COLLECTOR_MOE_TOKENS")
    requested_recorded_tokens = (
        _env_int_set("COLLECTOR_MOE_RECORDED_TOKENS")
        or _env_int_set("COLLECTOR_MOE_RANK_LOCAL_REPLAY_TOKENS")
        or requested_tokens
    )
    include_recorded = "recorded" in recorded_requested or (
        not recorded_requested and _has_recorded_distribution_file()
    )
    include_rank_local_replay = (
        not recorded_requested
        or "recorded" in recorded_requested
        or "rank_local_replay" in recorded_requested
    )
    seen_recorded_cases = set()
    seen_rank_local_replay_cases = set()

    for common_moe_testcase in get_common_moe_test_cases():
        model_name = common_moe_testcase.model_name

        model_moe_list = moe_list
        if model_name == "zai-org/GLM-5":
            model_moe_list = ["bfloat16"]
        elif model_name == "zai-org/GLM-5-FP8":
            model_moe_list = ["fp8_block"]
        elif model_name == "nvidia/GLM-5-NVFP4":
            model_moe_list = ["nvfp4"]
            if common_moe_testcase.ep != 1 or common_moe_testcase.tp >= 32:
                continue
        elif sm_version >= 120 and model_name in _SM120_NEMOTRON_NVFP4_MODELS:
            model_moe_list = [*model_moe_list, "nvfp4"]

        base_num_tokens_list = [
            num_tokens
            for num_tokens in common_moe_testcase.num_tokens_list
            if num_tokens <= 20480
        ]
        num_tokens_set = set(base_num_tokens_list)
        if include_rank_local_replay:
            replay_tokens = _rank_local_replay_tokens(
                ep_size=common_moe_testcase.ep,
                num_experts=common_moe_testcase.num_experts,
            )
            if requested_recorded_tokens is not None:
                replay_tokens = replay_tokens.intersection(requested_recorded_tokens)
            num_tokens_set.update(token for token in replay_tokens if token <= 20480)
        num_tokens_list = sorted(num_tokens_set)

        for moe_type, num_tokens in itertools.product(model_moe_list, num_tokens_list):
            if requested_moe_types is not None and moe_type not in requested_moe_types:
                continue
            base_token_requested = (
                requested_tokens is None or int(num_tokens) in requested_tokens
            )
            recorded_token_requested = (
                requested_recorded_tokens is None
                or int(num_tokens) in requested_recorded_tokens
            )
            if requested_tp_sizes is not None and int(common_moe_testcase.tp) not in requested_tp_sizes:
                continue
            if requested_ep_sizes is not None and int(common_moe_testcase.ep) not in requested_ep_sizes:
                continue
            if not moe_model_allows_quantization("sglang", model_name, moe_type):
                continue
            is_native_dsv4 = model_name.startswith("deepseek-ai/DeepSeek-V4-")
            if is_native_dsv4 and moe_type == "w4a8_mxfp4_mxfp8" and num_tokens > 8192:
                # DeepSeek-V4 FP4 experts are only exercised up to the SGLang
                # prefill chunk size. Larger synthetic masked-CuteDSL cases
                # allocate per-expert temporary buffers that exceed GB300 memory.
                continue
            if moe_type == "nvfp4" and is_native_dsv4:
                # Native DeepSeek-V4 experts are queried by AIC as
                # w4a8_mxfp4_mxfp8.  The SGLang kernel path is the same FP4
                # FlashInfer/CuteDSL path, but the perf row must use the AIC
                # quant-mode name so silicon lookup can find it.
                continue
            if (
                sm_version >= 120
                and moe_type == "nvfp4"
                and model_name in _SM120_NEMOTRON_NVFP4_MODELS
                and common_moe_testcase.ep == 1
                and (common_moe_testcase.inter_size // common_moe_testcase.tp) % 32 != 0
            ):
                # The SGLang 0.5.10 EP=1 NVFP4 path uses FlashInfer's TRTLLM
                # BF16xFP4 routed kernel, which requires the local intermediate
                # size to be divisible by 32. Keep non-divisible Nemotron
                # slices out of generated collection plans.
                continue
            # fp8_block requires hidden_size divisible by block group_size (128)
            if moe_type == "fp8_block" and (
                common_moe_testcase.hidden_size % 128 != 0 or common_moe_testcase.inter_size % 128 != 0
            ):
                continue
            if (
                moe_type == "fp8_block"
                and sm_version >= 120
                and common_moe_testcase.hidden_size == 4096
                and common_moe_testcase.inter_size == 14336
                and common_moe_testcase.topk == 2
                and common_moe_testcase.num_experts == 8
                and common_moe_testcase.tp == 32
                and (
                    num_tokens >= 16
                    or (common_moe_testcase.ep == 2 and num_tokens >= 8)
                    or (common_moe_testcase.ep == 4 and num_tokens >= 4)
                    or (common_moe_testcase.ep == 8 and num_tokens >= 2)
                )
            ):
                # SGLang 0.5.10 uses the default Triton fp8 block MoE config for
                # Mixtral on SM120 at this TP slice. These token counts require
                # 144 KiB shared memory, above the 99 KiB runtime limit.
                continue
            if (
                moe_type == "fp8_block"
                and sm_version >= 120
                and common_moe_testcase.hidden_size == 4096
                and common_moe_testcase.inter_size == 2688
                and common_moe_testcase.topk == 22
                and common_moe_testcase.num_experts == 512
                and (
                    (common_moe_testcase.tp == 2 and num_tokens >= 768)
                    or (common_moe_testcase.tp == 2 and common_moe_testcase.ep == 2 and num_tokens >= 320)
                    or (common_moe_testcase.tp == 2 and common_moe_testcase.ep == 4 and num_tokens >= 160)
                    or (common_moe_testcase.tp == 4 and common_moe_testcase.ep == 2 and num_tokens >= 320)
                    or (common_moe_testcase.tp == 4 and common_moe_testcase.ep == 4 and num_tokens >= 160)
                    or (common_moe_testcase.tp == 4 and common_moe_testcase.ep == 8 and num_tokens >= 80)
                    or (common_moe_testcase.tp == 4 and common_moe_testcase.ep == 16 and num_tokens >= 48)
                    or (common_moe_testcase.tp == 4 and common_moe_testcase.ep == 32 and num_tokens >= 32)
                    or (common_moe_testcase.tp == 4 and common_moe_testcase.ep == 64 and num_tokens >= 16)
                    or (common_moe_testcase.tp == 4 and num_tokens >= 768)
                    or (common_moe_testcase.tp == 8 and common_moe_testcase.ep == 2 and num_tokens >= 320)
                    or (common_moe_testcase.tp == 8 and common_moe_testcase.ep == 4 and num_tokens >= 160)
                    or (common_moe_testcase.tp == 8 and common_moe_testcase.ep == 8 and num_tokens >= 80)
                    or (common_moe_testcase.tp == 8 and common_moe_testcase.ep == 16 and num_tokens >= 48)
                    or (common_moe_testcase.tp == 8 and common_moe_testcase.ep == 32 and num_tokens >= 32)
                    or (common_moe_testcase.tp == 8 and num_tokens >= 768)
                    or (common_moe_testcase.tp == 2 and common_moe_testcase.ep == 8 and num_tokens >= 80)
                    or (common_moe_testcase.tp == 2 and common_moe_testcase.ep == 16 and num_tokens >= 48)
                    or (common_moe_testcase.tp == 2 and common_moe_testcase.ep == 32 and num_tokens >= 32)
                    or (common_moe_testcase.tp == 2 and common_moe_testcase.ep == 64)
                    or (common_moe_testcase.tp == 2 and common_moe_testcase.ep == 128)
                    or (common_moe_testcase.tp == 16 and num_tokens >= 768)
                    or (common_moe_testcase.tp == 16 and common_moe_testcase.ep == 2 and num_tokens >= 320)
                    or (common_moe_testcase.tp == 16 and common_moe_testcase.ep == 4 and num_tokens >= 160)
                    or (common_moe_testcase.tp == 16 and common_moe_testcase.ep == 8 and num_tokens >= 80)
                    or (common_moe_testcase.tp == 16 and common_moe_testcase.ep == 16 and num_tokens >= 48)
                    or (common_moe_testcase.tp == 32 and num_tokens >= 768)
                    or (common_moe_testcase.tp == 32 and common_moe_testcase.ep == 2 and num_tokens >= 320)
                    or (common_moe_testcase.tp == 32 and common_moe_testcase.ep == 4 and num_tokens >= 160)
                    or (common_moe_testcase.tp == 32 and common_moe_testcase.ep == 8 and num_tokens >= 80)
                )
            ):
                # SGLang 0.5.10 falls back to the default Triton fp8 block MoE
                # config for Nemotron-3 Super on SM120 for these TP/EP slices.
                # That config requires 144 KiB shared memory, above the 99 KiB
                # runtime limit.
                continue
            if (
                moe_type == "fp8_block"
                and sm_version >= 120
                and common_moe_testcase.hidden_size == 2048
                and common_moe_testcase.inter_size == 768
                and common_moe_testcase.topk == 8
                and common_moe_testcase.num_experts == 128
                and common_moe_testcase.tp == 4
                and (
                    num_tokens >= 160
                    or (common_moe_testcase.ep == 2 and num_tokens >= 80)
                    or (common_moe_testcase.ep == 4 and num_tokens >= 48)
                    or (common_moe_testcase.ep == 8 and num_tokens >= 32)
                    or (common_moe_testcase.ep == 16 and num_tokens >= 16)
                    or (common_moe_testcase.ep == 32 and num_tokens >= 8)
                    or (common_moe_testcase.ep == 64 and num_tokens >= 8)
                )
            ):
                # SGLang 0.5.10 also uses the default Triton fp8 block MoE config
                # for Qwen3-30B-A3B on SM120. For these larger token counts that
                # config requires 144 KiB shared memory, above the 99 KiB limit.
                continue
            if (
                moe_type == "fp8_block"
                and sm_version >= 120
                and common_moe_testcase.hidden_size == 4096
                and common_moe_testcase.inter_size == 1536
                and common_moe_testcase.topk == 8
                and common_moe_testcase.num_experts == 128
                and (
                    (
                        common_moe_testcase.tp == 8
                        and (
                            num_tokens >= 160
                            or (common_moe_testcase.ep == 2 and num_tokens >= 80)
                            or (common_moe_testcase.ep == 4 and num_tokens >= 48)
                            or (common_moe_testcase.ep == 8 and num_tokens >= 32)
                            or (common_moe_testcase.ep == 16 and num_tokens >= 16)
                            or (common_moe_testcase.ep == 32 and num_tokens >= 8)
                        )
                    )
                    or (
                        common_moe_testcase.tp == 16
                        and (
                            num_tokens >= 160
                            or (common_moe_testcase.ep == 2 and num_tokens >= 80)
                            or (common_moe_testcase.ep == 4 and num_tokens >= 48)
                            or (common_moe_testcase.ep == 8 and num_tokens >= 32)
                            or (common_moe_testcase.ep == 16 and num_tokens >= 16)
                        )
                    )
                    or (
                        common_moe_testcase.tp == 32
                        and (
                            num_tokens >= 160
                            or (common_moe_testcase.ep == 2 and num_tokens >= 80)
                            or (common_moe_testcase.ep == 4 and num_tokens >= 48)
                            or (common_moe_testcase.ep == 8 and num_tokens >= 32)
                        )
                    )
                )
            ):
                # SGLang 0.5.10 uses the default Triton fp8 block MoE config for
                # Qwen3-235B-A22B on SM120. For these token counts that config
                # requires 144 KiB shared memory, above the 99 KiB limit.
                continue
            if (
                moe_type == "fp8_block"
                and sm_version >= 120
                and common_moe_testcase.hidden_size == 6144
                and common_moe_testcase.inter_size == 2560
                and common_moe_testcase.topk == 8
                and common_moe_testcase.num_experts == 160
                and (
                    (
                        common_moe_testcase.tp == 8
                        and (
                            num_tokens >= 192
                            or (common_moe_testcase.ep == 2 and num_tokens >= 96)
                            or (common_moe_testcase.ep == 4 and num_tokens >= 48)
                            or (common_moe_testcase.ep == 8 and num_tokens >= 32)
                            or (common_moe_testcase.ep == 16 and num_tokens >= 16)
                            or (common_moe_testcase.ep == 32 and num_tokens >= 8)
                        )
                    )
                    or (
                        common_moe_testcase.tp == 16
                        and (
                            num_tokens >= 192
                            or (common_moe_testcase.ep == 2 and num_tokens >= 96)
                            or (common_moe_testcase.ep == 4 and num_tokens >= 48)
                            or (common_moe_testcase.ep == 8 and num_tokens >= 32)
                            or (common_moe_testcase.ep == 16 and num_tokens >= 16)
                        )
                    )
                    or (
                        common_moe_testcase.tp == 32
                        and (
                            num_tokens >= 192
                            or (common_moe_testcase.ep == 2 and num_tokens >= 96)
                            or (common_moe_testcase.ep == 4 and num_tokens >= 48)
                            or (common_moe_testcase.ep == 8 and num_tokens >= 32)
                        )
                    )
                )
            ):
                # SGLang 0.5.10 uses the default Triton fp8 block MoE config for
                # Qwen3-Coder-480B-A35B on SM120. For these token counts that
                # config requires 144 KiB shared memory, above the 99 KiB limit.
                continue
            if (
                moe_type == "fp8_block"
                and sm_version >= 120
                and common_moe_testcase.hidden_size == 4096
                and common_moe_testcase.inter_size == 1024
                and common_moe_testcase.topk == 10
                and common_moe_testcase.num_experts == 512
                and (
                    (common_moe_testcase.tp == 16 and num_tokens >= 768)
                    or (common_moe_testcase.tp == 16 and common_moe_testcase.ep == 2 and num_tokens >= 320)
                    or (common_moe_testcase.tp == 16 and common_moe_testcase.ep == 4 and num_tokens >= 160)
                    or (common_moe_testcase.tp == 16 and common_moe_testcase.ep == 8 and num_tokens >= 80)
                    or (common_moe_testcase.tp == 16 and common_moe_testcase.ep == 16 and num_tokens >= 48)
                    or (common_moe_testcase.tp == 32 and num_tokens >= 768)
                    or (common_moe_testcase.tp == 32 and common_moe_testcase.ep == 2 and num_tokens >= 320)
                    or (common_moe_testcase.tp == 32 and common_moe_testcase.ep == 4 and num_tokens >= 160)
                    or (common_moe_testcase.tp == 32 and common_moe_testcase.ep == 8 and num_tokens >= 80)
                )
            ):
                # SGLang 0.5.10 uses the default Triton fp8 block MoE config for
                # Qwen3.5-397B-A17B on SM120. For these token counts that config
                # requires 144 KiB shared memory, above the 99 KiB limit.
                continue
            if (
                moe_type == "fp8_block"
                and sm_version >= 120
                and common_moe_testcase.hidden_size == 6144
                and common_moe_testcase.inter_size == 2048
                and common_moe_testcase.topk == 8
                and common_moe_testcase.num_experts == 256
                and common_moe_testcase.tp == 32
                and (
                    num_tokens >= 320
                    or (common_moe_testcase.ep == 2 and num_tokens >= 160)
                    or (common_moe_testcase.ep == 4 and num_tokens >= 80)
                    or (common_moe_testcase.ep == 8 and num_tokens >= 48)
                )
            ):
                # SGLang 0.5.10 uses the default Triton fp8 block MoE config for
                # GLM-5 on SM120 at this TP slice. For these token counts that
                # config requires 144 KiB shared memory, above the 99 KiB limit.
                continue
            if (
                moe_type == "fp8_block"
                and sm_version >= 120
                and common_moe_testcase.hidden_size == 7168
                and common_moe_testcase.inter_size == 2048
                and common_moe_testcase.topk == 8
                and common_moe_testcase.num_experts == 256
                and common_moe_testcase.tp == 32
                and (
                    num_tokens >= 320
                    or (common_moe_testcase.ep == 2 and num_tokens >= 160)
                    or (common_moe_testcase.ep == 4 and num_tokens >= 80)
                    or (common_moe_testcase.ep == 8 and num_tokens >= 48)
                )
            ):
                # SGLang 0.5.10 uses the default Triton fp8 block MoE config for
                # DeepSeek-V3 on SM120 at this TP slice. For these token counts
                # that config requires 144 KiB shared memory, above the 99 KiB
                # limit.
                continue
            if (
                moe_type == "fp8_block"
                and sm_version >= 120
                and common_moe_testcase.hidden_size == 4096
                and common_moe_testcase.inter_size == 2048
                and common_moe_testcase.topk == 6
                and common_moe_testcase.num_experts == 256
                and common_moe_testcase.tp == 32
                and (
                    num_tokens >= 320
                    or (common_moe_testcase.ep == 2 and num_tokens >= 160)
                    or (common_moe_testcase.ep == 4 and num_tokens >= 80)
                    or (common_moe_testcase.ep == 8 and num_tokens >= 48)
                )
            ):
                # SGLang 0.5.10 uses the default Triton fp8 block MoE config for
                # DeepSeek-V4-Flash on SM120 at this TP slice. For these token
                # counts that config requires 144 KiB shared memory, above the
                # 99 KiB limit.
                continue
            if (
                moe_type == "fp8_block"
                and sm_version >= 120
                and common_moe_testcase.hidden_size == 7168
                and common_moe_testcase.inter_size == 3072
                and common_moe_testcase.topk == 6
                and common_moe_testcase.num_experts == 384
                and (
                    (common_moe_testcase.tp == 16 and num_tokens >= 512)
                    or (common_moe_testcase.tp == 16 and common_moe_testcase.ep == 2 and num_tokens >= 256)
                    or (common_moe_testcase.tp == 16 and common_moe_testcase.ep == 4 and num_tokens >= 128)
                    or (common_moe_testcase.tp == 16 and common_moe_testcase.ep == 8 and num_tokens >= 64)
                    or (common_moe_testcase.tp == 16 and common_moe_testcase.ep >= 16 and num_tokens >= 48)
                    or (common_moe_testcase.tp == 32 and num_tokens >= 512)
                    or (common_moe_testcase.tp == 32 and common_moe_testcase.ep == 2 and num_tokens >= 256)
                    or (common_moe_testcase.tp == 32 and common_moe_testcase.ep == 4 and num_tokens >= 128)
                    or (common_moe_testcase.tp == 32 and common_moe_testcase.ep == 8 and num_tokens >= 64)
                    or (common_moe_testcase.tp == 32 and common_moe_testcase.ep >= 16 and num_tokens >= 48)
                )
            ):
                # SGLang 0.5.10 uses the default Triton fp8 block MoE config for
                # DeepSeek-V4-Pro on SM120 for these TP/EP slices. For these
                # token counts that config requires 144 KiB shared memory,
                # above the 99 KiB limit.
                continue
            if (
                moe_type == "fp8_block"
                and sm_version >= 120
                and common_moe_testcase.hidden_size == 7168
                and common_moe_testcase.inter_size == 2048
                and common_moe_testcase.topk == 8
                and common_moe_testcase.num_experts == 384
                and common_moe_testcase.tp == 32
                and (
                    num_tokens >= 512
                    or (common_moe_testcase.ep == 2 and num_tokens >= 256)
                    or (common_moe_testcase.ep == 4 and num_tokens >= 128)
                    or (common_moe_testcase.ep == 8 and num_tokens >= 64)
                )
            ):
                # SGLang 0.5.10 uses the default Triton fp8 block MoE config for
                # Kimi-K2 on SM120 at this TP slice. For these token counts that
                # config requires 144 KiB shared memory, above the 99 KiB limit.
                continue
            if (
                moe_type == "fp8_block"
                and sm_version >= 120
                and common_moe_testcase.hidden_size == 3072
                and common_moe_testcase.inter_size == 1536
                and common_moe_testcase.topk == 8
                and common_moe_testcase.num_experts == 256
                and (
                    common_moe_testcase.tp == 16
                    or (common_moe_testcase.tp == 8 and common_moe_testcase.ep == 2 and num_tokens >= 160)
                    or (common_moe_testcase.tp == 8 and common_moe_testcase.ep == 4 and num_tokens >= 80)
                    or (common_moe_testcase.tp == 8 and common_moe_testcase.ep == 8 and num_tokens >= 48)
                    or (common_moe_testcase.tp == 8 and common_moe_testcase.ep == 16 and num_tokens >= 32)
                    or (common_moe_testcase.tp == 8 and common_moe_testcase.ep == 32 and num_tokens >= 16)
                    or (common_moe_testcase.tp == 8 and num_tokens >= 320)
                    or (common_moe_testcase.tp == 32 and common_moe_testcase.ep == 2 and num_tokens >= 160)
                    or (common_moe_testcase.tp == 32 and common_moe_testcase.ep == 4 and num_tokens >= 80)
                    or (common_moe_testcase.tp == 32 and common_moe_testcase.ep == 8 and num_tokens >= 48)
                    or (common_moe_testcase.tp == 32 and num_tokens >= 320)
                )
            ):
                # SGLang 0.5.10 uses the default Triton fp8 block MoE config for
                # MiniMax-M2.x on SM120. For these token counts that config
                # requires 144 KiB shared memory, above the 99 KiB limit.
                continue

            if moe_type in ("nvfp4", "w4a8_mxfp4_mxfp8"):
                shard_k = common_moe_testcase.inter_size // common_moe_testcase.tp
                # fp4_quantize requires weight dims divisible by 16 after TP sharding.
                # CuteDSL grouped GEMM additionally requires 16-byte contiguous alignment:
                # for fp4 (4-bit), that's 32 elements (16 * 8 // 4 = 32).
                # See: flashinfer/cute_dsl/blockscaled_gemm.py
                #   Sm100BlockScaledPersistentDenseGemmKernel.is_valid_tensor_alignment()
                if shard_k % 32 != 0:
                    continue

            # int4_wo (W4A16): packed K dims must be divisible by group_size (128).
            # w1 packed K = hidden_size // 2  → need hidden_size % 256 == 0
            # w2 packed K = shard_inter // 4 = inter_size // (2*tp) → need (inter_size // tp) % 256 == 0
            if moe_type == "int4_wo" and (
                common_moe_testcase.hidden_size % 256 != 0
                or (common_moe_testcase.inter_size // common_moe_testcase.tp) % 256 != 0
            ):
                continue
            if moe_type == "int4_wo" and common_moe_testcase.topk > (
                common_moe_testcase.num_experts // common_moe_testcase.ep
            ):
                # The SGLang int4 MoE path benchmarks the rank-0 local expert
                # slice. Cases where global top-k exceeds local experts fail
                # routing before kernel timing and are not valid single-rank
                # collector inputs.
                continue

            swiglu_limit = None
            # DeepSeek-V4 uses swiglu_limit=10
            if "DeepSeek-V4" in common_moe_testcase.model_name:
                swiglu_limit = 10

            base_case = [
                moe_type,
                num_tokens,
                common_moe_testcase.hidden_size,
                common_moe_testcase.inter_size,
                common_moe_testcase.topk,
                common_moe_testcase.num_experts,
                common_moe_testcase.tp,
                common_moe_testcase.ep,
                common_moe_testcase.model_name,
                common_moe_testcase.token_expert_distribution,
                common_moe_testcase.power_law_alpha,
                swiglu_limit,
            ]
            base_distribution = common_moe_testcase.token_expert_distribution
            if base_token_requested and (
                not recorded_requested or base_distribution in recorded_requested
            ):
                test_cases.append(base_case)

            if include_rank_local_replay and recorded_token_requested:
                for replay_distribution in _rank_local_replay_distributions(
                    num_tokens=num_tokens,
                    ep_size=common_moe_testcase.ep,
                    num_experts=common_moe_testcase.num_experts,
                ):
                    if moe_type == "int4_wo" and common_moe_testcase.ep >= 8:
                        # The Marlin W4A16 MoE kernel does not accept a
                        # rank-local replay batch whose local M is zero
                        # (Invalid MNK=[0, ...]).  Sparse recorded dummy
                        # distributions can legitimately produce an empty
                        # local-rank workload at larger EP.  Keep these cases
                        # out of the generic MoE collection plan; WideEP/DeepEP
                        # calibration uses the fp8 wideep_moe collector path.
                        continue
                    replay_key = (
                        moe_type,
                        num_tokens,
                        common_moe_testcase.hidden_size,
                        common_moe_testcase.inter_size,
                        common_moe_testcase.topk,
                        common_moe_testcase.num_experts,
                        common_moe_testcase.tp,
                        common_moe_testcase.ep,
                        common_moe_testcase.model_name,
                        replay_distribution,
                    )
                    if replay_key in seen_rank_local_replay_cases:
                        continue
                    seen_rank_local_replay_cases.add(replay_key)
                    replay_case = list(base_case)
                    replay_case[9] = replay_distribution
                    replay_case[10] = 0
                    test_cases.append(replay_case)

            recorded_key = (
                moe_type,
                num_tokens,
                common_moe_testcase.hidden_size,
                common_moe_testcase.inter_size,
                common_moe_testcase.topk,
                common_moe_testcase.num_experts,
                common_moe_testcase.tp,
                common_moe_testcase.ep,
                common_moe_testcase.model_name,
            )
            if (
                include_recorded
                and recorded_token_requested
                and recorded_key not in seen_recorded_cases
                and _recorded_distribution_has_case(
                    num_tokens,
                    common_moe_testcase.topk,
                    common_moe_testcase.num_experts,
                )
            ):
                seen_recorded_cases.add(recorded_key)
                recorded_case = list(base_case)
                recorded_case[9] = "recorded"
                recorded_case[10] = 0
                test_cases.append(recorded_case)

    return test_cases


class BenchmarkConfig(TypedDict):
    BLOCK_SIZE_M: int
    BLOCK_SIZE_N: int
    BLOCK_SIZE_K: int
    GROUP_SIZE_M: int
    num_warps: int
    num_stages: int


def benchmark_config(
    config: BenchmarkConfig,
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_nvfp4: bool = False,
    use_trtllm_bf16_fp4: bool = False,
    use_int4_w4a16: bool = False,
    use_mxfp4_w4a16: bool = False,
    use_mxfp4_w4a8: bool = False,
    block_shape: list[int] | None = None,
    num_iters: int = 10,
    distributed: str = "power_law",
    power_law_alpha: float = 0,
    workloads: list["Rank0Workload"] | None = None,
    swiglu_limit: float | None = None,
    moe_tp_size: int = 1,
    moe_ep_size: int = 1,
    model_name: str = "",
    phase: str = "generation",
) -> float:
    device = torch.device("cuda")
    use_mxfp4_moe = use_mxfp4_w4a16 or use_mxfp4_w4a8
    if workloads is not None:
        num_iters = len(workloads)
        num_tokens = max(workload["hidden_states"].shape[0] for workload in workloads)

    # 1. Gating Output Generation (not needed for Marlin int4 path which builds its own)
    if not (use_int4_w4a16 and _HAS_MARLIN_MOE):
        if workloads is not None:
            gating_output = None
        elif distributed == "uniform":
            gating_output = torch.randn(num_iters, num_tokens, num_experts, dtype=torch.float32, device=device)
        elif distributed == "balanced":
            gating_output = [balanced_logits(num_tokens, num_experts, topk).to(device) for _ in range(num_iters)]
        elif distributed == "power_law":
            gating_output = [
                power_law_logits_v3(num_tokens, num_experts, topk, 1, power_law_alpha).to(device)
                for _ in range(num_iters)
            ]
        elif distributed == "recorded":
            gating_output = [
                recorded_logits_v3(num_tokens, num_experts, topk, 1).to(device)
                for _ in range(num_iters)
            ]
        else:
            raise ValueError(f"Unsupported distributed mode: {distributed}")

    # 2. Setup based on Path
    if use_int4_w4a16 and _HAS_MARLIN_MOE:
        # Marlin int4 MoE path: repack GPTQ weights into Marlin tile layout
        # and call fused_marlin_moe which uses optimized CUDA kernels.
        num_bits = 4
        pack_factor = 8  # 32-bit int packs 8 x int4
        group_size = block_shape[1] if block_shape else 128

        # GPTQ-packed weights: (E, K // pack_factor, N) as int32
        w1_packed = torch.randint(
            -(2**31),
            2**31 - 1,
            (num_experts, hidden_size // pack_factor, shard_intermediate_size),
            dtype=torch.int32,
            device=device,
        )
        w2_packed = torch.randint(
            -(2**31),
            2**31 - 1,
            (num_experts, (shard_intermediate_size // 2) // pack_factor, hidden_size),
            dtype=torch.int32,
            device=device,
        )
        empty_perm = torch.empty((num_experts, 0), dtype=torch.int32, device=device)

        # Repack to Marlin layout: (E, K // 16, N * (num_bits // 2))
        w1_marlin = gptq_marlin_moe_repack(
            w1_packed,
            empty_perm,
            hidden_size,
            shard_intermediate_size,
            num_bits,
        )
        w2_marlin = gptq_marlin_moe_repack(
            w2_packed,
            empty_perm,
            shard_intermediate_size // 2,
            hidden_size,
            num_bits,
        )
        del w1_packed, w2_packed

        # Per-group scales: (E, K // group_size, N) — then permute for Marlin
        w1_scale = torch.randn(
            (num_experts, hidden_size // group_size, shard_intermediate_size),
            dtype=dtype,
            device=device,
        )
        w2_scale = torch.randn(
            (num_experts, (shard_intermediate_size // 2) // group_size, hidden_size),
            dtype=dtype,
            device=device,
        )
        w1_scale = marlin_moe_permute_scales(w1_scale, hidden_size, shard_intermediate_size, group_size)
        w2_scale = marlin_moe_permute_scales(w2_scale, shard_intermediate_size // 2, hidden_size, group_size)

        x = None if workloads is not None else torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)

        if workloads is None:
            if distributed == "power_law":
                gating_list = [
                    power_law_logits_v3(num_tokens, num_experts, topk, 1, power_law_alpha).to(device)
                    for _ in range(num_iters)
                ]
            elif distributed == "balanced":
                gating_list = [balanced_logits(num_tokens, num_experts, topk).to(device) for _ in range(num_iters)]
            elif distributed == "recorded":
                gating_list = [recorded_logits_v3(num_tokens, num_experts, topk, 1).to(device) for _ in range(num_iters)]
            else:
                gating_list = [
                    torch.randn(num_tokens, num_experts, dtype=torch.float32, device=device) for _ in range(num_iters)
                ]

        def run_op(i):
            if workloads is not None:
                current_hidden_states = workloads[i % num_iters]["hidden_states"]
                current_topk = workloads[i % num_iters]["topk_output"]
                # fused_marlin_moe asserts gating_output.shape[0] == hidden_states.shape[0],
                # but only uses topk_weights/topk_ids for routing. Provide a dummy.
                dummy_gating = torch.zeros(
                    current_hidden_states.shape[0],
                    num_experts,
                    device=current_hidden_states.device,
                    dtype=torch.float32,
                )
                # build_rank0_local_workload sets remote expert IDs to -1
                # and their weights to 0.  The Marlin CUDA kernel
                # (moe_wna16_marlin_gemm) indexes weight tensors by expert
                # ID without masking, so -1 causes illegal memory access.
                # Clamp to 0; the zero weight ensures no contribution.
                safe_topk_ids = current_topk.topk_ids.clamp(min=0)
                fused_marlin_moe(
                    current_hidden_states,
                    w1_marlin,
                    w2_marlin,
                    w1_scale,
                    w2_scale,
                    dummy_gating,
                    current_topk.topk_weights,
                    safe_topk_ids,
                    num_bits=num_bits,
                    is_k_full=True,
                )
            else:
                gating = gating_list[i % num_iters]
                new_topk = select_experts(x, gating, TopKConfig(top_k=topk))
                fused_marlin_moe(
                    x,
                    w1_marlin,
                    w2_marlin,
                    w1_scale,
                    w2_scale,
                    gating,
                    new_topk.topk_weights,
                    new_topk.topk_ids,
                    num_bits=num_bits,
                    is_k_full=True,
                )

    elif use_nvfp4 and use_trtllm_bf16_fp4 and workloads is None:
        from flashinfer.fused_moe import ActivationType, trtllm_fp4_block_scale_routed_moe
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            _pack_topk_for_flashinfer_routed,
            quantize_hidden_states_fp4,
        )

        if hidden_size % 32 != 0 or (shard_intermediate_size // 2) % 32 != 0:
            raise ValueError(
                "FlashInfer TRTLLM BF16xFP4 MoE requires hidden and intermediate dimensions "
                f"to be divisible by 32, got hidden_size={hidden_size} and "
                f"intermediate_size={shard_intermediate_size // 2}"
            )

        intermediate_size = shard_intermediate_size // 2
        x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
        router_logits_list = (
            gating_output
            if isinstance(gating_output, list)
            else [gating_output[i] for i in range(gating_output.shape[0])]
        )

        # Match GLM-5's ModelOpt FP4 + FlashInfer TRTLLM routed path. That path
        # quantizes activations to FP4 before invoking the routed MoE kernel;
        # passing BF16 activations directly selects a much slower BF16xFP4 path.
        w13_weight = torch.randint(
            0,
            256,
            (num_experts, shard_intermediate_size, hidden_size // 2),
            dtype=torch.uint8,
            device=device,
        )
        w2_weight = torch.randint(
            0,
            256,
            (num_experts, hidden_size, intermediate_size // 2),
            dtype=torch.uint8,
            device=device,
        )
        sf_block_size = 16
        w13_scale = torch.ones(
            (num_experts, shard_intermediate_size, hidden_size // sf_block_size),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        w2_scale = torch.ones(
            (num_experts, hidden_size, intermediate_size // sf_block_size),
            dtype=torch.float8_e4m3fn,
            device=device,
        )

        output = torch.empty(num_tokens, hidden_size, dtype=dtype, device=device)
        tune_max_num_tokens = max(1, 1 << (num_tokens - 1).bit_length())
        scale_ones = torch.ones(num_experts, dtype=torch.float32, device=device)
        input_scale_quant = torch.ones((), dtype=torch.float32, device=device)
        activation_type = ActivationType.Relu2 if _uses_relu2_moe_activation(model_name) else ActivationType.Swiglu
        topk_config = TopKConfig(
            top_k=topk,
            renormalize=True,
            scoring_func="sigmoid",
            routed_scaling_factor=1.0,
        )
        packed_topk_list = []
        for logits in router_logits_list:
            topk_output = select_experts(x, logits, topk_config)
            packed_topk_list.append(
                _pack_topk_for_flashinfer_routed(
                    topk_output.topk_ids,
                    topk_output.topk_weights,
                )
            )
        torch.cuda.synchronize()

        def run_op(i):
            x_fp4, x_scale = quantize_hidden_states_fp4(x, input_scale_quant)
            packed_topk = packed_topk_list[i % len(packed_topk_list)]
            trtllm_fp4_block_scale_routed_moe(
                topk_ids=packed_topk,
                routing_bias=None,
                hidden_states=x_fp4,
                hidden_states_scale=x_scale,
                gemm1_weights=w13_weight,
                gemm1_weights_scale=w13_scale,
                gemm1_bias=None,
                gemm1_alpha=None,
                gemm1_beta=None,
                gemm1_clamp_limit=None,
                gemm2_weights=w2_weight,
                gemm2_weights_scale=w2_scale,
                gemm2_bias=None,
                output1_scale_scalar=scale_ones,
                output1_scale_gate_scalar=scale_ones,
                output2_scale_scalar=scale_ones,
                num_experts=num_experts,
                top_k=packed_topk.shape[1],
                n_group=0,
                topk_group=0,
                intermediate_size=intermediate_size,
                local_expert_offset=0,
                local_num_experts=num_experts,
                routed_scaling_factor=None,
                routing_method_type=1,
                do_finalize=True,
                activation_type=activation_type,
                output=output,
                tune_max_num_tokens=tune_max_num_tokens,
            )

    elif use_nvfp4:
        if not HAS_FLASHINFER_CUTE:
            raise ImportError("FlashInfer CuteDSL not available")
        if not _HAS_SCALED_FP4_QUANT:
            raise ImportError(
                "scaled_fp4_quant not available (sglang.jit_kernel.nvfp4); "
                "NVFP4 MoE benchmarking requires this for correct weight layout"
            )

        # Global scales and Alpha
        input_gs = torch.ones(num_experts, device=device, dtype=torch.float32)
        w1_gs = torch.ones(num_experts, device=device, dtype=torch.float32)
        a2_gs = torch.ones(num_experts, device=device, dtype=torch.float32)
        w2_gs = torch.ones(num_experts, device=device, dtype=torch.float32)
        w1_alpha = torch.ones(num_experts, device=device, dtype=torch.float32)
        w2_alpha = torch.ones(num_experts, device=device, dtype=torch.float32)

        # Weight quantization
        w1_bf16 = torch.randn(num_experts, shard_intermediate_size, hidden_size, device=device, dtype=dtype)
        w2_bf16 = torch.randn(num_experts, hidden_size, shard_intermediate_size // 2, device=device, dtype=dtype)

        # Quantize weights per-expert using scaled_fp4_quant which produces
        # swizzled blockscales and maintains (num_experts, N, K//2) layout.
        w1_list_q, w1_list_bs = [], []
        for e in range(num_experts):
            q, bs = _scaled_fp4_quant(w1_bf16[e], w1_gs[e])
            w1_list_q.append(q)
            w1_list_bs.append(bs)
        w1 = torch.stack(w1_list_q)
        w1_bs = torch.stack(w1_list_bs)

        w2_list_q, w2_list_bs = [], []
        for e in range(num_experts):
            q, bs = _scaled_fp4_quant(w2_bf16[e], w2_gs[e])
            w2_list_q.append(q)
            w2_list_bs.append(bs)
        w2 = torch.stack(w2_list_q)
        w2_bs = torch.stack(w2_list_bs)

        def get_masked_m(logits):
            _, topk_idx = torch.topk(torch.softmax(logits, dim=1), topk, dim=-1)
            counts = [(topk_idx.view(-1) == i).sum() for i in range(num_experts)]
            return torch.tensor(counts, dtype=torch.int32, device=device)

        masked_m_list = (
            [workload["masked_m"] for workload in workloads]
            if workloads is not None
            else [get_masked_m(logits) for logits in gating_output]
        )

        # Calculate the maximum tokens any single expert will handle across all iterations
        max_m = 0
        for counts in masked_m_list:
            max_m = max(max_m, counts.max().item())
        # Align to 128 for kernel efficiency and safety
        max_m = (max_m + 127) // 128 * 128

        x_dispatched = torch.randn(num_experts, max_m, hidden_size, device=device, dtype=dtype)

        def run_op(i):
            flashinfer_cutedsl_moe_masked(
                hidden_states=(x_dispatched, None),
                input_global_scale=input_gs,
                w1=w1,
                w1_blockscale=w1_bs,
                w1_alpha=w1_alpha,
                w2=w2,
                a2_global_scale=a2_gs,
                w2_blockscale=w2_bs,
                w2_alpha=w2_alpha,
                masked_m=masked_m_list[i % num_iters],
            )
    elif use_mxfp4_moe:
        if not _HAS_SGLANG_MXFP4:
            raise ImportError("SGLang MXFP4 MoE support is not available")
        if workloads is not None:
            raise ValueError("MXFP4 benchmarking uses full-router logits, not rank-local workloads")

        previous_backend = _moe_utils.MOE_RUNNER_BACKEND
        _moe_utils.MOE_RUNNER_BACKEND = MoeRunnerBackend.FLASHINFER_MXFP4
        _patch_mxfp4_single_process_parallel(moe_tp_size=moe_tp_size, moe_ep_size=moe_ep_size)

        intermediate_size = shard_intermediate_size // 2 * moe_tp_size
        mxfp4_config_kwargs = {}
        if "is_checkpoint_mxfp4_serialized" in inspect.signature(Mxfp4Config).parameters:
            mxfp4_config_kwargs["is_checkpoint_mxfp4_serialized"] = True
        quant_config = Mxfp4Config(**mxfp4_config_kwargs)
        moe_layer = FusedMoE(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=0,
            top_k=topk,
            params_dtype=dtype,
            reduce_results=False,
            quant_config=quant_config,
            prefix="aic_sglang_mxfp4_moe",
        ).to(device)
        _moe_utils.MOE_RUNNER_BACKEND = previous_backend

        with torch.no_grad():
            moe_layer.w13_weight.zero_()
            moe_layer.w2_weight.zero_()
            moe_layer.w13_weight_scale.copy_(
                torch.ones_like(moe_layer.w13_weight_scale, dtype=torch.float8_e4m3fn).view(torch.uint8)
            )
            moe_layer.w2_weight_scale.copy_(
                torch.ones_like(moe_layer.w2_weight_scale, dtype=torch.float8_e4m3fn).view(torch.uint8)
            )
            moe_layer.w13_weight_bias.zero_()
            moe_layer.w2_weight_bias.zero_()
        moe_layer.quant_method.process_weights_after_loading(moe_layer)

        x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
        if distributed == "uniform":
            router_logits_list = [
                torch.randn(num_tokens, num_experts, dtype=torch.float32, device=device) for _ in range(num_iters)
            ]
        elif distributed == "balanced":
            router_logits_list = [balanced_logits(num_tokens, num_experts, topk).to(device) for _ in range(num_iters)]
        elif distributed == "power_law":
            router_logits_list = [
                power_law_logits_v3(num_tokens, num_experts, topk, moe_ep_size, power_law_alpha).to(device)
                for _ in range(num_iters)
            ]
        elif distributed == "recorded":
            router_logits_list = [
                recorded_logits_v3(num_tokens, num_experts, topk, moe_ep_size).to(device)
                for _ in range(num_iters)
            ]
        else:
            raise ValueError(f"Unsupported distributed mode: {distributed}")

        def run_op(i):
            moe_layer(
                x,
                BypassedTopKOutput(
                    hidden_states=x,
                    router_logits=router_logits_list[i % num_iters],
                    topk_config=TopKConfig(top_k=topk),
                ),
            )
    else:
        init_dtype = torch.bfloat16 if use_fp8_w8a8 else dtype
        x = None if workloads is not None else torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
        if use_int8_w8a16 or use_int8_w8a8:
            w1 = torch.randint(
                -127, 127, (num_experts, shard_intermediate_size, hidden_size), dtype=torch.int8, device=device
            )
            w2 = torch.randint(
                -127, 127, (num_experts, hidden_size, shard_intermediate_size // 2), dtype=torch.int8, device=device
            )
        elif use_int4_w4a16:
            # W4A16: 2 int4 values packed per int8 byte — K dimension halved.
            # w1 shape: (E, N=shard_inter, K_packed=hidden//2)
            # w2 shape: (E, N=hidden, K_packed=shard_inter//4)
            w1 = torch.randint(
                0, 127, (num_experts, shard_intermediate_size, hidden_size // 2), dtype=torch.int8, device=device
            )
            w2 = torch.randint(
                0, 127, (num_experts, hidden_size, shard_intermediate_size // 4), dtype=torch.int8, device=device
            )
        else:
            w1 = torch.randn(num_experts, shard_intermediate_size, hidden_size, dtype=init_dtype, device=device)
            w2 = torch.randn(num_experts, hidden_size, shard_intermediate_size // 2, dtype=init_dtype, device=device)

        w1_scale = w2_scale = a1_scale = a2_scale = None
        if use_int8_w8a16:
            w1_scale = torch.randn((num_experts, 2 * shard_intermediate_size), dtype=torch.float32, device=device)
            w2_scale = torch.randn((hidden_size, num_experts), dtype=torch.float32, device=device)
        elif use_int4_w4a16:
            # Per-group scales along K. The GPTQ kernel receives K = A.shape[1]
            # (unpacked hidden size), so scale groups are hidden_size // group_size,
            # NOT (hidden_size // 2) // group_size (the packed size).
            # w2's K is shard_intermediate_size // 2 (post silu_and_mul, unpacked).
            group_size = block_shape[1] if block_shape else 128
            w1_scale = torch.randn(
                (num_experts, shard_intermediate_size, hidden_size // group_size),
                dtype=torch.float32,
                device=device,
            )
            w2_scale = torch.randn(
                (num_experts, hidden_size, (shard_intermediate_size // 2) // group_size),
                dtype=torch.float32,
                device=device,
            )
        elif use_fp8_w8a8 or use_int8_w8a8:
            if use_int8_w8a8 and block_shape is None:
                w1_scale = torch.randn(num_experts, shard_intermediate_size, dtype=torch.float32, device=device)
                w2_scale = torch.randn(num_experts, hidden_size, dtype=torch.float32, device=device)
            elif block_shape is None:
                w1_scale = torch.randn(num_experts, dtype=torch.float32, device=device)
                w2_scale = torch.randn(num_experts, dtype=torch.float32, device=device)
                a1_scale = torch.randn(1, dtype=torch.float32, device=device)
                a2_scale = torch.randn(1, dtype=torch.float32, device=device)
            else:
                bn, bk = block_shape
                w1_scale = torch.rand(
                    (num_experts, (shard_intermediate_size + bn - 1) // bn, (hidden_size + bk - 1) // bk),
                    dtype=torch.float32,
                    device=device,
                )
                w2_scale = torch.rand(
                    (num_experts, (hidden_size + bn - 1) // bn, (shard_intermediate_size // 2 + bk - 1) // bk),
                    dtype=torch.float32,
                    device=device,
                )

        if use_fp8_w8a8:
            f8_type = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn
            w1, w2 = w1.to(f8_type), w2.to(f8_type)

        topk_output = (
            None
            if workloads is not None
            else select_experts(x, torch.randn(num_tokens, num_experts, device=device), TopKConfig(top_k=topk))
        )

        def run_op(i):
            from sglang.srt.layers.moe.fused_moe_triton import override_config

            if workloads is None:
                input_gating = gating_output[i % num_iters]
                new_topk = select_experts(x, input_gating, TopKConfig(top_k=topk))
                topk_output.topk_weights.copy_(new_topk.topk_weights)
                topk_output.topk_ids.copy_(new_topk.topk_ids)
                topk_output.router_logits.copy_(new_topk.router_logits)
                current_hidden_states = x
                current_topk_output = topk_output
            else:
                current_hidden_states = workloads[i % num_iters]["hidden_states"]
                current_topk_output = workloads[i % num_iters]["topk_output"]
                # build_rank0_local_workload sets remote expert IDs to -1
                # and their weights to 0.  The Triton fused_moe kernel
                # indexes weight tensors by expert ID without masking,
                # so -1 causes illegal memory access.
                # Clamp to 0; the zero weight ensures no contribution.
                current_topk_output = StandardTopKOutput(
                    topk_weights=current_topk_output.topk_weights,
                    topk_ids=current_topk_output.topk_ids.clamp(min=0),
                    router_logits=current_topk_output.router_logits,
                )

            with override_config(config):
                moe_runner_config = _make_moe_runner_config(swiglu_limit=swiglu_limit)
                fused_moe(
                    current_hidden_states,
                    w1,
                    w2,
                    current_topk_output,
                    moe_runner_config=moe_runner_config,
                    use_fp8_w8a8=use_fp8_w8a8,
                    use_int8_w8a8=use_int8_w8a8,
                    use_int8_w8a16=use_int8_w8a16,
                    use_int4_w4a16=use_int4_w4a16,
                    w1_scale=w1_scale,
                    w2_scale=w2_scale,
                    a1_scale=a1_scale,
                    a2_scale=a2_scale,
                    block_shape=block_shape,
                )

    # 3. Unified Execution Loop
    outside_loop_count = 5  # Repeat ops within kernel_func to increase accuracy for fast kernels

    def kernel_func():
        for i in range(outside_loop_count):
            run_op(i)

    with benchmark_with_power(
        device=device,
        kernel_func=kernel_func,
        num_warmups=5,
        num_runs=num_iters,
        repeat_n=1,
        use_cuda_graph=_use_cuda_graph_for_phase(phase),
        # sglang >=0.5.10 adds @torch.compile paths inside fused_experts_impl
        # (moe_sum_reduce_torch_compile) that can hang during CUDA graph capture.
        # allow_graph_fail gracefully falls back to eager execution.
        allow_graph_fail=True,
    ) as results:
        pass

    return results["latency_ms"] / outside_loop_count, results["power_stats"]


def _patch_mxfp4_single_process_parallel(*, moe_tp_size: int, moe_ep_size: int) -> None:
    """Patch SGLang distributed helpers so a collector process can time rank-0 MoE."""

    for module in (_moe_layer_mod, _std_dispatch_mod, _mxfp4_mod):
        if hasattr(module, "get_tp_group"):
            module.get_tp_group = lambda: None
        if hasattr(module, "is_allocation_symmetric"):
            module.is_allocation_symmetric = lambda: False
    _moe_layer_mod.get_moe_expert_parallel_world_size = lambda: moe_ep_size
    _moe_layer_mod.get_moe_expert_parallel_rank = lambda: 0
    _moe_layer_mod.get_moe_tensor_parallel_world_size = lambda: moe_tp_size
    _moe_layer_mod.get_moe_tensor_parallel_rank = lambda: 0
    _moe_layer_mod.create_kt_config_from_server_args = lambda server_args, layer_id: None
    _std_dispatch_mod.get_moe_expert_parallel_world_size = lambda: moe_ep_size
    _std_dispatch_mod.get_moe_expert_parallel_rank = lambda: 0


def benchmark(
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_nvfp4: bool = False,
    use_trtllm_bf16_fp4: bool = False,
    use_int4_w4a16: bool = False,
    use_mxfp4_w4a16: bool = False,
    use_mxfp4_w4a8: bool = False,
    block_shape: list[int] | None = None,
    distributed: str = "power_law",
    power_law_alpha: float = 0,
    workloads: list["Rank0Workload"] | None = None,
    swiglu_limit: float | None = None,
    moe_tp_size: int = 1,
    moe_ep_size: int = 1,
    model_name: str = "",
    phase: str = "generation",
) -> tuple[dict[str, int], float]:
    torch.cuda.manual_seed_all(0)
    benchmark_num_tokens = (
        max(workload["hidden_states"].shape[0] for workload in workloads) if workloads is not None else num_tokens
    )
    use_mxfp4_moe = use_mxfp4_w4a16 or use_mxfp4_w4a8

    if use_nvfp4 or use_mxfp4_moe or (use_int4_w4a16 and _HAS_MARLIN_MOE):
        # nvfp4 uses flashinfer cutedsl backend; int4_w4a16 uses Marlin CUDA
        # kernels — neither needs Triton tuning configs.
        kernel_time, power_stats = benchmark_config(
            None,
            benchmark_num_tokens,
            num_experts,
            shard_intermediate_size,
            hidden_size,
            topk,
            dtype,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_nvfp4,
            use_trtllm_bf16_fp4,
            use_int4_w4a16,
            use_mxfp4_w4a16,
            use_mxfp4_w4a8,
            block_shape,
            distributed=distributed,
            power_law_alpha=power_law_alpha,
            workloads=workloads,
            swiglu_limit=swiglu_limit,
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
            model_name=model_name,
            phase=phase,
        )
        return kernel_time, power_stats

    dtype_str = get_config_dtype_str(
        dtype,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        use_fp8_w8a8=use_fp8_w8a8,
    )
    # NOTE(woosuk): The current naming convention uses w2.shape[2], which
    # is the intermediate size after silu_and_mul.
    block_n = block_shape[0] if block_shape else 0
    block_k = block_shape[1] if block_shape else 0
    op_config = get_moe_configs(num_experts, shard_intermediate_size // 2, dtype_str, block_n, block_k)
    if op_config is None:
        config = get_default_config(
            benchmark_num_tokens,
            num_experts,
            shard_intermediate_size,
            hidden_size,
            topk,
            dtype_str,
            False,
            block_shape,
        )
    else:
        config = op_config[min(op_config.keys(), key=lambda x: abs(x - benchmark_num_tokens))]
    kernel_time, power_stats = benchmark_config(
        config,
        benchmark_num_tokens,
        num_experts,
        shard_intermediate_size,
        hidden_size,
        topk,
        dtype,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_nvfp4,
        False,
        use_int4_w4a16,
        use_mxfp4_w4a16,
        use_mxfp4_w4a8,
        block_shape,
        distributed=distributed,
        power_law_alpha=power_law_alpha,
        workloads=workloads,
        swiglu_limit=swiglu_limit,
        moe_tp_size=moe_tp_size,
        moe_ep_size=moe_ep_size,
        model_name=model_name,
        phase=phase,
    )
    return kernel_time, power_stats


class Rank0Workload(TypedDict):
    hidden_states: torch.Tensor
    topk_output: StandardTopKOutput
    masked_m: torch.Tensor


def _keep_aic_latency_sources() -> bool:
    for name in (
        "COLLECTOR_DSV3_KEEP_LATENCY_SOURCES",
        "COLLECTOR_DSV3_KEEP_MATERIALIZED_SOURCES",
    ):
        value = os.environ.get(name)
        if value is not None:
            return value.strip().lower() in ("1", "true", "yes", "on")
    return False


def _mean_float(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _p90_float(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(0.9 * len(ordered)) - 1))
    return float(ordered[index])


def _rank_local_replay_source_features(
    rank_results: list[tuple[float, dict | None]],
    replay_rank_workloads: list[list[Rank0Workload]],
) -> dict[str, object]:
    latencies = [float(result[0]) for result in rank_results]
    max_latency = max(latencies) if latencies else 0.0
    max_rank = latencies.index(max_latency) if latencies else -1

    rank_workload_counts: list[float] = []
    rank_rows_max: list[float] = []
    rank_rows_sum: list[float] = []
    rank_masked_m_max: list[float] = []
    rank_active_experts_max: list[float] = []
    rank_assignments_max: list[float] = []
    for workloads in replay_rank_workloads:
        rank_workload_counts.append(float(len(workloads)))
        row_counts = [float(workload["hidden_states"].shape[0]) for workload in workloads]
        masked_max_values: list[float] = []
        active_expert_values: list[float] = []
        assignment_values: list[float] = []
        for workload in workloads:
            masked_m = workload["masked_m"]
            if masked_m.numel() == 0:
                masked_max_values.append(0.0)
                active_expert_values.append(0.0)
                assignment_values.append(0.0)
                continue
            masked_max_values.append(float(masked_m.max().item()))
            active_expert_values.append(float((masked_m > 0).sum().item()))
            assignment_values.append(float(masked_m.sum().item()))

        rank_rows_max.append(max(row_counts) if row_counts else 0.0)
        rank_rows_sum.append(sum(row_counts))
        rank_masked_m_max.append(max(masked_max_values) if masked_max_values else 0.0)
        rank_active_experts_max.append(max(active_expert_values) if active_expert_values else 0.0)
        rank_assignments_max.append(max(assignment_values) if assignment_values else 0.0)

    return {
        "ordinary_rank_local_latency_mean": _mean_float(latencies),
        "ordinary_rank_local_latency_p90": _p90_float(latencies),
        "ordinary_rank_local_latency_max": max_latency,
        "ordinary_rank_local_latency_min": min(latencies) if latencies else 0.0,
        "ordinary_rank_local_latency_max_rank": max_rank,
        "ordinary_rank_local_latency_spread": (max_latency - min(latencies)) if latencies else 0.0,
        "ordinary_rank_local_workload_count_mean": _mean_float(rank_workload_counts),
        "ordinary_rank_local_rows_max": max(rank_rows_max) if rank_rows_max else 0.0,
        "ordinary_rank_local_rows_mean": _mean_float(rank_rows_max),
        "ordinary_rank_local_rows_sum_mean": _mean_float(rank_rows_sum),
        "ordinary_rank_local_masked_m_max": max(rank_masked_m_max) if rank_masked_m_max else 0.0,
        "ordinary_rank_local_active_experts_max": max(rank_active_experts_max) if rank_active_experts_max else 0.0,
        "ordinary_rank_local_assignments_max": max(rank_assignments_max) if rank_assignments_max else 0.0,
        "ordinary_rank_local_measurement_scope": "single_card_rank_local_replay",
    }


def _pack_rank_local_replay_for_ordinary_moe(
    local_topk_ids: torch.Tensor,
    *,
    table_num_tokens: int,
    phase: str,
    topk: int,
    moe_ep_size: int,
) -> torch.Tensor:
    """Convert assignment-expanded replay rows to ordinary fused-MoE token rows.

    WideEP/DeepEP replay is post-dispatch and may represent one received
    assignment per row.  Ordinary fused MoE expects rank-local token rows:
    only tokens that route to this local expert slice are materialized, with
    local expert IDs in the top-k columns.  This matches build_rank0_workloads()
    for balanced/power_law, which applies a token_mask before benchmarking.
    """

    local_topk_ids = local_topk_ids.to(dtype=torch.int32).contiguous()
    if local_topk_ids.numel() == 0:
        return local_topk_ids

    def apply_generation_decode_cap(rows: int) -> int:
        if phase != "generation" or int(table_num_tokens) <= int(
            os.environ.get("COLLECTOR_ORDINARY_MOE_DECODE_CHUNK_MIN_TOKENS", "512")
        ):
            return rows
        decode_cap = max(
            1,
            int(os.environ.get("COLLECTOR_ORDINARY_MOE_DECODE_CHUNK_ROWS", "4096")),
        )
        local_decode_rows = int(
            math.ceil(int(table_num_tokens) / max(1, int(moe_ep_size)))
        )
        return min(rows, local_decode_rows, decode_cap)

    valid_per_row = (local_topk_ids >= 0).sum(dim=1)
    is_assignment_expanded = bool(int(valid_per_row.max().item()) <= 1)
    if not is_assignment_expanded:
        target_rows = apply_generation_decode_cap(int(local_topk_ids.shape[0]))
        if target_rows >= int(local_topk_ids.shape[0]):
            return local_topk_ids
        return local_topk_ids[:target_rows].contiguous()

    current_rows = int(local_topk_ids.shape[0])
    if int(moe_ep_size) <= 1:
        estimated_local_token_rows = int(table_num_tokens)
    else:
        # Profile-free estimate of how many global token rows have at least
        # one of their top-k experts on a given EP rank.  It is the same
        # pre-dispatch token-row semantics as the balanced/power_law rank0
        # path, without using server/profile truth.
        local_hit_probability = 1.0 - (1.0 - 1.0 / float(moe_ep_size)) ** int(topk)
        estimated_local_token_rows = int(round(int(table_num_tokens) * local_hit_probability))
    target_rows = min(max(1, estimated_local_token_rows), current_rows)
    target_rows = apply_generation_decode_cap(target_rows)
    if current_rows <= target_rows and int(moe_ep_size) > 1:
        return local_topk_ids

    valid_ids = local_topk_ids[local_topk_ids >= 0].flatten()
    output = torch.full(
        (target_rows, int(topk)),
        -1,
        dtype=torch.int32,
        device=local_topk_ids.device,
    )
    if valid_ids.numel() == 0:
        return output

    capacity = target_rows * int(topk)
    valid_ids = valid_ids[:capacity]
    row_ids = torch.arange(
        int(valid_ids.numel()),
        device=local_topk_ids.device,
        dtype=torch.long,
    )
    output[row_ids // int(topk), row_ids % int(topk)] = valid_ids.to(torch.int32)
    return output.contiguous()


def build_replay_rank_workloads(
    *,
    num_tokens: int,
    hidden_size: int,
    topk: int,
    num_experts: int,
    moe_ep_size: int,
    distributed: str,
    dtype: torch.dtype,
    device: torch.device,
) -> list[list[Rank0Workload]]:
    replay_spec = _parse_rank_local_distribution(distributed)
    if replay_spec is None:
        raise ValueError(f"Not a rank-local replay distribution: {distributed}")
    source, phase, enable_eplb = replay_spec
    replay_dir = _get_rank_local_replay_dir()
    if replay_dir is None:
        raise FileNotFoundError("No rank-local MoE replay directory found")

    with (replay_dir / RANK_LOCAL_REPLAY_MANIFEST).open(
        newline="",
        encoding="utf-8",
    ) as f:
        matching_rows = [
            row
            for row in csv.DictReader(f)
            if row.get("phase") == phase
            and int(row.get("num_tokens", -1)) == int(num_tokens)
            and int(row.get("requested_ep_size", -1)) == int(moe_ep_size)
            and int(row.get("num_experts", -1)) == int(num_experts)
            and row.get("workload_source", "runtime") == source
            and (
                str(row.get("enable_eplb", "")).lower() in ("1", "true")
            )
            == enable_eplb
        ]
    if not matching_rows:
        raise FileNotFoundError(
            f"No ordinary MoE replay for tokens={num_tokens}, ep={moe_ep_size}, "
            f"experts={num_experts}, source={source}, eplb={enable_eplb}"
        )
    layer_id = max(int(row["layer_id"]) for row in matching_rows)
    materialization_methods = {
        row.get("materialization_method", "") for row in matching_rows
    }
    use_ordinary_token_replay = materialization_methods == {
        "single_card_deterministic_router_layout"
    }
    if use_ordinary_token_replay and os.environ.get(
        "COLLECTOR_ORDINARY_MOE_SYNTHETIC_REPLAY_FALLBACK",
        "false",
    ).lower() in ("1", "true", "yes"):
        raise RuntimeError(
            "Synthetic ordinary MoE replay fallback was removed from the default "
            "path; consume the materialized rank-local replay bundle instead."
        )
    samples = select_replay_workloads(
        replay_dir=replay_dir,
        phase=phase,
        table_num_tokens=num_tokens,
        layer_id=layer_id,
        ep_size=moe_ep_size,
        num_experts=num_experts,
        enable_eplb=enable_eplb,
        workload_source=source,
    )

    rank_workloads: list[list[Rank0Workload]] = [
        [] for _ in range(moe_ep_size)
    ]
    for sample in samples:
        for workload in sample:
            local_topk_ids = workload.local_topk_ids.to(
                device=device,
                dtype=torch.int32,
            )
            local_topk_ids = _pack_rank_local_replay_for_ordinary_moe(
                local_topk_ids,
                table_num_tokens=num_tokens,
                phase=phase,
                topk=topk,
                moe_ep_size=moe_ep_size,
            )
            topk_weights = torch.where(
                local_topk_ids >= 0,
                torch.full_like(
                    local_topk_ids,
                    1.0 / max(1, topk),
                    dtype=torch.float32,
                ),
                torch.zeros_like(local_topk_ids, dtype=torch.float32),
            )
            rank_workloads[workload.rank].append(
                {
                    "hidden_states": torch.randn(
                        int(local_topk_ids.shape[0]),
                        hidden_size,
                        dtype=dtype,
                        device=device,
                    ),
                    "topk_output": StandardTopKOutput(
                        topk_weights=topk_weights,
                        topk_ids=local_topk_ids,
                        router_logits=torch.empty(
                            (int(local_topk_ids.shape[0]), 0),
                            dtype=torch.float32,
                            device=device,
                        ),
                    ),
                    "masked_m": workload.masked_m.to(
                        device=device,
                        dtype=torch.int32,
                    ),
                }
            )
    return rank_workloads


def build_rank0_workloads(
    num_workloads: int,
    num_tokens: int,
    hidden_size: int,
    topk: int,
    num_experts: int,
    moe_ep_size: int,
    distributed: str,
    power_law_alpha: float | None,
    dtype: torch.dtype,
    device: torch.device,
) -> list[Rank0Workload]:
    workloads: list[Rank0Workload] = []
    experts_per_rank = num_experts // moe_ep_size

    for _ in range(num_workloads):
        if distributed == "power_law":
            if power_law_alpha is None:
                raise ValueError("power_law_alpha is required for power_law distribution")
            _, rank0_info = power_law_logits_v3(
                num_tokens,
                num_experts,
                topk,
                moe_ep_size,
                power_law_alpha,
                return_rank0_info=True,
            )
        elif distributed == "balanced":
            router_logits = balanced_logits(num_tokens, num_experts, topk).to(device=device, dtype=torch.float32)
            rank0_selected_slots = torch.topk(router_logits, topk, dim=-1).indices.to(torch.int64)
            rank0_token_mask = (rank0_selected_slots < experts_per_rank).any(dim=1)
            rank0_info = {
                "rank0_selected_slots": rank0_selected_slots[rank0_token_mask],
                "rank0_logits": router_logits[rank0_token_mask],
                "rank0_num_tokens": int(rank0_token_mask.sum().item()),
                "slots_per_rank": experts_per_rank,
            }
        elif distributed == "recorded":
            _, rank0_info = recorded_logits_v3(
                num_tokens,
                num_experts,
                topk,
                moe_ep_size,
                return_rank0_info=True,
            )
        else:
            raise ValueError(f"Unsupported distribution for rank0 workloads: {distributed}")

        rank0_local = build_rank0_local_workload(rank0_info)
        rank0_num_tokens = int(rank0_local["num_tokens"])
        workloads.append(
            {
                "hidden_states": torch.randn(rank0_num_tokens, hidden_size, dtype=dtype, device=device),
                "topk_output": StandardTopKOutput(
                    topk_weights=rank0_local["topk_weights"].to(device=device, dtype=torch.float32),
                    topk_ids=rank0_local["topk_ids"].to(device=device, dtype=torch.int32),
                    router_logits=torch.empty((rank0_num_tokens, 0), dtype=torch.float32, device=device),
                ),
                "masked_m": rank0_local["masked_m"].to(device=device, dtype=torch.int32),
            }
        )

    return workloads


def run_moe_torch(
    moe_type,
    num_tokens,
    hidden_size,
    inter_size,
    topk,
    num_experts,
    moe_tp_size,
    moe_ep_size,
    model_name,
    distributed="power_law",
    power_law_alpha=0,
    swiglu_limit=None,
    *,
    perf_filename,
    device="cuda:0",
):
    os.environ["COLLECTOR_CURRENT_OUTPUT_DIR"] = os.path.dirname(os.path.abspath(str(perf_filename))) or os.getcwd()
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    assert moe_type in [
        "fp8_block",
        "bfloat16",
        "nvfp4",
        "w4a8_mxfp4_mxfp8",
        "int4_wo",
        "w4a16_mxfp4",
    ], "only support moe type = fp8_block, bfloat16, nvfp4, int4_wo, w4a16_mxfp4, or w4a8_mxfp4_mxfp8"
    assert inter_size % moe_tp_size == 0, "inter_size % moe_tp_size must be 0"
    assert num_experts % moe_ep_size == 0, "num_experts must be divisible by moe_ep_size"

    num_local_experts = num_experts // moe_ep_size
    use_int4_w4a16 = moe_type == "int4_wo"
    # GPT-OSS uses W4A16 MXFP4; DeepSeek-V4 uses W4A8 MXFP4/MXFP8.
    # Both currently run through SGLang's FlashInfer MXFP4 MoE backend.
    use_mxfp4_w4a16 = moe_type == "w4a16_mxfp4"
    use_mxfp4_w4a8 = moe_type == "w4a8_mxfp4_mxfp8"
    use_mxfp4_moe = use_mxfp4_w4a16 or use_mxfp4_w4a8
    use_nvfp4_kernel = moe_type in ("nvfp4", "w4a8_mxfp4_mxfp8")
    use_trtllm_bf16_fp4 = moe_type == "nvfp4" and moe_ep_size == 1
    # int4_wo uses block_shape=[0, group_size] for grouped scales (group_size=128)
    if use_int4_w4a16:
        block_shape = [0, 128]
    elif moe_type == "fp8_block" and (inter_size // moe_tp_size) % 128 == 0 and hidden_size % 128 == 0:
        block_shape = [128, 128]
    else:
        block_shape = None

    rank0_workloads: list[Rank0Workload] | None = None
    replay_spec = _parse_rank_local_distribution(distributed)
    replay_rank_workloads: list[list[Rank0Workload]] | None = None
    if replay_spec is not None:
        if use_mxfp4_moe:
            raise ValueError("MXFP4 MoE does not support rank-local replay")
        replay_rank_workloads = build_replay_rank_workloads(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            topk=topk,
            num_experts=num_experts,
            moe_ep_size=moe_ep_size,
            distributed=distributed,
            dtype=torch.bfloat16,
            device=torch.device(device),
        )
    elif moe_ep_size > 1 and distributed in ("power_law", "balanced", "recorded") and not use_mxfp4_moe:
        rank0_workloads = build_rank0_workloads(
            num_workloads=5,
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            topk=topk,
            num_experts=num_experts,
            moe_ep_size=moe_ep_size,
            distributed=distributed,
            power_law_alpha=power_law_alpha if distributed == "power_law" else None,
            dtype=torch.bfloat16,
            device=torch.device(device),
        )

    output_distribution, output_phase = _public_distribution_for_output(
        distributed,
        power_law_alpha,
    )

    if replay_rank_workloads is not None:
        rank_results = []
        for rank, workloads in enumerate(replay_rank_workloads):
            if not workloads:
                raise ValueError(f"Replay has no workloads for rank={rank}")
            rank_results.append(
                benchmark(
                    num_tokens,
                    num_local_experts,
                    2 * inter_size // moe_tp_size,
                    hidden_size,
                    topk,
                    torch.bfloat16,
                    moe_type == "fp8_block",
                    False,
                    False,
                    use_nvfp4=use_nvfp4_kernel,
                    use_trtllm_bf16_fp4=use_trtllm_bf16_fp4,
                    use_int4_w4a16=use_int4_w4a16,
                    use_mxfp4_w4a16=False,
                    use_mxfp4_w4a8=False,
                    block_shape=block_shape,
                    distributed=distributed,
                    power_law_alpha=power_law_alpha,
                    workloads=workloads,
                    swiglu_limit=swiglu_limit,
                    moe_tp_size=moe_tp_size,
                    moe_ep_size=moe_ep_size,
                    model_name=model_name,
                    phase=output_phase,
                )
            )
        latency, power_stats = max(rank_results, key=lambda result: result[0])
        rank_local_source_features = (
            _rank_local_replay_source_features(rank_results, replay_rank_workloads)
            if _keep_aic_latency_sources()
            else {}
        )
    elif rank0_workloads is not None:
        latency, power_stats = benchmark(
            num_tokens,
            num_local_experts,
            2 * inter_size // moe_tp_size,
            hidden_size,
            topk,
            torch.bfloat16,
            moe_type == "fp8_block",
            False,
            False,
            use_nvfp4=use_nvfp4_kernel,
            use_trtllm_bf16_fp4=use_trtllm_bf16_fp4,
            use_int4_w4a16=use_int4_w4a16,
            use_mxfp4_w4a16=False,
            use_mxfp4_w4a8=False,
            block_shape=block_shape,
            distributed=distributed,
            power_law_alpha=power_law_alpha,
            workloads=rank0_workloads,
            swiglu_limit=swiglu_limit,
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
            model_name=model_name,
            phase=output_phase,
        )
        rank_local_source_features = {}
    else:
        latency, power_stats = benchmark(
            num_tokens,
            num_experts if use_mxfp4_moe else num_local_experts,
            2 * inter_size // moe_tp_size,
            hidden_size,
            topk,
            torch.bfloat16,
            moe_type == "fp8_block",
            False,
            False,
            use_nvfp4=use_nvfp4_kernel,
            use_trtllm_bf16_fp4=use_trtllm_bf16_fp4,
            use_int4_w4a16=use_int4_w4a16,
            use_mxfp4_w4a16=use_mxfp4_w4a16,
            use_mxfp4_w4a8=use_mxfp4_w4a8,
            block_shape=block_shape,
            distributed=distributed,
            power_law_alpha=power_law_alpha,
            swiglu_limit=swiglu_limit,
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
            model_name=model_name,
            phase=output_phase,
        )
        rank_local_source_features = {}

    perf_item = {
        "moe_dtype": moe_type,
        "num_tokens": num_tokens,
        "hidden_size": hidden_size,
        "inter_size": inter_size,
        "topk": topk,
        "num_experts": num_experts,
        "moe_tp_size": moe_tp_size,
        "moe_ep_size": moe_ep_size,
        "distribution": output_distribution,
        "phase": output_phase,
        "latency": latency,
    }
    perf_item.update(rank_local_source_features)

    log_perf(
        item_list=[perf_item],
        framework="SGLang",
        version=pkg_resources.get_distribution("sglang").version,
        device_name=torch.cuda.get_device_name(device),
        op_name="moe",
        kernel_source=(
            "sglang_flashinfer_trtllm_bf16_fp4_moe"
            if use_trtllm_bf16_fp4
            else "sglang_flashinfer_cutedsl_moe"
            if use_nvfp4_kernel
            else "sglang_marlin_moe"
            if moe_type == "int4_wo" and _HAS_MARLIN_MOE
            else "sglang_flashinfer_mxfp4_moe"
            if moe_type in {"w4a16_mxfp4", "w4a8_mxfp4_mxfp8"}
            else "sglang_fused_moe_triton"
        ),
        perf_filename=perf_filename,
        power_stats=power_stats,
    )


if __name__ == "__main__":
    from collector.registry_types import PerfFile

    test_cases = get_moe_test_cases()
    for test_case in test_cases:
        print(test_case)
        run_moe_torch(*test_case, perf_filename=PerfFile.MOE)
