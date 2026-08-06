"""AIC-facing adapters for the standalone analytical kernel models.

All public functions return latency in milliseconds, matching PerfDatabase.
The underlying archived models use microseconds and remain independent of AIC.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.kernelsim.bmm.model import estimate_mla_bmm
from aiconfigurator_core.sdk.kernelsim.dsa import (
    IndexMqaShape,
    IndexTopKShape,
    estimate_index_mqa,
    estimate_index_topk,
)
from aiconfigurator_core.sdk.kernelsim.fa import (
    AttentionShape,
    HardwareSpec,
    MlaModelOptions,
    MlaRequest,
    ModelOptions,
    estimate_attention,
    estimate_mla,
)
from aiconfigurator_core.sdk.kernelsim.fa.profiles import get_reference_profile
from aiconfigurator_core.sdk.kernelsim.gemm.model import (
    estimate_bf16_gemm,
    estimate_deepgemm_fp8,
    estimate_sglang_fp8,
)
from aiconfigurator_core.sdk.kernelsim.moe.model import estimate_sglang_moe

logger = logging.getLogger(__name__)

_LEVELS = frozenset({"standard", "low", "high"})
_FP8_GEMM_RECIPES = frozenset({"sglang", "deepgemm-hopper", "deepgemm-blackwell"})
_COMMUNICATION_DTYPES = frozenset({"half", "fp8", "int8"})


@dataclass(frozen=True)
class AnalyticalConfig:
    """User-selectable policy shared by all analytical operator queries."""

    level: str = "standard"
    fp8_gemm_recipe: str = "sglang"
    attention_algorithm: str = "fa2"
    communication_mode: str = "empirical"
    moe_dispatch_dtype: str = "half"
    moe_combine_dtype: str = "half"
    wideep_dispatch_dtype: str = "half"
    wideep_combine_dtype: str = "half"

    def __post_init__(self) -> None:
        level = self.level.strip().lower()
        recipe = self.fp8_gemm_recipe.strip().lower().replace("_", "-")
        algorithm = self.attention_algorithm.strip().lower()
        communication_mode = self.communication_mode.strip().lower()
        communication_dtypes = {
            "moe_dispatch_dtype": self.moe_dispatch_dtype.strip().lower(),
            "moe_combine_dtype": self.moe_combine_dtype.strip().lower(),
            "wideep_dispatch_dtype": self.wideep_dispatch_dtype.strip().lower(),
            "wideep_combine_dtype": self.wideep_combine_dtype.strip().lower(),
        }
        if level not in _LEVELS:
            raise ValueError("analytical level must be standard, low, or high")
        if recipe not in _FP8_GEMM_RECIPES:
            raise ValueError("FP8 GEMM recipe must be sglang, deepgemm-hopper, or deepgemm-blackwell")
        if algorithm not in {"fa2", "fa3"}:
            raise ValueError("attention algorithm must be fa2 or fa3")
        if communication_mode not in {"empirical", "silicon"}:
            raise ValueError("analytical communication mode must be empirical or silicon")
        for name, dtype in communication_dtypes.items():
            if dtype not in _COMMUNICATION_DTYPES:
                raise ValueError(f"{name} must be half, fp8, or int8")
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "fp8_gemm_recipe", recipe)
        object.__setattr__(self, "attention_algorithm", algorithm)
        object.__setattr__(self, "communication_mode", communication_mode)
        for name, dtype in communication_dtypes.items():
            object.__setattr__(self, name, dtype)

    def moe_communication_dtype(self, *, wideep: bool, dispatch: bool) -> common.CommQuantMode:
        """Return the configured dtype for analytical/SOL/empirical MoE communication."""
        prefix = "wideep" if wideep else "moe"
        phase = "dispatch" if dispatch else "combine"
        return common.CommQuantMode[getattr(self, f"{prefix}_{phase}_dtype")]


def warn_backend_compatibility(backend: str) -> None:
    if backend != common.BackendName.sglang.value:
        logger.warning(
            "ANALYTICAL mode is calibrated to SGLang operator implementations and collector "
            "boundaries; backend '%s' will be evaluated with the same formulas, but kernel "
            "selection and timing-boundary differences can reduce transfer accuracy.",
            backend,
        )


def fa_hardware(system: str, gpu: dict) -> HardwareSpec:
    """Build the fine-grained attention hardware model from the system YAML."""
    required = {
        "sm_count",
        "clock_hz",
        "shared_memory_per_sm_bytes",
        "l2_capacity_bytes",
        "l2_bandwidth_bytes_s",
        "vector_peak_flops",
        "mem_bw",
        "bfloat16_tc_flops",
    }
    missing = sorted(required - gpu.keys())
    if missing:
        raise ValueError(
            f"FA/MLA analytical hardware details are incomplete for system {system!r}; "
            f"missing gpu fields: {', '.join(missing)}"
        )
    peaks = {}
    key_map = {
        "bf16": "bfloat16_tc_flops",
        "fp16": "bfloat16_tc_flops",
        "fp8": "fp8_tc_flops",
        "fp32": "float32_flops",
    }
    for dtype, key in key_map.items():
        if key in gpu:
            peaks[dtype] = float(gpu[key])
    return HardwareSpec(
        sm_count=int(gpu["sm_count"]),
        clock_hz=float(gpu["clock_hz"]),
        shared_memory_per_sm_bytes=int(gpu["shared_memory_per_sm_bytes"]),
        l2_capacity_bytes=int(gpu["l2_capacity_bytes"]),
        l2_bandwidth_bytes_s=float(gpu["l2_bandwidth_bytes_s"]),
        hbm_bandwidth_bytes_s=float(gpu["mem_bw"]),
        matrix_peak_flops=peaks,
        vector_peak_flops=float(gpu["vector_peak_flops"]),
    )


def gemm_latency_ms(m: int, n: int, k: int, quant_mode, gpu: dict, config: AnalyticalConfig) -> float:
    if quant_mode == common.GEMMQuantMode.bfloat16:
        result = estimate_bf16_gemm(m, n, k, gpu["bfloat16_tc_flops"], gpu["mem_bw"], parameter_level=config.level)
    elif quant_mode in {
        common.GEMMQuantMode.fp8,
        common.GEMMQuantMode.fp8_static,
        common.GEMMQuantMode.fp8_block,
        common.GEMMQuantMode.fp8_ootb,
    }:
        try:
            peak = gpu["fp8_tc_flops"]
        except KeyError as error:
            raise ValueError("FP8 analytical GEMM requires gpu.fp8_tc_flops") from error
        if config.fp8_gemm_recipe == "sglang":
            result = estimate_sglang_fp8(m, n, k, peak, gpu["mem_bw"], parameter_level=config.level)
        else:
            architecture = config.fp8_gemm_recipe.removeprefix("deepgemm-")
            result = estimate_deepgemm_fp8(
                m,
                n,
                k,
                peak,
                gpu["mem_bw"],
                architecture,
                parameter_level=config.level,
            )
    else:
        raise ValueError(
            f"ANALYTICAL GEMM does not support quant mode {quant_mode.name!r}; "
            "supported modes are BF16 and FP8 variants"
        )
    return result.latency_us / 1000.0


def attention_latency_ms(
    *,
    system: str,
    gpu: dict,
    batch: int,
    query_length: int,
    kv_length: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: str,
    config: AnalyticalConfig,
) -> float:
    result = estimate_attention(
        fa_hardware(system, gpu),
        AttentionShape(
            batch_size=batch,
            query_length=query_length,
            kv_length_total=kv_length,
            query_heads=query_heads,
            kv_heads=kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        ),
        ModelOptions(
            algorithm=config.attention_algorithm,
            mode="profiled",
            estimate_level=config.level,
            assume_query_tile_l2_reuse=True,
        ),
    )
    return result.latency_us / 1000.0


def mla_latency_ms(
    *,
    system: str,
    gpu: dict,
    phase: str,
    batch: int,
    query_length: int,
    sequence_length: int,
    local_heads: int,
    dtype: str,
    config: AnalyticalConfig,
) -> float:
    result = estimate_mla(
        fa_hardware(system, gpu),
        MlaRequest(
            phase=phase,
            batch_size=batch,
            local_query_heads=local_heads,
            sequence_length=sequence_length,
            query_length=query_length,
            dtype=dtype,
        ),
        MlaModelOptions(
            algorithm=config.attention_algorithm,
            estimate_level=config.level,
        ),
    )
    return result.latency_us / 1000.0


def bmm_latency_ms(
    *,
    num_tokens: int,
    num_heads: int,
    if_pre: bool,
    dtype: str,
    peak_flops_s: float,
    mem_bandwidth_bytes_s: float,
    config: AnalyticalConfig,
) -> float:
    result = estimate_mla_bmm(
        num_tokens,
        num_heads,
        "pre" if if_pre else "post",
        dtype,
        peak_flops_s,
        mem_bandwidth_bytes_s,
        parameter_level=config.level,
    )
    return result.latency_us / 1000.0


def index_mqa_latency_ms(
    *,
    gpu: dict,
    layout: str,
    batch: int,
    query_length: int,
    context_length: int,
    index_heads: int,
    index_head_dim: int,
    config: AnalyticalConfig,
) -> float:
    required = {"sm_count", "clock_hz", "fp8_tc_flops", "mem_bw"}
    missing = sorted(required - gpu.keys())
    if missing:
        raise ValueError(f"DSA Index MQA analytical hardware fields missing: {', '.join(missing)}")
    result = estimate_index_mqa(
        IndexMqaShape(
            layout=layout,
            batch_size=batch,
            query_length=query_length,
            context_length=context_length,
            index_heads=index_heads,
            head_dim=index_head_dim,
        ),
        sm_count=int(gpu["sm_count"]),
        clock_hz=float(gpu["clock_hz"]),
        fp8_peak_flops_s=float(gpu["fp8_tc_flops"]),
        hbm_bandwidth_bytes_s=float(gpu["mem_bw"]),
        parameter_level=config.level,
    )
    return result.latency_us / 1000.0


def index_topk_latency_ms(
    *,
    gpu: dict,
    layout: str,
    batch: int,
    query_length: int,
    context_length: int,
    index_topk: int,
    config: AnalyticalConfig,
) -> float:
    result = estimate_index_topk(
        IndexTopKShape(
            layout=layout,
            batch_size=batch,
            query_length=query_length,
            context_length=context_length,
            topk=index_topk,
            variant="fused",
            score_distribution="standard",
        ),
        hbm_bandwidth_bytes_s=float(gpu["mem_bw"]),
        parameter_level=config.level,
    )
    return result.latency_us / 1000.0


def dsa_sparse_attention_latency_ms(
    *,
    gpu: dict,
    batch: int,
    query_length: int,
    selected_pairs: int,
    local_heads: int,
    qk_latent_dim: int,
    value_latent_dim: int,
    qk_nope_dim: int,
    output_value_dim: int,
    config: AnalyticalConfig,
) -> float:
    """Selected-KV MLA core using FA engineering efficiencies.

    Random gather, backend fusion and split scheduling remain uncalibrated; this
    is a granular no-table fallback, not a claim of FlashMLA kernel parity.
    """
    profile = get_reference_profile(config.level)
    peak_key = "bfloat16_tc_flops"
    if peak_key not in gpu:
        raise ValueError(f"DSA sparse attention analytical model requires gpu.{peak_key}")
    tokens = batch * query_length
    attention_flops = 2.0 * selected_pairs * local_heads * (qk_latent_dim + value_latent_dim)
    absorption_flops = 2.0 * tokens * local_heads * value_latent_dim * (qk_nope_dim + output_value_dim)
    flops = attention_flops + absorption_flops
    # One latent KV stream is shared by all query heads. Use average selected
    # rows per query for the unique-cache lower bound and retain output traffic.
    average_selected = selected_pairs / max(1, tokens)
    logical_bytes = (
        tokens * local_heads * qk_latent_dim * 2
        + batch * average_selected * qk_latent_dim * 2
        + tokens * local_heads * value_latent_dim * 2
        + local_heads * value_latent_dim * (qk_nope_dim + output_value_dim) * 2
    )
    compute_ms = flops / (float(gpu[peak_key]) * profile.compute_efficiency) * 1000
    memory_ms = logical_bytes / (float(gpu["mem_bw"]) * profile.hbm_efficiency) * 1000
    return profile.fixed_overhead_us / 1000 + max(compute_ms, memory_ms)


def moe_latency_ms(
    *,
    num_tokens: int,
    hidden_size: int,
    inter_size: int,
    topk: int,
    num_experts: int,
    moe_tp_size: int,
    moe_ep_size: int,
    quant_mode,
    gpu: dict,
    config: AnalyticalConfig,
) -> float:
    recipe_map = {
        "bfloat16": "bf16_triton",
        "fp8": "fp8_block_triton",
        "fp8_block": "fp8_block_triton",
        "nvfp4": "nvfp4_cutedsl",
    }
    try:
        recipe = recipe_map[quant_mode.name]
    except KeyError as error:
        raise ValueError(f"ANALYTICAL MoE does not support quant mode {quant_mode.name!r}") from error
    peak_key = {
        "bf16_triton": "bfloat16_tc_flops",
        "fp8_block_triton": "fp8_tc_flops",
        "nvfp4_cutedsl": "fp4_tc_flops",
    }[recipe]
    if peak_key not in gpu:
        raise ValueError(f"ANALYTICAL MoE recipe {recipe!r} requires gpu.{peak_key}")
    result = estimate_sglang_moe(
        recipe,
        num_tokens,
        hidden_size,
        inter_size,
        topk,
        num_experts,
        moe_tp_size,
        moe_ep_size,
        gpu[peak_key],
        gpu["mem_bw"],
        parameter_level=config.level,
    )
    return result.latency_us / 1000.0
