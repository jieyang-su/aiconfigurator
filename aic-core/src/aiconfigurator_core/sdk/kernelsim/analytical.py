"""AIC-facing adapters for the standalone analytical kernel models.

All public functions return latency in milliseconds, matching PerfDatabase.
The underlying archived models use microseconds and remain independent of AIC.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from aiconfigurator_core import (
    analytical_attention_latency_ms as _rust_analytical_attention_latency_ms,
    analytical_bmm_latency_ms as _rust_analytical_bmm_latency_ms,
    analytical_dsa_sparse_attention_latency_ms as _rust_analytical_dsa_sparse_attention_latency_ms,
    analytical_dsv4_topk_latency_ms as _rust_analytical_dsv4_topk_latency_ms,
    analytical_gemm_latency_ms as _rust_analytical_gemm_latency_ms,
    analytical_index_mqa_latency_ms as _rust_analytical_index_mqa_latency_ms,
    analytical_index_topk_latency_ms as _rust_analytical_index_topk_latency_ms,
    analytical_mla_latency_ms as _rust_analytical_mla_latency_ms,
    analytical_msa_index_latency_ms as _rust_analytical_msa_index_latency_ms,
    analytical_moe_latency_ms as _rust_analytical_moe_latency_ms,
)
from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.kernelsim.fa import HardwareSpec

logger = logging.getLogger(__name__)

_LEVELS = frozenset({"standard", "low", "high"})
_FP8_GEMM_RECIPES = frozenset({"sglang", "deepgemm-hopper", "deepgemm-blackwell"})
_COMMUNICATION_DTYPES = frozenset({"half", "fp8", "int8"})
_COMMUNICATION_PLACEMENTS = frozenset({"independent", "tp_first"})


def fa_hardware(system: str, gpu: dict) -> HardwareSpec:
    """Compatibility adapter for callers that still inspect FA hardware.

    Production Analytical queries use the Rust model directly. This helper is
    retained for diagnostics and older tests that compare the legacy Python FA
    shape model; it does not participate in the operator query path.
    """
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
    peaks = {"bf16": float(gpu["bfloat16_tc_flops"])}
    if gpu.get("fp8_tc_flops") is not None:
        peaks["fp8"] = float(gpu["fp8_tc_flops"])
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


@dataclass(frozen=True)
class AnalyticalConfig:
    """User-selectable policy shared by all analytical operator queries."""

    level: str = "standard"
    fp8_gemm_recipe: str = "sglang"
    attention_algorithm: str = "fa2"
    sparse_attention_head_quantum: int | None = None
    communication_mode: str = "empirical"
    moe_dispatch_dtype: str = "half"
    moe_combine_dtype: str = "half"
    wideep_dispatch_dtype: str = "half"
    wideep_combine_dtype: str = "half"
    communication_placement: str = "independent"

    def __post_init__(self) -> None:
        level = self.level.strip().lower()
        recipe = self.fp8_gemm_recipe.strip().lower().replace("_", "-")
        algorithm = self.attention_algorithm.strip().lower()
        communication_mode = self.communication_mode.strip().lower()
        communication_placement = self.communication_placement.strip().lower()
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
        if self.sparse_attention_head_quantum not in {None, 64, 128}:
            raise ValueError("sparse attention head quantum must be omitted, 64, or 128")
        if communication_mode not in {"empirical", "silicon"}:
            raise ValueError("analytical communication mode must be empirical or silicon")
        if communication_placement not in _COMMUNICATION_PLACEMENTS:
            raise ValueError("communication placement must be independent or tp_first")
        for name, dtype in communication_dtypes.items():
            if dtype not in _COMMUNICATION_DTYPES:
                raise ValueError(f"{name} must be half, fp8, or int8")
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "fp8_gemm_recipe", recipe)
        object.__setattr__(self, "attention_algorithm", algorithm)
        object.__setattr__(self, "communication_mode", communication_mode)
        object.__setattr__(self, "communication_placement", communication_placement)
        for name, dtype in communication_dtypes.items():
            object.__setattr__(self, name, dtype)

    def sparse_attention_executed_heads(self, logical_heads: int) -> int:
        """Apply an explicitly selected sparse-kernel head contract."""
        heads = int(logical_heads)
        if heads <= 0:
            raise ValueError("logical sparse attention heads must be positive")
        quantum = self.sparse_attention_head_quantum
        if quantum is None or heads % quantum == 0:
            return heads
        if quantum % heads == 0:
            return quantum
        raise ValueError(
            f"sparse attention heads={heads} are incompatible with the configured "
            f"head quantum={quantum}; the SGLang padding policy only pads divisors "
            "to one execution quantum"
        )

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


def gemm_latency_ms(m: int, n: int, k: int, quant_mode, gpu: dict, config: AnalyticalConfig) -> float:
    if not isinstance(quant_mode, common.GEMMQuantMode):
        raise ValueError(
            f"ANALYTICAL GEMM does not support quant mode {getattr(quant_mode, 'name', quant_mode)!r}; "
            "supported modes are BF16, W8A16, and FP8 variants"
        )
    return _rust_analytical_gemm_latency_ms(
        int(m),
        int(n),
        int(k),
        quant_mode.name,
        float(gpu["mem_bw"]),
        gpu.get("bfloat16_tc_flops"),
        gpu.get("fp8_tc_flops"),
        gpu.get("fp4_tc_flops"),
        config.level,
        config.fp8_gemm_recipe,
    )


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
    causal: bool = True,
    value_head_dim: int | None = None,
    kv_storage_dim: int | None = None,
    kv_cache_bytes_per_token: float | None = None,
    include_kv_cache_update: bool = True,
) -> float:
    return _rust_analytical_attention_latency_ms(
        int(batch), int(query_length), int(kv_length), int(query_heads), int(kv_heads), int(head_dim), dtype,
        float(gpu["mem_bw"]), gpu.get("bfloat16_tc_flops"), gpu.get("fp8_tc_flops"),
        gpu.get("sm_count"), gpu.get("clock_hz"), gpu.get("shared_memory_per_sm_bytes"),
        gpu.get("l2_capacity_bytes"), gpu.get("l2_bandwidth_bytes_s"), gpu.get("vector_peak_flops"),
        bool(causal), value_head_dim, kv_storage_dim, kv_cache_bytes_per_token,
        bool(include_kv_cache_update), config.level, config.attention_algorithm,
    )


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
    return _rust_analytical_mla_latency_ms(
        phase, int(batch), int(query_length), int(sequence_length), int(local_heads), dtype,
        float(gpu["mem_bw"]), gpu.get("bfloat16_tc_flops"), gpu.get("fp8_tc_flops"),
        gpu.get("sm_count"), gpu.get("clock_hz"), gpu.get("shared_memory_per_sm_bytes"),
        gpu.get("l2_capacity_bytes"), gpu.get("l2_bandwidth_bytes_s"), gpu.get("vector_peak_flops"),
        config.level, config.attention_algorithm,
    )


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
    return _rust_analytical_bmm_latency_ms(
        int(num_tokens), int(num_heads), bool(if_pre), dtype, float(peak_flops_s),
        float(mem_bandwidth_bytes_s), config.level,
    )


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
    dtype: str | None = None,
) -> float:
    required = {"sm_count", "clock_hz", "mem_bw"}
    missing = sorted(required - gpu.keys())
    if missing:
        raise ValueError(f"DSA Index MQA analytical hardware fields missing: {', '.join(missing)}")
    effective_dtype = (dtype or ("fp8" if gpu.get("fp8_tc_flops") is not None else "bf16")).strip().lower()
    if effective_dtype == "bf16" and gpu.get("bfloat16_tc_flops") is None:
        raise ValueError("DSA Index MQA analytical hardware fields missing: fp8_tc_flops or bfloat16_tc_flops")
    return _rust_analytical_index_mqa_latency_ms(
        layout, effective_dtype, int(batch), int(query_length), int(context_length), int(index_heads), int(index_head_dim),
        int(gpu["sm_count"]), float(gpu["clock_hz"]), float(gpu["mem_bw"]),
        gpu.get("fp8_tc_flops"), gpu.get("bfloat16_tc_flops"), config.level,
    )


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
    return _rust_analytical_index_topk_latency_ms(
        layout, int(batch), int(query_length), int(context_length), int(index_topk),
        float(gpu["mem_bw"]), config.level,
    )


def dsv4_topk_latency_ms(
    *,
    gpu: dict,
    variant: str,
    batch: int,
    fresh_tokens: int,
    prefix_tokens: int,
    index_topk: int,
    compression_ratio: int,
    config: AnalyticalConfig,
) -> float:
    return _rust_analytical_dsv4_topk_latency_ms(
        variant, int(batch), int(fresh_tokens), int(prefix_tokens), int(index_topk),
        int(compression_ratio), int(gpu["sm_count"]), config.level,
    )


def msa_index_latency_ms(
    *,
    gpu: dict,
    phase: str,
    batch: int,
    query_length: int,
    context_length: int,
    index_heads: int,
    index_head_dim: int,
    config: AnalyticalConfig,
) -> float:
    """Estimate M3's BF16 Triton block-score kernel, including block max."""
    required = {"sm_count", "mem_bw"}
    missing = sorted(required - gpu.keys())
    if missing:
        raise ValueError(f"MSA index analytical hardware fields missing: {', '.join(missing)}")
    return _rust_analytical_msa_index_latency_ms(
        phase, int(batch), int(query_length), int(context_length), int(index_heads), int(index_head_dim),
        int(gpu["sm_count"]), float(gpu["mem_bw"]), config.level,
    )


def msa_topk_elementwise_latency_ms(
    *, gpu: dict, rows: int, candidate_blocks: int, topk_blocks: int, config: AnalyticalConfig
) -> float:
    """Small conservative MSA TopK approximation.

    The production TopK is a sorting/reduction kernel, but its measured cost is
    small relative to index and main attention. Keep it as a launch-aware
    memory recipe until a dedicated MSA TopK calibration is justified.
    """
    if rows <= 0 or candidate_blocks <= 0 or topk_blocks <= 0:
        raise ValueError("MSA TopK shape values must be positive")
    bytes_moved = rows * (candidate_blocks * 4 + min(candidate_blocks, topk_blocks) * 4)
    # A small launch floor prevents the bytes-only estimate from collapsing to
    # zero for batch-one decode, without pretending this is a fitted TopK model.
    floor_us = 1.5
    memory_us = bytes_moved / float(gpu["mem_bw"]) * 1e6 / 0.35
    return (floor_us + memory_us) * {"low": 0.85, "standard": 1.0, "high": 1.25}[config.level] / 1000.0


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
    return _rust_analytical_dsa_sparse_attention_latency_ms(
        int(batch), int(query_length), int(selected_pairs), int(local_heads), int(qk_latent_dim),
        int(value_latent_dim), int(qk_nope_dim), int(output_value_dim), float(gpu["mem_bw"]),
        gpu.get("bfloat16_tc_flops"), gpu.get("sm_count"), gpu.get("clock_hz"),
        gpu.get("shared_memory_per_sm_bytes"), gpu.get("l2_capacity_bytes"),
        gpu.get("l2_bandwidth_bytes_s"), gpu.get("vector_peak_flops"), config.level,
        config.sparse_attention_head_quantum,
    )


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
    if not isinstance(quant_mode, common.MoEQuantMode):
        raise ValueError(
            f"ANALYTICAL MoE does not support quant mode {getattr(quant_mode, 'name', quant_mode)!r}"
        )
    return _rust_analytical_moe_latency_ms(
        int(num_tokens),
        int(hidden_size),
        int(inter_size),
        int(topk),
        int(num_experts),
        int(moe_tp_size),
        int(moe_ep_size),
        quant_mode.name,
        float(gpu["mem_bw"]),
        gpu.get("bfloat16_tc_flops"),
        gpu.get("fp8_tc_flops"),
        gpu.get("fp4_tc_flops"),
        config.level,
    )
