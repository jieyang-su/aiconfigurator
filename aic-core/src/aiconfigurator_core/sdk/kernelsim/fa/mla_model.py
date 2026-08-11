"""BF16 DeepSeek-default MLA adapter over the shared FA2/FA3 core."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .mla_profiles import (
    MLA_PROFILE_VERSION,
    MlaReferenceProfile,
    get_mla_reference_profile,
)
from .mla_schema import MlaModelOptions, MlaRequest
from .model import MODEL_VERSION, estimate_attention
from .schema import AttentionShape, HardwareSpec, ModelOptions

MLA_MODEL_VERSION = "2026-07-29.aic-mla-bf16-roofline-v2"


@dataclass(frozen=True)
class MlaEstimateResult:
    model_version: str
    fa_core_model_version: str
    profile_version: str
    latency_us: float
    algorithm: str
    request: dict[str, Any]
    geometry: dict[str, Any]
    options: dict[str, Any]
    profile: dict[str, Any]
    components_us: dict[str, float]
    resource: dict[str, Any]
    scheduling: dict[str, Any]
    tiles: dict[str, Any]
    work: dict[str, Any]
    assumptions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["assumptions"] = list(self.assumptions)
        return payload


def _geometry(request: MlaRequest) -> dict[str, Any]:
    if request.phase == "prefill":
        return {
            "query_length": request.effective_query_length,
            "kv_length_total": request.sequence_length,
            "query_heads": request.local_query_heads,
            "kv_heads": request.local_query_heads,
            "head_dim": 192,
            "value_head_dim": 128,
            "kv_storage_dim": 320,
            "include_kv_cache_update": False,
        }
    return {
        "query_length": 1,
        "kv_length_total": request.sequence_length,
        "query_heads": request.local_query_heads,
        "kv_heads": 1,
        "head_dim": 576,
        "value_head_dim": 512,
        "kv_storage_dim": 576,
        "include_kv_cache_update": True,
    }


def estimate_mla(
    hardware: HardwareSpec,
    request: MlaRequest,
    options: MlaModelOptions | None = None,
    profile: MlaReferenceProfile | None = None,
) -> MlaEstimateResult:
    """Estimate one BF16 MLA operation with a generic or explicit profile."""
    options = options or MlaModelOptions()
    profile = profile or get_mla_reference_profile(options.estimate_level)
    phase_profile = profile.phase_profile(request.config)
    geometry = _geometry(request)
    shape = AttentionShape(
        batch_size=request.batch_size,
        query_length=geometry["query_length"],
        kv_length_total=geometry["kv_length_total"],
        query_heads=geometry["query_heads"],
        kv_heads=geometry["kv_heads"],
        head_dim=geometry["head_dim"],
        value_head_dim=geometry["value_head_dim"],
        kv_storage_dim=geometry["kv_storage_dim"],
        dtype="bf16",
        output_dtype="bf16",
        causal=True,
        page_size=64,
        label=request.label,
    )
    core = estimate_attention(
        hardware,
        shape,
        ModelOptions(
            algorithm=options.algorithm,
            mode="analytical",
            br=options.br,
            bc=options.bc,
            decode_splits=options.decode_splits,
            decode_query_threshold=options.decode_query_threshold,
            max_decode_splits=options.max_decode_splits,
            min_kv_tiles_per_split=options.min_kv_tiles_per_split,
            overlap_fraction=options.overlap_fraction,
            include_kv_cache_update=geometry["include_kv_cache_update"],
            assume_gqa_hbm_reuse=True,
            assume_gqa_l2_reuse=True,
            assume_query_tile_l2_reuse=True,
        ),
    )
    cta_count = core.scheduling["cta_count_hq_mapping_with_splits"]
    waves = cta_count / hardware.sm_count
    wave_rounding = math.ceil(waves) / waves if waves > 1 else 1.0
    raw_resource_us = core.resources_us["raw_resource_roof_us"]
    selected_resource_us = raw_resource_us
    if phase_profile.wave_mode == "rounded":
        selected_resource_us *= wave_rounding
    adjusted_resource_us = selected_resource_us / phase_profile.resource_efficiency
    task_service_us = 0.0
    if request.phase == "decode":
        task_service_us = (
            core.scheduling["fractional_task_waves"] * phase_profile.kv_task_cycles / hardware.clock_hz * 1e6
        )
    latency_us = phase_profile.fixed_overhead_us + adjusted_resource_us + task_service_us
    return MlaEstimateResult(
        model_version=MLA_MODEL_VERSION,
        fa_core_model_version=MODEL_VERSION,
        profile_version=MLA_PROFILE_VERSION,
        latency_us=latency_us,
        algorithm=options.algorithm,
        request=request.to_dict(),
        geometry=geometry,
        options=options.to_dict(),
        profile={
            "profile_id": profile.profile_id,
            "level": profile.level,
            "description": profile.description,
            "phase": asdict(phase_profile),
            "source_scope": profile.source_scope,
        },
        components_us={
            "fixed_overhead": phase_profile.fixed_overhead_us,
            "raw_resource": raw_resource_us,
            "selected_resource_before_efficiency": selected_resource_us,
            "resource_adjusted": adjusted_resource_us,
            "task_service": task_service_us,
        },
        resource={
            "raw_bottleneck": core.resources_us["raw_bottleneck"],
            "wave_mode": phase_profile.wave_mode,
            "cta_waves": waves,
            "wave_rounding_factor": wave_rounding,
            "resource_efficiency": phase_profile.resource_efficiency,
        },
        scheduling=core.scheduling,
        tiles=core.tiles,
        work=core.work,
        assumptions=(
            "DeepSeek-default MLA dimensions are fixed by phase.",
            "Only BF16 is supported; FP8 MLA profiles are intentionally unpublished.",
            "Prefill uses ideal query-tile KV L2 reuse and rounded CTA waves.",
            "Decode uses linear resource waves plus equivalent KV-task service.",
            "Profiles are engineering scenarios, not statistical confidence bounds.",
        ),
    )


__all__ = ["MLA_MODEL_VERSION", "MlaEstimateResult", "estimate_mla"]
