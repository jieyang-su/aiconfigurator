"""AIC adapter for the Kimi-K3 KDA kernel-level analytical model.

The adapter deliberately accepts one ``KDAKernel`` query at a time.  It does
not compose a KDA layer: GEMM, norm, and collective terms remain separate AIC
operations and preserve the model graph's existing accounting boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from .model import (
    KdaHardware,
    KdaShape,
    KernelEstimate,
    estimate_conv,
    estimate_fused_decode,
    estimate_fused_verify,
    estimate_recurrence,
    estimate_scan,
    get_profile,
)
from .v4 import V4Saturation, predict_chunk_v4

# These are measured hardware capabilities, not fitted latency coefficients.
# L40S intentionally remains absent: its v2 path must not invent an SM count.
_VERIFIED_SM_COUNTS = {
    "b200_sxm": 148,
    "b300_sxm": 148,
    "gb200": 148,
    "gb300": 148,
    "h100_sxm": 132,
    "h200_sxm": 132,
    "rtx_pro_6000_server": 188,
}


@dataclass(frozen=True)
class KdaAnalyticalEstimate:
    """One core-kernel result, in microseconds, with planning provenance."""

    central_us: float
    lower_us: float
    upper_us: float
    confidence: str
    policy: str
    kernel: str
    roofline_branch: str


def _base_waves(shape: KdaShape, sm_count: int) -> float:
    """Static CTA ledger for the fixed 64-token chunk_kda pipeline."""
    chunks = ceil(shape.seq_len / 64)
    heads = shape.local_heads
    batch = shape.batch_size
    programs = (
        chunks * batch * heads
        + chunks * 16 * batch * heads
        + chunks * 4 * batch * heads
        + ceil(shape.seq_len / 16) * batch * heads
        + chunks * batch * heads
        + chunks * batch * heads
        + ceil(shape.head_dim / 32) * batch * heads
        + ceil(shape.head_dim / 64) * chunks * batch * heads
    )
    return programs / sm_count


def _kernel_estimate(
    *,
    kernel_source: str,
    phase: str,
    shape: KdaShape,
    hardware: KdaHardware,
    route: str,
    profile_level: str,
) -> KernelEstimate:
    profile = get_profile(profile_level)
    if kernel_source in {"causal_conv1d_fn_qkv3", "causal_conv1d_update"}:
        return estimate_conv(
            shape, {"context": "prefill", "generation": "decode", "verify": "verify"}[phase], hardware, profile, route
        )
    if kernel_source in {"chunk_kda", "chunk_kda_with_fused_gate", "flashkda_fwd"}:
        return estimate_scan(shape, hardware, profile, route)
    if kernel_source in {"fused_kda_decode", "kda_fused_decode"}:
        return estimate_fused_decode(shape, hardware, profile)
    if kernel_source == "fused_kda_decode_mtp_dspark":
        return estimate_fused_verify(shape, hardware, profile, route)
    if kernel_source in {
        "fused_recurrent_kda_packed_decode",
        "fused_sigmoid_gating_delta_rule_update",
        "fused_recurrent_kda",
    }:
        model_phase = "verify" if phase == "verify" else "decode"
        return estimate_recurrence(shape, model_phase, hardware, profile, route)
    raise ValueError(f"unsupported analytical KDA kernel source {kernel_source!r}")


def estimate_kernel(
    *,
    system: str,
    gpu: dict,
    backend: str,
    kernel_source: str,
    phase: str,
    batch_size: int,
    seq_len: int | None,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    d_conv: int,
    level: str = "standard",
) -> KdaAnalyticalEstimate:
    """Estimate a single AIC KDA kernel query without reading a perf table."""
    if backend not in {"sglang", "vllm"}:
        raise ValueError(f"ANALYTICAL KDA supports sglang or vllm, got {backend!r}")
    if phase not in {"context", "generation", "verify"}:
        raise ValueError(f"unsupported KDA phase {phase!r}")
    per_request_tokens = int(seq_len or 1)
    shape = KdaShape(
        batch_size=int(batch_size),
        local_heads=int(num_v_heads),
        head_dim=int(head_v_dim),
        conv_width=int(d_conv),
        seq_len=per_request_tokens,
        draft_tokens=per_request_tokens if phase == "verify" else 0,
    )
    if head_k_dim != head_v_dim:
        raise ValueError("ANALYTICAL KDA currently requires equal K/V head dimensions")
    hardware = KdaHardware(float(gpu["mem_bw"]), float(gpu["bfloat16_tc_flops"]))
    central = _kernel_estimate(
        kernel_source=kernel_source,
        phase=phase,
        shape=shape,
        hardware=hardware,
        route=backend,
        profile_level=level,
    )
    low = _kernel_estimate(
        kernel_source=kernel_source,
        phase=phase,
        shape=shape,
        hardware=hardware,
        route=backend,
        profile_level="low",
    )
    high = _kernel_estimate(
        kernel_source=kernel_source,
        phase=phase,
        shape=shape,
        hardware=hardware,
        route=backend,
        profile_level="high",
    )
    policy = "v2_fallback"
    central_us = central.latency_us
    confidence = "medium"
    if kernel_source in {"chunk_kda", "chunk_kda_with_fused_gate"} and phase == "context" and backend == "sglang":
        sm_count = _VERIFIED_SM_COUNTS.get(system)
        if sm_count is not None:
            central_us = predict_chunk_v4(
                shape, hardware, get_profile(level), _base_waves(shape, sm_count), V4Saturation(0.04, 0.48, 32.0)
            )
            policy = "v4_central_low_confidence" if system == "rtx_pro_6000_server" else "v4_central"
            confidence = "low" if system == "rtx_pro_6000_server" else "medium"
    return KdaAnalyticalEstimate(
        central_us=central_us,
        lower_us=min(low.latency_us, central_us, high.latency_us),
        upper_us=max(low.latency_us, central_us, high.latency_us),
        confidence=confidence,
        policy=policy,
        kernel=central.kernel,
        roofline_branch=central.roofline_branch,
    )
