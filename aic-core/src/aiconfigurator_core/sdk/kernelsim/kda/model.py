"""Analytical Kimi Delta Attention (KDA) core and module composition model.

This experiment intentionally models only KDA-specific kernels. GEMM,
elementwise, and collective latency are inputs to ``estimate_kda_module`` and
remain owned by their existing AIC models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, isfinite
from typing import Literal

MODEL_VERSION = "2026-08-10.kimi-k3-kda-core-v4"
Phase = Literal["prefill", "decode", "verify"]
Route = Literal["sglang", "vllm"]
Confidence = Literal["high", "medium", "low"]
CentralMode = Literal["v2", "v4"]


@dataclass(frozen=True)
class KdaHardware:
    """Minimum hardware contract. Rates use bytes/s and FLOPs/s."""

    hbm_bandwidth_bytes_s: float
    bf16_peak_flops_s: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite value")


@dataclass(frozen=True)
class KdaShape:
    """One rank-local KDA request. ``seq_len`` is required for prefill/verify."""

    batch_size: int
    local_heads: int
    head_dim: int = 128
    conv_width: int = 4
    seq_len: int = 1
    draft_tokens: int = 0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name == "draft_tokens":
                if int(value) != value or value < 0:
                    raise ValueError("draft_tokens must be a nonnegative integer")
                continue
            if int(value) != value or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.draft_tokens > self.seq_len:
            raise ValueError("draft_tokens cannot exceed seq_len")

    @property
    def projection_width(self) -> int:
        return self.local_heads * self.head_dim

    @property
    def state_bytes_per_request(self) -> int:
        return self.local_heads * self.head_dim * self.head_dim * 4


@dataclass(frozen=True)
class KdaProfile:
    """Generic, backend-neutral scenarios rather than hardware lookup profiles."""

    level: str
    hbm_efficiency: float
    compute_efficiency: float
    conv_launch_us: float
    vllm_prefill_conv_host_us: float
    scan_launch_us: float
    chunk_hbm_efficiency: float
    recurrent_launch_us: float
    verify_launch_us: float
    fused_decode_launch_us: float

    def __post_init__(self) -> None:
        if self.level not in {"low", "standard", "high"}:
            raise ValueError("level must be low, standard, or high")
        if not 0 < self.hbm_efficiency <= 1 or not 0 < self.compute_efficiency <= 1:
            raise ValueError("efficiencies must be in (0, 1]")
        if not 0 < self.chunk_hbm_efficiency <= 1:
            raise ValueError("chunk_hbm_efficiency must be in (0, 1]")
        for name in (
            "conv_launch_us",
            "vllm_prefill_conv_host_us",
            "scan_launch_us",
            "recurrent_launch_us",
            "verify_launch_us",
            "fused_decode_launch_us",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")


_PROFILES = {
    # chunk_kda is a materialized, multi-kernel Triton pipeline. Its effective
    # bandwidth is lower than the single fused kernels sharing hbm_efficiency.
    "low": KdaProfile("low", 0.88, 0.75, 1.5, 180.0, 7.0, 0.42, 4.5, 5.0, 5.0),
    # vLLM's prefill conv collector carries a roughly 420us host metadata
    # boundary across three serial Q/K/V calls (~140us/call). This is a
    # collector-boundary term, not a GPU kernel floor and is only used on the
    # explicit vLLM prefill route.
    "standard": KdaProfile("standard", 0.68, 0.55, 2.0, 140.0, 10.0, 0.30, 6.0, 7.0, 7.0),
    "high": KdaProfile("high", 0.48, 0.38, 3.0, 280.0, 15.0, 0.22, 9.0, 11.0, 11.0),
}


def get_profile(level: str = "standard") -> KdaProfile:
    try:
        return _PROFILES[level.strip().lower()]
    except KeyError as error:
        raise ValueError("level must be low, standard, or high") from error


@dataclass(frozen=True)
class KernelEstimate:
    kernel: str
    phase: Phase
    latency_us: float
    launch_us: float
    memory_us: float
    compute_us: float
    roofline_branch: str
    flops: int
    hbm_read_bytes: int
    hbm_write_bytes: int
    fp32_state_bytes: int
    metadata: dict[str, int | str]

    @property
    def hbm_bytes(self) -> int:
        return self.hbm_read_bytes + self.hbm_write_bytes


@dataclass(frozen=True)
class ExistingModuleTerms:
    """Latency supplied by existing AIC GEMM, elementwise, and comm components."""

    norm_us: float = 0.0
    qkvg_projection_us: float = 0.0
    forget_gate_projections_us: float = 0.0
    output_norm_us: float = 0.0
    output_projection_us: float = 0.0
    tp_allreduce_us: float = 0.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a nonnegative finite value")

    @property
    def total_us(self) -> float:
        return sum(asdict(self).values())


@dataclass(frozen=True)
class ModuleEstimate:
    phase: Phase
    route: Route
    latency_us: float
    existing_components_us: float
    kda_core_us: float
    core_kernels: tuple[KernelEstimate, ...]
    folded_existing_terms: tuple[str, ...]


@dataclass(frozen=True)
class KdaPlanningEstimate:
    """No-GPU planning result with an explicit uncertainty contract."""

    central_us: float
    lower_us: float
    upper_us: float
    confidence: Confidence
    assumptions: tuple[str, ...]
    workload_sensitive_us: float | None = None


def _rate_time_us(work: int, rate: float, efficiency: float) -> float:
    return work / (rate * efficiency) * 1e6


def _estimate(
    *,
    kernel: str,
    phase: Phase,
    launch_us: float,
    flops: int,
    read_bytes: int,
    write_bytes: int,
    state_bytes: int,
    hardware: KdaHardware,
    profile: KdaProfile,
    metadata: dict[str, int | str],
) -> KernelEstimate:
    memory_us = _rate_time_us(read_bytes + write_bytes, hardware.hbm_bandwidth_bytes_s, profile.hbm_efficiency)
    compute_us = _rate_time_us(flops, hardware.bf16_peak_flops_s, profile.compute_efficiency)
    branch = "memory" if memory_us >= compute_us else "compute"
    return KernelEstimate(
        kernel=kernel,
        phase=phase,
        latency_us=launch_us + max(memory_us, compute_us),
        launch_us=launch_us,
        memory_us=memory_us,
        compute_us=compute_us,
        roofline_branch=branch,
        flops=flops,
        hbm_read_bytes=read_bytes,
        hbm_write_bytes=write_bytes,
        fp32_state_bytes=state_bytes,
        metadata=metadata,
    )


def estimate_conv(
    shape: KdaShape, phase: Phase, hardware: KdaHardware, profile: KdaProfile, route: Route = "sglang"
) -> KernelEstimate:
    """KDA Q/K/V causal convolution. Context is three serial calls; decode is packed."""
    tokens = shape.batch_size * (shape.seq_len if phase in {"prefill", "verify"} else 1)
    width = shape.projection_width
    channels = 3 * width
    # Sliding-window tiles reuse most K-1 taps on chip. Treating every tap as
    # a new HBM read overstates long-context SGLang measurements. The 1.8x
    # factor keeps compulsory input reads plus tile-edge reloads explicit.
    read_bytes = int(tokens * channels * 2 * 1.8)
    write_bytes = tokens * channels * 2
    flops = tokens * channels * shape.conv_width * 2
    return _estimate(
        kernel="causal_conv1d_fn_qkv3" if phase == "prefill" else "causal_conv1d_update",
        phase=phase,
        # The collector's prefill row wraps three independent Q/K/V calls.
        # vLLM deliberately disables graph capture here because each call
        # performs host-side metadata transfer; retain that timed boundary.
        launch_us=(
            3 * (profile.conv_launch_us + (profile.vllm_prefill_conv_host_us if route == "vllm" else 0.0))
            if phase == "prefill"
            # Packed update has a materially higher service floor than the
            # prefill GPU-core call; verify uses a smaller but nonzero floor.
            # These multipliers are route-neutral and are validated against
            # the kernel family, not fitted to a hardware name.
            else profile.conv_launch_us * (2.5 if phase == "decode" else 1.5)
        ),
        flops=flops,
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        state_bytes=0,
        hardware=hardware,
        profile=profile,
        metadata={"tokens": tokens, "conv_channels": channels, "conv_calls": 3 if phase == "prefill" else 1},
    )


def estimate_scan(shape: KdaShape, hardware: KdaHardware, profile: KdaProfile, route: Route) -> KernelEstimate:
    """Prefill chunk scan. Partial chunks pay one state round trip too."""
    tokens = shape.batch_size * shape.seq_len
    width = shape.projection_width
    state = shape.state_bytes_per_request
    chunks = ceil(shape.seq_len / 64)
    chunk_state_bytes = chunks * state * shape.batch_size
    read_bytes = tokens * 4 * width * 2 + state * shape.batch_size + chunk_state_bytes
    write_bytes = tokens * width * 2 + state * shape.batch_size + chunk_state_bytes
    if route == "sglang":
        # chunk_kda_fwd materializes A/Aqk (T,H,64) and the w/u/kg/v_new
        # streams before the final output kernel. A/Aqk use FP32 at creation;
        # the other streams are BF16. These are real global-memory boundaries,
        # not a fitted shape correction.
        matrix_elements = tokens * shape.local_heads * 64
        read_bytes += matrix_elements * 10 + tokens * width * 20
        write_bytes += matrix_elements * 10 + tokens * width * 20
    # Delta-rule state update plus output application: eight D^2 FLOPs per head-token.
    flops = tokens * shape.local_heads * shape.head_dim * shape.head_dim * 8
    return _estimate(
        kernel="flashkda_fwd" if route == "vllm" else "chunk_kda",
        phase="prefill",
        launch_us=profile.scan_launch_us,
        flops=flops,
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        state_bytes=state,
        hardware=hardware,
        profile=KdaProfile(
            profile.level,
            profile.chunk_hbm_efficiency if route == "sglang" else profile.hbm_efficiency,
            profile.compute_efficiency,
            profile.conv_launch_us,
            profile.vllm_prefill_conv_host_us,
            profile.scan_launch_us,
            profile.chunk_hbm_efficiency,
            profile.recurrent_launch_us,
            profile.verify_launch_us,
            profile.fused_decode_launch_us,
        ),
        metadata={
            "tokens": tokens,
            "chunk_size": 64,
            "chunks_per_request": chunks,
            "materialized_workspace": route == "sglang",
        },
    )


def estimate_recurrence(
    shape: KdaShape, phase: Phase, hardware: KdaHardware, profile: KdaProfile, route: Route
) -> KernelEstimate:
    """Packed decode recurrence or chain verify recurrence, excluding convolution."""
    if phase not in {"decode", "verify"}:
        raise ValueError("recurrence supports decode or verify only")
    per_request_tokens = shape.draft_tokens if phase == "verify" else 1
    tokens = shape.batch_size * per_request_tokens
    width = shape.projection_width
    state = shape.state_bytes_per_request
    read_bytes = tokens * 4 * width * 2 + state * shape.batch_size
    write_bytes = tokens * width * 2 + state * (tokens if phase == "verify" else shape.batch_size)
    flops = tokens * shape.local_heads * shape.head_dim * shape.head_dim * 8
    kernel = (
        "fused_recurrent_kda"
        if route == "vllm" and phase == "verify"
        else ("fused_sigmoid_gating_delta_rule_update" if phase == "verify" else "fused_recurrent_kda_packed_decode")
    )
    launch_us = (3.0 if phase == "decode" else 5.0) if route == "vllm" else (5.0 if phase == "decode" else 5.5)
    return _estimate(
        kernel=kernel,
        phase=phase,
        # Independent recurrent tables have route-specific service boundaries;
        # fused callers use their own composite launch floor.
        launch_us=launch_us,
        flops=flops,
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        state_bytes=state,
        hardware=hardware,
        profile=profile,
        metadata={"tokens": tokens, "draft_tokens": per_request_tokens},
    )


def estimate_fused_decode(shape: KdaShape, hardware: KdaHardware, profile: KdaProfile) -> KernelEstimate:
    """One fused decode kernel: packed conv, recurrence, and gated output norm."""
    conv = estimate_conv(shape, "decode", hardware, profile, "vllm")
    recurrence = estimate_recurrence(shape, "decode", hardware, profile, "vllm")
    tokens = shape.batch_size
    output_norm_bytes = tokens * shape.projection_width * 6
    output_norm_flops = tokens * shape.projection_width * 5
    return _estimate(
        kernel="fused_kda_decode",
        phase="decode",
        launch_us=profile.fused_decode_launch_us,
        flops=conv.flops + recurrence.flops + output_norm_flops,
        read_bytes=conv.hbm_read_bytes + recurrence.hbm_read_bytes + output_norm_bytes,
        write_bytes=conv.hbm_write_bytes + recurrence.hbm_write_bytes,
        state_bytes=shape.state_bytes_per_request,
        hardware=hardware,
        profile=profile,
        metadata={"fuses": "conv,recurrent,gated_output_norm", "tokens": tokens},
    )


def estimate_fused_verify(shape: KdaShape, hardware: KdaHardware, profile: KdaProfile, route: Route) -> KernelEstimate:
    """One DSPARK verify kernel covering conv update and chain recurrence."""
    if shape.draft_tokens <= 0:
        raise ValueError("fused verify requires draft_tokens > 0")
    conv = estimate_conv(shape, "verify", hardware, profile, route)
    recurrence = estimate_recurrence(shape, "verify", hardware, profile, route)
    # A fused kernel pays one verify service floor; its semantic traffic remains
    # the sum of the two constituent paths.
    return _estimate(
        kernel="fused_kda_decode_mtp_dspark",
        phase="verify",
        launch_us=profile.verify_launch_us,
        flops=conv.flops + recurrence.flops,
        read_bytes=conv.hbm_read_bytes + recurrence.hbm_read_bytes,
        write_bytes=conv.hbm_write_bytes + recurrence.hbm_write_bytes,
        state_bytes=shape.state_bytes_per_request,
        hardware=hardware,
        profile=profile,
        metadata={"fuses": "conv,recurrent", "tokens": shape.batch_size * shape.draft_tokens},
    )


def estimate_kda_core(
    shape: KdaShape,
    phase: Phase,
    route: Route,
    hardware: KdaHardware,
    profile: KdaProfile | None = None,
    *,
    fused_decode: bool = False,
) -> tuple[KernelEstimate, ...]:
    """Return the actual KDA core kernel sequence for one phase."""
    profile = profile or get_profile()
    if phase == "prefill":
        return (estimate_conv(shape, phase, hardware, profile, route), estimate_scan(shape, hardware, profile, route))
    if phase == "decode":
        if fused_decode:
            return (estimate_fused_decode(shape, hardware, profile),)
        return (
            estimate_conv(shape, phase, hardware, profile, route),
            estimate_recurrence(shape, phase, hardware, profile, route),
        )
    if shape.draft_tokens <= 0:
        raise ValueError("verify requires draft_tokens > 0")
    if fused_decode:
        return (estimate_fused_verify(shape, hardware, profile, route),)
    return (
        estimate_conv(shape, phase, hardware, profile, route),
        estimate_recurrence(shape, phase, hardware, profile, route),
    )


def estimate_kda_module(
    shape: KdaShape,
    phase: Phase,
    route: Route,
    hardware: KdaHardware,
    existing: ExistingModuleTerms,
    profile: KdaProfile | None = None,
    *,
    fused_decode: bool = False,
) -> ModuleEstimate:
    """Compose KDA core with latency returned by existing AIC components."""
    kernels = estimate_kda_core(shape, phase, route, hardware, profile, fused_decode=fused_decode)
    folded: tuple[str, ...] = ()
    existing_us = existing.total_us
    if phase == "decode" and fused_decode:
        # This core already includes the gated output norm. Do not charge it twice.
        existing_us -= existing.output_norm_us
        folded = ("output_norm_us",)
    core_us = sum(kernel.latency_us for kernel in kernels)
    return ModuleEstimate(
        phase=phase,
        route=route,
        latency_us=existing_us + core_us,
        existing_components_us=existing_us,
        kda_core_us=core_us,
        core_kernels=kernels,
        folded_existing_terms=folded,
    )


def estimate_kda_plan(
    shape: KdaShape,
    phase: Phase,
    route: Route,
    hardware: KdaHardware,
    existing: ExistingModuleTerms | None = None,
    *,
    fused_decode: bool = False,
    base_waves: float | None = None,
    central_mode: CentralMode = "v2",
) -> KdaPlanningEstimate:
    """Return a bounded no-GPU estimate rather than a falsely exact latency.

    ``base_waves`` is optional execution-ledger evidence for SGLang prefill.
    When provided, ``workload_sensitive_us`` reports the experimental v4
    chunk correction separately and never replaces the v2 central estimate.
    """

    def total(profile: KdaProfile) -> float:
        if existing is None:
            return sum(
                item.latency_us
                for item in estimate_kda_core(shape, phase, route, hardware, profile, fused_decode=fused_decode)
            )
        return estimate_kda_module(
            shape, phase, route, hardware, existing, profile, fused_decode=fused_decode
        ).latency_us

    lower = total(get_profile("low"))
    v2_central = total(get_profile("standard"))
    central = v2_central
    upper = total(get_profile("high"))
    workload_sensitive = None
    assumptions = ["v2 central roofline", "existing GEMM/norm/communication terms are externally supplied"]
    confidence: Confidence = "medium"
    if route == "vllm" and phase == "prefill":
        confidence = "low"
        assumptions.append("vLLM prefill convolution collector includes host metadata transfer")
    elif fused_decode and phase == "verify":
        confidence = "high"
        assumptions.append("single fused KDA core boundary")
    elif fused_decode and phase == "decode":
        assumptions.append("fused decode has one core boundary but moderate cross-hardware residual")
    elif route == "sglang" and phase == "prefill":
        assumptions.append("chunk_kda materialized workspace is modeled; runtime autotune remains unknown")
    if base_waves is not None:
        if base_waves <= 0:
            raise ValueError("base_waves must be positive when provided")
        if route == "sglang" and phase == "prefill":
            # Local import keeps the v2 API independent from the experimental
            # v4 implementation and its offline fitting utilities.
            from .v4 import V4Saturation, predict_chunk_v4

            v4_scan = predict_chunk_v4(
                shape,
                hardware,
                get_profile("standard"),
                base_waves,
                V4Saturation(0.04, 0.48, 32.0),
            )
            v2_scan = estimate_scan(shape, hardware, get_profile("standard"), "sglang").latency_us
            workload_sensitive = v2_central - v2_scan + v4_scan
            assumptions.append("v4 workload-sensitive value is experimental, not the central estimate")
    if central_mode == "v4":
        if workload_sensitive is None:
            raise ValueError("v4 central mode requires SGLang prefill with positive base_waves")
        central = workload_sensitive
        assumptions[-1] = "v4 workload-sensitive value selected as the no-GPU planning central scenario"
        assumptions.append("v4 is not approved as the production AIC SOL fallback")
    return KdaPlanningEstimate(
        central_us=central,
        lower_us=min(lower, central, upper),
        upper_us=max(lower, central, upper),
        confidence=confidence,
        assumptions=tuple(assumptions),
        workload_sensitive_us=workload_sensitive,
    )
