"""KernelSim model for MiniMax-M3's BF16 Triton MSA index score.

This is intentionally separate from DSA's FP8 DeepGEMM-style index model.
The MSA score kernel includes the 128-token block-max reduction in its own
boundary.  TopK, cache updates and main attention are not part of this model.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

MSA_INDEX_MODEL_VERSION = "2026-08-11.msa-triton-index-v1"
MSA_INDEX_LIMITATION = (
    "Provisional MiniMax MSA Triton index model calibrated on one H100 SXM; "
    "BF16, head_dim=128 and block_size=128 only. It excludes TopK, cache "
    "updates, norm/RoPE and main attention and is not cross-GPU validated."
)


class MsaIndexModelWarning(UserWarning):
    """Warning for the deliberately limited MSA calibration."""


@dataclass(frozen=True)
class MsaIndexShape:
    phase: str
    batch_size: int
    query_length: int
    context_length: int
    index_heads: int = 4
    head_dim: int = 128
    block_size: int = 128
    dtype_bytes: int = 2

    def __post_init__(self) -> None:
        if self.phase not in {"prefill", "decode"}:
            raise ValueError("phase must be prefill or decode")
        for name in ("batch_size", "query_length", "context_length", "index_heads", "head_dim", "block_size", "dtype_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or int(value) <= 0:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        if self.query_length > self.context_length:
            raise ValueError("query_length must not exceed context_length")
        if self.phase == "decode" and self.query_length != 1:
            raise ValueError("decode MSA index recipe requires query_length=1")


@dataclass(frozen=True)
class MsaIndexParameters:
    prefill_floor_us: float = 2.829717731734192
    prefill_eta_mem: float = 0.5808596715912914
    decode_floor_us: float = 3.8241479943009493
    decode_eta_mem: float = 0.9238947076161088
    latency_scale: float = 1.0


_STANDARD = MsaIndexParameters()
_PROFILES = {
    "standard": _STANDARD,
    "low": MsaIndexParameters(latency_scale=0.85),
    "high": MsaIndexParameters(latency_scale=1.25),
}


def get_msa_index_parameters(level: str = "standard") -> MsaIndexParameters:
    try:
        return _PROFILES[level.strip().lower()]
    except KeyError as exc:
        raise ValueError("parameter_level must be standard, low, or high") from exc


@dataclass(frozen=True)
class MsaIndexEstimate:
    latency_us: float
    floor_us: float
    memory_us: float
    modeled_bytes: float
    tasks: int
    waves: int
    executed_heads: int
    warnings: tuple[str, ...]


def _next_pow2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def estimate_msa_index(shape: MsaIndexShape, *, sm_count: int, hbm_bandwidth_bytes_s: float,
                       parameter_level: str = "standard", params: MsaIndexParameters | None = None) -> MsaIndexEstimate:
    if sm_count <= 0 or hbm_bandwidth_bytes_s <= 0:
        raise ValueError("sm_count and hbm_bandwidth_bytes_s must be positive")
    if params is not None and parameter_level != "standard":
        raise ValueError("custom params require parameter_level=standard")
    p = params or get_msa_index_parameters(parameter_level)
    notes = [MSA_INDEX_LIMITATION]
    if (shape.head_dim, shape.block_size, shape.dtype_bytes) != (128, 128, 2):
        notes.append("shape is outside the calibrated BF16 head_dim=128/block_size=128 recipe")
    warnings.warn(MSA_INDEX_LIMITATION, MsaIndexModelWarning, stacklevel=2)

    if shape.phase == "prefill":
        tasks = shape.batch_size * shape.index_heads * math.ceil(shape.query_length / 128)
        waves = math.ceil(tasks / sm_count)
        modeled_bytes = waves * shape.context_length * shape.head_dim * shape.dtype_bytes * sm_count
        floor_us = p.prefill_floor_us
        memory_us = modeled_bytes / hbm_bandwidth_bytes_s / p.prefill_eta_mem * 1e6
        executed_heads = shape.index_heads
    else:
        executed_heads = max(16, _next_pow2(shape.index_heads))
        # The Triton decode recipe uses a fixed head execution tile and a
        # context-dependent block grid; this is not logical head-linear work.
        tasks = shape.batch_size
        waves = math.ceil(tasks / sm_count)
        blocks = math.ceil(shape.context_length / shape.block_size)
        modeled_bytes = (
            shape.batch_size * shape.context_length * shape.head_dim * shape.dtype_bytes
            + shape.batch_size * executed_heads * blocks * 4
        )
        floor_us = p.decode_floor_us
        memory_us = modeled_bytes / hbm_bandwidth_bytes_s / p.decode_eta_mem * 1e6
    scale = p.latency_scale
    return MsaIndexEstimate(
        latency_us=(floor_us + memory_us) * scale,
        floor_us=floor_us * scale,
        memory_us=memory_us * scale,
        modeled_bytes=modeled_bytes,
        tasks=tasks,
        waves=waves,
        executed_heads=executed_heads,
        warnings=tuple(notes),
    )
