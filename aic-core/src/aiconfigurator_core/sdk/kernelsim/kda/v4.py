"""Experimental v4 workload-saturation candidate for chunk_kda only."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

from .model import KdaHardware, KdaProfile, KdaShape, estimate_scan


@dataclass(frozen=True)
class V4Saturation:
    eta_floor: float
    eta_ceiling: float
    saturation_waves: float

    def __post_init__(self) -> None:
        if not (0 < self.eta_floor <= self.eta_ceiling <= 1):
            raise ValueError("efficiency bounds must satisfy 0 < floor <= ceiling <= 1")
        if self.saturation_waves <= 0:
            raise ValueError("saturation_waves must be positive")

    def efficiency(self, base_waves: float) -> float:
        fraction = 1.0 - exp(-max(base_waves, 0.0) / self.saturation_waves)
        return self.eta_floor + (self.eta_ceiling - self.eta_floor) * fraction


def predict_chunk_v4(
    shape: KdaShape,
    hardware: KdaHardware,
    profile: KdaProfile,
    base_waves: float,
    saturation: V4Saturation,
) -> float:
    """Adjust only chunk_kda's memory branch; v2 remains the reference model."""
    estimate = estimate_scan(shape, hardware, profile, "sglang")
    eta = saturation.efficiency(base_waves)
    memory_us = estimate.memory_us * profile.chunk_hbm_efficiency / eta
    return estimate.launch_us + max(memory_us, estimate.compute_us)
