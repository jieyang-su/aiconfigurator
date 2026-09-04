"""Hardware-agnostic empirical parameter levels for the FA roofline model."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

PROFILE_VERSION = "2026-07-24.aic-fa-generic-profile-v2"


@dataclass(frozen=True)
class ReferenceProfile:
    """One generic latency level shared by all hardware and dtypes.

    ``high`` means a high-latency conservative estimate; ``low`` means a
    low-latency optimistic estimate. These levels are engineering scenarios,
    not statistical confidence bounds.
    """

    level: str
    fixed_overhead_us: float
    hbm_efficiency: float
    l2_efficiency: float
    compute_efficiency: float
    kv_task_cycles: float
    description: str

    def to_dict(self) -> dict:
        result = asdict(self)
        result["profile_version"] = PROFILE_VERSION
        return result


# Rounded scenarios synthesized from the A100/H100 study. They intentionally
# retain no GPU, architecture, backend, phase or dtype keys.
_PROFILES = {
    "standard": ReferenceProfile(
        level="standard",
        fixed_overhead_us=12.5,
        hbm_efficiency=0.72,
        l2_efficiency=0.80,
        compute_efficiency=0.55,
        kv_task_cycles=6000.0,
        description="Central engineering estimate for general deployment.",
    ),
    "high": ReferenceProfile(
        level="high",
        fixed_overhead_us=16.0,
        hbm_efficiency=0.50,
        l2_efficiency=0.60,
        compute_efficiency=0.35,
        kv_task_cycles=8000.0,
        description="High-latency conservative scenario.",
    ),
    "low": ReferenceProfile(
        level="low",
        fixed_overhead_us=10.0,
        hbm_efficiency=0.92,
        l2_efficiency=0.95,
        compute_efficiency=0.80,
        kv_task_cycles=3500.0,
        description="Low-latency optimistic scenario.",
    ),
}


def all_profiles() -> Iterable[ReferenceProfile]:
    return tuple(_PROFILES[level] for level in ("standard", "high", "low"))


def get_reference_profile(level: str = "standard") -> ReferenceProfile:
    normalized = level.strip().lower()
    try:
        return _PROFILES[normalized]
    except KeyError as error:
        raise ValueError("estimate_level must be standard, high, or low") from error


__all__ = [
    "PROFILE_VERSION",
    "ReferenceProfile",
    "all_profiles",
    "get_reference_profile",
]
