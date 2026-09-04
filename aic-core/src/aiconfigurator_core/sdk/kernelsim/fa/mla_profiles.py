"""Hardware-agnostic BF16 MLA engineering profiles."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MLA_PROFILE_VERSION = "2026-07-29.aic-mla-bf16-generic-profile-v1"


def _positive(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result


@dataclass(frozen=True)
class MlaPhaseProfile:
    fixed_overhead_us: float
    resource_efficiency: float
    wave_mode: str
    kv_task_cycles: float = 0.0

    def __post_init__(self) -> None:
        _positive("fixed_overhead_us", self.fixed_overhead_us)
        efficiency = _positive("resource_efficiency", self.resource_efficiency)
        if efficiency > 1:
            raise ValueError("resource_efficiency must be <= 1")
        wave_mode = self.wave_mode.strip().lower()
        if wave_mode not in {"linear", "rounded"}:
            raise ValueError("wave_mode must be linear or rounded")
        object.__setattr__(self, "wave_mode", wave_mode)
        cycles = float(self.kv_task_cycles)
        if not math.isfinite(cycles) or cycles < 0:
            raise ValueError("kv_task_cycles must be finite and non-negative")
        object.__setattr__(self, "kv_task_cycles", cycles)


@dataclass(frozen=True)
class MlaReferenceProfile:
    profile_id: str
    level: str
    description: str
    prefill_bf16: MlaPhaseProfile
    decode_bf16: MlaPhaseProfile
    source_scope: Mapping[str, Any] | None = None

    def phase_profile(self, config: str) -> MlaPhaseProfile:
        if config == "prefill_bf16":
            return self.prefill_bf16
        if config == "decode_bf16":
            return self.decode_bf16
        raise ValueError(f"unsupported MLA profile config {config!r}")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["profile_version"] = MLA_PROFILE_VERSION
        result["profiles"] = {
            "prefill_bf16": result.pop("prefill_bf16"),
            "decode_bf16": result.pop("decode_bf16"),
        }
        return result

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MlaReferenceProfile:
        if not isinstance(payload, Mapping):
            raise TypeError("MLA profile must be a JSON object")
        payload = dict(payload)
        payload.pop("profile_version", None)
        profiles = payload.pop("profiles", None)
        if not isinstance(profiles, Mapping):
            raise TypeError("MLA profile must contain a profiles object")
        unknown_profiles = sorted(set(profiles) - {"prefill_bf16", "decode_bf16"})
        if unknown_profiles:
            raise ValueError("unsupported MLA profile groups: " + ", ".join(unknown_profiles))
        missing = {"prefill_bf16", "decode_bf16"} - set(profiles)
        if missing:
            raise ValueError("missing MLA profile groups: " + ", ".join(sorted(missing)))
        allowed = {"profile_id", "level", "description", "source_scope"}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"unknown MLA profile fields: {', '.join(unknown)}")
        return cls(
            **payload,
            prefill_bf16=MlaPhaseProfile(**dict(profiles["prefill_bf16"])),
            decode_bf16=MlaPhaseProfile(**dict(profiles["decode_bf16"])),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> MlaReferenceProfile:
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


_PROFILES = {
    "low": MlaReferenceProfile(
        profile_id="mla-bf16-low-v1",
        level="low",
        description="Low-latency optimistic BF16 MLA engineering scenario.",
        prefill_bf16=MlaPhaseProfile(10.0, 0.90, "rounded"),
        decode_bf16=MlaPhaseProfile(10.0, 0.95, "linear", 4500.0),
    ),
    "standard": MlaReferenceProfile(
        profile_id="mla-bf16-standard-v1",
        level="standard",
        description="Central hardware-agnostic BF16 MLA engineering estimate.",
        prefill_bf16=MlaPhaseProfile(12.5, 0.70, "rounded"),
        decode_bf16=MlaPhaseProfile(12.5, 0.85, "linear", 7000.0),
    ),
    "high": MlaReferenceProfile(
        profile_id="mla-bf16-high-v1",
        level="high",
        description="High-latency conservative BF16 MLA engineering scenario.",
        prefill_bf16=MlaPhaseProfile(16.0, 0.50, "rounded"),
        decode_bf16=MlaPhaseProfile(16.0, 0.60, "linear", 9000.0),
    ),
}


def all_mla_profiles() -> tuple[MlaReferenceProfile, ...]:
    return tuple(_PROFILES[level] for level in ("low", "standard", "high"))


def get_mla_reference_profile(level: str = "standard") -> MlaReferenceProfile:
    normalized = level.strip().lower()
    try:
        return _PROFILES[normalized]
    except KeyError as error:
        raise ValueError("MLA estimate_level must be low, standard, or high") from error


__all__ = [
    "MLA_PROFILE_VERSION",
    "MlaPhaseProfile",
    "MlaReferenceProfile",
    "all_mla_profiles",
    "get_mla_reference_profile",
]
