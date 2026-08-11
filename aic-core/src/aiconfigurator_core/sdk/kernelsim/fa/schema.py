"""Validated public inputs for the standalone FlashAttention roofline model."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2

_DTYPE_BYTES = {
    "fp8": 1,
    "float8": 1,
    "fp16": 2,
    "float16": 2,
    "bf16": 2,
    "bfloat16": 2,
    "fp32": 4,
    "float32": 4,
}

_DTYPE_CANONICAL = {
    "float8": "fp8",
    "float16": "fp16",
    "bfloat16": "bf16",
    "float32": "fp32",
}


def canonical_dtype(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("dtype must be a string")
    dtype = value.strip().lower()
    if dtype not in _DTYPE_BYTES:
        supported = ", ".join(sorted(_DTYPE_BYTES))
        raise ValueError(f"unsupported dtype {value!r}; supported aliases: {supported}")
    return _DTYPE_CANONICAL.get(dtype, dtype)


def dtype_bytes(value: str) -> int:
    return _DTYPE_BYTES[canonical_dtype(value)]


def _positive_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(frozen=True)
class HardwareSpec:
    """Hardware facts required by the analytical model.

    Rates are aggregate dense device rates, not sparse peaks. The L2 bandwidth
    is a requested/sustained modeling rate and must be supplied directly because
    product data sheets rarely publish it.
    """

    sm_count: int
    clock_hz: float
    shared_memory_per_sm_bytes: int
    l2_capacity_bytes: int
    l2_bandwidth_bytes_s: float
    hbm_bandwidth_bytes_s: float
    matrix_peak_flops: Mapping[str, float]
    vector_peak_flops: float
    exp_flop_equivalent: float = 35.0
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"hardware schema_version must be {SCHEMA_VERSION}, got {self.schema_version}")
        _positive_integer("sm_count", self.sm_count)
        _positive_integer("shared_memory_per_sm_bytes", self.shared_memory_per_sm_bytes)
        _positive_integer("l2_capacity_bytes", self.l2_capacity_bytes)
        for name in (
            "clock_hz",
            "l2_bandwidth_bytes_s",
            "hbm_bandwidth_bytes_s",
            "vector_peak_flops",
            "exp_flop_equivalent",
        ):
            _positive_number(name, getattr(self, name))
        if not isinstance(self.matrix_peak_flops, Mapping) or not self.matrix_peak_flops:
            raise ValueError("matrix_peak_flops must be a non-empty dtype-to-FLOP/s map")
        normalized: dict[str, float] = {}
        for dtype, peak in self.matrix_peak_flops.items():
            canonical = canonical_dtype(dtype)
            normalized[canonical] = _positive_number(f"matrix_peak_flops[{dtype!r}]", peak)
        object.__setattr__(self, "matrix_peak_flops", normalized)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HardwareSpec:
        if not isinstance(payload, Mapping):
            raise TypeError("hardware configuration must be a JSON object")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"unknown hardware fields: {', '.join(unknown)}")
        return cls(**dict(payload))

    @classmethod
    def from_json(cls, path: str | Path) -> HardwareSpec:
        return cls.from_dict(_read_json(path))

    def matrix_peak(self, dtype: str) -> float:
        canonical = canonical_dtype(dtype)
        try:
            return float(self.matrix_peak_flops[canonical])
        except KeyError as error:
            supported = ", ".join(sorted(self.matrix_peak_flops))
            raise ValueError(
                f"hardware has no dense matrix peak for {canonical}; configured precisions: {supported}"
            ) from error

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AttentionShape:
    """Logical attention shape.

    ``kv_length_total`` includes history and current query tokens. Q/K/V use a
    common effective dtype; mixed Q and KV-cache precision is intentionally not
    inferred from a misleading external ``attn_dtype`` label. ``head_dim`` is
    the QK reduction width. The optional value and storage widths support
    asymmetric values and shared-latent KV layouts.
    """

    batch_size: int
    query_length: int
    kv_length_total: int
    query_heads: int
    kv_heads: int
    head_dim: int
    dtype: str
    causal: bool = True
    output_dtype: str | None = None
    page_size: int | None = 64
    label: str = ""
    value_head_dim: int | None = None
    kv_storage_dim: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "batch_size",
            "query_length",
            "kv_length_total",
            "query_heads",
            "kv_heads",
            "head_dim",
        ):
            _positive_integer(name, getattr(self, name))
        if self.kv_length_total < self.query_length:
            raise ValueError("kv_length_total must be >= query_length")
        if self.query_heads % self.kv_heads:
            raise ValueError("query_heads must be divisible by kv_heads")
        if not isinstance(self.causal, bool):
            raise TypeError("causal must be boolean")
        object.__setattr__(self, "dtype", canonical_dtype(self.dtype))
        output = self.output_dtype
        if output is None:
            output = "bf16" if self.dtype == "fp8" else self.dtype
        object.__setattr__(self, "output_dtype", canonical_dtype(output))
        if self.page_size is not None:
            _positive_integer("page_size", self.page_size)
        value_dim = self.head_dim if self.value_head_dim is None else self.value_head_dim
        _positive_integer("value_head_dim", value_dim)
        object.__setattr__(self, "value_head_dim", value_dim)
        storage_dim = self.head_dim + value_dim if self.kv_storage_dim is None else self.kv_storage_dim
        _positive_integer("kv_storage_dim", storage_dim)
        object.__setattr__(self, "kv_storage_dim", storage_dim)

    @property
    def history_length(self) -> int:
        return self.kv_length_total - self.query_length

    @property
    def phase(self) -> str:
        if self.history_length == 0:
            return "prefill"
        return "decode" if self.query_length == 1 else "prefix"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AttentionShape:
        if not isinstance(payload, Mapping):
            raise TypeError("attention shape must be a JSON object")
        payload = dict(payload)
        declared_history = payload.pop("history_length", None)
        declared_phase = payload.pop("phase", None)
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"unknown shape fields: {', '.join(unknown)}")
        shape = cls(**payload)
        if declared_history is not None and declared_history != shape.history_length:
            raise ValueError("declared history_length does not match kv_length_total-query_length")
        if declared_phase is not None and declared_phase != shape.phase:
            raise ValueError("declared phase does not match the shape")
        return shape

    @classmethod
    def many_from_json(cls, path: str | Path) -> list[AttentionShape]:
        payload = _read_json(path)
        if isinstance(payload, Mapping) and "shapes" in payload:
            payload = payload["shapes"]
        if isinstance(payload, Mapping):
            payload = [payload]
        if not isinstance(payload, list) or not payload:
            raise ValueError("shape input must be an object, a list, or {'shapes': [...]}")
        return [cls.from_dict(item) for item in payload]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["history_length"] = self.history_length
        result["phase"] = self.phase
        return result


@dataclass(frozen=True)
class ModelOptions:
    """Execution assumptions kept separate from hardware facts."""

    algorithm: str = "fa2"
    mode: str = "profiled"
    estimate_level: str = "standard"
    br: int | None = None
    bc: int | None = None
    decode_splits: str | int = "auto"
    decode_query_threshold: int = 16
    max_decode_splits: int = 128
    min_kv_tiles_per_split: int = 4
    account_for_parallelism: bool = True
    assume_gqa_hbm_reuse: bool = True
    assume_gqa_l2_reuse: bool = True
    store_lse: bool = False
    include_kv_cache_update: bool = True
    overlap_fraction: float | None = None
    assume_query_tile_l2_reuse: bool = False

    def __post_init__(self) -> None:
        algorithm = self.algorithm.strip().lower()
        if algorithm not in {"fa2", "fa3"}:
            raise ValueError("algorithm must be fa2 or fa3")
        object.__setattr__(self, "algorithm", algorithm)
        mode = self.mode.strip().lower()
        if mode not in {"analytical", "profiled"}:
            raise ValueError("mode must be analytical or profiled")
        object.__setattr__(self, "mode", mode)
        level = self.estimate_level.strip().lower()
        if level not in {"standard", "high", "low"}:
            raise ValueError("estimate_level must be standard, high, or low")
        object.__setattr__(self, "estimate_level", level)
        for name in ("br", "bc"):
            value = getattr(self, name)
            if value is not None:
                _positive_integer(name, value)
        if self.decode_splits != "auto":
            _positive_integer("decode_splits", self.decode_splits)
        for name in (
            "decode_query_threshold",
            "max_decode_splits",
            "min_kv_tiles_per_split",
        ):
            _positive_integer(name, getattr(self, name))
        if self.overlap_fraction is not None:
            overlap = float(self.overlap_fraction)
            if not math.isfinite(overlap) or not 0 <= overlap <= 1:
                raise ValueError("overlap_fraction must be in [0, 1]")
            object.__setattr__(self, "overlap_fraction", overlap)
        for name in (
            "account_for_parallelism",
            "assume_gqa_hbm_reuse",
            "assume_gqa_l2_reuse",
            "store_lse",
            "include_kv_cache_update",
            "assume_query_tile_l2_reuse",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "SCHEMA_VERSION",
    "AttentionShape",
    "HardwareSpec",
    "ModelOptions",
    "canonical_dtype",
    "dtype_bytes",
]
