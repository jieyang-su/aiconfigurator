"""Validated BF16 MLA operator inputs and model options."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .schema import ModelOptions, canonical_dtype

MLA_SCHEMA_VERSION = 2
FP8_UNSUPPORTED_MESSAGE = (
    "FP8 MLA is not supported by the production model: the available SGLang "
    "0.5.9 measurements show substantial backend performance degradation, so "
    "no transferable FP8 profile is published"
)


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class MlaRequest:
    """DeepSeek-default MLA request.

    ``sequence_length`` is the total attended KV length. ``query_length`` is
    optional for prefill and defaults to ``sequence_length``; decode always
    uses one query token. ``local_query_heads`` is the post-TP head count seen
    by one GPU.
    """

    phase: str
    batch_size: int
    local_query_heads: int
    sequence_length: int
    dtype: str = "bf16"
    label: str = ""
    query_length: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.phase, str):
            raise TypeError("phase must be a string")
        phase = self.phase.strip().lower()
        if phase not in {"prefill", "decode"}:
            raise ValueError("phase must be prefill or decode")
        object.__setattr__(self, "phase", phase)
        for name in ("batch_size", "local_query_heads", "sequence_length"):
            _positive_integer(name, getattr(self, name))
        if self.query_length is not None:
            _positive_integer("query_length", self.query_length)
            if phase == "prefill" and self.query_length > self.sequence_length:
                raise ValueError("prefill query_length must be <= sequence_length")
            if phase == "decode" and self.query_length != 1:
                raise ValueError("decode query_length must be 1 when provided")
        dtype = canonical_dtype(self.dtype)
        if dtype == "fp8":
            raise ValueError(FP8_UNSUPPORTED_MESSAGE)
        if dtype != "bf16":
            raise ValueError("MLA production profiles currently support BF16 only")
        object.__setattr__(self, "dtype", dtype)
        if not isinstance(self.label, str):
            raise TypeError("label must be a string")

    @property
    def config(self) -> str:
        return f"{self.phase}_bf16"

    @property
    def effective_query_length(self) -> int:
        if self.phase == "decode":
            return 1
        return self.sequence_length if self.query_length is None else self.query_length

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MlaRequest:
        if not isinstance(payload, Mapping):
            raise TypeError("MLA request must be a JSON object")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"unknown MLA request fields: {', '.join(unknown)}")
        return cls(**dict(payload))

    @classmethod
    def many_from_json(cls, path: str | Path) -> list[MlaRequest]:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, Mapping) and "requests" in payload:
            payload = payload["requests"]
        if isinstance(payload, Mapping):
            payload = [payload]
        if not isinstance(payload, list) or not payload:
            raise ValueError("MLA input must be an object, a list, or {'requests': [...]}")
        return [cls.from_dict(item) for item in payload]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["config"] = self.config
        return result


@dataclass(frozen=True)
class MlaModelOptions:
    """Execution choices separate from hardware and MLA geometry."""

    algorithm: str = "fa2"
    estimate_level: str = "standard"
    br: int | None = None
    bc: int | None = None
    decode_splits: str | int = "auto"
    decode_query_threshold: int = 16
    max_decode_splits: int = 128
    min_kv_tiles_per_split: int = 4
    overlap_fraction: float | None = None

    def __post_init__(self) -> None:
        validated = ModelOptions(
            algorithm=self.algorithm,
            mode="analytical",
            estimate_level=self.estimate_level,
            br=self.br,
            bc=self.bc,
            decode_splits=self.decode_splits,
            decode_query_threshold=self.decode_query_threshold,
            max_decode_splits=self.max_decode_splits,
            min_kv_tiles_per_split=self.min_kv_tiles_per_split,
            overlap_fraction=self.overlap_fraction,
        )
        object.__setattr__(self, "algorithm", validated.algorithm)
        object.__setattr__(self, "estimate_level", validated.estimate_level)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "FP8_UNSUPPORTED_MESSAGE",
    "MLA_SCHEMA_VERSION",
    "MlaModelOptions",
    "MlaRequest",
]
