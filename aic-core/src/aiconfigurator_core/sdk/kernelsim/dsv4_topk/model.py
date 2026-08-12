"""Provisional DeepSeek-V4 CSA TopK v1/v2 KernelSim models."""

from __future__ import annotations

import math
import warnings as python_warnings
from dataclasses import dataclass

DSV4_TOPK_MODEL_VERSION = "2026-08-11.aic-dsv4-topk-v1"
CALIBRATED_TOPK = 512
PROVISIONAL_PRO_TOPK = 1024
CALIBRATED_COMPRESSION = 4

LIMITED_SCOPE_MESSAGE = (
    "The DeepSeek-V4 TopK model is provisional. It was fitted from SGLang "
    "0.5.14 CUDA-graph kernel measurements using top-last synthetic scores. "
    "Flash topK=512 has multi-platform tables; Pro topK=1024 v1 has only a "
    "targeted H100 calibration, while v2 reuses a paired K-invariant profile. "
    "H100 is the calibration platform; "
    "H200 and Blackwell tables are validation evidence, not independent fitted "
    "profiles. Other K values, score distributions, backends and kernels are "
    "low-confidence extrapolations."
)


class Dsv4TopKWarning(UserWarning):
    """Warning for estimates outside the provisional V4 TopK calibration."""


def _positive_int(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class Dsv4TopKShape:
    variant: str
    batch_size: int
    fresh_tokens: int
    prefix_tokens: int = 0
    topk: int = CALIBRATED_TOPK
    compression_ratio: int = CALIBRATED_COMPRESSION

    def __post_init__(self) -> None:
        variant = self.variant.strip().lower()
        if variant not in {"v1", "v2"}:
            raise ValueError("variant must be v1 (prefill) or v2 (decode)")
        object.__setattr__(self, "variant", variant)
        for name in ("batch_size", "fresh_tokens", "topk", "compression_ratio"):
            object.__setattr__(self, name, _positive_int(name, getattr(self, name)))
        prefix = int(self.prefix_tokens)
        if prefix < 0:
            raise ValueError("prefix_tokens must be non-negative")
        object.__setattr__(self, "prefix_tokens", prefix)
        if variant == "v2" and self.fresh_tokens != 1:
            raise ValueError("V4 TopK v2 models normal decode and requires fresh_tokens=1")


@dataclass(frozen=True)
class Dsv4TopKParameters:
    v1_floor_us: float = 1.8876796777852949
    v1_scan_ns_per_k: float = 0.6315947962407802
    v1_row_us: float = 0.30470508398506463
    v1_active_row_us: float = 0.05464873077023488
    v1_log_us: float = 0.1328188658023386
    v1_pro_floor_us: float = 2.0732305051062827
    v1_pro_scan_ns_per_k: float = 0.5393153049428494
    v1_pro_row_us: float = 0.36599741163961375
    v1_pro_active_row_us: float = 0.0
    v1_pro_log_us: float = 0.0
    v2_one_floor_us: float = 2.303645789924076
    v2_one_log_us: float = 0.2910326893486602
    v2_one_batch_log_us: float = 0.052747428779569026
    v2_two_floor_us: float = 7.43770538863625
    v2_two_log_us: float = 2.0351509138085078
    v2_two_batch_log_us: float = 0.01348700968793443
    v2_cluster_floor_us: float = 8.014170795757238
    v2_cluster_log_us: float = 2.004893480458116
    v2_cluster_batch_log_us: float = 0.16093564095181337
    v2_cluster_persistent_us: float = 1.0
    latency_scale: float = 1.0


_STANDARD = Dsv4TopKParameters()
_PROFILES = {
    "standard": _STANDARD,
    "high": Dsv4TopKParameters(**{**_STANDARD.__dict__, "latency_scale": 1.30}),
    "low": Dsv4TopKParameters(**{**_STANDARD.__dict__, "latency_scale": 0.75}),
}


def get_dsv4_topk_parameters(level: str = "standard") -> Dsv4TopKParameters:
    try:
        return _PROFILES[level.strip().lower()]
    except KeyError as error:
        raise ValueError("level must be low, standard, or high") from error


@dataclass(frozen=True)
class Dsv4TopKEstimate:
    latency_us: float
    variant: str
    parameter_level: str
    scope: str
    warnings: tuple[str, ...]
    compressed_context: int
    active_rows: int
    critical_scan: float
    regime: str


def _floor_prefix_sum(last: int, divisor: int) -> int:
    """Return sum(floor(x/divisor), x=0..last)."""
    if last < 0:
        return 0
    groups, remainder = divmod(last, divisor)
    return divisor * groups * (groups - 1) // 2 + groups * (remainder + 1)


def _v1_features(shape: Dsv4TopKShape, sm_count: int) -> tuple[int, int, float, float, float]:
    ratio = shape.compression_ratio
    query = shape.fresh_tokens
    prefix = shape.prefix_tokens
    max_context = max(1, (prefix + query) // ratio)
    first_active = max(1, ratio * (shape.topk + 1) - prefix)
    active_per_request = max(0, query - first_active + 1)
    active_rows = shape.batch_size * active_per_request
    if active_per_request:
        lower = prefix + first_active - 1
        upper = prefix + query
        per_request_scan = _floor_prefix_sum(upper, ratio) - _floor_prefix_sum(lower, ratio)
        total_scan = shape.batch_size * per_request_scan
        critical_scan = max(float(max_context), total_scan / sm_count)
    else:
        critical_scan = 0.0
    query_rows = shape.batch_size * query
    return max_context, active_rows, critical_scan, query_rows / sm_count, active_rows / sm_count


def _estimate_v1(shape: Dsv4TopKShape, sm_count: int, params: Dsv4TopKParameters) -> Dsv4TopKEstimate:
    context, active_rows, critical_scan, row_waves, active_waves = _v1_features(shape, sm_count)
    if shape.topk == PROVISIONAL_PRO_TOPK:
        floor_us = params.v1_pro_floor_us
        scan_ns_per_k = params.v1_pro_scan_ns_per_k
        row_us = params.v1_pro_row_us
        active_row_us = params.v1_pro_active_row_us
        log_us = params.v1_pro_log_us
    else:
        floor_us = params.v1_floor_us
        scan_ns_per_k = params.v1_scan_ns_per_k
        row_us = params.v1_row_us
        active_row_us = params.v1_active_row_us
        log_us = params.v1_log_us
    if context <= shape.topk:
        latency = 0.0
        regime = "trivial-bypassed"
    else:
        latency = (
            floor_us
            + scan_ns_per_k * critical_scan / 1000.0
            + row_us * row_waves
            + active_row_us * active_waves
            + log_us * math.log2(context / shape.topk)
        )
        regime = "v1-radix-per-row"
    return Dsv4TopKEstimate(
        latency_us=latency * params.latency_scale,
        variant=shape.variant,
        parameter_level="custom",
        scope="h100_sglang_0.5.14_dsv4_flash_toplast",
        warnings=(),
        compressed_context=context,
        active_rows=active_rows,
        critical_scan=critical_scan,
        regime=regime,
    )


def _estimate_v2(shape: Dsv4TopKShape, params: Dsv4TopKParameters) -> Dsv4TopKEstimate:
    context = max(1, (shape.prefix_tokens + shape.fresh_tokens) // shape.compression_ratio)
    if context <= shape.topk:
        latency = 0.0
        regime = "trivial-bypassed"
    else:
        batch_log = math.log2(shape.batch_size)
        if context <= 16384:
            latency = (
                params.v2_one_floor_us
                + params.v2_one_log_us * math.log2(context / shape.topk)
                + params.v2_one_batch_log_us * batch_log
            )
            regime = "v2-register-one-pass"
        elif context <= 32768:
            latency = (
                params.v2_two_floor_us
                + params.v2_two_log_us * math.log2(context / 16384)
                + params.v2_two_batch_log_us * batch_log
            )
            regime = "v2-register-two-pass"
        else:
            persistent = max(0.0, shape.batch_size / 15.0 - 1.0)
            latency = (
                params.v2_cluster_floor_us
                + params.v2_cluster_log_us * math.log2(context / 32768)
                + params.v2_cluster_batch_log_us * batch_log
                + params.v2_cluster_persistent_us * persistent
            )
            regime = "v2-cluster"
    return Dsv4TopKEstimate(
        latency_us=latency * params.latency_scale,
        variant=shape.variant,
        parameter_level="custom",
        scope="h100_sglang_0.5.14_dsv4_flash_toplast",
        warnings=(),
        compressed_context=context,
        active_rows=shape.batch_size if context > shape.topk else 0,
        critical_scan=float(context),
        regime=regime,
    )


def estimate_dsv4_topk(
    shape: Dsv4TopKShape,
    *,
    sm_count: int,
    parameter_level: str = "standard",
    params: Dsv4TopKParameters | None = None,
) -> Dsv4TopKEstimate:
    if not isinstance(shape, Dsv4TopKShape):
        raise TypeError("shape must be a Dsv4TopKShape")
    sm_count = _positive_int("sm_count", sm_count)
    level = parameter_level.strip().lower()
    if params is not None and level != "standard":
        raise ValueError("custom params cannot be combined with a non-standard parameter_level")
    selected = params or get_dsv4_topk_parameters(level)
    messages = [LIMITED_SCOPE_MESSAGE]
    if shape.topk == PROVISIONAL_PRO_TOPK:
        messages.append(
            "topk=1024 Pro v1 is fitted from 23 targeted H100 shapes only; cross-hardware transfer is not validated"
        )
    elif shape.topk != CALIBRATED_TOPK:
        messages.append(f"topk={shape.topk} is not calibrated; supported fitted K values are 512 and 1024")
    if shape.compression_ratio != CALIBRATED_COMPRESSION:
        messages.append("compression_ratio differs from the calibrated CSA c4 cache")
    compressed = (shape.prefix_tokens + shape.fresh_tokens) // shape.compression_ratio
    if compressed > 262144:
        messages.append("compressed context exceeds the measured and V4 v2 nominal 262144-candidate range")
    python_warnings.warn(" ".join(messages), Dsv4TopKWarning, stacklevel=2)
    estimate = _estimate_v1(shape, sm_count, selected) if shape.variant == "v1" else _estimate_v2(shape, selected)
    return Dsv4TopKEstimate(
        **{
            **estimate.__dict__,
            "parameter_level": "custom" if params is not None else level,
            "warnings": tuple(messages),
        }
    )
