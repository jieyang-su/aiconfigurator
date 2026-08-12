"""Table-free granular DSA operators used by ANALYTICAL mode."""

from __future__ import annotations

import math
from typing import ClassVar

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.operations.base import Operation
from aiconfigurator_core.sdk.performance_result import PerformanceResult


def _result_source(database) -> str:
    return "analytical" if database._default_database_mode == common.DatabaseMode.ANALYTICAL else "estimated"


class DSAIndexScore(Operation):
    """Granular FP8 Index MQA score kernel for DSA producer layers."""

    _CP_AWARE: ClassVar[bool] = True

    def __init__(
        self,
        name: str,
        scale_factor: float,
        *,
        layout: str,
        index_heads: int,
        index_head_dim: int,
        index_topk: int,
        cp_size: int = 1,
        context_stride: int = 1,
    ) -> None:
        super().__init__(name, scale_factor)
        self._layout = layout
        self._index_heads = index_heads
        self._index_head_dim = index_head_dim
        self._index_topk = index_topk
        self._cp_size = cp_size
        self._context_stride = context_stride
        if context_stride <= 0:
            raise ValueError("context_stride must be positive")

    def query(self, database, **kwargs) -> PerformanceResult:
        batch = int(kwargs["batch_size"])
        sequence = int(kwargs["s"])
        prefix = int(kwargs.get("prefix", 0)) if self._layout == "ragged" else 0
        full_context = prefix + sequence if self._layout == "ragged" else sequence
        indexed_context = max(1, full_context // self._context_stride)
        if self._layout == "ragged" and indexed_context <= self._index_topk:
            return PerformanceResult(0.0, energy=0.0, source="analytical")
        query_length = math.ceil(sequence / self._cp_size) if self._layout == "ragged" else 1

        from aiconfigurator_core.sdk.kernelsim.analytical import index_mqa_latency_ms

        def estimate(query: int) -> float:
            return index_mqa_latency_ms(
                gpu=database.system_spec["gpu"],
                layout=self._layout,
                batch=batch,
                query_length=query,
                context_length=indexed_context,
                index_heads=self._index_heads,
                index_head_dim=self._index_head_dim,
                config=database._analytical_config,
            )

        if self._layout == "ragged" and query_length > indexed_context:
            chunks, remainder = divmod(query_length, indexed_context)
            latency = chunks * estimate(indexed_context)
            if remainder:
                latency += estimate(remainder)
        else:
            latency = estimate(query_length)
        return PerformanceResult(
            latency * self._scale_factor,
            energy=0.0,
            source=_result_source(database),
        )

    def get_weights(self, **kwargs):
        return 0.0


class DSATopKSelect(Operation):
    """Granular FP32-score TopK and index-transform kernel."""

    _CP_AWARE: ClassVar[bool] = True

    def __init__(
        self,
        name: str,
        scale_factor: float,
        *,
        layout: str,
        index_topk: int,
        cp_size: int = 1,
        context_stride: int = 1,
        kernel_recipe: str = "dsa",
    ) -> None:
        super().__init__(name, scale_factor)
        self._layout = layout
        self._index_topk = index_topk
        self._cp_size = cp_size
        self._context_stride = context_stride
        self._kernel_recipe = kernel_recipe.strip().lower()
        if context_stride <= 0:
            raise ValueError("context_stride must be positive")
        if self._kernel_recipe not in {"dsa", "dsv4"}:
            raise ValueError("kernel_recipe must be dsa or dsv4")

    def query(self, database, **kwargs) -> PerformanceResult:
        batch = int(kwargs["batch_size"])
        sequence = int(kwargs["s"])
        prefix = int(kwargs.get("prefix", 0)) if self._layout == "ragged" else 0
        full_context = prefix + sequence if self._layout == "ragged" else sequence
        indexed_context = max(1, full_context // self._context_stride)
        if self._layout == "ragged" and indexed_context <= self._index_topk:
            return PerformanceResult(0.0, energy=0.0, source="analytical")
        query_length = math.ceil(sequence / self._cp_size) if self._layout == "ragged" else 1

        if self._kernel_recipe == "dsv4":
            from aiconfigurator_core.sdk.kernelsim.analytical import dsv4_topk_latency_ms

            if self._layout == "ragged":
                local_fresh = query_length
                local_prefix = prefix + max(0, sequence - query_length)
                variant = "v1"
            else:
                local_fresh = 1
                local_prefix = max(0, sequence - 1)
                variant = "v2"
            latency = dsv4_topk_latency_ms(
                gpu=database.system_spec["gpu"],
                variant=variant,
                batch=batch,
                fresh_tokens=local_fresh,
                prefix_tokens=local_prefix,
                index_topk=self._index_topk,
                compression_ratio=self._context_stride,
                config=database._analytical_config,
            )
        else:
            from aiconfigurator_core.sdk.kernelsim.analytical import index_topk_latency_ms

            latency = index_topk_latency_ms(
                gpu=database.system_spec["gpu"],
                layout=self._layout,
                batch=batch,
                query_length=query_length,
                context_length=indexed_context,
                index_topk=self._index_topk,
                config=database._analytical_config,
            )
        return PerformanceResult(
            latency * self._scale_factor,
            energy=0.0,
            source=_result_source(database),
        )

    def get_weights(self, **kwargs):
        return 0.0


class DSASparseAttention(Operation):
    """Granular selected-KV sparse MLA core, excluding index score and TopK."""

    _CP_AWARE: ClassVar[bool] = True

    def __init__(
        self,
        name: str,
        scale_factor: float,
        *,
        layout: str,
        local_heads: int,
        index_topk: int,
        qk_latent_dim: int = 576,
        value_latent_dim: int = 512,
        qk_nope_dim: int = 128,
        output_value_dim: int = 128,
        cp_size: int = 1,
    ) -> None:
        super().__init__(name, scale_factor)
        self._layout = layout
        self._local_heads = local_heads
        self._index_topk = index_topk
        self._qk_latent_dim = qk_latent_dim
        self._value_latent_dim = value_latent_dim
        self._qk_nope_dim = qk_nope_dim
        self._output_value_dim = output_value_dim
        self._cp_size = cp_size

    @staticmethod
    def _causal_pairs(batch: int, query: int, prefix: int, limit: int) -> int:
        full = prefix + query
        if prefix >= limit:
            return batch * query * limit
        if full <= limit:
            return batch * (full * (full + 1) - prefix * (prefix + 1)) // 2
        ramp = batch * (limit * (limit + 1) - prefix * (prefix + 1)) // 2
        return ramp + batch * (full - limit) * limit

    def query(self, database, **kwargs) -> PerformanceResult:
        batch = int(kwargs["batch_size"])
        sequence = int(kwargs["s"])
        if self._layout == "ragged":
            prefix = int(kwargs.get("prefix", 0))
            query_length = math.ceil(sequence / self._cp_size)
            local_prefix = prefix + max(0, sequence - query_length)
            pairs = self._causal_pairs(batch, query_length, local_prefix, self._index_topk)
        else:
            query_length = 1
            pairs = batch * min(sequence, self._index_topk)

        from aiconfigurator_core.sdk.kernelsim.analytical import dsa_sparse_attention_latency_ms

        latency = dsa_sparse_attention_latency_ms(
            gpu=database.system_spec["gpu"],
            batch=batch,
            query_length=query_length,
            selected_pairs=pairs,
            local_heads=self._local_heads,
            qk_latent_dim=self._qk_latent_dim,
            value_latent_dim=self._value_latent_dim,
            qk_nope_dim=self._qk_nope_dim,
            output_value_dim=self._output_value_dim,
            config=database._analytical_config,
        )
        return PerformanceResult(
            latency * self._scale_factor,
            energy=0.0,
            source=_result_source(database),
        )

    def get_weights(self, **kwargs):
        return 0.0
