"""Table-free granular DSA operators used by ANALYTICAL mode."""
from __future__ import annotations
import math
from aiconfigurator_core.sdk.kernelsim.analytical import dsa_sparse_attention_latency_ms, index_mqa_latency_ms, index_topk_latency_ms
from aiconfigurator_core.sdk.operations.base import Operation
from aiconfigurator_core.sdk.performance_result import PerformanceResult

class DSAIndexScore(Operation):
    _CP_AWARE = True
    def __init__(self, name, scale_factor, *, layout, index_heads, index_head_dim, index_topk, cp_size=1):
        super().__init__(name, scale_factor); self._layout, self._index_heads, self._index_head_dim, self._index_topk, self._cp_size = layout, index_heads, index_head_dim, index_topk, cp_size
    def query(self, database, **kwargs):
        b, s = int(kwargs["batch_size"]), int(kwargs["s"]); p = int(kwargs.get("prefix", 0)) if self._layout == "ragged" else 0; c = p + s if self._layout == "ragged" else s
        if self._layout == "ragged" and c <= self._index_topk: return PerformanceResult(0.0, energy=0.0, source="analytical")
        t = index_mqa_latency_ms(gpu=database.system_spec["gpu"], layout=self._layout, batch=b, query_length=math.ceil(s / self._cp_size) if self._layout == "ragged" else 1, context_length=c, index_heads=self._index_heads, index_head_dim=self._index_head_dim, config=database._analytical_config)
        return PerformanceResult(t * self._scale_factor, energy=0.0, source="analytical")
    def get_weights(self, **kwargs): return 0.0

class DSATopKSelect(Operation):
    _CP_AWARE = True
    def __init__(self, name, scale_factor, *, layout, index_topk, cp_size=1): super().__init__(name, scale_factor); self._layout, self._index_topk, self._cp_size = layout, index_topk, cp_size
    def query(self, database, **kwargs):
        b, s = int(kwargs["batch_size"]), int(kwargs["s"]); p = int(kwargs.get("prefix", 0)) if self._layout == "ragged" else 0; c = p + s if self._layout == "ragged" else s
        if self._layout == "ragged" and c <= self._index_topk: return PerformanceResult(0.0, energy=0.0, source="analytical")
        t = index_topk_latency_ms(gpu=database.system_spec["gpu"], layout=self._layout, batch=b, query_length=math.ceil(s / self._cp_size) if self._layout == "ragged" else 1, context_length=c, index_topk=self._index_topk, config=database._analytical_config)
        return PerformanceResult(t * self._scale_factor, energy=0.0, source="analytical")
    def get_weights(self, **kwargs): return 0.0

class DSASparseAttention(Operation):
    _CP_AWARE = True
    def __init__(self, name, scale_factor, *, layout, local_heads, index_topk, qk_latent_dim=576, value_latent_dim=512, qk_nope_dim=128, output_value_dim=128, cp_size=1):
        super().__init__(name, scale_factor); self._layout, self._local_heads, self._index_topk, self._qk_latent_dim, self._value_latent_dim, self._qk_nope_dim, self._output_value_dim, self._cp_size = layout, local_heads, index_topk, qk_latent_dim, value_latent_dim, qk_nope_dim, output_value_dim, cp_size
    def query(self, database, **kwargs):
        b, s = int(kwargs["batch_size"]), int(kwargs["s"]); q = math.ceil(s / self._cp_size) if self._layout == "ragged" else 1; pairs = b * q * min(self._index_topk, int(kwargs.get("prefix", 0)) + s)
        t = dsa_sparse_attention_latency_ms(gpu=database.system_spec["gpu"], batch=b, query_length=q, selected_pairs=pairs, local_heads=self._local_heads, qk_latent_dim=self._qk_latent_dim, value_latent_dim=self._value_latent_dim, qk_nope_dim=self._qk_nope_dim, output_value_dim=self._output_value_dim, config=database._analytical_config)
        return PerformanceResult(t * self._scale_factor, energy=0.0, source="analytical")
    def get_weights(self, **kwargs): return 0.0
