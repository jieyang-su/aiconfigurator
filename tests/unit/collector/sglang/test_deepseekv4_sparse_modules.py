# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from unittest.mock import MagicMock

import pytest

_saved_mock = None
if isinstance(sys.modules.get("torch"), MagicMock):
    _saved_mock = sys.modules.pop("torch")

try:
    import torch
except ImportError:
    if _saved_mock is not None:
        sys.modules["torch"] = _saved_mock
    pytest.skip("real torch required for sparse collector tests", allow_module_level=True)

from collector.sglang import deepseekv4_sparse_modules as sparse_modules


@pytest.mark.unit
def test_bench_hca_attn_uses_torch_subprocess_after_illegal_access(monkeypatch):
    monkeypatch.setenv("COLLECTOR_DSV4_HCA_TORCH_FALLBACK", "fatal")

    def fail_flash(*args, **kwargs):
        raise torch.AcceleratorError("CUDA error: an illegal memory access was encountered")

    fallback_calls = []

    def fake_fallback(M, past_kv, *, batch_size=1, device="cuda:0"):
        fallback_calls.append((M, past_kv, batch_size, str(device)))
        return {"latency_ms": 1.23, "power_stats": None, "backend": "torch_eager"}

    monkeypatch.setattr(sparse_modules, "_bench_flash_mla_sparse", fail_flash)
    monkeypatch.setattr(sparse_modules, "_bench_hca_attn_torch_subprocess", fake_fallback)

    result = sparse_modules._bench_hca_attn(64, 2048, batch_size=4, tp_size=1, device="cuda:0")

    assert result["latency_ms"] == pytest.approx(1.23)
    assert result["backend"] == "torch_eager"
    assert fallback_calls == [(64, 2048, 4, "cuda:0")]


@pytest.mark.unit
def test_bench_hca_attn_does_not_fallback_for_non_illegal_access(monkeypatch):
    monkeypatch.setenv("COLLECTOR_DSV4_HCA_TORCH_FALLBACK", "fatal")

    def fail_flash(*args, **kwargs):
        raise RuntimeError("some unrelated failure")

    monkeypatch.setattr(sparse_modules, "_bench_flash_mla_sparse", fail_flash)

    with pytest.raises(RuntimeError, match="unrelated failure"):
        sparse_modules._bench_hca_attn(64, 2048, batch_size=1, tp_size=1, device="cuda:0")