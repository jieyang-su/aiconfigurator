# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiconfigurator.sdk.backends.sglang_backend import SGLANGBackend


class _FakeDatabase:
    def __init__(self, mem_gib: int):
        self.system_spec = {"gpu": {"mem_capacity": mem_gib * (1 << 30)}}


def test_sglang_scheduler_config_uses_ctx_tokens_as_compat_fallback():
    backend = SGLANGBackend()

    cfg = backend._build_sglang_scheduler_config(
        _FakeDatabase(80),
        batch_size=4,
        ctx_tokens=2048,
        sglang_chunked_prefill_size=None,
        sglang_max_prefill_tokens=None,
        sglang_max_running_requests=None,
        sglang_enable_mixed_chunk=None,
    )

    assert cfg.chunked_prefill_size == 2048
    assert cfg.max_prefill_tokens == 16384
    assert cfg.max_running_requests == 4
    assert cfg.enable_mixed_chunk is False


def test_sglang_scheduler_config_uses_memory_default_without_ctx_tokens():
    backend = SGLANGBackend()

    cfg = backend._build_sglang_scheduler_config(
        _FakeDatabase(80),
        batch_size=4,
        ctx_tokens=0,
        sglang_chunked_prefill_size=None,
        sglang_max_prefill_tokens=None,
        sglang_max_running_requests=None,
        sglang_enable_mixed_chunk=None,
    )

    assert cfg.chunked_prefill_size == 8192


def test_sglang_scheduler_config_respects_explicit_values():
    backend = SGLANGBackend()

    cfg = backend._build_sglang_scheduler_config(
        _FakeDatabase(80),
        batch_size=4,
        ctx_tokens=2048,
        sglang_chunked_prefill_size=4096,
        sglang_max_prefill_tokens=8192,
        sglang_max_running_requests=2,
        sglang_enable_mixed_chunk=True,
    )

    assert cfg.chunked_prefill_size == 4096
    assert cfg.max_prefill_tokens == 8192
    assert cfg.max_running_requests == 2
    assert cfg.enable_mixed_chunk is True