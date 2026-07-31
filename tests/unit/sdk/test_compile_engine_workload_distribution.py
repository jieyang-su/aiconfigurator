# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from aiconfigurator.sdk import engine


@pytest.mark.unit
@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("balanced", "balanced"),
        (None, "power_law"),
    ],
)
def test_compile_engine_forwards_optional_workload_distribution(monkeypatch, requested, expected):
    captured = {}

    def fake_get_model(model_path, model_config, backend):
        captured["model_config"] = model_config
        return SimpleNamespace(
            architecture="DeepseekV3ForCausalLM",
            context_ops=[],
            generation_ops=[],
            encoder_ops=[],
            config=model_config,
        )

    monkeypatch.setattr(engine, "get_model", fake_get_model)
    monkeypatch.setattr(engine, "_maybe_load_database", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        engine.aiconfigurator_core,
        "engine_spec_bincode_from_json",
        lambda spec_json: b"compiled",
    )

    result = engine.compile_engine(
        "test-model",
        "h100_pcie",
        "sglang",
        backend_version="0.5.9",
        tp_size=4,
        attention_dp_size=1,
        moe_tp_size=1,
        moe_ep_size=4,
        workload_distribution=requested,
    )

    assert result == b"compiled"
    assert captured["model_config"].workload_distribution == expected
