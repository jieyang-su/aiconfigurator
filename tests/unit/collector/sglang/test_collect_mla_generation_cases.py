# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch


def _load_collector():
    """Load the case generator without requiring a CUDA-enabled torch install."""
    fake_helper = types.ModuleType("helper")
    fake_helper.get_sm_version = lambda: 90
    fake_helper.log_perf = lambda **kwargs: None
    fake_helper.benchmark_with_power = lambda **kwargs: None

    fake_torch = types.ModuleType("torch")
    module_name = "test_collect_mla_module_generation_cases_target"
    module_path = Path(__file__).parents[4] / "collector/sglang/collect_mla_module.py"

    with patch.dict(sys.modules, {"helper": fake_helper, "torch": fake_torch}):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def _shape_set(cases):
    return {(case[1], case[0]) for case in cases}


def test_generation_includes_batch_24_and_long_kv_for_large_batches():
    module = _load_collector()
    with patch.object(module, "_get_precision_combos", return_value=[("bfloat16", "bfloat16", "bfloat16")]):
        shapes = _shape_set(module.get_generation_test_cases("mla"))

    assert (24, 131072) in shapes
    assert (32, 32768) in shapes


def test_generation_does_not_filter_on_batch_times_past_kv():
    module = _load_collector()
    with patch.object(module, "_get_precision_combos", return_value=[("bfloat16", "bfloat16", "bfloat16")]):
        shapes = _shape_set(module.get_generation_test_cases("mla"))

    assert (1024, 131072) in shapes


def test_generation_uses_strict_past_kv_plus_new_token_limit():
    module = _load_collector()
    with patch.object(module, "_get_precision_combos", return_value=[("bfloat16", "bfloat16", "bfloat16")]):
        excluded = _shape_set(module.get_generation_test_cases("mla", max_sequence_length=32769))
        included = _shape_set(module.get_generation_test_cases("mla", max_sequence_length=32770))

    assert (24, 32768) not in excluded
    assert (24, 32768) in included


def test_supported_model_max_sequence_lengths_come_from_cached_configs():
    module = _load_collector()

    assert module._get_model_max_sequence_length("deepseek-ai/DeepSeek-V3") == 163840
    assert module._get_model_max_sequence_length("deepseek-ai/DeepSeek-V3.2") == 163840
    assert module._get_model_max_sequence_length("zai-org/GLM-5") == 202752
