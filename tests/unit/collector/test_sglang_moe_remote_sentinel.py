# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest


@pytest.mark.unit
def test_raw_moe_benchmark_preserves_remote_expert_sentinel():
    source_path = Path(__file__).resolve().parents[3] / "collector" / "sglang" / "collect_moe.py"
    source = source_path.read_text(encoding="utf-8")

    assert "topk_ids=current_topk_output.topk_ids.clamp(min=0)" not in source
    assert 'current_topk_output = workloads[i % num_iters]["topk_output"]' in source
