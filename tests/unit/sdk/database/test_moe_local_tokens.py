# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations import MoE, PerformanceResult


pytestmark = pytest.mark.unit


def _make_moe(workload_distribution="balanced", **kwargs) -> MoE:
    return MoE(
        "context_moe",
        1,
        7168,
        2048,
        8,
        256,
        1,
        8,
        common.MoEQuantMode.fp8_block,
        workload_distribution,
        1,
        is_context=True,
        **kwargs,
    )


def test_h20_local_token_divisor_is_applied_with_ceil():
    database = MagicMock()
    database.system = "h20_pcie"
    database.query_moe.return_value = PerformanceResult(3.0, energy=30.0, source="silicon")
    op = _make_moe(local_token_divisor_by_system={"h20_pcie": 8})

    result = op.query(database, x=8193)

    assert float(result) == 3.0
    assert database.query_moe.call_args.kwargs["num_tokens"] == 1025


def test_other_system_keeps_global_token_count():
    database = MagicMock()
    database.system = "h100_pcie"
    database.query_moe.return_value = PerformanceResult(3.0, energy=30.0, source="silicon")
    op = _make_moe(local_token_divisor_by_system={"h20_pcie": 8})

    op.query(database, x=8193)

    assert database.query_moe.call_args.kwargs["num_tokens"] == 8193


def test_divisor_can_be_limited_to_balanced_distribution():
    database = MagicMock()
    database.system = "h20_pcie"
    database.query_moe.return_value = PerformanceResult(3.0, energy=30.0, source="silicon")
    op = _make_moe(
        workload_distribution="recorded",
        local_token_divisor_by_system={"h20_pcie": 8},
        local_token_divisor_distributions={"balanced"},
    )

    op.query(database, x=8192)

    assert database.query_moe.call_args.kwargs["num_tokens"] == 8192


def test_local_token_divisor_must_be_positive():
    with pytest.raises(ValueError, match="must be positive"):
        _make_moe(local_token_divisor_by_system={"h20_pcie": 0})
