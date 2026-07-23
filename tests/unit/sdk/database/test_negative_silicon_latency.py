# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.perf_database import PerfDataNotAvailableError, PerfDatabase
from aiconfigurator.sdk.performance_result import PerformanceResult

pytestmark = pytest.mark.unit


def _query(database_mode):
    database = object.__new__(PerfDatabase)
    database._apply_dsv4_operator_calibration = lambda result, roofline_scale=None: result
    return database._query_silicon_or_hybrid(
        get_silicon=lambda: PerformanceResult(-1.25, energy=0.0),
        get_empirical=lambda: 2.5,
        database_mode=database_mode,
        error_msg="test silicon query",
    )


def test_negative_silicon_latency_is_an_explicit_database_miss():
    with pytest.raises(PerfDataNotAvailableError, match=r"negative latency -1\.25 ms"):
        _query(common.DatabaseMode.SILICON)


def test_negative_silicon_latency_falls_back_in_hybrid_mode():
    result = _query(common.DatabaseMode.HYBRID)

    assert result == 2.5
    assert result.source == "empirical"
