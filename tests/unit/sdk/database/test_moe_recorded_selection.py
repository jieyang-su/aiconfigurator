# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.common import PerfDataFilename
from aiconfigurator.sdk.operations.moe import (
    MoE,
    _normalize_ordinary_moe_distribution_for_load,
    _ordinary_recorded_phase_distribution,
    _select_moe_leaf,
)
from aiconfigurator.sdk.perf_database import LoadedOpData, PerfDataNotAvailableError, PerfDatabase

pytestmark = pytest.mark.unit


def _shape_leaf(points: dict[int, float], *, moe_ep_size: int = 4) -> dict:
    return {
        8: {
            256: {
                7168: {
                    2048: {
                        1: {
                            moe_ep_size: {
                                token: {"latency": latency, "power": 0.0, "energy": 0.0}
                                for token, latency in points.items()
                            }
                        }
                    }
                }
            }
        }
    }


def _database(monkeypatch, distributions: dict[str, dict]) -> PerfDatabase:
    database = object.__new__(PerfDatabase)
    database.backend = common.BackendName.sglang.value
    database._default_database_mode = common.DatabaseMode.HYBRID
    database._moe_data = LoadedOpData(
        {common.MoEQuantMode.fp8_block: distributions},
        PerfDataFilename.moe,
        "unit-test-moe.parquet",
    )
    database._apply_dsv4_operator_calibration = lambda result, roofline_scale=None: result
    monkeypatch.setattr(MoE, "load_data", classmethod(lambda cls, db: None))
    return database


def _query(
    database: PerfDatabase,
    *,
    num_tokens: int = 48,
    is_context: bool = True,
    moe_ep_size: int = 4,
):
    return database.query_moe(
        num_tokens=num_tokens,
        hidden_size=7168,
        inter_size=2048,
        topk=8,
        num_experts=256,
        moe_tp_size=1,
        moe_ep_size=moe_ep_size,
        quant_mode=common.MoEQuantMode.fp8_block,
        workload_distribution="recorded",
        is_context=is_context,
        database_mode=common.DatabaseMode.HYBRID,
        strict_workload_distribution=True,
    )


def test_recorded_phase_names_are_normalized_for_ordinary_moe():
    assert _ordinary_recorded_phase_distribution("recorded", True) == "recorded_context_no_eplb"
    assert _ordinary_recorded_phase_distribution("recorded", False) == "recorded_generation_no_eplb"
    assert (
        _normalize_ordinary_moe_distribution_for_load("recorded_no_eplb", "context")
        == "recorded_context_no_eplb"
    )
    assert (
        _normalize_ordinary_moe_distribution_for_load(
            "recorded_dummy_fixture_generation_rank_local_no_eplb",
            "generation",
        )
        == "recorded_generation_no_eplb"
    )


def test_strict_recorded_query_selects_context_and_generation_rows(monkeypatch):
    database = _database(
        monkeypatch,
        {
            "recorded_context_no_eplb": _shape_leaf({32: 0.10, 64: 0.20}),
            "recorded_generation_no_eplb": _shape_leaf({32: 0.30, 64: 0.40}),
            "balanced": _shape_leaf({32: 9.0, 64: 9.0}),
        },
    )

    context = _query(database, is_context=True)
    generation = _query(database, is_context=False)

    assert float(context) == pytest.approx(0.15)
    assert float(generation) == pytest.approx(0.35)


def test_non_strict_shape_selection_skips_empty_defaultdict_leaf():
    def tree():
        return defaultdict(tree)

    recorded = tree()
    balanced = tree()
    balanced[8][256][7168][2048][1][4] = {32: {"latency": 1.0}}

    distribution, leaf = _select_moe_leaf(
        {
            "recorded_context_no_eplb": recorded,
            "balanced": balanced,
        },
        "recorded_context_no_eplb",
        topk=8,
        num_experts=256,
        hidden_size=7168,
        inter_size=2048,
        moe_tp_size=1,
        moe_ep_size=4,
    )

    assert distribution == "balanced"
    assert sorted(leaf) == [32]


def test_strict_recorded_query_rejects_distribution_fallback(monkeypatch):
    database = _database(
        monkeypatch,
        {"balanced": _shape_leaf({32: 9.0, 64: 9.0})},
    )

    with pytest.raises(PerfDataNotAvailableError, match="Strict MoE workload distribution selection failed"):
        _query(database)


def test_strict_recorded_query_rejects_shape_fallback(monkeypatch):
    database = _database(
        monkeypatch,
        {
            "recorded_context_no_eplb": _shape_leaf({32: 0.10, 64: 0.20}),
            "balanced": _shape_leaf({32: 9.0, 64: 9.0}, moe_ep_size=8),
        },
    )

    with pytest.raises(PerfDataNotAvailableError, match="Strict MoE shape selection failed"):
        _query(database, moe_ep_size=8)


@pytest.mark.parametrize("num_tokens", [16, 80])
def test_strict_recorded_query_rejects_out_of_range_tokens(monkeypatch, num_tokens):
    database = _database(
        monkeypatch,
        {"recorded_context_no_eplb": _shape_leaf({32: 0.10, 64: 0.20})},
    )

    with pytest.raises(PerfDataNotAvailableError, match="outside collected range"):
        _query(database, num_tokens=num_tokens)
