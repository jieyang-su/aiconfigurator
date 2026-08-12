# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations import ContextKVBProjGEMM, FallbackOp, MLAConcatK, MLAModule, PerformanceResult
from aiconfigurator.sdk.perf_database import LoadedOpData, PerfDataNotAvailableError

pytestmark = pytest.mark.unit


def _module_data():
    return {
        common.FMHAQuantMode.bfloat16: {
            common.KVCacheQuantMode.bfloat16: {
                common.GEMMQuantMode.bfloat16: {
                    128: {
                        16: {
                            0: {64: {2: {"latency": 0.1, "energy": 1.0}}},
                            128: {64: {2: {"latency": 0.3, "energy": 3.0}}},
                        }
                    }
                }
            }
        }
    }


def test_context_module_uses_measured_prefix_axis_without_blanket_scaling(stub_perf_db, monkeypatch):
    stub_perf_db._default_database_mode = common.DatabaseMode.SILICON
    stub_perf_db._context_mla_module_data = LoadedOpData(
        _module_data(), common.PerfDataFilename.mla_context_module, "prefix-aware"
    )
    monkeypatch.setattr(MLAModule, "load_data", classmethod(lambda cls, database: None))

    def query(prefix):
        return MLAModule._query_context_mla_module_table(
            stub_perf_db,
            b=2,
            s=64,
            prefix=prefix,
            num_heads=16,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            native_num_heads=128,
        )

    assert float(query(0)) == pytest.approx(0.1)
    assert float(query(128)) == pytest.approx(0.3)
    assert 0.1 < float(query(64)) < 0.3
    with pytest.raises(PerfDataNotAvailableError, match="prefix bracket"):
        query(256)


def test_context_module_rejects_prefix_without_bracket(stub_perf_db, monkeypatch):
    data = _module_data()
    del data[common.FMHAQuantMode.bfloat16][common.KVCacheQuantMode.bfloat16][common.GEMMQuantMode.bfloat16][128][16][
        128
    ]
    stub_perf_db._default_database_mode = common.DatabaseMode.SILICON
    stub_perf_db._context_mla_module_data = LoadedOpData(
        data, common.PerfDataFilename.mla_context_module, "prefix-zero-only"
    )
    monkeypatch.setattr(MLAModule, "load_data", classmethod(lambda cls, database: None))

    with pytest.raises(PerfDataNotAvailableError, match="prefix bracket"):
        MLAModule._query_context_mla_module_table(
            stub_perf_db,
            b=2,
            s=64,
            prefix=64,
            num_heads=16,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            native_num_heads=128,
        )


def test_missing_prefix_module_data_falls_back_to_granular(stub_perf_db, monkeypatch):
    data = _module_data()
    del data[common.FMHAQuantMode.bfloat16][common.KVCacheQuantMode.bfloat16][common.GEMMQuantMode.bfloat16][128][16][
        128
    ]
    stub_perf_db._default_database_mode = common.DatabaseMode.SILICON
    stub_perf_db._context_mla_module_data = LoadedOpData(
        data, common.PerfDataFilename.mla_context_module, "prefix-zero-only"
    )
    monkeypatch.setattr(MLAModule, "load_data", classmethod(lambda cls, database: None))
    primary = MLAModule(
        "module",
        1.0,
        True,
        16,
        common.KVCacheQuantMode.bfloat16,
        common.FMHAQuantMode.bfloat16,
        common.GEMMQuantMode.bfloat16,
        native_num_heads=128,
    )
    fallback = MagicMock()
    fallback.query.return_value = PerformanceResult(0.7, energy=7.0, source="analytical")
    fallback.get_weights.return_value = 0.0

    result = FallbackOp("mla", primary, [fallback]).query(
        stub_perf_db,
        batch_size=2,
        s=64,
        prefix=64,
    )

    assert float(result) == pytest.approx(0.7)
    assert result.source == "analytical"
    fallback.query.assert_called_once()


def test_context_kv_projection_adds_prefix_before_cp_sharding():
    database = MagicMock()
    database.query_gemm.return_value = PerformanceResult(1.0, energy=2.0)
    op = ContextKVBProjGEMM(
        "kv_b",
        1.0,
        32768,
        512,
        common.GEMMQuantMode.bfloat16,
        seq_split=2,
    )

    op.query(database, x=16, batch_size=2, prefix=10)

    database.query_gemm.assert_called_once_with(18, 32768, 512, common.GEMMQuantMode.bfloat16)


def test_mla_concat_uses_full_k_token_count_and_local_heads():
    database = MagicMock()
    database.query_mem_op.return_value = PerformanceResult(0.5, energy=1.0, source="empirical")
    op = MLAConcatK("concat", 2.0, num_heads=32)

    result = op.query(database, batch_size=2, s=4, prefix=6)

    tokens = 2 * (4 + 6)
    expected_bytes = tokens * 32 * 128 * 2 + tokens * 64 * 2 + tokens * 32 * 192 * 2
    database.query_mem_op.assert_called_once_with(expected_bytes)
    assert float(result) == pytest.approx(1.0)
    assert result.energy == pytest.approx(2.0)
