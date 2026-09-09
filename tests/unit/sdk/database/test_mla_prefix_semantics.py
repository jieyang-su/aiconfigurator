# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rust-engine regression coverage for MLA prefix handling.

The old Python ``MLAConcatK`` and per-call ``MLAModule`` query helpers were
retired. Context MLA module data now stores measured total-sequence points;
the Rust operator receives ``s`` and ``prefix`` and applies prefix correction
during evaluation. The model's concat work is an ElementWise op.
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations import ElementWise, FallbackOp, MLAModule
from aiconfigurator_core.sdk.engine import _evaluate_single_op
from aiconfigurator_core.sdk.perf_database import PerfDatabase

pytestmark = pytest.mark.unit


_SYSTEM_YAML = {
    "data_dir": "data",
    "gpu": {
        "sm_version": 90,
        "mem_bw": 4_800_000_000_000.0,
        "mem_bw_empirical_scaling_factor": 0.8,
        "mem_empirical_constant_latency": 0.000003,
        "bfloat16_tc_flops": 989_000_000_000_000.0,
        "fp8_tc_flops": 1_978_000_000_000_000.0,
    },
    "node": {
        "num_gpus_per_node": 8,
        "inter_node_bw": 50_000_000_000.0,
        "intra_node_bw": 450_000_000_000.0,
        "p2p_latency": 0.00001,
    },
}


def _module_row(*, isl: int, batch_size: int = 2, latency: float = 0.1) -> dict:
    return {
        "framework": "vllm",
        "version": "test",
        "device": "NVIDIA H100",
        "op_name": "mla_context_module",
        "kernel_source": "default",
        "model": "deepseek-ai/DeepSeek-V3",
        "architecture": "DeepseekV3ForCausalLM",
        "mla_dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "gemm_type": "bfloat16",
        "num_heads": 16,
        "batch_size": batch_size,
        "isl": isl,
        "tp_size": 1,
        "step": 0,
        "latency": latency,
    }


def _write_module_table(root: Path, rows: list[dict]) -> None:
    path = root / "data" / "vllm" / "test" / "mla_context_module_perf.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = {name: [row[name] for row in rows] for name in rows[0]}
    pq.write_table(pa.table({name: pa.array(values) for name, values in columns.items()}), path)


def _database(tmp_path: Path, rows: list[dict] | None = None) -> PerfDatabase:
    root = tmp_path / "systems"
    root.mkdir()
    (root / "testsys.yaml").write_text(yaml.safe_dump(_SYSTEM_YAML), encoding="utf-8")
    # The directory itself makes the engine use the requested SILICON view;
    # it may contain only the table needed by a particular test.
    (root / "data" / "vllm" / "test").mkdir(parents=True)
    if rows:
        _write_module_table(root, rows)
    return PerfDatabase(
        "testsys",
        "vllm",
        "test",
        systems_root=str(root),
        database_mode="SILICON",
        strict_provenance=False,
    )


def _evaluate(database, op, *, batch_size: int, s: int, prefix: int = 0):
    return _evaluate_single_op(
        database,
        op,
        is_context=True,
        batch_size=batch_size,
        s=s,
        prefix=prefix,
    )


def _mla_module() -> MLAModule:
    return MLAModule(
        "context_mla_module",
        1.0,
        True,
        16,
        common.KVCacheQuantMode.bfloat16,
        common.FMHAQuantMode.bfloat16,
        common.GEMMQuantMode.bfloat16,
        native_num_heads=128,
    )


def test_context_module_uses_full_sequence_and_prefix_correction(tmp_path):
    database = _database(
        tmp_path,
        [
            _module_row(isl=64, latency=0.1),
            _module_row(isl=128, latency=0.3),
        ],
    )

    no_prefix = _evaluate(database, _mla_module(), batch_size=2, s=64)
    with_prefix = _evaluate(database, _mla_module(), batch_size=2, s=64, prefix=64)

    assert float(no_prefix) == pytest.approx(0.1)
    # The 128-token measured point is corrected for the 64-token cached
    # prefix: (128^2 - 64^2) / 128^2 = 0.75.
    assert float(with_prefix) == pytest.approx(0.225)
    assert no_prefix.source == with_prefix.source == "silicon"


def test_missing_module_source_falls_back_through_rust_engine(tmp_path):
    database = _database(tmp_path)
    fallback = FallbackOp(
        "context_mla_block",
        primary=_mla_module(),
        fallback=[ElementWise("context_mla_concat_k", 1.0, 1, 1)],
    )

    result = _evaluate(database, fallback, batch_size=2, s=64)

    assert float(result) > 0.0
    assert result.source == "empirical"


def test_mla_concat_is_modeled_as_elementwise_work():
    # The model graph uses 2 * (dim_in + dim_out) bytes per token, which is
    # the same byte count formerly assembled by MLAConcatK.
    op = ElementWise("context_mla_concat_k", 2.0, 32 * 128 + 64, 32 * 192)

    assert op.get_weights() == 0.0
    spec = op._spec_json()
    assert '"name":"context_mla_concat_k"' in spec
    assert '"bytes_per_token":20608.0' in spec
