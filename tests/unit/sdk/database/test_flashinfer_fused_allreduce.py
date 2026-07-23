# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.common import PerfDataFilename
from aiconfigurator.sdk.operations.communication import (
    FusedAllReduceResidualRMSNorm,
    load_flashinfer_fused_allreduce_data,
)
from aiconfigurator.sdk.perf_database import LoadedOpData, PerfDataNotAvailableError, PerfDatabase

pytestmark = pytest.mark.unit


def make_database(tmp_path: Path, monkeypatch) -> PerfDatabase:
    csv_path = tmp_path / "flashinfer_fused_allreduce_perf.txt"
    csv_path.write_text(
        "framework,version,device,op_name,kernel_source,dtype,num_gpus,"
        "token_num,hidden_size,message_size,pattern,execution_mode,latency\n"
        "SGLang,0.5.9,H100,op,source,bfloat16,4,32,7168,229376,auto,eager,0.02\n"
        "SGLang,0.5.9,H100,op,source,bfloat16,4,64,7168,458752,auto,eager,0.04\n",
        encoding="utf-8",
    )
    database = object.__new__(PerfDatabase)
    database._flashinfer_fused_allreduce_data = LoadedOpData(
        load_flashinfer_fused_allreduce_data(str(csv_path)),
        PerfDataFilename.flashinfer_fused_allreduce,
        str(csv_path),
    )
    monkeypatch.setattr(
        FusedAllReduceResidualRMSNorm,
        "load_data",
        classmethod(lambda cls, db: None),
    )
    return database


def test_flashinfer_fused_allreduce_interpolates_tokens(tmp_path, monkeypatch):
    database = make_database(tmp_path, monkeypatch)

    result = database.query_flashinfer_fused_allreduce(
        common.CommQuantMode.half,
        tp_size=4,
        token_num=48,
        hidden_size=7168,
        pattern="auto",
        execution_mode="eager",
    )

    assert float(result) == pytest.approx(0.03)


def test_flashinfer_fused_allreduce_rejects_uncollected_tp(tmp_path, monkeypatch):
    database = make_database(tmp_path, monkeypatch)

    with pytest.raises(PerfDataNotAvailableError):
        database.query_flashinfer_fused_allreduce(
            common.CommQuantMode.half,
            tp_size=2,
            token_num=32,
            hidden_size=7168,
        )


def test_flashinfer_fused_allreduce_rejects_extrapolation(tmp_path, monkeypatch):
    database = make_database(tmp_path, monkeypatch)

    with pytest.raises(PerfDataNotAvailableError):
        database.query_flashinfer_fused_allreduce(
            common.CommQuantMode.half,
            tp_size=4,
            token_num=128,
            hidden_size=7168,
        )
