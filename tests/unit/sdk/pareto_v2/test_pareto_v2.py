from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.pareto_v2 import (
    _best_worker_match,
    _frontier_mask_2d,
    _worker_envelope,
    _worker_templates,
    sweep_disagg_pareto_v2,
)

pytestmark = pytest.mark.unit


def _worker(*, parallel: str, bs: int, tpot: float, ttft: float, rate: float, gpus: int, tp: int = 1) -> dict:
    row = dict.fromkeys(common.ColumnsStatic, 0)
    row.update(
        model="test",
        isl=8,
        osl=4,
        prefix=0,
        bs=bs,
        global_bs=bs,
        ttft=ttft,
        tpot=tpot,
        request_latency=ttft + tpot * 3,
        **{
            "seq/s": rate,
            "seq/s/gpu": rate / gpus,
            "tokens/s": rate * 4,
            "tokens/s/gpu": rate * 4 / gpus,
            "tokens/s/user": 1000 / tpot,
            "num_total_gpus": gpus,
        },
        tp=tp,
        pp=1,
        dp=1,
        moe_tp=1,
        moe_ep=1,
        cp=1,
        parallel=parallel,
        backend="sglang",
        version="test",
        system="test",
    )
    return row


def test_frontier_retains_tradeoffs_and_equal_objective_configs():
    x = np.asarray([10.0, 8.0, 7.0, 10.0, 11.0])
    y = np.asarray([5.0, 7.0, 4.0, 5.0, 4.0])
    assert _frontier_mask_2d(x, y, maximize_x=True, maximize_y=True).tolist() == [True, True, False, True, True]


def test_worker_envelope_matches_pairwise_dominance():
    df = pd.DataFrame(
        [
            _worker(parallel="a", bs=1, tpot=10, ttft=10, rate=10, gpus=1),
            _worker(parallel="a", bs=2, tpot=9, ttft=10, rate=9, gpus=1),
            _worker(parallel="a", bs=3, tpot=11, ttft=10, rate=8, gpus=1),
            _worker(parallel="b", bs=1, tpot=8, ttft=10, rate=7, gpus=2),
        ]
    )
    result = _worker_envelope(df, role="decode", require_same_tp=False)
    assert set(zip(result["num_total_gpus"], result["bs"], strict=True)) == {(1, 1), (1, 2), (2, 1)}


def test_worker_envelope_preserves_legacy_reported_gpu_dimension():
    ep_worker = _worker(parallel="ep", bs=1, tpot=10, ttft=10, rate=8, gpus=8)
    ep_worker.update(tp=1, pp=1, dp=1, moe_ep=8)
    dp_worker = _worker(parallel="dp", bs=2, tpot=9, ttft=9, rate=9, gpus=8)
    dp_worker.update(tp=1, pp=1, dp=8, moe_ep=1)
    result = _worker_envelope(pd.DataFrame([ep_worker, dp_worker]), role="decode", require_same_tp=True)
    assert set(result["parallel"]) == {"ep", "dp"}


def test_vector_worker_match_equals_brute_force():
    templates = _worker_templates(2, 4, range(1, 5), range(1, 5), {8, 12, 16}, None, None)
    match = _best_worker_match(13.0, 7.0, templates, 0.9, 0.92)
    brute = max(
        (
            (min(13.0 * p * 0.9, 7.0 * d * 0.92) / (2 * p + 4 * d), p, d)
            for p in range(1, 5)
            for d in range(1, 5)
            if 2 * p + 4 * d in {8, 12, 16}
        ),
        key=lambda item: item[0],
    )
    assert match == (brute[1], brute[2])


def test_v2_preserves_high_tps_point_removed_by_efficiency_topk(monkeypatch):
    prefill = pd.DataFrame(
        [
            _worker(parallel="p-fast", bs=1, tpot=1, ttft=10, rate=30, gpus=1),
            _worker(parallel="p-wide", bs=1, tpot=1, ttft=5, rate=80, gpus=4),
        ]
    )
    decode = pd.DataFrame(
        [
            _worker(parallel="d-fast", bs=1, tpot=5, ttft=1, rate=6, gpus=1),
            _worker(parallel="d-efficient", bs=8, tpot=10, ttft=1, rate=20, gpus=1),
            _worker(parallel="d-dominated", bs=4, tpot=12, ttft=1, rate=10, gpus=1),
        ]
    )

    calls = iter([prefill, decode])
    monkeypatch.setattr("aiconfigurator.sdk.pareto_v2._get_disagg_worker_candidates", lambda **_: next(calls))
    result = sweep_disagg_pareto_v2(
        model_path="test",
        runtime_config=config.RuntimeConfig(isl=8, osl=4, ttft=100, tpot=20),
        prefill_database=MagicMock(),
        prefill_backend_name="sglang",
        prefill_model_config=config.ModelConfig(),
        prefill_parallel_config_list=[(1, 1, 1, 1, 1, 1)],
        prefill_latency_correction=1,
        decode_database=MagicMock(),
        decode_backend_name="sglang",
        decode_model_config=config.ModelConfig(),
        decode_parallel_config_list=[(1, 1, 1, 1, 1, 1)],
        decode_latency_correction=1,
        prefill_max_num_tokens=8,
        decode_max_num_tokens=8,
        prefill_num_worker_list=[1, 2],
        decode_num_worker_list=[1, 2],
        num_gpu_list=[2, 3, 4, 6, 8, 10, 12],
    )
    assert set(result["tokens/s/user"]) == {100.0, 200.0}
    diagnostics = result.attrs["pareto_v2_diagnostics"]
    assert diagnostics["raw_decode_workers"] == 3
    # Pre-match worker pruning is intentionally disabled while AIC uses
    # different GPU accounting in matching and result reporting.
    assert diagnostics["envelope_decode_workers"] == 3
    assert diagnostics["frontier_points"] == 2


def test_v2_does_not_prune_worker_under_legacy_gpu_accounting_mismatch(monkeypatch):
    slower = _worker(parallel="p-slower", bs=1, tpot=1, ttft=10, rate=6.586, gpus=8)
    faster = _worker(parallel="p-faster", bs=1, tpot=1, ttft=9, rate=6.859, gpus=8)
    for row in (slower, faster):
        row.update(tp=1, pp=1, dp=1, moe_ep=8, cp=8)
    decode = _worker(parallel="d", bs=84, tpot=37.995, ttft=0, rate=17.289, gpus=8)
    decode.update(tp=1, pp=1, dp=8, moe_ep=8)
    calls = iter([pd.DataFrame([slower, faster]), pd.DataFrame([decode])])
    monkeypatch.setattr("aiconfigurator.sdk.pareto_v2._get_disagg_worker_candidates", lambda **_: next(calls))

    result = sweep_disagg_pareto_v2(
        model_path="test",
        runtime_config=config.RuntimeConfig(isl=8, osl=1024, ttft=100, tpot=100),
        prefill_database=MagicMock(),
        prefill_backend_name="sglang",
        prefill_model_config=config.ModelConfig(),
        prefill_parallel_config_list=[(1, 1, 1, 1, 8, 8)],
        prefill_latency_correction=1,
        decode_database=MagicMock(),
        decode_backend_name="sglang",
        decode_model_config=config.ModelConfig(),
        decode_parallel_config_list=[(1, 1, 8, 1, 8, 1)],
        decode_latency_correction=1,
        prefill_max_num_tokens=8,
        decode_max_num_tokens=84,
        prefill_num_worker_list=list(range(1, 33)),
        decode_num_worker_list=list(range(1, 33)),
        num_gpu_list=[1, 2, 4, 8, 16, 24, 32],
    )
    row = result.iloc[0]
    assert row["(p)parallel"] == "p-slower"
    assert row["(p)workers"] == 3
    assert row["(d)workers"] == 1


def test_v2_applies_corrected_ttft_and_request_latency(monkeypatch):
    rejected_ttft = _worker(parallel="p-rejected", bs=1, tpot=1, ttft=60, rate=100, gpus=1)
    accepted = _worker(parallel="p-accepted", bs=1, tpot=1, ttft=10, rate=20, gpus=1)
    decode = _worker(parallel="d", bs=1, tpot=5, ttft=0, rate=20, gpus=1)
    calls = iter([pd.DataFrame([rejected_ttft, accepted]), pd.DataFrame([decode])])
    monkeypatch.setattr("aiconfigurator.sdk.pareto_v2._get_disagg_worker_candidates", lambda **_: next(calls))
    result = sweep_disagg_pareto_v2(
        model_path="test",
        runtime_config=config.RuntimeConfig(isl=8, osl=4, ttft=100, tpot=20, request_latency=30),
        prefill_database=MagicMock(),
        prefill_backend_name="sglang",
        prefill_model_config=config.ModelConfig(),
        prefill_parallel_config_list=[(1, 1, 1, 1, 1, 1)],
        prefill_latency_correction=1,
        decode_database=MagicMock(),
        decode_backend_name="sglang",
        decode_model_config=config.ModelConfig(),
        decode_parallel_config_list=[(1, 1, 1, 1, 1, 1)],
        decode_latency_correction=1,
        prefill_max_num_tokens=8,
        decode_max_num_tokens=1,
        prefill_num_worker_list=[1],
        decode_num_worker_list=[1],
        num_gpu_list=[2],
    )
    assert result.iloc[0]["(p)parallel"] == "p-accepted"
    assert result.iloc[0]["request_latency"] == 25


def test_v2_rejects_autoscale():
    with pytest.raises(ValueError, match="does not support autoscale"):
        sweep_disagg_pareto_v2(
            model_path="test",
            runtime_config=config.RuntimeConfig(),
            prefill_database=MagicMock(),
            prefill_backend_name="sglang",
            prefill_model_config=config.ModelConfig(),
            prefill_parallel_config_list=[],
            prefill_latency_correction=1,
            decode_database=MagicMock(),
            decode_backend_name="sglang",
            decode_model_config=config.ModelConfig(),
            decode_parallel_config_list=[],
            decode_latency_correction=1,
            autoscale=True,
        )
