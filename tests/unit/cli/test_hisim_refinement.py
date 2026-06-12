# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from aiconfigurator.cli.hisim_refinement import (
    SubprocessHisimEvaluator,
    map_hisim_metrics_to_aic_row,
    refine_task_result,
    search_best_rps,
    select_refinement_candidates,
)
from aiconfigurator.cli import hisim_refinement_worker as worker
from aiconfigurator.cli.main import build_experiment_task_configs

pytestmark = pytest.mark.unit


def _task(ttft=100.0, tpot=20.0, request_latency=None):
    runtime = SimpleNamespace(ttft=ttft, tpot=tpot, request_latency=request_latency)
    return SimpleNamespace(config=SimpleNamespace(runtime_config=runtime))


def _refine_task_config():
    runtime = SimpleNamespace(
        ttft=100.0,
        tpot=20.0,
        request_latency=None,
        isl=512,
        osl=32,
        prefix=0,
    )
    return SimpleNamespace(
        config=SimpleNamespace(runtime_config=runtime),
        refinement={
            "enabled": True,
            "backend": "hisim",
            "serving_modes": ["agg"],
            "candidate_top_k": 1,
            "frontier_buffer_pct": 0.0,
            "sla_metric": "mean",
            "capacity_mode": "auto",
            "merge_analytical_results": True,
            "load_search": {
                "max_iters": 0,
                "initial_upper_scale": 1.0,
                "max_upper_scale": 1.0,
                "num_operating_points": 1,
                "target_concurrency_points": [],
            },
        },
        serving_mode="agg",
        model_path="model",
        backend_name="sglang",
        backend_version="test",
    )


def test_select_refinement_candidates_topk_buffer_and_stable_dedupe():
    df = pd.DataFrame(
        [
            {"system": "h200", "parallel": "tp1", "bs": 1, "ctx_tokens": 100, "tokens/s/user": 10, "tokens/s/gpu": 120},
            {"system": "h200", "parallel": "tp2", "bs": 1, "ctx_tokens": 100, "tokens/s/user": 20, "tokens/s/gpu": 100},
            {"system": "h200", "parallel": "tp3", "bs": 1, "ctx_tokens": 100, "tokens/s/user": 20, "tokens/s/gpu": 92},
            {"system": "h200", "parallel": "tp3", "bs": 1, "ctx_tokens": 100, "tokens/s/user": 20, "tokens/s/gpu": 91},
            {"system": "h200", "parallel": "tp4", "bs": 1, "ctx_tokens": 100, "tokens/s/user": 20, "tokens/s/gpu": 80},
        ]
    )

    selected = select_refinement_candidates(
        df,
        candidate_top_k=1,
        frontier_buffer_pct=0.10,
        x_axis_col="tokens/s/user",
    )

    assert selected["parallel"].tolist() == ["tp1", "tp2", "tp3"]


def test_select_refinement_candidates_includes_top_x_axis_candidates():
    df = pd.DataFrame(
        [
            {"system": "h200", "parallel": "high_y", "bs": 1, "ctx_tokens": 100, "tokens/s/user": 10, "tokens/s/gpu": 200},
            {"system": "h200", "parallel": "balanced", "bs": 1, "ctx_tokens": 100, "tokens/s/user": 100, "tokens/s/gpu": 100},
            {"system": "h200", "parallel": "high_x_dominated", "bs": 1, "ctx_tokens": 100, "tokens/s/user": 99, "tokens/s/gpu": 10},
            {"system": "h200", "parallel": "low", "bs": 1, "ctx_tokens": 100, "tokens/s/user": 20, "tokens/s/gpu": 20},
        ]
    )

    selected = select_refinement_candidates(
        df,
        candidate_top_k=2,
        frontier_buffer_pct=0.0,
        x_axis_col="tokens/s/user",
    )

    assert selected["parallel"].tolist() == ["high_y", "balanced", "high_x_dominated"]


def test_select_refinement_candidates_auto_capacity_dedupes_across_seed_bs():
    df = pd.DataFrame(
        [
            {"system": "h200", "num_total_gpus": 8, "parallel": "tp8", "bs": 1, "ctx_tokens": 100, "tokens/s/user": 100, "tokens/s/gpu": 10},
            {"system": "h200", "num_total_gpus": 8, "parallel": "tp8", "bs": 8, "ctx_tokens": 100, "tokens/s/user": 40, "tokens/s/gpu": 80},
            {"system": "h200", "num_total_gpus": 8, "parallel": "tp4", "bs": 4, "ctx_tokens": 100, "tokens/s/user": 60, "tokens/s/gpu": 60},
        ]
    )

    selected = select_refinement_candidates(
        df,
        candidate_top_k=3,
        frontier_buffer_pct=0.0,
        x_axis_col="tokens/s/user",
        capacity_mode="auto",
    )

    assert selected["parallel"].tolist() == ["tp8", "tp4"]
    assert selected.loc[selected["parallel"] == "tp8", "bs"].item() == 8


def test_map_hisim_metrics_to_aic_row_preserves_aic_and_recomputes_throughput():
    candidate = pd.Series(
        {
            "ttft": 10.0,
            "tpot": 2.0,
            "request_latency": 100.0,
            "seq/s": 8.0,
            "tokens/s": 800.0,
            "tokens/s/gpu": 200.0,
            "num_total_gpus": 4,
        }
    )
    metrics = {
        "mean_ttft_ms": 12.0,
        "mean_tpot_ms": 4.0,
        "mean_e2e_latency_ms": 300.0,
        "request_throughput": 7.0,
        "output_throughput": 700.0,
        "p99_ttft_ms": 20.0,
        "p99_tpot_ms": 8.0,
        "p99_e2e_latency_ms": 500.0,
        "mean_queue_ms": 3.0,
        "prefix_cache_reused_ratio": 0.5,
    }

    row = map_hisim_metrics_to_aic_row(candidate, metrics, best_rps=7.2)

    assert row["aic_ttft"] == 10.0
    assert row["ttft"] == 12.0
    assert row["tpot"] == 4.0
    assert row["request_latency"] == 300.0
    assert row["seq/s"] == 7.0
    assert row["tokens/s/gpu"] == 175.0
    assert row["tokens/s/user"] == 250.0
    assert row["hisim_best_rps"] == 7.2
    assert row["hisim_p99_e2e_latency_ms"] == 500.0


def test_search_best_rps_converges_to_largest_mean_sla_point():
    class MockEvaluator:
        def evaluate(self, candidate, request_rate):
            ok = request_rate <= 10.0
            return {
                "mean_ttft_ms": 50.0 if ok else 150.0,
                "mean_tpot_ms": 10.0 if ok else 30.0,
                "request_throughput": request_rate,
                "output_throughput": request_rate * 100,
            }

    result = search_best_rps(
        MockEvaluator(),
        pd.Series({"seq/s": 8.0}),
        _task(),
        {
            "sla_metric": "mean",
            "load_search": {
                "max_iters": 10,
                "rps_tolerance_pct": 1,
                "initial_upper_scale": 1.5,
                "max_upper_scale": 4.0,
            },
        },
    )

    assert 9.8 <= result.best_rps <= 10.05
    assert result.metrics["request_throughput"] == pytest.approx(result.best_rps)
    assert len(result.operating_points) == 5
    assert result.operating_points[-1]["request_rate"] == pytest.approx(result.best_rps)


def test_search_best_rps_adds_target_concurrency_points():
    class MockEvaluator:
        def evaluate(self, candidate, request_rate):
            return {
                "mean_ttft_ms": 20.0,
                "mean_tpot_ms": 5.0,
                "mean_e2e_latency_ms": 1000.0,
                "request_throughput": request_rate,
                "output_throughput": request_rate * 10,
            }

    result = search_best_rps(
        MockEvaluator(),
        pd.Series({"seq/s": 8.0, "request_latency": 1000.0}),
        _task(ttft=100.0, tpot=20.0, request_latency=2000.0),
        {
            "sla_metric": "mean",
            "load_search": {
                "max_iters": 10,
                "rps_tolerance_pct": 1,
                "initial_upper_scale": 1.0,
                "max_upper_scale": 1.0,
                "num_operating_points": 1,
                "target_concurrency_points": [1, 2],
                "target_concurrency_refine_iters": 1,
            },
        },
    )

    target_points = [
        point for point in result.operating_points
        if point.get("point_type") == "target_concurrency"
    ]
    assert [point["target_concurrency"] for point in target_points] == [1.0, 2.0]
    assert [point["request_rate"] for point in target_points] == pytest.approx([1.0, 2.0])


def test_worker_auto_capacity_does_not_use_candidate_bs():
    candidate = {"bs": 8, "global_bs": 32}
    task = {"isl": 512, "osl": 32}
    hisim_cfg = {}

    assert worker._resolve_max_running_requests(candidate, hisim_cfg, "auto") is None
    assert worker._estimate_max_total_tokens(candidate, task, hisim_cfg, "auto") is None
    assert worker._resolve_max_running_requests(candidate, hisim_cfg, "candidate_bs") == 8
    assert worker._estimate_max_total_tokens(candidate, task, hisim_cfg, "candidate_bs") == 4352


def test_worker_writes_predictor_backend_override(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "_target_model_config", lambda model_path: {"name": model_path})
    job = {
        "task": {
            "model_path": "model",
            "backend_name": "trtllm",
            "backend_version": "1.3.0rc10",
        },
        "candidate": {
            "tp": 2,
            "pp": 1,
            "dp": 4,
            "moe_tp": 2,
            "moe_ep": 2,
            "gemm": "nvfp4",
            "kvcache": "fp8",
        },
        "refinement": {
            "capacity_mode": "auto",
            "hisim": {
                "platform_accelerator_name": "PRO6000_clos32",
                "predictor_device_name": "PRO6000_clos32",
                "predictor_backend_name": "trtllm",
                "predictor_backend_version": "1.3.0rc10",
            },
        },
        "output_dir": str(tmp_path),
    }

    config_path = worker._write_hisim_config(job)

    with open(config_path, encoding="utf-8") as fh:
        config = json.load(fh)
    assert config["scheduler"]["backend_name"] == "trtllm"
    assert config["scheduler"]["backend_version"] == "1.3.0rc10"


def test_refine_task_result_merges_all_analytical_rows_with_hisim_rows():
    class MockEvaluator:
        def evaluate(self, candidate, request_rate):
            return {
                "mean_ttft_ms": 10.0,
                "mean_tpot_ms": 5.0,
                "mean_e2e_latency_ms": 200.0,
                "request_throughput": request_rate,
                "output_throughput": request_rate * 10,
            }

    analytical = pd.DataFrame(
        [
            {"system": "h200", "num_total_gpus": 8, "parallel": "tp8", "bs": 1, "ctx_tokens": 512, "tokens/s/user": 100, "tokens/s/gpu": 10, "seq/s": 1.0, "ttft": 10.0, "tpot": 10.0, "request_latency": 320.0},
            {"system": "h200", "num_total_gpus": 8, "parallel": "tp8", "bs": 8, "ctx_tokens": 512, "tokens/s/user": 40, "tokens/s/gpu": 80, "seq/s": 8.0, "ttft": 80.0, "tpot": 25.0, "request_latency": 900.0},
        ]
    )

    result = refine_task_result(
        exp_name="exp",
        task_config=_refine_task_config(),
        task_result={"pareto_df": analytical},
        evaluator=MockEvaluator(),
    )

    assert set(result["pareto_df"]["refinement_source"]) == {
        "aic_analytical",
        "hisim_refined",
    }
    assert len(result["pareto_df"][result["pareto_df"]["refinement_source"] == "aic_analytical"]) == 2
    assert not result["hisim_refined_df"].empty


def test_refine_task_result_can_refine_candidates_in_parallel():
    calls = []

    class MockEvaluator:
        def evaluate(self, candidate, request_rate):
            calls.append(candidate["parallel"])
            return {
                "mean_ttft_ms": 10.0,
                "mean_tpot_ms": 5.0,
                "mean_e2e_latency_ms": 200.0,
                "request_throughput": request_rate,
                "output_throughput": request_rate * 10,
            }

    task_config = _refine_task_config()
    task_config.refinement["candidate_top_k"] = 2
    task_config.refinement["max_workers"] = 2
    analytical = pd.DataFrame(
        [
            {"system": "h200", "num_total_gpus": 8, "parallel": "tp8", "bs": 8, "ctx_tokens": 512, "tokens/s/user": 80, "tokens/s/gpu": 80, "seq/s": 8.0, "ttft": 80.0, "tpot": 25.0, "request_latency": 900.0},
            {"system": "h200", "num_total_gpus": 4, "parallel": "tp4", "bs": 4, "ctx_tokens": 512, "tokens/s/user": 100, "tokens/s/gpu": 40, "seq/s": 4.0, "ttft": 40.0, "tpot": 10.0, "request_latency": 400.0},
        ]
    )

    result = refine_task_result(
        exp_name="exp",
        task_config=task_config,
        task_result={"pareto_df": analytical},
        evaluator=MockEvaluator(),
    )

    assert set(calls) == {"tp8", "tp4"}
    assert len(result["hisim_refined_df"]) == 2


def test_subprocess_evaluator_uses_persistent_cache(tmp_path, monkeypatch):
    calls = 0

    class FakePopen:
        def __init__(self, cmd, stdout, stderr, text, env):
            nonlocal calls
            calls += 1
            self.args = cmd
            self.returncode = 0
            self.stdout = StringIO("")
            self.stderr = StringIO("")
            job = json.loads(Path(cmd[-1]).read_text(encoding="utf-8"))
            Path(job["progress_path"]).write_text(
                json.dumps({"stage": "complete", "elapsed_s": 0.1}) + "\n",
                encoding="utf-8",
            )
            Path(job["result_path"]).write_text(
                json.dumps({"metrics": {"request_throughput": 1.0, "calls": calls}}),
                encoding="utf-8",
            )

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr("aiconfigurator.cli.hisim_refinement.subprocess.Popen", FakePopen)
    runtime = SimpleNamespace(
        isl=512,
        osl=32,
        prefix=0,
        ttft=100.0,
        tpot=20.0,
        request_latency=None,
    )
    task_config = SimpleNamespace(
        config=SimpleNamespace(runtime_config=runtime),
        model_path="model",
        backend_name="trtllm",
        backend_version="1.3.0rc10",
        serving_mode="agg",
        enable_chunked_prefill=False,
    )
    refinement = {
        "cache": {"enabled": True, "dir": str(tmp_path / "cache")},
        "hisim": {
            "output_dir": str(tmp_path / "runs"),
            "repo_path": None,
            "timeout_s": 10,
        },
        "workload": {
            "dataset": "random_ids",
            "seed": 42,
            "num_prompts_min": 4,
            "duration_s": 1,
            "input_len": 512,
            "output_len": 32,
            "prefix_hit_rate": 0.0,
        },
    }
    evaluator = SubprocessHisimEvaluator(refinement, task_config, "exp")
    candidate = pd.Series({"parallel": "tp1", "seq/s": 1.0})

    first = evaluator.evaluate(candidate, 1.25)
    second = evaluator.evaluate(candidate, 1.25)

    assert calls == 1
    assert first == second
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1


def test_build_experiment_task_configs_merges_root_and_per_exp_refinement():
    config = {
        "refinement": {
            "enabled": True,
            "candidate_top_k": 3,
            "workload": {"seed": 7},
            "hisim": {"repo_path": "/tmp/hisim"},
        },
        "exps": ["exp_a", "exp_b"],
        "exp_a": {
            "serving_mode": "agg",
            "model_path": "Qwen/Qwen3-32B",
            "total_gpus": 8,
            "system_name": "h200_sxm",
            "prefer_nccl_for_custom_allreduce": True,
            "disable_hybrid_shared_layer": True,
            "refinement": {"workload": {"duration_s": 30}},
        },
        "exp_b": {
            "serving_mode": "agg",
            "model_path": "Qwen/Qwen3-32B",
            "total_gpus": 8,
            "system_name": "h200_sxm",
            "refinement": {"enabled": False},
        },
    }

    tasks = build_experiment_task_configs(config=config)

    assert tasks["exp_a"].refinement["enabled"] is True
    assert tasks["exp_a"].refinement["candidate_top_k"] == 3
    assert tasks["exp_a"].refinement["workload"]["seed"] == 7
    assert tasks["exp_a"].refinement["workload"]["duration_s"] == 30
    assert tasks["exp_a"].prefer_nccl_for_custom_allreduce is True
    assert tasks["exp_a"].disable_hybrid_shared_layer is True
    assert tasks["exp_b"].refinement["enabled"] is False


def test_worker_applies_aic_runtime_env(monkeypatch):
    monkeypatch.delenv("AIC_PREFER_NCCL_FOR_CUSTOM_ALLREDUCE", raising=False)
    monkeypatch.delenv("AIC_DISABLE_HYBRID_SHARED_LAYER", raising=False)

    worker._apply_aic_runtime_env(
        {
            "prefer_nccl_for_custom_allreduce": True,
            "disable_hybrid_shared_layer": "true",
        }
    )

    assert os.environ["AIC_PREFER_NCCL_FOR_CUSTOM_ALLREDUCE"] == "1"
    assert os.environ["AIC_DISABLE_HYBRID_SHARED_LAYER"] == "1"
