# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _add_hisim_repo_to_path(repo_path: str | None) -> None:
    if not repo_path:
        return
    repo = Path(repo_path)
    for path in (repo / "src", repo):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _filter_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(callable_obj)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _emit_progress(job: dict[str, Any], stage: str, **fields: Any) -> None:
    progress_path = job.get("progress_path")
    if not progress_path:
        return
    started_at = float(job.setdefault("_started_at", time.monotonic()))
    record = {
        "time": time.time(),
        "elapsed_s": time.monotonic() - started_at,
        "stage": stage,
    }
    record.update(fields)
    try:
        with Path(progress_path).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass


def _dtype_from_quant(value: str | None, default: str = "FP16") -> str:
    if not value:
        return default
    value = str(value).lower()
    if "bf" in value or "bfloat" in value:
        return "BF16"
    if "fp8" in value:
        return "FP8"
    if "int8" in value:
        return "INT8"
    if "fp4" in value or "nvfp4" in value:
        return "FP4"
    if "int4" in value:
        return "INT4"
    return default


def _capacity_mode(refinement: dict[str, Any]) -> str:
    mode = str((refinement or {}).get("capacity_mode", "auto")).lower()
    return "candidate_bs" if mode in {"candidate_bs", "candidate-bs", "bs"} else "auto"


def _apply_aic_runtime_env(task: dict[str, Any]) -> None:
    env_map = {
        "prefer_nccl_for_custom_allreduce": "AIC_PREFER_NCCL_FOR_CUSTOM_ALLREDUCE",
        "disable_hybrid_shared_layer": "AIC_DISABLE_HYBRID_SHARED_LAYER",
    }
    for config_key, env_key in env_map.items():
        value = task.get(config_key)
        if value is not None:
            if isinstance(value, str):
                enabled = value.strip().lower() in {"1", "true", "yes", "on"}
            else:
                enabled = bool(value)
            os.environ[env_key] = "1" if enabled else "0"


def _resolve_max_running_requests(
    candidate: dict[str, Any],
    hisim_cfg: dict[str, Any],
    capacity_mode: str,
) -> int | None:
    override = hisim_cfg.get("max_running_requests")
    if override is not None:
        return int(override)
    if capacity_mode == "candidate_bs":
        return int(candidate.get("bs") or candidate.get("global_bs") or 1)
    return None


def _estimate_max_total_tokens(
    candidate: dict[str, Any],
    task: dict[str, Any],
    hisim_cfg: dict[str, Any],
    capacity_mode: str,
) -> int | None:
    override = hisim_cfg.get("max_total_tokens")
    if override is not None:
        return int(override)
    if capacity_mode != "candidate_bs":
        return None
    bs = int(candidate.get("bs") or candidate.get("global_bs") or 1)
    isl = int(task.get("isl") or 1)
    osl = int(task.get("osl") or 1)
    ctx_tokens = int(candidate.get("ctx_tokens") or isl)
    return max(bs * (isl + osl), ctx_tokens + bs * osl)


def _write_hisim_config(job: dict[str, Any]) -> str:
    task = job["task"]
    candidate = job["candidate"]
    refinement = job["refinement"]
    hisim_cfg = refinement.get("hisim", {})
    capacity_mode = _capacity_mode(refinement)
    output_dir = Path(job["output_dir"])
    config_path = output_dir / "hisim_config.json"

    platform: dict[str, Any] = {
        "accelerator": {"name": hisim_cfg["platform_accelerator_name"]},
        "num_device_per_node": int(hisim_cfg.get("num_device_per_node", 8)),
    }
    for key in (
        "disk_read_bandwidth_gb",
        "disk_write_bandwidth_gb",
        "memory_read_bandwidth_gb",
        "memory_write_bandwidth_gb",
    ):
        if hisim_cfg.get(key) is not None:
            platform[key] = hisim_cfg[key]

    predictor: dict[str, Any] = {
        "name": "aiconfigurator",
        "device_name": hisim_cfg["predictor_device_name"],
        "database_mode": hisim_cfg.get("database_mode", "SILICON"),
        "prefill_scale_factor": hisim_cfg.get("prefill_scale_factor", 1),
        "decode_scale_factor": hisim_cfg.get("decode_scale_factor", 1),
    }
    for key in ("database_path", "xgb_model_path"):
        if hisim_cfg.get(key) is not None:
            predictor[key] = hisim_cfg[key]

    scheduler = {
        "tp_size": int(candidate.get("tp", 1) or 1),
        "pp_size": int(candidate.get("pp", 1) or 1),
        "dp_size": int(candidate.get("dp", 1) or 1),
        "moe_tp_size": int(candidate.get("moe_tp", candidate.get("tp", 1)) or 1),
        "ep_size": int(candidate.get("moe_ep", 1) or 1),
        "data_type": hisim_cfg.get("data_type") or _dtype_from_quant(candidate.get("gemm")),
        "kv_cache_data_type": hisim_cfg.get("kv_cache_data_type") or _dtype_from_quant(candidate.get("kvcache")),
        "backend_name": hisim_cfg.get("predictor_backend_name") or task.get("backend_name") or "sglang",
        "backend_version": hisim_cfg.get("predictor_backend_version") or task.get("backend_version"),
        "capacity_mode": capacity_mode,
    }
    max_running_requests = _resolve_max_running_requests(candidate, hisim_cfg, capacity_mode)
    if max_running_requests is not None:
        scheduler["max_running_requests"] = max_running_requests

    config = {
        "model": _target_model_config(task["model_path"]),
        "platform": platform,
        "predictor": predictor,
        "scheduler": scheduler,
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return str(config_path)


def _target_model_config(model_path: str) -> dict[str, Any]:
    from aiconfigurator.sdk.utils import get_model_config_from_model_path

    model_info = get_model_config_from_model_path(model_path) or {}
    raw_config = dict(model_info.get("raw_config") or {})
    raw_config["name"] = model_path
    raw_config["model_path"] = model_path
    return raw_config


def _build_server_args(job: dict[str, Any]) -> Any:
    from sglang.srt.server_args import ServerArgs

    task = job["task"]
    candidate = job["candidate"]
    refinement = job["refinement"]
    hisim_cfg = refinement.get("hisim", {})
    capacity_mode = _capacity_mode(refinement)
    ctx_tokens = int(candidate.get("ctx_tokens") or task.get("isl") or 1)
    max_running_requests = _resolve_max_running_requests(candidate, hisim_cfg, capacity_mode)
    max_total_tokens = _estimate_max_total_tokens(candidate, task, hisim_cfg, capacity_mode)

    kwargs: dict[str, Any] = {
        "model_path": hisim_cfg.get("server_model_path") or task["model_path"],
        "load_format": hisim_cfg.get("load_format", "dummy"),
        "device": hisim_cfg.get("server_device", "cpu"),
        "tp_size": 1,
        "max_prefill_tokens": ctx_tokens,
    }
    if max_running_requests is not None:
        kwargs["max_running_requests"] = max_running_requests
    if max_total_tokens is not None:
        kwargs["max_total_tokens"] = max_total_tokens
    if hisim_cfg.get("enable_hierarchical_cache") is not None:
        kwargs["enable_hierarchical_cache"] = bool(hisim_cfg["enable_hierarchical_cache"])
    if task.get("enable_chunked_prefill"):
        kwargs["chunked_prefill_size"] = ctx_tokens
    kwargs.update(hisim_cfg.get("server_args", {}) or {})

    return ServerArgs(**_filter_kwargs(ServerArgs, kwargs))


def _dummy_tokenizer_for_random_ids(model_path: str) -> Any:
    from aiconfigurator.sdk.utils import get_model_config_from_model_path

    try:
        model_info = get_model_config_from_model_path(model_path) or {}
        vocab_size = int(model_info.get("vocab_size") or 129280)
    except Exception:
        vocab_size = 129280
    return SimpleNamespace(vocab_size=vocab_size)


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    _emit_progress(job, "start")
    refinement = job["refinement"]
    hisim_cfg = refinement.get("hisim", {})
    _apply_aic_runtime_env(job.get("task", {}))
    _add_hisim_repo_to_path(hisim_cfg.get("repo_path"))
    _emit_progress(job, "env_ready")

    output_dir = Path(job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = _write_hisim_config(job)
    os.environ["HISIM_CONFIG_PATH"] = config_path
    os.environ["HISIM_OUTPUT_DIR"] = str(output_dir)
    os.environ["HISIM_SIMULATION_MODE"] = "OFFLINE"
    _emit_progress(job, "config_written", config_path=config_path)

    workload = job["task"]["workload"]
    seed = int(workload.get("seed", 42))
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    _emit_progress(job, "seeded", seed=seed)

    from hisim.dataset import DatasetArgs, get_dataset
    from hisim.simulation.sglang.sglang_bench import SGLangBenchmarkRunner
    from hisim.simulation.types import BenchmarkConfig
    _emit_progress(job, "imports_ready")

    task = job["task"]
    request_rate = float(job["request_rate"])
    duration_s = float(workload.get("duration_s", 60))
    num_prompts = max(int(workload.get("num_prompts_min", 512)), int(request_rate * duration_s + 0.999999))
    input_len = int(workload["input_len"])
    output_len = int(workload["output_len"])

    dataset_args = DatasetArgs(
        name=workload.get("dataset", "random_ids"),
        num_prompts=num_prompts,
        min_input_len=input_len,
        max_input_len=input_len,
        min_output_len=output_len,
        max_output_len=output_len,
        prefix_hit_rate=workload.get("prefix_hit_rate"),
        seed=seed,
    )
    benchmark_config = BenchmarkConfig(request_rate=request_rate, ignore_request_timestamp=True)
    _emit_progress(
        job,
        "dataset_args_ready",
        dataset=dataset_args.name,
        num_prompts=num_prompts,
        input_len=input_len,
        output_len=output_len,
        request_rate=request_rate,
    )

    _emit_progress(job, "runner_init_start")
    runner = SGLangBenchmarkRunner(server_args=_build_server_args(job))
    _emit_progress(job, "runner_init_done")
    try:
        if dataset_args.name in {"random_ids", "identical_ids"}:
            _emit_progress(job, "dataset_build_start")
            dataset = get_dataset(
                dataset_args,
                tokenizer=_dummy_tokenizer_for_random_ids(task["model_path"]),
            )
            _emit_progress(job, "dataset_build_done", num_prompts=len(dataset))
            _emit_progress(job, "benchmark_start")
            metrics = runner.benchmark(benchmark_config, dataset=dataset)
        else:
            _emit_progress(job, "benchmark_start")
            metrics = runner.benchmark(benchmark_config, dataset_args=dataset_args)
        _emit_progress(job, "benchmark_done")
    finally:
        _emit_progress(job, "runner_shutdown_start")
        runner.shutdown()
        _emit_progress(job, "runner_shutdown_done")
    if metrics is None:
        raise RuntimeError("hisim runner returned no metrics")
    metrics["hisim_capacity_mode"] = _capacity_mode(refinement)
    metrics["hisim_max_running_requests"] = getattr(
        runner.server_args, "max_running_requests", None
    )
    metrics["hisim_max_total_tokens"] = getattr(runner.server_args, "max_total_tokens", None)
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        metrics_path.write_text(json.dumps(metrics) + "\n", encoding="utf-8")
    _emit_progress(
        job,
        "metrics_ready",
        request_throughput=metrics.get("request_throughput"),
        output_throughput=metrics.get("output_throughput"),
        mean_ttft_ms=metrics.get("mean_ttft_ms"),
        mean_tpot_ms=metrics.get("mean_tpot_ms"),
    )
    return {"metrics": metrics, "output_dir": str(output_dir), "config_path": config_path}


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    if len(argv) != 1:
        raise SystemExit("usage: python -m aiconfigurator.cli.hisim_refinement_worker JOB_JSON")
    job_path = Path(argv[0])
    job = json.loads(job_path.read_text(encoding="utf-8"))
    try:
        result = run_job(job)
        _emit_progress(job, "result_write_start", result_path=job["result_path"])
        Path(job["result_path"]).write_text(json.dumps(result, indent=2), encoding="utf-8")
        _emit_progress(job, "complete", result_path=job["result_path"])
        print(json.dumps({"ok": True, "result_path": job["result_path"]}))
        return 0
    except Exception as exc:
        _emit_progress(job, "failed", error=repr(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
