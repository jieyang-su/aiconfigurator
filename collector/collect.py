# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Top-level collector entrypoint.

This script resolves the requested backend, framework version, model/SM case
plan, and op registry entry, then runs the selected collector functions and
writes perf files. It is the orchestration layer for collector v2; individual
modules own benchmark setup, while `model_cases.py` and YAML own case selection.
"""

import contextlib
import functools
import os
import warnings

from helper import get_device_module, get_device_str


def setup_warning_filters():
    """Configure warning filters to suppress known non-critical warnings"""

    # Suppress the modelopt transformers version warning
    warnings.filterwarnings(
        "ignore",
        message="transformers version .* is incompatible with nvidia-modelopt",
        category=UserWarning,
        module="modelopt",
    )

    # Suppress the cuda.cudart deprecation warning
    warnings.filterwarnings("ignore", message="The cuda.cudart module is deprecated", category=FutureWarning)

    warnings.filterwarnings("ignore", message="The cuda.cuda module is deprecated", category=FutureWarning)

    # Suppress TensorRT-LLM specific warnings if needed
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorrt_llm")

    # Suppress flashinfer warnings
    warnings.filterwarnings("ignore", message="Prebuilt kernels not found", module="flashinfer")

    # Suppress torch operator override warnings (flash_attn kernel re-registration)
    warnings.filterwarnings(
        "ignore",
        message="Warning only once for all operators.*",
        category=UserWarning,
    )

    # Suppress pynvml deprecation warning from torch.cuda
    warnings.filterwarnings(
        "ignore",
        message="The pynvml package is deprecated",
        category=FutureWarning,
    )


import random
import resource

from tqdm import tqdm

try:
    import torch
except ModuleNotFoundError:
    torch = None

setup_warning_filters()

import argparse
import cProfile
import importlib
import importlib.util
import io
import json
import multiprocessing as mp
import pstats
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter
from datetime import datetime
from inspect import Parameter, signature
from pathlib import Path

from helper import (
    EXIT_CODE_RESTART,
    create_test_case_id,
    finalize_perf_files,
    find_perf_csv_outputs,
    save_error_report,
    setup_logging,
    setup_signal_handlers,
)

logger = None
RESUME_SCHEMA_VERSION = "collector-resume-v1"
STALL_THRESHOLD = 30  # iterations (x 0.5 s sleep = 15 s) before stall bailout

_DYNAMIC_FALSE_VALUES = {"0", "false", "no", "off"}
_DSV3_LATENCY_WORKSPACE_ENV = "COLLECTOR_DSV3_LATENCY_WORKSPACE_DIR"

_DSEEK_V3_EP8_CONTEXT_TOKENS = "8 16 32 64 128 512 640 1536 2048 2560 4096 5120 8192 10240 12288 14336 16384"
_DSEEK_V3_EP8_GENERATION_TOKENS = "8 32 40 64 128 288 512 896 1024"
_DSEEK_V3_ORDINARY_MOE_TOKENS = (
    "1 2 4 8 32 40 64 128 288 512 640 896 1024 "
    "1536 2048 2560 4096 5120 8192 10240 12288 14336 16384"
)
_DSEEK_V3_ORDINARY_MOE_REPLAY_TOKENS = (
    "1 2 4 8 16 32 40 64 128 288 512 640 896 1024 "
    "1536 2048 2560 4096 5120 8192 10240 12288 14336 16384"
)
_DSEEK_V3_RECORDED_MOE_EP_SIZES = "1,2,4,8,16,32"
_DSEEK_V3_MOE_DISTRIBUTION_GENERATION_TOKENS = (
    "8 32 40 64 128 288 512 640 896 1024 "
    "1536 2048 2560 4096 5120 8192 10240 12288 14336 16384"
)
_DSEEK_V3_ORDINARY_MOE_RECORDED_TOKENS = (
    "1 2 4 8 32 40 64 128 288 512 640 896 1024 "
    "1536 2048 2560 4096 5120 8192 10240 12288 14336 16384"
)
_DSEEK_V3_LEGACY_BASE_OP_CASES = (
    "attention_context",
    "attention_encoder",
    "attention_generation",
)


def _require_torch():
    if torch is None:
        raise RuntimeError("PyTorch is required to run collectors. Use --plan-only to inspect collector v2 YAML plans.")
    return torch


def _cuda_available() -> bool:
    return torch is not None and torch.cuda.is_available()


def _xpu_available() -> bool:
    return torch is not None and hasattr(torch, "xpu") and torch.xpu.is_available()


def _resolve_perf_filename(perf_filename: str) -> str:
    """Resolve bare perf filenames into the active collector run directory."""
    if os.path.isabs(perf_filename) or os.path.dirname(perf_filename):
        return perf_filename

    log_dir = os.environ.get("COLLECTOR_LOG_DIR", "").strip()
    if not log_dir:
        return perf_filename

    if not os.path.isabs(log_dir):
        log_dir = os.path.abspath(log_dir)
    return os.path.join(log_dir, perf_filename)


def _perf_output_roots() -> list[Path]:
    """Return directories that may receive collector perf staging files."""
    roots = [Path.cwd()]
    if os.environ.get("COLLECTOR_LOG_DIR", "").strip():
        roots.append(Path(_resolve_perf_filename("__collector_probe_perf.txt")).parent)

    seen = set()
    unique_roots = []
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_roots.append(root)
    return unique_roots


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_recorded_materialization_source_path(
    *,
    env_names: tuple[str, ...],
    run_dir: Path,
    phase: str,
) -> Path | None:
    for env_name in env_names:
        override = os.environ.get(env_name, "").strip()
        if override:
            return Path(override).resolve()
    filename = (
        "wideep_context_moe_perf.txt"
        if phase == "context"
        else "wideep_generation_moe_perf.txt"
    )
    raw_source = _dsv3_latency_source_root(run_dir) / "raw_collector_source" / filename
    if raw_source.exists():
        return raw_source.resolve()
    return (run_dir / filename).resolve()


def _bool_env_enabled(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in _DYNAMIC_FALSE_VALUES


def _keep_dsv3_latency_sources() -> bool:
    """Keep raw/materialized AIC MoE sources for offline latency experiments.

    The newer name describes the intended use more clearly: these files are
    not validation truth, but collector-owned inputs for trying alternate
    latency source selection.  Keep the historical env var as a compatibility
    alias for existing scripts.
    """

    if "COLLECTOR_DSV3_KEEP_LATENCY_SOURCES" in os.environ:
        return _bool_env_enabled("COLLECTOR_DSV3_KEEP_LATENCY_SOURCES", False)
    return _bool_env_enabled("COLLECTOR_DSV3_KEEP_MATERIALIZED_SOURCES", False)


def _keep_dsv3_materialized_sources() -> bool:
    return _keep_dsv3_latency_sources()


def _dsv3_clean_latency_enabled() -> bool:
    return _bool_env_enabled("COLLECTOR_DSV3_CLEAN_LATENCY", True)


def _dsv3_latency_source_root(run_dir: Path) -> Path:
    workspace = os.environ.get(_DSV3_LATENCY_WORKSPACE_ENV, "").strip()
    if workspace:
        return Path(workspace).resolve()
    return run_dir


def _capture_dsv3_latency_sources(run_dir: Path) -> bool:
    return _keep_dsv3_latency_sources() or _dsv3_latency_source_root(run_dir) != run_dir


@contextlib.contextmanager
def _dsv3_latency_workspace(args, ops: list[str] | None, run_dir: Path):
    if (
        not _is_deepseek_v3_ep8_flow(args, ops)
        or not _dsv3_clean_latency_enabled()
        or _keep_dsv3_latency_sources()
    ):
        yield
        return

    previous = os.environ.get(_DSV3_LATENCY_WORKSPACE_ENV)
    with tempfile.TemporaryDirectory(prefix="dsv3_latency_sources_") as tmp_dir:
        os.environ[_DSV3_LATENCY_WORKSPACE_ENV] = tmp_dir
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop(_DSV3_LATENCY_WORKSPACE_ENV, None)
            else:
                os.environ[_DSV3_LATENCY_WORKSPACE_ENV] = previous


@contextlib.contextmanager
def _dsv3_materialized_source_dir(run_dir: Path, dirname: str):
    if _keep_dsv3_materialized_sources():
        path = run_dir / dirname
        path.mkdir(parents=True, exist_ok=True)
        yield path
        return
    workspace_root = _dsv3_latency_source_root(run_dir)
    if workspace_root != run_dir:
        path = workspace_root / dirname
        path.mkdir(parents=True, exist_ok=True)
        yield path
        return
    with tempfile.TemporaryDirectory(prefix=f"{dirname}_") as tmp_dir:
        yield Path(tmp_dir)


def _preserve_raw_collector_file(run_dir: Path, path: Path) -> None:
    """Save the pre-materialization collector output once per run."""

    if not _capture_dsv3_latency_sources(run_dir):
        return
    if not path.exists():
        return
    raw_dir = _dsv3_latency_source_root(run_dir) / "raw_collector_source"
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / path.name
    if destination.exists():
        return
    shutil.copy2(path, destination)


def _preserve_dsv3_latency_source_input(
    *,
    run_dir: Path,
    role: str,
    path: Path | None,
) -> None:
    """Copy a source-family input used by Recorded materialization.

    These files are captured into an internal temporary workspace when clean
    latency is enabled.  They are only persisted in the run directory when
    COLLECTOR_DSV3_KEEP_LATENCY_SOURCES is enabled.
    """

    if not _capture_dsv3_latency_sources(run_dir) or path is None or not path.exists():
        return
    bundle_dir = _dsv3_latency_source_root(run_dir) / "aic_latency_source_bundle" / "recorded_materialization_inputs"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    destination = bundle_dir / f"{role}{path.suffix or '.txt'}"
    shutil.copy2(path, destination)


def _preserve_dsv3_latency_source_manifest(
    *,
    run_dir: Path,
    inputs: dict[str, Path | None],
) -> None:
    if not _capture_dsv3_latency_sources(run_dir):
        return
    bundle_dir = _dsv3_latency_source_root(run_dir) / "aic_latency_source_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "description": (
            "AIC-only source files preserved for offline MoE latency source "
            "selection. These files are not server/profile truth."
        ),
        "env": {
            "COLLECTOR_DSV3_KEEP_LATENCY_SOURCES": os.environ.get(
                "COLLECTOR_DSV3_KEEP_LATENCY_SOURCES",
                "",
            ),
            "COLLECTOR_DSV3_KEEP_MATERIALIZED_SOURCES": os.environ.get(
                "COLLECTOR_DSV3_KEEP_MATERIALIZED_SOURCES",
                "",
            ),
        },
        "recorded_materialization_inputs": {
            role: "" if path is None else str(path)
            for role, path in sorted(inputs.items())
        },
    }
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _csv_ints(values: list[int]) -> str:
    return ",".join(str(value) for value in values)


def _visible_gpu_count_for_defaults() -> int:
    raw = os.environ.get("COLLECTOR_MOE_DISTRIBUTION_VISIBLE_DEVICES")
    if not raw:
        raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw:
        return max(1, len([item for item in raw.split(",") if item.strip()]))
    if torch is not None and torch.cuda.is_available():
        return max(1, int(torch.cuda.device_count() or 1))
    return 1


def _env_ints(name: str) -> list[int] | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    return [int(item) for item in raw.replace(",", " ").split() if item.strip()]


def _append_env_words(name: str, values: tuple[str, ...]) -> bool:
    before = os.environ.get(name, "")
    items = [item.strip() for item in before.replace(",", " ").split() if item.strip()]
    changed = False
    for value in values:
        if value not in items:
            items.append(value)
            changed = True
    if changed:
        os.environ[name] = " ".join(items)
    return changed


def _is_deepseek_v3_ep8_flow(args, ops: list[str] | None) -> bool:
    requested_ops = set(ops or [])
    if args.backend != "sglang":
        return False
    model_path = (args.model_path or "").lower()
    if "deepseek" not in model_path or "v3" not in model_path:
        return False
    return "moe_token_distribution" in requested_ops and (
        "moe" in requested_ops or "wideep_moe" in requested_ops
    )


def _maybe_apply_deepseek_v3_ep8_defaults(args, ops: list[str] | None) -> None:
    if not _is_deepseek_v3_ep8_flow(args, ops):
        return
    legacy_base_changed = _append_env_words("COLLECTOR_LEGACY_BASE_OP_CASES", _DSEEK_V3_LEGACY_BASE_OP_CASES)
    # DeepSeek-V3 MoE calibration is profile-free by default: each requested EP
    # shape is materialized as rank-local replay and benchmarked by a single GPU
    # worker.  Visible GPUs may run independent cases in parallel, but no case
    # uses multiple physical GPUs as real EP ranks unless single-GPU simulation
    # is explicitly disabled.
    os.environ.setdefault("COLLECTOR_DSV3_EP8_SINGLE_GPU_SIM", "true")
    single_gpu_sim = _bool_env_enabled("COLLECTOR_DSV3_EP8_SINGLE_GPU_SIM", True)
    if single_gpu_sim:
        os.environ.setdefault("COLLECTOR_MOE_DISTRIBUTION_SINGLE_CARD_EP_SIM", "true")
        os.environ.setdefault("COLLECTOR_MOE_DISTRIBUTION_MATERIALIZED_PROFILE", "deterministic")
        if (
            os.environ.get("COLLECTOR_MOE_DISTRIBUTION_MATERIALIZED_PROFILE", "")
            .strip()
            .lower()
            != "deterministic"
        ):
            if logger is not None:
                logger.warning(
                    "Ignoring COLLECTOR_MOE_DISTRIBUTION_MATERIALIZED_PROFILE=%s; "
                    "single-card EP simulation now only supports deterministic replay",
                    os.environ.get("COLLECTOR_MOE_DISTRIBUTION_MATERIALIZED_PROFILE"),
                )
            os.environ["COLLECTOR_MOE_DISTRIBUTION_MATERIALIZED_PROFILE"] = "deterministic"
        os.environ.setdefault("COLLECTOR_MOE_DISTRIBUTION_LOAD_FORMAT", "dummy")
        os.environ.setdefault("COLLECTOR_MOE_DISTRIBUTION_WORKLOAD_SOURCE", "dummy")
        os.environ.setdefault("COLLECTOR_MOE_RANK_LOCAL_REPLAY_PHASES", "context,generation")
        os.environ.setdefault("COLLECTOR_WIDEEP_MOE_REPLAY_SOURCE", "dummy")
        os.environ.setdefault("COLLECTOR_WIDEEP_MOE_DISTRIBUTIONS", "recorded uniform power_law")
        os.environ.setdefault("COLLECTOR_DSV3_SINGLE_CARD_CASE_PARALLEL", "true")
    materialized_distribution = _bool_env_enabled(
        "COLLECTOR_MOE_DISTRIBUTION_SINGLE_CARD_EP_SIM",
        False,
    )
    before = (
        os.environ.get("COLLECTOR_MOE_DISTRIBUTION_EP_SIZES"),
        os.environ.get("COLLECTOR_WIDEEP_MOE_EP_SIZES"),
        os.environ.get("COLLECTOR_MOE_DISTRIBUTION_TOKENS"),
        os.environ.get("COLLECTOR_WIDEEP_MOE_PREFILL_TOKENS"),
        os.environ.get("COLLECTOR_MOE_DISTRIBUTION_GENERATION_TOKENS"),
        os.environ.get("COLLECTOR_WIDEEP_MOE_DECODE_TOKENS"),
        os.environ.get("COLLECTOR_DSV3_EP8_SINGLE_GPU_SIM"),
        os.environ.get("COLLECTOR_MOE_DISTRIBUTION_SINGLE_CARD_EP_SIM"),
        os.environ.get("COLLECTOR_MOE_DISTRIBUTION_MATERIALIZED_PROFILE"),
    )
    # Real-EP mode is retained as an explicit escape hatch.  It caps live
    # recorder EP sizes to visible GPUs and defaults WideEP to EP8 only.
    if single_gpu_sim and not materialized_distribution:
        visible = _visible_gpu_count_for_defaults()
        supported_live_eps = [ep for ep in (1, 2, 4, 8) if ep <= visible]
        requested_live_eps = _env_ints("COLLECTOR_MOE_DISTRIBUTION_EP_SIZES")
        if requested_live_eps is None:
            os.environ["COLLECTOR_MOE_DISTRIBUTION_EP_SIZES"] = _csv_ints(supported_live_eps)
        else:
            capped = [ep for ep in requested_live_eps if ep <= visible]
            os.environ["COLLECTOR_MOE_DISTRIBUTION_EP_SIZES"] = _csv_ints(capped or [1])
        os.environ.setdefault("COLLECTOR_WIDEEP_MOE_EP_SIZES", "8")
    if materialized_distribution:
        os.environ.setdefault("COLLECTOR_MOE_DISTRIBUTION_EP_SIZES", _DSEEK_V3_RECORDED_MOE_EP_SIZES)
        os.environ.setdefault("COLLECTOR_WIDEEP_MOE_EP_SIZES", "2,4,8")
        os.environ.setdefault("COLLECTOR_MOE_DISTRIBUTION_EPLB_MODES", "false,true")
    ordinary_replay_tokens = (
        _DSEEK_V3_ORDINARY_MOE_REPLAY_TOKENS
        if materialized_distribution
        else _DSEEK_V3_EP8_CONTEXT_TOKENS
    )
    os.environ.setdefault("COLLECTOR_MOE_DISTRIBUTION_TOKENS", ordinary_replay_tokens)
    os.environ.setdefault("COLLECTOR_WIDEEP_MOE_PREFILL_TOKENS", _DSEEK_V3_EP8_CONTEXT_TOKENS)
    os.environ.setdefault(
        "COLLECTOR_MOE_DISTRIBUTION_GENERATION_TOKENS",
        _DSEEK_V3_MOE_DISTRIBUTION_GENERATION_TOKENS
        if materialized_distribution
        else ordinary_replay_tokens,
    )
    os.environ.setdefault("COLLECTOR_WIDEEP_MOE_DECODE_TOKENS", _DSEEK_V3_EP8_GENERATION_TOKENS)
    os.environ.setdefault("COLLECTOR_MOE_TOKENS", _DSEEK_V3_ORDINARY_MOE_TOKENS)
    os.environ.setdefault("COLLECTOR_MOE_RECORDED_TOKENS", _DSEEK_V3_ORDINARY_MOE_RECORDED_TOKENS)
    os.environ.setdefault("COLLECTOR_MOE_EP_SIZES", _DSEEK_V3_RECORDED_MOE_EP_SIZES)
    if "COLLECTOR_MOE_DISTRIBUTION_EPLB_MODES" not in os.environ:
        wideep_eplb = os.environ.get("COLLECTOR_WIDEEP_MOE_ENABLE_EPLB")
        if wideep_eplb is not None:
            os.environ["COLLECTOR_MOE_DISTRIBUTION_EPLB_MODES"] = (
                "true"
                if wideep_eplb.strip().lower() not in _DYNAMIC_FALSE_VALUES
                else "false"
            )
    after = (
        os.environ.get("COLLECTOR_MOE_DISTRIBUTION_EP_SIZES"),
        os.environ.get("COLLECTOR_WIDEEP_MOE_EP_SIZES"),
        os.environ.get("COLLECTOR_MOE_DISTRIBUTION_TOKENS"),
        os.environ.get("COLLECTOR_WIDEEP_MOE_PREFILL_TOKENS"),
        os.environ.get("COLLECTOR_MOE_DISTRIBUTION_GENERATION_TOKENS"),
        os.environ.get("COLLECTOR_WIDEEP_MOE_DECODE_TOKENS"),
        os.environ.get("COLLECTOR_DSV3_EP8_SINGLE_GPU_SIM"),
        os.environ.get("COLLECTOR_MOE_DISTRIBUTION_SINGLE_CARD_EP_SIM"),
        os.environ.get("COLLECTOR_MOE_DISTRIBUTION_MATERIALIZED_PROFILE"),
    )
    if logger is not None and before != after:
        logger.info(
            "DeepSeek-V3 MoE calibration defaults applied: "
            "COLLECTOR_MOE_DISTRIBUTION_EP_SIZES=%s, "
            "COLLECTOR_WIDEEP_MOE_EP_SIZES=%s, "
            "context_tokens=%s, generation_tokens=%s, "
            "single_gpu_sim=%s, single_card_ep_sim=%s, materialized_profile=%s",
            after[0],
            after[1],
            after[2],
            after[4],
            after[6],
            after[7],
            after[8],
        )
        if single_gpu_sim:
            logger.info(
                "DeepSeek-V3 MoE single-card EP simulation: each case uses one "
                "GPU worker with deterministic profile-free rank-local replay; "
                "visible GPUs may parallelize independent cases, but they are "
                "not used as real EP ranks. "
                "Historical bootstrap replay is not used. Server/profile data is "
                "used only by the final validation compare."
            )
        if legacy_base_changed:
            logger.info(
                "DeepSeek-V3 full collection: using legacy base case sweeps for "
                "non-MoE attention ops (%s); MoE/WideEP keeps recorded calibration points.",
                os.environ.get("COLLECTOR_LEGACY_BASE_OP_CASES"),
            )


def _read_perf_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write_perf_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    import csv

    merged_fields = list(fieldnames)
    seen = set(merged_fields)
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                merged_fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=merged_fields)
        writer.writeheader()
        writer.writerows(rows)


ORDINARY_MOE_COMPACT_FIELDS = [
    "framework",
    "version",
    "device",
    "op_name",
    "kernel_source",
    "moe_dtype",
    "num_tokens",
    "hidden_size",
    "inter_size",
    "topk",
    "num_experts",
    "moe_tp_size",
    "moe_ep_size",
    "distribution",
    "phase",
    "latency",
]

WIDEEP_MOE_COMPACT_FIELDS = [
    "framework",
    "version",
    "device",
    "op_name",
    "kernel_source",
    "moe_dtype",
    "num_tokens",
    "hidden_size",
    "inter_size",
    "topk",
    "num_experts",
    "moe_tp_size",
    "moe_ep_size",
    "distribution",
    "gemm_path",
    "kernel_regime",
    "latency",
]


def _compact_fields_for_perf(filename: str) -> list[str]:
    if filename == "moe_perf.txt":
        return ORDINARY_MOE_COMPACT_FIELDS
    if filename in {"wideep_context_moe_perf.txt", "wideep_generation_moe_perf.txt"}:
        return WIDEEP_MOE_COMPACT_FIELDS
    raise ValueError(f"Unsupported compact perf table: {filename}")


def _write_compact_perf_rows(path: Path, rows: list[dict[str, str]]) -> None:
    import csv

    fieldnames = _compact_fields_for_perf(path.name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fieldnames} for row in rows)


def _sync_run_tables_to_latency_source_root(run_dir: Path) -> Path:
    source_root = _dsv3_latency_source_root(run_dir)
    if source_root == run_dir:
        _sync_materialized_tables_to_clean_latency_inputs(source_root)
        return source_root

    source_root.mkdir(parents=True, exist_ok=True)
    for name in (
        "collection_summary_sglang.json",
        "collector.log",
        "collector_errors.log",
        "moe_token_distribution_perf.txt",
        "moe_perf.txt",
        "wideep_context_moe_perf.txt",
        "wideep_generation_moe_perf.txt",
    ):
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, source_root / name)
    for dirname in (
        "raw_collector_source",
        "recorded_materialized_source",
        "ordinary_moe_materialized_source",
        "aic_latency_source_bundle",
    ):
        src = run_dir / dirname
        dst = source_root / dirname
        if src.exists() and not dst.exists():
            shutil.copytree(src, dst)
    _sync_materialized_tables_to_clean_latency_inputs(source_root)
    return source_root


def _sync_materialized_tables_to_clean_latency_inputs(source_root: Path) -> None:
    """Feed clean latency with full AIC materialized tables, not compact outputs."""

    preferred_sources = {
        "moe_perf.txt": source_root / "ordinary_moe_materialized_source" / "moe_perf.txt",
        "wideep_context_moe_perf.txt": source_root
        / "recorded_materialized_source"
        / "wideep_context_moe_perf.txt",
        "wideep_generation_moe_perf.txt": source_root
        / "recorded_materialized_source"
        / "wideep_generation_moe_perf.txt",
    }
    for filename, src in preferred_sources.items():
        if src.exists():
            shutil.copy2(src, source_root / filename)


def _preserve_full_header_table(
    *,
    run_dir: Path,
    filename: str,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    if not _keep_dsv3_latency_sources():
        return
    full_dir = run_dir / "aic_latency_source_bundle" / "final_full_header"
    full_dir.mkdir(parents=True, exist_ok=True)
    _write_perf_rows(full_dir / filename, fieldnames, rows)


def _wideep_table_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        str(row.get("moe_ep_size", "")),
        str(row.get("num_tokens", "")),
        str(row.get("distribution", "")),
    )


def _install_recorded_latency_into_run_tables(
    *,
    run_dir: Path,
    validation_source_dir: Path,
) -> None:
    """Make collected WideEP MoE tables simulation-facing.

    Prefer materialized Recorded rows when a validation source exists: that is
    the collector-owned source-selection step for context sparse/dense and
    generation small/main families.  If materialized rows are unavailable, fall
    back to applying the shared recorded latency policy to the current run rows.
    Baseline distributions stay exactly as collected.
    """

    repo_root = _repo_root()
    try:
        from moe_hybrid_policy import apply_profile_free_hybrid_latency
    except ModuleNotFoundError:
        sys.path.insert(0, str(repo_root / "collector"))
        from moe_hybrid_policy import apply_profile_free_hybrid_latency

    recorded_distributions = {"recorded", "recorded_no_eplb", "recorded_eplb"}

    for filename in ("wideep_context_moe_perf.txt", "wideep_generation_moe_perf.txt"):
        run_path = run_dir / filename
        if not run_path.exists():
            continue

        run_fields, run_rows = _read_perf_rows(run_path)
        materialized_path = validation_source_dir / filename
        materialized_by_key = {}
        materialized_rows = []
        if materialized_path.exists():
            _, materialized_rows = _read_perf_rows(materialized_path)
            materialized_by_key = {
                _wideep_table_key(row): row
                for row in materialized_rows
                if row.get("distribution") in recorded_distributions
            }
        phase = "context" if filename.startswith("wideep_context") else "generation"
        replaced = 0
        if materialized_by_key:
            materialized_ep_sizes = {key[0] for key in materialized_by_key}
            output_rows = []
            for row in run_rows:
                if row.get("distribution") not in recorded_distributions:
                    output_rows.append(row)
                    continue
                if str(row.get("moe_ep_size", "")) in materialized_ep_sizes:
                    continue
                merged = apply_profile_free_hybrid_latency(row, phase=phase)
                if not merged.get("origin_latency"):
                    merged["origin_latency"] = str(row.get("latency", ""))
                merged["latency_policy_scope"] = "profile_free_hybrid_recorded"
                output_rows.append(merged)
            output_rows.extend(materialized_by_key[key] for key in sorted(materialized_by_key))
            replaced = len(materialized_by_key)
        else:
            output_rows = []
            for row in run_rows:
                if row.get("distribution") not in recorded_distributions:
                    output_rows.append(row)
                    continue
                merged = apply_profile_free_hybrid_latency(row, phase=phase)
                if not merged.get("origin_latency"):
                    merged["origin_latency"] = str(row.get("latency", ""))
                merged["latency_policy_scope"] = "profile_free_hybrid_recorded"
                output_rows.append(merged)
                replaced += 1

        if replaced:
            _preserve_raw_collector_file(run_dir, run_path)
            fields = list(run_fields)
            for key in [
                "origin_latency",
                "latency_policy_scope",
                "materialization_role",
                "materialization_source_family",
            ]:
                if key not in fields:
                    fields.append(key)
            _preserve_full_header_table(
                run_dir=run_dir,
                filename=run_path.name,
                fieldnames=fields,
                rows=output_rows,
            )
            _write_compact_perf_rows(run_path, output_rows)
            logger.info(
                "Installed DeepSeek-V3 profile-free Recorded latency into %s "
                "(replaced %d Recorded rows; materialized_rows=%d; compact table keeps latency as final value)",
                run_path,
                replaced,
                len(materialized_by_key),
            )


def _recorded_materialization_inputs(run_dir: Path) -> dict[str, Path | None]:
    return {
        "context_sparse": _resolve_recorded_materialization_source_path(
            env_names=(
                "COLLECTOR_DSV3_RECORDED_CONTEXT_SPARSE",
                "COLLECTOR_DSV3_EP8_CONTEXT_SPARSE",
            ),
            run_dir=run_dir,
            phase="context",
        ),
        "context_dense_noeplb": _resolve_recorded_materialization_source_path(
            env_names=(
                "COLLECTOR_DSV3_RECORDED_CONTEXT_DENSE_NOEPLB",
                "COLLECTOR_DSV3_EP8_CONTEXT_DENSE_NOEPLB",
            ),
            run_dir=run_dir,
            phase="context",
        ),
        "context_dense_eplb": _resolve_recorded_materialization_source_path(
            env_names=(
                "COLLECTOR_DSV3_RECORDED_CONTEXT_DENSE_EPLB",
                "COLLECTOR_DSV3_EP8_CONTEXT_DENSE_EPLB",
            ),
            run_dir=run_dir,
            phase="context",
        ),
        "generation_small_noeplb": _resolve_recorded_materialization_source_path(
            env_names=(
                "COLLECTOR_DSV3_RECORDED_GENERATION_SMALL_NOEPLB",
                "COLLECTOR_DSV3_EP8_GENERATION_SMALL_NOEPLB",
            ),
            run_dir=run_dir,
            phase="generation",
        ),
        "generation_small_eplb": _resolve_recorded_materialization_source_path(
            env_names=(
                "COLLECTOR_DSV3_RECORDED_GENERATION_SMALL_EPLB",
                "COLLECTOR_DSV3_EP8_GENERATION_SMALL_EPLB",
            ),
            run_dir=run_dir,
            phase="generation",
        ),
        "generation_main_noeplb": _resolve_recorded_materialization_source_path(
            env_names=(
                "COLLECTOR_DSV3_RECORDED_GENERATION_MAIN_NOEPLB",
                "COLLECTOR_DSV3_EP8_GENERATION_MAIN_NOEPLB",
            ),
            run_dir=run_dir,
            phase="generation",
        ),
        "generation_main_eplb": _resolve_recorded_materialization_source_path(
            env_names=(
                "COLLECTOR_DSV3_RECORDED_GENERATION_MAIN_EPLB",
                "COLLECTOR_DSV3_EP8_GENERATION_MAIN_EPLB",
            ),
            run_dir=run_dir,
            phase="generation",
        ),
    }


def _run_recorded_materialization_postprocess(args, ops: list[str] | None) -> None:
    if not _is_deepseek_v3_ep8_flow(args, ops):
        return
    if not _bool_env_enabled("COLLECTOR_DSV3_RECORDED_MATERIALIZATION", True):
        if logger is not None:
            logger.info("DeepSeek-V3 Recorded materialization is disabled by COLLECTOR_DSV3_RECORDED_MATERIALIZATION")
        return

    run_dir = Path(os.environ.get("COLLECTOR_LOG_DIR", ".")).resolve()
    required_local = [
        run_dir / "wideep_context_moe_perf.txt",
        run_dir / "wideep_generation_moe_perf.txt",
    ]
    missing_local = [path for path in required_local if not path.exists()]
    if missing_local:
        if logger is not None:
            logger.warning(
                "Skip DeepSeek-V3 Recorded materialization because current run is missing:\n  "
                + "\n  ".join(str(path) for path in missing_local)
            )
        return

    for path in required_local:
        _preserve_raw_collector_file(run_dir, path)

    inputs = _recorded_materialization_inputs(run_dir)
    missing_inputs = [
        name
        for name, path in inputs.items()
        if path is None or not path.exists()
    ]
    if missing_inputs:
        if logger is not None:
            logger.warning(
                "Skip DeepSeek-V3 Recorded materialization because AIC source inputs are missing:\n  "
                + "\n  ".join(missing_inputs)
                + "\nCurrent-run WideEP tables are used by default; explicit COLLECTOR_DSV3_RECORDED_* source env vars may override them."
            )
        return

    try:
        _preserve_dsv3_latency_source_manifest(run_dir=run_dir, inputs=inputs)
        for role, path in inputs.items():
            _preserve_dsv3_latency_source_input(run_dir=run_dir, role=role, path=path)

        from moe_recorded_materialization import materialize_recorded_wideep_tables_from_paths

        with _dsv3_materialized_source_dir(run_dir, "recorded_materialized_source") as validation_source_dir:
            materialize_recorded_wideep_tables_from_paths(
                base_validation_source=run_dir,
                context_sparse=inputs["context_sparse"],
                context_dense_noeplb=inputs["context_dense_noeplb"],
                context_dense_eplb=inputs["context_dense_eplb"],
                generation_small_noeplb=inputs["generation_small_noeplb"],
                generation_small_eplb=inputs["generation_small_eplb"],
                generation_main_noeplb=inputs["generation_main_noeplb"],
                generation_main_eplb=inputs["generation_main_eplb"],
                output_dir=validation_source_dir,
            )
            _install_recorded_latency_into_run_tables(
                run_dir=run_dir,
                validation_source_dir=validation_source_dir,
            )
            if logger is not None:
                if _keep_dsv3_materialized_sources():
                    logger.info("DeepSeek-V3 Recorded materialized source saved: %s", validation_source_dir)
                else:
                    logger.info(
                        "DeepSeek-V3 Recorded materialization installed into run tables "
                        "(temporary source not saved; set COLLECTOR_DSV3_KEEP_LATENCY_SOURCES=true to keep it)"
                    )
    except Exception:
        if logger is not None:
            logger.exception("DeepSeek-V3 Recorded materialization failed")
        return


def _run_ordinary_moe_materialization_postprocess(args, ops: list[str] | None) -> None:
    if not _is_deepseek_v3_ep8_flow(args, ops):
        return
    requested_ops = set(ops or [])
    if "moe" not in requested_ops:
        return
    if not _bool_env_enabled("COLLECTOR_DSV3_ORDINARY_MOE_MATERIALIZATION", True):
        if logger is not None:
            logger.info(
                "DeepSeek-V3 ordinary MoE materialization is disabled by "
                "COLLECTOR_DSV3_ORDINARY_MOE_MATERIALIZATION"
            )
        return

    run_dir = Path(os.environ.get("COLLECTOR_LOG_DIR", ".")).resolve()
    moe_perf = run_dir / "moe_perf.txt"
    replay_manifest = run_dir / "moe_token_distribution_replay" / "manifest.csv"
    if not moe_perf.exists() or not replay_manifest.exists():
        if logger is not None:
            logger.warning(
                "Skip DeepSeek-V3 ordinary MoE materialization because current "
                "run is missing moe_perf.txt or rank-local replay manifest."
            )
        return

    repo_root = _repo_root()
    try:
        from sglang.dsv3_ordinary_moe_materialization import (
            materialize_ordinary_moe_profile_free_source,
            summarize_ordinary_moe_replay_shape,
        )
    except ModuleNotFoundError:
        sys.path.insert(0, str(repo_root / "collector"))
        from sglang.dsv3_ordinary_moe_materialization import (
            materialize_ordinary_moe_profile_free_source,
            summarize_ordinary_moe_replay_shape,
        )

    try:
        with _dsv3_materialized_source_dir(run_dir, "ordinary_moe_materialized_source") as materialized_dir:
            shape_csv = materialized_dir / "ordinary_moe_replay_shape.csv"
            summarize_ordinary_moe_replay_shape(
                data_dir=run_dir,
                output=shape_csv,
            )
            materialize_ordinary_moe_profile_free_source(
                data_dir=run_dir,
                shape_csv=shape_csv,
                generation_shape_csv=shape_csv,
                output_dir=materialized_dir,
            )
            materialized_moe_perf = materialized_dir / "moe_perf.txt"
            if not materialized_moe_perf.exists():
                raise FileNotFoundError(materialized_moe_perf)
            _preserve_raw_collector_file(run_dir, moe_perf)
            materialized_fields, materialized_rows = _read_perf_rows(materialized_moe_perf)
            _preserve_full_header_table(
                run_dir=run_dir,
                filename=moe_perf.name,
                fieldnames=materialized_fields,
                rows=materialized_rows,
            )
            _write_compact_perf_rows(moe_perf, materialized_rows)
            if logger is not None:
                if _keep_dsv3_materialized_sources():
                    logger.info("DeepSeek-V3 ordinary MoE materialized source saved: %s", materialized_dir)
                else:
                    logger.info(
                        "DeepSeek-V3 ordinary MoE materialization installed as compact moe_perf.txt "
                        "(temporary source not saved; set COLLECTOR_DSV3_KEEP_LATENCY_SOURCES=true to keep it)"
                    )
    except subprocess.CalledProcessError as exc:
        if logger is not None:
            logger.warning(
                "DeepSeek-V3 ordinary MoE materialization failed: %s\nstdout:\n%s\nstderr:\n%s",
                exc,
                exc.stdout,
                exc.stderr,
            )
        return
    except Exception:
        if logger is not None:
            logger.exception("DeepSeek-V3 ordinary MoE materialization failed")
        return

    if logger is not None:
        logger.info(
            "Installed DeepSeek-V3 ordinary MoE profile-free Recorded latency "
            "into %s (compact table keeps latency as final value)",
            moe_perf,
        )


def _run_clean_latency_postprocess(args, ops: list[str] | None) -> None:
    """Install collector-native clean latency into final compact MoE tables."""

    if not _is_deepseek_v3_ep8_flow(args, ops):
        return
    if getattr(args, "smoke", False):
        if logger is not None:
            logger.info(
                "Skip DeepSeek-V3 clean latency in --smoke mode because sampled "
                "WideEP sources are intentionally incomplete."
            )
        return
    if not _dsv3_clean_latency_enabled():
        if logger is not None:
            logger.info("DeepSeek-V3 clean latency is disabled by COLLECTOR_DSV3_CLEAN_LATENCY")
        return

    run_dir = Path(os.environ.get("COLLECTOR_LOG_DIR", ".")).resolve()
    source_root = _sync_run_tables_to_latency_source_root(run_dir)
    wideep_required = [
        source_root / "wideep_context_moe_perf.txt",
        source_root / "wideep_generation_moe_perf.txt",
        source_root / "aic_latency_source_bundle" / "recorded_materialization_inputs",
    ]
    ordinary_required = [
        source_root / "moe_perf.txt",
        source_root / "raw_collector_source" / "moe_perf.txt",
    ]
    required = list(wideep_required)
    if any(path.exists() for path in ordinary_required):
        required.extend(ordinary_required)
    missing = [path for path in required if not path.exists()]
    if missing:
        if logger is not None:
            logger.warning(
                "Skip DeepSeek-V3 clean latency because required AIC source files are missing:\n  "
                + "\n  ".join(str(path) for path in missing)
                + "\nClean latency uses collector-owned AIC sources; set COLLECTOR_DSV3_KEEP_LATENCY_SOURCES=1 only when you need to persist them."
            )
        return
    install_filenames = ["wideep_context_moe_perf.txt", "wideep_generation_moe_perf.txt"]
    if all(path.exists() for path in ordinary_required):
        install_filenames.insert(0, "moe_perf.txt")
    elif logger is not None:
        logger.info(
            "DeepSeek-V3 clean latency will install WideEP tables only; ordinary MoE source files are absent."
        )

    repo_root = _repo_root()
    try:
        from moe_clean_latency import build_clean_latency_tables
    except ModuleNotFoundError:
        sys.path.insert(0, str(repo_root / "collector"))
        from moe_clean_latency import build_clean_latency_tables

    def _install_candidate_table(candidate_dir: Path, filename: str) -> None:
        candidate_path = candidate_dir / filename
        if not candidate_path.exists():
            raise FileNotFoundError(candidate_path)
        fields, rows = _read_perf_rows(candidate_path)
        _preserve_full_header_table(
            run_dir=run_dir,
            filename=filename,
            fieldnames=fields,
            rows=rows,
        )
        _write_compact_perf_rows(run_dir / filename, rows)

    try:
        if _keep_dsv3_latency_sources():
            candidate_dir = run_dir / "clean_latency_debug" / "candidate_full"
            build_clean_latency_tables(
                source_dir=source_root,
                candidate_dir=candidate_dir,
                write_origin_dir=True,
            )
            for filename in install_filenames:
                _install_candidate_table(candidate_dir, filename)
            if logger is not None:
                logger.info("DeepSeek-V3 clean latency candidate source saved: %s", candidate_dir)
        else:
            with tempfile.TemporaryDirectory(prefix="dsv3_clean_latency_") as tmp_dir:
                candidate_dir = Path(tmp_dir) / "candidate"
                build_clean_latency_tables(
                    source_dir=source_root,
                    candidate_dir=candidate_dir,
                    write_origin_dir=False,
                )
                for filename in install_filenames:
                    _install_candidate_table(candidate_dir, filename)
        if logger is not None:
            logger.info(
                "Installed DeepSeek-V3 clean latency into compact MoE tables "
                "(latency column is the final recorded latency value)."
            )
    except Exception:
        if logger is not None:
            logger.exception("DeepSeek-V3 clean latency postprocess failed")
        return


def _wideep_registry_for_backend(backend: str) -> list:
    module_name = f"collector.wideep.{backend}.registry"
    try:
        spec = importlib.util.find_spec(module_name)
    except ModuleNotFoundError:
        return []
    if spec is None:
        return []
    return list(importlib.import_module(module_name).REGISTRY)


def _registry_with_requested_wideep(registry: list, backend: str, ops: list[str] | None, case_plan=None) -> list:
    wideep_registry = _wideep_registry_for_backend(backend)
    if not wideep_registry:
        return registry

    requested_ops = set(ops if ops is not None else (case_plan.ops if case_plan is not None else []))
    requested_wideep_ops = requested_ops & {entry.op for entry in wideep_registry}
    if not requested_wideep_ops:
        return registry

    if logger is not None:
        logger.info(f"WideEP registry active for {backend}: {sorted(requested_wideep_ops)}")
    return [*registry, *wideep_registry]


class ResumeCheckpoint:
    """Tracks which tasks are done so a collection run can be resumed.

    Always writes checkpoint files.  When ``--resume`` is passed the existing
    checkpoint is loaded and done tasks are skipped; otherwise the checkpoint
    is overwritten from scratch (so a future ``--resume`` can pick up).
    """

    FLUSH_INTERVAL_SEC = 2.0

    def __init__(self, backend: str, module_name: str, run_func_name: str, checkpoint_dir: str):
        self.module_name = module_name
        self._dirty = False
        self._last_flush = 0.0
        self._metadata = {
            "schema": RESUME_SCHEMA_VERSION,
            "backend": backend,
            "module": module_name,
            "run_func": run_func_name,
        }
        self._done: set[str] = set()
        self._failed: set[str] = set()
        self._expected_failed: set[str] = set()

        safe_name = module_name.replace("/", "_").replace(":", "_")
        self._path = Path(checkpoint_dir).expanduser().resolve() / backend / f"{safe_name}.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def load_existing(self):
        """Load an existing checkpoint for resume.  Raises on mismatch."""
        if not self._path.exists():
            logger.info(f"{self.module_name}: no checkpoint found, starting fresh")
            return

        try:
            with open(self._path) as f:
                data = json.load(f)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint {self._path}: {e}. Run without --resume to start fresh."
            ) from e

        for key in ("schema", "backend", "module", "run_func"):
            if data.get(key) != self._metadata[key]:
                raise RuntimeError(
                    f"{self.module_name}: checkpoint mismatch "
                    f"({key}: {data.get(key)} != {self._metadata[key]}). "
                    "Run without --resume to start fresh."
                )

        self._done = set(data.get("done", []))
        self._failed = set(data.get("failed", []))
        self._expected_failed = set(data.get("expected_failed", []))
        logger.info(
            f"{self.module_name}: loaded checkpoint — {len(self._done)} passed, "
            f"{len(self._failed)} failed, {len(self._expected_failed)} expected failed"
        )

    # -- public API -------------------------------------------------------

    def filter_done(self, task_infos: list[dict], retry_failed: bool = False) -> list[dict]:
        """Return only tasks that need to run.

        By default, skips both passed and failed tasks. With retry_failed=True,
        previously failed tasks are retried.
        """
        skip_set = (
            (self._done | self._expected_failed)
            if retry_failed
            else (self._done | self._failed | self._expected_failed)
        )
        runnable = [t for t in task_infos if t["id"] not in skip_set]
        skipped_done = sum(1 for t in task_infos if t["id"] in self._done)
        skipped_failed = sum(1 for t in task_infos if t["id"] in self._failed)
        skipped_expected = sum(1 for t in task_infos if t["id"] in self._expected_failed)
        retrying = sum(1 for t in runnable if t["id"] in self._failed) if retry_failed else 0
        if skipped_done or skipped_failed or skipped_expected or retrying:
            parts = [f"skipping {skipped_done} passed"]
            parts.append(f"skipping {skipped_expected} expected failed")
            if retry_failed:
                parts.append(f"retrying {retrying} previously failed")
            else:
                parts.append(f"skipping {skipped_failed} failed")
            parts.append(f"running {len(runnable)}")
            logger.info(f"{self.module_name}: {', '.join(parts)}")
        return runnable

    def mark_passed(self, task_id: str):
        """Mark a task as successfully completed. Skipped on resume."""
        self._done.add(task_id)
        self._failed.discard(task_id)  # if it was previously failed, it passed now
        self._expected_failed.discard(task_id)
        self._dirty = True
        self.flush()

    def mark_failed(self, task_id: str):
        """Mark a task as attempted but failed. Retried on resume."""
        self._failed.add(task_id)
        self._dirty = True
        self.flush()

    def mark_expected_failed(self, task_id: str):
        """Mark a task as covered by an expected SM/framework exception."""
        self._expected_failed.add(task_id)
        self._failed.discard(task_id)
        self._dirty = True
        self.flush()

    # Keep mark_done as alias for backwards compat
    mark_done = mark_passed

    def flush(self, force: bool = False):
        if not self._dirty:
            return
        now = time.time()
        if not force and (now - self._last_flush) < self.FLUSH_INTERVAL_SEC:
            return

        data = {
            **self._metadata,
            "updated_at": datetime.now().isoformat(),
            "done": sorted(self._done),
            "failed": sorted(self._failed),
            "expected_failed": sorted(self._expected_failed),
        }
        tmp_path = self._path.with_suffix(".json.tmp")
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self._path)
        self._dirty = False
        self._last_flush = now


class ProfilerContext:
    """Context manager for profiling collector execution"""

    def __init__(self, backend: str, enabled: bool = False):
        self.enabled = enabled
        self.backend = backend
        self.profiler = None
        self.start_time = None
        self.log_dir = None

    def __enter__(self):
        if self.enabled:
            self.profiler = cProfile.Profile()
            self.profiler.enable()
            self.start_time = time.perf_counter()
            self.log_dir = os.environ.get("COLLECTOR_LOG_DIR", "")
            if not self.log_dir:
                self.log_dir = "."
            logger.info("Profiling enabled - running sequentially in main process (no parallel workers)")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled or self.profiler is None:
            return

        self.profiler.disable()
        profile_file = os.path.join(self.log_dir, f"collector_profile_{self.backend}.prof")
        self.profiler.dump_stats(profile_file)

        # Calculate elapsed time
        end_time = time.perf_counter()
        elapsed_time = end_time - self.start_time if self.start_time else 0

        logger.info("=" * 80)
        logger.info("PROFILING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total elapsed time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
        logger.info(f"Profile file: {profile_file}")
        logger.info("=" * 80)

        # Print slow operations ranked by tottime and cumtime
        stats = pstats.Stats(self.profiler)
        stats.strip_dirs()

        # Get top functions by tottime (time spent in the function itself, excluding subcalls)
        logger.info("Top 20 functions by tottime (time in function excluding subcalls):")
        logger.info("=" * 80)
        stream = io.StringIO()
        import sys

        old_stdout = sys.stdout
        sys.stdout = stream
        try:
            stats.sort_stats("tottime")
            stats.print_stats(20)
        finally:
            sys.stdout = old_stdout
        for line in stream.getvalue().split("\n"):
            if line.strip():
                logger.info(line)

        # Get top functions by cumtime (cumulative time including subcalls)
        logger.info("=" * 80)
        logger.info("Top 20 functions by cumtime (cumulative time including subcalls):")
        logger.info("=" * 80)
        stream = io.StringIO()
        sys.stdout = stream
        try:
            stats.sort_stats("cumtime")
            stats.print_stats(20)
        finally:
            sys.stdout = old_stdout
        for line in stream.getvalue().split("\n"):
            if line.strip():
                logger.info(line)

        logger.info("=" * 80)
        logger.info(f"Full profile saved to: {profile_file}")


def _expected_failure_for_task(task, task_index, expected_failure_context):
    if not expected_failure_context:
        return None
    from collector.model_cases import expected_failure_for_test_case

    return expected_failure_for_test_case(
        task,
        plan=expected_failure_context["plan"],
        full_module_name=expected_failure_context["full_module_name"],
        run_func_name=expected_failure_context["run_func_name"],
        runtime_version=expected_failure_context.get("runtime_version"),
        index=task_index or 0,
    )


def _expected_failure_label(details):
    if not details:
        return "expected collector exception"
    parts = [details.get("reason_type", "expected_exception")]
    if details.get("reference_source"):
        parts.append(f"source={details['reference_source']}")
    if details.get("label"):
        parts.append(f"label={details['label']}")
    if details.get("reason"):
        parts.append(details["reason"])
    return "; ".join(str(part) for part in parts if part)


def _is_cuda_fatal_exception(exc, torch_mod) -> bool:
    accelerator_error = getattr(torch_mod, "AcceleratorError", ())
    is_cuda_fatal = isinstance(exc, accelerator_error) if accelerator_error else False
    if not is_cuda_fatal:
        error_text = str(exc).lower()
        fatal_markers = (
            "illegal memory access",
            "unspecified launch failure",
            "cuda_error_launch_failed",
            "cublas_status_execution_failed",
            "cublas_status_internal_error",
            "cublas_status_alloc_failed",
        )
        is_cuda_fatal = any(marker in error_text for marker in fatal_markers)
    if not is_cuda_fatal:
        # DSLCudaRuntimeError from CUTLASS DSL also corrupts CUDA context but
        # is not a torch.AcceleratorError subclass.
        is_cuda_fatal = type(exc).__name__ == "DSLCudaRuntimeError"
    return is_cuda_fatal


def _summarize_expected_skips(skipped):
    counts = Counter(item.get("reason_type", "expected_exception") for item in skipped)
    return ", ".join(f"{reason_type}={count}" for reason_type, count in sorted(counts.items()))


def collect_module_safe(
    module_name,
    test_type,
    get_test_cases_func,
    run_func,
    num_processes,
    resume_options=None,
    expected_failure_context=None,
):
    """
    Safely collect module with comprehensive error handling

    Args:
        num_processes: Number of parallel processes to use. If 0, runs sequentially in main process.
    """
    full_name = f"{module_name}.{test_type}"
    logger.info(f"Starting collection: {full_name}")

    try:
        # Get test cases
        test_cases = get_test_cases_func()
        logger.info(f"Generated {len(test_cases)} test cases for {full_name}")

        # Run collection
        errors = parallel_run(
            test_cases,
            run_func,
            num_processes,
            full_name,
            resume_options=resume_options,
            expected_failure_context=expected_failure_context,
        )

        return errors

    except Exception as e:
        logger.exception(f"Failed to collect {full_name}")
        return [
            {
                "module": full_name,
                "error_type": "ModuleCollectionFailure",
                "error_message": str(e),
                "traceback": traceback.format_exc(),
            }
        ]


def worker(
    queue,
    device_id: int,
    func,
    progress_value,
    lock,
    error_queue=None,
    done_tasks=None,
    failed_tasks=None,
    expected_failed_tasks=None,
    module_name="unknown",
    current_task_ids=None,
    consumed_sentinel=None,
    expected_failure_context=None,
):
    """worker with automatic logging setup"""

    # Disable core dumps — GPU crashes are expected and handled via error_queue;
    # without this, each SIGSEGV/SIGABRT writes a multi-GB core file to disk.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    setup_warning_filters()  # Must run in each spawned process

    # Setup logging for this worker - reads config from environment automatically
    worker_logger = setup_logging(worker_id=device_id)

    # Setup signal handlers
    setup_signal_handlers(device_id)

    # Setup device
    torch_mod = _require_torch()
    device = torch_mod.device(f"{get_device_str()}:{device_id}")
    get_device_module().set_device(device)
    worker_logger.info(f"Worker {device_id} initialized for {module_name}")

    # Process tasks
    while True:
        task_info = queue.get()
        if task_info is None:
            if current_task_ids is not None:
                current_task_ids[device_id] = None
            if consumed_sentinel is not None:
                consumed_sentinel[device_id] = True
            worker_logger.debug("Received termination signal")
            break

        # Handle both old format (tuple) and new format (dict)
        if isinstance(task_info, dict):
            task_id = task_info.get("id", "unknown")
            task = task_info.get("params", task_info)
            task_index = task_info.get("index", 0)
        else:
            task = task_info
            task_id = create_test_case_id(task, "unknown", module_name)
            task_index = 0

        if current_task_ids is not None:
            current_task_ids[device_id] = task_id

        try:
            worker_logger.debug(f"Starting task {task_id}")
            func(*task, device=device)
            worker_logger.debug(f"Completed task {task_id}")

            # Mark done ONLY on success — failed tasks should be retried on resume
            if done_tasks is not None:
                try:
                    done_tasks[task_id] = True
                except Exception:
                    pass
            # Clear task ID on success so crash handler knows it completed
            if current_task_ids is not None:
                current_task_ids[device_id] = None
        except SystemExit as e:
            # EXIT_CODE_RESTART: task completed successfully, worker exits to free GPU memory
            # (e.g., MOE collectors call sys.exit(EXIT_CODE_RESTART) after finishing)
            if e.code == EXIT_CODE_RESTART:
                if done_tasks is not None:
                    try:
                        done_tasks[task_id] = True
                    except Exception:
                        pass
                if current_task_ids is not None:
                    current_task_ids[device_id] = None
            raise  # re-raise so the worker actually exits
        except Exception as e:
            expected_failure = _expected_failure_for_task(task, task_index, expected_failure_context)
            if expected_failure:
                worker_logger.warning(
                    f"Task {task_id} hit expected collector exception "
                    f"({_expected_failure_label(expected_failure)}); continuing"
                )
                if expected_failed_tasks is not None:
                    try:
                        expected_failed_tasks[task_id] = True
                    except Exception:
                        pass
                if current_task_ids is not None:
                    current_task_ids[device_id] = None
                for handler in worker_logger.handlers:
                    handler.flush()
                if _is_cuda_fatal_exception(e, torch_mod):
                    worker_logger.warning(
                        f"Expected fatal {type(e).__name__} encountered on task {task_id}. "
                        f"Worker {device_id} exiting to reset GPU context."
                    )
                    for handler in worker_logger.handlers:
                        handler.flush()
                    exit(0)
                continue

            # Build comprehensive error info
            error_info = {
                "module": module_name,
                "device_id": device_id,
                "task_id": task_id,
                "task_params": str(task),
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat(),
            }

            # Report error to queue BEFORE any exit
            if error_queue:
                error_queue.put(error_info)

            worker_logger.exception(f"Task {task_id} failed")

            # Track failed task for checkpoint
            if failed_tasks is not None:
                try:
                    failed_tasks[task_id] = True
                except Exception:
                    pass
            # Clear task ID so crash handler knows it was handled
            if current_task_ids is not None:
                current_task_ids[device_id] = None

            # Force flush logs before any potential exit
            for handler in worker_logger.handlers:
                handler.flush()

            if _is_cuda_fatal_exception(e, torch_mod):
                worker_logger.warning(
                    f"Fatal {type(e).__name__} encountered on task {task_id}. "
                    f"Worker {device_id} exiting to reset GPU context. "
                    f"Progress: {progress_value.value}"
                )
                # Flush logs again after warning
                for handler in worker_logger.handlers:
                    handler.flush()
                # Exiting with non-zero code will add an additional error to the summary,
                # which we don't want (error already reported above).
                exit(0)
        finally:
            with lock:
                progress_value.value += 1

            # Periodic memory cleanup to reduce fragmentation
            if progress_value.value % 100 == 0:
                import gc

                gc.collect()
                get_device_module().empty_cache()


def parallel_run(tasks, func, num_processes, module_name="unknown", resume_options=None, expected_failure_context=None):
    """parallel runner with error collection

    Args:
        num_processes: Number of parallel processes. If 0, runs sequentially in main process.
    """
    # func may be a functools.partial (perf_filename bound by collect_ops),
    # which lacks __name__. Fall back to partial.func to get the wrapped function.
    func_name = getattr(func, "__name__", None) or getattr(func, "func", func).__name__
    raw_task_infos = []
    for i, task in enumerate(tasks):
        if isinstance(task, dict) and "id" in task and "params" in task:
            task_id = task["id"]
            task_params = task["params"]
        else:
            task_id = create_test_case_id(task, func_name, module_name)
            task_params = task
        raw_task_infos.append({"id": task_id, "params": task_params, "index": i})
    task_info_by_id = {task_info["id"]: task_info for task_info in raw_task_infos}

    checkpoint_dir = (
        resume_options.get("checkpoint_dir", ".collector_checkpoint") if resume_options else ".collector_checkpoint"
    )
    resume_tracker = ResumeCheckpoint(
        backend=resume_options.get("backend", "unknown") if resume_options else "unknown",
        module_name=module_name,
        run_func_name=func_name,
        checkpoint_dir=checkpoint_dir,
    )

    if resume_options and resume_options.get("resume"):
        resume_tracker.load_existing()
        retry_failed = resume_options.get("retry_failed", False)
        task_infos = resume_tracker.filter_done(raw_task_infos, retry_failed=retry_failed)
    else:
        task_infos = raw_task_infos

    if not task_infos:
        logger.info(f"{module_name}: no tasks to run")
        return []

    queue = mp.Queue()
    error_queue = mp.Queue()
    processes = []

    manager = mp.Manager()
    progress_value = manager.Value("i", 0)
    lock = manager.Lock()

    # Track process health
    process_stats = {i: {"restarts": 0, "errors": []} for i in range(num_processes)}

    # Per-worker flag: True once a worker has consumed its None sentinel.
    # Used to decide whether a replacement sentinel is needed on restart.
    consumed_sentinel = manager.dict(dict.fromkeys(range(num_processes), False))
    current_task_ids = manager.dict(dict.fromkeys(range(num_processes), None))
    # Synchronous record of completed task IDs.  Workers write here via
    # manager RPC in their finally block — same mechanism as progress_value,
    # so it is guaranteed to be visible before the worker touches the next
    # task.  Unlike mp.Queue (async feeder thread) this cannot be lost when
    # a worker is killed by a signal on a subsequent task.
    done_tasks = manager.dict()
    failed_tasks = manager.dict()
    expected_failed_tasks = manager.dict()

    def start_process(device_id):
        p = mp.Process(
            target=worker,
            args=(
                queue,
                device_id,
                func,
                progress_value,
                lock,
                error_queue,
                done_tasks,
                failed_tasks,
                expected_failed_tasks,
                module_name,
                current_task_ids,
                consumed_sentinel,
                expected_failure_context,
            ),
        )
        p.start()
        logger.info(f"Started worker process {p.pid} on device {device_id}")
        return p

    def create_process_exit_error(device_id, exit_code):
        if exit_code in (None, 0, EXIT_CODE_RESTART):
            return None

        if exit_code < 0:
            signum = -exit_code
            try:
                signame = signal.Signals(signum).name
            except Exception:
                signame = f"SIG{signum}"
            reason = f"terminated by signal {signum} ({signame})"
            error_type = "WorkerSignalCrash"
        else:
            reason = f"exited with status {exit_code}"
            error_type = "WorkerAbnormalExit"

        logger.error(f"Process {device_id} ({module_name}) {reason}")

        return {
            "module": module_name,
            "device_id": device_id,
            "task_id": "process_exit",
            "task_params": None,
            "error_type": error_type,
            "error_message": reason,
            "traceback": "",
            "exit_code": exit_code,
            "timestamp": datetime.now().isoformat(),
        }

    def sync_done_to_checkpoint():
        for task_id in list(done_tasks.keys()):
            resume_tracker.mark_passed(task_id)
            try:
                del done_tasks[task_id]
            except KeyError:
                pass
        for task_id in list(failed_tasks.keys()):
            resume_tracker.mark_failed(task_id)
            try:
                del failed_tasks[task_id]
            except KeyError:
                pass
        for task_id in list(expected_failed_tasks.keys()):
            resume_tracker.mark_expected_failed(task_id)
            try:
                del expected_failed_tasks[task_id]
            except KeyError:
                pass

    # Start processes
    for device_id in range(num_processes):
        processes.append(start_process(device_id))

    # Queue tasks with IDs
    for task_info in task_infos:
        queue.put(task_info)

    # Add termination signals
    for _ in range(len(processes)):
        queue.put(None)

    # Monitor progress with error collection
    errors = []

    with tqdm(total=len(task_infos), desc=f"{module_name}", dynamic_ncols=True, leave=True) as pbar:
        last_progress = 0
        stall_count = 0
        last_error_count = 0

        if num_processes == 0:
            # Special handling for --profile
            # Run tasks sequentially in main process
            torch_mod = _require_torch()
            device = torch_mod.device(f"{get_device_str()}:0")
            get_device_module().set_device(device)

            for task_info in task_infos:
                task_id = task_info["id"]
                task_params = task_info["params"]

                try:
                    func(*task_params, device=device)
                    resume_tracker.mark_passed(task_id)
                except Exception as e:
                    expected_failure = _expected_failure_for_task(
                        task_params,
                        task_info.get("index", 0),
                        expected_failure_context,
                    )
                    if expected_failure:
                        resume_tracker.mark_expected_failed(task_id)
                        logger.warning(
                            f"Task {task_id} hit expected collector exception "
                            f"({_expected_failure_label(expected_failure)}); continuing"
                        )
                        pbar.update(1)
                        progress_value.value += 1
                        resume_tracker.flush()
                        continue
                    resume_tracker.mark_failed(task_id)
                    error_info = {
                        "module": module_name,
                        "device_id": 0,
                        "task_id": task_id,
                        "task_params": str(task_params),
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "traceback": traceback.format_exc(),
                        "timestamp": datetime.now().isoformat(),
                    }
                    errors.append(error_info)
                    logger.exception(f"Task {task_id} failed")

                pbar.update(1)
                progress_value.value += 1
                if len(errors) > 0:
                    pbar.set_postfix({"errors": len(errors)})
                resume_tracker.flush()
            resume_tracker.flush(force=True)

        while progress_value.value < len(task_infos):
            # Drain errors
            while not error_queue.empty():
                error = error_queue.get()
                errors.append(error)
                process_stats[error["device_id"]]["errors"].append(error["task_id"])
            sync_done_to_checkpoint()

            # Update postfix only if count changed
            if len(errors) != last_error_count:
                pbar.set_postfix({"errors": len(errors)})
                last_error_count = len(errors)

            if progress_value.value == last_progress:
                stall_count += 1
                if stall_count > STALL_THRESHOLD:
                    logger.warning(f"Progress stalled at {progress_value.value}/{len(task_infos)}")
                    stall_count = 0
            else:
                stall_count = 0
                last_progress = progress_value.value

            # Check process health — only restart if there is still work
            # remaining.  Workers that consumed a None sentinel or finished
            # via sys.exit(EXIT_CODE_RESTART) should not be restarted once
            # all tasks are dispatched, otherwise the new worker blocks
            # forever on queue.get().
            for i, p in enumerate(processes):
                if p is None:
                    continue

                if not p.is_alive():
                    exit_code = p.exitcode
                    active_task_id = current_task_ids.get(i)
                    process_stats[i]["restarts"] += 1
                    if exit_code == EXIT_CODE_RESTART:
                        logger.debug(
                            f"Process {i} completed task and exited normally for release gpu memory"
                            f"(completed tasks: {process_stats[i]['restarts']})"
                        )
                    else:
                        logger.warning(
                            f"Process {i} died (exit code: {exit_code}, "
                            f"restarts: {process_stats[i]['restarts']}, "
                            f"errors: {len(process_stats[i]['errors'])})"
                        )

                    # Mark active task as failed if the process died while running it
                    expected_failure = None
                    if active_task_id is not None and active_task_id not in done_tasks:
                        task_info = task_info_by_id.get(active_task_id)
                        if task_info is not None:
                            expected_failure = _expected_failure_for_task(
                                task_info["params"],
                                task_info.get("index", 0),
                                expected_failure_context,
                            )
                        if expected_failure:
                            logger.warning(
                                f"Task {active_task_id} exited through an expected collector exception "
                                f"({_expected_failure_label(expected_failure)}); continuing"
                            )
                            try:
                                expected_failed_tasks[active_task_id] = True
                            except Exception:
                                pass
                        else:
                            try:
                                failed_tasks[active_task_id] = True
                            except Exception:
                                pass
                        current_task_ids[i] = None
                        with lock:
                            progress_value.value += 1

                    crash_error = (
                        None
                        if active_task_id is not None and expected_failure
                        else create_process_exit_error(i, exit_code)
                    )
                    if crash_error:
                        errors.append(crash_error)
                        process_stats[i]["errors"].append("process_exit")
                        pbar.set_postfix({"errors": len(errors)})
                        last_error_count = len(errors)

                    if process_stats[i]["restarts"] > 8192:
                        logger.error(f"Process {i} exceeded restart limit, not restarting")
                        processes[i] = None
                        continue

                    if consumed_sentinel.get(i, False):
                        processes[i] = None
                        continue

                    remaining = len(task_infos) - progress_value.value
                    if remaining > 0:
                        processes[i] = start_process(i)
                    else:
                        processes[i] = None

            current = progress_value.value
            if current > pbar.n:
                pbar.update(current - pbar.n)

            resume_tracker.flush()
            time.sleep(0.5)
        sync_done_to_checkpoint()

    # Collect remaining errors
    while not error_queue.empty():
        errors.append(error_queue.get())
    sync_done_to_checkpoint()
    resume_tracker.flush(force=True)

    # Wait for processes
    for p in processes:
        if p is None:
            continue
        p.join(timeout=42)
        if p.is_alive():
            logger.warning(f"Process {p.pid} did not terminate, forcing...")
            p.terminate()

    # Shutdown manager to clean up resources (semaphores, etc.)
    manager.shutdown()

    # Log summary
    if errors:
        log_dir = os.environ.get("COLLECTOR_LOG_DIR", "")
        logger.error(f"{module_name}: Completed with {len(errors)} errors")
        error_file = f"{log_dir}/errors_{module_name}.json"
        save_error_report(errors, error_file)
        logger.error(f"Error details saved to {error_file}")
    else:
        logger.info(f"{module_name}: Completed successfully with no errors")

    return errors


def collect_ops(
    num_processes: int,
    collections: list[dict],
    runtime_version: str | None = None,
    limit: int | None = None,
    shuffle: bool = False,
    shuffle_seed: int = 42,
    backend: str = "unknown",
    resume_options: dict | None = None,
    model_path: str | None = None,
    case_plan=None,
) -> list[dict]:
    """Run collection for a list of resolved collection entries.

    Each entry must have: name, type, module, get_func, run_func.
    Version resolution and op filtering are handled upstream by
    version_resolver.build_collections(). If runtime_version is provided,
    per-module __compat__ is validated and incompatible ops fail explicitly.
    If limit is provided, the number of test cases is limited to the limit.
    If shuffle is True, the test cases are shuffled with the given seed.
    """

    class CompatibilityError(RuntimeError):
        """Raised when a resolved collector module is incompatible."""

    check_compat = None
    if runtime_version:
        from collector.version_resolver import _check_compat as check_compat

    @contextlib.contextmanager
    def _collector_model_path(model_path: str | None):
        previous = os.environ.get("COLLECTOR_MODEL_PATH")
        if model_path:
            os.environ["COLLECTOR_MODEL_PATH"] = model_path
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("COLLECTOR_MODEL_PATH", None)
            else:
                os.environ["COLLECTOR_MODEL_PATH"] = previous

    def _get_test_cases(get_func, model_path: str | None):
        if not model_path:
            return get_func()
        with _collector_model_path(model_path):
            sig = signature(get_func)
            params = sig.parameters
            if "model_path" in params or any(param.kind == Parameter.VAR_KEYWORD for param in params.values()):
                return get_func(model_path=model_path)
            return get_func()

    all_errors = []

    for collection in collections:
        try:
            op_plan = case_plan.op_cases.get(collection["type"]) if case_plan is not None else None
            if case_plan is not None and op_plan is None:
                logger.info(f"Skipping {collection['name']}.{collection['type']} — not in collector v2 case plan")
                continue
            module_name = collection["module"]
            get_module = __import__(module_name, fromlist=[collection["get_func"]])
            run_module = __import__(module_name, fromlist=[collection["run_func"]])

            # Fail this op explicitly if declared compatibility doesn't match runtime.
            if check_compat:
                declared = getattr(get_module, "__compat__", None)
                if declared:
                    try:
                        if not check_compat(declared, runtime_version):
                            if _xpu_available():
                                # Disable vllm xpu runtime version check for now
                                logger.warning(
                                    f"module {module_name} declares __compat__={declared!r}, \
                                    runtime is v{runtime_version}"
                                )
                            else:
                                raise CompatibilityError(
                                    f"module {module_name} declares __compat__={declared!r}, \
                                        runtime is v{runtime_version}"
                                )
                    except ValueError as e:
                        raise CompatibilityError(f"invalid __compat__ {declared!r}: {e}") from e

            get_func = getattr(get_module, collection["get_func"])
            run_func = getattr(run_module, collection["run_func"])
            resolved_perf_filename = _resolve_perf_filename(collection["perf_filename"])
            run_func = functools.partial(
                run_func,
                perf_filename=resolved_perf_filename,
            )

            def get_func_with_limit(get_func=get_func):
                previous_output_dir = os.environ.get("COLLECTOR_CURRENT_OUTPUT_DIR")
                os.environ["COLLECTOR_CURRENT_OUTPUT_DIR"] = os.path.dirname(
                    os.path.abspath(str(resolved_perf_filename))
                )
                try:
                    cases = _get_test_cases(get_func, model_path)
                finally:
                    if previous_output_dir is None:
                        os.environ.pop("COLLECTOR_CURRENT_OUTPUT_DIR", None)
                    else:
                        os.environ["COLLECTOR_CURRENT_OUTPUT_DIR"] = previous_output_dir
                skipped = []
                if op_plan is not None:
                    from collector.model_cases import filter_test_cases_with_report

                    full_module_name = f"{collection['name']}.{collection['type']}"
                    run_func_name = getattr(run_func, "__name__", None) or getattr(run_func, "func", run_func).__name__
                    before_count = len(cases)
                    cases, skipped = filter_test_cases_with_report(
                        cases,
                        plan=op_plan,
                        full_module_name=full_module_name,
                        run_func_name=run_func_name,
                        runtime_version=runtime_version,
                    )
                    message = f"{full_module_name}: collector v2 case plan kept {len(cases)}/{before_count} cases"
                    if skipped:
                        message += (
                            f"; skipped {len(skipped)} expected SM exceptions ({_summarize_expected_skips(skipped)})"
                        )
                    logger.info(message)
                if shuffle:
                    rng = random.Random(shuffle_seed)
                    rng.shuffle(cases)
                if limit is not None:
                    if collection["type"] == "moe_token_distribution" and limit > 0:
                        by_ep_size = {}
                        for case in cases:
                            if not isinstance(case, (list, tuple)) or len(case) < 4:
                                continue
                            ep_size = case[3]
                            current = by_ep_size.get(ep_size)
                            if current is None or case[0] < current[0]:
                                by_ep_size[ep_size] = case
                        if by_ep_size:
                            selected = [by_ep_size[ep_size] for ep_size in sorted(by_ep_size)[:limit]]
                            remaining = [case for case in cases if case not in selected]
                            cases = [*selected, *remaining[: max(0, limit - len(selected))]]
                            return cases[:limit]
                    if collection["type"] == "moe" and limit > 0:
                        recorded_cases = [
                            case
                            for case in cases
                            if isinstance(case, (list, tuple))
                            and len(case) > 9
                            and (
                                case[9] == "recorded"
                                or "_rank_local_" in str(case[9])
                            )
                        ]
                        if recorded_cases:
                            first_recorded = recorded_cases[0]
                            remaining = [case for case in cases if case is not first_recorded]
                            cases = [first_recorded, *remaining[: max(0, limit - 1)]]
                            return cases
                    cases = cases[:limit]
                return cases

            merged_resume = {**(resume_options or {}), "backend": backend}
            collection_num_processes = collection.get("num_processes")
            if collection_num_processes is None:
                collection_num_processes = num_processes
            else:
                logger.info(
                    f"{collection['name']}.{collection['type']}: overriding worker count "
                    f"{num_processes} -> {collection_num_processes}"
                )
            errors = collect_module_safe(
                collection["name"],
                collection["type"],
                get_func_with_limit,
                run_func,
                collection_num_processes,
                resume_options=merged_resume,
                expected_failure_context=(
                    {
                        "plan": op_plan,
                        "full_module_name": f"{collection['name']}.{collection['type']}",
                        "run_func_name": getattr(run_func, "__name__", None)
                        or getattr(run_func, "func", run_func).__name__,
                        "runtime_version": runtime_version,
                    }
                    if op_plan is not None
                    else None
                ),
            )
            all_errors.extend(errors)

        except Exception as e:
            logger.exception(f"Failed to process {collection['name']}.{collection['type']}")
            all_errors.append(
                {
                    "module": f"{collection['name']}.{collection['type']}",
                    "error_type": "CompatibilityError" if isinstance(e, CompatibilityError) else type(e).__name__,
                    "error_message": str(e),
                    "traceback": traceback.format_exc(),
                }
            )

    return all_errors


def collect_sglang(
    num_processes: int,
    ops: list[str] | None = None,
    limit: int | None = None,
    shuffle: bool = False,
    resume_options: dict | None = None,
    model_path: str | None = None,
    case_plan=None,
):
    """Collect performance data for SGLang with enhanced error tracking"""
    from collector.sglang.registry import REGISTRY
    from collector.version_resolver import build_collections

    os.environ["FLASHINFER_LOG_LEVEL"] = "ERROR"

    try:
        from importlib.metadata import version as get_version

        package_version = get_version("sglang")
        version = package_version
        if package_version in {"0.0.0", "0.0.0.dev0"}:
            # Editable/source SGLang builds can carry placeholder package
            # metadata even though their API is a released branch.  Reuse the
            # same structural detector as the collectors so compatibility
            # routing reflects the runtime API rather than stale wheel metadata.
            from collector.sglang.version_compat import sglang_version_branch

            branch = sglang_version_branch()
            version = "0.5.12" if branch == "current" else "0.5.10"
            logger.info(
                "SGLang package version %s resolved to API version %s (%s branch)",
                package_version,
                version,
                branch,
            )
        else:
            logger.info(f"SGLang version: {version}")
    except Exception:
        logger.exception("SGLang is not installed")
        return

    registry = _registry_with_requested_wideep(REGISTRY, "sglang", ops, case_plan)
    collections = build_collections(registry, "sglang", version, ops, logger=logger)
    all_errors = collect_ops(
        num_processes,
        collections,
        version,
        limit=limit,
        shuffle=shuffle,
        backend="sglang",
        resume_options=resume_options,
        model_path=model_path,
        case_plan=case_plan,
    )

    generate_collection_summary(all_errors, "sglang", version)


def collect_vllm(
    num_processes: int,
    ops: list[str] | None = None,
    limit: int | None = None,
    shuffle: bool = False,
    resume_options: dict | None = None,
    model_path: str | None = None,
    case_plan=None,
):
    """Collect performance data for vLLM"""
    from collector.version_resolver import build_collections

    if _cuda_available():
        from collector.vllm.registry import REGISTRY
    elif _xpu_available():
        from collector.vllm.registry import REGISTRY_XPU as REGISTRY
    else:
        raise RuntimeError("No supported hardware detected. Neither CUDA nor XPU is available.")

    try:
        from vllm.version import __version__ as vllm_version

        version = vllm_version
    except Exception:
        logger.exception("vLLM is not installed. Please install it from https://github.com/vllm-project/vllm")
        return

    registry = _registry_with_requested_wideep(REGISTRY, "vllm", ops, case_plan)
    collections = build_collections(registry, "vllm", version, ops, logger=logger)
    all_errors = collect_ops(
        num_processes,
        collections,
        version,
        limit=limit,
        shuffle=shuffle,
        backend="vllm",
        resume_options=resume_options,
        model_path=model_path,
        case_plan=case_plan,
    )

    generate_collection_summary(all_errors, "vllm", version)


def collect_trtllm(
    num_processes: int,
    ops: list[str] | None = None,
    limit: int | None = None,
    shuffle: bool = False,
    resume_options: dict | None = None,
    model_path: str | None = None,
    case_plan=None,
):
    """Collect performance data for TensorRT LLM with enhanced error tracking"""
    from collector.trtllm.registry import REGISTRY
    from collector.version_resolver import build_collections

    os.environ["TLLM_LOG_LEVEL"] = "ERROR"
    os.environ["TRTLLM_DG_ENABLED"] = "1"
    os.environ["FLASHINFER_LOG_LEVEL"] = "ERROR"

    try:
        with (
            open(os.devnull, "w") as _null,
            contextlib.redirect_stdout(_null),
            contextlib.redirect_stderr(_null),
        ):
            import tensorrt_llm
        version = tensorrt_llm.__version__
        logger.info(f"TensorRT LLM version: {version}")
    except Exception:
        logger.exception("TensorRT LLM is not installed")
        return

    registry = _registry_with_requested_wideep(REGISTRY, "trtllm", ops, case_plan)
    collections = build_collections(registry, "trtllm", version, ops, logger=logger)
    all_errors = collect_ops(
        num_processes,
        collections,
        version,
        limit=limit,
        shuffle=shuffle,
        backend="trtllm",
        resume_options=resume_options,
        model_path=model_path,
        case_plan=case_plan,
    )

    generate_collection_summary(all_errors, "trtllm", version)


def generate_collection_summary(all_errors, backend, version):
    """Generate comprehensive collection summary"""
    summary = {
        "backend": backend,
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "total_errors": len(all_errors),
        "errors_by_module": {},
        "errors_by_type": {},
    }

    for error in all_errors:
        module = error.get("module", "unknown")
        error_type = error.get("error_type", "unknown")

        summary["errors_by_module"][module] = summary["errors_by_module"].get(module, 0) + 1
        summary["errors_by_type"][error_type] = summary["errors_by_type"].get(error_type, 0) + 1

    log_dir = os.environ.get("COLLECTOR_LOG_DIR", "")

    # Save summary
    summary_file = f"{log_dir}/collection_summary_{backend}.json"
    with open(summary_file, "w") as f:
        json.dump({"summary": summary, "errors": all_errors}, f, indent=2)

    # Print summary
    logger.info("=" * 60)
    logger.info(f"COLLECTION SUMMARY - {backend} v{version}")
    logger.info("=" * 60)
    logger.info(f"Total errors: {summary['total_errors']}")

    if summary["errors_by_module"]:
        logger.info("\nErrors by module:")
        for module, count in sorted(summary["errors_by_module"].items()):
            logger.info(f"  {module}: {count}")

    if summary["errors_by_type"]:
        logger.info("\nErrors by type:")
        for error_type, count in sorted(summary["errors_by_type"].items()):
            logger.info(f"  {error_type}: {count}")

    logger.info(f"\nDetailed error report saved to: {summary_file}")


def _all_op_names() -> list[str]:
    """Collect all unique op names across normal and WideEP registries."""
    from collector.sglang.registry import REGISTRY as SGLANG_REG
    from collector.trtllm.registry import REGISTRY as TRTLLM_REG
    from collector.vllm.registry import REGISTRY as VLLM_REG

    seen = set()
    ops = []
    registries = [
        TRTLLM_REG,
        VLLM_REG,
        SGLANG_REG,
        _wideep_registry_for_backend("trtllm"),
        _wideep_registry_for_backend("vllm"),
        _wideep_registry_for_backend("sglang"),
    ]
    for reg in registries:
        for entry in reg:
            if entry.op not in seen:
                seen.add(entry.op)
                ops.append(entry.op)
    return ops


def main():
    global logger
    parser = argparse.ArgumentParser(description="Collect performance data for backends")
    parser.add_argument("--backend", type=str, choices=["trtllm", "sglang", "vllm"], default="trtllm")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--ops",
        nargs="*",
        type=str,
        choices=_all_op_names(),
        help="Run only specified collection items. Leave empty to run all. "
        "Available ops vary by backend — see backend-specific registry.py for details.",
        default=None,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: randomly sample 4 test cases per op to verify the collector runs end-to-end",
    )
    parser.add_argument(
        "--measure_power",
        action="store_true",
        help="Enable power monitoring during kernel execution (samples at 100ms intervals)",
    )
    parser.add_argument(
        "--power_test_duration_sec",
        type=float,
        default=1.0,
        help="Minimum duration for kernel runs when power measurement is enabled (default: 1.0s)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume collection from checkpoint, skipping passed and failed tasks",
    )
    parser.add_argument(
        "--resume-retry-failed",
        action="store_true",
        help="When resuming, retry previously failed tasks instead of skipping them. Requires --resume.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=".collector_checkpoint",
        help="Directory for per-module resume checkpoints (default: .collector_checkpoint)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of test cases per collection (useful for debugging)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle test cases before applying --limit (uses seed 42 for reproducibility)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Collector v2 model path (for example 'MiniMaxAI/MiniMax-M2.5'). "
        "When set, collect.py resolves collector/cases/models/<architecture>_cases.yaml by model alias, "
        "then runs only the planned ops/cases.",
    )
    parser.add_argument(
        "--model-architecture",
        type=str,
        default=None,
        help="Collector v2 model architecture (for example 'Qwen3MoeForCausalLM'). "
        "Defaults to resolving the architecture case file from --model-path aliases.",
    )
    parser.add_argument(
        "--model-cases",
        type=str,
        default=None,
        help="Optional path to a model cases YAML file. Defaults to collector/cases/models/<architecture>_cases.yaml.",
    )
    parser.add_argument(
        "--model-cases-full",
        action="store_true",
        help="Collector v2 full mode: aggregate base op cases plus every model cases YAML file.",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="GPU type for resolving collector v2 SM exceptions, for example b200_sxm. "
        "The SM version is read from src/aiconfigurator/systems/<gpu>.yaml unless --sm is provided.",
    )
    parser.add_argument(
        "--sm",
        type=int,
        default=None,
        help="Explicit SM version for collector v2 exceptions, for example 100. Overrides --gpu SM resolution.",
    )
    parser.add_argument(
        "--sm-exceptions",
        type=str,
        default=None,
        help=(
            "Optional path to an SM exceptions YAML file. "
            "Defaults to collector/cases/sm_exceptions/sm<version>_exceptions.yaml."
        ),
    )
    parser.add_argument(
        "--gpu-exceptions",
        type=str,
        default=None,
        help="Deprecated alias for --sm-exceptions.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the collector v2 case plan and exit without running collectors.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile the collector run and save output ",
    )
    parser.add_argument(
        "--sglang-version-branch",
        choices=["auto", "v0.5.10", "0.5.10", "v0.5.12", "0.5.12"],
        default=os.environ.get("COLLECTOR_SGLANG_VERSION_BRANCH", "auto"),
        help=(
            "SGLang API branch for collector compatibility. Use v0.5.10 for the "
            "old compressed attention backend; use v0.5.12 for the dsv4 backend."
        ),
    )
    parser.add_argument(
        "--keep-csv",
        action="store_true",
        help="Keep collector CSV staging files instead of finalizing *_perf.txt outputs to parquet.",
    )
    args = parser.parse_args()
    ops = args.ops
    os.environ["COLLECTOR_SGLANG_VERSION_BRANCH"] = args.sglang_version_branch
    _dsv4_auto_expand = False
    case_plan = None
    logger_message = None
    if args.plan_only and not (args.model_path or args.model_architecture or args.model_cases or args.model_cases_full):
        parser.error("--plan-only requires --model-path, --model-architecture, --model-cases, or --model-cases-full")
    if args.model_path or args.model_architecture or args.model_cases or args.model_cases_full:
        from collector.case_generator import is_dsv4_attention_model
        from collector.model_cases import build_collection_case_plan

        if args.model_path:
            os.environ["COLLECTOR_MODEL_PATH"] = args.model_path
        else:
            os.environ.pop("COLLECTOR_MODEL_PATH", None)

        case_plan = build_collection_case_plan(
            backend=args.backend,
            model_path=args.model_path,
            model_architecture=args.model_architecture,
            gpu_type=args.gpu,
            sm_version=args.sm,
            model_cases_path=args.model_cases,
            sm_exceptions_path=args.sm_exceptions or args.gpu_exceptions,
            full=args.model_cases_full,
        )
        if case_plan.model_path:
            os.environ["COLLECTOR_MODEL_PATH"] = case_plan.model_path

        planned_ops = case_plan.ops
        if args.ops is None:
            ops = planned_ops
        else:
            requested_ops = set(args.ops)
            ops = [op for op in planned_ops if op in requested_ops]
            missing_ops = requested_ops - set(ops)
            if missing_ops:
                parser.error(
                    "Requested ops are not present in the collector v2 case plan: " + ", ".join(sorted(missing_ops))
                )

        if (args.model_path or args.model_architecture) and not case_plan.model_cases_paths:
            logger_message = (
                "No collector v2 model cases YAML found for "
                f"model_path={args.model_path!r}, model_architecture={args.model_architecture!r}; "
                "using base op cases only plus legacy model filtering."
            )

        # DeepSeek-V4 Flash/Pro special-case: when no explicit ``--ops`` is
        # given, collect the full set of AIC data needed by the model. The
        # attention path uses prefix-aware full modules, so the old sparse
        # kernel correction files are intentionally not part of the default.
        if args.ops is None and args.model_path and is_dsv4_attention_model(args.model_path):
            ops = [
                "gemm",
                "moe",
                "mhc_module",
                "dsv4_csa_context_module",
                "dsv4_hca_context_module",
                "dsv4_csa_generation_module",
                "dsv4_hca_generation_module",
            ]
            _dsv4_auto_expand = True
        if args.plan_only:
            log_dict = case_plan.to_log_dict()
            log_dict["ops"] = ops
            print(json.dumps(log_dict, indent=2))
            return
    else:
        os.environ.pop("COLLECTOR_MODEL_PATH", None)
        os.environ.pop("COLLECTOR_LOCAL_MODEL_PATH", None)

    # Setup logging - debug flag is handled inside setup_logging
    if logger is None:
        # Use short label when V4-Flash auto-expanded ops to several names
        # (the joined scope may exceed Linux filename length limit).
        if _dsv4_auto_expand:
            log_scope = ["dsv4"]
        elif args.model_cases_full:
            log_scope = ["model_cases_full"]
        else:
            log_scope = ops if ops else ["all"]
        logger = setup_logging(scope=log_scope, debug=args.debug)
    elif args.debug:
        # Update log level if debug flag changed
        setup_logging(debug=args.debug)

    if logger_message:
        logger.warning(logger_message)
    if case_plan is not None:
        logger.info("Collector v2 case plan active:")
        for key, value in case_plan.to_log_dict().items():
            logger.info(f"  {key}: {value}")
        if ops and args.ops is None:
            logger.info(f"  expanded to model-specific ops: {ops}")
    elif args.model_path:
        logger.info(f"Legacy model filter active: collecting only for '{args.model_path}'")

    resume_options = {
        "resume": args.resume,
        "checkpoint_dir": args.checkpoint_dir,
        "retry_failed": args.resume_retry_failed,
    }
    if args.resume_retry_failed and not args.resume:
        parser.error("--resume-retry-failed requires --resume")
    if args.resume:
        logger.info(
            f"Resume enabled: dir={Path(args.checkpoint_dir).expanduser()}"
            + (" (retrying previously failed tasks)" if args.resume_retry_failed else "")
        )

    _require_torch()

    _maybe_apply_deepseek_v3_ep8_defaults(args, ops)

    # Determine number of processes (0 = sequential mode for profiling)
    if args.profile:
        num_processes = 0
        logger.info("Starting collection in sequential mode (profiling enabled)")
    else:
        num_processes = get_device_module().device_count()
        if (
            _is_deepseek_v3_ep8_flow(args, ops)
            and _bool_env_enabled("COLLECTOR_DSV3_EP8_SINGLE_GPU_SIM", True)
            and _bool_env_enabled("COLLECTOR_MOE_DISTRIBUTION_SINGLE_CARD_EP_SIM", False)
            and not _bool_env_enabled("COLLECTOR_DSV3_SINGLE_CARD_CASE_PARALLEL", False)
        ):
            if num_processes != 1:
                logger.info(
                    "DeepSeek-V3 single-card EP simulation: overriding worker "
                    "count %s -> 1 because COLLECTOR_DSV3_SINGLE_CARD_CASE_PARALLEL=false. "
                    "Set it to true to parallelize independent cases across visible GPUs.",
                    num_processes,
                )
            num_processes = 1
        logger.info(f"Starting collection with {num_processes} GPU processes")

    # Set environment variables for worker processes
    if args.measure_power:
        os.environ["COLLECTOR_MEASURE_POWER"] = "true"
        os.environ["COLLECTOR_POWER_MIN_DURATION"] = str(args.power_test_duration_sec)
        logger.info(f"Power monitoring enabled (min duration: {args.power_test_duration_sec}s)")
    else:
        os.environ["COLLECTOR_MEASURE_POWER"] = "false"

    # Suppress torch operator override warnings in spawned workers
    # (env var takes effect at interpreter startup, before any module imports)
    os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:torch.library"

    shuffle = args.shuffle
    limit = args.limit
    if args.smoke:
        os.environ["COLLECTOR_SMOKE"] = "1"
        shuffle = True
        limit = args.limit if args.limit is not None else 4
        logger.info(f"Smoke test mode enabled — sampling {limit} random test cases per op")
    else:
        os.environ.pop("COLLECTOR_SMOKE", None)

    # Warn if profiling without limit (profiling can be very slow)
    if args.profile and limit is None:
        logger.warning(
            "Profiling is enabled but --limit is not set. "
            "Profiling all test cases can be very slow. "
            "Consider using --limit to restrict the number of test cases."
        )

    # Disable core dumps — GPU crashes are expected and handled; core files waste disk.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    # Only set multiprocessing start method if not profiling (profiling uses sequential mode via num_processes=0)
    if not args.profile:
        mp.set_start_method("spawn")

    output_roots = _perf_output_roots()
    existing_perf_outputs = {
        path.resolve(): path.stat().st_mtime_ns
        for output_root in output_roots
        for path in find_perf_csv_outputs(output_root)
    }

    def was_touched_by_run(path: Path) -> bool:
        resolved = path.resolve()
        return resolved not in existing_perf_outputs or path.stat().st_mtime_ns != existing_perf_outputs[resolved]

    # Use profiling context manager
    with ProfilerContext(args.backend, enabled=args.profile):
        if args.backend == "trtllm":
            collect_trtllm(
                num_processes,
                ops,
                limit=limit,
                shuffle=shuffle,
                resume_options=resume_options,
                model_path=case_plan.model_path if case_plan is not None else None,
                case_plan=case_plan,
            )
        elif args.backend == "sglang":
            collect_sglang(
                num_processes,
                ops,
                limit=limit,
                shuffle=shuffle,
                resume_options=resume_options,
                model_path=case_plan.model_path if case_plan is not None else None,
                case_plan=case_plan,
            )
        elif args.backend == "vllm":
            collect_vllm(
                num_processes,
                ops,
                limit=limit,
                shuffle=shuffle,
                resume_options=resume_options,
                model_path=case_plan.model_path if case_plan is not None else None,
                case_plan=case_plan,
            )

    run_dir = Path(os.environ.get("COLLECTOR_LOG_DIR", ".")).resolve()
    with _dsv3_latency_workspace(args, ops, run_dir):
        _run_recorded_materialization_postprocess(args, ops)
        _run_ordinary_moe_materialization_postprocess(args, ops)
        _run_clean_latency_postprocess(args, ops)

    if args.keep_csv:
        logger.info("Keeping collector CSV staging files because --keep-csv was passed")
    else:
        touched_perf_outputs = [
            path
            for output_root in output_roots
            for path in find_perf_csv_outputs(output_root)
            if was_touched_by_run(path)
        ]
        if touched_perf_outputs:
            logger.info(
                "Finalizing collector CSV staging files as parquet:\n  "
                + "\n  ".join(str(path) for path in touched_perf_outputs)
            )
        converted = finalize_perf_files(touched_perf_outputs)
        if converted:
            logger.info(f"Finalized {len(converted)} collector perf files as parquet")


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    main()
