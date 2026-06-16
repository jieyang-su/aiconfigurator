# SPDX-License-Identifier: Apache-2.0

"""SGLang MoE token-distribution collector.

Collects real SGLang router/expert load through the built-in expert
distribution recorder and writes AIC perf-style rows.  The collector uses dummy
weights and a small layer count so it can be run as part of normal collector v2
flows without relying on old experiment directories.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from collector.helper import _resolve_local_model_path, resolve_subprocess_visible_device

DEFAULT_TOKENS = [128, 512, 2048, 4096]
DEFAULT_EP_SIZES = [1, 2, 4, 8]
DEFAULT_TOPK = 8
DEFAULT_NUM_EXPERTS = 256


def _env_int_list(name: str, default: list[int]) -> list[int]:
    raw = os.environ.get(name)
    if not raw:
        return default
    return [int(item) for item in raw.replace(",", " ").split() if item.strip()]


def _selected_model_path() -> str:
    if os.environ.get("COLLECTOR_MOE_DISTRIBUTION_MODEL_PATH"):
        return os.environ["COLLECTOR_MOE_DISTRIBUTION_MODEL_PATH"]
    if os.environ.get("MOE_MODEL_PATH"):
        return os.environ["MOE_MODEL_PATH"]
    if os.environ.get("COLLECTOR_MODEL_PATH"):
        return os.environ["COLLECTOR_MODEL_PATH"]
    for candidate in (
        Path("/model/DeepSeek-V3.1"),
        Path("/model/DeepSeek-V3"),
        Path("/model"),
    ):
        if (candidate / "config.json").exists():
            return str(candidate)
    return "deepseek-ai/DeepSeek-V3.1"


def _runtime_model_path(model_id: str) -> str:
    if os.environ.get("COLLECTOR_MOE_DISTRIBUTION_MODEL_PATH"):
        return _resolve_local_model_path(os.environ["COLLECTOR_MOE_DISTRIBUTION_MODEL_PATH"])
    if os.environ.get("MOE_MODEL_PATH"):
        return _resolve_local_model_path(os.environ["MOE_MODEL_PATH"])
    for candidate in (
        Path("/model/DeepSeek-V3.1"),
        Path("/model/DeepSeek-V3"),
        Path("/model"),
    ):
        if (candidate / "config.json").exists():
            return str(candidate)
    return _resolve_local_model_path(model_id)


def _get_requested_num_layers() -> int:
    raw = os.environ.get("COLLECTOR_MOE_DISTRIBUTION_NUM_LAYERS")
    if raw is None:
        return 4
    value = int(raw)
    if value < 1:
        raise ValueError(f"SGLANG_TEST_NUM_LAYERS must be >= 1, got {value}")
    return value


def _first_k_dense_replace(num_layers: int) -> int:
    raw = os.environ.get("COLLECTOR_MOE_DISTRIBUTION_FIRST_K_DENSE_REPLACE")
    if raw is not None:
        return int(raw)
    return min(3, max(num_layers - 1, 0))


def _all_visible_devices() -> str:
    raw = os.environ.get("COLLECTOR_MOE_DISTRIBUTION_VISIBLE_DEVICES")
    if raw:
        return raw
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited:
        return inherited
    count = torch.cuda.device_count()
    if count > 0:
        return ",".join(str(i) for i in range(count))
    return "0"


def get_moe_distribution_test_cases(model_path: str | None = None):
    tokens = _env_int_list(
        "COLLECTOR_MOE_DISTRIBUTION_TOKENS",
        _env_int_list("COLLECTOR_WIDEEP_MOE_PREFILL_TOKENS", DEFAULT_TOKENS),
    )
    ep_sizes = _env_int_list(
        "COLLECTOR_MOE_DISTRIBUTION_EP_SIZES",
        _env_int_list(
            "COLLECTOR_MOE_DISTRIBUTION_EP_SIZE",
            _env_int_list("COLLECTOR_WIDEEP_MOE_EP_SIZES", DEFAULT_EP_SIZES),
        ),
    )
    batch_size = int(os.environ.get("COLLECTOR_MOE_DISTRIBUTION_BATCH_SIZE", "1"))
    output_len = int(os.environ.get("COLLECTOR_MOE_DISTRIBUTION_OUTPUT_LEN", "0"))
    return [
        [int(token), batch_size, output_len, int(ep_size)]
        for token in sorted(set(tokens))
        for ep_size in sorted(set(ep_sizes))
    ]


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def run_moe_distribution(num_tokens, batch_size, output_len, ep_size=None, *, perf_filename, device="cuda:0"):
    gpu_id = int(getattr(device, "index", 0) or 0)
    visible_device = resolve_subprocess_visible_device(gpu_id)
    model_id = _selected_model_path()
    model_path = _runtime_model_path(model_id)
    num_layers = _get_requested_num_layers()
    first_k_dense_replace = _first_k_dense_replace(num_layers)
    topk = int(os.environ.get("COLLECTOR_MOE_DISTRIBUTION_TOPK", str(DEFAULT_TOPK)))
    num_experts = int(os.environ.get("COLLECTOR_MOE_DISTRIBUTION_NUM_EXPERTS", str(DEFAULT_NUM_EXPERTS)))
    recorder_mode = os.environ.get("COLLECTOR_MOE_DISTRIBUTION_RECORDER_MODE", "stat")
    buffer_size = os.environ.get("COLLECTOR_MOE_DISTRIBUTION_BUFFER_SIZE", "-1")
    load_format = os.environ.get("COLLECTOR_MOE_DISTRIBUTION_LOAD_FORMAT", "auto")
    mem_fraction_static = float(os.environ.get("COLLECTOR_MOE_DISTRIBUTION_MEM_FRACTION_STATIC", "0.5"))
    visible_device_count = len([item for item in _all_visible_devices().split(",") if item.strip()])
    tp_size = int(os.environ.get("COLLECTOR_MOE_DISTRIBUTION_TP_SIZE", str(min(8, visible_device_count))))
    requested_ep_size = int(
        ep_size if ep_size is not None else os.environ.get("COLLECTOR_MOE_DISTRIBUTION_EP_SIZE", str(tp_size))
    )
    fallback_ep1 = os.environ.get("COLLECTOR_MOE_DISTRIBUTION_EP1_FALLBACK_TO_EP2", "true").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    runtime_ep_size = 2 if requested_ep_size == 1 and fallback_ep1 else requested_ep_size
    requested_enable_eplb = os.environ.get("COLLECTOR_MOE_DISTRIBUTION_ENABLE_EPLB", "true").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    enable_eplb = requested_enable_eplb and runtime_ep_size > 1

    with tempfile.TemporaryDirectory(prefix="aic_moe_distribution_") as tmp_dir:
        code = r'''
import csv
import json
import os
from pathlib import Path

import torch

from sglang.srt.environ import envs

model_path = os.environ["AIC_MOE_DISTRIBUTION_MODEL_PATH"]
num_tokens = int(os.environ["AIC_MOE_DISTRIBUTION_NUM_TOKENS"])
batch_size = int(os.environ["AIC_MOE_DISTRIBUTION_BATCH_SIZE"])
output_len = int(os.environ["AIC_MOE_DISTRIBUTION_OUTPUT_LEN"])
num_layers = int(os.environ["AIC_MOE_DISTRIBUTION_NUM_LAYERS"])
first_k_dense_replace = int(os.environ["AIC_MOE_DISTRIBUTION_FIRST_K_DENSE_REPLACE"])
topk = int(os.environ["AIC_MOE_DISTRIBUTION_TOPK"])
num_experts = int(os.environ["AIC_MOE_DISTRIBUTION_NUM_EXPERTS"])
recorder_dir = Path(os.environ["AIC_MOE_DISTRIBUTION_RECORDER_DIR"])
output_csv = Path(os.environ["AIC_MOE_DISTRIBUTION_OUTPUT_CSV"])

envs.SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR.set(str(recorder_dir))

from sglang.srt.entrypoints.engine import Engine

engine = Engine(
    model_path=model_path,
    load_format=os.environ["AIC_MOE_DISTRIBUTION_LOAD_FORMAT"],
    trust_remote_code=True,
    skip_tokenizer_init=True,
    mem_fraction_static=float(os.environ["AIC_MOE_DISTRIBUTION_MEM_FRACTION_STATIC"]),
    tp_size=int(os.environ["AIC_MOE_DISTRIBUTION_TP_SIZE"]),
    ep_size=int(os.environ["AIC_MOE_DISTRIBUTION_RUNTIME_EP_SIZE"]),
    enable_eplb=os.environ["AIC_MOE_DISTRIBUTION_ENABLE_EPLB"].lower() == "true",
    disable_cuda_graph=True,
    disable_overlap_schedule=True,
    expert_distribution_recorder_mode=os.environ["AIC_MOE_DISTRIBUTION_RECORDER_MODE"],
    expert_distribution_recorder_buffer_size=int(os.environ["AIC_MOE_DISTRIBUTION_BUFFER_SIZE"]),
    json_model_override_args=json.dumps(
        {
            "num_hidden_layers": num_layers,
            "first_k_dense_replace": first_k_dense_replace,
        }
    ),
)

try:
    input_ids = [[1000 + ((i + j) % 1000) for i in range(num_tokens)] for j in range(batch_size)]
    engine.start_expert_distribution_record()
    engine.generate(
        input_ids=input_ids,
        sampling_params={"temperature": 0, "max_new_tokens": output_len},
    )
    engine.stop_expert_distribution_record()
    engine.dump_expert_distribution_record()
finally:
    engine.shutdown()

pt_files = sorted(recorder_dir.glob("expert_distribution_recorder_*.pt"))
if not pt_files:
    raise RuntimeError(f"No expert distribution recorder output found in {recorder_dir}")
data = torch.load(pt_files[-1], map_location="cpu", weights_only=True)
logical_count = data["logical_count"].to(torch.float32)
if logical_count.ndim == 3:
    logical_count = logical_count.sum(dim=0)
elif logical_count.ndim != 2:
    raise RuntimeError(f"Unexpected logical_count shape: {tuple(logical_count.shape)}")

rows = []
for logical_layer_index in range(logical_count.shape[0]):
    counts = logical_count[logical_layer_index]
    total = float(counts.sum().item())
    active = int((counts > 0).sum().item())
    max_assignments = float(counts.max().item()) if counts.numel() else 0.0
    mean = float(counts.mean().item()) if counts.numel() else 0.0
    std = float(counts.std(unbiased=False).item()) if counts.numel() else 0.0
    nonzero = counts[counts > 0]
    nonzero_mean = float(nonzero.mean().item()) if active else 0.0
    rows.append(
        {
            "framework": "SGLang",
            "version": os.environ.get("AIC_MOE_DISTRIBUTION_SGLANG_VERSION", ""),
            "device": torch.cuda.get_device_name(0),
            "op_name": "moe_token_distribution",
            "kernel_source": "expert_distribution_recorder",
            "model": os.environ.get("AIC_MOE_DISTRIBUTION_MODEL_ID", model_path),
            "phase": "context",
            "distribution": "recorded",
            "recorder_tp_size": int(os.environ["AIC_MOE_DISTRIBUTION_TP_SIZE"]),
            "recorder_ep_size": int(os.environ["AIC_MOE_DISTRIBUTION_EP_SIZE"]),
            "recorder_runtime_ep_size": int(os.environ["AIC_MOE_DISTRIBUTION_RUNTIME_EP_SIZE"]),
            "recorder_enable_eplb": os.environ["AIC_MOE_DISTRIBUTION_ENABLE_EPLB"].lower() == "true",
            "recorder_source": os.environ["AIC_MOE_DISTRIBUTION_SOURCE"],
            "num_tokens": num_tokens * batch_size,
            "layer_id": first_k_dense_replace + logical_layer_index,
            "topk": topk,
            "num_experts": num_experts,
            "total_assignments": total,
            "active_experts": active,
            "max_assignments": max_assignments,
            "mean_assignments": mean,
            "std_assignments": std,
            "cv": std / mean if mean else "",
            "max_over_mean": max_assignments / mean if mean else "",
            "nonzero_mean_assignments": nonzero_mean,
            "max_over_nonzero_mean": max_assignments / nonzero_mean if nonzero_mean else "",
            "expert_assignments_json": json.dumps([float(value) for value in counts.tolist()]),
        }
    )

with output_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
'''

        output_csv = Path(tmp_dir) / "moe_token_distribution_rows.csv"
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": _all_visible_devices(),
            "AIC_MOE_DISTRIBUTION_MODEL_PATH": model_path,
                "AIC_MOE_DISTRIBUTION_MODEL_ID": model_id,
                "AIC_MOE_DISTRIBUTION_LOAD_FORMAT": load_format,
                "AIC_MOE_DISTRIBUTION_MEM_FRACTION_STATIC": str(mem_fraction_static),
                "AIC_MOE_DISTRIBUTION_TP_SIZE": str(tp_size),
                "AIC_MOE_DISTRIBUTION_EP_SIZE": str(requested_ep_size),
                "AIC_MOE_DISTRIBUTION_RUNTIME_EP_SIZE": str(runtime_ep_size),
                "AIC_MOE_DISTRIBUTION_ENABLE_EPLB": "true" if enable_eplb else "false",
                "AIC_MOE_DISTRIBUTION_SOURCE": (
                    "fallback_from_ep2_for_ep1" if runtime_ep_size != requested_ep_size else "runtime"
                ),
                "AIC_MOE_DISTRIBUTION_NUM_TOKENS": str(num_tokens),
                "AIC_MOE_DISTRIBUTION_BATCH_SIZE": str(batch_size),
                "AIC_MOE_DISTRIBUTION_OUTPUT_LEN": str(output_len),
                "AIC_MOE_DISTRIBUTION_NUM_LAYERS": str(num_layers),
                "AIC_MOE_DISTRIBUTION_FIRST_K_DENSE_REPLACE": str(first_k_dense_replace),
                "AIC_MOE_DISTRIBUTION_TOPK": str(topk),
                "AIC_MOE_DISTRIBUTION_NUM_EXPERTS": str(num_experts),
                "AIC_MOE_DISTRIBUTION_RECORDER_MODE": recorder_mode,
                "AIC_MOE_DISTRIBUTION_BUFFER_SIZE": str(buffer_size),
                "AIC_MOE_DISTRIBUTION_RECORDER_DIR": str(Path(tmp_dir) / "expert_distribution"),
                "AIC_MOE_DISTRIBUTION_OUTPUT_CSV": str(output_csv),
            }
        )
        try:
            from importlib.metadata import version as get_version

            env["AIC_MOE_DISTRIBUTION_SGLANG_VERSION"] = get_version("sglang")
        except Exception:
            env["AIC_MOE_DISTRIBUTION_SGLANG_VERSION"] = ""

        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=tmp_dir,
            env=env,
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "MoE distribution subprocess failed "
                f"(exit={proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )

        with output_csv.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        _write_rows(Path(perf_filename), rows)
