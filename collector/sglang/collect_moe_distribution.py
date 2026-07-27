# SPDX-License-Identifier: Apache-2.0

"""SGLang MoE token-distribution collector.

Collects real SGLang router/expert load through the built-in expert
distribution recorder and writes AIC perf-style rows plus versioned rank-local
replay bundles.  It does not profile the complete multi-rank routed/compute
server boundary; that validation remains an optional calibration step.
"""

from __future__ import annotations

import csv
import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

from collector.helper import _resolve_local_model_path, resolve_subprocess_visible_device

DEFAULT_TOKENS = [
    1,
    2,
    4,
    8,
    16,
    32,
    48,
    64,
    80,
    96,
    128,
    160,
    192,
    256,
    320,
    384,
    512,
    640,
    768,
    1024,
    1536,
    2048,
    2560,
    3072,
    4096,
    5120,
    6144,
    8192,
    10240,
    12288,
    14336,
    16384,
    18888,
]
DEFAULT_GENERATION_TOKENS = [
    8,
    16,
    32,
    40,
    64,
    96,
    128,
    144,
    160,
    192,
    224,
    256,
    288,
    320,
    384,
    512,
    640,
    768,
    896,
    1024,
    1280,
]
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


def _visible_device_count() -> int:
    return len([item for item in _all_visible_devices().split(",") if item.strip()])


def _default_ep_sizes_for_visible_devices() -> list[int]:
    visible = max(1, _visible_device_count())
    sizes = [ep for ep in DEFAULT_EP_SIZES if ep <= visible]
    return sizes or [1]


def _env_bool_list(name: str, default: list[bool]) -> list[bool]:
    raw = os.environ.get(name)
    if not raw:
        return default
    values = []
    for item in raw.replace(",", " ").split():
        normalized = item.strip().lower()
        if normalized in ("1", "true", "yes", "on"):
            values.append(True)
        elif normalized in ("0", "false", "no", "off"):
            values.append(False)
        else:
            raise ValueError(f"{name} contains invalid boolean {item!r}")
    return values


def get_moe_distribution_test_cases(model_path: str | None = None):
    tokens = _env_int_list(
        "COLLECTOR_MOE_DISTRIBUTION_TOKENS",
        _env_int_list("COLLECTOR_WIDEEP_MOE_PREFILL_TOKENS", DEFAULT_TOKENS),
    )
    generation_tokens = _env_int_list(
        "COLLECTOR_MOE_DISTRIBUTION_GENERATION_TOKENS",
        _env_int_list("COLLECTOR_WIDEEP_MOE_DECODE_TOKENS", DEFAULT_GENERATION_TOKENS),
    )
    ep_sizes = _env_int_list(
        "COLLECTOR_MOE_DISTRIBUTION_EP_SIZES",
        _env_int_list(
            "COLLECTOR_MOE_DISTRIBUTION_EP_SIZE",
            _env_int_list(
                "COLLECTOR_WIDEEP_MOE_EP_SIZES",
                _default_ep_sizes_for_visible_devices(),
            ),
        ),
    )
    visible = _visible_device_count()
    too_large = [ep_size for ep_size in ep_sizes if ep_size > visible]
    if too_large and not _single_card_ep_sim_enabled():
        raise ValueError(
            "moe_token_distribution records real EP ranks and cannot request "
            f"EP sizes {too_large} with only {visible} visible GPUs. "
            "Unset COLLECTOR_*_EP_SIZES to auto-sweep supported EPs, or run "
            "the recorder on a machine with enough GPUs."
        )
    batch_size = int(os.environ.get("COLLECTOR_MOE_DISTRIBUTION_BATCH_SIZE", "1"))
    output_len = int(os.environ.get("COLLECTOR_MOE_DISTRIBUTION_OUTPUT_LEN", "1"))
    eplb_modes = _env_bool_list(
        "COLLECTOR_MOE_DISTRIBUTION_EPLB_MODES",
        [False, True],
    )
    phases = {
        item.strip().lower()
        for item in os.environ.get(
            "COLLECTOR_MOE_DISTRIBUTION_PHASES",
            "context generation",
        ).replace(",", " ").split()
        if item.strip()
    }
    invalid_phases = phases - {"context", "generation"}
    if invalid_phases:
        raise ValueError(
            "COLLECTOR_MOE_DISTRIBUTION_PHASES must contain only "
            f"context/generation, got {sorted(invalid_phases)}"
        )

    def valid_eplb_modes_for_ep(ep_size: int) -> list[bool]:
        # EPLB needs more than one EP rank.  Do not materialize EP1+EPLB as a
        # requested case; otherwise downstream ordinary MoE can discover an
        # eplb-looking replay whose runtime EPLB was silently disabled.
        return [enable_eplb for enable_eplb in eplb_modes if not (int(ep_size) == 1 and enable_eplb)]

    cases = []
    if "context" in phases:
        cases.extend(
            [
                [int(token), batch_size, output_len, int(ep_size), enable_eplb, "context"]
                for token in sorted(set(tokens))
                for ep_size in sorted(set(ep_sizes))
                for enable_eplb in valid_eplb_modes_for_ep(ep_size)
            ]
        )
    if "generation" in phases:
        for global_tokens in sorted(set(generation_tokens)):
            for ep_size in sorted(set(ep_sizes)):
                if _single_card_ep_sim_enabled():
                    cases.extend(
                        [
                            int(global_tokens),
                            1,
                            max(2, output_len),
                            int(ep_size),
                            enable_eplb,
                            "generation",
                        ]
                        for enable_eplb in valid_eplb_modes_for_ep(ep_size)
                    )
                    continue
                if global_tokens % ep_size:
                    continue
                for enable_eplb in valid_eplb_modes_for_ep(ep_size):
                    cases.append(
                        [
                            1,
                            max(1, global_tokens // ep_size),
                            max(2, output_len),
                            int(ep_size),
                            enable_eplb,
                            "generation",
                        ]
                    )
    return cases


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = []
    existing_fields = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fields = list(reader.fieldnames or ())
            existing_rows = list(reader)
    fields = list(existing_fields)
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing_rows)
        writer.writerows(rows)


def _terminate_subprocess_session(pg_leader_pid: int) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pg_leader_pid, signal.SIGTERM)
    time.sleep(1.0)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pg_leader_pid, signal.SIGKILL)


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _diagnostics_path(perf_path: Path) -> Path | None:
    if not _bool_env("COLLECTOR_MOE_DISTRIBUTION_KEEP_DIAGNOSTICS", False):
        return None
    return perf_path.parent / "moe_token_distribution_diagnostics.csv"


def _single_card_ep_sim_enabled() -> bool:
    return _bool_env("COLLECTOR_MOE_DISTRIBUTION_SINGLE_CARD_EP_SIM", False)


def _moe_distribution_perf_rows(
    *,
    replay_rows: list[dict[str, object]],
    replay_file: str,
    model_id: str,
    workload_source: str,
    source_label: str,
    load_format: str,
    requested_ep_size: int,
    runtime_ep_size: int,
    requested_enable_eplb: bool,
    runtime_enable_eplb: bool,
    tp_size: int,
    topk: int,
    num_experts: int,
    device_name: str,
    materialization_method: str | None = None,
) -> list[dict[str, object]]:
    try:
        from importlib.metadata import version as get_version

        sglang_version = get_version("sglang")
    except Exception:
        sglang_version = ""
    rows = []
    for replay_row in replay_rows:
        total = float(replay_row["total_assignments"])
        active = int(replay_row["active_experts"])
        max_assignments = float(replay_row["max_assignments"])
        mean = float(replay_row["mean_assignments"])
        rows.append(
            {
                "framework": "SGLang",
                "version": sglang_version,
                "device": device_name,
                "op_name": "moe_token_distribution",
                "kernel_source": f"expert_distribution_recorder_per_token_{workload_source}",
                "model": model_id,
                "phase": replay_row["phase"],
                "distribution": f"recorded_{workload_source}",
                "recorder_tp_size": tp_size,
                "recorder_ep_size": requested_ep_size,
                "recorder_runtime_ep_size": runtime_ep_size,
                "recorder_enable_eplb": requested_enable_eplb,
                "recorder_runtime_enable_eplb": runtime_enable_eplb,
                "recorder_source": source_label,
                "recorder_materialization_method": materialization_method or "",
                "recorder_load_format": load_format,
                "num_tokens": replay_row["num_tokens"],
                "layer_id": replay_row["layer_id"],
                "topk": topk,
                "num_experts": num_experts,
                "total_assignments": total,
                "active_experts": active,
                "max_assignments": max_assignments,
                "mean_assignments": mean,
                "std_assignments": "",
                "cv": "",
                "max_over_mean": max_assignments / mean if mean else "",
                "nonzero_mean_assignments": "",
                "max_over_nonzero_mean": "",
                "expert_assignments_json": replay_row.get("expert_assignments_json", ""),
                "schema_version": replay_row["schema_version"],
                "sample_id": replay_row["sample_id"],
                "replay_file": replay_file,
                "replay_sha256": replay_row["replay_sha256"],
            }
        )
    return rows


def run_moe_distribution(
    num_tokens,
    batch_size,
    output_len,
    ep_size=None,
    enable_eplb=None,
    phase="both",
    *,
    perf_filename,
    device="cuda:0",
):
    if isinstance(device, str):
        gpu_id = int(device.rsplit(":", 1)[-1]) if ":" in device else 0
    else:
        gpu_id = int(getattr(device, "index", 0) or 0)
    visible_device = resolve_subprocess_visible_device(gpu_id)
    model_id = _selected_model_path()
    model_path = _runtime_model_path(model_id)
    num_layers = _get_requested_num_layers()
    first_k_dense_replace = _first_k_dense_replace(num_layers)
    topk = int(os.environ.get("COLLECTOR_MOE_DISTRIBUTION_TOPK", str(DEFAULT_TOPK)))
    num_experts = int(os.environ.get("COLLECTOR_MOE_DISTRIBUTION_NUM_EXPERTS", str(DEFAULT_NUM_EXPERTS)))
    recorder_mode = os.environ.get("COLLECTOR_MOE_DISTRIBUTION_RECORDER_MODE", "per_token")
    buffer_size = os.environ.get("COLLECTOR_MOE_DISTRIBUTION_BUFFER_SIZE", "-1")
    # AIC operator collection intentionally uses dummy weights by default: the
    # MoE microbench calibrates kernel/runtime shape behavior and must not
    # require loading full model weights.  ``workload_source`` tracks this
    # separately from the measured single-card replay latency.
    load_format = os.environ.get("COLLECTOR_MOE_DISTRIBUTION_LOAD_FORMAT", "dummy")
    mem_fraction_static = float(os.environ.get("COLLECTOR_MOE_DISTRIBUTION_MEM_FRACTION_STATIC", "0.5"))
    workload_source = os.environ.get(
        "COLLECTOR_MOE_DISTRIBUTION_WORKLOAD_SOURCE",
        "dummy" if load_format == "dummy" else "runtime",
    )
    extrapolate_on_failure = _bool_env(
        "COLLECTOR_MOE_DISTRIBUTION_EXTRAPOLATE_ON_FAILURE",
        load_format == "dummy",
    )
    visible_device_count = len([item for item in _all_visible_devices().split(",") if item.strip()])
    requested_ep_size = int(
        ep_size
        if ep_size is not None
        else os.environ.get(
            "COLLECTOR_MOE_DISTRIBUTION_EP_SIZE",
            str(min(8, visible_device_count)),
        )
    )
    fallback_ep1 = os.environ.get("COLLECTOR_MOE_DISTRIBUTION_EP1_FALLBACK_TO_EP2", "false").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    runtime_ep_size = 2 if requested_ep_size == 1 and fallback_ep1 else requested_ep_size
    tp_size = int(
        os.environ.get(
            "COLLECTOR_MOE_DISTRIBUTION_TP_SIZE",
            str(runtime_ep_size),
        )
    )
    if tp_size > visible_device_count and not _single_card_ep_sim_enabled():
        raise ValueError(
            f"MoE distribution TP={tp_size} requires {tp_size} visible GPUs, "
            f"but CUDA_VISIBLE_DEVICES exposes {visible_device_count}"
        )
    if (
        (runtime_ep_size > tp_size or tp_size % runtime_ep_size)
        and not _single_card_ep_sim_enabled()
    ):
        raise ValueError(
            f"MoE distribution requires EP to divide TP, got "
            f"TP={tp_size}, EP={runtime_ep_size}"
        )
    requested_enable_eplb = (
        bool(enable_eplb)
        if enable_eplb is not None
        else os.environ.get("COLLECTOR_MOE_DISTRIBUTION_ENABLE_EPLB", "true").lower()
        in ("1", "true", "yes", "on")
    )
    runtime_enable_eplb = requested_enable_eplb and runtime_ep_size > 1
    phase = str(phase or "both").lower()
    if phase not in ("both", "context", "generation"):
        raise ValueError(f"Unsupported MoE distribution phase: {phase}")

    # The recorder runs from a temporary working directory. Resolve the perf
    # destination in the parent process so a relative caller path cannot place
    # replay bundles inside that temporary directory and silently delete them
    # when the subprocess exits.
    perf_path = Path(perf_filename).resolve()
    replay_dir = perf_path.parent / "moe_token_distribution_replay"
    if _single_card_ep_sim_enabled():
        if phase == "both":
            raise ValueError(
                "single-card materialized MoE distribution requires an explicit "
                "context or generation phase"
            )
        from collector.wideep.sglang.rank_local_moe_replay import (
            materialize_synthetic_replay_bundle,
        )

        table_num_tokens = (
            int(num_tokens) * int(batch_size)
            if phase == "context"
            else int(num_tokens)
        )
        bundle_path, replay_rows = materialize_synthetic_replay_bundle(
            output_dir=replay_dir,
            model=model_id,
            requested_ep_size=requested_ep_size,
            enable_eplb=requested_enable_eplb,
            topk=topk,
            num_logical_experts=num_experts,
            first_moe_layer_id=first_k_dense_replace,
            num_layers=num_layers,
            phase=phase,
            table_num_tokens=table_num_tokens,
            workload_source=workload_source,
            diagnostics_path=_diagnostics_path(perf_path),
        )
        materialization_method = "single_card_deterministic_router_layout"
        replay_file = str(bundle_path.relative_to(replay_dir.parent))
        rows = _moe_distribution_perf_rows(
            replay_rows=replay_rows,
            replay_file=replay_file,
            model_id=model_id,
            workload_source=workload_source,
            source_label=f"single_card_materialized:{materialization_method}",
            load_format=load_format,
            materialization_method=materialization_method,
            requested_ep_size=requested_ep_size,
            runtime_ep_size=1,
            requested_enable_eplb=requested_enable_eplb,
            runtime_enable_eplb=False,
            tp_size=1,
            topk=topk,
            num_experts=num_experts,
            device_name=(
                torch.cuda.get_device_name(gpu_id)
                if torch.cuda.is_available()
                else "single-card-materialized"
            ),
        )
        _write_rows(perf_path, rows)
        return

    repo_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="aic_moe_distribution_") as tmp_dir:
        code = r'''
import csv
import json
import os
from pathlib import Path

import torch

from sglang.srt.environ import envs
from sglang.srt.eplb import expert_distribution as expert_distribution_module
from sglang.srt.layers.moe.token_dispatcher import deepep as deepep_dispatcher_module
from collector.wideep.sglang.rank_local_moe_replay import materialize_replay_bundle

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

# SGLang's detail recorder assumes the internode layout tensor always exists.
# DeepEP returns None for num_tokens_per_rdma_rank on pure intranode EP, so the
# upstream gatherer crashes while serializing an otherwise valid dispatch.
# Patch only that serialization boundary before Engine forks its workers.
def _compat_on_deepep_dispatch_normal(
    self,
    layer_idx,
    local_physical_count_of_layer,
    num_tokens_per_rank,
    num_tokens_per_rdma_rank,
    num_tokens_per_expert,
    recv_topk_ids=None,
):
    self._misc_objects.append(
        dict(
            layer_id=layer_idx,
            local_physical_count_of_layer=list(local_physical_count_of_layer),
            recv_topk_ids=(
                recv_topk_ids.cpu().clone()
                if recv_topk_ids is not None
                else None
            ),
            num_tokens_per_rank=(
                num_tokens_per_rank.cpu().tolist()
                if num_tokens_per_rank is not None
                else []
            ),
            num_tokens_per_rdma_rank=(
                num_tokens_per_rdma_rank.cpu().tolist()
                if num_tokens_per_rdma_rank is not None
                else []
            ),
            num_tokens_per_expert=(
                num_tokens_per_expert.cpu().tolist()
                if num_tokens_per_expert is not None
                else []
            ),
        )
    )


expert_distribution_module._DetailSinglePassGatherer.on_deepep_dispatch_normal = (
    _compat_on_deepep_dispatch_normal
)


def _tensor_layout(value):
    if value is None:
        return None
    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "dtype": str(value.dtype).removeprefix("torch."),
        "is_contiguous": value.is_contiguous(),
    }


_original_deepep_dispatch = deepep_dispatcher_module.DeepEPDispatcher.dispatch


def _dispatch_with_layout_record(self, hidden_states, topk_output):
    output = _original_deepep_dispatch(self, hidden_states, topk_output)
    recorder = expert_distribution_module.get_global_expert_distribution_recorder()
    if not getattr(recorder, "recording", False):
        return output
    current_layer = getattr(recorder, "_current_layer_idx", None)
    layer_id = getattr(current_layer, "value", None)
    gatherers = getattr(recorder, "_single_pass_gatherers", {})
    for gatherer in gatherers.values():
        misc_objects = getattr(gatherer, "_misc_objects", None)
        if misc_objects is None:
            continue
        is_low_latency = isinstance(
            output,
            deepep_dispatcher_module.DeepEPLLDispatchOutput,
        )
        record = {
            "aic_record_type": "dispatch_layout",
            "phase": "generation" if is_low_latency else "context",
            "layer_id": layer_id,
            "hidden_states": _tensor_layout(output.hidden_states),
            "hidden_states_scale": _tensor_layout(output.hidden_states_scale),
            "topk_ids": _tensor_layout(output.topk_ids),
            "topk_weights": _tensor_layout(output.topk_weights),
        }
        if is_low_latency:
            record.update(
                masked_m=_tensor_layout(output.masked_m),
                expected_m=int(output.expected_m),
            )
        else:
            record.update(
                num_recv_tokens_per_expert=list(
                    output.num_recv_tokens_per_expert
                )
            )
        misc_objects.append(record)
    return output


deepep_dispatcher_module.DeepEPDispatcher.dispatch = _dispatch_with_layout_record

from sglang.srt.entrypoints.engine import Engine

engine_kwargs = dict(
    model_path=model_path,
    load_format=os.environ["AIC_MOE_DISTRIBUTION_LOAD_FORMAT"],
    trust_remote_code=True,
    skip_tokenizer_init=True,
    mem_fraction_static=float(os.environ["AIC_MOE_DISTRIBUTION_MEM_FRACTION_STATIC"]),
    tp_size=int(os.environ["AIC_MOE_DISTRIBUTION_TP_SIZE"]),
    ep_size=int(os.environ["AIC_MOE_DISTRIBUTION_RUNTIME_EP_SIZE"]),
    enable_eplb=os.environ["AIC_MOE_DISTRIBUTION_RUNTIME_ENABLE_EPLB"].lower() == "true",
    moe_runner_backend="deep_gemm",
    disable_cuda_graph=True,
    disable_overlap_schedule=True,
    # SGLang's per-token recorder stores routed expert IDs only.  Fused shared
    # experts append an extra ID to topk_ids (DeepSeek-V3 becomes 8 + 1), while
    # the recorder buffer is intentionally sized for the routed top-k.
    disable_shared_experts_fusion=True,
    skip_server_warmup=True,
    expert_distribution_recorder_mode=os.environ["AIC_MOE_DISTRIBUTION_RECORDER_MODE"],
    expert_distribution_recorder_buffer_size=int(os.environ["AIC_MOE_DISTRIBUTION_BUFFER_SIZE"]),
    json_model_override_args=json.dumps(
        {
            "num_hidden_layers": num_layers,
            "first_k_dense_replace": first_k_dense_replace,
        }
    ),
)
if int(os.environ["AIC_MOE_DISTRIBUTION_RUNTIME_EP_SIZE"]) > 1:
    engine_kwargs.update(moe_a2a_backend="deepep", deepep_mode="auto")
engine = Engine(**engine_kwargs)

try:
    input_ids = [[1000 + ((i + j) % 1000) for i in range(num_tokens)] for j in range(batch_size)]
    engine.start_expert_distribution_record()
    engine.generate(
        input_ids=input_ids,
        sampling_params={"temperature": 0, "max_new_tokens": max(1, output_len)},
    )
    engine.stop_expert_distribution_record()
    engine.dump_expert_distribution_record()
finally:
    engine.shutdown()

pt_files = sorted(recorder_dir.glob("expert_distribution_recorder_*.pt"))
if not pt_files:
    raise RuntimeError(f"No expert distribution recorder output found in {recorder_dir}")
diagnostics_csv = os.environ.get("AIC_MOE_DISTRIBUTION_DIAGNOSTICS_CSV") or None
bundle_path, replay_rows = materialize_replay_bundle(
    recorder_dir=recorder_dir,
    output_dir=os.environ["AIC_MOE_DISTRIBUTION_REPLAY_DIR"],
    model=os.environ.get("AIC_MOE_DISTRIBUTION_MODEL_ID", model_path),
    requested_ep_size=int(os.environ["AIC_MOE_DISTRIBUTION_EP_SIZE"]),
    runtime_ep_size=int(os.environ["AIC_MOE_DISTRIBUTION_RUNTIME_EP_SIZE"]),
    enable_eplb=os.environ["AIC_MOE_DISTRIBUTION_REQUESTED_ENABLE_EPLB"].lower() == "true",
    topk=topk,
    num_logical_experts=num_experts,
    first_moe_layer_id=first_k_dense_replace,
    context_table_num_tokens=(
        0
        if os.environ.get(
            "AIC_MOE_DISTRIBUTION_SKIP_CONTEXT_REPLAY",
            "false",
        ).lower() == "true"
        else num_tokens * batch_size
    ),
    generation_table_num_tokens=(
        0
        if os.environ["AIC_MOE_DISTRIBUTION_PHASE"] == "context"
        else batch_size * int(os.environ["AIC_MOE_DISTRIBUTION_EP_SIZE"])
    ),
    workload_source=os.environ["AIC_MOE_DISTRIBUTION_WORKLOAD_SOURCE"],
    diagnostics_path=diagnostics_csv,
)
replay_file = str(
    Path(bundle_path).relative_to(
        Path(os.environ["AIC_MOE_DISTRIBUTION_REPLAY_DIR"]).parent
    )
)

rows = []
for replay_row in replay_rows:
    total = float(replay_row["total_assignments"])
    active = int(replay_row["active_experts"])
    max_assignments = float(replay_row["max_assignments"])
    mean = float(replay_row["mean_assignments"])
    rows.append(
        {
            "framework": "SGLang",
            "version": os.environ.get("AIC_MOE_DISTRIBUTION_SGLANG_VERSION", ""),
            "device": torch.cuda.get_device_name(0),
            "op_name": "moe_token_distribution",
            "kernel_source": (
                "expert_distribution_recorder_per_token_"
                + os.environ["AIC_MOE_DISTRIBUTION_WORKLOAD_SOURCE"]
            ),
            "model": os.environ.get("AIC_MOE_DISTRIBUTION_MODEL_ID", model_path),
            "phase": replay_row["phase"],
            "distribution": (
                "recorded_"
                + os.environ["AIC_MOE_DISTRIBUTION_WORKLOAD_SOURCE"]
            ),
            "recorder_tp_size": int(os.environ["AIC_MOE_DISTRIBUTION_TP_SIZE"]),
            "recorder_ep_size": int(os.environ["AIC_MOE_DISTRIBUTION_EP_SIZE"]),
            "recorder_runtime_ep_size": int(os.environ["AIC_MOE_DISTRIBUTION_RUNTIME_EP_SIZE"]),
            "recorder_enable_eplb": os.environ["AIC_MOE_DISTRIBUTION_REQUESTED_ENABLE_EPLB"].lower() == "true",
            "recorder_runtime_enable_eplb": os.environ["AIC_MOE_DISTRIBUTION_RUNTIME_ENABLE_EPLB"].lower() == "true",
            "recorder_source": os.environ["AIC_MOE_DISTRIBUTION_SOURCE"],
            "recorder_load_format": os.environ["AIC_MOE_DISTRIBUTION_LOAD_FORMAT"],
            "num_tokens": replay_row["num_tokens"],
            "layer_id": replay_row["layer_id"],
            "topk": topk,
            "num_experts": num_experts,
            "total_assignments": total,
            "active_experts": active,
            "max_assignments": max_assignments,
            "mean_assignments": mean,
            "std_assignments": "",
            "cv": "",
            "max_over_mean": max_assignments / mean if mean else "",
            "nonzero_mean_assignments": "",
            "max_over_nonzero_mean": "",
            "expert_assignments_json": replay_row.get("expert_assignments_json", ""),
            "schema_version": replay_row["schema_version"],
            "sample_id": replay_row["sample_id"],
            # Portable relative path rooted at the perf file directory.  The
            # perf table and replay directory can therefore be moved together
            # into systems data without retaining a container-local path.
            "replay_file": replay_file,
            "replay_sha256": replay_row["replay_sha256"],
        }
    )
if not rows:
    raise RuntimeError(
        "No MoE distribution replay rows were materialized for "
        f"phase={os.environ['AIC_MOE_DISTRIBUTION_PHASE']}"
    )

with output_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
'''

        output_csv = Path(tmp_dir) / "moe_token_distribution_rows.csv"
        env = os.environ.copy()
        inherited_pythonpath = env.get("PYTHONPATH", "")
        compat_dir = Path(__file__).resolve().parent / "moe_distribution_compat"
        env["PYTHONPATH"] = os.pathsep.join(
            item
            for item in (
                str(compat_dir),
                str(repo_root),
                str(repo_root / "src"),
                inherited_pythonpath,
            )
            if item
        )
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": _all_visible_devices(),
                "AIC_MOE_DISTRIBUTION_MODEL_PATH": model_path,
                "AIC_MOE_DISTRIBUTION_MODEL_ID": model_id,
                "AIC_MOE_DISTRIBUTION_LOAD_FORMAT": load_format,
                "AIC_MOE_DISTRIBUTION_WORKLOAD_SOURCE": (
                    workload_source
                ),
                "AIC_MOE_DISTRIBUTION_MEM_FRACTION_STATIC": str(mem_fraction_static),
                "AIC_MOE_DISTRIBUTION_TP_SIZE": str(tp_size),
                "AIC_MOE_DISTRIBUTION_EP_SIZE": str(requested_ep_size),
                "AIC_MOE_DISTRIBUTION_RUNTIME_EP_SIZE": str(runtime_ep_size),
                "AIC_MOE_DISTRIBUTION_REQUESTED_ENABLE_EPLB": (
                    "true" if requested_enable_eplb else "false"
                ),
                "AIC_MOE_DISTRIBUTION_RUNTIME_ENABLE_EPLB": (
                    "true" if runtime_enable_eplb else "false"
                ),
                "AIC_MOE_DISTRIBUTION_SOURCE": (
                    "fallback_from_ep2_for_ep1"
                    if runtime_ep_size != requested_ep_size
                    else (
                        "runtime_eplb_disabled_for_ep1"
                        if requested_enable_eplb and not runtime_enable_eplb
                        else "runtime"
                    )
                ),
                "AIC_MOE_DISTRIBUTION_NUM_TOKENS": str(num_tokens),
                "AIC_MOE_DISTRIBUTION_BATCH_SIZE": str(batch_size),
                "AIC_MOE_DISTRIBUTION_OUTPUT_LEN": str(output_len),
                "AIC_MOE_DISTRIBUTION_PHASE": phase,
                "AIC_MOE_DISTRIBUTION_NUM_LAYERS": str(num_layers),
                "AIC_MOE_DISTRIBUTION_FIRST_K_DENSE_REPLACE": str(first_k_dense_replace),
                "AIC_MOE_DISTRIBUTION_TOPK": str(topk),
                "AIC_MOE_DISTRIBUTION_NUM_EXPERTS": str(num_experts),
                "AIC_MOE_DISTRIBUTION_RECORDER_MODE": recorder_mode,
                "AIC_MOE_DISTRIBUTION_BUFFER_SIZE": str(buffer_size),
                "AIC_MOE_DISTRIBUTION_RECORDER_DIR": str(Path(tmp_dir) / "expert_distribution"),
                "AIC_MOE_DISTRIBUTION_OUTPUT_CSV": str(output_csv),
                "AIC_MOE_DISTRIBUTION_REPLAY_DIR": str(replay_dir),
                "AIC_MOE_DISTRIBUTION_DIAGNOSTICS_CSV": (
                    str(_diagnostics_path(perf_path) or "")
                ),
                "AIC_MOE_DISTRIBUTION_SKIP_CONTEXT_REPLAY": (
                    "true"
                    if phase == "generation"
                    else os.environ.get(
                        "COLLECTOR_MOE_DISTRIBUTION_SKIP_CONTEXT_REPLAY",
                        "false",
                    )
                ),
            }
        )
        try:
            from importlib.metadata import version as get_version

            env["AIC_MOE_DISTRIBUTION_SGLANG_VERSION"] = get_version("sglang")
        except Exception:
            env["AIC_MOE_DISTRIBUTION_SGLANG_VERSION"] = ""

        timeout_seconds = int(
            os.environ.get("COLLECTOR_MOE_DISTRIBUTION_TIMEOUT_SECONDS", "600")
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", code],
            cwd=tmp_dir,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            # Engine workers are grandchildren of this subprocess. Killing only
            # the Python parent leaves scheduler processes and CUDA contexts
            # behind, so terminate the complete session created above.
            _terminate_subprocess_session(proc.pid)
            try:
                stdout, stderr = proc.communicate(timeout=15)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                stdout, stderr = proc.communicate()
            raise RuntimeError(
                "MoE distribution subprocess timed out "
                f"after {timeout_seconds}s\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            ) from error
        if proc.returncode != 0:
            # A failed recorder subprocess can leave SGLang scheduler
            # grandchildren alive in the separate session. Best-effort
            # terminate the whole session before any fallback or retry logic.
            _terminate_subprocess_session(proc.pid)
            error_message = (
                "MoE distribution subprocess failed "
                f"(exit={proc.returncode})\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            )
            if extrapolate_on_failure:
                try:
                    from collector.wideep.sglang.rank_local_moe_replay import (
                        materialize_extrapolated_replay_bundle,
                    )

                    manifest_path = replay_dir / "manifest.csv"
                    if not manifest_path.is_file():
                        raise FileNotFoundError(manifest_path)
                    bundle_path, replay_rows, donor_tokens = (
                        materialize_extrapolated_replay_bundle(
                            replay_dir=replay_dir,
                            model=model_id,
                            requested_ep_size=requested_ep_size,
                            runtime_ep_size=runtime_ep_size,
                            enable_eplb=requested_enable_eplb,
                            topk=topk,
                            num_logical_experts=num_experts,
                            context_table_num_tokens=num_tokens * batch_size,
                            workload_source=workload_source,
                        )
                    )
                    replay_file = str(bundle_path.relative_to(replay_dir.parent))
                    rows = _moe_distribution_perf_rows(
                        replay_rows=replay_rows,
                        replay_file=replay_file,
                        model_id=model_id,
                        workload_source=workload_source,
                        source_label=f"extrapolated_from_ctx{donor_tokens}",
                        load_format=load_format,
                        requested_ep_size=requested_ep_size,
                        runtime_ep_size=runtime_ep_size,
                        requested_enable_eplb=requested_enable_eplb,
                        runtime_enable_eplb=runtime_enable_eplb,
                        tp_size=tp_size,
                        topk=topk,
                        num_experts=num_experts,
                        device_name=torch.cuda.get_device_name(gpu_id),
                    )
                    _write_rows(perf_path, rows)
                    return
                except Exception as fallback_error:
                    raise RuntimeError(
                        error_message
                        + "\nReplay extrapolation fallback also failed:\n"
                        + repr(fallback_error)
                    ) from fallback_error
            raise RuntimeError(error_message)

        with output_csv.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        _write_rows(perf_path, rows)
        _terminate_subprocess_session(proc.pid)
