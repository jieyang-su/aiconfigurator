#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run a pure nsys WideEP generation smoke/truth client.

This helper starts SGLang directly without torch-profiler, waits for health,
performs warmup generations, then wraps the measured generation in an NVTX
range so nsys can locate it.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[5]
MOE_CALIBRATION_DIR = REPO_ROOT / "tools" / "moe_calibration"
if str(MOE_CALIBRATION_DIR) not in sys.path:
    sys.path.insert(0, str(MOE_CALIBRATION_DIR))

from dsv3_moe_dense_refresh import generate, wait_ready


def _start_server_cuda_profiler(port: int, decode_steps: int) -> None:
    """Start SGLang's CUDA-profiler control in the kernel-owning workers."""
    response = requests.post(
        f"http://127.0.0.1:{port}/start_profile",
        json={
            "activities": ["CUDA_PROFILER"],
            "num_steps": decode_steps,
            "profile_by_stage": False,
            "profile_prefix": "aic_nsys",
        },
        timeout=120,
    )
    response.raise_for_status()


def _server_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--tokenizer-path",
        args.tokenizer_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--tp-size",
        str(args.ep),
        "--ep-size",
        str(args.ep),
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "low_latency",
        "--disable-shared-experts-fusion",
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--max-total-tokens",
        str(args.max_total_tokens),
        "--max-prefill-tokens",
        str(args.max_prefill_tokens),
        "--chunked-prefill-size",
        str(args.chunked_prefill_size),
        "--watchdog-timeout",
        str(args.watchdog_timeout),
        "--json-model-override-args",
        f'{{"num_hidden_layers":{args.num_hidden_layers},"first_k_dense_replace":{args.first_k_dense_replace}}}',
    ]
    if args.eplb == "eplb_on":
        cmd.append("--enable-eplb")
    cuda_graph_max_bs = os.environ.get("CUDA_GRAPH_MAX_BS")
    if cuda_graph_max_bs:
        cmd.extend(["--cuda-graph-max-bs", cuda_graph_max_bs])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--eplb", choices=["eplb_off", "eplb_on"], required=True)
    parser.add_argument("--token", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--warmup-repetitions", type=int, default=3)
    parser.add_argument("--measured-repetitions", type=int, default=1)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--server-ready-timeout", type=int, default=1800)
    parser.add_argument("--watchdog-timeout", type=int, default=3600)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-hidden-layers", type=int, default=6)
    parser.add_argument("--first-k-dense-replace", type=int, default=3)
    parser.add_argument("--mem-fraction-static", type=float, default=0.5)
    parser.add_argument("--max-total-tokens", type=int, default=20000)
    parser.add_argument("--max-prefill-tokens", type=int, default=16384)
    parser.add_argument("--chunked-prefill-size", type=int, default=16384)
    parser.add_argument("--seed-base", type=int, default=1)
    parser.add_argument("--range-name", required=True)
    parser.add_argument("--server-log", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.server_log.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", "")
    extra_paths = [env.get("AIC_SRC", "/cold/tair-kvcache/aiconfigurator") + "/src"]
    extra_paths.append(env.get("AIC_SRC", "/cold/tair-kvcache/aiconfigurator"))
    extra_paths.append(env.get("SGLANG_SRC", "/sgl-workspace/sglang") + "/python")
    env["PYTHONPATH"] = ":".join(extra_paths + ([env["PYTHONPATH"]] if env["PYTHONPATH"] else []))
    # This enables only SGLang's semantic record_function annotations.  The
    # nsys path never starts torch.profiler; CUDA-profiler control is requested
    # separately through /start_profile below.
    env["SGLANG_AIC_MOE_PROFILE"] = "1"
    env["SGLANG_AIC_NSYS_SEMANTIC_MARKERS"] = env.get(
        "SGLANG_AIC_NSYS_SEMANTIC_MARKERS", "1"
    )
    env["SGLANG_AIC_NSYS_CUDA_EVENTS"] = env.get(
        "SGLANG_AIC_NSYS_CUDA_EVENTS", "1"
    )
    env["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] = env.get(
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK", "1024"
    )
    (args.output_dir / "server_env_snapshot.txt").write_text(
        "\n".join(
            f"{key}={env.get(key, )}"
            for key in [
                "PYTHONPATH",
                "SGLANG_AIC_MOE_PROFILE",
                "SGLANG_AIC_NSYS_SEMANTIC_MARKERS",
                "SGLANG_AIC_NSYS_CUDA_EVENTS",
                "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK",
                "CUDA_GRAPH_MAX_BS",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    if args.eplb == "eplb_on":
        env["ENABLE_EPLB"] = "1"

    server = subprocess.Popen(
        _server_cmd(args),
        env=env,
        stdout=args.server_log.open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )

    try:
        wait_ready(args.port, args.server_ready_timeout)
        for rep in range(args.warmup_repetitions):
            generate(
                args.port,
                batch_size=args.batch_size,
                input_len=1,
                seed=100000 + args.seed_base * 100 + rep,
                max_new_tokens=args.samples,
                prompt_source="sharegpt",
                dataset_path=args.dataset_path,
                tokenizer_path=args.tokenizer_path,
            )
        # One request has one prefill forward plus ``samples`` decode forwards.
        # Capture all of them; the later WideEP-generation parser must drop
        # prefill and apply the decode-only representative-chunk rule.
        _start_server_cuda_profiler(args.port, args.samples + 1)
        generate(
            args.port,
            batch_size=args.batch_size,
            input_len=1,
            seed=200000 + args.seed_base,
            max_new_tokens=args.samples,
            prompt_source="sharegpt",
            dataset_path=args.dataset_path,
            tokenizer_path=args.tokenizer_path,
        )
    finally:
        try:
            server.send_signal(signal.SIGTERM)
        except Exception:
            pass
        try:
            server.wait(timeout=30)
        except Exception:
            try:
                server.kill()
            except Exception:
                pass


if __name__ == "__main__":
    main()
