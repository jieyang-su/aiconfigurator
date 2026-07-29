#!/usr/bin/env python3
"""Collect synchronized FlashInfer fused all-reduce + RMSNorm latency.

Run this script with ``torchrun``.  It exercises the same SGLang wrapper used
by ``RMSNorm.forward_with_allreduce_fusion``: BF16, residual enabled,
``use_oneshot=None`` (FlashInfer auto selection), and the production defaults
for completion and FP32 accumulation.  Rank 0 writes the maximum rank-local
CUDA-event latency for each sample so the table represents collective service
time rather than the fastest participant.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import importlib.metadata
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist

from sglang.srt.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    graph_capture,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.flashinfer_comm_fusion import (
    cleanup_flashinfer_workspace,
    flashinfer_allreduce_residual_rmsnorm,
)


DEFAULT_TOKENS = (1, 2, 4, 8, 16, 24, 32, 40, 64, 128, 256, 512)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=DEFAULT_TOKENS)
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--ops-per-graph", type=int, default=10)
    return parser.parse_args()


def rank_max(value: float, device: torch.device) -> float:
    sample = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(sample, op=dist.ReduceOp.MAX)
    return float(sample.item())


def production_op(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    result = flashinfer_allreduce_residual_rmsnorm(
        input_tensor=x,
        residual=residual,
        weight=weight,
    )
    if result[0] is None or result[1] is None:
        raise RuntimeError("SGLang did not activate FlashInfer fused all-reduce")
    return result


def time_eager(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    *,
    warmups: int,
    samples: int,
) -> list[float]:
    for _ in range(warmups):
        production_op(x, residual, weight)
    torch.cuda.synchronize()

    values = []
    for _ in range(samples):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        production_op(x, residual, weight)
        end.record()
        end.synchronize()
        values.append(rank_max(start.elapsed_time(end), x.device))
    return values


def time_graph(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    *,
    warmups: int,
    samples: int,
    ops_per_graph: int,
) -> list[float]:
    production_op(x, residual, weight)
    torch.cuda.synchronize()
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with graph_capture() as capture:
        with torch.cuda.graph(graph, stream=capture.stream):
            for _ in range(ops_per_graph):
                production_op(x, residual, weight)

    for _ in range(warmups):
        graph.replay()
    torch.cuda.synchronize()

    values = []
    for _ in range(samples):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        local_ms = start.elapsed_time(end) / ops_per_graph
        values.append(rank_max(local_ms, x.device))

    del graph
    gc.collect()
    torch.cuda.empty_cache()
    return values


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "stdev_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main() -> int:
    args = parse_args()
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("launch with torchrun")
    if any(token <= 0 for token in args.tokens):
        raise ValueError("all token counts must be positive")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size <= 1:
        raise ValueError("FlashInfer fused all-reduce requires world_size > 1")

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.set_default_device(device)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    try:
        for token_num in sorted(set(args.tokens)):
            x = torch.randn(
                token_num,
                args.hidden_size,
                dtype=torch.bfloat16,
                device=device,
            )
            residual = torch.randn_like(x)
            weight = torch.ones(
                args.hidden_size,
                dtype=torch.bfloat16,
                device=device,
            )

            eager = time_eager(
                x,
                residual,
                weight,
                warmups=args.warmups,
                samples=args.samples,
            )
            graph = time_graph(
                x,
                residual,
                weight,
                warmups=args.warmups,
                samples=args.samples,
                ops_per_graph=args.ops_per_graph,
            )
            if rank == 0:
                print(
                    f"tokens={token_num} eager={statistics.fmean(eager):.6f}ms "
                    f"graph={statistics.fmean(graph):.6f}ms",
                    flush=True,
                )
                for execution_mode, values in (("eager", eager), ("graph", graph)):
                    summary = summarize(values)
                    rows.append(
                        {
                            "framework": "SGLang",
                            "version": importlib.metadata.version("sglang"),
                            "device": torch.cuda.get_device_name(device),
                            "op_name": "flashinfer_fused_allreduce_residual_rmsnorm",
                            "kernel_source": f"flashinfer_auto_{execution_mode}",
                            "dtype": "bfloat16",
                            "num_gpus": world_size,
                            "token_num": token_num,
                            "hidden_size": args.hidden_size,
                            "message_size": token_num * args.hidden_size,
                            "pattern": "auto",
                            "execution_mode": execution_mode,
                            "latency": summary["mean_ms"],
                        }
                    )
                    diagnostics.append(
                        {
                            "token_num": token_num,
                            "execution_mode": execution_mode,
                            "rank_aggregation": "maximum latency across ranks per sample",
                            **summary,
                            "samples_ms": values,
                        }
                    )
    finally:
        with contextlib.suppress(Exception):
            cleanup_flashinfer_workspace()
        with contextlib.suppress(Exception):
            dist.barrier()
        cleanup_dist_env_and_memory(shutdown_ray=False)

    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0])
        with args.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        sidecar = args.output.with_suffix(".diagnostics.json")
        sidecar.write_text(
            json.dumps(
                {
                    "schema": "aiconfigurator.flashinfer_fused_allreduce_collection.v1",
                    "world_size": world_size,
                    "hidden_size": args.hidden_size,
                    "warmups": args.warmups,
                    "samples": args.samples,
                    "ops_per_graph": args.ops_per_graph,
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "rows": diagnostics,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
