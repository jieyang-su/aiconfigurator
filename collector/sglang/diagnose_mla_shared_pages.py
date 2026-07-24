#!/usr/bin/env python3
"""Compare FA3 MLA decode latency under three physical KV-page topologies.

This is a focused diagnostic companion to ``collect_mla_module.py``.  It uses
the same ModelRunner, attention module, dummy latent input, and CUDA Graph
execution path.  The only changed input between cases is the physical page
mapping stored in ``ReqToTokenPool.req_to_token``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from collect_mla_module import cleanup_distributed, load_model_runner


TOPOLOGIES = ("independent", "pair_shared_shuffled", "all_shared")


def build_mixed_group_ids(num_groups: int, prompts_per_group: int, seed: int) -> list[int]:
    """Match gsp_workload.py's mixed two-round dispatch policy."""
    rng = random.Random(seed)
    dispatch: list[int] = []
    previous_group: int | None = None
    for _occurrence_id in range(prompts_per_group):
        round_groups = list(range(num_groups))
        rng.shuffle(round_groups)
        if previous_group is not None and round_groups[0] == previous_group:
            swap_index = next(i for i, group_id in enumerate(round_groups[1:], 1) if group_id != previous_group)
            round_groups[0], round_groups[swap_index] = round_groups[swap_index], round_groups[0]
        dispatch.extend(round_groups)
        previous_group = round_groups[-1]
    return dispatch


def apply_topology(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    base_page_table: torch.Tensor,
    topology: str,
    shared_prefix_len: int,
    pair_group_ids: list[int],
) -> dict:
    """Restore independent mappings, then alias the requested shared prefix."""
    active_len = base_page_table.shape[1]
    req_to_token[req_pool_indices, :active_len] = base_page_table

    if topology == "pair_shared_shuffled":
        first_row_by_group: dict[int, int] = {}
        for row_index, group_id in enumerate(pair_group_ids):
            source_index = first_row_by_group.setdefault(group_id, row_index)
            if source_index != row_index:
                target_pool_index = int(req_pool_indices[row_index].item())
                source_pool_index = int(req_pool_indices[source_index].item())
                req_to_token[target_pool_index, :shared_prefix_len].copy_(
                    req_to_token[source_pool_index, :shared_prefix_len]
                )
    elif topology == "all_shared":
        source_pool_index = int(req_pool_indices[0].item())
        for row_index in range(1, len(req_pool_indices)):
            target_pool_index = int(req_pool_indices[row_index].item())
            req_to_token[target_pool_index, :shared_prefix_len].copy_(
                req_to_token[source_pool_index, :shared_prefix_len]
            )
    elif topology != "independent":
        raise ValueError(f"Unknown topology: {topology}")

    torch.cuda.synchronize()
    rows = req_to_token[req_pool_indices, :active_len]
    logical_page_count = int(rows.numel())
    unique_page_count = int(torch.unique(rows).numel())
    return {
        "logical_page_count": logical_page_count,
        "unique_page_count": unique_page_count,
        "unique_over_logical": unique_page_count / logical_page_count,
    }


def cuda_time_graph(graph: torch.cuda.CUDAGraph, replays_per_sample: int, sample_count: int) -> list[float]:
    samples_ms: list[float] = []
    for _ in range(sample_count):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays_per_sample):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples_ms.append(start.elapsed_time(end) / replays_per_sample)
    return samples_ms


def profile_graph(graph: torch.cuda.CUDAGraph, trace_path: Path, replay_count: int) -> list[dict]:
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=False) as prof:
        for _ in range(replay_count):
            graph.replay()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(trace_path))

    rows = []
    for event in prof.key_averages():
        device_us = float(
            getattr(event, "self_cuda_time_total", 0.0)
            or getattr(event, "self_device_time_total", 0.0)
            or 0.0
        )
        if device_us <= 0:
            continue
        rows.append(
            {
                "name": event.key,
                "self_device_total_ms": device_us / 1000.0,
                "self_device_per_replay_ms": device_us / 1000.0 / replay_count,
                "count": int(event.count),
            }
        )
    rows.sort(key=lambda item: item["self_device_total_ms"], reverse=True)
    return rows[:20]


def summarize(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {
        "samples_ms": samples,
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=32768)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--shared-prefix-len", type=int, default=29491)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--samples-per-cycle", type=int, default=5)
    parser.add_argument("--replays-per-sample", type=int, default=100)
    parser.add_argument("--profile-replays", type=int, default=5)
    args = parser.parse_args()

    if args.batch_size != 32:
        raise ValueError("This diagnostic currently defines a 16x2 mixed topology and requires batch_size=32")
    if not 0 < args.shared_prefix_len < args.seq_len:
        raise ValueError("shared_prefix_len must be in (0, seq_len)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(11)
    np.random.seed(11)
    random.seed(11)
    torch.cuda.set_device(0)

    pair_group_ids = build_mixed_group_ids(num_groups=16, prompts_per_group=2, seed=11)
    expected_group_ids = [
        10, 5, 0, 1, 9, 4, 6, 2, 3, 15, 11, 7, 12, 8, 13, 14,
        3, 15, 14, 13, 4, 12, 5, 7, 6, 0, 10, 11, 8, 1, 2, 9,
    ]
    if pair_group_ids != expected_group_ids:
        raise AssertionError(f"seed-11 mixed dispatch drifted: {pair_group_ids}")

    started_at = time.time()
    model_runner = None
    results: dict = {
        "schema": "mla_shared_page_diagnostic.v1",
        "started_at_unix_s": started_at,
        "target_gpu_host_index": int(os.environ.get("TARGET_GPU_HOST_INDEX", "-1")),
        "target_gpu_uuid": os.environ.get("TARGET_GPU_UUID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "num_heads": args.num_heads,
        "shared_prefix_len": args.shared_prefix_len,
        "shared_prefix_ratio": args.shared_prefix_len / args.seq_len,
        "pair_group_ids": pair_group_ids,
        "collector_precision": {
            "compute_dtype": "bfloat16",
            "kv_cache_dtype": "bfloat16",
            "gemm_type": "bfloat16",
            "load_format": os.environ.get("SGLANG_LOAD_FORMAT", "dummy"),
            "test_num_layers": int(os.environ.get("SGLANG_TEST_NUM_LAYERS", "2")),
        },
        "measurement": {
            "cycles": args.cycles,
            "samples_per_cycle": args.samples_per_cycle,
            "replays_per_sample": args.replays_per_sample,
            "profile_replays": args.profile_replays,
        },
        "topologies": {name: {"samples_ms": [], "profile_top_kernels": []} for name in TOPOLOGIES},
    }

    try:
        model_runner = load_model_runner(
            model_path="deepseek-ai/DeepSeek-V3",
            head_num=args.num_heads,
            kv_cache_dtype="bfloat16",
            attention_backend="fa3",
            device="cuda:0",
            gemm_type="bfloat16",
        )
        results["device_name"] = torch.cuda.get_device_name(0)
        results["sglang_attention_backend"] = model_runner.server_args.attention_backend

        from sglang.srt.layers.communicator import AttentionInputs, get_attn_tp_context
        from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from sglang.srt.mem_cache.chunk_cache import ChunkCache
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        from sglang.srt.sampling.sampling_params import SamplingParams
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
        from sglang.srt.utils import BumpAllocator

        attention_module = model_runner.model.model.layers[0].self_attn
        q_lora_rank = getattr(attention_module, "q_lora_rank", 1536) or 1536
        kv_lora_rank = getattr(attention_module, "kv_lora_rank", 512)
        qk_rope_head_dim = getattr(attention_module, "qk_rope_head_dim", 64)
        qkv_latent_dim = q_lora_rank + kv_lora_rank + qk_rope_head_dim

        def dummy_qkv_latent_func(hidden_states, _forward_batch):
            return torch.randn(
                hidden_states.shape[0],
                qkv_latent_dim,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        reqs = []
        for request_index in range(args.batch_size):
            req = Req(
                rid=str(request_index),
                origin_input_text="",
                origin_input_ids=list(torch.randint(0, 10000, (args.seq_len,)).tolist()),
                sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
            )
            req.prefix_indices = torch.empty((0,), dtype=torch.int64)
            req.fill_ids = req.origin_input_ids
            req.extend_input_len = len(req.fill_ids)
            req.logprob_start_len = 0
            req.cached_tokens = 0
            req.already_computed = 0
            reqs.append(req)

        cache_params = CacheInitParams(
            disable=True,
            req_to_token_pool=model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
            page_size=model_runner.token_to_kv_pool_allocator.page_size,
        )
        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
            tree_cache=ChunkCache(cache_params),
            model_config=model_runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.prepare_for_extend()
        batch.output_ids = torch.randint(0, 10000, (args.batch_size,), dtype=torch.int64, device="cuda")
        batch.prepare_for_decode()

        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
        active_seq_len = int(forward_batch.seq_lens_cpu.max().item())
        base_page_table = model_runner.req_to_token_pool.req_to_token[
            batch.req_pool_indices, :active_seq_len
        ].clone()
        results["active_seq_len"] = active_seq_len

        decode_hidden = torch.randn(
            args.batch_size,
            model_runner.model.config.hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
        )
        decode_positions = torch.full((args.batch_size,), args.seq_len, device="cuda")
        zero_allocator = BumpAllocator(buffer_size=2048, dtype=torch.float32, device="cuda")
        attn_inputs = AttentionInputs(decode_hidden, forward_batch, dummy_qkv_latent_func)
        get_attn_tp_context().set_attn_inputs(attn_inputs)

        def kernel_func():
            attention_module(
                positions=decode_positions,
                hidden_states=decode_hidden,
                forward_batch=forward_batch,
                zero_allocator=zero_allocator,
            )

        cycle_orders = [
            ("independent", "pair_shared_shuffled", "all_shared"),
            ("all_shared", "pair_shared_shuffled", "independent"),
            ("pair_shared_shuffled", "independent", "all_shared"),
        ]
        for cycle_index in range(args.cycles):
            for topology in cycle_orders[cycle_index % len(cycle_orders)]:
                topology_meta = apply_topology(
                    model_runner.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    base_page_table,
                    topology,
                    args.shared_prefix_len,
                    pair_group_ids,
                )
                model_runner.attn_backend.init_forward_metadata(forward_batch)

                for _ in range(8):
                    kernel_func()
                torch.cuda.synchronize()

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    kernel_func()
                torch.cuda.synchronize()
                for _ in range(5):
                    graph.replay()
                torch.cuda.synchronize()

                samples = cuda_time_graph(graph, args.replays_per_sample, args.samples_per_cycle)
                item = results["topologies"][topology]
                item["samples_ms"].extend(samples)
                item["mapping"] = topology_meta
                print(
                    json.dumps(
                        {
                            "cycle": cycle_index,
                            "topology": topology,
                            "samples_ms": samples,
                            **topology_meta,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

                if cycle_index == 0 and args.profile_replays > 0:
                    trace_path = args.output_dir / f"{topology}.trace.json"
                    item["profile_top_kernels"] = profile_graph(graph, trace_path, args.profile_replays)
                    item["profile_trace"] = str(trace_path)

                del graph
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        for topology in TOPOLOGIES:
            item = results["topologies"][topology]
            item.update(summarize(item.pop("samples_ms")))

        independent_mean = results["topologies"]["independent"]["mean_ms"]
        for topology in TOPOLOGIES[1:]:
            mean_ms = results["topologies"][topology]["mean_ms"]
            results["topologies"][topology]["speedup_vs_independent"] = independent_mean / mean_ms
            results["topologies"][topology]["saved_ms_vs_independent"] = independent_mean - mean_ms

        pair_mean = results["topologies"]["pair_shared_shuffled"]["mean_ms"]
        all_mean = results["topologies"]["all_shared"]["mean_ms"]
        results["incremental_all_vs_pair"] = {
            "saved_ms": pair_mean - all_mean,
            "speedup": pair_mean / all_mean,
        }
        results["completed_at_unix_s"] = time.time()
        results["elapsed_s"] = results["completed_at_unix_s"] - started_at

        output_path = args.output_dir / "results.json"
        output_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"results_path": str(output_path)}, sort_keys=True), flush=True)
    finally:
        if model_runner is not None:
            model_runner.req_to_token_pool.clear()
            model_runner.token_to_kv_pool_allocator.clear()
        cleanup_distributed()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
