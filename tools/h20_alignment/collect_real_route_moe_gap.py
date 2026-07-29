#!/usr/bin/env python3
"""Measure an H20 rank-local MoE replay from real-weight SGLang routing.

The input recorder directory must contain one ExpertDistributionRecorder
artifact per EP rank.  The script first materializes an exact replay bundle,
then benchmarks the maximum rank-local ordinary-MoE latency on one H20.  All
outputs remain in the caller-provided staging directory; this script does not
install or overwrite a performance database table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from collector.sglang.collect_moe import run_moe_torch
from collector.wideep.sglang.rank_local_moe_replay import (
    MANIFEST_FILENAME,
    materialize_replay_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorder-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-tokens", type=int, default=8192)
    parser.add_argument("--generation-tokens", type=int, default=8)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_fresh_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    recorder_files = sorted(
        args.recorder_dir.glob("expert_distribution_recorder_*.pt")
    )
    if len(recorder_files) != 8:
        raise ValueError(
            f"expected exactly 8 rank recorder files, found {len(recorder_files)}"
        )
    if args.context_tokens <= 0 or args.generation_tokens < 0:
        raise ValueError("context tokens must be positive and generation tokens nonnegative")

    require_fresh_output_dir(args.output_dir)
    replay_dir = args.output_dir / "moe_token_distribution_replay"
    diagnostics_path = args.output_dir / "replay_count_diagnostics.csv"
    bundle_path, replay_rows = materialize_replay_bundle(
        recorder_dir=args.recorder_dir,
        output_dir=replay_dir,
        model="DeepSeek-V3-layers39-realweights",
        requested_ep_size=8,
        runtime_ep_size=8,
        enable_eplb=False,
        topk=8,
        num_logical_experts=256,
        first_moe_layer_id=3,
        context_table_num_tokens=args.context_tokens,
        generation_table_num_tokens=args.generation_tokens,
        workload_source="realweights_tp8ep8_a1",
        diagnostics_path=diagnostics_path,
    )

    os.environ["COLLECTOR_MOE_RANK_LOCAL_REPLAY_DIR"] = str(replay_dir)
    os.environ["COLLECTOR_MOE_RANK_LOCAL_REPLAY_PHASES"] = "context"
    perf_path = args.output_dir / "moe_perf.txt"
    run_moe_torch(
        "fp8_block",
        args.context_tokens,
        7168,
        2048,
        8,
        256,
        1,
        8,
        "deepseek-ai/DeepSeek-V3",
        distributed="recorded_realweights_tp8ep8_a1_rank_local_no_eplb",
        power_law_alpha=0,
        perf_filename=str(perf_path),
        device=args.device,
    )

    artifacts = [bundle_path, replay_dir / MANIFEST_FILENAME, diagnostics_path, perf_path]
    summary = {
        "schema": "aiconfigurator.h20_real_route_moe_gap.v1",
        "scope": "staging_only_not_installed",
        "device": args.device,
        "context_tokens": args.context_tokens,
        "generation_tokens": args.generation_tokens,
        "recorder_files": [
            {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in recorder_files
        ],
        "replay_row_count": len(replay_rows),
        "artifacts": [
            {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in artifacts
        ],
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
