#!/usr/bin/env python3
"""Migrate the frozen SGLang DeepSeek-V3 MoE measurements to the unified schema.

The input is the legacy ``systems/data/<system>/sglang/<version>`` directory
from the Recorded collector.  Truth data is deliberately not read here: this
tool only republishes independently collected MoE measurements and the router
distribution sidecar used for Recorded replay.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


ORDINARY_COLUMNS = [
    "framework", "version", "device", "op_name", "kernel_source", "moe_dtype",
    "num_tokens", "hidden_size", "inter_size", "topk", "num_experts",
    "moe_tp_size", "moe_ep_size", "distribution", "latency",
]
EXPERT_COLUMNS = [
    "framework", "version", "device", "op_name", "kernel_source", "moe_dtype",
    "distribution", "inference_phase", "num_tokens", "hidden_size", "inter_size",
    "topk", "num_experts", "num_slots", "moe_tp_size", "moe_ep_size", "latency",
]


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _write(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns: {missing}")
    result = df[columns].copy()
    for column in ("num_tokens", "hidden_size", "inter_size", "topk", "num_experts",
                   "num_slots", "moe_tp_size", "moe_ep_size"):
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="raise").astype("int64")
    result["latency"] = pd.to_numeric(result["latency"], errors="raise").astype(float)
    if result.empty:
        raise ValueError(f"{path.name}: source has no rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(path, index=False)


def _migrate_platform(source: Path, destination: Path) -> None:
    ordinary = _read(source / "moe_perf.txt")
    _write(ordinary, destination / "moe_perf.parquet", ORDINARY_COLUMNS)

    expert_frames = []
    for filename, phase in (
        ("wideep_context_moe_perf.txt", "context"),
        ("wideep_generation_moe_perf.txt", "generation"),
    ):
        frame = _read(source / filename).copy()
        frame["inference_phase"] = phase
        frame["num_slots"] = frame["num_experts"]
        # Legacy spelling is retained only by the legacy adapter.  New rows
        # must use the kernel-source spelling consumed by the unified SDK.
        frame["kernel_source"] = frame["kernel_source"].replace(
            {
                "deepepmoe": "deepep_moe",
                "aic_recorded_wideep_context_v3": "deepep_moe",
                "aic_recorded_wideep_generation_v1": "deepep_moe",
            }
        )
        frame["op_name"] = "moe_ep"
        expert_frames.append(frame)
    _write(pd.concat(expert_frames, ignore_index=True), destination / "moe_expert_compute_perf.parquet", EXPERT_COLUMNS)

    distribution = _read(source / "moe_token_distribution_perf.txt")
    distribution.to_parquet(destination / "moe_token_distribution_perf.parquet", index=False)

    metadata = {
        "schema_version": 1,
        "provenance": "migrated_recorded_dsv3",
        "runtime": {"framework": "sglang", "version": source.name},
        "tables": {
            "moe_perf": {"status": "complete", "rows": len(ordinary)},
            "moe_expert_compute_perf": {"status": "complete", "rows": len(pd.concat(expert_frames))},
            "moe_token_distribution_perf": {"status": "complete", "rows": len(distribution)},
        },
        # These fields document provenance for humans; schema v1 intentionally
        # permits them and the loader ignores them.
        "source_directory": "legacy systems/data/<system>/sglang/0.5.9-dsv3",
        "source_files": [
            "moe_perf.txt", "wideep_context_moe_perf.txt",
            "wideep_generation_moe_perf.txt", "moe_token_distribution_perf.txt",
        ],
        "truth_used_as_input": False,
        "recorded_distribution_is_independent_sidecar": True,
    }
    (destination / "collection_meta.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--version", default="0.5.9-dsv3")
    args = parser.parse_args()

    for system in ("h20_sxm", "h100_sxm"):
        source = args.source_root / system / "sglang" / args.version
        destination = args.destination_root / system / "moe" / "sglang" / args.version
        if not source.is_dir():
            raise FileNotFoundError(source)
        _migrate_platform(source, destination)
        print(f"migrated {system}: {source} -> {destination}")


if __name__ == "__main__":
    main()
