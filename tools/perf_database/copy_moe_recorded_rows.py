#!/usr/bin/env python3
"""Copy the ordinary DeepSeek-V3 recorded MoE rows into a PCIe table.

The copied latency strings are preserved byte-for-byte at the CSV-field level.
Only the source ``phase`` representation is normalized into the destination
distribution names expected by the ordinary MoE loader.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


SOURCE_DISTRIBUTION = "recorded_no_eplb"
PHASE_TO_DESTINATION_DISTRIBUTION = {
    "context": "recorded_context_no_eplb",
    "generation": "recorded_generation_no_eplb",
}
FILTERS = {
    "op_name": "moe",
    "kernel_source": "sglang_fused_moe_triton",
    "moe_dtype": "fp8_block",
    "hidden_size": "7168",
    "inter_size": "2048",
    "topk": "8",
    "num_experts": "256",
    "moe_tp_size": "1",
    "moe_ep_size": "4",
    "distribution": SOURCE_DISTRIBUTION,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV header missing: {path}")
        return list(reader.fieldnames), list(reader)


def selected_source_rows(source_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected = [
        row
        for row in source_rows
        if all(row.get(field) == value for field, value in FILTERS.items())
        and row.get("phase") in PHASE_TO_DESTINATION_DISTRIBUTION
    ]
    counts = Counter(row["phase"] for row in selected)
    if set(counts) != set(PHASE_TO_DESTINATION_DISTRIBUTION):
        raise ValueError(
            "source must contain both context and generation recorded/no-EPLB rows; "
            f"found {dict(counts)}"
        )
    expected_tokens = {
        1,
        2,
        4,
        8,
        32,
        40,
        64,
        128,
        288,
        512,
        640,
        896,
        1024,
        1536,
        2048,
        2560,
        4096,
        5120,
        8192,
        10240,
        12288,
        14336,
        16384,
    }
    tokens_by_phase = {
        phase: {int(row["num_tokens"]) for row in selected if row["phase"] == phase}
        for phase in PHASE_TO_DESTINATION_DISTRIBUTION
    }
    if counts != {"context": len(expected_tokens), "generation": len(expected_tokens)}:
        raise ValueError(
            "unexpected selected source grid; expected 23 context and generation rows "
            f"with tokens {sorted(expected_tokens)}, found {dict(counts)}"
        )
    for phase, tokens in tokens_by_phase.items():
        if tokens != expected_tokens:
            raise ValueError(
                f"{phase} recorded source token grid is incomplete: "
                f"expected {sorted(expected_tokens)}, found {sorted(tokens)}"
            )
    return selected


def destination_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row["distribution"],
        row["moe_dtype"],
        row["hidden_size"],
        row["inter_size"],
        row["topk"],
        row["num_experts"],
        row["moe_tp_size"],
        row["moe_ep_size"],
        row["num_tokens"],
    )


def normalize_for_destination(
    row: dict[str, str], destination_fields: list[str]
) -> dict[str, str]:
    normalized = {field: row.get(field, "") for field in destination_fields}
    normalized["distribution"] = PHASE_TO_DESTINATION_DISTRIBUTION[row["phase"]]
    return normalized


def write_output(
    base_target: Path,
    output: Path,
    destination_fields: list[str],
    appended_rows: list[dict[str, str]],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with base_target.open("rb") as source, output.open("wb") as destination:
        shutil.copyfileobj(source, destination)
        destination.seek(0, 2)
        if destination.tell() and not base_target.read_bytes().endswith(b"\n"):
            destination.write(b"\n")
    with output.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=destination_fields,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writerows(appended_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--base-target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-ref", default="origin/dev@1913348")
    parser.add_argument(
        "--expected-source-sha256",
        default=None,
        help="Optionally pin the source bytes; the observed SHA is always recorded.",
    )
    args = parser.parse_args()

    source_hash = sha256(args.source)
    if args.expected_source_sha256 and source_hash != args.expected_source_sha256:
        raise ValueError(
            f"source SHA256 mismatch: {source_hash} != {args.expected_source_sha256}"
        )

    source_fields, source_rows = read_rows(args.source)
    destination_fields, destination_rows = read_rows(args.base_target)
    if "phase" not in source_fields:
        raise ValueError("source table has no phase column")
    required_destination_fields = set(source_fields) - {"phase"}
    missing_fields = required_destination_fields - set(destination_fields)
    if missing_fields:
        raise ValueError(f"destination table is missing fields: {sorted(missing_fields)}")

    selected = selected_source_rows(source_rows)
    appended = [
        normalize_for_destination(row, destination_fields) for row in selected
    ]
    existing_keys = {destination_key(row) for row in destination_rows}
    appended_keys = [destination_key(row) for row in appended]
    duplicate_keys = [key for key in appended_keys if key in existing_keys]
    if duplicate_keys:
        raise ValueError(
            "destination already contains selected recorded rows; first duplicate key: "
            f"{duplicate_keys[0]}"
        )
    if len(set(appended_keys)) != len(appended_keys):
        raise ValueError("selected source rows contain duplicate destination keys")

    write_output(args.base_target, args.output, destination_fields, appended)
    output_fields, output_rows = read_rows(args.output)
    if output_fields != destination_fields:
        raise AssertionError("output header changed")
    if len(output_rows) != len(destination_rows) + len(appended):
        raise AssertionError("output row count does not match base plus copied rows")

    phase_tokens: dict[str, list[int]] = defaultdict(list)
    for row in selected:
        phase_tokens[row["phase"]].append(int(row["num_tokens"]))
    manifest = {
        "schema": "aiconfigurator.moe_recorded_copy_provenance.v1",
        "source_ref": args.source_ref,
        "source_path": str(args.source.resolve()),
        "source_sha256": source_hash,
        "source_lfs_oid": f"sha256:{source_hash}",
        "base_target_path": str(args.base_target.resolve()),
        "base_target_sha256": sha256(args.base_target),
        "output_path": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "filters": FILTERS,
        "phase_distribution_mapping": PHASE_TO_DESTINATION_DISTRIBUTION,
        "copied_row_count": len(appended),
        "copied_rows_by_phase": dict(Counter(row["phase"] for row in selected)),
        "tokens_by_phase": {
            phase: sorted(tokens) for phase, tokens in phase_tokens.items()
        },
        "latency_origin": "SXM-derived recorded latency",
        "latency_values_transformed": False,
        "pcie_sxm_scaling_applied": False,
        "excluded": [
            "WideEP/DeepEP",
            "EPLB",
            "H20",
            "other model shapes",
            "balanced and power-law distributions",
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
