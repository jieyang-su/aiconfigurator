#!/usr/bin/env python3
"""Verify fixed inputs for recorded MoE generalization work."""

from __future__ import annotations

import argparse
from pathlib import Path
import csv
import sys

try:
    import yaml
except ImportError:  # pragma: no cover - fallback for minimal envs
    yaml = None


ROOT = Path(__file__).resolve().parents[4]


def _load_manifest(path: Path) -> dict:
    text = path.read_text()
    if yaml is not None:
        return yaml.safe_load(text)

    raise RuntimeError("PyYAML is required to parse the manifest")


def _count_lines(path: Path) -> int | None:
    if not path.exists() or not path.is_file():
        return None
    with path.open("rb") as f:
        return sum(1 for _ in f)


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.manifest
    manifest = _load_manifest(manifest_path)
    rows: list[dict[str, object]] = []

    def add(label: str, path_value: str, kind: str) -> None:
        path = _resolve(path_value)
        exists = path.exists()
        rows.append(
            {
                "label": label,
                "kind": kind,
                "path": str(path),
                "exists": exists,
                "line_count": _count_lines(path) if exists and path.is_file() else "",
            }
        )

    add("truth.point_error_table", manifest["truth"]["point_error_table"], "truth")
    add("truth.previous_materialized_candidate_dir", manifest["truth"]["previous_materialized_candidate_dir"], "dir")

    for platform, spec in manifest["aic_collector_inputs"].items():
        if "collector_dir" in spec:
            add(f"{platform}.collector_dir", spec["collector_dir"], "dir")
        add(f"{platform}.raw_source_dir", spec["raw_source_dir"], "dir")
        for table_name, table_path in spec["raw_tables"].items():
            add(f"{platform}.raw_tables.{table_name}", table_path, "raw_table")
        for name, path_value in spec.get("copied_final_dirs", {}).items():
            add(f"{platform}.copied_final_dirs.{name}", path_value, "dir")

    for name, path_value in manifest["offline_artifacts"].items():
        add(f"offline_artifacts.{name}", path_value, "dir")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "kind", "path", "exists", "line_count"])
        writer.writeheader()
        writer.writerows(rows)

    missing = [row for row in rows if not row["exists"]]
    print(f"Wrote {args.output}")
    print(f"checked={len(rows)} missing={len(missing)}")
    for row in missing:
        print(f"MISSING {row['label']}: {row['path']}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
