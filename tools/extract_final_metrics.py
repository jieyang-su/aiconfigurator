#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

PATS = {
    "best_throughput": r"Best Throughput.*?:\s*([0-9.]+)",
    "per_gpu_throughput": r"Per-GPU Throughput.*?:\s*([0-9.]+)",
    "per_user_throughput": r"Per-User Throughput.*?:\s*([0-9.]+)",
    "ttft_ms": r"TTFT.*?:\s*([0-9.]+)",
    "tpot_ms": r"TPOT.*?:\s*([0-9.]+)",
    "request_latency_ms": r"Request Latency.*?:\s*([0-9.]+)",
    "best_experiment": r"Best Experiment Chosen.*?:\s*([A-Za-z0-9_/-]+)",
}

# Broad patterns to tolerate different summary formats
PARALLEL_PATS = {
    "tp": [r"\btp\s*[:=]\s*([0-9]+)", r"\bparallel\s*[:=]\s*([0-9]+)"],
    "pp": [r"\bpp\s*[:=]\s*([0-9]+)"],
    "replicas": [r"\breplicas\s*[:=]\s*([0-9]+)"],
    "bs": [r"\bbs\s*[:=]\s*([0-9]+)", r"\bbatch(?:_size)?\s*[:=]\s*([0-9]+)"],
}


def first_match(text: str, patterns: list[str]) -> str:
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1)
    return ""


def parse(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    out: dict[str, str] = {}
    for k, p in PATS.items():
        m = re.search(p, text, re.IGNORECASE)
        out[k] = m.group(1) if m else ""
    for k, patterns in PARALLEL_PATS.items():
        out[k] = first_match(text, patterns)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scaleup-log", required=True)
    ap.add_argument("--scaleout-log", required=True)
    ap.add_argument("--output-csv", required=True)
    a = ap.parse_args()

    su = parse(Path(a.scaleup_log))
    so = parse(Path(a.scaleout_log))

    ordered_metrics = [*PATS.keys(), "tp", "pp", "replicas", "bs"]
    rows = []
    for k in ordered_metrics:
        row = {"metric": k, "scaleup_32": su.get(k, ""), "scaleout_2x16": so.get(k, ""), "delta": ""}
        try:
            row["delta"] = str(float(row["scaleup_32"]) - float(row["scaleout_2x16"]))
        except Exception:
            row["delta"] = ""
        rows.append(row)

    out = Path(a.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "scaleup_32", "scaleout_2x16", "delta"])
        w.writeheader()
        w.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
