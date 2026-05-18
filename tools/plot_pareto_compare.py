#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

X_CANDIDATES = ["tokens/s/gpu", "per_gpu_throughput", "throughput_per_gpu", "throughput_gpu", "x", "throughput"]
Y_CANDIDATES = ["tokens/s/user", "per_user_throughput", "throughput_per_user", "throughput_user", "y", "latency", "ttft"]


def pick_col(fieldnames: list[str], requested: str, candidates: list[str]) -> str:
    if requested in fieldnames:
        return requested
    lower_map = {c.lower(): c for c in fieldnames}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    raise ValueError(f"Cannot find a compatible column for '{requested}'. Available: {fieldnames}")


def read_xy(path: Path, requested_x: str, requested_y: str) -> tuple[list[float], list[float], str, str]:
    xs: list[float] = []
    ys: list[float] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        x_col = pick_col(reader.fieldnames, requested_x, X_CANDIDATES)
        y_col = pick_col(reader.fieldnames, requested_y, Y_CANDIDATES)
        for r in reader:
            try:
                xs.append(float(r[x_col]))
                ys.append(float(r[y_col]))
            except Exception:
                pass
    return xs, ys, x_col, y_col


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scaleup-csv", required=True)
    ap.add_argument("--scaleout-csv", required=True)
    ap.add_argument("--x-col", default="tokens/s/gpu")
    ap.add_argument("--y-col", default="tokens/s/user")
    ap.add_argument("--title", default="Scale-up vs Scale-out Pareto")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    import matplotlib.pyplot as plt

    sx, sy, sx_col, sy_col = read_xy(Path(a.scaleup_csv), a.x_col, a.y_col)
    ox, oy, ox_col, oy_col = read_xy(Path(a.scaleout_csv), a.x_col, a.y_col)

    plt.figure(figsize=(8, 5))
    plt.plot(sx, sy, "o-", label=f"scaleup_32 ({sx_col}/{sy_col})")
    plt.plot(ox, oy, "o-", label=f"scaleout_2x16 ({ox_col}/{oy_col})")
    plt.xlabel(a.x_col)
    plt.ylabel(a.y_col)
    plt.title(a.title)
    plt.legend()
    plt.tight_layout()
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(a.output, dpi=150)
    print(a.output)


if __name__ == "__main__":
    main()
