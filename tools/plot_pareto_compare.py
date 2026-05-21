#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

X_CANDIDATES = ["tokens/s/gpu", "per_gpu_throughput", "throughput_per_gpu", "throughput_gpu", "x", "throughput"]
Y_CANDIDATES = ["tokens/s/user", "per_user_throughput", "throughput_per_user", "throughput_user", "y", "latency", "ttft"]

AXIS_LABELS = {
    "tokens/s/user": "Interactivity (tokens/s/user)",
    "tokens/s/gpu": "Token Throughput per GPU (tokens/s/gpu)",
}


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


def axis_label(requested: str, resolved: str) -> str:
    return AXIS_LABELS.get(resolved, AXIS_LABELS.get(requested, requested))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scaleup-csv")
    ap.add_argument("--scaleout-csv")
    ap.add_argument("--scaleup-label", default="scaleup")
    ap.add_argument("--scaleout-label", default="scaleout")
    ap.add_argument("--x-col", default="tokens/s/gpu")
    ap.add_argument("--y-col", default="tokens/s/user")
    ap.add_argument("--title", default="Scale-up vs Scale-out Pareto")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    if not a.scaleup_csv and not a.scaleout_csv:
        raise SystemExit("At least one of --scaleup-csv or --scaleout-csv must be provided")

    import matplotlib.pyplot as plt

    sx = sy = ox = oy = []
    sx_col = sy_col = ox_col = oy_col = ""
    if a.scaleup_csv:
        sx, sy, sx_col, sy_col = read_xy(Path(a.scaleup_csv), a.x_col, a.y_col)
    if a.scaleout_csv:
        ox, oy, ox_col, oy_col = read_xy(Path(a.scaleout_csv), a.x_col, a.y_col)

    plt.figure(figsize=(8, 5))
    if a.scaleup_csv:
        plt.plot(sx, sy, "o-", label=a.scaleup_label)
    if a.scaleout_csv:
        plt.plot(ox, oy, "o-", label=a.scaleout_label)

    resolved_x = sx_col or ox_col or a.x_col
    resolved_y = sy_col or oy_col or a.y_col
    plt.xlabel(axis_label(a.x_col, resolved_x))
    plt.ylabel(axis_label(a.y_col, resolved_y))
    plt.title(a.title)
    if a.scaleup_csv or a.scaleout_csv:
        plt.legend()
    plt.tight_layout()
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(a.output, dpi=150)
    print(a.output)


if __name__ == "__main__":
    main()
