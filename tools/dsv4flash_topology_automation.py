#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def run_cmd(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    return p.returncode


def find_pareto_by_mode(root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in root.rglob("pareto.csv"):
        s = str(p).lower()
        if "disagg" in s:
            out.setdefault("disagg", p)
        elif "agg" in s:
            out.setdefault("agg", p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="One-command DS-V4 Flash topology compare automation")
    ap.add_argument("--config", required=True, help="JSON config path")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))

    out_dir = Path(cfg["out_dir"])
    scaleup_dir = out_dir / "scaleup"
    scaleout_dir = out_dir / "scaleout"
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    common = [
        "aiconfigurator", "cli", "default",
        "--systems-paths", cfg["systems_path"],
        "--model", cfg["model"],
        "--total-gpus", str(cfg["total_gpus"]),
        "--backend", cfg["backend"],
        "--backend-version", cfg["backend_version"],
        "--isl", str(cfg["isl"]),
        "--osl", str(cfg["osl"]),
        "--gemm-quant-mode", cfg["gemm_quant_mode"],
        "--database-mode", cfg["database_mode"],
        "--ttft", str(cfg["ttft"]),
        "--tpot", str(cfg["tpot"]),
    ]

    rc1 = run_cmd(common + ["--system", cfg["scaleup_system"], "--save-dir", str(scaleup_dir)], out_dir / "output_scaleup_32.log")
    rc2 = run_cmd(common + ["--system", cfg["scaleout_system"], "--save-dir", str(scaleout_dir)], out_dir / "output_scaleout_2x16.log")

    if rc1 != 0 or rc2 != 0:
        raise SystemExit(f"run failed: scaleup_rc={rc1}, scaleout_rc={rc2}")

    up = find_pareto_by_mode(scaleup_dir)
    out = find_pareto_by_mode(scaleout_dir)

    modes = ["agg", "disagg"]
    for m in modes:
        up_p = up.get(m)
        out_p = out.get(m)
        if not up_p or not out_p:
            print(f"skip mode={m}: missing pareto. scaleup={up_p} scaleout={out_p}")
            continue
        canon_up = scaleup_dir / f"pareto_{m}.csv"
        canon_out = scaleout_dir / f"pareto_{m}.csv"
        shutil.copy2(up_p, canon_up)
        shutil.copy2(out_p, canon_out)

        plot_cmd = [
            "uv", "run", "--frozen", "python", "tools/plot_pareto_compare.py",
            "--scaleup-csv", str(canon_up),
            "--scaleout-csv", str(canon_out),
            "--x-col", cfg.get("x_col", "tokens/s/gpu"),
            "--y-col", cfg.get("y_col", "tokens/s/user"),
            "--title", f"DS-V4 Flash {m} Scale-up vs Scale-out",
            "--output", str(plot_dir / f"pareto_compare_{m}.png"),
        ]
        prc = subprocess.run(plot_cmd, text=True)
        if prc.returncode != 0:
            print(f"plot failed for mode={m}")

    extract_cmd = [
        "python", "tools/extract_final_metrics.py",
        "--scaleup-log", str(out_dir / "output_scaleup_32.log"),
        "--scaleout-log", str(out_dir / "output_scaleout_2x16.log"),
        "--output-csv", str(out_dir / "compare_single_point.csv"),
    ]
    subprocess.run(extract_cmd, check=False)

    print("done")


if __name__ == "__main__":
    main()
