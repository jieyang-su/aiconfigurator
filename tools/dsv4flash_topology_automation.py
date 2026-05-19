#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"


def _display_label(system_name: str, fallback: str) -> str:
    return system_name or fallback


def _requested_modes(cfg: dict) -> list[str]:
    fixed = cfg.get("fixed_parallel") or {}
    search = cfg.get("search_parallel") or {}
    modes = [m for m in ["agg", "disagg"] if fixed.get(m) or search.get(m)]
    if not modes:
        return ["agg", "disagg"]
    return modes


def _has_custom_parallel(cfg: dict) -> bool:
    return bool(cfg.get("fixed_parallel") or cfg.get("search_parallel"))


def run_cmd(cmd: list[str], log_path: Path, cwd: Path | None = None, env: dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, cwd=cwd, env=env)
    return p.returncode


def find_pareto_by_mode(root: Path) -> dict[str, Path]:
    return find_result_csv_by_mode(root, "pareto.csv")


def find_best_config_by_mode(root: Path) -> dict[str, Path]:
    return find_result_csv_by_mode(root, "best_config_topn.csv")


def find_result_csv_by_mode(root: Path, filename: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in root.rglob(filename):
        s = str(p).lower()
        if "disagg" in s:
            current = out.get("disagg")
            if current is None or p.stat().st_mtime > current.stat().st_mtime:
                out["disagg"] = p
        elif "agg" in s:
            current = out.get("agg")
            if current is None or p.stat().st_mtime > current.stat().st_mtime:
                out["agg"] = p
    return out


def _copy_quant_overrides(cfg: dict) -> dict:
    out: dict[str, object] = {}
    for key in [
        "gemm_quant_mode",
        "moe_quant_mode",
        "kvcache_quant_mode",
        "fmha_quant_mode",
        "comm_quant_mode",
    ]:
        if key in cfg:
            out[key] = cfg[key]
    return out


def _int_list(value: object) -> list[int]:
    if isinstance(value, list):
        return [int(v) for v in value]
    return [int(value)]


def _fixed_worker_config(cfg: dict, fixed: dict) -> dict:
    worker_cfg = _copy_quant_overrides(cfg)
    worker_cfg.update(
        {
            "num_gpu_per_worker": _int_list(fixed["num_gpu_per_worker"]),
            "tp_list": _int_list(fixed.get("tp", 1)),
            "pp_list": _int_list(fixed.get("pp", 1)),
            "dp_list": _int_list(fixed.get("dp", 1)),
            "moe_tp_list": _int_list(fixed.get("moe_tp", 1)),
            "moe_ep_list": _int_list(fixed.get("moe_ep", 1)),
        }
    )
    if "enable_wideep" in fixed:
        worker_cfg["enable_wideep"] = bool(fixed["enable_wideep"])
    if "enable_eplb" in fixed:
        worker_cfg["enable_eplb"] = bool(fixed["enable_eplb"])
    if "moe_backend" in fixed:
        worker_cfg["moe_backend"] = fixed["moe_backend"]
    if "attention_backend" in fixed:
        worker_cfg["attention_backend"] = fixed["attention_backend"]
    return worker_cfg


def _base_exp_config(cfg: dict, system_name: str) -> dict:
    exp_cfg = {
        "mode": "patch",
        "model_path": cfg["model"],
        "total_gpus": cfg["total_gpus"],
        "system_name": system_name,
        "backend_name": cfg["backend"],
        "backend_version": cfg["backend_version"],
        "database_mode": cfg["database_mode"],
        "isl": cfg["isl"],
        "osl": cfg["osl"],
        "ttft": cfg["ttft"],
        "tpot": cfg["tpot"],
    }
    for key in ["prefix", "request_latency", "enable_wideep", "enable_eplb", "moe_backend", "attention_backend"]:
        if key in cfg:
            exp_cfg[key] = cfg[key]
    return exp_cfg


def _apply_exp_overrides(exp_cfg: dict, overrides: dict | None) -> None:
    if not overrides:
        return
    for key in ["prefix", "request_latency", "enable_wideep", "enable_eplb", "moe_backend", "attention_backend"]:
        if key in overrides:
            exp_cfg[key] = overrides[key]


def _build_custom_experiment_yaml(cfg: dict, system_name: str) -> dict:
    fixed = cfg.get("fixed_parallel") or {}
    search = cfg.get("search_parallel") or {}
    exp_yaml: dict[str, object] = {"exps": []}

    for mode in ["agg", "disagg"]:
        if fixed.get(mode) and search.get(mode):
            raise ValueError(
                f"Mode '{mode}' is configured in both fixed_parallel and search_parallel; choose only one."
            )

    def _kind_and_spec(mode: str) -> tuple[str, dict] | None:
        if fixed.get(mode):
            return "fixed", fixed[mode]
        if search.get(mode):
            return "search", search[mode]
        return None

    agg_spec = _kind_and_spec("agg")
    if agg_spec:
        agg_kind, agg_cfg = agg_spec
        agg_exp = _base_exp_config(cfg, system_name)
        _apply_exp_overrides(agg_exp, agg_cfg)
        agg_exp["serving_mode"] = "agg"
        agg_exp["config"] = {"worker_config": _fixed_worker_config(cfg, agg_cfg)}
        agg_name = f"agg_{agg_kind}"
        exp_yaml["exps"].append(agg_name)
        exp_yaml[agg_name] = agg_exp

    disagg_spec = _kind_and_spec("disagg")
    if disagg_spec:
        disagg_kind, disagg_cfg = disagg_spec
        if "prefill" not in disagg_cfg or "decode" not in disagg_cfg:
            raise ValueError("disagg config must provide both 'prefill' and 'decode' sections")
        if "prefill_workers" not in disagg_cfg or "decode_workers" not in disagg_cfg:
            raise ValueError("disagg config must provide both 'prefill_workers' and 'decode_workers'")

        disagg_exp = _base_exp_config(cfg, system_name)
        _apply_exp_overrides(disagg_exp, disagg_cfg)
        disagg_exp["serving_mode"] = "disagg"
        disagg_exp["decode_system_name"] = system_name
        replica_gpu_list = _int_list(disagg_cfg.get("num_gpu_per_replica", cfg["total_gpus"]))
        prefill_worker_list = _int_list(disagg_cfg["prefill_workers"])
        decode_worker_list = _int_list(disagg_cfg["decode_workers"])
        disagg_exp["config"] = {
            "prefill_worker_config": _fixed_worker_config(cfg, disagg_cfg["prefill"]),
            "decode_worker_config": _fixed_worker_config(cfg, disagg_cfg["decode"]),
            "replica_config": {
                "num_gpu_per_replica": replica_gpu_list,
                "max_gpu_per_replica": max(replica_gpu_list),
                "prefill_num_worker_list": prefill_worker_list,
                "decode_num_worker_list": decode_worker_list,
                "max_prefill_worker": max(prefill_worker_list),
                "max_decode_worker": max(decode_worker_list),
            },
        }
        disagg_name = f"disagg_{disagg_kind}"
        exp_yaml["exps"].append(disagg_name)
        exp_yaml[disagg_name] = disagg_exp

    if not exp_yaml["exps"]:
        raise ValueError("No modes found in fixed_parallel/search_parallel")
    return exp_yaml


def _run_aic(cfg: dict, system_name: str, save_dir: Path, log_path: Path) -> int:
    custom_parallel = _has_custom_parallel(cfg)
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC_DIR) if not existing_pythonpath else f"{SRC_DIR}:{existing_pythonpath}"
    if custom_parallel:
        exp_yaml = _build_custom_experiment_yaml(cfg, system_name)
        save_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".yaml", prefix="topology_fixed_", dir=save_dir, delete=False
        ) as tmp:
            yaml.safe_dump(exp_yaml, tmp, sort_keys=False)
            yaml_path = tmp.name
        cmd = [
            sys.executable,
            "-m",
            "aiconfigurator.cli.main",
            "exp",
            "--systems-paths",
            cfg["systems_path"],
            "--yaml-path",
            yaml_path,
            "--save-dir",
            str(save_dir),
        ]
        return run_cmd(cmd, log_path, cwd=REPO_ROOT, env=env)

    cmd = [
        sys.executable,
        "-m",
        "aiconfigurator.cli.main",
        "default",
        "--systems-paths",
        cfg["systems_path"],
        "--model",
        cfg["model"],
        "--total-gpus",
        str(cfg["total_gpus"]),
        "--backend",
        cfg["backend"],
        "--backend-version",
        cfg["backend_version"],
        "--isl",
        str(cfg["isl"]),
        "--osl",
        str(cfg["osl"]),
        "--gemm-quant-mode",
        cfg["gemm_quant_mode"],
        "--database-mode",
        cfg["database_mode"],
        "--ttft",
        str(cfg["ttft"]),
        "--tpot",
        str(cfg["tpot"]),
        "--system",
        system_name,
        "--save-dir",
        str(save_dir),
    ]
    return run_cmd(cmd, log_path, cwd=REPO_ROOT, env=env)


def _read_first_csv_row(csv_path: Path) -> dict[str, str]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty csv: {csv_path}")
    return rows[0]


def _read_best_pareto_row(csv_path: Path) -> dict[str, str]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty pareto csv: {csv_path}")
    return max(rows, key=lambda row: float(row.get("tokens/s/gpu_cluster") or 0.0))


def _best_rows_by_mode(
    best_config_paths: dict[str, Path], pareto_paths: dict[str, Path], modes: list[str]
) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for mode in modes:
        best_path = best_config_paths.get(mode)
        if best_path is not None:
            rows[mode] = _read_first_csv_row(best_path)
            continue
        pareto_path = pareto_paths.get(mode)
        if pareto_path is None:
            raise ValueError(f"missing best_config_topn.csv and pareto.csv for mode={mode}")
        rows[mode] = _read_best_pareto_row(pareto_path)
    return rows


def _row_value(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return ""


def _write_compare_summary(
    scaleup_best: dict[str, Path],
    scaleout_best: dict[str, Path],
    scaleup: dict[str, Path],
    scaleout: dict[str, Path],
    output_csv: Path,
    scaleup_label: str,
    scaleout_label: str,
    modes: list[str],
) -> None:
    rows = []
    scaleup_rows = _best_rows_by_mode(scaleup_best, scaleup, modes)
    scaleout_rows = _best_rows_by_mode(scaleout_best, scaleout, modes)

    for mode in modes:
        su_row = scaleup_rows[mode]
        so_row = scaleout_rows[mode]
        metric_pairs = [
            ("best_throughput", _row_value(su_row, "tokens/s/gpu_cluster"), _row_value(so_row, "tokens/s/gpu_cluster")),
            ("per_gpu_throughput", _row_value(su_row, "tokens/s/gpu"), _row_value(so_row, "tokens/s/gpu")),
            ("per_user_throughput", _row_value(su_row, "tokens/s/user"), _row_value(so_row, "tokens/s/user")),
            ("ttft_ms", _row_value(su_row, "ttft"), _row_value(so_row, "ttft")),
            ("tpot_ms", _row_value(su_row, "tpot"), _row_value(so_row, "tpot")),
            ("request_latency_ms", _row_value(su_row, "request_latency"), _row_value(so_row, "request_latency")),
        ]

        if mode == "agg":
            metric_pairs.extend(
                [
                    ("tp", _row_value(su_row, "tp"), _row_value(so_row, "tp")),
                    ("pp", _row_value(su_row, "pp"), _row_value(so_row, "pp")),
                    ("dp", _row_value(su_row, "dp"), _row_value(so_row, "dp")),
                    ("moe_tp", _row_value(su_row, "moe_tp"), _row_value(so_row, "moe_tp")),
                    ("moe_ep", _row_value(su_row, "moe_ep"), _row_value(so_row, "moe_ep")),
                    ("bs", _row_value(su_row, "bs"), _row_value(so_row, "bs")),
                ]
            )
        else:
            metric_pairs.extend(
                [
                    ("prefill_workers", _row_value(su_row, "(p)workers"), _row_value(so_row, "(p)workers")),
                    ("decode_workers", _row_value(su_row, "(d)workers"), _row_value(so_row, "(d)workers")),
                    ("prefill_tp", _row_value(su_row, "(p)tp"), _row_value(so_row, "(p)tp")),
                    ("decode_tp", _row_value(su_row, "(d)tp"), _row_value(so_row, "(d)tp")),
                    ("prefill_dp", _row_value(su_row, "(p)dp"), _row_value(so_row, "(p)dp")),
                    ("decode_dp", _row_value(su_row, "(d)dp"), _row_value(so_row, "(d)dp")),
                    ("prefill_moe_ep", _row_value(su_row, "(p)moe_ep"), _row_value(so_row, "(p)moe_ep")),
                    ("decode_moe_ep", _row_value(su_row, "(d)moe_ep"), _row_value(so_row, "(d)moe_ep")),
                    ("prefill_bs", _row_value(su_row, "(p)bs"), _row_value(so_row, "(p)bs")),
                    ("decode_bs", _row_value(su_row, "(d)bs"), _row_value(so_row, "(d)bs")),
                ]
            )

        for metric, su, so in metric_pairs:
            delta = ""
            try:
                delta = str(float(su) - float(so))
            except Exception:
                pass
            rows.append({"mode": mode, "metric": metric, scaleup_label: su, scaleout_label: so, "delta": delta})

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "metric", scaleup_label, scaleout_label, "delta"])
        writer.writeheader()
        writer.writerows(rows)


def _compact_best_config(mode: str, row: dict[str, str]) -> dict[str, str]:
    compact = {
        "tokens/s/gpu_cluster": _row_value(row, "tokens/s/gpu_cluster"),
        "tokens/s/gpu": _row_value(row, "tokens/s/gpu"),
        "tokens/s/user": _row_value(row, "tokens/s/user"),
        "ttft": _row_value(row, "ttft"),
        "tpot": _row_value(row, "tpot"),
        "request_latency": _row_value(row, "request_latency"),
    }
    if mode == "agg":
        compact.update(
            {
                "tp": _row_value(row, "tp"),
                "pp": _row_value(row, "pp"),
                "dp": _row_value(row, "dp"),
                "moe_tp": _row_value(row, "moe_tp"),
                "moe_ep": _row_value(row, "moe_ep"),
                "bs": _row_value(row, "bs"),
            }
        )
    else:
        compact.update(
            {
                "prefill_workers": _row_value(row, "(p)workers"),
                "decode_workers": _row_value(row, "(d)workers"),
                "prefill_tp": _row_value(row, "(p)tp"),
                "decode_tp": _row_value(row, "(d)tp"),
                "prefill_dp": _row_value(row, "(p)dp"),
                "decode_dp": _row_value(row, "(d)dp"),
                "prefill_moe_ep": _row_value(row, "(p)moe_ep"),
                "decode_moe_ep": _row_value(row, "(d)moe_ep"),
                "prefill_bs": _row_value(row, "(p)bs"),
                "decode_bs": _row_value(row, "(d)bs"),
            }
        )
    return compact


def _print_and_save_best_configs(
    scaleup_best: dict[str, Path],
    scaleout_best: dict[str, Path],
    scaleup: dict[str, Path],
    scaleout: dict[str, Path],
    out_dir: Path,
    scaleup_label: str,
    scaleout_label: str,
    modes: list[str],
) -> None:
    best_dir = out_dir / "best_configs"
    best_dir.mkdir(parents=True, exist_ok=True)
    scaleup_rows = _best_rows_by_mode(scaleup_best, scaleup, modes)
    scaleout_rows = _best_rows_by_mode(scaleout_best, scaleout, modes)

    for mode in modes:
        su_row = scaleup_rows[mode]
        so_row = scaleout_rows[mode]

        print(f"[{mode}] {scaleup_label}: {json.dumps(_compact_best_config(mode, su_row), ensure_ascii=False)}")
        print(f"[{mode}] {scaleout_label}: {json.dumps(_compact_best_config(mode, so_row), ensure_ascii=False)}")

        with (best_dir / f"best_config_{mode}_scaleup.json").open("w", encoding="utf-8") as f:
            json.dump(su_row, f, indent=2, ensure_ascii=False)
        with (best_dir / f"best_config_{mode}_scaleout.json").open("w", encoding="utf-8") as f:
            json.dump(so_row, f, indent=2, ensure_ascii=False)


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
    scaleup_label = _display_label(cfg.get("scaleup_system", ""), "scaleup")
    scaleout_label = _display_label(cfg.get("scaleout_system", ""), "scaleout")
    scaleup_log = out_dir / "output_scaleup.log"
    scaleout_log = out_dir / "output_scaleout.log"
    requested_modes = _requested_modes(cfg)

    rc1 = _run_aic(cfg, cfg["scaleup_system"], scaleup_dir, scaleup_log)
    rc2 = _run_aic(cfg, cfg["scaleout_system"], scaleout_dir, scaleout_log)

    if rc1 != 0 or rc2 != 0:
        raise SystemExit(f"run failed: scaleup_rc={rc1}, scaleout_rc={rc2}")

    up = find_pareto_by_mode(scaleup_dir)
    out = find_pareto_by_mode(scaleout_dir)
    up_best = find_best_config_by_mode(scaleup_dir)
    out_best = find_best_config_by_mode(scaleout_dir)
    up_all = find_result_csv_by_mode(scaleup_dir, "all_results.csv")
    out_all = find_result_csv_by_mode(scaleout_dir, "all_results.csv")

    modes = requested_modes
    missing_required_modes: list[str] = []
    for m in modes:
        up_p = up.get(m)
        out_p = out.get(m)
        if not up_p or not out_p:
            print(f"skip mode={m}: missing pareto. scaleup={up_p} scaleout={out_p}")
            missing_required_modes.append(m)
            continue
        canon_up = scaleup_dir / f"pareto_{m}.csv"
        canon_out = scaleout_dir / f"pareto_{m}.csv"
        shutil.copy2(up_p, canon_up)
        shutil.copy2(out_p, canon_out)

        up_all_p = up_all.get(m)
        out_all_p = out_all.get(m)
        if up_all_p and out_all_p:
            canon_up_all = scaleup_dir / f"all_results_{m}.csv"
            canon_out_all = scaleout_dir / f"all_results_{m}.csv"
            shutil.copy2(up_all_p, canon_up_all)
            shutil.copy2(out_all_p, canon_out_all)

            full_plot_cmd = [
                "uv", "run", "--frozen", "python", "tools/plot_pareto_compare.py",
                "--scaleup-csv", str(canon_up_all),
                "--scaleout-csv", str(canon_out_all),
                "--scaleup-label", scaleup_label,
                "--scaleout-label", scaleout_label,
                "--x-col", cfg.get("x_col", "tokens/s/user"),
                "--y-col", cfg.get("y_col", "tokens/s/gpu"),
                "--title", f"DS-V4 Flash {m} All Candidates",
                "--output", str(plot_dir / f"full_compare_{m}.png"),
            ]
            full_rc = subprocess.run(full_plot_cmd, text=True, cwd=REPO_ROOT)
            if full_rc.returncode != 0:
                print(f"full plot failed for mode={m}")

        plot_cmd = [
            "uv", "run", "--frozen", "python", "tools/plot_pareto_compare.py",
            "--scaleup-csv", str(canon_up),
            "--scaleout-csv", str(canon_out),
            "--scaleup-label", scaleup_label,
            "--scaleout-label", scaleout_label,
            "--x-col", cfg.get("x_col", "tokens/s/user"),
            "--y-col", cfg.get("y_col", "tokens/s/gpu"),
            "--title", f"DS-V4 Flash {m} Scale-up vs Scale-out",
            "--output", str(plot_dir / f"pareto_compare_{m}.png"),
        ]
        prc = subprocess.run(plot_cmd, text=True, cwd=REPO_ROOT)
        if prc.returncode != 0:
            print(f"plot failed for mode={m}")

    if missing_required_modes:
        missing_str = ", ".join(missing_required_modes)
        raise SystemExit(
            f"missing requested pareto for mode(s): {missing_str}. Check {scaleup_log} and {scaleout_log} for details."
        )

    summary_csv = out_dir / "compare_single_point.csv"
    if _has_custom_parallel(cfg):
        _write_compare_summary(up_best, out_best, up, out, summary_csv, scaleup_label, scaleout_label, modes)
    else:
        extract_cmd = [
            "python", "tools/extract_final_metrics.py",
            "--scaleup-log", str(scaleup_log),
            "--scaleout-log", str(scaleout_log),
            "--scaleup-label", scaleup_label,
            "--scaleout-label", scaleout_label,
            "--output-csv", str(summary_csv),
        ]
        subprocess.run(extract_cmd, check=False)


    _print_and_save_best_configs(up_best, out_best, up, out, out_dir, scaleup_label, scaleout_label, modes)
    print("done")


if __name__ == "__main__":
    main()
