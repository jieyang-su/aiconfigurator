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


def _resolve_perf_source_from_cfg(cfg: dict, system_name: str) -> dict[str, str] | None:
    systems_path = Path(cfg["systems_path"])
    if not systems_path.is_absolute():
        systems_path = (REPO_ROOT / systems_path).resolve()
    system_yaml = systems_path / f"{system_name}.yaml"
    if not system_yaml.exists():
        return None
    try:
        system_spec = yaml.safe_load(system_yaml.read_text(encoding="utf-8")) or {}
        data_dir = system_spec.get("data_dir")
        if not data_dir:
            return None
    except Exception:
        return None

    data_path = systems_path / data_dir / cfg["backend"] / cfg["backend_version"]
    return {
        "systems_root": str(systems_path),
        "system_yaml_path": str(system_yaml),
        "data_path": str(data_path.resolve()),
    }


def _resolve_aiconfigurator_file(env: dict[str, str]) -> str:
    probe = [
        sys.executable,
        "-c",
        "import aiconfigurator; print(getattr(aiconfigurator, '__file__', '<unknown>'))",
    ]
    try:
        result = subprocess.run(
            probe,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env=env,
            check=False,
        )
        out = (result.stdout or "").strip()
        return out if out else "<unresolved>"
    except Exception:
        return "<unresolved>"


def _build_automation_context_lines(cfg: dict, system_name: str, env: dict[str, str]) -> list[str]:
    systems_path = Path(cfg["systems_path"])
    if not systems_path.is_absolute():
        systems_path = (REPO_ROOT / systems_path).resolve()

    perf_source = _resolve_perf_source_from_cfg(cfg, system_name)
    lines = [
        f"[automation-context] sys.executable={sys.executable}",
        f"[automation-context] aiconfigurator.__file__={_resolve_aiconfigurator_file(env)}",
        f"[automation-context] resolved systems path={systems_path}",
    ]
    if perf_source is None:
        lines.append("[automation-context] perf file source=<unresolved>")
    else:
        lines.append(
            "[automation-context] perf file source: "
            f"systems_root={perf_source['systems_root']} "
            f"system_yaml={perf_source['system_yaml_path']} "
            f"data_path={perf_source['data_path']}"
        )
    return lines


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
    with log_path.open("a", encoding="utf-8") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, cwd=cwd, env=env)
    return p.returncode


def find_result_csv_by_mode(root: Path, filename: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in root.rglob(filename):
        rel_parts = [part.lower() for part in p.relative_to(root).parts]
        rel_path = "/".join(rel_parts)
        is_disagg = any(part == "disagg" or part.startswith("disagg_") for part in rel_parts)
        is_agg = any(part == "agg" or part.startswith("agg_") for part in rel_parts)
        if is_disagg:
            current = out.get("disagg")
            if current is None or p.stat().st_mtime > current.stat().st_mtime:
                out["disagg"] = p
        elif is_agg:
            current = out.get("agg")
            if current is None or p.stat().st_mtime > current.stat().st_mtime:
                out["agg"] = p
    return out


def find_pareto_by_mode(root: Path) -> dict[str, Path]:
    return find_result_csv_by_mode(root, "pareto.csv")


def find_best_config_by_mode(root: Path) -> dict[str, Path]:
    return find_result_csv_by_mode(root, "best_config_topn.csv")


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
    context_lines = _build_automation_context_lines(cfg, system_name, env)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        for line in context_lines:
            print(line)
            f.write(line + "\n")

    if custom_parallel:
        exp_yaml = _build_custom_experiment_yaml(cfg, system_name)
        save_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".yaml", prefix="single_system_fixed_", dir=save_dir, delete=False
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


def _best_row_for_mode(best_config_paths: dict[str, Path], pareto_paths: dict[str, Path], mode: str) -> dict[str, str]:
    best_path = best_config_paths.get(mode)
    if best_path is not None:
        return _read_first_csv_row(best_path)
    pareto_path = pareto_paths.get(mode)
    if pareto_path is None:
        raise ValueError(f"missing best_config_topn.csv and pareto.csv for mode={mode}")
    return _read_best_pareto_row(pareto_path)


def _best_row_for_mode_or_none(
    best_config_paths: dict[str, Path], pareto_paths: dict[str, Path], mode: str
) -> dict[str, str] | None:
    try:
        return _best_row_for_mode(best_config_paths, pareto_paths, mode)
    except ValueError:
        return None


def _row_value(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return ""


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


def _write_mode_summary(agg_row: dict[str, str] | None, disagg_row: dict[str, str] | None, output_csv: Path) -> None:
    agg_row = agg_row or {}
    disagg_row = disagg_row or {}
    metric_pairs = [
        ("best_throughput", _row_value(agg_row, "tokens/s/gpu_cluster"), _row_value(disagg_row, "tokens/s/gpu_cluster")),
        ("per_gpu_throughput", _row_value(agg_row, "tokens/s/gpu"), _row_value(disagg_row, "tokens/s/gpu")),
        ("per_user_throughput", _row_value(agg_row, "tokens/s/user"), _row_value(disagg_row, "tokens/s/user")),
        ("ttft_ms", _row_value(agg_row, "ttft"), _row_value(disagg_row, "ttft")),
        ("tpot_ms", _row_value(agg_row, "tpot"), _row_value(disagg_row, "tpot")),
        ("request_latency_ms", _row_value(agg_row, "request_latency"), _row_value(disagg_row, "request_latency")),
        ("agg_tp", _row_value(agg_row, "tp"), ""),
        ("agg_dp", _row_value(agg_row, "dp"), ""),
        ("agg_moe_ep", _row_value(agg_row, "moe_ep"), ""),
        ("disagg_prefill_workers", "", _row_value(disagg_row, "(p)workers")),
        ("disagg_decode_workers", "", _row_value(disagg_row, "(d)workers")),
        ("disagg_prefill_tp", "", _row_value(disagg_row, "(p)tp")),
        ("disagg_decode_tp", "", _row_value(disagg_row, "(d)tp")),
    ]

    rows = []
    for metric, agg_value, disagg_value in metric_pairs:
        delta = ""
        try:
            delta = str(float(agg_value) - float(disagg_value))
        except Exception:
            pass
        rows.append({"metric": metric, "agg": agg_value, "disagg": disagg_value, "delta": delta})

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "agg", "disagg", "delta"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="One-command DS-V4 Flash single-system agg vs disagg automation")
    ap.add_argument("--config", required=True, help="JSON config path")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))

    out_dir = Path(cfg["out_dir"])
    run_dir = out_dir / "run"
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "output.log"
    requested_modes = _requested_modes(cfg)

    rc = _run_aic(cfg, cfg["system"], run_dir, log_path)
    if rc != 0:
        raise SystemExit(f"run failed: rc={rc}. Check {log_path}")

    pareto = find_pareto_by_mode(run_dir)
    best = find_best_config_by_mode(run_dir)
    all_results = find_result_csv_by_mode(run_dir, "all_results.csv")

    available_modes: list[str] = []
    for m in requested_modes:
        p = pareto.get(m)
        if not p:
            print(f"skip mode={m}: missing pareto. path={p}")
            continue
        available_modes.append(m)
        shutil.copy2(p, run_dir / f"pareto_{m}.csv")
        a = all_results.get(m)
        if a:
            shutil.copy2(a, run_dir / f"all_results_{m}.csv")

    if not available_modes:
        missing_str = ", ".join(requested_modes)
        raise SystemExit(f"no pareto generated for requested mode(s): {missing_str}. Check {log_path} for details.")

    agg_all = run_dir / "all_results_agg.csv"
    disagg_all = run_dir / "all_results_disagg.csv"
    agg_pareto = run_dir / "pareto_agg.csv"
    disagg_pareto = run_dir / "pareto_disagg.csv"

    if agg_all.exists() or disagg_all.exists():
        full_plot_cmd = [
            "uv", "run", "--frozen", "python", "tools/plot_pareto_compare.py",
            "--scaleup-label", "agg",
            "--scaleout-label", "disagg",
            "--x-col", cfg.get("x_col", "tokens/s/user"),
            "--y-col", cfg.get("y_col", "tokens/s/gpu"),
            "--title", "DS-V4 Flash agg vs disagg All Candidates",
            "--output", str(plot_dir / "full_compare_agg_vs_disagg.png"),
        ]
        if agg_all.exists():
            full_plot_cmd.extend(["--scaleup-csv", str(agg_all)])
        if disagg_all.exists():
            full_plot_cmd.extend(["--scaleout-csv", str(disagg_all)])
        full_rc = subprocess.run(full_plot_cmd, text=True, cwd=REPO_ROOT)
        if full_rc.returncode != 0:
            print("full plot failed")

    pareto_plot_cmd = [
        "uv", "run", "--frozen", "python", "tools/plot_pareto_compare.py",
        "--scaleup-label", "agg",
        "--scaleout-label", "disagg",
        "--x-col", cfg.get("x_col", "tokens/s/user"),
        "--y-col", cfg.get("y_col", "tokens/s/gpu"),
        "--title", "DS-V4 Flash agg vs disagg",
        "--output", str(plot_dir / "pareto_compare_agg_vs_disagg.png"),
    ]
    if agg_pareto.exists():
        pareto_plot_cmd.extend(["--scaleup-csv", str(agg_pareto)])
    if disagg_pareto.exists():
        pareto_plot_cmd.extend(["--scaleout-csv", str(disagg_pareto)])
    prc = subprocess.run(pareto_plot_cmd, text=True, cwd=REPO_ROOT)
    if prc.returncode != 0:
        print("pareto plot failed")

    agg_row = _best_row_for_mode_or_none(best, pareto, "agg")
    disagg_row = _best_row_for_mode_or_none(best, pareto, "disagg")
    _write_mode_summary(agg_row, disagg_row, out_dir / "compare_agg_disagg_single_point.csv")

    best_dir = out_dir / "best_configs"
    best_dir.mkdir(parents=True, exist_ok=True)
    if agg_row is not None:
        with (best_dir / "best_config_agg.json").open("w", encoding="utf-8") as f:
            json.dump(agg_row, f, indent=2, ensure_ascii=False)
        print(f"[agg] {json.dumps(_compact_best_config('agg', agg_row), ensure_ascii=False)}")
    else:
        print("[agg] unavailable")

    if disagg_row is not None:
        with (best_dir / "best_config_disagg.json").open("w", encoding="utf-8") as f:
            json.dump(disagg_row, f, indent=2, ensure_ascii=False)
        print(f"[disagg] {json.dumps(_compact_best_config('disagg', disagg_row), ensure_ascii=False)}")
    else:
        print("[disagg] unavailable")
    print("done")


if __name__ == "__main__":
    main()
