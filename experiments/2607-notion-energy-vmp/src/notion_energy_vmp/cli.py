from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

from .data import load_instance, load_yaml, write_instance_artifacts, write_prepared_data
from .model import build_model, configure_solver
from .reporting import write_no_solution_summary, write_solution_reports


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Notion energy-aware two-stage VM placement MILP")
    p.add_argument("--config", required=True)
    p.add_argument("--run-dir")
    p.add_argument("--time-limit", type=float)
    p.add_argument("--mip-gap", type=float)
    p.add_argument("--threads", type=int)
    p.add_argument("--num-scenarios", type=int)
    p.add_argument("--target-total-vms", type=int)
    p.add_argument("--num-servers", type=int)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--validate-only", action="store_true", help="Deprecated alias for --prepare-only")
    p.add_argument("--overwrite-prepared", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = copy.deepcopy(load_yaml(args.config))
    if args.time_limit is not None:
        cfg["solver"]["time_limit_seconds"] = args.time_limit
    if args.mip_gap is not None:
        cfg["solver"]["mip_gap"] = args.mip_gap
    if args.threads is not None:
        cfg["solver"]["threads"] = args.threads
    if args.num_scenarios is not None:
        cfg["experiment"]["num_scenarios"] = args.num_scenarios
        cfg["energy_data"]["scenario_dates"] = cfg["energy_data"]["scenario_dates"][: args.num_scenarios]
    if args.target_total_vms is not None:
        cfg["experiment"]["target_total_vms"] = args.target_total_vms
        cfg["experiment"]["class_counts"] = None
    if args.num_servers is not None:
        cfg["experiment"]["num_servers"] = args.num_servers

    root = Path(cfg["workspace_root"]).expanduser().resolve()
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = root / run_dir
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = root / "runs" / f"{cfg['experiment']['name']}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    print(f"[1/4] Loading and validating data -> {run_dir}", flush=True)
    instance = load_instance(cfg)
    write_instance_artifacts(instance, run_dir)
    prepared_dir = write_prepared_data(
        instance,
        run_dir / "prepared_data",
        overwrite=args.overwrite_prepared,
    )
    print(f"Canonical prepared data -> {prepared_dir}", flush=True)
    print(json.dumps(instance.summary(), indent=2, ensure_ascii=False), flush=True)
    if args.prepare_only or args.validate_only:
        print("Data preparation and audit completed; model was not built or solved.", flush=True)
        return 0

    print("[2/4] Building deterministic-equivalent MILP", flush=True)
    artifacts = build_model(instance)
    model = artifacts.model
    configure_solver(model, cfg["solver"], run_dir)
    if cfg["solver"].get("write_lp", True):
        model.write(str(run_dir / "model.lp"))
    if cfg["solver"].get("write_mps", True):
        model.write(str(run_dir / "model.mps"))

    print("[3/4] Starting Gurobi optimization", flush=True)
    model.optimize()
    if model.SolCount > 0:
        print("[4/4] Writing incumbent solution and metrics", flush=True)
        summary = write_solution_reports(artifacts, run_dir)
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        return 0

    print("[4/4] No incumbent solution; writing diagnostics", flush=True)
    try:
        from gurobipy import GRB

        if model.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
            model.computeIIS()
            model.write(str(run_dir / "iis.ilp"))
    finally:
        write_no_solution_summary(model, run_dir)
    return 2


if __name__ == "__main__":
    sys.exit(main())
