from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .data import build_instance, load_config
from .incumbent import IncumbentSnapshotWriter, snapshot_destination
from .model import build_model, configure_solver
from .reporting import (
    write_instance_artifacts,
    write_no_solution_summary,
    write_solution_reports,
)


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the isolated 30-minute, server-minimum-time two-stage stochastic "
            "VM-placement experiment."
        )
    )
    parser.add_argument("--config", required=True, help="Baseline or derived YAML config")
    parser.add_argument("--run-dir", help="Absolute path or path relative to this experiment")
    parser.add_argument("--prepare-only", action="store_true", help="Validate/write data, then stop")
    parser.add_argument("--build-only", action="store_true", help="Build/write the MILP, then stop")
    parser.add_argument("--time-limit", type=float)
    parser.add_argument("--mip-gap", type=float)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--num-scenarios", type=int)
    parser.add_argument("--num-servers", type=int)
    parser.add_argument("--on-demand-count", type=int)
    parser.add_argument("--spot-count", type=int)
    parser.add_argument("--batch-job-count", type=int)
    parser.add_argument(
        "--save-incumbent-snapshot",
        action="store_true",
        help=(
            "Atomically overwrite incumbent_latest.sol whenever Gurobi finds "
            "a new incumbent"
        ),
    )
    return parser.parse_args(argv)


def _apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    resolved = copy.deepcopy(config)
    experiment = resolved.setdefault("experiment", {})
    solver = resolved.setdefault("solver", {})
    class_counts = experiment.get("class_counts")
    if not isinstance(class_counts, dict):
        class_counts = {}
        experiment["class_counts"] = class_counts

    # Materialize effective defaults before resolved_config.yaml is written.
    solver.setdefault("mip_gap", 0.001)
    solver.setdefault("threads", 8)
    solver.setdefault("time_limit_seconds", None)
    solver.setdefault("seed", 42)
    solver.setdefault("numeric_focus", 1)
    solver.setdefault("presolve", 2)
    solver.setdefault("write_lp", True)
    solver.setdefault("write_mps", False)
    incumbent_snapshot = solver.get("incumbent_snapshot")
    if incumbent_snapshot is None:
        incumbent_snapshot = {}
        solver["incumbent_snapshot"] = incumbent_snapshot
    if not isinstance(incumbent_snapshot, dict):
        raise TypeError("solver.incumbent_snapshot must be a mapping")
    incumbent_snapshot.setdefault("enabled", False)
    incumbent_snapshot.setdefault("filename", "incumbent_latest.sol")

    if args.time_limit is not None:
        solver["time_limit_seconds"] = args.time_limit
    if args.mip_gap is not None:
        solver["mip_gap"] = args.mip_gap
    if args.threads is not None:
        solver["threads"] = args.threads
    if args.num_scenarios is not None:
        experiment["num_scenarios"] = args.num_scenarios
    if args.num_servers is not None:
        experiment["num_servers"] = args.num_servers
    if args.on_demand_count is not None:
        class_counts["on_demand"] = args.on_demand_count
    if args.spot_count is not None:
        class_counts["spot"] = args.spot_count
    if args.batch_job_count is not None:
        class_counts.pop("batch_candidate", None)
        class_counts["batch_jobs"] = args.batch_job_count
    if getattr(args, "save_incumbent_snapshot", False):
        incumbent_snapshot["enabled"] = True
    return resolved


def _resolve_config_path(value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    # Prefer this experiment's own config namespace. This prevents a root-level
    # compatibility config with the same relative name from being loaded while
    # another experiment is running.
    experiment_candidate = (EXPERIMENT_ROOT / candidate).resolve()
    if experiment_candidate.is_file():
        return experiment_candidate
    return (Path.cwd() / candidate).resolve()


def _resolve_run_dir(config: dict, explicit: str | None) -> Path:
    if explicit:
        candidate = Path(explicit).expanduser()
        return candidate.resolve() if candidate.is_absolute() else (EXPERIMENT_ROOT / candidate).resolve()
    name = str(config.get("experiment", {}).get("name", "server_min_baseline"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")
    return (EXPERIMENT_ROOT / "runs" / f"{name}_{stamp}").resolve()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = _resolve_config_path(args.config)
    config = _apply_overrides(load_config(config_path), args)
    run_dir = _resolve_run_dir(config, args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    print(f"[1/4] Building and validating 30-minute inputs -> {run_dir}", flush=True)
    instance = build_instance(config, config_path=config_path)
    prepared_dir = write_instance_artifacts(instance, run_dir / "prepared_data")
    summary = instance.summary() if hasattr(instance, "summary") else {}
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"Prepared-data audit -> {prepared_dir}", flush=True)
    if args.prepare_only:
        print("Preparation completed; no Gurobi model was built.", flush=True)
        return 0

    print("[2/4] Building deterministic-equivalent MILP", flush=True)
    artifacts = build_model(instance)
    model = artifacts.model
    configure_solver(model, config.get("solver", {}), run_dir)
    solver_config = config.get("solver", {})
    if solver_config.get("write_lp", True):
        model.write(str(run_dir / "model.lp"))
    if solver_config.get("write_mps", False):
        model.write(str(run_dir / "model.mps"))
    if args.build_only:
        build_summary = {
            "status": "BUILT_NOT_SOLVED",
            "num_variables": int(model.NumVars),
            "num_binary_variables": int(model.NumBinVars),
            "num_constraints": int(model.NumConstrs),
            "num_general_constraints": int(model.NumGenConstrs),
        }
        (run_dir / "build_summary.json").write_text(
            json.dumps(build_summary, indent=2), encoding="utf-8"
        )
        print(json.dumps(build_summary, indent=2), flush=True)
        return 0

    print("[3/4] Starting Gurobi optimization", flush=True)
    snapshot_path = snapshot_destination(solver_config, run_dir)
    snapshot_writer = None
    if snapshot_path is not None:
        snapshot_writer = IncumbentSnapshotWriter(
            model,
            snapshot_path,
            constant_revenue_offset=sum(artifacts.objective_constants.values()),
        )
        print(
            "Incumbent snapshots will atomically overwrite "
            f"{snapshot_path}",
            flush=True,
        )
        model.optimize(snapshot_writer)
    else:
        model.optimize()
    if model.SolCount > 0:
        if snapshot_writer is not None:
            snapshot_writer.finalize(model)
        print("[4/4] Writing solution and fidelity audits", flush=True)
        solution_summary = write_solution_reports(artifacts, run_dir)
        print(json.dumps(solution_summary, indent=2, ensure_ascii=False), flush=True)
        return 0

    print("[4/4] No incumbent; writing diagnostics", flush=True)
    try:
        from gurobipy import GRB

        if model.Status == GRB.INF_OR_UNBD:
            model.Params.DualReductions = 0
            if snapshot_writer is not None:
                model.optimize(snapshot_writer)
            else:
                model.optimize()
        if model.Status == GRB.INFEASIBLE:
            model.computeIIS()
            model.write(str(run_dir / "iis.ilp"))
    finally:
        write_no_solution_summary(model, run_dir)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
