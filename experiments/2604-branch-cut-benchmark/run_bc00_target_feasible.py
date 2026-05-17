import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

from gurobipy import GRB


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
BENCHMARK_SCRIPT = EXPERIMENT_DIR / "run_12vm_branch_cut_benchmark.py"
CUT_EXPERIMENT_DIR = REPO_ROOT / "experiments" / "2604-cut-experiment"
DECOMP_EXPERIMENT_DIR = REPO_ROOT / "experiments" / "2604-decomposition-experiment"

if str(CUT_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(CUT_EXPERIMENT_DIR))
if str(DECOMP_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DECOMP_EXPERIMENT_DIR))

from decomposition_starts import build_start_strategy  # noqa: E402
from run_model import apply_var_hints  # noqa: E402


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("branch_cut_benchmark", BENCHMARK_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description="bc00 기준선에서 목적값 상한 이하 해를 하나 찾습니다.")
    parser.add_argument("--profile", type=str, default="bc00_control_state_phi")
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--objective-cap", type=float, required=True)
    parser.add_argument("--known-lb", type=float, default=None)
    parser.add_argument("--threads", type=int, default=min(32, os.cpu_count() or 8))
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario-seed", type=int, default=42)
    parser.add_argument("--solver-seed", type=int, default=0)
    parser.add_argument("--scenario-count", type=int, default=10)
    parser.add_argument("--on-demand", type=int, default=2)
    parser.add_argument("--spot", type=int, default=2)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--max-vcpu", type=int, default=4)
    parser.add_argument("--server-capacity", type=float, default=4.0)
    parser.add_argument("--min-avg-cpu", type=float, default=20.0)
    parser.add_argument("--no-rel-heur-time", type=float, default=3600.0)
    parser.add_argument("--time-limit", type=float, default=0.0)
    parser.add_argument("--heuristics", type=float, default=0.5)
    parser.add_argument("--mip-focus", type=int, default=1)
    parser.add_argument("--seed-strategy", type=str, default="")
    parser.add_argument("--seed-application", type=str, default="both")
    parser.add_argument("--seed-time-limit", type=float, default=300.0)
    parser.add_argument("--seed-threads", type=int, default=8)
    parser.add_argument("--hint-priority", type=int, default=20)
    return parser.parse_args()


def build_case_name(on_demand, spot, batch, scenario_count, server_capacity):
    total_vm = on_demand + spot + batch
    return (
        f"chance_2sp_toy_{total_vm}vm_combination_od_sp_bj_"
        f"od{on_demand}_sp{spot}_bj{batch}_sc{scenario_count}_"
        f"cap{int(server_capacity)}_avg20_objenergy"
    )


def objective_tag(value):
    text = f"{value:.3f}"
    return text.replace(".", "p")


def build_output_dirs(total_vm, server_capacity, max_vcpu, objective_cap, no_rel_heur_time, profile_name, run_tag):
    suffix = (
        f"chance_2sp_toy_{total_vm}vm_bc00_target_cap{int(server_capacity)}_"
        f"vcpu{int(max_vcpu)}_obj{objective_tag(objective_cap)}_{profile_name}"
    )
    if no_rel_heur_time and no_rel_heur_time > 0.0:
        suffix += f"_norel{int(no_rel_heur_time)}"
    if run_tag:
        suffix += f"_{run_tag}"
    analysis_dir = EXPERIMENT_DIR / "results" / "analysis" / suffix
    runs_group_dir = EXPERIMENT_DIR / "results" / "runs" / analysis_dir.name
    return analysis_dir, runs_group_dir


def main():
    args = parse_args()
    benchmark = load_benchmark_module()
    total_vm = args.on_demand + args.spot + args.batch
    analysis_dir, runs_group_dir = build_output_dirs(
        total_vm=total_vm,
        server_capacity=args.server_capacity,
        max_vcpu=args.max_vcpu,
        objective_cap=args.objective_cap,
        no_rel_heur_time=args.no_rel_heur_time,
        profile_name=args.profile,
        run_tag=args.run_tag,
    )

    benchmark.CASE_CONFIG = {
        "case": "od_sp_bj",
        "label": "OD + SP + BJ",
        "on_demand": args.on_demand,
        "spot": args.spot,
        "batch": args.batch,
    }
    benchmark.ANALYSIS_DIR = analysis_dir
    benchmark.RUNS_GROUP_DIR = runs_group_dir

    analysis_dir.mkdir(parents=True, exist_ok=True)
    runs_group_dir.mkdir(parents=True, exist_ok=True)

    case_name = build_case_name(
        on_demand=args.on_demand,
        spot=args.spot,
        batch=args.batch,
        scenario_count=args.scenario_count,
        server_capacity=args.server_capacity,
    )
    instance_dir = benchmark.DATA_ROOT / case_name
    if not (instance_dir / "instance.json").exists():
        benchmark.build_instance(
            source_csv=benchmark.SOURCE_CSV,
            output_dir=instance_dir,
            instance_name=case_name,
            seed=args.seed,
            scenario_count=args.scenario_count,
            on_demand_count=args.on_demand,
            spot_count=args.spot,
            batch_count=args.batch,
            max_vcpu=args.max_vcpu,
            min_avg_cpu=args.min_avg_cpu,
            server_capacity=args.server_capacity,
            epsilon_od=0.10,
            epsilon_sp=0.20,
            rho=0.80,
            objective_type="energy",
            lambda_migration=0.0,
            energy_idle=benchmark.ENERGY_IDLE,
            energy_cpu=benchmark.ENERGY_CPU,
            energy_migration=benchmark.ENERGY_MIGRATION,
            scenario_seed=args.scenario_seed,
        )

    config = next(profile for profile in benchmark.BENCHMARK_PROFILES if profile["name"] == args.profile)
    results_dir = runs_group_dir / case_name / config["name"]
    results_dir.mkdir(parents=True, exist_ok=True)

    data = benchmark.load_instance_data(instance_dir)
    model, variables = benchmark.build_model(
        data=data,
        results_dir=results_dir,
        time_limit=float(args.time_limit),
        mip_gap=args.mip_gap,
        log_name="solver.log",
        threads=args.threads,
        server_limit=None,
        no_rel_heur_time=float(args.no_rel_heur_time),
        cut_profile=config["cut_profile"],
    )

    start_values = None
    seed_metadata = None
    seed_runtime_seconds = 0.0
    if args.seed_strategy:
        seed_start = time.perf_counter()
        start_values, seed_metadata = build_start_strategy(
            strategy_name=args.seed_strategy,
            data=data,
            results_dir=results_dir / "seed_generation",
            cut_profile=config["cut_profile"],
            strategy_time_limit=float(args.seed_time_limit),
            strategy_threads=int(args.seed_threads),
        )
        seed_runtime_seconds = time.perf_counter() - seed_start

    objective_expr = model.getObjective()
    objective_tolerance = max(1e-6, abs(float(args.objective_cap)) * 1e-6)
    model.addConstr(objective_expr <= float(args.objective_cap) + objective_tolerance, name="objective_cap")
    if args.known_lb is not None:
        lower_bound_tolerance = max(1e-6, abs(float(args.known_lb)) * 1e-6)
        model.addConstr(
            objective_expr >= float(args.known_lb) - lower_bound_tolerance,
            name="objective_known_lb",
        )
    model.setParam("SolutionLimit", 1)
    model.setParam("MIPFocus", int(args.mip_focus))
    model.setParam("Heuristics", float(args.heuristics))
    model.setParam("Cutoff", float(args.objective_cap) + objective_tolerance)
    if int(args.solver_seed) > 0:
        model.setParam("Seed", int(args.solver_seed))

    manifest = {
        "case_name": case_name,
        "profile": config["name"],
        "threads": args.threads,
        "mip_gap": args.mip_gap,
        "time_limit": args.time_limit if args.time_limit > 0 else None,
        "on_demand": args.on_demand,
        "spot": args.spot,
        "batch": args.batch,
        "max_vcpu": args.max_vcpu,
        "server_capacity": args.server_capacity,
        "min_avg_cpu": args.min_avg_cpu,
        "no_rel_heur_time": args.no_rel_heur_time,
        "objective_cap": args.objective_cap,
        "known_lb": args.known_lb,
        "analysis_dir": str(analysis_dir),
        "runs_group_dir": str(runs_group_dir),
        "feasibility_mode": True,
        "mip_focus": args.mip_focus,
        "heuristics": args.heuristics,
        "solver_seed": args.solver_seed,
        "profile_name": args.profile,
        "run_tag": args.run_tag,
        "seed_strategy": args.seed_strategy or None,
        "seed_application": args.seed_application if args.seed_strategy else None,
        "seed_time_limit": float(args.seed_time_limit) if args.seed_strategy else None,
        "seed_threads": int(args.seed_threads) if args.seed_strategy else None,
        "hint_priority": int(args.hint_priority) if args.seed_strategy else None,
        "seed_runtime_seconds": seed_runtime_seconds,
        "seed_metadata": seed_metadata,
    }
    (analysis_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    benchmark.apply_fixed_values(variables, None)
    if args.seed_strategy and args.seed_application in {"start", "both"}:
        benchmark.apply_start_values(variables, start_values)
    else:
        benchmark.apply_start_values(variables, None)
    if args.seed_strategy and args.seed_application in {"hint", "both"}:
        apply_var_hints(variables, start_values, hint_priority=int(args.hint_priority))
    model.optimize()

    summary = benchmark.build_summary(model, data)
    summary["runtime_seconds"] = float(model.Runtime)
    summary["wall_clock_solve_seconds"] = float(model.Runtime)
    summary["solve_time_limit_seconds"] = args.time_limit if args.time_limit > 0 else None
    summary["experiment_name"] = config["name"]
    summary["experiment_notes"] = "objective cap feasibility search"
    summary["objective_cap"] = float(args.objective_cap)
    summary["known_lb"] = None if args.known_lb is None else float(args.known_lb)
    summary["feasibility_mode"] = True
    summary["solution_limit"] = 1
    summary["mip_focus"] = int(args.mip_focus)
    summary["heuristics"] = float(args.heuristics)
    summary["seed_strategy"] = args.seed_strategy or None
    summary["seed_application"] = args.seed_application if args.seed_strategy else None
    summary["seed_time_limit"] = float(args.seed_time_limit) if args.seed_strategy else None
    summary["seed_threads"] = int(args.seed_threads) if args.seed_strategy else None
    summary["seed_runtime_seconds"] = seed_runtime_seconds
    summary["seed_metadata"] = seed_metadata

    outputs = None
    if model.SolCount > 0:
        summary_updates, outputs = benchmark.extract_solution_tables(data, variables)
        summary.update(summary_updates)
        summary["objective_cap_satisfied"] = bool(summary["objective_value"] <= float(args.objective_cap) + objective_tolerance)
    else:
        summary["objective_cap_satisfied"] = False

    if outputs is not None:
        benchmark.write_solution_outputs(results_dir, summary, outputs)
    else:
        benchmark.write_summary_only(results_dir, summary)

    row = {
        "profile": config["name"],
        "instance_name": case_name,
        "cut_profile": config["cut_profile"],
        "status_name": summary.get("status_name"),
        "has_solution": summary.get("has_solution"),
        "runtime_seconds": summary.get("runtime_seconds"),
        "objective_value": summary.get("objective_value"),
        "objective_bound": summary.get("objective_bound"),
        "mip_gap": summary.get("mip_gap"),
        "node_count": summary.get("node_count"),
        "iter_count": summary.get("iter_count"),
        "work_units": summary.get("work_units"),
        "used_server_count": summary.get("used_server_count"),
        "results_dir": str(results_dir),
        "objective_cap": float(args.objective_cap),
        "known_lb": None if args.known_lb is None else float(args.known_lb),
        "objective_cap_satisfied": summary.get("objective_cap_satisfied"),
        "feasibility_mode": True,
        "seed_strategy": args.seed_strategy or None,
        "seed_application": args.seed_application if args.seed_strategy else None,
        "seed_runtime_seconds": seed_runtime_seconds,
    }

    import pandas as pd

    pd.DataFrame([row]).to_csv(analysis_dir / "final_summary.csv", index=False)

    if model.SolCount > 0 and summary.get("objective_cap_satisfied"):
        print(
            f"[found] status={summary['status_name']} objective={summary['objective_value']:.6f} "
            f"bound={summary['objective_bound']:.6f} runtime={summary['runtime_seconds']:.1f}s"
        )
    else:
        print(
            f"[not-found] status={summary['status_name']} solutions={summary['solution_count']} "
            f"runtime={summary['runtime_seconds']:.1f}s"
        )


if __name__ == "__main__":
    main()
