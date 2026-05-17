import argparse
import importlib.util
import json
import os
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
BENCHMARK_SCRIPT = EXPERIMENT_DIR / "run_12vm_branch_cut_benchmark.py"

def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("branch_cut_benchmark", BENCHMARK_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description="bc00 기준선을 time limit 없이 단독 실행합니다.")
    parser.add_argument("--threads", type=int, default=min(32, os.cpu_count() or 8))
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--time-limit", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario-seed", type=int, default=42)
    parser.add_argument("--scenario-count", type=int, default=10)
    parser.add_argument("--on-demand", type=int, default=4)
    parser.add_argument("--spot", type=int, default=4)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--max-vcpu", type=int, default=8)
    parser.add_argument("--server-capacity", type=float, default=8.0)
    parser.add_argument("--min-avg-cpu", type=float, default=20.0)
    parser.add_argument("--no-rel-heur-time", type=float, default=0.0)
    return parser.parse_args()


def build_case_name(on_demand, spot, batch, scenario_count, server_capacity):
    total_vm = on_demand + spot + batch
    return (
        f"chance_2sp_toy_{total_vm}vm_combination_od_sp_bj_"
        f"od{on_demand}_sp{spot}_bj{batch}_sc{scenario_count}_"
        f"cap{int(server_capacity)}_avg20_objenergy"
    )


def build_output_dirs(total_vm, server_capacity, max_vcpu, no_rel_heur_time, time_limit):
    suffix = f"chance_2sp_toy_{total_vm}vm_bc00_open_cap{int(server_capacity)}_vcpu{int(max_vcpu)}"
    if no_rel_heur_time and no_rel_heur_time > 0.0:
        suffix += f"_norel{int(no_rel_heur_time)}"
    if time_limit and time_limit > 0.0:
        suffix += f"_t{int(time_limit)}"
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
        no_rel_heur_time=args.no_rel_heur_time,
        time_limit=args.time_limit,
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

    config = next(profile for profile in benchmark.BENCHMARK_PROFILES if profile["name"] == "bc00_control_state_phi")
    config = dict(config)
    config["no_rel_heur_time"] = args.no_rel_heur_time

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
        "analysis_dir": str(analysis_dir),
        "runs_group_dir": str(runs_group_dir),
    }
    (analysis_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    row = benchmark.solve_single_experiment(
        (
            case_name,
            instance_dir,
            config,
            float(args.time_limit),
            args.mip_gap,
            args.threads,
        )
    )

    output_path = analysis_dir / "final_summary.csv"
    output_path.write_text("", encoding="utf-8")
    import pandas as pd

    pd.DataFrame([row]).to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
