import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from build_dataset import build_instance
from run_model import solve_instance
from visualize import create_case_plots, create_ratio_comparison_figure


EXPERIMENT_DIR = Path(__file__).resolve().parent
EXPERIMENT_NAME = EXPERIMENT_DIR.name
REPO_ROOT = EXPERIMENT_DIR.parents[1]
SOURCE_CSV = REPO_ROOT / "data" / "processed" / "2601-initial-toy-model" / "sample_vm_data.csv"
DATA_ROOT = REPO_ROOT / "data" / "processed" / EXPERIMENT_NAME
RESULTS_ROOT = EXPERIMENT_DIR / "results" / "runs"
ANALYSIS_DIR = EXPERIMENT_DIR / "results" / "analysis" / "chance_2sp_toy_30vm_ratio_sweep_cap4"
DEFAULT_THREADS = 8

CASE_CONFIGS = [
    {"case": "balanced", "label": "balanced", "on_demand": 10, "spot": 10, "batch": 10},
    {"case": "service_heavy", "label": "service-heavy", "on_demand": 18, "spot": 6, "batch": 6},
    {"case": "spot_heavy", "label": "spot-heavy", "on_demand": 6, "spot": 18, "batch": 6},
    {"case": "batch_heavy", "label": "batch-heavy", "on_demand": 6, "spot": 6, "batch": 18},
]


def parse_args():
    parser = argparse.ArgumentParser(description="30-VM 비율 스윕을 실행합니다.")
    parser.add_argument("--source-csv", type=Path, default=SOURCE_CSV)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario-count", type=int, default=10)
    parser.add_argument("--time-limit", type=int, default=1800)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    return parser.parse_args()


def build_case_name(case_config, scenario_count):
    return (
        f"chance_2sp_toy_30vm_{case_config['case']}_"
        f"od{case_config['on_demand']}_sp{case_config['spot']}_bj{case_config['batch']}_"
        f"sc{scenario_count}_cap4"
    )


def collect_case_row(case_config, case_name, summary, visualization_outputs):
    migration_events = pd.read_csv(visualization_outputs["migration_events_csv"])
    spot_suspensions = pd.read_csv(visualization_outputs["spot_suspension_events_csv"])
    total_vm = case_config["on_demand"] + case_config["spot"] + case_config["batch"]

    return {
        "case": case_config["label"],
        "instance_name": case_name,
        "total_vm": total_vm,
        "on_demand": case_config["on_demand"],
        "spot": case_config["spot"],
        "batch": case_config["batch"],
        "on_demand_share": case_config["on_demand"] / total_vm,
        "spot_share": case_config["spot"] / total_vm,
        "batch_share": case_config["batch"] / total_vm,
        "status_name": summary.get("status_name"),
        "used_server_count": summary.get("used_server_count"),
        "selected_solution_source": summary.get("selected_solution_source"),
        "fallback_attempted": summary.get("fallback_attempted"),
        "fallback_selected": summary.get("fallback_selected"),
        "runtime_seconds": summary.get("runtime_seconds"),
        "total_runtime_seconds": summary.get("total_runtime_seconds"),
        "mip_gap": summary.get("mip_gap"),
        "objective_value": summary.get("objective_value"),
        "primary_objective_value": summary.get("primary_objective_value"),
        "migration_count": summary.get("migration_count"),
        "actual_migration_event_count": len(migration_events),
        "peak_realized_server_utilization": summary.get("peak_realized_server_utilization"),
        "peak_overbooking_ratio": summary.get("peak_overbooking_ratio"),
        "peak_cluster_overbooking_ratio": summary.get("peak_cluster_overbooking_ratio"),
        "max_server_violation_probability": summary.get("max_server_violation_probability"),
        "max_spot_suspension_probability": summary.get("max_spot_suspension_probability"),
        "min_spot_completion_ratio": summary.get("min_spot_completion_ratio"),
        "total_gamma_activations": summary.get("total_gamma_activations"),
        "total_phi_activations": summary.get("total_phi_activations"),
        "worst_realized_scenario": summary.get("worst_realized_scenario"),
        "visualized_scenario": visualization_outputs["scenario"],
        "suspended_gantt_scenario": visualization_outputs.get("suspended_scenario"),
        "clean_gantt_scenario": visualization_outputs.get("clean_scenario"),
        "spot_suspension_event_count": len(spot_suspensions),
    }


def run_single_case(task):
    case_index, case_config, source_csv, seed, scenario_count, time_limit, mip_gap, threads = task
    source_csv = Path(source_csv)
    case_name = build_case_name(case_config, scenario_count)
    instance_dir = DATA_ROOT / case_name
    results_dir = RESULTS_ROOT / case_name

    build_instance(
        source_csv=source_csv,
        output_dir=instance_dir,
        instance_name=case_name,
        seed=seed + case_index,
        scenario_count=scenario_count,
        on_demand_count=case_config["on_demand"],
        spot_count=case_config["spot"],
        batch_count=case_config["batch"],
        max_vcpu=4,
        server_capacity=4.0,
        epsilon_od=0.10,
        epsilon_sp=0.20,
        rho=0.80,
        lambda_migration=0.50,
        scenario_seed=seed,
    )
    summary = solve_instance(
        instance_dir=instance_dir,
        results_dir=results_dir,
        time_limit=time_limit,
        mip_gap=mip_gap,
        threads=threads,
        use_fallback=True,
    )
    visualization_outputs = create_case_plots(instance_dir, results_dir)
    return collect_case_row(case_config, case_name, summary, visualization_outputs)


def main():
    args = parse_args()
    source_csv = args.source_csv.resolve()

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [
        (index, case_config, str(source_csv), args.seed, args.scenario_count, args.time_limit, args.mip_gap, args.threads)
        for index, case_config in enumerate(CASE_CONFIGS)
    ]

    comparison_rows = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(run_single_case, task): task[1]["label"] for task in tasks}
        for future in as_completed(futures):
            case_label = futures[future]
            case_row = future.result()
            comparison_rows.append(case_row)
            print(
                f"[done] {case_label}: status={case_row['status_name']}, "
                f"used_servers={case_row['used_server_count']}, "
                f"overbooking={case_row['peak_overbooking_ratio']:.3f}"
            )

    comparison_df = pd.DataFrame(comparison_rows).sort_values(["on_demand", "spot", "batch"], ascending=[False, False, False]).reset_index(drop=True)
    comparison_csv = ANALYSIS_DIR / "ratio_sweep_summary.csv"
    comparison_png = ANALYSIS_DIR / "ratio_sweep_comparison.png"

    comparison_df.to_csv(comparison_csv, index=False)
    create_ratio_comparison_figure(comparison_df, comparison_png)

    print(f"saved: {comparison_csv}")
    print(f"saved: {comparison_png}")


if __name__ == "__main__":
    main()
