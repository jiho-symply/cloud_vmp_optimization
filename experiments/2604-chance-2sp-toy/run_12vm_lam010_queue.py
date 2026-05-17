import argparse
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
ANALYSIS_DIR = EXPERIMENT_DIR / "results" / "analysis" / "chance_2sp_toy_12vm_cap8_avg20_lam010_queue"
RUNS_GROUP_DIR = RESULTS_ROOT / ANALYSIS_DIR.name

LAMBDA_VALUE = 0.1
CASE_GROUPS = {
    "ratio": [
        {"case": "balanced", "label": "balanced", "on_demand": 4, "spot": 4, "batch": 4},
        {"case": "service_heavy", "label": "service-heavy", "on_demand": 8, "spot": 2, "batch": 2},
        {"case": "spot_heavy", "label": "spot-heavy", "on_demand": 2, "spot": 8, "batch": 2},
        {"case": "batch_heavy", "label": "batch-heavy", "on_demand": 2, "spot": 2, "batch": 8},
    ],
    "combination": [
        {"case": "od_only", "label": "OD only", "on_demand": 12, "spot": 0, "batch": 0},
        {"case": "od_sp", "label": "OD + SP", "on_demand": 6, "spot": 6, "batch": 0},
        {"case": "od_bj", "label": "OD + BJ", "on_demand": 6, "spot": 0, "batch": 6},
        {"case": "od_sp_bj", "label": "OD + SP + BJ", "on_demand": 4, "spot": 4, "batch": 4},
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="12-VM / cap8 / avg_cpu>=20 / lambda=0.1 실험을 큐 방식으로 실행합니다.")
    parser.add_argument("--source-csv", type=Path, default=SOURCE_CSV)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario-seed", type=int, default=42)
    parser.add_argument("--scenario-count", type=int, default=10)
    parser.add_argument("--time-limit", type=int, default=7200)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def lambda_tag(lambda_value):
    return f"lam{int(round(lambda_value * 100)):03d}"


def build_case_name(group_name, case_config, scenario_count):
    total_vm = case_config["on_demand"] + case_config["spot"] + case_config["batch"]
    return (
        f"chance_2sp_toy_{total_vm}vm_{group_name}_{case_config['case']}_"
        f"od{case_config['on_demand']}_sp{case_config['spot']}_bj{case_config['batch']}_"
        f"sc{scenario_count}_cap8_avg20_{lambda_tag(LAMBDA_VALUE)}"
    )


def read_csv_safe(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def collect_case_row(group_name, case_config, case_name, summary, visualization_outputs):
    migration_events = read_csv_safe(visualization_outputs["migration_events_csv"])
    spot_suspensions = read_csv_safe(visualization_outputs["spot_suspension_events_csv"])
    total_vm = case_config["on_demand"] + case_config["spot"] + case_config["batch"]

    return {
        "group": group_name,
        "case": case_config["label"],
        "instance_name": case_name,
        "lambda_migration": LAMBDA_VALUE,
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
    (
        group_name,
        case_config,
        case_seed,
        source_csv,
        scenario_seed,
        scenario_count,
        time_limit,
        mip_gap,
        threads,
    ) = task

    source_csv = Path(source_csv)
    case_name = build_case_name(group_name, case_config, scenario_count)
    instance_dir = DATA_ROOT / case_name
    results_dir = RUNS_GROUP_DIR / case_name
    results_dir.parent.mkdir(parents=True, exist_ok=True)

    build_instance(
        source_csv=source_csv,
        output_dir=instance_dir,
        instance_name=case_name,
        seed=case_seed,
        scenario_count=scenario_count,
        on_demand_count=case_config["on_demand"],
        spot_count=case_config["spot"],
        batch_count=case_config["batch"],
        max_vcpu=8,
        min_avg_cpu=20.0,
        server_capacity=8.0,
        epsilon_od=0.10,
        epsilon_sp=0.20,
        rho=0.80,
        lambda_migration=LAMBDA_VALUE,
        scenario_seed=scenario_seed,
    )
    summary = solve_instance(
        instance_dir=instance_dir,
        results_dir=results_dir,
        time_limit=time_limit,
        mip_gap=mip_gap,
        threads=threads,
        use_fallback=False,
    )
    visualization_outputs = create_case_plots(instance_dir, results_dir)
    return collect_case_row(group_name, case_config, case_name, summary, visualization_outputs)


def main():
    args = parse_args()
    source_csv = args.source_csv.resolve()

    RUNS_GROUP_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    case_seed_map = {}
    case_order = 0
    for group_name, case_list in CASE_GROUPS.items():
        for case_config in case_list:
            case_seed_map[(group_name, case_config["case"])] = args.seed + case_order
            case_order += 1

    tasks = []
    for group_name, case_list in CASE_GROUPS.items():
        for case_config in case_list:
            case_seed = case_seed_map[(group_name, case_config["case"])]
            tasks.append(
                (
                    group_name,
                    case_config,
                    case_seed,
                    str(source_csv),
                    args.scenario_seed,
                    args.scenario_count,
                    args.time_limit,
                    args.mip_gap,
                    args.threads,
                )
            )

    comparison_rows = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(run_single_case, task): (
                task[0],
                task[1]["label"],
            )
            for task in tasks
        }
        for future in as_completed(futures):
            group_name, case_label = futures[future]
            case_row = future.result()
            comparison_rows.append(case_row)
            print(
                f"[done] {group_name} / {case_label}: "
                f"status={case_row['status_name']}, "
                f"servers={case_row['used_server_count']}, "
                f"actual_migrations={case_row['actual_migration_event_count']}"
            )

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        ["group", "case"], ascending=[True, True]
    ).reset_index(drop=True)

    summary_csv = ANALYSIS_DIR / "all_jobs_summary.csv"
    comparison_df.to_csv(summary_csv, index=False)

    for group_name in CASE_GROUPS:
        subset = comparison_df.loc[comparison_df["group"] == group_name].copy()
        if subset.empty:
            continue
        output_png = ANALYSIS_DIR / f"{group_name}_{lambda_tag(LAMBDA_VALUE)}.png"
        create_ratio_comparison_figure(subset, output_png)

    print(f"saved: {summary_csv}")


if __name__ == "__main__":
    main()
