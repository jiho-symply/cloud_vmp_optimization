import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
ANALYSIS_DIR = EXPERIMENT_DIR / "results" / "analysis" / "chance_2sp_toy_24vm_cap8_avg20_lambda_queue"

LAMBDA_VALUES = [0.1, 0.0]
CASE_GROUPS = {
    "ratio": [
        {"case": "balanced", "label": "balanced", "on_demand": 8, "spot": 8, "batch": 8},
        {"case": "service_heavy", "label": "service-heavy", "on_demand": 14, "spot": 5, "batch": 5},
        {"case": "spot_heavy", "label": "spot-heavy", "on_demand": 5, "spot": 14, "batch": 5},
        {"case": "batch_heavy", "label": "batch-heavy", "on_demand": 5, "spot": 5, "batch": 14},
    ],
    "combination": [
        {"case": "od_only", "label": "OD only", "on_demand": 24, "spot": 0, "batch": 0},
        {"case": "od_sp", "label": "OD + SP", "on_demand": 12, "spot": 12, "batch": 0},
        {"case": "od_bj", "label": "OD + BJ", "on_demand": 12, "spot": 0, "batch": 12},
        {"case": "od_sp_bj", "label": "OD + SP + BJ", "on_demand": 8, "spot": 8, "batch": 8},
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="24-VM / cap8 / avg_cpu>=20 실험을 큐 방식으로 실행합니다.")
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


def build_case_name(group_name, case_config, scenario_count, lambda_value):
    total_vm = case_config["on_demand"] + case_config["spot"] + case_config["batch"]
    return (
        f"chance_2sp_toy_{total_vm}vm_{group_name}_{case_config['case']}_"
        f"od{case_config['on_demand']}_sp{case_config['spot']}_bj{case_config['batch']}_"
        f"sc{scenario_count}_cap8_avg20_{lambda_tag(lambda_value)}"
    )


def read_csv_safe(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def collect_case_row(group_name, case_config, case_name, lambda_value, summary, visualization_outputs):
    migration_events = read_csv_safe(visualization_outputs["migration_events_csv"])
    spot_suspensions = read_csv_safe(visualization_outputs["spot_suspension_events_csv"])
    total_vm = case_config["on_demand"] + case_config["spot"] + case_config["batch"]

    return {
        "group": group_name,
        "case": case_config["label"],
        "instance_name": case_name,
        "lambda_migration": lambda_value,
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
        lambda_value,
    ) = task

    source_csv = Path(source_csv)
    case_name = build_case_name(group_name, case_config, scenario_count, lambda_value)
    instance_dir = DATA_ROOT / case_name
    results_dir = RESULTS_ROOT / case_name

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
        lambda_migration=lambda_value,
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
    return collect_case_row(group_name, case_config, case_name, lambda_value, summary, visualization_outputs)


def create_lambda_migration_figure(comparison_df, output_path):
    plot_df = comparison_df.copy()
    plot_df["job_label"] = plot_df["group"] + " | " + plot_df["case"]
    labels = plot_df["job_label"].drop_duplicates().tolist()
    lambda_values = sorted(plot_df["lambda_migration"].drop_duplicates().tolist(), reverse=True)
    index_map = {label: idx for idx, label in enumerate(labels)}
    x = np.arange(len(labels))
    width = 0.34

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    for offset_index, lambda_value in enumerate(lambda_values):
        subset = plot_df.loc[plot_df["lambda_migration"] == lambda_value].copy()
        subset["xpos"] = subset["job_label"].map(index_map)
        shift = (offset_index - (len(lambda_values) - 1) / 2.0) * width
        axes[0].bar(
            subset["xpos"] + shift,
            subset["actual_migration_event_count"],
            width=width,
            label=f"lambda={lambda_value:.1f}",
        )
        axes[1].bar(
            subset["xpos"] + shift,
            subset["used_server_count"],
            width=width,
            label=f"lambda={lambda_value:.1f}",
        )

    axes[0].set_title("migration penalty에 따른 실제 migration event 수 비교")
    axes[0].set_ylabel("actual migration events")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    axes[1].set_title("migration penalty에 따른 사용 서버 수 비교")
    axes[1].set_ylabel("used servers")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    source_csv = args.source_csv.resolve()

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
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
            for lambda_value in LAMBDA_VALUES:
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
                        lambda_value,
                    )
                )

    comparison_rows = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(run_single_case, task): (
                task[0],
                task[1]["label"],
                task[9],
            )
            for task in tasks
        }
        for future in as_completed(futures):
            group_name, case_label, lambda_value = futures[future]
            case_row = future.result()
            comparison_rows.append(case_row)
            print(
                f"[done] {group_name} / {case_label} / lambda={lambda_value:.1f}: "
                f"status={case_row['status_name']}, "
                f"servers={case_row['used_server_count']}, "
                f"actual_migrations={case_row['actual_migration_event_count']}"
            )

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        ["group", "case", "lambda_migration"], ascending=[True, True, False]
    ).reset_index(drop=True)

    summary_csv = ANALYSIS_DIR / "all_jobs_summary.csv"
    summary_png = ANALYSIS_DIR / "lambda_migration_comparison.png"
    comparison_df.to_csv(summary_csv, index=False)

    for group_name in CASE_GROUPS:
        for lambda_value in LAMBDA_VALUES:
            subset = comparison_df.loc[
                (comparison_df["group"] == group_name)
                & (comparison_df["lambda_migration"] == lambda_value)
            ].copy()
            if subset.empty:
                continue
            output_png = ANALYSIS_DIR / f"{group_name}_{lambda_tag(lambda_value)}.png"
            create_ratio_comparison_figure(subset, output_png)

    migration_compare = comparison_df.pivot_table(
        index=["group", "case", "on_demand", "spot", "batch"],
        columns="lambda_migration",
        values=["used_server_count", "migration_count", "actual_migration_event_count"],
        aggfunc="first",
    )
    migration_compare.columns = [
        f"{metric}_lambda_{str(lambda_value).replace('.', '_')}"
        for metric, lambda_value in migration_compare.columns
    ]
    migration_compare = migration_compare.reset_index()
    migration_compare_csv = ANALYSIS_DIR / "lambda_migration_comparison.csv"
    migration_compare.to_csv(migration_compare_csv, index=False)

    create_lambda_migration_figure(comparison_df, summary_png)

    print(f"saved: {summary_csv}")
    print(f"saved: {migration_compare_csv}")
    print(f"saved: {summary_png}")


if __name__ == "__main__":
    main()
