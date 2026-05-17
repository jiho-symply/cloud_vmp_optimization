import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from build_dataset import build_instance
from run_model import solve_instance
from visualize import create_case_plots, create_ratio_comparison_figure


EXPERIMENT_DIR = Path(__file__).resolve().parent
EXPERIMENT_NAME = EXPERIMENT_DIR.name
REPO_ROOT = EXPERIMENT_DIR.parents[1]
SOURCE_CSV = REPO_ROOT / "data" / "processed" / "2601-initial-toy-model" / "sample_vm_data.csv"
DATA_ROOT = REPO_ROOT / "data" / "processed" / EXPERIMENT_NAME
RESULTS_ROOT = EXPERIMENT_DIR / "results" / "runs"
ANALYSIS_DIR = EXPERIMENT_DIR / "results" / "analysis" / "chance_2sp_toy_12vm_cap8_avg20_energy_queue"
RUNS_GROUP_DIR = RESULTS_ROOT / ANALYSIS_DIR.name

OBJECTIVE_TYPE = "energy"
ENERGY_IDLE = 100.0
ENERGY_CPU = 300.0
ENERGY_MIGRATION = 50.0
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
    parser = argparse.ArgumentParser(description="12-VM / cap8 / avg_cpu>=20 에너지 최소화 실험을 큐 방식으로 실행합니다.")
    parser.add_argument("--source-csv", type=Path, default=SOURCE_CSV)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario-seed", type=int, default=42)
    parser.add_argument("--scenario-count", type=int, default=10)
    parser.add_argument("--time-limit", type=int, default=7200)
    parser.add_argument("--norel-pre-time", type=float, default=0.0)
    parser.add_argument("--alternate-norel-time", type=float, default=0.0)
    parser.add_argument("--alternate-main-time", type=float, default=0.0)
    parser.add_argument("--alternate-max-rounds", type=int, default=6)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=sorted(CASE_GROUPS.keys()),
        default=sorted(CASE_GROUPS.keys()),
        help="실행할 케이스 그룹을 고릅니다. 예: --groups combination",
    )
    return parser.parse_args()


def build_case_name(group_name, case_config, scenario_count):
    total_vm = case_config["on_demand"] + case_config["spot"] + case_config["batch"]
    return (
        f"chance_2sp_toy_{total_vm}vm_{group_name}_{case_config['case']}_"
        f"od{case_config['on_demand']}_sp{case_config['spot']}_bj{case_config['batch']}_"
        f"sc{scenario_count}_cap8_avg20_objenergy"
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
    migration_csv = visualization_outputs.get("migration_events_csv")
    suspension_csv = visualization_outputs.get("spot_suspension_events_csv")
    migration_events = read_csv_safe(migration_csv) if migration_csv else pd.DataFrame()
    spot_suspensions = read_csv_safe(suspension_csv) if suspension_csv else pd.DataFrame()
    total_vm = case_config["on_demand"] + case_config["spot"] + case_config["batch"]

    return {
        "group": group_name,
        "case": case_config["label"],
        "instance_name": case_name,
        "objective_type": OBJECTIVE_TYPE,
        "energy_idle_param": ENERGY_IDLE,
        "energy_cpu_param": ENERGY_CPU,
        "energy_migration_param": ENERGY_MIGRATION,
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
        "solve_strategy": summary.get("solve_strategy"),
        "fallback_attempted": summary.get("fallback_attempted"),
        "fallback_selected": summary.get("fallback_selected"),
        "runtime_seconds": summary.get("runtime_seconds"),
        "total_runtime_seconds": summary.get("total_runtime_seconds"),
        "norel_phase_attempted": summary.get("norel_phase_attempted"),
        "norel_phase_status_name": summary.get("norel_phase_status_name"),
        "norel_phase_has_solution": summary.get("norel_phase_has_solution"),
        "norel_phase_runtime_seconds": summary.get("norel_phase_runtime_seconds"),
        "main_phase_runtime_seconds": summary.get("main_phase_runtime_seconds"),
        "alternate_norel_time": summary.get("alternate_norel_time"),
        "alternate_main_time": summary.get("alternate_main_time"),
        "alternate_max_rounds": summary.get("alternate_max_rounds"),
        "phase_count": summary.get("phase_count"),
        "incumbent_updates": summary.get("incumbent_updates"),
        "best_bound_seen": summary.get("best_bound_seen"),
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
        "total_energy": summary.get("total_energy"),
        "idle_energy": summary.get("idle_energy"),
        "cpu_energy": summary.get("cpu_energy"),
        "migration_energy": summary.get("migration_energy"),
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
        norel_pre_time,
        alternate_norel_time,
        alternate_main_time,
        alternate_max_rounds,
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
        objective_type=OBJECTIVE_TYPE,
        lambda_migration=0.0,
        energy_idle=ENERGY_IDLE,
        energy_cpu=ENERGY_CPU,
        energy_migration=ENERGY_MIGRATION,
        scenario_seed=scenario_seed,
    )
    summary = solve_instance(
        instance_dir=instance_dir,
        results_dir=results_dir,
        time_limit=time_limit,
        mip_gap=mip_gap,
        threads=threads,
        norel_pre_time=norel_pre_time,
        alternate_norel_time=alternate_norel_time,
        alternate_main_time=alternate_main_time,
        alternate_max_rounds=alternate_max_rounds,
        use_fallback=False,
    )
    visualization_outputs = {
        "scenario": None,
        "suspended_scenario": None,
        "clean_scenario": None,
        "spot_activity_csv": None,
        "spot_suspension_events_csv": None,
        "migration_events_csv": None,
    }
    if summary.get("has_solution"):
        visualization_outputs = create_case_plots(instance_dir, results_dir)
    return collect_case_row(group_name, case_config, case_name, summary, visualization_outputs)


def create_energy_breakdown_figure(comparison_df, output_path):
    plot_df = comparison_df.copy()
    plot_df["job_label"] = plot_df["group"] + " | " + plot_df["case"]
    plot_df = plot_df.sort_values(["group", "case"]).reset_index(drop=True)
    x = np.arange(len(plot_df))

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), facecolor="#F6F3EE")

    axes[0].bar(x, plot_df["idle_energy"], color="#9ECAE1", label="Idle energy")
    axes[0].bar(x, plot_df["cpu_energy"], bottom=plot_df["idle_energy"], color="#F28E2B", label="CPU energy")
    axes[0].bar(
        x,
        plot_df["migration_energy"],
        bottom=plot_df["idle_energy"] + plot_df["cpu_energy"],
        color="#8E6CBE",
        label="Migration energy",
    )
    axes[0].set_title("케이스별 에너지 소모 분해", loc="left")
    axes[0].set_ylabel("energy")
    axes[0].legend(frameon=True)
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(plot_df["job_label"], rotation=25, ha="right")

    axes[1].scatter(
        plot_df["peak_overbooking_ratio"] * 100.0,
        plot_df["total_energy"],
        s=120,
        c=plot_df["used_server_count"],
        cmap="viridis",
        edgecolors="white",
        linewidth=1.0,
    )
    for row in plot_df.itertuples(index=False):
        axes[1].annotate(
            row.case,
            (row.peak_overbooking_ratio * 100.0, row.total_energy),
            xytext=(7, 5),
            textcoords="offset points",
            fontsize=9,
        )
    axes[1].set_title("에너지와 overbooking 관계", loc="left")
    axes[1].set_xlabel("peak overbooking ratio (%)")
    axes[1].set_ylabel("total energy")
    axes[1].grid(True, axis="both", alpha=0.25)

    axes[1].set_xticks(np.arange(0, max(130, float(plot_df["peak_overbooking_ratio"].max() * 100.0) + 10), 10))
    axes[1].set_xlim(left=95)
    axes[1].figure.colorbar(
        plt.cm.ScalarMappable(cmap="viridis", norm=Normalize(plot_df["used_server_count"].min(), plot_df["used_server_count"].max())),
        ax=axes[1],
        label="used servers",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    source_csv = args.source_csv.resolve()
    selected_groups = {group_name: CASE_GROUPS[group_name] for group_name in args.groups}

    RUNS_GROUP_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    case_seed_map = {}
    case_order = 0
    for group_name, case_list in selected_groups.items():
        for case_config in case_list:
            case_seed_map[(group_name, case_config["case"])] = args.seed + case_order
            case_order += 1

    tasks = []
    for group_name, case_list in selected_groups.items():
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
                    args.norel_pre_time,
                    args.alternate_norel_time,
                    args.alternate_main_time,
                    args.alternate_max_rounds,
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
                f"energy={case_row['total_energy']:.3f}, "
                f"servers={case_row['used_server_count']}"
            )

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        ["group", "case"], ascending=[True, True]
    ).reset_index(drop=True)

    summary_csv = ANALYSIS_DIR / "all_jobs_summary.csv"
    comparison_df.to_csv(summary_csv, index=False)

    for group_name in selected_groups:
        subset = comparison_df.loc[comparison_df["group"] == group_name].copy()
        if subset.empty:
            continue
        output_png = ANALYSIS_DIR / f"{group_name}_energy.png"
        create_ratio_comparison_figure(subset, output_png)

    energy_png = ANALYSIS_DIR / "energy_breakdown.png"
    create_energy_breakdown_figure(comparison_df, energy_png)

    print(f"saved: {summary_csv}")
    print(f"saved: {energy_png}")


if __name__ == "__main__":
    main()
