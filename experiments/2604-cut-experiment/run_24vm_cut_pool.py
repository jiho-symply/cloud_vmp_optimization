import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from build_dataset import build_instance
from cut_profiles import CUT_PROFILES
from run_model import solve_instance


EXPERIMENT_DIR = Path(__file__).resolve().parent
EXPERIMENT_NAME = EXPERIMENT_DIR.name
REPO_ROOT = EXPERIMENT_DIR.parents[1]
SOURCE_CSV = REPO_ROOT / "data" / "processed" / "2601-initial-toy-model" / "sample_vm_data.csv"
DATA_ROOT = REPO_ROOT / "data" / "processed" / EXPERIMENT_NAME
RESULTS_ROOT = EXPERIMENT_DIR / "results" / "runs"
ANALYSIS_DIR = EXPERIMENT_DIR / "results" / "analysis" / "cut_24vm_cap8_avg20_energy_pool"
RUNS_GROUP_DIR = RESULTS_ROOT / ANALYSIS_DIR.name

ENERGY_IDLE = 100.0
ENERGY_CPU = 300.0
ENERGY_MIGRATION = 50.0
CASE_GROUPS = {
    "combination": [
        {"case": "od_only", "label": "OD only", "on_demand": 24, "spot": 0, "batch": 0},
        {"case": "od_sp", "label": "OD + SP", "on_demand": 12, "spot": 12, "batch": 0},
        {"case": "od_bj", "label": "OD + BJ", "on_demand": 12, "spot": 0, "batch": 12},
        {"case": "od_sp_bj", "label": "OD + SP + BJ", "on_demand": 8, "spot": 8, "batch": 8},
    ],
}
PROFILE_ORDER = [
    "baseline",
    "activation",
    "spot_link",
    "pairwise_cover",
    "triple_cover",
    "uptime_symmetry",
    "builtin_aggressive",
    "combined_light",
    "combined_full",
]


def parse_args():
    parser = argparse.ArgumentParser(description="24VM cap8 avg20 cut 실험을 병렬 pool로 실행합니다.")
    parser.add_argument("--source-csv", type=Path, default=SOURCE_CSV)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario-seed", type=int, default=42)
    parser.add_argument("--scenario-count", type=int, default=10)
    parser.add_argument("--time-limit", type=int, default=3600)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--profiles", nargs="+", choices=PROFILE_ORDER, default=PROFILE_ORDER)
    return parser.parse_args()


def build_case_name(case_config, scenario_count):
    total_vm = case_config["on_demand"] + case_config["spot"] + case_config["batch"]
    return (
        f"chance_2sp_toy_{total_vm}vm_combination_{case_config['case']}_"
        f"od{case_config['on_demand']}_sp{case_config['spot']}_bj{case_config['batch']}_"
        f"sc{scenario_count}_cap8_avg20_objenergy"
    )


def ensure_instances(source_csv, seed, scenario_seed, scenario_count):
    instance_records = []
    for case_index, case_config in enumerate(CASE_GROUPS["combination"]):
        case_name = build_case_name(case_config, scenario_count)
        instance_dir = DATA_ROOT / case_name
        build_instance(
            source_csv=source_csv,
            output_dir=instance_dir,
            instance_name=case_name,
            seed=seed + case_index,
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
            objective_type="energy",
            lambda_migration=0.0,
            energy_idle=ENERGY_IDLE,
            energy_cpu=ENERGY_CPU,
            energy_migration=ENERGY_MIGRATION,
            scenario_seed=scenario_seed,
        )
        instance_records.append((case_config, case_name, instance_dir))
    return instance_records


def run_single_job(task):
    case_config, case_name, instance_dir, profile_name, time_limit, mip_gap, threads = task
    results_dir = RUNS_GROUP_DIR / case_name / profile_name
    summary = solve_instance(
        instance_dir=instance_dir,
        results_dir=results_dir,
        time_limit=time_limit,
        mip_gap=mip_gap,
        threads=threads,
        norel_pre_time=0.0,
        use_fallback=False,
        cut_profile=profile_name,
    )
    return {
        "case": case_config["label"],
        "case_key": case_config["case"],
        "instance_name": case_name,
        "profile": profile_name,
        "status_name": summary.get("status_name"),
        "runtime_seconds": summary.get("runtime_seconds"),
        "objective_value": summary.get("objective_value"),
        "objective_bound": summary.get("objective_bound"),
        "mip_gap": summary.get("mip_gap"),
        "node_count": summary.get("node_count"),
        "iter_count": summary.get("iter_count"),
        "work_units": summary.get("work_units"),
        "used_server_count": summary.get("used_server_count"),
        "selected_solution_source": summary.get("selected_solution_source"),
        "cut_profile_category": summary.get("cut_profile_category"),
        "cut_profile_description": summary.get("cut_profile_description"),
        "solver_cut_params": json.dumps(summary.get("solver_cut_params", {}), ensure_ascii=False, sort_keys=True),
        "cut_counts": json.dumps(summary.get("cut_counts", {}), ensure_ascii=False, sort_keys=True),
        "results_dir": str(results_dir),
    }


def create_runtime_figure(summary_df, output_path):
    pivot = summary_df.pivot(index="profile", columns="case", values="runtime_seconds").reindex(PROFILE_ORDER)
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="#F7F5F1")
    image = ax.imshow(pivot.fillna(0.0).to_numpy(), aspect="auto", cmap="YlOrBr")
    ax.set_title("Runtime by cut profile", loc="left", pad=16)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for row_idx, profile_name in enumerate(pivot.index):
        for col_idx, case_name in enumerate(pivot.columns):
            value = pivot.iloc[row_idx, col_idx]
            if pd.notna(value):
                ax.text(col_idx, row_idx, f"{value:.0f}", ha="center", va="center", fontsize=8, color="#1F1F1F")
    fig.colorbar(image, ax=ax, label="runtime (s)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_gap_figure(summary_df, output_path):
    plot_df = summary_df.copy()
    plot_df["mip_gap_pct"] = plot_df["mip_gap"].fillna(0.0) * 100.0
    fig, ax = plt.subplots(figsize=(13, 6), facecolor="#F7F5F1")
    for case_name, subset in plot_df.groupby("case"):
        subset = subset.set_index("profile").reindex(PROFILE_ORDER).reset_index()
        ax.plot(subset["profile"], subset["mip_gap_pct"], marker="o", linewidth=2.0, label=case_name)
    ax.set_title("Final MIP gap by cut profile", loc="left", pad=16)
    ax.set_ylabel("MIP gap (%)")
    ax.set_xlabel("cut profile")
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=25)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    source_csv = args.source_csv.resolve()

    RUNS_GROUP_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    instance_records = ensure_instances(
        source_csv=source_csv,
        seed=args.seed,
        scenario_seed=args.scenario_seed,
        scenario_count=args.scenario_count,
    )

    tasks = []
    for case_config, case_name, instance_dir in instance_records:
        for profile_name in args.profiles:
            tasks.append(
                (
                    case_config,
                    case_name,
                    instance_dir,
                    profile_name,
                    args.time_limit,
                    args.mip_gap,
                    args.threads,
                )
            )

    rows = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(run_single_job, task): (task[0]["label"], task[3])
            for task in tasks
        }
        for future in as_completed(futures):
            case_label, profile_name = futures[future]
            row = future.result()
            rows.append(row)
            print(
                f"[done] {case_label} / {profile_name}: "
                f"status={row['status_name']}, runtime={row['runtime_seconds']:.1f}s, gap={0.0 if row['mip_gap'] is None else row['mip_gap'] * 100.0:.2f}%"
            )

    summary_df = pd.DataFrame(rows).sort_values(["case", "profile"]).reset_index(drop=True)
    summary_csv = ANALYSIS_DIR / "cut_experiment_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    runtime_png = ANALYSIS_DIR / "runtime_heatmap.png"
    gap_png = ANALYSIS_DIR / "gap_profile.png"
    create_runtime_figure(summary_df, runtime_png)
    create_gap_figure(summary_df, gap_png)

    print(f"saved: {summary_csv}")
    print(f"saved: {runtime_png}")
    print(f"saved: {gap_png}")


if __name__ == "__main__":
    main()
