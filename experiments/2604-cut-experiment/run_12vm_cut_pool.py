import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from build_dataset import build_instance
from run_model import solve_instance


EXPERIMENT_DIR = Path(__file__).resolve().parent
EXPERIMENT_NAME = EXPERIMENT_DIR.name
REPO_ROOT = EXPERIMENT_DIR.parents[1]
SOURCE_CSV = REPO_ROOT / "data" / "processed" / "2601-initial-toy-model" / "sample_vm_data.csv"
DATA_ROOT = REPO_ROOT / "data" / "processed" / EXPERIMENT_NAME
RESULTS_ROOT = EXPERIMENT_DIR / "results" / "runs"
ANALYSIS_DIR = EXPERIMENT_DIR / "results" / "analysis" / "cut_12vm_od_sp_bj_cap8_avg20_energy_pool"
RUNS_GROUP_DIR = RESULTS_ROOT / ANALYSIS_DIR.name

ENERGY_IDLE = 100.0
ENERGY_CPU = 300.0
ENERGY_MIGRATION = 50.0
CASE_CONFIG = {"case": "od_sp_bj", "label": "OD + SP + BJ", "on_demand": 4, "spot": 4, "batch": 4}
PROFILE_ORDER = [
    "baseline",
    "activation",
    "spot_server_link",
    "spot_time_link",
    "pairwise_cover",
    "triple_cover",
    "uptime_symmetry",
    "solver_cover_focus",
    "solver_clique_focus",
    "solver_implied_focus",
    "solver_lift_focus",
    "builtin_aggressive",
    "combined_light",
    "combined_full",
]


def parse_args():
    parser = argparse.ArgumentParser(description="12VM OD+SP+BJ cut 실험을 병렬 pool로 실행합니다.")
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


def build_case_name(scenario_count):
    total_vm = CASE_CONFIG["on_demand"] + CASE_CONFIG["spot"] + CASE_CONFIG["batch"]
    return (
        f"chance_2sp_toy_{total_vm}vm_combination_{CASE_CONFIG['case']}_"
        f"od{CASE_CONFIG['on_demand']}_sp{CASE_CONFIG['spot']}_bj{CASE_CONFIG['batch']}_"
        f"sc{scenario_count}_cap8_avg20_objenergy"
    )


def ensure_instance(source_csv, seed, scenario_seed, scenario_count):
    case_name = build_case_name(scenario_count)
    instance_dir = DATA_ROOT / case_name
    build_instance(
        source_csv=source_csv,
        output_dir=instance_dir,
        instance_name=case_name,
        seed=seed,
        scenario_count=scenario_count,
        on_demand_count=CASE_CONFIG["on_demand"],
        spot_count=CASE_CONFIG["spot"],
        batch_count=CASE_CONFIG["batch"],
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
    return case_name, instance_dir


def run_single_job(task):
    case_name, instance_dir, profile_name, time_limit, mip_gap, threads = task
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
        "case": CASE_CONFIG["label"],
        "case_key": CASE_CONFIG["case"],
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


def create_runtime_bar(summary_df, output_path):
    plot_df = summary_df.set_index("profile").reindex(PROFILE_ORDER).reset_index()
    fig, ax = plt.subplots(figsize=(13, 6), facecolor="#F7F5F1")
    ax.bar(plot_df["profile"], plot_df["runtime_seconds"], color="#D9822B", edgecolor="#1F1F1F", linewidth=0.9)
    ax.set_title("Runtime by cut profile", loc="left", pad=16)
    ax.set_ylabel("runtime (s)")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.25)
    for row in plot_df.itertuples(index=False):
        if pd.notna(row.runtime_seconds):
            ax.text(row.profile, row.runtime_seconds, f"{row.runtime_seconds:.0f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_gap_bar(summary_df, output_path):
    plot_df = summary_df.set_index("profile").reindex(PROFILE_ORDER).reset_index()
    plot_df["mip_gap_pct"] = plot_df["mip_gap"].fillna(0.0) * 100.0
    fig, ax = plt.subplots(figsize=(13, 6), facecolor="#F7F5F1")
    ax.bar(plot_df["profile"], plot_df["mip_gap_pct"], color="#5C8D89", edgecolor="#1F1F1F", linewidth=0.9)
    ax.set_title("Final MIP gap by cut profile", loc="left", pad=16)
    ax.set_ylabel("MIP gap (%)")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    source_csv = args.source_csv.resolve()

    RUNS_GROUP_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    case_name, instance_dir = ensure_instance(
        source_csv=source_csv,
        seed=args.seed,
        scenario_seed=args.scenario_seed,
        scenario_count=args.scenario_count,
    )

    tasks = [
        (case_name, instance_dir, profile_name, args.time_limit, args.mip_gap, args.threads)
        for profile_name in args.profiles
    ]

    rows = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(run_single_job, task): task[2]
            for task in tasks
        }
        for future in as_completed(futures):
            profile_name = futures[future]
            row = future.result()
            rows.append(row)
            print(
                f"[done] {profile_name}: "
                f"status={row['status_name']}, runtime={row['runtime_seconds']:.1f}s, gap={0.0 if row['mip_gap'] is None else row['mip_gap'] * 100.0:.2f}%"
            )

    summary_df = pd.DataFrame(rows).sort_values(["profile"]).reset_index(drop=True)
    summary_csv = ANALYSIS_DIR / "cut_experiment_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    runtime_png = ANALYSIS_DIR / "runtime_bar.png"
    gap_png = ANALYSIS_DIR / "gap_bar.png"
    create_runtime_bar(summary_df, runtime_png)
    create_gap_bar(summary_df, gap_png)

    print(f"saved: {summary_csv}")
    print(f"saved: {runtime_png}")
    print(f"saved: {gap_png}")


if __name__ == "__main__":
    main()
