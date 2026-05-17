import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from build_dataset import build_instance
from run_model import solve_instance
from run_12vm_energy_queue import collect_case_row, create_energy_breakdown_figure
from visualize import create_case_plots, create_ratio_comparison_figure


EXPERIMENT_DIR = Path(__file__).resolve().parent
EXPERIMENT_NAME = EXPERIMENT_DIR.name
REPO_ROOT = EXPERIMENT_DIR.parents[1]
SOURCE_CSV = REPO_ROOT / "data" / "processed" / "2601-initial-toy-model" / "sample_vm_data.csv"
DATA_ROOT = REPO_ROOT / "data" / "processed" / EXPERIMENT_NAME
RESULTS_ROOT = EXPERIMENT_DIR / "results" / "runs"
ANALYSIS_DIR = EXPERIMENT_DIR / "results" / "analysis" / "chance_2sp_toy_6vm_cap8_avg20_energy_queue"
RUNS_GROUP_DIR = RESULTS_ROOT / ANALYSIS_DIR.name

OBJECTIVE_TYPE = "energy"
ENERGY_IDLE = 100.0
ENERGY_CPU = 300.0
ENERGY_MIGRATION = 50.0
CASE_GROUPS = {
    "combination": [
        {"case": "od_only", "label": "OD only", "on_demand": 6, "spot": 0, "batch": 0},
        {"case": "od_sp", "label": "OD + SP", "on_demand": 3, "spot": 3, "batch": 0},
        {"case": "od_bj", "label": "OD + BJ", "on_demand": 3, "spot": 0, "batch": 3},
        {"case": "od_sp_bj", "label": "OD + SP + BJ", "on_demand": 2, "spot": 2, "batch": 2},
    ]
}


def parse_args():
    parser = argparse.ArgumentParser(description="6-VM / cap8 / avg_cpu>=20 에너지 최소화 조합 벤치마크를 큐 방식으로 실행합니다.")
    parser.add_argument("--source-csv", type=Path, default=SOURCE_CSV)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario-seed", type=int, default=42)
    parser.add_argument("--scenario-count", type=int, default=10)
    parser.add_argument("--time-limit", type=int, default=0)
    parser.add_argument("--norel-pre-time", type=float, default=600.0)
    parser.add_argument("--alternate-norel-time", type=float, default=0.0)
    parser.add_argument("--alternate-main-time", type=float, default=0.0)
    parser.add_argument("--alternate-max-rounds", type=int, default=6)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    return parser.parse_args()


def build_case_name(case_config, scenario_count):
    total_vm = case_config["on_demand"] + case_config["spot"] + case_config["batch"]
    return (
        f"chance_2sp_toy_{total_vm}vm_combination_{case_config['case']}_"
        f"od{case_config['on_demand']}_sp{case_config['spot']}_bj{case_config['batch']}_"
        f"sc{scenario_count}_cap8_avg20_objenergy"
    )


def run_single_case(task):
    (
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
    case_name = build_case_name(case_config, scenario_count)
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

    return collect_case_row("combination", case_config, case_name, summary, visualization_outputs)


def main():
    args = parse_args()
    source_csv = args.source_csv.resolve()

    RUNS_GROUP_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    tasks = []
    for case_index, case_config in enumerate(CASE_GROUPS["combination"]):
        tasks.append(
            (
                case_config,
                args.seed + case_index,
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
            executor.submit(run_single_case, task): task[0]["label"]
            for task in tasks
        }
        for future in as_completed(futures):
            case_label = futures[future]
            case_row = future.result()
            comparison_rows.append(case_row)
            print(
                f"[done] {case_label}: "
                f"status={case_row['status_name']}, "
                f"energy={case_row['total_energy']:.3f}, "
                f"servers={case_row['used_server_count']}"
            )

    comparison_df = pd.DataFrame(comparison_rows).sort_values(["group", "case"]).reset_index(drop=True)
    summary_csv = ANALYSIS_DIR / "all_jobs_summary.csv"
    comparison_df.to_csv(summary_csv, index=False)

    comparison_png = ANALYSIS_DIR / "combination_energy.png"
    create_ratio_comparison_figure(comparison_df, comparison_png)

    energy_png = ANALYSIS_DIR / "energy_breakdown.png"
    create_energy_breakdown_figure(comparison_df, energy_png)

    print(f"saved: {summary_csv}")
    print(f"saved: {comparison_png}")
    print(f"saved: {energy_png}")


if __name__ == "__main__":
    main()
