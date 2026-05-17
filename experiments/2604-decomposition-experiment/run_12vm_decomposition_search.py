import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
EXPERIMENT_NAME = EXPERIMENT_DIR.name
REPO_ROOT = EXPERIMENT_DIR.parents[1]
CUT_EXPERIMENT_DIR = REPO_ROOT / "experiments" / "2604-cut-experiment"
if str(CUT_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(CUT_EXPERIMENT_DIR))

from build_dataset import build_instance  # noqa: E402
from run_model import load_instance_data, solve_instance  # noqa: E402

from decomposition_starts import build_start_strategy  # noqa: E402


SOURCE_CSV = REPO_ROOT / "data" / "processed" / "2601-initial-toy-model" / "sample_vm_data.csv"
DATA_ROOT = REPO_ROOT / "data" / "processed" / EXPERIMENT_NAME
ANALYSIS_DIR = EXPERIMENT_DIR / "results" / "analysis" / "chance_2sp_toy_12vm_decomposition_search"
RUNS_GROUP_DIR = EXPERIMENT_DIR / "results" / "runs" / ANALYSIS_DIR.name

ENERGY_IDLE = 100.0
ENERGY_CPU = 300.0
ENERGY_MIGRATION = 50.0

CASE_CONFIG = {
    "case": "od_sp_bj",
    "label": "OD + SP + BJ",
    "on_demand": 4,
    "spot": 4,
    "batch": 4,
}

PROFILE_CONFIGS = [
    {"name": "ref_combo_phi_mass", "cut_profile": "combo_barrier_state_uptime_phi_mass", "start_strategy": None},
    {"name": "ref_state_link", "cut_profile": "state_link", "start_strategy": None},
    {"name": "rep_mean_combo_phi_mass", "cut_profile": "combo_barrier_state_uptime_phi_mass", "start_strategy": "representative_mean"},
    {"name": "rep_peak_combo_phi_mass", "cut_profile": "combo_barrier_state_uptime_phi_mass", "start_strategy": "representative_peak"},
    {"name": "scenario_best_combo_phi_mass", "cut_profile": "combo_barrier_state_uptime_phi_mass", "start_strategy": "single_scenario_best"},
    {"name": "scenario_consensus_combo_phi_mass", "cut_profile": "combo_barrier_state_uptime_phi_mass", "start_strategy": "scenario_consensus"},
    {"name": "progressive_mean_combo_phi_mass", "cut_profile": "combo_barrier_state_uptime_phi_mass", "start_strategy": "progressive_mean"},
    {"name": "greedy_rule_combo_phi_mass", "cut_profile": "combo_barrier_state_uptime_phi_mass", "start_strategy": "greedy_rule_mean"},
]


def parse_args():
    parser = argparse.ArgumentParser(description="12VM decomposition-style warm-start experiment")
    parser.add_argument("--source-csv", type=Path, default=SOURCE_CSV)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario-seed", type=int, default=42)
    parser.add_argument("--scenario-count", type=int, default=10)
    parser.add_argument("--time-limit", type=int, default=900)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--strategy-time-limit", type=int, default=120)
    parser.add_argument("--strategy-threads", type=int, default=2)
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


def precompute_starts(data, strategy_time_limit, strategy_threads):
    start_cache = {None: {"start_values": None, "metadata": {"strategy": None}}}
    starts_dir = ANALYSIS_DIR / "starts"
    starts_dir.mkdir(parents=True, exist_ok=True)

    for strategy_name in sorted({profile["start_strategy"] for profile in PROFILE_CONFIGS if profile["start_strategy"]}):
        start_values, metadata = build_start_strategy(
            strategy_name=strategy_name,
            data=data,
            results_dir=starts_dir,
            cut_profile="combo_barrier_state_uptime_phi_mass",
            strategy_time_limit=strategy_time_limit,
            strategy_threads=strategy_threads,
        )
        start_cache[strategy_name] = {
            "start_values": start_values,
            "metadata": metadata,
        }

    metadata_rows = []
    for strategy_name, payload in start_cache.items():
        metadata_rows.append(
            {
                "start_strategy": strategy_name or "none",
                "has_start": payload["start_values"] is not None,
                "metadata": json.dumps(payload["metadata"], ensure_ascii=False),
            }
        )
    pd.DataFrame(metadata_rows).to_csv(ANALYSIS_DIR / "start_generation_summary.csv", index=False)
    return start_cache


def run_single_job(task):
    case_name, instance_dir, profile_config, time_limit, mip_gap, threads, start_payload = task
    profile_name = profile_config["name"]
    results_dir = RUNS_GROUP_DIR / case_name / profile_name
    summary = solve_instance(
        instance_dir=instance_dir,
        results_dir=results_dir,
        time_limit=time_limit,
        mip_gap=mip_gap,
        threads=threads,
        norel_pre_time=0.0,
        use_fallback=False,
        cut_profile=profile_config["cut_profile"],
        start_values=start_payload["start_values"],
    )
    return {
        "profile": profile_name,
        "cut_profile": profile_config["cut_profile"],
        "start_strategy": profile_config["start_strategy"] or "none",
        "start_available": start_payload["start_values"] is not None,
        "start_metadata": json.dumps(start_payload["metadata"], ensure_ascii=False),
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
    }


def frame_to_markdown(frame):
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in frame.itertuples(index=False):
        values = []
        for value in row:
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(summary_df):
    ordered = summary_df.sort_values(["mip_gap", "objective_value"]).reset_index(drop=True)
    lines = [
        "# 12VM Decomposition Strategy Review",
        "",
        "## 실험 설정",
        "",
        "- 인스턴스: `12VM`, `OD + SP + BJ = 4 + 4 + 4`, `cap=8`, `avg_cpu_mean >= 20`, `scenario=10`",
        "- 목적함수: 에너지 최소화",
        "- full MIP 시간 제한: `900초`",
        "- 비교 기준: 기존 우수 프로파일(`combo_barrier_state_uptime_phi_mass`, `state_link`)과 decomposition-style warm start 전략",
        "",
        "## 결과 요약",
        "",
        frame_to_markdown(
            ordered[
                [
                    "profile",
                    "cut_profile",
                    "start_strategy",
                    "objective_value",
                    "objective_bound",
                    "mip_gap",
                    "used_server_count",
                    "node_count",
                ]
            ]
        ),
        "",
    ]
    (ANALYSIS_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_GROUP_DIR.mkdir(parents=True, exist_ok=True)

    case_name, instance_dir = ensure_instance(
        source_csv=args.source_csv.resolve(),
        seed=args.seed,
        scenario_seed=args.scenario_seed,
        scenario_count=args.scenario_count,
    )
    data = load_instance_data(instance_dir)
    start_cache = precompute_starts(
        data=data,
        strategy_time_limit=args.strategy_time_limit,
        strategy_threads=args.strategy_threads,
    )

    tasks = [
        (
            case_name,
            instance_dir,
            profile_config,
            args.time_limit,
            args.mip_gap,
            args.threads,
            start_cache[profile_config["start_strategy"]],
        )
        for profile_config in PROFILE_CONFIGS
    ]

    rows = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(run_single_job, task): task[2]["name"] for task in tasks}
        for future in as_completed(futures):
            profile_name = futures[future]
            row = future.result()
            rows.append(row)
            gap_value = row["mip_gap"] * 100.0 if row["mip_gap"] is not None else float("nan")
            print(
                f"{profile_name}: status={row['status_name']}, runtime={row['runtime_seconds']:.1f}s, gap={gap_value:.2f}%"
            )

    summary_df = pd.DataFrame(rows).sort_values(["profile"]).reset_index(drop=True)
    summary_df.to_csv(ANALYSIS_DIR / "summary.csv", index=False)
    write_report(summary_df)
    print(f"analysis_dir={ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
