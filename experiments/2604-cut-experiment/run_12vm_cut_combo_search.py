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
ANALYSIS_DIR = EXPERIMENT_DIR / "results" / "analysis" / "cut_12vm_od_sp_bj_cap8_avg20_combo_search"
RUNS_GROUP_DIR = EXPERIMENT_DIR / "results" / "runs" / ANALYSIS_DIR.name

ENERGY_IDLE = 100.0
ENERGY_CPU = 300.0
ENERGY_MIGRATION = 50.0
CASE_CONFIG = {"case": "od_sp_bj", "label": "OD + SP + BJ", "on_demand": 4, "spot": 4, "batch": 4}

SCREEN_PROFILE_ORDER = [
    "baseline",
    "strategy_barrier_nocrossover",
    "uptime_symmetry",
    "delta_gamma_link_only",
    "state_link",
    "eta_aggregate_load",
    "spot_bridge_lower",
    "spot_bridge_aggregate",
    "eta_cover_general",
    "combo_barrier_uptime",
    "combo_barrier_delta_gamma",
    "combo_barrier_state_link",
    "combo_barrier_eta_aggregate",
    "combo_barrier_spot_bridge_lower",
    "combo_barrier_eta_cover_general",
]


def parse_args():
    parser = argparse.ArgumentParser(description="12VM cut+root-strategy combination screening")
    parser.add_argument("--source-csv", type=Path, default=SOURCE_CSV)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario-seed", type=int, default=42)
    parser.add_argument("--scenario-count", type=int, default=10)
    parser.add_argument("--screen-time-limit", type=int, default=600)
    parser.add_argument("--final-time-limit", type=int, default=1200)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--finalists", type=int, default=4)
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
    stage_name, case_name, instance_dir, profile_name, time_limit, mip_gap, threads = task
    results_dir = RUNS_GROUP_DIR / stage_name / case_name / profile_name
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
        "stage": stage_name,
        "case": CASE_CONFIG["label"],
        "case_key": CASE_CONFIG["case"],
        "instance_name": case_name,
        "profile": profile_name,
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
        "selected_solution_source": summary.get("selected_solution_source"),
        "solver_cut_params": json.dumps(summary.get("solver_cut_params", {}), ensure_ascii=False, sort_keys=True),
        "cut_counts": json.dumps(summary.get("cut_counts", {}), ensure_ascii=False, sort_keys=True),
        "results_dir": str(results_dir),
    }


def run_stage(stage_name, case_name, instance_dir, profiles, time_limit, mip_gap, max_workers, threads):
    tasks = [
        (stage_name, case_name, instance_dir, profile_name, time_limit, mip_gap, threads)
        for profile_name in profiles
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_job, task): task[3] for task in tasks}
        for future in as_completed(futures):
            profile_name = futures[future]
            row = future.result()
            rows.append(row)
            gap_value = row["mip_gap"] * 100.0 if row["mip_gap"] is not None else float("nan")
            print(
                f"[{stage_name}] {profile_name}: "
                f"status={row['status_name']}, runtime={row['runtime_seconds']:.1f}s, gap={gap_value:.2f}%"
            )
    return pd.DataFrame(rows).sort_values(["profile"]).reset_index(drop=True)


def add_relative_metrics(summary_df):
    summary_df = summary_df.copy()
    for column in ["objective_value", "objective_bound", "mip_gap", "runtime_seconds"]:
        summary_df[column] = pd.to_numeric(summary_df[column], errors="coerce")
    baseline = summary_df.loc[summary_df["profile"] == "baseline"].iloc[0]
    baseline_obj = float(baseline["objective_value"])
    baseline_bound = float(baseline["objective_bound"])
    baseline_gap = float(baseline["mip_gap"])
    summary_df["incumbent_improve_pct"] = (
        (baseline_obj - summary_df["objective_value"]) / baseline_obj * 100.0
    )
    summary_df["bound_improve_pct"] = (
        (summary_df["objective_bound"] - baseline_bound) / baseline_bound * 100.0
    )
    summary_df["gap_improve_pct"] = (
        (baseline_gap - summary_df["mip_gap"]) / baseline_gap * 100.0
    )
    summary_df["screen_score"] = (
        0.55 * summary_df["gap_improve_pct"]
        + 0.25 * summary_df["incumbent_improve_pct"]
        + 0.20 * summary_df["bound_improve_pct"]
    )
    return summary_df


def choose_finalists(screen_df, finalist_count):
    ranked = screen_df.loc[(screen_df["profile"] != "baseline") & screen_df["has_solution"]].copy()
    ranked = ranked.sort_values(
        ["screen_score", "mip_gap", "objective_value"],
        ascending=[False, True, True],
    )
    finalists = ["baseline"] + ranked["profile"].head(finalist_count).tolist()
    return finalists, ranked


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


def create_metric_bar(summary_df, output_path, metric, ylabel, color):
    fig, ax = plt.subplots(figsize=(14, 6), facecolor="#F7F5F1")
    ax.bar(summary_df["profile"], summary_df[metric], color=color, edgecolor="#1F1F1F", linewidth=0.9)
    ax.set_title(output_path.stem.replace("_", " "), loc="left", pad=18)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_markdown_report(screen_df, final_df, finalists, output_path, screen_time_limit, final_time_limit):
    lines = [
        "# 12VM Cut + Root Strategy Search",
        "",
        "## 실험 설계",
        "",
        "- 고정 인스턴스: `12VM`, `OD + SP + BJ = 4 + 4 + 4`, `cap=8`, `avg_cpu_mean >= 20`, `scenario=10`",
        "- 목적함수: 에너지 최소화",
        f"- 1차 screening: 각 단일 컷 또는 cut+root 전략 조합을 `{screen_time_limit}초`로 비교",
        f"- 2차 재평가: screening 상위 후보와 `baseline`을 `{final_time_limit}초`로 재평가",
        "",
        "## 최종 재평가 대상",
        "",
    ]
    for profile in finalists:
        lines.append(f"- `{profile}`")
    lines += [
        "",
        "## Screening 상위 결과",
        "",
        frame_to_markdown(
            screen_df[[
                "profile",
                "objective_value",
                "objective_bound",
                "mip_gap",
                "incumbent_improve_pct",
                "bound_improve_pct",
                "gap_improve_pct",
                "screen_score",
            ]].head(10)
        ),
        "",
        "## Final 결과",
        "",
        frame_to_markdown(
            final_df[[
                "profile",
                "objective_value",
                "objective_bound",
                "mip_gap",
                "incumbent_improve_pct",
                "bound_improve_pct",
                "gap_improve_pct",
            ]].sort_values(["mip_gap", "objective_value"])
        ),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


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

    screen_df = run_stage(
        stage_name="screen",
        case_name=case_name,
        instance_dir=instance_dir,
        profiles=SCREEN_PROFILE_ORDER,
        time_limit=args.screen_time_limit,
        mip_gap=args.mip_gap,
        max_workers=args.max_workers,
        threads=args.threads,
    )
    screen_df = add_relative_metrics(screen_df)
    screen_csv = ANALYSIS_DIR / "screen_summary.csv"
    screen_df.to_csv(screen_csv, index=False)

    finalists, ranked_screen = choose_finalists(screen_df, args.finalists)
    ranked_csv = ANALYSIS_DIR / "screen_ranking.csv"
    ranked_screen.to_csv(ranked_csv, index=False)

    final_df = run_stage(
        stage_name="final",
        case_name=case_name,
        instance_dir=instance_dir,
        profiles=finalists,
        time_limit=args.final_time_limit,
        mip_gap=args.mip_gap,
        max_workers=args.max_workers,
        threads=args.threads,
    )
    final_df = add_relative_metrics(final_df)
    final_csv = ANALYSIS_DIR / "final_summary.csv"
    final_df.to_csv(final_csv, index=False)

    create_metric_bar(
        summary_df=ranked_screen.head(12),
        output_path=ANALYSIS_DIR / "screen_score_bar.png",
        metric="screen_score",
        ylabel="screen score",
        color="#B36A3C",
    )
    final_plot_df = final_df.sort_values(["mip_gap", "objective_value"]).copy()
    final_plot_df["mip_gap_pct"] = final_plot_df["mip_gap"] * 100.0
    create_metric_bar(
        summary_df=final_plot_df,
        output_path=ANALYSIS_DIR / "final_gap_bar.png",
        metric="mip_gap_pct",
        ylabel="MIP gap (%)",
        color="#4E7B86",
    )

    report_path = ANALYSIS_DIR / "SEARCH_REPORT.md"
    write_markdown_report(
        screen_df=ranked_screen,
        final_df=final_df,
        finalists=finalists,
        output_path=report_path,
        screen_time_limit=args.screen_time_limit,
        final_time_limit=args.final_time_limit,
    )

    print(f"saved: {screen_csv}")
    print(f"saved: {ranked_csv}")
    print(f"saved: {final_csv}")
    print(f"saved: {report_path}")


if __name__ == "__main__":
    main()
