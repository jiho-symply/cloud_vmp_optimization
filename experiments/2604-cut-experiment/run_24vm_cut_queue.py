import argparse
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
ANALYSIS_DIR = EXPERIMENT_DIR / "results" / "analysis" / "chance_2sp_toy_24vm_cap8_avg20_cut_pool"
RUNS_GROUP_DIR = RESULTS_ROOT / ANALYSIS_DIR.name

LAMBDA_VALUE = 0.1
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
MANUAL_CUT_COLUMNS = [
    "activation_upper",
    "used_upper",
    "spot_completion_server",
    "spot_delta_server",
    "pairwise_cover",
    "triple_cover",
    "uptime_symmetry",
]


def parse_args():
    parser = argparse.ArgumentParser(description="24VM cut profile 벤치마크를 큐 방식으로 실행합니다.")
    parser.add_argument("--source-csv", type=Path, default=SOURCE_CSV)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario-seed", type=int, default=42)
    parser.add_argument("--scenario-count", type=int, default=10)
    parser.add_argument("--time-limit", type=int, default=3600)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=sorted(CASE_GROUPS.keys()),
        default=sorted(CASE_GROUPS.keys()),
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=sorted(CUT_PROFILES.keys()),
        default=sorted(CUT_PROFILES.keys()),
    )
    return parser.parse_args()


def lambda_tag(lambda_value):
    return f"lam{int(round(lambda_value * 100)):03d}"


def build_instance_name(group_name, case_config, scenario_count):
    total_vm = case_config["on_demand"] + case_config["spot"] + case_config["batch"]
    return (
        f"chance_2sp_toy_{total_vm}vm_{group_name}_{case_config['case']}_"
        f"od{case_config['on_demand']}_sp{case_config['spot']}_bj{case_config['batch']}_"
        f"sc{scenario_count}_cap8_avg20_{lambda_tag(LAMBDA_VALUE)}"
    )


def build_results_name(instance_name, cut_profile):
    return f"{instance_name}_cut_{cut_profile}"


def profile_category(cut_profile):
    flags = CUT_PROFILES[cut_profile]
    categories = []
    if flags.get("activation_upper") or flags.get("spot_completion_server") or flags.get("spot_delta_server"):
        categories.append("logic")
    if flags.get("pairwise_cover") or flags.get("triple_cover"):
        categories.append("cover")
    if flags.get("uptime_symmetry"):
        categories.append("symmetry")
    if flags.get("builtin_aggressive"):
        categories.append("solver-cut")
    if not categories:
        return "baseline"
    return "+".join(categories)


def profile_description(cut_profile):
    flags = CUT_PROFILES[cut_profile]
    labels = []
    if flags.get("activation_upper"):
        labels.append("u/u_used activation upper bound")
    if flags.get("spot_completion_server"):
        labels.append("server-conditioned spot completion")
    if flags.get("spot_delta_server"):
        labels.append("delta-server linking")
    if flags.get("pairwise_cover"):
        labels.append("pairwise cover cut")
    if flags.get("triple_cover"):
        labels.append("triple cover cut")
    if flags.get("uptime_symmetry"):
        labels.append("uptime symmetry")
    if flags.get("builtin_aggressive"):
        labels.append("aggressive built-in Gurobi cuts")
    return ", ".join(labels) if labels else "추가 cut 없음"


def prepare_instances(selected_groups, source_csv, seed, scenario_seed, scenario_count):
    instance_records = []
    case_order = 0
    for group_name, case_list in selected_groups.items():
        for case_config in case_list:
            case_seed = seed + case_order
            case_order += 1
            instance_name = build_instance_name(group_name, case_config, scenario_count)
            instance_dir = DATA_ROOT / instance_name
            build_instance(
                source_csv=source_csv,
                output_dir=instance_dir,
                instance_name=instance_name,
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
            instance_records.append(
                {
                    "group": group_name,
                    "case": case_config["label"],
                    "case_config": case_config,
                    "instance_name": instance_name,
                    "instance_dir": instance_dir,
                }
            )
    return instance_records


def summarize_row(instance_record, cut_profile, summary):
    row = {
        "group": instance_record["group"],
        "case": instance_record["case"],
        "instance_name": instance_record["instance_name"],
        "cut_profile": cut_profile,
        "cut_category": profile_category(cut_profile),
        "cut_description": profile_description(cut_profile),
        "lambda_migration": LAMBDA_VALUE,
        "status_name": summary.get("status_name"),
        "runtime_seconds": summary.get("runtime_seconds"),
        "total_runtime_seconds": summary.get("total_runtime_seconds"),
        "mip_gap": summary.get("mip_gap"),
        "objective_value": summary.get("objective_value"),
        "objective_bound": summary.get("objective_bound"),
        "node_count": summary.get("node_count"),
        "iter_count": summary.get("iter_count"),
        "work_units": summary.get("work_units"),
        "solution_count": summary.get("solution_count"),
        "used_server_count": summary.get("used_server_count"),
        "migration_count": summary.get("migration_count"),
        "actual_migration_event_count": summary.get("actual_migration_event_count"),
        "peak_realized_server_utilization": summary.get("peak_realized_server_utilization"),
        "peak_overbooking_ratio": summary.get("peak_overbooking_ratio"),
        "max_server_violation_probability": summary.get("max_server_violation_probability"),
        "max_spot_suspension_probability": summary.get("max_spot_suspension_probability"),
        "min_spot_completion_ratio": summary.get("min_spot_completion_ratio"),
        "manual_cut_total": sum(summary.get("cut_counts", {}).values()),
    }
    for cut_name in MANUAL_CUT_COLUMNS:
        row[f"cut_count_{cut_name}"] = int(summary.get("cut_counts", {}).get(cut_name, 0))
    return row


def run_single_task(task):
    instance_record, cut_profile, time_limit, mip_gap, threads = task
    results_name = build_results_name(instance_record["instance_name"], cut_profile)
    results_dir = RUNS_GROUP_DIR / cut_profile / results_name
    results_dir.parent.mkdir(parents=True, exist_ok=True)

    summary = solve_instance(
        instance_dir=instance_record["instance_dir"],
        results_dir=results_dir,
        time_limit=time_limit,
        mip_gap=mip_gap,
        threads=threads,
        norel_pre_time=0.0,
        use_fallback=False,
        cut_profile=cut_profile,
    )
    return summarize_row(instance_record, cut_profile, summary)


def create_runtime_figure(summary_df, output_path):
    plot_df = summary_df.copy()
    plot_df["job_label"] = plot_df["group"] + " | " + plot_df["case"]
    pivot = plot_df.pivot(index="cut_profile", columns="job_label", values="runtime_seconds").sort_index()

    fig, ax = plt.subplots(figsize=(16, 7))
    image = ax.imshow(pivot.fillna(0.0).to_numpy(), aspect="auto", cmap="YlOrRd")
    ax.set_title("Runtime Heatmap by Cut Profile")
    ax.set_xlabel("case")
    ax.set_ylabel("cut profile")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    fig.colorbar(image, ax=ax, label="runtime (sec)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_gap_figure(summary_df, output_path):
    plot_df = summary_df.copy()
    plot_df["job_label"] = plot_df["group"] + " | " + plot_df["case"]
    pivot = plot_df.pivot(index="cut_profile", columns="job_label", values="mip_gap").sort_index()

    fig, ax = plt.subplots(figsize=(16, 7))
    image = ax.imshow(pivot.fillna(0.0).to_numpy(), aspect="auto", cmap="PuBuGn")
    ax.set_title("Final MIP Gap Heatmap by Cut Profile")
    ax.set_xlabel("case")
    ax.set_ylabel("cut profile")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    fig.colorbar(image, ax=ax, label="MIP gap")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_manifest(summary_df, output_path):
    profile_df = pd.DataFrame(
        [
            {
                "cut_profile": profile_name,
                "category": profile_category(profile_name),
                "description": profile_description(profile_name),
                "enabled_flags": ", ".join(key for key, value in profile.items() if value),
                "solver_params": "built-in aggressive cuts" if profile.get("builtin_aggressive") else "(default)",
            }
            for profile_name, profile in CUT_PROFILES.items()
        ]
    )

    lines = [
        "# 24VM Cut Experiment",
        "",
        "## 실험 설정",
        "",
        f"- instance: 24VM / cap8 / avg_cpu>=20 / lambda={LAMBDA_VALUE}",
        "- NoRel heuristic: 사용하지 않음",
        "- time limit: 3600초",
        "- threads per job: 8",
        "",
        "## cut profile 목록",
        "",
    ]
    for row in profile_df.itertuples(index=False):
        lines.append(f"### {row.cut_profile}")
        lines.append(f"- category: {row.category}")
        lines.append(f"- description: {row.description}")
        lines.append(f"- enabled flags: {row.enabled_flags or '(none)'}")
        lines.append(f"- solver params: {row.solver_params or '(default)'}")
        lines.append("")

    if not summary_df.empty:
        best_runtime = (
            summary_df.groupby("cut_profile", as_index=False)["runtime_seconds"]
            .median()
            .sort_values("runtime_seconds")
        )
        lines.extend(["## 현재까지의 중앙값 runtime", ""])
        for row in best_runtime.itertuples(index=False):
            lines.append(f"- {row.cut_profile}: {row.runtime_seconds:.2f} sec")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    selected_groups = {group_name: CASE_GROUPS[group_name] for group_name in args.groups}
    source_csv = args.source_csv.resolve()

    RUNS_GROUP_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    instance_records = prepare_instances(
        selected_groups=selected_groups,
        source_csv=source_csv,
        seed=args.seed,
        scenario_seed=args.scenario_seed,
        scenario_count=args.scenario_count,
    )

    tasks = []
    for instance_record in instance_records:
        for cut_profile in args.profiles:
            tasks.append(
                (
                    instance_record,
                    cut_profile,
                    args.time_limit,
                    args.mip_gap,
                    args.threads,
                )
            )

    rows = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        future_map = {
            executor.submit(run_single_task, task): (task[0]["case"], task[1])
            for task in tasks
        }
        for future in as_completed(future_map):
            case_label, cut_profile = future_map[future]
            row = future.result()
            rows.append(row)
            print(
                f"[done] {case_label} / {cut_profile}: "
                f"status={row['status_name']}, runtime={row['runtime_seconds']}, gap={row['mip_gap']}"
            )

    summary_df = pd.DataFrame(rows).sort_values(["cut_profile", "group", "case"]).reset_index(drop=True)
    summary_csv = ANALYSIS_DIR / "all_jobs_summary.csv"
    runtime_png = ANALYSIS_DIR / "runtime_heatmap.png"
    gap_png = ANALYSIS_DIR / "gap_heatmap.png"
    manifest_md = ANALYSIS_DIR / "CUT_MANIFEST.md"

    summary_df.to_csv(summary_csv, index=False)
    create_runtime_figure(summary_df, runtime_png)
    create_gap_figure(summary_df, gap_png)
    write_manifest(summary_df, manifest_md)

    print(f"saved: {summary_csv}")
    print(f"saved: {runtime_png}")
    print(f"saved: {gap_png}")
    print(f"saved: {manifest_md}")


if __name__ == "__main__":
    main()
