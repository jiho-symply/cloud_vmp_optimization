import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
EXPERIMENT_NAME = EXPERIMENT_DIR.name
REPO_ROOT = EXPERIMENT_DIR.parents[1]
CUT_EXPERIMENT_DIR = REPO_ROOT / "experiments" / "2604-cut-experiment"
if str(CUT_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(CUT_EXPERIMENT_DIR))
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from build_dataset import build_instance  # noqa: E402
from run_model import (  # noqa: E402
    apply_fixed_values,
    apply_start_values,
    apply_var_hints,
    build_model,
    build_summary,
    capture_start_values,
    extract_solution_tables,
    load_instance_data,
    write_solution_outputs,
    write_summary_only,
)

from decomposition_starts import build_start_strategy, filter_named_values  # noqa: E402


SOURCE_CSV = REPO_ROOT / "data" / "processed" / "2601-initial-toy-model" / "sample_vm_data.csv"
DATA_ROOT = REPO_ROOT / "data" / "processed" / EXPERIMENT_NAME
ANALYSIS_DIR = EXPERIMENT_DIR / "results" / "analysis" / "chance_2sp_toy_12vm_decomposition_full_t3600"
RUNS_GROUP_DIR = EXPERIMENT_DIR / "results" / "runs" / ANALYSIS_DIR.name

ENERGY_IDLE = 100.0
ENERGY_CPU = 300.0
ENERGY_MIGRATION = 50.0
CASE_CONFIG = {"case": "od_sp_bj", "label": "OD + SP + BJ", "on_demand": 4, "spot": 4, "batch": 4}

DECOMPOSITION_PROFILES = [
    {
        "name": "control_combo_phi",
        "cut_profile": "combo_barrier_state_uptime_phi_mass",
        "strategy_type": "plain",
        "notes": "전체 원문제를 그대로 푸는 baseline",
    },
    {
        "name": "control_state_link",
        "cut_profile": "state_link",
        "strategy_type": "plain",
        "notes": "state_link 중심 bound 강화 baseline",
    },
    {
        "name": "rep_peak_start_combo",
        "cut_profile": "combo_barrier_state_uptime_phi_mass",
        "strategy_type": "start",
        "start_strategy": "representative_peak",
        "notes": "representative peak 해를 warm start로 주입",
    },
    {
        "name": "rep_peak_hint_combo",
        "cut_profile": "combo_barrier_state_uptime_phi_mass",
        "strategy_type": "hint",
        "start_strategy": "representative_peak",
        "notes": "representative peak 해를 var hint로 주입",
    },
    {
        "name": "scenario_consensus_hint_combo",
        "cut_profile": "combo_barrier_state_uptime_phi_mass",
        "strategy_type": "hint",
        "start_strategy": "scenario_consensus",
        "notes": "시나리오 합의 해를 var hint로 주입",
    },
    {
        "name": "kwon_threshold_hint_combo",
        "cut_profile": "combo_barrier_state_uptime_phi_mass",
        "strategy_type": "hint",
        "start_strategy": "kwon_threshold_mean",
        "notes": "Kwon threshold 규칙 기반 seed를 var hint로 주입",
    },
    {
        "name": "kwon_rolling_hint_combo",
        "cut_profile": "combo_barrier_state_uptime_phi_mass",
        "strategy_type": "hint",
        "start_strategy": "rolling_horizon_mean",
        "notes": "rolling horizon seed를 var hint로 주입",
    },
    {
        "name": "state_link_rep_peak_hint",
        "cut_profile": "state_link",
        "strategy_type": "hint",
        "start_strategy": "representative_peak",
        "notes": "state_link에 representative peak hint 결합",
    },
    {
        "name": "local_branch_rep_peak_combo",
        "cut_profile": "combo_barrier_state_uptime_phi_mass",
        "strategy_type": "local_branch",
        "start_strategy": "representative_peak",
        "local_branch_radius": 16,
        "notes": "representative peak seed 주변 local branching",
    },
    {
        "name": "fixopt_rep_peak_combo",
        "cut_profile": "combo_barrier_state_uptime_phi_mass",
        "strategy_type": "fix_opt",
        "start_strategy": "representative_peak",
        "window_size": 4,
        "notes": "representative peak seed 기준 fix-and-optimize",
    },
    {
        "name": "local_branch_control_seed_combo",
        "cut_profile": "combo_barrier_state_uptime_phi_mass",
        "strategy_type": "local_branch",
        "start_strategy": "control_seed_combo",
        "local_branch_radius": 16,
        "notes": "짧게 푼 control incumbent 주변 local branching",
    },
    {
        "name": "fixopt_control_seed_combo",
        "cut_profile": "combo_barrier_state_uptime_phi_mass",
        "strategy_type": "fix_opt",
        "start_strategy": "control_seed_combo",
        "window_size": 4,
        "notes": "짧게 푼 control incumbent 기준 fix-and-optimize",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="12VM decomposition full benchmark with a single 3600-second budget")
    parser.add_argument("--source-csv", type=Path, default=SOURCE_CSV)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario-seed", type=int, default=42)
    parser.add_argument("--scenario-count", type=int, default=10)
    parser.add_argument("--time-limit", type=int, default=3600)
    parser.add_argument("--seed-time-limit", type=int, default=600)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--strategy-threads", type=int, default=4)
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


def build_local_branching_constraint(model, variables, center_values, radius):
    if not center_values:
        return None

    terms = []
    for name in ("u_used", "x", "y", "z"):
        for key, value in center_values.get(name, {}).items():
            if key not in variables[name]:
                continue
            var = variables[name][key]
            if value > 0.5:
                terms.append(1.0 - var)
            else:
                terms.append(var)

    if not terms:
        return None

    return model.addConstr(sum(terms) <= int(radius), name="local_branching")


def build_fixopt_windows(data, window_size):
    times = list(data["times"])
    return [times[index : index + window_size] for index in range(0, len(times), window_size)]


def build_outside_window_fix_values(data, center_values, window_times):
    window_set = set(int(time_value) for time_value in window_times)
    fixed_values = {"x": {}, "y": {}, "z": {}, "u": {}}

    for workload_id in data["spot_ids"]:
        for server in data["servers"]:
            key = (workload_id, server)
            fixed_values["y"][key] = center_values["y"].get(key, 0.0)

    for workload_id in data["on_demand_ids"]:
        for time_value in data["od_active"][workload_id]:
            if int(time_value) in window_set:
                continue
            for server in data["servers"]:
                key = (workload_id, server, time_value)
                fixed_values["x"][key] = center_values["x"].get(key, 0.0)

    for batch_job_id in data["batch_ids"]:
        source_time = int(data["batch_source_time"].get(batch_job_id, data["times"][0]))
        if source_time in window_set:
            continue
        for server in data["servers"]:
            for time_value in data["times"]:
                key = (batch_job_id, server, time_value)
                fixed_values["z"][key] = center_values["z"].get(key, 0.0)

    for server in data["servers"]:
        for time_value in data["times"]:
            if int(time_value) in window_set:
                continue
            key = (server, time_value)
            fixed_values["u"][key] = center_values["u"].get(key, 0.0)

    return fixed_values


def run_single_pass(
    data,
    results_dir,
    cut_profile,
    time_limit,
    mip_gap,
    threads,
    log_name,
    start_values=None,
    hint_values=None,
    fixed_values=None,
    local_branch_radius=None,
    hint_priority=10,
    return_start_values=False,
):
    results_dir.mkdir(parents=True, exist_ok=True)
    model, variables = build_model(
        data=data,
        results_dir=results_dir,
        time_limit=time_limit,
        mip_gap=mip_gap,
        log_name=log_name,
        threads=threads,
        server_limit=None,
        cut_profile=cut_profile,
    )
    apply_fixed_values(variables, fixed_values)
    apply_start_values(variables, start_values)
    apply_var_hints(variables, hint_values, hint_priority=hint_priority)
    if local_branch_radius is not None:
        build_local_branching_constraint(model, variables, start_values or hint_values, local_branch_radius)
    model.optimize()
    summary = build_summary(model, data)
    incumbent_values = None
    if model.SolCount > 0:
        incumbent_values = capture_start_values(variables)
        summary_updates, outputs = extract_solution_tables(data, variables)
        summary.update(summary_updates)
        write_solution_outputs(results_dir, summary, outputs)
    else:
        write_summary_only(results_dir, summary)
    if return_start_values:
        return summary, incumbent_values
    return summary


def run_fixopt_profile(data, results_dir, profile, time_limit, mip_gap, threads, start_values):
    wall_start = time.perf_counter()
    center_values = start_values
    windows = build_fixopt_windows(data, profile.get("window_size", 2))
    per_window_time = max(0.1, float(time_limit) / max(1, len(windows)))
    phase_rows = []
    total_node_count = 0.0
    total_iter_count = 0.0
    total_work_units = 0.0
    last_summary = None

    for window_index, window_times in enumerate(windows, start=1):
        phase_dir = results_dir / f"window_{window_index:02d}"
        phase_dir.mkdir(parents=True, exist_ok=True)
        fixed_values = build_outside_window_fix_values(data, center_values, window_times)
        summary, incumbent_values = run_single_pass(
            data=data,
            results_dir=phase_dir,
            cut_profile=profile["cut_profile"],
            time_limit=per_window_time,
            mip_gap=mip_gap,
            threads=threads,
            log_name="solver.log",
            start_values=center_values,
            hint_values=center_values,
            fixed_values=fixed_values,
            hint_priority=20,
            return_start_values=True,
        )
        last_summary = summary
        total_node_count += float(summary.get("node_count") or 0.0)
        total_iter_count += float(summary.get("iter_count") or 0.0)
        total_work_units += float(summary.get("work_units") or 0.0)
        if incumbent_values is not None:
            center_values = filter_named_values(incumbent_values, ("u", "u_used", "x", "m", "y", "z"))
        phase_rows.append(
            {
                "window_index": window_index,
                "window_times": list(window_times),
                "status_name": summary.get("status_name"),
                "has_solution": summary.get("has_solution"),
                "objective_value": summary.get("objective_value"),
                "objective_bound": summary.get("objective_bound"),
                "mip_gap": summary.get("mip_gap"),
                "runtime_seconds": summary.get("runtime_seconds"),
            }
        )

    pd.DataFrame(phase_rows).to_csv(results_dir / "fixopt_phase_history.csv", index=False)
    if last_summary is None:
        return None

    total_summary = dict(last_summary)
    total_summary["runtime_seconds"] = time.perf_counter() - wall_start
    total_summary["node_count"] = total_node_count
    total_summary["iter_count"] = total_iter_count
    total_summary["work_units"] = total_work_units
    write_summary_only(results_dir, total_summary)
    return total_summary


def build_control_seed(data, results_dir, time_limit, threads):
    seed_dir = results_dir / "control_seed_combo"
    seed_dir.mkdir(parents=True, exist_ok=True)
    model, variables = build_model(
        data=data,
        results_dir=seed_dir,
        time_limit=time_limit,
        mip_gap=0.01,
        log_name="solver.log",
        threads=threads,
        server_limit=None,
        cut_profile="combo_barrier_state_uptime_phi_mass",
    )
    model.setParam("MIPFocus", 1)
    model.setParam("Heuristics", 0.3)
    model.setParam("NoRelHeurTime", float(time_limit))
    model.optimize()
    summary = build_summary(model, data)
    incumbent_values = None
    if model.SolCount > 0:
        incumbent_values = capture_start_values(variables)
        summary_updates, outputs = extract_solution_tables(data, variables)
        summary.update(summary_updates)
        write_solution_outputs(seed_dir, summary, outputs)
    else:
        write_summary_only(seed_dir, summary)
    if incumbent_values is None:
        return None, {"strategy": "control_seed_combo", "summary": summary}
    return (
        filter_named_values(incumbent_values, ("u", "u_used", "x", "m", "y", "z")),
        {"strategy": "control_seed_combo", "summary": summary},
    )


def requires_optimization_seed(strategy_name):
    return strategy_name in {
        "representative_mean",
        "representative_peak",
        "single_scenario_best",
        "scenario_consensus",
        "progressive_mean",
        "rolling_horizon_mean",
        "rolling_horizon_peak",
        "control_seed_combo",
    }


def build_profile_seed(profile, data, results_dir, seed_time_limit, strategy_threads):
    strategy_name = profile.get("start_strategy")
    if strategy_name is None:
        return None, {"strategy": None}, 0.0
    if not requires_optimization_seed(strategy_name):
        start_values, metadata = build_start_strategy(
            strategy_name=strategy_name,
            data=data,
            results_dir=results_dir,
            cut_profile=profile["cut_profile"],
            strategy_time_limit=0.0,
            strategy_threads=strategy_threads,
        )
        return start_values, metadata, 0.0
    if strategy_name == "control_seed_combo":
        start_values, metadata = build_control_seed(
            data=data,
            results_dir=results_dir,
            time_limit=seed_time_limit,
            threads=strategy_threads,
        )
        return start_values, metadata, float(seed_time_limit)
    start_values, metadata = build_start_strategy(
        strategy_name=strategy_name,
        data=data,
        results_dir=results_dir,
        cut_profile=profile["cut_profile"],
        strategy_time_limit=seed_time_limit,
        strategy_threads=strategy_threads,
    )
    return start_values, metadata, float(seed_time_limit)


def run_profile(task):
    profile, case_name, instance_dir, total_time_limit, seed_time_limit, mip_gap, threads, strategy_threads = task
    data = load_instance_data(instance_dir)
    results_dir = RUNS_GROUP_DIR / case_name / profile["name"]
    results_dir.mkdir(parents=True, exist_ok=True)

    wall_start = time.perf_counter()
    start_values, start_metadata, consumed_seed_time = build_profile_seed(
        profile=profile,
        data=data,
        results_dir=results_dir / "seed_generation",
        seed_time_limit=seed_time_limit,
        strategy_threads=strategy_threads,
    )
    solve_time_limit = max(0.1, float(total_time_limit) - float(consumed_seed_time))

    if profile["strategy_type"] == "plain":
        summary = run_single_pass(
            data=data,
            results_dir=results_dir,
            cut_profile=profile["cut_profile"],
            time_limit=total_time_limit,
            mip_gap=mip_gap,
            threads=threads,
            log_name="solver.log",
        )
    elif start_values is None:
        summary = run_single_pass(
            data=data,
            results_dir=results_dir,
            cut_profile=profile["cut_profile"],
            time_limit=solve_time_limit,
            mip_gap=mip_gap,
            threads=threads,
            log_name="solver_fallback.log",
        )
    elif profile["strategy_type"] == "start":
        summary = run_single_pass(
            data=data,
            results_dir=results_dir,
            cut_profile=profile["cut_profile"],
            time_limit=solve_time_limit,
            mip_gap=mip_gap,
            threads=threads,
            log_name="solver.log",
            start_values=start_values,
        )
    elif profile["strategy_type"] == "hint":
        summary = run_single_pass(
            data=data,
            results_dir=results_dir,
            cut_profile=profile["cut_profile"],
            time_limit=solve_time_limit,
            mip_gap=mip_gap,
            threads=threads,
            log_name="solver.log",
            hint_values=start_values,
            hint_priority=20,
        )
    elif profile["strategy_type"] == "local_branch":
        summary = run_single_pass(
            data=data,
            results_dir=results_dir,
            cut_profile=profile["cut_profile"],
            time_limit=solve_time_limit,
            mip_gap=mip_gap,
            threads=threads,
            log_name="solver.log",
            start_values=start_values,
            hint_values=start_values,
            local_branch_radius=profile.get("local_branch_radius", 16),
            hint_priority=20,
        )
    elif profile["strategy_type"] == "fix_opt":
        summary = run_fixopt_profile(
            data=data,
            results_dir=results_dir,
            profile=profile,
            time_limit=solve_time_limit,
            mip_gap=mip_gap,
            threads=threads,
            start_values=start_values,
        )
    else:
        raise ValueError(f"Unknown strategy type: {profile['strategy_type']}")

    wall_clock_seconds = time.perf_counter() - wall_start
    runtime_seconds = wall_clock_seconds if summary is None else wall_clock_seconds

    return {
        "stage": "full",
        "case": CASE_CONFIG["label"],
        "case_key": CASE_CONFIG["case"],
        "instance_name": case_name,
        "profile": profile["name"],
        "cut_profile": profile["cut_profile"],
        "strategy_type": profile["strategy_type"],
        "start_strategy": profile.get("start_strategy", "none"),
        "notes": profile.get("notes", ""),
        "status_name": None if summary is None else summary.get("status_name"),
        "has_solution": False if summary is None else summary.get("has_solution"),
        "runtime_seconds": runtime_seconds,
        "objective_value": None if summary is None else summary.get("objective_value"),
        "objective_bound": None if summary is None else summary.get("objective_bound"),
        "mip_gap": None if summary is None else summary.get("mip_gap"),
        "node_count": None if summary is None else summary.get("node_count"),
        "iter_count": None if summary is None else summary.get("iter_count"),
        "work_units": None if summary is None else summary.get("work_units"),
        "used_server_count": None if summary is None else summary.get("used_server_count"),
        "results_dir": str(results_dir),
        "start_available": start_values is not None,
        "start_metadata": json.dumps(start_metadata, ensure_ascii=False),
        "seed_budget_seconds": consumed_seed_time,
        "solve_budget_seconds": solve_time_limit if profile["strategy_type"] != "plain" else float(total_time_limit),
        "total_budget_seconds": float(total_time_limit),
    }


def run_benchmark(case_name, instance_dir, profiles, total_time_limit, seed_time_limit, mip_gap, max_workers, threads, strategy_threads):
    tasks = [
        (
            profile,
            case_name,
            instance_dir,
            total_time_limit,
            seed_time_limit,
            mip_gap,
            threads,
            strategy_threads,
        )
        for profile in profiles
    ]

    rows = []
    if int(max_workers) <= 1:
        for task in tasks:
            row = run_profile(task)
            rows.append(row)
            gap_text = "nan"
            if row["mip_gap"] is not None:
                gap_text = f"{float(row['mip_gap']) * 100.0:.2f}%"
            print(
                f"[full] {row['profile']}: status={row['status_name']}, "
                f"runtime={row['runtime_seconds']:.1f}s, gap={gap_text}"
            )
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_profile, task): task[0]["name"] for task in tasks}
            for future in as_completed(futures):
                profile_name = futures[future]
                row = future.result()
                rows.append(row)
                gap_text = "nan"
                if row["mip_gap"] is not None:
                    gap_text = f"{float(row['mip_gap']) * 100.0:.2f}%"
                print(
                    f"[full] {profile_name}: status={row['status_name']}, "
                    f"runtime={row['runtime_seconds']:.1f}s, gap={gap_text}"
                )
    return pd.DataFrame(rows).sort_values(["profile"]).reset_index(drop=True)


def add_relative_metrics(summary_df):
    summary_df = summary_df.copy()
    for column in ["objective_value", "objective_bound", "mip_gap", "runtime_seconds"]:
        summary_df[column] = pd.to_numeric(summary_df[column], errors="coerce")

    baseline_rows = summary_df.loc[summary_df["profile"] == "control_combo_phi"]
    if baseline_rows.empty:
        return summary_df

    baseline = baseline_rows.iloc[0]
    baseline_obj = baseline["objective_value"]
    baseline_bound = baseline["objective_bound"]
    baseline_gap = baseline["mip_gap"]

    if pd.notna(baseline_obj) and baseline_obj != 0:
        summary_df["incumbent_improve_pct"] = (baseline_obj - summary_df["objective_value"]) / baseline_obj * 100.0
    else:
        summary_df["incumbent_improve_pct"] = pd.NA

    if pd.notna(baseline_bound) and baseline_bound != 0:
        summary_df["bound_improve_pct"] = (summary_df["objective_bound"] - baseline_bound) / baseline_bound * 100.0
    else:
        summary_df["bound_improve_pct"] = pd.NA

    if pd.notna(baseline_gap) and baseline_gap != 0:
        summary_df["gap_improve_pct"] = (baseline_gap - summary_df["mip_gap"]) / baseline_gap * 100.0
    else:
        summary_df["gap_improve_pct"] = pd.NA

    return summary_df


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


def write_report(summary_df, output_path, total_time_limit, seed_time_limit):
    lines = [
        "# 12VM decomposition full benchmark",
        "",
        "## 실험 설정",
        "",
        "- 고정 인스턴스: `12VM`, `OD + SP + BJ = 4 + 4 + 4`, `cap=8`, `avg_cpu_mean >= 20`, `scenario=10`",
        "- 목적함수: 에너지 최소화",
        f"- profile별 총 예산: `{total_time_limit}`초",
        f"- optimization 기반 seed는 profile별 총 예산 안에서 최대 `{seed_time_limit}`초를 사용하고, 나머지를 본 solve 또는 refinement에 사용",
        "- rule 기반 seed (`kwon_threshold_mean`)는 별도 solver 시간을 쓰지 않음",
        "- screening/final 분리를 없애고 모든 profile을 동일 규칙으로 한 번씩 실행",
        "",
        "## 실행 대상",
        "",
    ]
    for profile in DECOMPOSITION_PROFILES:
        lines.append(f"- `{profile['name']}`: {profile['notes']}")

    lines += [
        "",
        "## 결과 요약",
        "",
        frame_to_markdown(
            summary_df[
                [
                    "profile",
                    "strategy_type",
                    "start_strategy",
                    "has_solution",
                    "objective_value",
                    "objective_bound",
                    "mip_gap",
                    "used_server_count",
                    "runtime_seconds",
                    "seed_budget_seconds",
                    "solve_budget_seconds",
                ]
            ].sort_values(["has_solution", "objective_value", "mip_gap"], ascending=[False, True, True])
        ),
        "",
        "## 해석 시 주의사항",
        "",
        "- `local_branch_*`, `fixopt_*`는 제한된 이웃 또는 부분문제를 푸는 전략이므로 reported gap은 전체 원문제의 전역 gap과 동일하게 읽으면 안 된다.",
        "- 반면 이 전략들의 feasible incumbent 품질과 사용 서버 수, migration/energy 구성은 비교할 수 있다.",
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

    summary_df = run_benchmark(
        case_name=case_name,
        instance_dir=instance_dir,
        profiles=DECOMPOSITION_PROFILES,
        total_time_limit=args.time_limit,
        seed_time_limit=args.seed_time_limit,
        mip_gap=args.mip_gap,
        max_workers=args.max_workers,
        threads=args.threads,
        strategy_threads=args.strategy_threads,
    )
    summary_df = add_relative_metrics(summary_df)
    summary_df.to_csv(ANALYSIS_DIR / "full_summary.csv", index=False)
    write_report(
        summary_df=summary_df,
        output_path=ANALYSIS_DIR / "REPORT.md",
        total_time_limit=args.time_limit,
        seed_time_limit=args.seed_time_limit,
    )
    print(f"analysis_dir={ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
