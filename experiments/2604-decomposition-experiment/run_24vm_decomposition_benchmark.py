import argparse
import json
import math
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
ANALYSIS_DIR = EXPERIMENT_DIR / "results" / "analysis" / "chance_2sp_toy_24vm_decomposition_benchmark"
RUNS_GROUP_DIR = EXPERIMENT_DIR / "results" / "runs" / ANALYSIS_DIR.name

ENERGY_IDLE = 100.0
ENERGY_CPU = 300.0
ENERGY_MIGRATION = 50.0
CASE_CONFIG = {"case": "od_sp_bj", "label": "OD + SP + BJ", "on_demand": 8, "spot": 8, "batch": 8}

DECOMPOSITION_PROFILES = [
    {
        "name": "control_combo_phi",
        "cut_profile": "combo_barrier_state_uptime_phi_mass",
        "strategy_type": "plain",
        "notes": "decomposition 미적용 baseline",
    },
    {
        "name": "rep_peak_start_combo",
        "cut_profile": "combo_barrier_state_uptime_phi_mass",
        "strategy_type": "start",
        "start_strategy": "representative_peak",
        "notes": "집계 peak surrogate 해를 직접 warm start로 주입",
    },
    {
        "name": "kwon_threshold_hint_combo",
        "cut_profile": "combo_barrier_state_uptime_phi_mass",
        "strategy_type": "hint",
        "start_strategy": "kwon_threshold_mean",
        "notes": "Kwon 계열 threshold dispatch heuristic",
    },
    {
        "name": "local_branch_rep_peak_combo",
        "cut_profile": "combo_barrier_state_uptime_phi_mass",
        "strategy_type": "local_branch",
        "start_strategy": "representative_peak",
        "local_branch_radius": 40,
        "notes": "representative peak seed 주변 local branching",
    },
    {
        "name": "fixopt_rep_peak_combo",
        "cut_profile": "combo_barrier_state_uptime_phi_mass",
        "strategy_type": "fix_opt",
        "start_strategy": "representative_peak",
        "window_size": 6,
        "notes": "representative peak seed 기준 time-window fix-and-optimize",
    },
    {
        "name": "local_branch_control_seed_combo",
        "cut_profile": "combo_barrier_state_uptime_phi_mass",
        "strategy_type": "local_branch",
        "start_strategy": "control_seed_combo",
        "local_branch_radius": 40,
        "notes": "짧게 푼 control incumbent 주변 local branching",
    },
    {
        "name": "fixopt_control_seed_combo",
        "cut_profile": "combo_barrier_state_uptime_phi_mass",
        "strategy_type": "fix_opt",
        "start_strategy": "control_seed_combo",
        "window_size": 6,
        "notes": "짧게 푼 control incumbent 기준 time-window fix-and-optimize",
    },
]

CONTROL_PROFILES = ("control_combo_phi",)


def parse_args():
    parser = argparse.ArgumentParser(description="24VM decomposition benchmark with control-seed refinement")
    parser.add_argument("--source-csv", type=Path, default=SOURCE_CSV)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario-seed", type=int, default=42)
    parser.add_argument("--scenario-count", type=int, default=10)
    parser.add_argument("--screen-time-limit", type=int, default=300)
    parser.add_argument("--final-time-limit", type=int, default=900)
    parser.add_argument("--strategy-time-limit", type=int, default=60)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--strategy-threads", type=int, default=2)
    parser.add_argument("--finalists", type=int, default=5)
    parser.add_argument("--resume-screen", action="store_true")
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
    outputs = None
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
    center_values = start_values
    windows = build_fixopt_windows(data, profile.get("window_size", 2))
    per_window_time = max(5, int(math.ceil(time_limit / max(1, len(windows)))))
    phase_rows = []
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
            local_branch_radius=None,
            hint_priority=20,
            return_start_values=True,
        )
        last_summary = summary
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
            }
        )

    pd.DataFrame(phase_rows).to_csv(results_dir / "fixopt_phase_history.csv", index=False)
    if last_summary is not None:
        write_summary_only(results_dir, last_summary)
    return last_summary


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


def precompute_starts(data, strategy_time_limit, strategy_threads):
    start_cache = {None: {"start_values": None, "metadata": {"strategy": None}}}
    strategies = sorted({profile.get("start_strategy") for profile in DECOMPOSITION_PROFILES if profile.get("start_strategy")})
    starts_dir = ANALYSIS_DIR / "starts"
    starts_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for strategy_name in strategies:
        if strategy_name == "control_seed_combo":
            start_values, metadata = build_control_seed(
                data=data,
                results_dir=starts_dir,
                time_limit=strategy_time_limit,
                threads=strategy_threads,
            )
        else:
            start_values, metadata = build_start_strategy(
                strategy_name=strategy_name,
                data=data,
                results_dir=starts_dir,
                cut_profile="combo_barrier_state_uptime_phi_mass",
                strategy_time_limit=strategy_time_limit,
                strategy_threads=strategy_threads,
            )
        start_cache[strategy_name] = {"start_values": start_values, "metadata": metadata}
        rows.append(
            {
                "start_strategy": strategy_name,
                "has_start": start_values is not None,
                "metadata": json.dumps(metadata, ensure_ascii=False),
            }
        )
    pd.DataFrame(rows).to_csv(ANALYSIS_DIR / "start_generation_summary.csv", index=False)
    return start_cache


def run_profile(task):
    stage_name, case_name, instance_dir, profile, time_limit, mip_gap, threads, start_payload = task
    data = load_instance_data(instance_dir)
    results_dir = RUNS_GROUP_DIR / stage_name / case_name / profile["name"]
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = None
    start_values = start_payload["start_values"]
    has_seed = start_values is not None

    if profile["strategy_type"] == "plain" or (profile["strategy_type"] != "plain" and not has_seed):
        summary = run_single_pass(
            data=data,
            results_dir=results_dir,
            cut_profile=profile["cut_profile"],
            time_limit=time_limit,
            mip_gap=mip_gap,
            threads=threads,
            log_name="solver.log" if profile["strategy_type"] == "plain" else "solver_fallback.log",
        )
    elif profile["strategy_type"] == "start":
        summary = run_single_pass(
            data=data,
            results_dir=results_dir,
            cut_profile=profile["cut_profile"],
            time_limit=time_limit,
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
            time_limit=time_limit,
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
            time_limit=time_limit,
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
            time_limit=time_limit,
            mip_gap=mip_gap,
            threads=threads,
            start_values=start_values,
        )
    else:
        raise ValueError(f"Unknown strategy type: {profile['strategy_type']}")

    return {
        "stage": stage_name,
        "case": CASE_CONFIG["label"],
        "case_key": CASE_CONFIG["case"],
        "instance_name": case_name,
        "profile": profile["name"],
        "cut_profile": profile["cut_profile"],
        "strategy_type": profile["strategy_type"],
        "start_strategy": profile.get("start_strategy", "none"),
        "notes": profile.get("notes", ""),
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
        "start_available": start_values is not None,
        "start_metadata": json.dumps(start_payload["metadata"], ensure_ascii=False),
    }


def run_stage(stage_name, case_name, instance_dir, profiles, time_limit, mip_gap, max_workers, threads, start_cache):
    tasks = [
        (
            stage_name,
            case_name,
            instance_dir,
            profile,
            time_limit,
            mip_gap,
            threads,
            start_cache.get(profile.get("start_strategy"), {"start_values": None, "metadata": {"strategy": None}}),
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
                gap_text = f"{row['mip_gap'] * 100.0:.2f}%"
            print(
                f"[{stage_name}] {row['profile']}: status={row['status_name']}, "
                f"runtime={row['runtime_seconds']:.1f}s, gap={gap_text}"
            )
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_profile, task): task[3]["name"] for task in tasks}
            for future in as_completed(futures):
                profile_name = futures[future]
                row = future.result()
                rows.append(row)
                gap_text = "nan"
                if row["mip_gap"] is not None:
                    gap_text = f"{row['mip_gap'] * 100.0:.2f}%"
                print(
                    f"[{stage_name}] {profile_name}: status={row['status_name']}, "
                    f"runtime={row['runtime_seconds']:.1f}s, gap={gap_text}"
                )
    return pd.DataFrame(rows).sort_values(["profile"]).reset_index(drop=True)


def load_stage_from_run_dirs(stage_name, case_name, profiles, start_cache):
    rows = []
    for profile in profiles:
        results_dir = RUNS_GROUP_DIR / stage_name / case_name / profile["name"]
        summary_path = results_dir / "summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path, "r", encoding="utf-8") as file:
            summary = json.load(file)
        start_payload = start_cache.get(profile.get("start_strategy"), {"start_values": None, "metadata": {"strategy": None}})
        rows.append(
            {
                "stage": stage_name,
                "case": CASE_CONFIG["label"],
                "case_key": CASE_CONFIG["case"],
                "instance_name": case_name,
                "profile": profile["name"],
                "cut_profile": profile["cut_profile"],
                "strategy_type": profile["strategy_type"],
                "start_strategy": profile.get("start_strategy", "none"),
                "notes": profile.get("notes", ""),
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
                "start_available": start_payload["start_values"] is not None,
                "start_metadata": json.dumps(start_payload["metadata"], ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows).sort_values(["profile"]).reset_index(drop=True)


def add_relative_metrics(summary_df):
    summary_df = summary_df.copy()
    for column in ["objective_value", "objective_bound", "mip_gap", "runtime_seconds"]:
        summary_df[column] = pd.to_numeric(summary_df[column], errors="coerce")

    baseline = summary_df.loc[summary_df["profile"] == "control_combo_phi"].iloc[0]
    baseline_obj = float(baseline["objective_value"])
    baseline_bound = float(baseline["objective_bound"])
    baseline_gap = float(baseline["mip_gap"])

    summary_df["incumbent_improve_pct"] = (baseline_obj - summary_df["objective_value"]) / baseline_obj * 100.0
    summary_df["bound_improve_pct"] = (summary_df["objective_bound"] - baseline_bound) / baseline_bound * 100.0
    summary_df["gap_improve_pct"] = (baseline_gap - summary_df["mip_gap"]) / baseline_gap * 100.0
    summary_df["screen_score"] = (
        0.50 * summary_df["gap_improve_pct"]
        + 0.30 * summary_df["incumbent_improve_pct"]
        + 0.20 * summary_df["bound_improve_pct"]
    )
    return summary_df


def choose_finalists(screen_df, finalist_count):
    ranked = screen_df.loc[~screen_df["profile"].isin(CONTROL_PROFILES) & screen_df["has_solution"]].copy()
    ranked = ranked.sort_values(["screen_score", "mip_gap", "objective_value"], ascending=[False, True, True])
    finalists = list(CONTROL_PROFILES) + ranked["profile"].head(finalist_count - len(CONTROL_PROFILES)).tolist()
    profile_map = {profile["name"]: profile for profile in DECOMPOSITION_PROFILES}
    return [profile_map[name] for name in finalists], ranked


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


def write_report(screen_df, final_df, finalists, output_path, screen_time_limit, final_time_limit):
    lines = [
        "# 24VM decomposition benchmark",
        "",
        "## 실험 설정",
        "",
        "- 고정 인스턴스: `24VM`, `OD + SP + BJ = 8 + 8 + 8`, `cap=8`, `avg_cpu_mean >= 20`, `scenario=10`",
        "- 목적함수: 에너지 최소화",
        f"- 1차 screening: `{screen_time_limit}`초",
        f"- 2차 final: `{final_time_limit}`초",
        "- baseline: `control_combo_phi`",
        "- decomposition 축: representative surrogate, Kwon threshold, control incumbent seed, local branching, fix-and-optimize",
        "",
        "## 최종 재실행 대상",
        "",
    ]
    for profile in finalists:
        lines.append(f"- `{profile['name']}`")

    lines += [
        "",
        "## screening 결과",
        "",
        frame_to_markdown(
            screen_df[
                [
                    "profile",
                    "cut_profile",
                    "strategy_type",
                    "start_strategy",
                    "objective_value",
                    "objective_bound",
                    "mip_gap",
                    "screen_score",
                ]
            ].sort_values(["screen_score", "mip_gap"], ascending=[False, True])
        ),
        "",
        "## final 결과",
        "",
        frame_to_markdown(
            final_df[
                [
                    "profile",
                    "cut_profile",
                    "strategy_type",
                    "start_strategy",
                    "objective_value",
                    "objective_bound",
                    "mip_gap",
                    "used_server_count",
                    "node_count",
                ]
            ].sort_values(["mip_gap", "objective_value"])
        ),
        "",
        "## 문헌 기반 해석",
        "",
        "- `representative_peak`는 Kwon의 time-stability / surrogate 축과 유사하게, 불확실 수요를 보수적 대표 시나리오로 축약해 초기해를 만드는 전략이다.",
        "- `kwon_threshold_mean`은 rule-constrained threshold policy 아이디어를 VM/server activation 규칙으로 옮긴 것이다.",
        "- `control_seed_combo`는 원문제를 짧은 시간만 먼저 풀어 incumbent를 확보한 뒤, 그 해를 후속 neighborhood search의 기준점으로 쓰는 2단계 전략이다.",
        "- `local_branch`와 `fix_opt`는 seed를 기준으로 제한된 neighborhood 또는 time-window 부분문제를 푸는 후처리 단계다.",
        "",
        "## 해석 시 주의사항",
        "",
        "- `control_combo_phi`, `kwon_threshold_hint_combo`, `rep_peak_start_combo`는 전체 원문제를 그대로 푼 결과다.",
        "- `local_branch_*`, `fixopt_*`는 제한된 이웃 또는 부분문제를 푼 결과이므로, reported gap은 전체 원문제의 전역 gap과 다르게 읽어야 한다.",
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
    data = load_instance_data(instance_dir)
    start_cache = precompute_starts(
        data=data,
        strategy_time_limit=args.strategy_time_limit,
        strategy_threads=args.strategy_threads,
    )

    if args.resume_screen:
        screen_df = load_stage_from_run_dirs(
            stage_name="screen",
            case_name=case_name,
            profiles=DECOMPOSITION_PROFILES,
            start_cache=start_cache,
        )
    else:
        screen_df = run_stage(
            stage_name="screen",
            case_name=case_name,
            instance_dir=instance_dir,
            profiles=DECOMPOSITION_PROFILES,
            time_limit=args.screen_time_limit,
            mip_gap=args.mip_gap,
            max_workers=args.max_workers,
            threads=args.threads,
            start_cache=start_cache,
        )
    screen_df = add_relative_metrics(screen_df)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    screen_path = ANALYSIS_DIR / "screen_summary.csv"
    screen_df.to_csv(screen_path, index=False)

    finalists, ranked = choose_finalists(screen_df, args.finalists)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(ANALYSIS_DIR / "screen_ranking.csv", index=False)

    final_df = run_stage(
        stage_name="final",
        case_name=case_name,
        instance_dir=instance_dir,
        profiles=finalists,
        time_limit=args.final_time_limit,
        mip_gap=args.mip_gap,
        max_workers=args.max_workers,
        threads=args.threads,
        start_cache=start_cache,
    )
    final_df = add_relative_metrics(final_df)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    final_path = ANALYSIS_DIR / "final_summary.csv"
    final_df.to_csv(final_path, index=False)

    write_report(
        screen_df=screen_df,
        final_df=final_df,
        finalists=finalists,
        output_path=ANALYSIS_DIR / "SEARCH_REPORT.md",
        screen_time_limit=args.screen_time_limit,
        final_time_limit=args.final_time_limit,
    )
    print(f"analysis_dir={ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
