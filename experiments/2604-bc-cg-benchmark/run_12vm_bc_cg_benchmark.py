import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import gurobipy as gp
import pandas as pd
from gurobipy import GRB


EXPERIMENT_DIR = Path(__file__).resolve().parent
EXPERIMENT_NAME = EXPERIMENT_DIR.name
REPO_ROOT = EXPERIMENT_DIR.parents[1]
CUT_EXPERIMENT_DIR = REPO_ROOT / "experiments" / "2604-cut-experiment"
DECOMP_EXPERIMENT_DIR = REPO_ROOT / "experiments" / "2604-decomposition-experiment"

if str(CUT_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(CUT_EXPERIMENT_DIR))
if str(DECOMP_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DECOMP_EXPERIMENT_DIR))

from build_dataset import build_instance  # noqa: E402
from decomposition_starts import build_start_strategy  # noqa: E402
from run_model import (  # noqa: E402
    apply_fixed_values,
    apply_start_values,
    apply_var_hints,
    build_cover_candidates,
    build_cover_expr,
    build_greedy_minimal_cover,
    build_model,
    build_summary,
    extract_solution_tables,
    load_instance_data,
    write_solution_outputs,
    write_summary_only,
)

from column_generation_starts import build_column_generation_start  # noqa: E402


SOURCE_CSV = REPO_ROOT / "data" / "processed" / "2601-initial-toy-model" / "sample_vm_data.csv"
DATA_ROOT = REPO_ROOT / "data" / "processed" / EXPERIMENT_NAME
ANALYSIS_DIR = EXPERIMENT_DIR / "results" / "analysis" / "chance_2sp_toy_12vm_bc_cg_t10800"
RUNS_GROUP_DIR = EXPERIMENT_DIR / "results" / "runs" / ANALYSIS_DIR.name

ENERGY_IDLE = 100.0
ENERGY_CPU = 300.0
ENERGY_MIGRATION = 50.0
CASE_CONFIG = {"case": "od_sp_bj", "label": "OD + SP + BJ", "on_demand": 4, "spot": 4, "batch": 4}


SETTINGS_EXPERIMENTS = [
    {
        "category": "settings",
        "name": "set_state_link_plain",
        "cut_profile": "state_link",
        "notes": "기존 결과에서 gap 기준 최상위였던 state_link plain",
    },
    {
        "category": "settings",
        "name": "set_state_phi_plain",
        "cut_profile": "state_link_phi_mass",
        "notes": "state_link에 phi scenario mass만 더한 lean 세팅",
    },
    {
        "category": "settings",
        "name": "set_combo_state_phi_barrier",
        "cut_profile": "combo_barrier_state_phi_mass",
        "notes": "barrier와 state/phi를 결합한 lean combo",
    },
    {
        "category": "settings",
        "name": "set_combo_state_uptime_phi_barrier",
        "cut_profile": "combo_barrier_state_uptime_phi_mass",
        "notes": "기존 combined baseline",
    },
    {
        "category": "settings",
        "name": "set_rep_peak_hint_state_phi",
        "cut_profile": "combo_barrier_state_phi_mass",
        "seed_strategy": "representative_peak",
        "seed_application": "hint",
        "notes": "representative peak hint를 lean combo에 결합",
    },
    {
        "category": "settings",
        "name": "set_kwon_threshold_hint_state_phi",
        "cut_profile": "combo_barrier_state_phi_mass",
        "seed_strategy": "kwon_threshold_mean",
        "seed_application": "hint",
        "notes": "Kwon threshold hint를 lean combo에 결합",
    },
]

BRANCH_AND_CUT_EXPERIMENTS = [
    {
        "category": "branch_cut",
        "name": "bc_server_branch",
        "cut_profile": "combo_barrier_state_phi_mass",
        "branch_scheme": "server_first",
        "notes": "서버 활성화 계열 변수 우선 branching",
    },
    {
        "category": "branch_cut",
        "name": "bc_budget_branch",
        "cut_profile": "combo_barrier_state_phi_mass",
        "branch_scheme": "budget_first",
        "notes": "eta, delta, phi, gamma 우선 branching",
    },
    {
        "category": "branch_cut",
        "name": "bc_general_cover_usercuts",
        "cut_profile": "combo_barrier_state_phi_mass",
        "dynamic_general_cover": True,
        "notes": "fractional node에서 general minimal cover user cut 분리",
    },
    {
        "category": "branch_cut",
        "name": "bc_eta_usercuts",
        "cut_profile": "combo_barrier_state_phi_mass",
        "dynamic_eta_cover": True,
        "notes": "fractional node에서 eta fixed cover user cut 분리",
    },
    {
        "category": "branch_cut",
        "name": "bc_spot_usercuts",
        "cut_profile": "combo_barrier_state_phi_mass",
        "dynamic_spot_bridge": True,
        "notes": "spot bridge / delta-gamma cut을 node violation 시에만 분리",
    },
    {
        "category": "branch_cut",
        "name": "bc_budget_general_cover_combo",
        "cut_profile": "combo_barrier_state_phi_mass",
        "branch_scheme": "budget_first",
        "dynamic_general_cover": True,
        "notes": "budget-first branching과 general cover user cut 결합",
    },
]

COLUMN_GENERATION_EXPERIMENTS = [
    {
        "category": "column_generation",
        "name": "cg_mean_rmp_start",
        "cut_profile": "combo_barrier_state_phi_mass",
        "seed_application": "both",
        "cg_profile": {
            "anchor_strategy": "representative_mean",
            "pool_strategies": ["representative_mean", "greedy_rule_mean"],
            "surrogate_mode": "mean",
            "include_constant_paths": True,
            "pricing_rounds": 0,
            "migration_weight": 8.0,
        },
        "notes": "mean surrogate 기반 restricted master start",
    },
    {
        "category": "column_generation",
        "name": "cg_peak_rmp_start",
        "cut_profile": "combo_barrier_state_phi_mass",
        "seed_application": "both",
        "cg_profile": {
            "anchor_strategy": "representative_peak",
            "pool_strategies": ["representative_peak", "representative_mean"],
            "surrogate_mode": "peak",
            "include_constant_paths": True,
            "pricing_rounds": 0,
            "migration_weight": 8.0,
        },
        "notes": "peak surrogate 기반 restricted master start",
    },
    {
        "category": "column_generation",
        "name": "cg_mixed_rmp_start",
        "cut_profile": "combo_barrier_state_phi_mass",
        "seed_application": "both",
        "cg_profile": {
            "anchor_strategy": "representative_peak",
            "pool_strategies": [
                "representative_mean",
                "representative_peak",
                "rolling_horizon_mean",
                "kwon_threshold_mean",
            ],
            "surrogate_mode": "mean",
            "include_constant_paths": True,
            "pricing_rounds": 0,
            "migration_weight": 8.0,
        },
        "notes": "mean/peak/rolling/threshold path pool을 섞은 restricted master",
    },
    {
        "category": "column_generation",
        "name": "cg_priced_rmp_start",
        "cut_profile": "combo_barrier_state_phi_mass",
        "seed_application": "both",
        "cg_profile": {
            "anchor_strategy": "representative_peak",
            "pool_strategies": [
                "representative_mean",
                "representative_peak",
                "rolling_horizon_mean",
                "kwon_threshold_mean",
            ],
            "surrogate_mode": "mean",
            "include_constant_paths": True,
            "pricing_rounds": 2,
            "migration_weight": 8.0,
        },
        "notes": "heuristic pricing으로 path column을 두 차례 추가하는 start",
    },
]

ALL_EXPERIMENTS = SETTINGS_EXPERIMENTS + BRANCH_AND_CUT_EXPERIMENTS + COLUMN_GENERATION_EXPERIMENTS


def parse_args():
    parser = argparse.ArgumentParser(description="12VM benchmark for promising settings, branch-and-cut, and column-generation-like starts")
    parser.add_argument("--source-csv", type=Path, default=SOURCE_CSV)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario-seed", type=int, default=42)
    parser.add_argument("--scenario-count", type=int, default=10)
    parser.add_argument("--time-limit", type=int, default=10800)
    parser.add_argument("--seed-time-limit", type=int, default=1200)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--strategy-threads", type=int, default=4)
    parser.add_argument("--profiles", nargs="*", default=None)
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


def set_branch_priorities(variables, scheme):
    if not scheme:
        return {}

    schemes = {
        "server_first": {
            "u_used": 100,
            "u": 90,
            "y": 80,
            "x": 60,
            "z": 50,
            "m": 40,
            "eta": 20,
            "delta": 20,
            "phi": 10,
            "gamma": 10,
        },
        "budget_first": {
            "delta": 100,
            "eta": 100,
            "phi": 80,
            "gamma": 80,
            "u_used": 60,
            "u": 50,
            "y": 40,
            "x": 20,
            "z": 20,
            "m": 10,
        },
    }
    priority_map = schemes[scheme]
    applied = {}
    for variable_name, priority in priority_map.items():
        if variable_name not in variables:
            continue
        for _, var in variables[variable_name].items():
            var.BranchPriority = int(priority)
        applied[variable_name] = int(priority)
    return applied


def prepare_general_cover_rows(data, include_types=("od", "sp", "bj")):
    rows = []
    for time_value in data["times"]:
        for scenario_name in data["scenarios"]:
            cover_items = build_greedy_minimal_cover(
                data=data,
                time_value=time_value,
                scenario_name=scenario_name,
                capacity=data["capacity"],
                include_types=include_types,
            )
            if cover_items:
                rows.append((time_value, scenario_name, tuple(cover_items)))
    return rows


def prepare_eta_cover_rows(data):
    rows = []
    for time_value in data["times"]:
        for scenario_name in data["scenarios"]:
            cover_items = build_greedy_minimal_cover(
                data=data,
                time_value=time_value,
                scenario_name=scenario_name,
                capacity=data["capacity"],
                include_types=("od", "bj"),
            )
            if cover_items:
                rows.append((time_value, scenario_name, tuple(cover_items)))
    return rows


def prepare_spot_bridge_rows(data):
    rows = []
    for workload_id in data["spot_ids"]:
        for server in data["servers"]:
            for time_value in data["spot_active"][workload_id]:
                for scenario_name in data["scenarios"]:
                    rows.append((workload_id, server, time_value, scenario_name))
    return rows


def build_bc_callback(data, variables, config):
    general_cover_rows = prepare_general_cover_rows(data) if config.get("dynamic_general_cover") else []
    eta_rows = prepare_eta_cover_rows(data) if config.get("dynamic_eta_cover") else []
    spot_rows = prepare_spot_bridge_rows(data) if config.get("dynamic_spot_bridge") else []
    max_cuts_per_node = int(config.get("max_cuts_per_node", 24))
    tolerance = float(config.get("cut_tolerance", 1e-5))
    cut_counter = Counter()

    u = variables["u"]
    phi = variables["phi"]
    eta = variables["eta"]
    gamma = variables["gamma"]
    delta = variables["delta"]
    y = variables["y"]
    a = variables["a"]

    def callback(model, where):
        if where != GRB.Callback.MIPNODE:
            return
        node_status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
        if node_status != GRB.OPTIMAL:
            return

        cuts_added = 0

        for time_value, scenario_name, cover_items in general_cover_rows:
            if cuts_added >= max_cuts_per_node:
                break
            for server in data["servers"]:
                if model.cbGetNodeRel(u[server, time_value]) <= 1e-6:
                    continue
                lhs_vars = [build_cover_expr(variables, item, server, time_value, scenario_name) for item in cover_items]
                lhs_value = sum(model.cbGetNodeRel(var) for var in lhs_vars)
                rhs_value = len(cover_items) - 1 + model.cbGetNodeRel(phi[server, time_value, scenario_name])
                if lhs_value <= rhs_value + tolerance:
                    continue
                model.cbCut(gp.quicksum(lhs_vars) <= len(cover_items) - 1 + phi[server, time_value, scenario_name])
                cut_counter["general_cover"] += 1
                cuts_added += 1
                if cuts_added >= max_cuts_per_node:
                    break

        for time_value, scenario_name, cover_items in eta_rows:
            if cuts_added >= max_cuts_per_node:
                break
            for server in data["servers"]:
                if model.cbGetNodeRel(u[server, time_value]) <= 1e-6:
                    continue
                lhs_vars = [build_cover_expr(variables, item, server, time_value, scenario_name) for item in cover_items]
                lhs_value = sum(model.cbGetNodeRel(var) for var in lhs_vars)
                rhs_value = len(cover_items) - 1 + model.cbGetNodeRel(eta[server, scenario_name])
                if lhs_value <= rhs_value + tolerance:
                    continue
                model.cbCut(gp.quicksum(lhs_vars) <= len(cover_items) - 1 + eta[server, scenario_name])
                cut_counter["eta_cover_fixed"] += 1
                cuts_added += 1
                if cuts_added >= max_cuts_per_node:
                    break

        for workload_id, server, time_value, scenario_name in spot_rows:
            if cuts_added >= max_cuts_per_node:
                break
            if model.cbGetNodeRel(y[workload_id, server]) <= 1e-6:
                continue
            y_value = model.cbGetNodeRel(y[workload_id, server])
            gamma_value = model.cbGetNodeRel(gamma[server, time_value, scenario_name])
            delta_value = model.cbGetNodeRel(delta[workload_id, scenario_name])
            a_value = model.cbGetNodeRel(a[workload_id, server, time_value, scenario_name])

            bridge_rhs = y_value - gamma_value - delta_value
            if bridge_rhs > a_value + tolerance:
                model.cbCut(
                    a[workload_id, server, time_value, scenario_name]
                    >= y[workload_id, server] - gamma[server, time_value, scenario_name] - delta[workload_id, scenario_name]
                )
                cut_counter["spot_bridge"] += 1
                cuts_added += 1
                if cuts_added >= max_cuts_per_node:
                    break

            delta_rhs = y_value + gamma_value - 1.0
            if delta_rhs > delta_value + tolerance and cuts_added < max_cuts_per_node:
                model.cbCut(
                    delta[workload_id, scenario_name]
                    >= y[workload_id, server] + gamma[server, time_value, scenario_name] - 1
                )
                cut_counter["delta_gamma"] += 1
                cuts_added += 1

    return callback, cut_counter, {
        "general_cover_candidates": len(general_cover_rows),
        "eta_cover_candidates": len(eta_rows),
        "spot_bridge_candidates": len(spot_rows),
        "max_cuts_per_node": max_cuts_per_node,
    }


def prepare_seed(data, results_dir, config, seed_time_limit, strategy_threads):
    if config.get("seed_strategy"):
        seed_start = time.perf_counter()
        start_values, metadata = build_start_strategy(
            strategy_name=config["seed_strategy"],
            data=data,
            results_dir=results_dir / "seed_generation",
            cut_profile=config.get("seed_cut_profile", config["cut_profile"]),
            strategy_time_limit=seed_time_limit,
            strategy_threads=strategy_threads,
        )
        runtime_seconds = time.perf_counter() - seed_start
        return start_values, metadata, runtime_seconds

    if config.get("cg_profile"):
        seed_start = time.perf_counter()
        start_values, metadata = build_column_generation_start(
            data=data,
            results_dir=results_dir / "seed_generation",
            cut_profile=config.get("seed_cut_profile", config["cut_profile"]),
            profile_name=config["name"],
            config=config["cg_profile"],
            time_limit=seed_time_limit,
            strategy_threads=strategy_threads,
        )
        runtime_seconds = time.perf_counter() - seed_start
        return start_values, metadata, runtime_seconds

    return None, None, 0.0


def solve_single_experiment(task):
    (
        case_name,
        instance_dir,
        config,
        total_time_limit,
        seed_time_limit,
        mip_gap,
        threads,
        strategy_threads,
    ) = task

    results_dir = RUNS_GROUP_DIR / config["category"] / case_name / config["name"]
    results_dir.mkdir(parents=True, exist_ok=True)
    data = load_instance_data(instance_dir)

    start_values, seed_metadata, seed_runtime = prepare_seed(
        data=data,
        results_dir=results_dir,
        config=config,
        seed_time_limit=seed_time_limit,
        strategy_threads=strategy_threads,
    )

    solve_time_limit = max(1.0, float(total_time_limit) - float(seed_runtime))

    model, variables = build_model(
        data=data,
        results_dir=results_dir,
        time_limit=solve_time_limit,
        mip_gap=mip_gap,
        log_name="solver.log",
        threads=threads,
        server_limit=None,
        no_rel_heur_time=0.0,
        cut_profile=config["cut_profile"],
    )

    branch_priority_map = set_branch_priorities(variables, config.get("branch_scheme"))
    apply_fixed_values(variables, None)

    seed_application = config.get("seed_application")
    if seed_application in {"start", "both"}:
        apply_start_values(variables, start_values)
    if seed_application in {"hint", "both"}:
        apply_var_hints(variables, start_values, hint_priority=20)

    callback = None
    callback_metadata = None
    callback_cut_counter = Counter()
    if any(config.get(flag) for flag in ("dynamic_general_cover", "dynamic_eta_cover", "dynamic_spot_bridge")):
        model.setParam("PreCrush", 1)
        callback, callback_cut_counter, callback_metadata = build_bc_callback(data, variables, config)

    solve_start = time.perf_counter()
    if callback is None:
        model.optimize()
    else:
        model.optimize(callback)
    solve_runtime = time.perf_counter() - solve_start

    summary = build_summary(model, data)
    summary["runtime_seconds"] = float(seed_runtime) + float(model.Runtime)
    summary["wall_clock_solve_seconds"] = float(solve_runtime)
    summary["seed_runtime_seconds"] = float(seed_runtime)
    summary["seed_time_limit_seconds"] = float(seed_time_limit) if seed_application else 0.0
    summary["solve_time_limit_seconds"] = float(solve_time_limit)
    summary["total_budget_seconds"] = float(total_time_limit)
    summary["experiment_name"] = config["name"]
    summary["experiment_category"] = config["category"]
    summary["experiment_notes"] = config.get("notes")
    summary["branch_scheme"] = config.get("branch_scheme")
    summary["branch_priorities"] = branch_priority_map
    summary["seed_application"] = seed_application
    summary["seed_metadata"] = seed_metadata
    summary["callback_metadata"] = callback_metadata
    summary["callback_cut_counts"] = dict(callback_cut_counter)

    outputs = None
    if model.SolCount > 0:
        summary_updates, outputs = extract_solution_tables(data, variables)
        summary.update(summary_updates)

    if outputs is not None:
        write_solution_outputs(results_dir, summary, outputs)
    else:
        write_summary_only(results_dir, summary)

    row = {
        "category": config["category"],
        "profile": config["name"],
        "instance_name": case_name,
        "cut_profile": config["cut_profile"],
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
        "seed_runtime_seconds": summary.get("seed_runtime_seconds"),
        "solve_time_limit_seconds": summary.get("solve_time_limit_seconds"),
        "branch_scheme": config.get("branch_scheme"),
        "seed_application": seed_application,
        "dynamic_general_cover": bool(config.get("dynamic_general_cover")),
        "dynamic_eta_cover": bool(config.get("dynamic_eta_cover")),
        "dynamic_spot_bridge": bool(config.get("dynamic_spot_bridge")),
        "branch_priorities": json.dumps(branch_priority_map, ensure_ascii=False, sort_keys=True),
        "seed_metadata": json.dumps(seed_metadata, ensure_ascii=False),
        "callback_metadata": json.dumps(callback_metadata, ensure_ascii=False) if callback_metadata else "",
        "callback_cut_counts": json.dumps(dict(callback_cut_counter), ensure_ascii=False, sort_keys=True),
        "results_dir": str(results_dir),
        "notes": config.get("notes", ""),
    }
    return row


def build_task_rows(case_name, instance_dir, total_time_limit, seed_time_limit, mip_gap, threads, strategy_threads, profiles=None):
    selected_names = set(profiles or [])
    selected_configs = [config for config in ALL_EXPERIMENTS if not selected_names or config["name"] in selected_names]
    return [
        (
            case_name,
            instance_dir,
            config,
            total_time_limit,
            seed_time_limit,
            mip_gap,
            threads,
            strategy_threads,
        )
        for config in selected_configs
    ]


def write_experiment_catalog(output_path):
    catalog = pd.DataFrame(ALL_EXPERIMENTS)
    catalog.to_csv(output_path, index=False)


def main():
    args = parse_args()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_GROUP_DIR.mkdir(parents=True, exist_ok=True)

    case_name, instance_dir = ensure_instance(
        source_csv=args.source_csv,
        seed=args.seed,
        scenario_seed=args.scenario_seed,
        scenario_count=args.scenario_count,
    )

    write_experiment_catalog(ANALYSIS_DIR / "experiment_catalog.csv")
    tasks = build_task_rows(
        case_name=case_name,
        instance_dir=instance_dir,
        total_time_limit=args.time_limit,
        seed_time_limit=args.seed_time_limit,
        mip_gap=args.mip_gap,
        threads=args.threads,
        strategy_threads=args.strategy_threads,
        profiles=args.profiles,
    )

    rows = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        future_map = {executor.submit(solve_single_experiment, task): task[2]["name"] for task in tasks}
        for future in as_completed(future_map):
            profile_name = future_map[future]
            row = future.result()
            rows.append(row)
            summary_df = pd.DataFrame(rows).sort_values(["category", "profile"]).reset_index(drop=True)
            summary_df.to_csv(ANALYSIS_DIR / "live_summary.csv", index=False)
            gap_text = ""
            if row["mip_gap"] is not None:
                gap_text = f", gap={float(row['mip_gap']) * 100.0:.2f}%"
            print(
                f"[done] {row['category']} / {profile_name}: "
                f"status={row['status_name']}, runtime={float(row['runtime_seconds']):.1f}s{gap_text}"
            )

    summary_df = pd.DataFrame(rows).sort_values(["category", "profile"]).reset_index(drop=True)
    summary_df.to_csv(ANALYSIS_DIR / "final_summary.csv", index=False)


if __name__ == "__main__":
    main()
