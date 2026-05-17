import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import gurobipy as gp
import pandas as pd
from gurobipy import GRB


EXPERIMENT_DIR = Path(__file__).resolve().parent
EXPERIMENT_NAME = EXPERIMENT_DIR.name
REPO_ROOT = EXPERIMENT_DIR.parents[1]
CUT_EXPERIMENT_DIR = REPO_ROOT / "experiments" / "2604-cut-experiment"

if str(CUT_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(CUT_EXPERIMENT_DIR))

from build_dataset import build_instance  # noqa: E402
from run_model import (  # noqa: E402
    apply_fixed_values,
    apply_start_values,
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


SOURCE_CSV = REPO_ROOT / "data" / "processed" / "2601-initial-toy-model" / "sample_vm_data.csv"
DATA_ROOT = REPO_ROOT / "data" / "processed" / EXPERIMENT_NAME
ANALYSIS_DIR = EXPERIMENT_DIR / "results" / "analysis" / "chance_2sp_toy_12vm_branch_cut_round2_t10800"
RUNS_GROUP_DIR = EXPERIMENT_DIR / "results" / "runs" / ANALYSIS_DIR.name

ENERGY_IDLE = 100.0
ENERGY_CPU = 300.0
ENERGY_MIGRATION = 50.0
CASE_CONFIG = {"case": "od_sp_bj", "label": "OD + SP + BJ", "on_demand": 4, "spot": 4, "batch": 4}


BENCHMARK_PROFILES = [
    {
        "name": "bc00_control_state_phi",
        "cut_profile": "state_link_phi_mass",
        "notes": "동적 user cut 없이 state-link + phi-mass만 사용하는 기준선",
    },
    {
        "name": "bc01_static_state_phi_eta_general",
        "cut_profile": "state_phi_eta_cover_general",
        "notes": "state-link + phi-mass 위에 eta general cover를 정적으로 추가",
    },
    {
        "name": "bc02_static_state_phi_eta_fixed",
        "cut_profile": "state_phi_eta_cover_fixed",
        "notes": "state-link + phi-mass 위에 eta fixed cover를 정적으로 추가",
    },
    {
        "name": "bc03_static_state_phi_delta_mass",
        "cut_profile": "state_phi_delta_mass",
        "notes": "state-link + phi-mass 위에 delta-mass linking을 정적으로 추가",
    },
    {
        "name": "bc04_static_state_phi_aggregate",
        "cut_profile": "state_phi_aggregate_fixed_load",
        "notes": "state-link + phi-mass 위에 aggregate fixed-load 컷을 정적으로 추가",
    },
    {
        "name": "bc05_branch_server_first",
        "cut_profile": "state_link_phi_mass",
        "branch_scheme": "server_first",
        "notes": "컷은 고정하고 server-first branching priority만 적용",
    },
    {
        "name": "bc06_root_eta_lifted",
        "cut_profile": "state_link_phi_mass",
        "families": ["eta_lifted"],
        "schedule_mode": "root_only",
        "root_barrier": True,
        "root_pass_limit": 2,
        "stop_root_on_zero_cut": True,
        "notes": "1-step lifted eta cover를 root에서만 분리",
    },
    {
        "name": "bc07_periodic_hybrid_server",
        "cut_profile": "state_link_phi_mass",
        "families": ["eta_general", "eta_lifted", "eta_mass", "eta_mass_window"],
        "schedule_mode": "periodic",
        "window_size": 2,
        "branch_scheme": "server_first",
        "root_barrier": True,
        "periodic_every_n_nodes": 250,
        "root_pass_limit": 3,
        "stop_root_on_zero_cut": True,
        "max_cuts_per_call": 24,
        "persist_cuts": True,
        "max_cut_age": 3,
        "notes": "server-first + lifted / projected cover를 root와 periodic node에서 분리",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="12VM branch-and-cut 2차 벤치마크를 실행합니다.")
    parser.add_argument("--source-csv", type=Path, default=SOURCE_CSV)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario-seed", type=int, default=42)
    parser.add_argument("--scenario-count", type=int, default=10)
    parser.add_argument("--time-limit", type=int, default=10800)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
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

    priority_map = {
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
    }
    applied = {}
    if scheme != "server_first":
        return applied
    for variable_name, priority in priority_map.items():
        if variable_name not in variables:
            continue
        for _, var in variables[variable_name].items():
            var.BranchPriority = int(priority)
        applied[variable_name] = int(priority)
    return applied


def make_item_key(item):
    return item[0], item[1]


def build_simple_lifted_items(data, time_value, scenario_name, cover_items, include_types=("od", "sp", "bj")):
    if not cover_items:
        return ()

    total_cover = sum(item[2] for item in cover_items)
    largest_cover = max(item[2] for item in cover_items)
    threshold = float(data["capacity"]) - (total_cover - largest_cover)
    cover_keys = {make_item_key(item) for item in cover_items}
    lifted_items = []

    for item in build_cover_candidates(data, time_value, scenario_name, threshold=0.0, include_types=include_types):
        if make_item_key(item) in cover_keys:
            continue
        if item[2] > threshold + 1e-9:
            lifted_items.append(item)
    return tuple(lifted_items)


def prepare_cover_row_map(data, include_types=("od", "sp", "bj")):
    row_map = {}
    for time_value in data["times"]:
        for scenario_name in data["scenarios"]:
            cover_items = build_greedy_minimal_cover(
                data=data,
                time_value=time_value,
                scenario_name=scenario_name,
                capacity=data["capacity"],
                include_types=include_types,
            )
            if not cover_items:
                continue
            row_map[(time_value, scenario_name)] = {
                "time": time_value,
                "scenario": scenario_name,
                "cover": tuple(cover_items),
                "lifted": build_simple_lifted_items(
                    data=data,
                    time_value=time_value,
                    scenario_name=scenario_name,
                    cover_items=cover_items,
                    include_types=include_types,
                ),
            }
    return row_map


def prepare_window_rows(data, row_map, window_size):
    rows = []
    times = list(data["times"])
    for scenario_name in data["scenarios"]:
        for start_index in range(0, len(times) - window_size + 1):
            time_slice = times[start_index : start_index + window_size]
            entries = []
            for time_value in time_slice:
                row = row_map.get((time_value, scenario_name))
                if row is None:
                    entries = []
                    break
                entries.append((time_value, row["cover"]))
            if entries:
                rows.append(
                    {
                        "scenario": scenario_name,
                        "entries": tuple(entries),
                        "window_size": window_size,
                        "times": tuple(time_slice),
                    }
                )
    return rows


def prepare_mass_rows(data, row_map):
    rows = []
    scenario_prob = data["scenario_prob"]
    scale = 1.0 / min(scenario_prob.values())

    for time_value in data["times"]:
        scenario_entries = []
        rhs_coeff = 0.0
        for scenario_name in data["scenarios"]:
            row = row_map.get((time_value, scenario_name))
            if row is None:
                continue
            scenario_entries.append((scenario_name, row["cover"]))
            rhs_coeff += scenario_prob[scenario_name] * (len(row["cover"]) - 1)
        if scenario_entries:
            rows.append(
                {
                    "time": time_value,
                    "entries": tuple(scenario_entries),
                    "rhs_coeff": rhs_coeff,
                    "scale": scale,
                }
            )
    return rows


def prepare_mass_window_rows(data, row_map, window_size):
    rows = []
    scenario_prob = data["scenario_prob"]
    scale = 1.0 / min(scenario_prob.values())
    times = list(data["times"])

    for start_index in range(0, len(times) - window_size + 1):
        time_slice = times[start_index : start_index + window_size]
        scenario_entries = []
        rhs_coeff_by_time = defaultdict(float)
        for scenario_name in data["scenarios"]:
            scenario_rows = []
            valid = True
            for time_value in time_slice:
                row = row_map.get((time_value, scenario_name))
                if row is None:
                    valid = False
                    break
                scenario_rows.append((time_value, row["cover"]))
                rhs_coeff_by_time[time_value] += scenario_prob[scenario_name] * (len(row["cover"]) - 1)
            if valid:
                scenario_entries.append((scenario_name, tuple(scenario_rows)))
        if scenario_entries:
            rows.append(
                {
                    "times": tuple(time_slice),
                    "entries": tuple(scenario_entries),
                    "window_size": window_size,
                    "rhs_coeff_by_time": dict(rhs_coeff_by_time),
                    "scale": scale,
                }
            )
    return rows


def should_process_node(model, callback_state, config):
    node_status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
    if node_status != GRB.OPTIMAL:
        return False, None, None

    node_count = int(model.cbGet(GRB.Callback.MIPNODE_NODCNT))
    mode = config.get("schedule_mode", "root_only")
    root_pass_limit = int(config.get("root_pass_limit", 6))

    if node_count == 0:
        if callback_state["stop_root_separation"]:
            return False, node_count, "root"
        if callback_state["root_calls"] >= root_pass_limit:
            return False, node_count, "root"
        callback_state["root_calls"] += 1
        callback_state["last_processed_node"] = node_count
        return True, node_count, "root"

    if mode != "periodic":
        return False, node_count, "tree"

    every_n_nodes = int(config.get("periodic_every_n_nodes", 250))
    if node_count == callback_state["last_processed_node"]:
        return False, node_count, "tree"
    if node_count % every_n_nodes != 0:
        return False, node_count, "tree"

    callback_state["periodic_calls"] += 1
    callback_state["last_processed_node"] = node_count
    return True, node_count, "tree"


def build_bc_callback(data, variables, config):
    row_map = prepare_cover_row_map(data)
    family_rows = {
        "phi_general": tuple(row_map.values()),
        "eta_general": tuple(row_map.values()),
        "eta_lifted": tuple(row_map.values()),
    }
    window_size = int(config.get("window_size", 2))
    family_rows["eta_window"] = tuple(prepare_window_rows(data, row_map, window_size))
    family_rows["eta_mass"] = tuple(prepare_mass_rows(data, row_map))
    family_rows["eta_mass_window"] = tuple(prepare_mass_window_rows(data, row_map, window_size))

    callback_state = {
        "calls": 0,
        "processed_calls": 0,
        "root_calls": 0,
        "periodic_calls": 0,
        "last_processed_node": None,
        "stop_root_separation": False,
        "active_pool": set(),
        "pool_miss_count": defaultdict(int),
        "pool_peak": 0,
    }
    cut_counter = Counter()

    servers = data["servers"]
    scenario_prob = data["scenario_prob"]
    epsilon_od = float(data["epsilon_od"])
    families = tuple(config.get("families", ()))
    max_cuts_per_call = int(config.get("max_cuts_per_call", 24))
    cut_tolerance = float(config.get("cut_tolerance", 1e-5))
    persist_cuts = bool(config.get("persist_cuts", False))
    max_cut_age = int(config.get("max_cut_age", 3))

    u = variables["u"]
    phi = variables["phi"]
    eta = variables["eta"]
    stop_root_on_zero_cut = bool(config.get("stop_root_on_zero_cut", False))

    def compute_order(family_name, rows):
        if not persist_cuts:
            return list(enumerate(rows))
        active = []
        inactive = []
        for row_index, row in enumerate(rows):
            if (family_name, row_index) in callback_state["active_pool"]:
                active.append((row_index, row))
            else:
                inactive.append((row_index, row))
        return active + inactive

    def update_pool(family_name, row_index, violated):
        key = (family_name, row_index)
        if not persist_cuts:
            return
        if violated:
            callback_state["active_pool"].add(key)
            callback_state["pool_miss_count"][key] = 0
            callback_state["pool_peak"] = max(callback_state["pool_peak"], len(callback_state["active_pool"]))
        elif key in callback_state["active_pool"]:
            callback_state["pool_miss_count"][key] += 1
            if callback_state["pool_miss_count"][key] > max_cut_age:
                callback_state["active_pool"].discard(key)
                callback_state["pool_miss_count"].pop(key, None)

    def callback(model, where):
        if where != GRB.Callback.MIPNODE:
            return

        callback_state["calls"] += 1
        should_process, node_count, phase = should_process_node(model, callback_state, config)
        if not should_process:
            return

        callback_state["processed_calls"] += 1
        cuts_added = 0
        family_hits = Counter()
        u_rel = {
            (server, time_value): model.cbGetNodeRel(u[server, time_value])
            for server in servers
            for time_value in data["times"]
        }

        for family_name in families:
            if cuts_added >= max_cuts_per_call:
                break
            rows = family_rows[family_name]
            for row_index, row in compute_order(family_name, rows):
                if cuts_added >= max_cuts_per_call:
                    break
                violated = False

                if family_name == "phi_general":
                    time_value = row["time"]
                    scenario_name = row["scenario"]
                    for server in servers:
                        if u_rel[server, time_value] <= 1e-6:
                            continue
                        lhs_vars = [build_cover_expr(variables, item, server, time_value, scenario_name) for item in row["cover"]]
                        lhs_value = sum(model.cbGetNodeRel(var) for var in lhs_vars)
                        rhs_value = (len(row["cover"]) - 1) * u_rel[server, time_value] + model.cbGetNodeRel(
                            phi[server, time_value, scenario_name]
                        )
                        if lhs_value - rhs_value <= cut_tolerance:
                            continue
                        model.cbCut(
                            gp.quicksum(lhs_vars)
                            <= (len(row["cover"]) - 1) * u[server, time_value] + phi[server, time_value, scenario_name]
                        )
                        cut_counter["phi_general"] += 1
                        family_hits["phi_general"] += 1
                        cuts_added += 1
                        violated = True
                        break

                elif family_name == "eta_general":
                    time_value = row["time"]
                    scenario_name = row["scenario"]
                    for server in servers:
                        if u_rel[server, time_value] <= 1e-6:
                            continue
                        lhs_vars = [build_cover_expr(variables, item, server, time_value, scenario_name) for item in row["cover"]]
                        lhs_value = sum(model.cbGetNodeRel(var) for var in lhs_vars)
                        rhs_value = (len(row["cover"]) - 1) * u_rel[server, time_value] + model.cbGetNodeRel(
                            eta[server, scenario_name]
                        )
                        if lhs_value - rhs_value <= cut_tolerance:
                            continue
                        model.cbCut(
                            gp.quicksum(lhs_vars)
                            <= (len(row["cover"]) - 1) * u[server, time_value] + eta[server, scenario_name]
                        )
                        cut_counter["eta_general"] += 1
                        family_hits["eta_general"] += 1
                        cuts_added += 1
                        violated = True
                        break

                elif family_name == "eta_lifted":
                    time_value = row["time"]
                    scenario_name = row["scenario"]
                    for server in servers:
                        if u_rel[server, time_value] <= 1e-6:
                            continue
                        lhs_vars = [build_cover_expr(variables, item, server, time_value, scenario_name) for item in row["cover"]]
                        lhs_vars.extend(
                            build_cover_expr(variables, item, server, time_value, scenario_name) for item in row["lifted"]
                        )
                        lhs_value = sum(model.cbGetNodeRel(var) for var in lhs_vars)
                        rhs_value = (len(row["cover"]) - 1) * u_rel[server, time_value] + model.cbGetNodeRel(
                            eta[server, scenario_name]
                        )
                        if lhs_value - rhs_value <= cut_tolerance:
                            continue
                        model.cbCut(
                            gp.quicksum(lhs_vars)
                            <= (len(row["cover"]) - 1) * u[server, time_value] + eta[server, scenario_name]
                        )
                        cut_counter["eta_lifted"] += 1
                        family_hits["eta_lifted"] += 1
                        cuts_added += 1
                        violated = True
                        break

                elif family_name == "eta_window":
                    scenario_name = row["scenario"]
                    for server in servers:
                        if sum(u_rel[server, time_value] for time_value in row["times"]) <= 1e-6:
                            continue
                        lhs_vars = []
                        for time_value, cover_items in row["entries"]:
                            lhs_vars.extend(
                                build_cover_expr(variables, item, server, time_value, scenario_name) for item in cover_items
                            )
                        lhs_value = sum(model.cbGetNodeRel(var) for var in lhs_vars)
                        rhs_value = row["window_size"] * model.cbGetNodeRel(eta[server, scenario_name])
                        for time_value, cover_items in row["entries"]:
                            rhs_value += (len(cover_items) - 1) * u_rel[server, time_value]
                        if lhs_value - rhs_value <= cut_tolerance:
                            continue
                        rhs_expr = row["window_size"] * eta[server, scenario_name]
                        for time_value, cover_items in row["entries"]:
                            rhs_expr += (len(cover_items) - 1) * u[server, time_value]
                        model.cbCut(gp.quicksum(lhs_vars) <= rhs_expr)
                        cut_counter["eta_window"] += 1
                        family_hits["eta_window"] += 1
                        cuts_added += 1
                        violated = True
                        break

                elif family_name == "eta_mass":
                    time_value = row["time"]
                    for server in servers:
                        if u_rel[server, time_value] <= 1e-6:
                            continue
                        lhs_terms = []
                        lhs_value = 0.0
                        for scenario_name, cover_items in row["entries"]:
                            weight = row["scale"] * scenario_prob[scenario_name]
                            scenario_vars = [
                                build_cover_expr(variables, item, server, time_value, scenario_name) for item in cover_items
                            ]
                            lhs_terms.extend((weight, var) for var in scenario_vars)
                            lhs_value += weight * sum(model.cbGetNodeRel(var) for var in scenario_vars)
                        rhs_value = row["scale"] * row["rhs_coeff"] * u_rel[server, time_value] + row["scale"] * epsilon_od
                        if lhs_value - rhs_value <= cut_tolerance:
                            continue
                        model.cbCut(
                            gp.quicksum(weight * var for weight, var in lhs_terms)
                            <= row["scale"] * row["rhs_coeff"] * u[server, time_value] + row["scale"] * epsilon_od
                        )
                        cut_counter["eta_mass"] += 1
                        family_hits["eta_mass"] += 1
                        cuts_added += 1
                        violated = True
                        break

                elif family_name == "eta_mass_window":
                    for server in servers:
                        if sum(u_rel[server, time_value] for time_value in row["times"]) <= 1e-6:
                            continue
                        lhs_terms = []
                        lhs_value = 0.0
                        for scenario_name, scenario_rows in row["entries"]:
                            weight = row["scale"] * scenario_prob[scenario_name]
                            for time_value, cover_items in scenario_rows:
                                scenario_vars = [
                                    build_cover_expr(variables, item, server, time_value, scenario_name)
                                    for item in cover_items
                                ]
                                lhs_terms.extend((weight, var) for var in scenario_vars)
                                lhs_value += weight * sum(model.cbGetNodeRel(var) for var in scenario_vars)
                        rhs_value = row["scale"] * row["window_size"] * epsilon_od
                        for time_value in row["times"]:
                            rhs_value += row["scale"] * row["rhs_coeff_by_time"][time_value] * u_rel[server, time_value]
                        if lhs_value - rhs_value <= cut_tolerance:
                            continue
                        rhs_expr = row["scale"] * row["window_size"] * epsilon_od
                        for time_value in row["times"]:
                            rhs_expr += row["scale"] * row["rhs_coeff_by_time"][time_value] * u[server, time_value]
                        model.cbCut(gp.quicksum(weight * var for weight, var in lhs_terms) <= rhs_expr)
                        cut_counter["eta_mass_window"] += 1
                        family_hits["eta_mass_window"] += 1
                        cuts_added += 1
                        violated = True
                        break

                update_pool(family_name, row_index, violated)

        callback_state["last_family_hits"] = dict(family_hits)
        callback_state["last_phase"] = phase
        callback_state["last_node_count"] = node_count
        if phase == "root" and cuts_added == 0 and stop_root_on_zero_cut:
            callback_state["stop_root_separation"] = True

    metadata = {
        "families": list(families),
        "schedule_mode": config.get("schedule_mode", "root_only"),
        "window_size": int(config.get("window_size", 2)),
        "root_pass_limit": int(config.get("root_pass_limit", 6)),
        "periodic_every_n_nodes": int(config.get("periodic_every_n_nodes", 250)),
        "persist_cuts": bool(config.get("persist_cuts", False)),
        "max_cut_age": int(config.get("max_cut_age", 3)),
        "max_cuts_per_call": max_cuts_per_call,
        "candidate_counts": {
            family_name: len(family_rows.get(family_name, ()))
            for family_name in ("phi_general", "eta_general", "eta_lifted", "eta_window", "eta_mass", "eta_mass_window")
        },
    }
    return callback, cut_counter, callback_state, metadata


def solve_single_experiment(task):
    case_name, instance_dir, config, total_time_limit, mip_gap, threads = task

    results_dir = RUNS_GROUP_DIR / case_name / config["name"]
    results_dir.mkdir(parents=True, exist_ok=True)
    data = load_instance_data(instance_dir)

    model, variables = build_model(
        data=data,
        results_dir=results_dir,
        time_limit=float(total_time_limit),
        mip_gap=mip_gap,
        log_name="solver.log",
        threads=threads,
        server_limit=None,
        no_rel_heur_time=float(config.get("no_rel_heur_time", 0.0)),
        cut_profile=config["cut_profile"],
    )

    branch_priority_map = set_branch_priorities(variables, config.get("branch_scheme"))
    apply_fixed_values(variables, None)
    apply_start_values(variables, None)

    callback = None
    callback_metadata = None
    callback_cut_counter = Counter()
    callback_state = None
    if config.get("families"):
        if config.get("root_barrier"):
            model.setParam("Method", 2)
            model.setParam("Crossover", 0)
        model.setParam("PreCrush", 1)
        callback, callback_cut_counter, callback_state, callback_metadata = build_bc_callback(data, variables, config)

    solve_start = time.perf_counter()
    if callback is None:
        model.optimize()
    else:
        model.optimize(callback)
    solve_runtime = time.perf_counter() - solve_start

    summary = build_summary(model, data)
    summary["runtime_seconds"] = float(model.Runtime)
    summary["wall_clock_solve_seconds"] = float(solve_runtime)
    summary["solve_time_limit_seconds"] = float(total_time_limit)
    summary["experiment_name"] = config["name"]
    summary["experiment_notes"] = config.get("notes", "")
    summary["branch_scheme"] = config.get("branch_scheme")
    summary["branch_priorities"] = branch_priority_map
    summary["callback_metadata"] = callback_metadata
    if callback_state is not None:
        summary["callback_runtime_metadata"] = {
            "calls": callback_state["calls"],
            "processed_calls": callback_state["processed_calls"],
            "root_calls": callback_state["root_calls"],
            "periodic_calls": callback_state["periodic_calls"],
            "stop_root_separation": callback_state["stop_root_separation"],
            "pool_peak": callback_state["pool_peak"],
            "last_phase": callback_state.get("last_phase"),
            "last_node_count": callback_state.get("last_node_count"),
            "last_family_hits": callback_state.get("last_family_hits", {}),
        }
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
        "branch_scheme": config.get("branch_scheme", ""),
        "families": json.dumps(config.get("families", []), ensure_ascii=False),
        "schedule_mode": config.get("schedule_mode", ""),
        "branch_priorities": json.dumps(branch_priority_map, ensure_ascii=False, sort_keys=True),
        "callback_metadata": json.dumps(callback_metadata, ensure_ascii=False) if callback_metadata else "",
        "callback_runtime_metadata": json.dumps(summary.get("callback_runtime_metadata", {}), ensure_ascii=False),
        "callback_cut_counts": json.dumps(dict(callback_cut_counter), ensure_ascii=False, sort_keys=True),
        "results_dir": str(results_dir),
        "notes": config.get("notes", ""),
    }
    return row


def build_task_rows(case_name, instance_dir, total_time_limit, mip_gap, threads, profiles=None):
    selected_names = set(profiles or [])
    selected_configs = [config for config in BENCHMARK_PROFILES if not selected_names or config["name"] in selected_names]
    return [
        (
            case_name,
            instance_dir,
            config,
            total_time_limit,
            mip_gap,
            threads,
        )
        for config in selected_configs
    ]


def write_experiment_catalog(output_path):
    pd.DataFrame(BENCHMARK_PROFILES).to_csv(output_path, index=False)


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
        mip_gap=args.mip_gap,
        threads=args.threads,
        profiles=args.profiles,
    )

    rows = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        future_map = {executor.submit(solve_single_experiment, task): task[2]["name"] for task in tasks}
        for future in as_completed(future_map):
            profile_name = future_map[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "profile": profile_name,
                    "instance_name": case_name,
                    "cut_profile": "",
                    "status_name": "FAILED",
                    "has_solution": False,
                    "runtime_seconds": None,
                    "objective_value": None,
                    "objective_bound": None,
                    "mip_gap": None,
                    "node_count": None,
                    "iter_count": None,
                    "work_units": None,
                    "used_server_count": None,
                    "branch_scheme": "",
                    "families": "",
                    "schedule_mode": "",
                    "branch_priorities": "",
                    "callback_metadata": "",
                    "callback_runtime_metadata": "",
                    "callback_cut_counts": "",
                    "results_dir": "",
                    "notes": f"FAILED: {exc}",
                }
            rows.append(row)
            summary_df = pd.DataFrame(rows).sort_values(["profile"]).reset_index(drop=True)
            summary_df.to_csv(ANALYSIS_DIR / "live_summary.csv", index=False)
            gap_text = ""
            if row["mip_gap"] is not None:
                gap_text = f", gap={float(row['mip_gap']) * 100.0:.2f}%"
            runtime_text = "NA" if row["runtime_seconds"] is None else f"{float(row['runtime_seconds']):.1f}s"
            print(f"[done] {profile_name}: status={row['status_name']}, runtime={runtime_text}{gap_text}")

    summary_df = pd.DataFrame(rows).sort_values(["profile"]).reset_index(drop=True)
    summary_df.to_csv(ANALYSIS_DIR / "final_summary.csv", index=False)


if __name__ == "__main__":
    main()
