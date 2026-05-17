import math
import sys
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
CUT_EXPERIMENT_DIR = REPO_ROOT / "experiments" / "2604-cut-experiment"
DECOMP_EXPERIMENT_DIR = REPO_ROOT / "experiments" / "2604-decomposition-experiment"

if str(CUT_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(CUT_EXPERIMENT_DIR))
if str(DECOMP_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DECOMP_EXPERIMENT_DIR))

from decomposition_starts import (  # noqa: E402
    build_start_strategy,
    finalize_first_stage_start,
    make_aggregated_scenario_data,
    split_equal_time_budget,
)


def path_signature_from_start(data, start_values, workload_id):
    active_times = data["od_active"][workload_id]
    signature = []
    for time_value in active_times:
        chosen_server = next(
            (
                server
                for server in data["servers"]
                if start_values["x"].get((workload_id, server, time_value), 0.0) > 0.5
            ),
            data["servers"][0],
        )
        signature.append(int(chosen_server))
    return tuple(signature)


def constant_server_path(data, workload_id, server):
    return tuple(int(server) for _ in data["od_active"][workload_id])


def migration_count(path_signature):
    if not path_signature:
        return 0
    return sum(
        1 for previous_server, current_server in zip(path_signature[:-1], path_signature[1:]) if previous_server != current_server
    )


def build_fixed_loads(reduced_data, anchor_start):
    scenario_name = reduced_data["scenarios"][0]
    fixed_load = {(server, time_value): 0.0 for server in reduced_data["servers"] for time_value in reduced_data["times"]}
    fixed_occ = {(server, time_value): 0 for server in reduced_data["servers"] for time_value in reduced_data["times"]}

    for workload_id in reduced_data["spot_ids"]:
        assigned_server = next(
            (
                server
                for server in reduced_data["servers"]
                if anchor_start["y"].get((workload_id, server), 0.0) > 0.5
            ),
            None,
        )
        if assigned_server is None:
            continue
        for time_value in reduced_data["spot_active"][workload_id]:
            fixed_load[assigned_server, time_value] += reduced_data["d_sp"][workload_id, time_value, scenario_name]
            fixed_occ[assigned_server, time_value] = 1

    for batch_job_id in reduced_data["batch_ids"]:
        for server in reduced_data["servers"]:
            for time_value in reduced_data["times"]:
                if anchor_start["z"].get((batch_job_id, server, time_value), 0.0) > 0.5:
                    fixed_load[server, time_value] += reduced_data["d_batch"][batch_job_id, scenario_name]
                    fixed_occ[server, time_value] = 1

    return fixed_load, fixed_occ


def build_path_pool(data, reduced_data, start_map, strategy_names, include_constant_paths):
    path_pool = {workload_id: {} for workload_id in data["on_demand_ids"]}

    for strategy_name in strategy_names:
        start_values = start_map.get(strategy_name)
        if start_values is None:
            continue
        for workload_id in data["on_demand_ids"]:
            signature = path_signature_from_start(data, start_values, workload_id)
            path_pool[workload_id][signature] = {
                "source": strategy_name,
                "signature": signature,
                "migrations": migration_count(signature),
            }

    if include_constant_paths:
        for workload_id in data["on_demand_ids"]:
            for server in data["servers"]:
                signature = constant_server_path(data, workload_id, server)
                path_pool[workload_id].setdefault(
                    signature,
                    {
                        "source": f"constant_s{server}",
                        "signature": signature,
                        "migrations": 0,
                    },
                )

    return path_pool


def solve_restricted_master(data, reduced_data, anchor_start, path_pool, migration_weight, time_limit, threads):
    scenario_name = reduced_data["scenarios"][0]
    fixed_load, fixed_occ = build_fixed_loads(reduced_data, anchor_start)

    model = gp.Model("od_path_rmp")
    model.setParam("LogToConsole", 0)
    if time_limit is not None and float(time_limit) > 0.0:
        model.setParam("TimeLimit", float(time_limit))
    if threads is not None:
        model.setParam("Threads", int(threads))

    lambda_index = []
    for workload_id in reduced_data["on_demand_ids"]:
        for path_signature in path_pool[workload_id]:
            lambda_index.append((workload_id, path_signature))

    choose = model.addVars(lambda_index, vtype=GRB.BINARY, name="lambda")
    u = model.addVars(reduced_data["servers"], reduced_data["times"], vtype=GRB.BINARY, name="u")

    model.addConstrs(
        gp.quicksum(choose[workload_id, path_signature] for path_signature in path_pool[workload_id]) == 1
        for workload_id in reduced_data["on_demand_ids"]
    )

    for server in reduced_data["servers"]:
        for time_value in reduced_data["times"]:
            load_expr = fixed_load[server, time_value]
            for workload_id in reduced_data["on_demand_ids"]:
                active_times = reduced_data["od_active"][workload_id]
                if time_value not in active_times:
                    continue
                time_index = active_times.index(time_value)
                demand = reduced_data["d_od"][workload_id, time_value, scenario_name]
                for path_signature in path_pool[workload_id]:
                    if path_signature[time_index] != server:
                        continue
                    load_expr += demand * choose[workload_id, path_signature]

            model.addConstr(load_expr <= reduced_data["capacity"] * u[server, time_value], name=f"cap[{server},{time_value}]")
            if fixed_occ[server, time_value]:
                model.addConstr(u[server, time_value] == 1, name=f"fixed_occ[{server},{time_value}]")

    idle_cost = gp.quicksum(u[server, time_value] for server in reduced_data["servers"] for time_value in reduced_data["times"])
    migration_cost = gp.quicksum(
        path_pool[workload_id][path_signature]["migrations"] * choose[workload_id, path_signature]
        for workload_id, path_signature in lambda_index
    )
    model.setObjective(100.0 * idle_cost + float(migration_weight) * migration_cost, GRB.MINIMIZE)
    model.optimize()

    if model.SolCount <= 0:
        return None, {
            "status_name": "NO_SOLUTION",
            "objective_value": None,
            "selected_paths": {},
            "fixed_load_peak": max(fixed_load.values()) if fixed_load else 0.0,
        }

    selected_paths = {}
    for workload_id in reduced_data["on_demand_ids"]:
        selected_paths[workload_id] = next(
            (
                path_signature
                for path_signature in path_pool[workload_id]
                if choose[workload_id, path_signature].X > 0.5
            ),
            next(iter(path_pool[workload_id])),
        )

    return selected_paths, {
        "status_name": "OPTIMAL" if model.Status == GRB.OPTIMAL else "TIME_LIMIT",
        "objective_value": float(model.ObjVal),
        "selected_path_count": len(selected_paths),
        "fixed_load_peak": max(fixed_load.values()) if fixed_load else 0.0,
    }


def build_penalties_from_paths(reduced_data, anchor_start, selected_paths):
    scenario_name = reduced_data["scenarios"][0]
    fixed_load, _ = build_fixed_loads(reduced_data, anchor_start)
    total_load = dict(fixed_load)

    for workload_id, path_signature in selected_paths.items():
        active_times = reduced_data["od_active"][workload_id]
        for time_index, time_value in enumerate(active_times):
            server = path_signature[time_index]
            total_load[server, time_value] += reduced_data["d_od"][workload_id, time_value, scenario_name]

    penalties = {}
    for server in reduced_data["servers"]:
        for time_value in reduced_data["times"]:
            utilization = total_load[server, time_value] / reduced_data["capacity"]
            penalties[server, time_value] = max(0.0, utilization - 0.80) * 12.0 + 0.02 * server
    return penalties


def generate_priced_path(reduced_data, workload_id, penalties, migration_weight):
    active_times = reduced_data["od_active"][workload_id]
    if not active_times:
        return tuple()

    scenario_name = reduced_data["scenarios"][0]
    servers = reduced_data["servers"]

    dp = {}
    first_time = active_times[0]
    for server in servers:
        demand = reduced_data["d_od"][workload_id, first_time, scenario_name]
        dp[(0, server)] = (
            penalties[server, first_time] * demand + 0.02 * server,
            [server],
        )

    for time_index in range(1, len(active_times)):
        time_value = active_times[time_index]
        next_dp = {}
        demand = reduced_data["d_od"][workload_id, time_value, scenario_name]
        for server in servers:
            best_value = None
            best_path = None
            for previous_server in servers:
                previous_value, previous_path = dp[(time_index - 1, previous_server)]
                candidate_value = (
                    previous_value
                    + penalties[server, time_value] * demand
                    + float(migration_weight) * (1 if server != previous_server else 0)
                )
                if best_value is None or candidate_value < best_value - 1e-9:
                    best_value = candidate_value
                    best_path = previous_path + [server]
            next_dp[(time_index, server)] = (best_value, best_path)
        dp = next_dp

    best_value = None
    best_path = None
    final_index = len(active_times) - 1
    for server in servers:
        candidate_value, candidate_path = dp[(final_index, server)]
        if best_value is None or candidate_value < best_value - 1e-9:
            best_value = candidate_value
            best_path = candidate_path
    return tuple(best_path or [])


def build_start_from_selected_paths(data, anchor_start, selected_paths):
    start_values = {name: {} for name in ("u", "u_used", "x", "m", "y", "z")}
    start_values["y"].update(anchor_start["y"])
    start_values["z"].update(anchor_start["z"])

    x_choice = {}
    for workload_id in data["on_demand_ids"]:
        path_signature = selected_paths[workload_id]
        active_times = data["od_active"][workload_id]
        for time_index, time_value in enumerate(active_times):
            chosen_server = path_signature[time_index]
            x_choice[workload_id, time_value] = chosen_server
            for server in data["servers"]:
                start_values["x"][workload_id, server, time_value] = 1.0 if server == chosen_server else 0.0

    finalize_first_stage_start(data, start_values, x_choice=x_choice)
    return start_values


def build_column_generation_start(
    data,
    results_dir,
    cut_profile,
    profile_name,
    config,
    time_limit,
    strategy_threads,
):
    unique_strategies = []
    for strategy_name in [config["anchor_strategy"], *config["pool_strategies"]]:
        if strategy_name not in unique_strategies:
            unique_strategies.append(strategy_name)

    per_strategy_time = split_equal_time_budget(time_limit, max(1, len(unique_strategies)))
    start_map = {}
    metadata_rows = []
    for strategy_name in unique_strategies:
        start_values, metadata = build_start_strategy(
            strategy_name=strategy_name,
            data=data,
            results_dir=results_dir / "starts" / strategy_name,
            cut_profile=cut_profile,
            strategy_time_limit=per_strategy_time,
            strategy_threads=strategy_threads,
        )
        start_map[strategy_name] = start_values
        metadata_rows.append({"strategy": strategy_name, "has_start": start_values is not None, "metadata": metadata})

    anchor_start = start_map.get(config["anchor_strategy"])
    if anchor_start is None:
        for strategy_name in unique_strategies:
            if start_map.get(strategy_name) is not None:
                anchor_start = start_map[strategy_name]
                break
    if anchor_start is None:
        return None, {
            "strategy": profile_name,
            "status_name": "NO_START",
            "metadata_rows": metadata_rows,
        }

    reduced_data = make_aggregated_scenario_data(data, config["surrogate_mode"])
    path_pool = build_path_pool(
        data=data,
        reduced_data=reduced_data,
        start_map=start_map,
        strategy_names=config["pool_strategies"],
        include_constant_paths=config.get("include_constant_paths", True),
    )

    selected_paths, rmp_summary = solve_restricted_master(
        data=data,
        reduced_data=reduced_data,
        anchor_start=anchor_start,
        path_pool=path_pool,
        migration_weight=config.get("migration_weight", 5.0),
        time_limit=config.get("rmp_time_limit", per_strategy_time),
        threads=strategy_threads,
    )
    if selected_paths is None:
        return anchor_start, {
            "strategy": profile_name,
            "status_name": "FALLBACK_ANCHOR",
            "metadata_rows": metadata_rows,
            "rmp_summary": rmp_summary,
        }

    pricing_rounds = int(config.get("pricing_rounds", 0))
    for _ in range(pricing_rounds):
        penalties = build_penalties_from_paths(reduced_data, anchor_start, selected_paths)
        added_columns = 0
        for workload_id in data["on_demand_ids"]:
            priced_signature = generate_priced_path(
                reduced_data=reduced_data,
                workload_id=workload_id,
                penalties=penalties,
                migration_weight=config.get("migration_weight", 5.0),
            )
            if not priced_signature:
                continue
            if priced_signature in path_pool[workload_id]:
                continue
            path_pool[workload_id][priced_signature] = {
                "source": "heuristic_pricing",
                "signature": priced_signature,
                "migrations": migration_count(priced_signature),
            }
            added_columns += 1

        if added_columns == 0:
            break

        selected_paths, rmp_summary = solve_restricted_master(
            data=data,
            reduced_data=reduced_data,
            anchor_start=anchor_start,
            path_pool=path_pool,
            migration_weight=config.get("migration_weight", 5.0),
            time_limit=config.get("rmp_time_limit", per_strategy_time),
            threads=strategy_threads,
        )
        if selected_paths is None:
            return anchor_start, {
                "strategy": profile_name,
                "status_name": "FALLBACK_ANCHOR_AFTER_PRICING",
                "metadata_rows": metadata_rows,
                "rmp_summary": rmp_summary,
            }

    start_values = build_start_from_selected_paths(data, anchor_start, selected_paths)
    metadata = {
        "strategy": profile_name,
        "status_name": "CG_START_READY",
        "metadata_rows": metadata_rows,
        "rmp_summary": rmp_summary,
        "path_pool_sizes": {workload_id: len(pool) for workload_id, pool in path_pool.items()},
        "anchor_strategy": config["anchor_strategy"],
        "surrogate_mode": config["surrogate_mode"],
        "pricing_rounds": pricing_rounds,
    }
    return start_values, metadata
