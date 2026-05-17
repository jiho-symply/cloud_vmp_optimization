import argparse
import itertools
import json
from pathlib import Path

import gurobipy as gp
import pandas as pd
from gurobipy import GRB

from cut_profiles import CUT_PROFILES


EXPERIMENT_DIR = Path(__file__).resolve().parent
EXPERIMENT_NAME = EXPERIMENT_DIR.name
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DEFAULT_INSTANCE_DIR = REPO_ROOT / "data" / "processed" / EXPERIMENT_NAME / "chance_2sp_toy_30vm_od10_sp10_bj10_sc10_cap4"
def parse_args():
    parser = argparse.ArgumentParser(description="chance-constrained 2SP toy model을 풉니다.")
    parser.add_argument("--instance-dir", type=Path, default=DEFAULT_INSTANCE_DIR)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--time-limit", type=int, default=1800)
    parser.add_argument("--norel-pre-time", type=float, default=0.0)
    parser.add_argument("--alternate-norel-time", type=float, default=0.0)
    parser.add_argument("--alternate-main-time", type=float, default=0.0)
    parser.add_argument("--alternate-max-rounds", type=int, default=6)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--server-limit", type=int, default=None)
    parser.add_argument("--no-rel-heur-time", type=float, default=0.0)
    parser.add_argument("--cut-profile", type=str, default="baseline", choices=sorted(CUT_PROFILES.keys()))
    parser.add_argument("--use-fallback", action="store_true")
    return parser.parse_args()


def status_name(status_code):
    status_names = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
    }
    return status_names.get(status_code, str(status_code))


def resolve_cut_profile(cut_profile):
    if cut_profile not in CUT_PROFILES:
        raise ValueError(f"Unknown cut profile: {cut_profile}")
    return dict(CUT_PROFILES[cut_profile])


def increment_cut_count(cut_stats, key, amount):
    if amount <= 0:
        return
    cut_stats[key] = int(cut_stats.get(key, 0)) + int(amount)


def build_server_time_rhs(data, variables, server, time_value):
    rhs_terms = []
    for workload_id in data["on_demand_ids"]:
        if time_value in data["od_active"][workload_id]:
            rhs_terms.append(variables["x"][workload_id, server, time_value])
    for workload_id in data["spot_ids"]:
        if time_value in data["spot_active"][workload_id]:
            rhs_terms.append(variables["y"][workload_id, server])
    for batch_job_id in data["batch_ids"]:
        rhs_terms.append(variables["z"][batch_job_id, server, time_value])
    return gp.quicksum(rhs_terms)


def build_cover_candidates(data, time_value, scenario_name, threshold, include_types=("od", "sp", "bj")):
    include_types = set(include_types)
    candidates = []
    if "od" in include_types:
        for workload_id in data["on_demand_ids"]:
            if time_value in data["od_active"][workload_id]:
                demand = data["d_od"][workload_id, time_value, scenario_name]
                if demand > threshold:
                    candidates.append(("od", workload_id, float(demand)))
    if "sp" in include_types:
        for workload_id in data["spot_ids"]:
            if time_value in data["spot_active"][workload_id]:
                demand = data["d_sp"][workload_id, time_value, scenario_name]
                if demand > threshold:
                    candidates.append(("sp", workload_id, float(demand)))
    if "bj" in include_types:
        for batch_job_id in data["batch_ids"]:
            demand = data["d_batch"][batch_job_id, scenario_name]
            if demand > threshold:
                candidates.append(("bj", batch_job_id, float(demand)))
    candidates.sort(key=lambda item: item[2], reverse=True)
    return candidates


def build_greedy_minimal_cover(data, time_value, scenario_name, capacity, include_types=("od", "sp", "bj")):
    candidates = build_cover_candidates(
        data=data,
        time_value=time_value,
        scenario_name=scenario_name,
        threshold=0.0,
        include_types=include_types,
    )
    if not candidates:
        return None

    cover = []
    total_demand = 0.0
    for item in candidates:
        cover.append(item)
        total_demand += item[2]
        if total_demand > capacity + 1e-9:
            break

    if total_demand <= capacity + 1e-9:
        return None

    cover.sort(key=lambda item: item[2])
    changed = True
    while changed:
        changed = False
        current_total = sum(item[2] for item in cover)
        for index, item in enumerate(list(cover)):
            if current_total - item[2] > capacity + 1e-9:
                del cover[index]
                changed = True
                break

    cover.sort(key=lambda item: item[2], reverse=True)
    return cover


def build_cover_expr(variables, item, server, time_value, scenario_name):
    item_type, item_id, _ = item
    if item_type == "od":
        return variables["x"][item_id, server, time_value]
    if item_type == "sp":
        return variables["a"][item_id, server, time_value, scenario_name]
    return variables["z"][item_id, server, time_value]


def configure_solver_cut_parameters(model, cut_flags):
    param_values = dict(cut_flags.get("solver_params", {}))
    for param_name, param_value in param_values.items():
        model.setParam(param_name, param_value)
    return param_values


def load_instance_data(instance_dir):
    with open(instance_dir / "instance.json", "r", encoding="utf-8") as file:
        instance = json.load(file)

    metadata = pd.read_csv(instance_dir / "workload_metadata.csv")
    scenario_time_series = pd.read_csv(instance_dir / "scenario_time_series.csv")
    batch_jobs = pd.read_csv(instance_dir / "batch_jobs.csv")
    batch_job_demands = pd.read_csv(instance_dir / "batch_job_demands.csv")

    servers = list(range(instance["num_servers"]))
    times = [int(time) for time in instance["time_periods"]]
    scenarios = [scenario["scenario"] for scenario in instance["scenarios"]]
    scenario_prob = {scenario["scenario"]: float(scenario["probability"]) for scenario in instance["scenarios"]}

    on_demand_ids = metadata.loc[metadata["workload_type"] == "on_demand", "workload_id"].tolist()
    spot_ids = metadata.loc[metadata["workload_type"] == "spot", "workload_id"].tolist()
    batch_ids = batch_jobs["batch_job_id"].tolist()

    od_active = (
        scenario_time_series.loc[scenario_time_series["workload_type"] == "on_demand"]
        .groupby("workload_id")["time"]
        .apply(lambda series: sorted(series.astype(int).unique().tolist()))
        .to_dict()
    )
    od_transitions = {
        workload_id: list(zip(active_times[:-1], active_times[1:]))
        for workload_id, active_times in od_active.items()
    }
    spot_active = (
        scenario_time_series.loc[scenario_time_series["workload_type"] == "spot"]
        .groupby("workload_id")["time"]
        .apply(lambda series: sorted(series.astype(int).unique().tolist()))
        .to_dict()
    )

    d_od = {}
    d_sp = {}
    for row in scenario_time_series.itertuples(index=False):
        key = (row.workload_id, int(row.time), row.scenario)
        if row.workload_type == "on_demand":
            d_od[key] = float(row.demand)
        else:
            d_sp[key] = float(row.demand)

    d_batch = {
        (row.batch_job_id, row.scenario): float(row.demand)
        for row in batch_job_demands.itertuples(index=False)
    }
    batch_parents = batch_jobs.set_index("batch_job_id")["parent_workload_id"].to_dict() if not batch_jobs.empty else {}
    batch_source_time = batch_jobs.set_index("batch_job_id")["source_time"].to_dict() if not batch_jobs.empty else {}

    return {
        "instance": instance,
        "metadata": metadata,
        "scenario_time_series": scenario_time_series,
        "batch_jobs": batch_jobs,
        "batch_job_demands": batch_job_demands,
        "servers": servers,
        "times": times,
        "scenarios": scenarios,
        "scenario_prob": scenario_prob,
        "on_demand_ids": on_demand_ids,
        "spot_ids": spot_ids,
        "batch_ids": batch_ids,
        "batch_parents": batch_parents,
        "batch_source_time": batch_source_time,
        "od_active": od_active,
        "od_transitions": od_transitions,
        "spot_active": spot_active,
        "d_od": d_od,
        "d_sp": d_sp,
        "d_batch": d_batch,
        "capacity": float(instance["server_capacity"]),
        "big_m": float(instance["big_m"]),
        "epsilon_od": float(instance["chance_constraints"]["epsilon_od"]),
        "epsilon_sp": float(instance["chance_constraints"]["epsilon_sp"]),
        "rho": float(instance["chance_constraints"]["rho"]),
        "objective_type": instance.get("objective", {}).get("type", "server_count"),
        "lambda_migration": float(instance.get("objective", {}).get("lambda_migration", 0.0)),
        "energy_idle": float(instance.get("objective", {}).get("energy_idle", 0.0)),
        "energy_cpu": float(instance.get("objective", {}).get("energy_cpu", 0.0)),
        "energy_migration": float(instance.get("objective", {}).get("energy_migration", 0.0)),
    }

def build_variables(model, data):
    servers = data["servers"]
    times = data["times"]
    scenarios = data["scenarios"]
    on_demand_ids = data["on_demand_ids"]
    spot_ids = data["spot_ids"]
    batch_ids = data["batch_ids"]
    od_active = data["od_active"]
    od_transitions = data["od_transitions"]
    spot_active = data["spot_active"]

    return {
        "u": model.addVars(servers, times, vtype=GRB.BINARY, name="u"),
        "u_used": model.addVars(servers, vtype=GRB.BINARY, name="u_used"),
        "x": model.addVars(
            [(i, s, t) for i in on_demand_ids for s in servers for t in od_active[i]],
            vtype=GRB.BINARY,
            name="x",
        ),
        "m": model.addVars(
            [(i, prev_t, curr_t) for i in on_demand_ids for prev_t, curr_t in od_transitions[i]],
            vtype=GRB.BINARY,
            name="m",
        ),
        "y": model.addVars([(j, s) for j in spot_ids for s in servers], vtype=GRB.BINARY, name="y"),
        "z": model.addVars([(k, s, t) for k in batch_ids for s in servers for t in times], vtype=GRB.BINARY, name="z"),
        "a": model.addVars(
            [(j, s, t, xi) for j in spot_ids for s in servers for t in spot_active[j] for xi in scenarios],
            vtype=GRB.BINARY,
            name="a",
        ),
        "gamma": model.addVars(servers, times, scenarios, vtype=GRB.BINARY, name="gamma"),
        "phi": model.addVars(servers, times, scenarios, vtype=GRB.BINARY, name="phi"),
        "eta": model.addVars(servers, scenarios, vtype=GRB.BINARY, name="eta"),
        "delta": model.addVars(spot_ids, scenarios, vtype=GRB.BINARY, name="delta"),
        "load": model.addVars(servers, times, scenarios, lb=0.0, vtype=GRB.CONTINUOUS, name="load"),
        "power_load": model.addVars(servers, times, scenarios, lb=0.0, vtype=GRB.CONTINUOUS, name="power_load"),
        "server_energy": model.addVars(servers, lb=0.0, vtype=GRB.CONTINUOUS, name="server_energy"),
    }


def add_first_stage_constraints(model, data, variables):
    servers = data["servers"]
    times = data["times"]
    on_demand_ids = data["on_demand_ids"]
    spot_ids = data["spot_ids"]
    batch_ids = data["batch_ids"]
    od_active = data["od_active"]
    od_transitions = data["od_transitions"]
    spot_active = data["spot_active"]

    u = variables["u"]
    u_used = variables["u_used"]
    x = variables["x"]
    m = variables["m"]
    y = variables["y"]
    z = variables["z"]

    model.addConstrs(gp.quicksum(x[i, s, t] for s in servers) == 1 for i in on_demand_ids for t in od_active[i])
    model.addConstrs(x[i, s, t] <= u[s, t] for i in on_demand_ids for s in servers for t in od_active[i])
    model.addConstrs(
        m[i, prev_t, curr_t] >= x[i, s, curr_t] - x[i, s, prev_t]
        for i in on_demand_ids
        for s in servers
        for prev_t, curr_t in od_transitions[i]
    )
    model.addConstrs(
        m[i, prev_t, curr_t] >= x[i, s, prev_t] - x[i, s, curr_t]
        for i in on_demand_ids
        for s in servers
        for prev_t, curr_t in od_transitions[i]
    )

    model.addConstrs(gp.quicksum(y[j, s] for s in servers) == 1 for j in spot_ids)
    model.addConstrs(y[j, s] <= u[s, t] for j in spot_ids for s in servers for t in spot_active[j])

    model.addConstrs(gp.quicksum(z[k, s, t] for s in servers for t in times) == 1 for k in batch_ids)
    model.addConstrs(z[k, s, t] <= u[s, t] for k in batch_ids for s in servers for t in times)

    model.addConstrs(u_used[s] >= u[s, t] for s in servers for t in times)
    model.addConstrs(u_used[s] <= gp.quicksum(u[s, t] for t in times) for s in servers)


def add_second_stage_constraints(model, data, variables):
    servers = data["servers"]
    times = data["times"]
    scenarios = data["scenarios"]
    scenario_prob = data["scenario_prob"]
    on_demand_ids = data["on_demand_ids"]
    spot_ids = data["spot_ids"]
    batch_ids = data["batch_ids"]
    od_active = data["od_active"]
    spot_active = data["spot_active"]
    d_od = data["d_od"]
    d_sp = data["d_sp"]
    d_batch = data["d_batch"]
    capacity = data["capacity"]
    big_m = data["big_m"]
    epsilon_od = data["epsilon_od"]
    epsilon_sp = data["epsilon_sp"]
    rho = data["rho"]

    y = variables["y"]
    z = variables["z"]
    a = variables["a"]
    gamma = variables["gamma"]
    phi = variables["phi"]
    eta = variables["eta"]
    delta = variables["delta"]
    load = variables["load"]
    power_load = variables["power_load"]
    x = variables["x"]
    u = variables["u"]

    model.addConstrs(
        a[j, s, t, xi] <= y[j, s]
        for j in spot_ids
        for s in servers
        for t in spot_active[j]
        for xi in scenarios
    )
    model.addConstrs(
        a[j, s, t, xi] <= 1 - gamma[s, t, xi]
        for j in spot_ids
        for s in servers
        for t in spot_active[j]
        for xi in scenarios
    )
    for s in servers:
        for t in times:
            for xi in scenarios:
                on_demand_load = gp.quicksum(
                    d_od[i, t, xi] * x[i, s, t]
                    for i in on_demand_ids
                    if t in od_active[i]
                )
                spot_load = gp.quicksum(
                    d_sp[j, t, xi] * a[j, s, t, xi]
                    for j in spot_ids
                    if t in spot_active[j]
                )
                batch_load = gp.quicksum(d_batch[k, xi] * z[k, s, t] for k in batch_ids)

                model.addConstr(load[s, t, xi] == on_demand_load + spot_load + batch_load)
                model.addConstr(load[s, t, xi] <= capacity * u[s, t] + big_m * phi[s, t, xi])
                model.addConstr(eta[s, xi] >= phi[s, t, xi])
                model.addConstr(power_load[s, t, xi] <= load[s, t, xi])
                model.addConstr(power_load[s, t, xi] <= capacity * u[s, t])
                model.addConstr(power_load[s, t, xi] >= load[s, t, xi] - big_m * phi[s, t, xi])
                model.addConstr(power_load[s, t, xi] >= capacity * u[s, t] - big_m * (1 - phi[s, t, xi]))

    model.addConstrs(gp.quicksum(scenario_prob[xi] * eta[s, xi] for xi in scenarios) <= epsilon_od for s in servers)
    model.addConstrs(eta[s, xi] <= gp.quicksum(phi[s, t, xi] for t in times) for s in servers for xi in scenarios)

    model.addConstrs(
        delta[j, xi] >= 1 - gp.quicksum(a[j, s, t, xi] for s in servers)
        for j in spot_ids
        for t in spot_active[j]
        for xi in scenarios
    )
    model.addConstrs(
        delta[j, xi] <= gp.quicksum(
            1 - gp.quicksum(a[j, s, t, xi] for s in servers)
            for t in spot_active[j]
        )
        for j in spot_ids
        for xi in scenarios
    )
    model.addConstrs(gp.quicksum(scenario_prob[xi] * delta[j, xi] for xi in scenarios) <= epsilon_sp for j in spot_ids)
    model.addConstrs(
        gp.quicksum(a[j, s, t, xi] for s in servers for t in spot_active[j]) >= rho * len(spot_active[j])
        for j in spot_ids
        for xi in scenarios
    )


def add_experimental_cuts(model, data, variables, cut_flags, cut_stats):
    servers = data["servers"]
    times = data["times"]
    scenarios = data["scenarios"]
    scenario_prob = data["scenario_prob"]
    rho = data["rho"]
    capacity = data["capacity"]
    big_m = data["big_m"]
    epsilon_od = data["epsilon_od"]
    epsilon_sp = data["epsilon_sp"]
    spot_ids = data["spot_ids"]
    od_active = data["od_active"]
    spot_active = data["spot_active"]
    d_od = data["d_od"]
    d_batch = data["d_batch"]
    u = variables["u"]
    u_used = variables["u_used"]
    x = variables["x"]
    y = variables["y"]
    z = variables["z"]
    a = variables["a"]
    gamma = variables["gamma"]
    delta = variables["delta"]
    phi = variables["phi"]
    eta = variables["eta"]
    load = variables["load"]

    if cut_flags.get("activation_upper"):
        local_count = 0
        for server in servers:
            for time_value in times:
                rhs = build_server_time_rhs(data, variables, server, time_value)
                model.addConstr(u[server, time_value] <= rhs, name=f"cut_activation_upper[{server},{time_value}]")
                local_count += 1
        increment_cut_count(cut_stats, "activation_upper", local_count)

        local_count = 0
        for server in servers:
            rhs = gp.quicksum(build_server_time_rhs(data, variables, server, time_value) for time_value in times)
            model.addConstr(u_used[server] <= rhs, name=f"cut_used_upper[{server}]")
            local_count += 1
        increment_cut_count(cut_stats, "used_upper", local_count)

    if cut_flags.get("spot_completion_server"):
        local_count = 0
        for workload_id in data["spot_ids"]:
            active_times = data["spot_active"][workload_id]
            active_count = len(active_times)
            if active_count == 0:
                continue
            for server in servers:
                for scenario_name in scenarios:
                    model.addConstr(
                        gp.quicksum(a[workload_id, server, time_value, scenario_name] for time_value in active_times)
                        >= rho * active_count * y[workload_id, server],
                        name=f"cut_spot_completion_server[{workload_id},{server},{scenario_name}]",
                    )
                    local_count += 1
        increment_cut_count(cut_stats, "spot_completion_server", local_count)

    if cut_flags.get("spot_delta_server"):
        local_count = 0
        for workload_id in data["spot_ids"]:
            active_times = data["spot_active"][workload_id]
            active_count = len(active_times)
            if active_count == 0:
                continue
            for server in servers:
                for scenario_name in scenarios:
                    model.addConstr(
                        gp.quicksum(a[workload_id, server, time_value, scenario_name] for time_value in active_times)
                        >= active_count * (y[workload_id, server] - delta[workload_id, scenario_name]),
                        name=f"cut_spot_delta_server[{workload_id},{server},{scenario_name}]",
                    )
                    local_count += 1
        increment_cut_count(cut_stats, "spot_delta_server", local_count)

    if cut_flags.get("spot_time_lower"):
        local_count = 0
        for workload_id in data["spot_ids"]:
            for server in servers:
                for time_value in data["spot_active"][workload_id]:
                    for scenario_name in scenarios:
                        model.addConstr(
                            a[workload_id, server, time_value, scenario_name]
                            >= y[workload_id, server] - delta[workload_id, scenario_name],
                            name=f"cut_spot_time_lower[{workload_id},{server},{time_value},{scenario_name}]",
                        )
                        local_count += 1
        increment_cut_count(cut_stats, "spot_time_lower", local_count)

    if cut_flags.get("delta_gamma_link"):
        local_count = 0
        for workload_id in data["spot_ids"]:
            for server in servers:
                for time_value in data["spot_active"][workload_id]:
                    for scenario_name in scenarios:
                        model.addConstr(
                            delta[workload_id, scenario_name]
                            >= y[workload_id, server] + gamma[server, time_value, scenario_name] - 1,
                            name=f"cut_delta_gamma_link[{workload_id},{server},{time_value},{scenario_name}]",
                        )
                        local_count += 1
        increment_cut_count(cut_stats, "delta_gamma_link", local_count)

    if cut_flags.get("delta_mass_gamma"):
        local_count = 0
        for time_value in times:
            active_spot_ids = [workload_id for workload_id in spot_ids if time_value in spot_active[workload_id]]
            active_count = len(active_spot_ids)
            if active_count == 0:
                continue
            for server in servers:
                assigned_spots = gp.quicksum(y[workload_id, server] for workload_id in active_spot_ids)
                for scenario_name in scenarios:
                    model.addConstr(
                        gp.quicksum(delta[workload_id, scenario_name] for workload_id in active_spot_ids)
                        >= assigned_spots - active_count * (1 - gamma[server, time_value, scenario_name]),
                        name=f"cut_delta_mass_gamma[{server},{time_value},{scenario_name}]",
                    )
                    local_count += 1
        increment_cut_count(cut_stats, "delta_mass_gamma", local_count)

    if cut_flags.get("spot_gamma_aggregate"):
        local_count = 0
        for workload_id in data["spot_ids"]:
            active_times = data["spot_active"][workload_id]
            active_count = len(active_times)
            if active_count == 0:
                continue
            for server in servers:
                for scenario_name in scenarios:
                    model.addConstr(
                        gp.quicksum(gamma[server, time_value, scenario_name] for time_value in active_times)
                        <= active_count * (delta[workload_id, scenario_name] + 1 - y[workload_id, server]),
                        name=f"cut_spot_gamma_aggregate[{workload_id},{server},{scenario_name}]",
                    )
                    local_count += 1
        increment_cut_count(cut_stats, "spot_gamma_aggregate", local_count)

    if cut_flags.get("spot_bridge_lower"):
        local_count = 0
        for workload_id in data["spot_ids"]:
            for server in servers:
                for time_value in data["spot_active"][workload_id]:
                    for scenario_name in scenarios:
                        model.addConstr(
                            a[workload_id, server, time_value, scenario_name]
                            >= y[workload_id, server]
                            - gamma[server, time_value, scenario_name]
                            - delta[workload_id, scenario_name],
                            name=f"cut_spot_bridge_lower[{workload_id},{server},{time_value},{scenario_name}]",
                        )
                        local_count += 1
        increment_cut_count(cut_stats, "spot_bridge_lower", local_count)

    if cut_flags.get("spot_bridge_aggregate"):
        local_count = 0
        for workload_id in data["spot_ids"]:
            active_times = data["spot_active"][workload_id]
            active_count = len(active_times)
            if active_count == 0:
                continue
            for server in servers:
                for scenario_name in scenarios:
                    model.addConstr(
                        gp.quicksum(a[workload_id, server, time_value, scenario_name] for time_value in active_times)
                        >= active_count * y[workload_id, server]
                        - gp.quicksum(gamma[server, time_value, scenario_name] for time_value in active_times)
                        - active_count * delta[workload_id, scenario_name],
                        name=f"cut_spot_bridge_aggregate[{workload_id},{server},{scenario_name}]",
                    )
                    local_count += 1
        increment_cut_count(cut_stats, "spot_bridge_aggregate", local_count)

    if cut_flags.get("state_link"):
        local_count = 0
        for server in servers:
            for time_value in times:
                for scenario_name in scenarios:
                    model.addConstr(
                        phi[server, time_value, scenario_name] <= u[server, time_value],
                        name=f"cut_phi_state_link[{server},{time_value},{scenario_name}]",
                    )
                    model.addConstr(
                        gamma[server, time_value, scenario_name] <= u[server, time_value],
                        name=f"cut_gamma_state_link[{server},{time_value},{scenario_name}]",
                    )
                    local_count += 2
        for server in servers:
            for scenario_name in scenarios:
                model.addConstr(
                    eta[server, scenario_name] <= gp.quicksum(u[server, time_value] for time_value in times),
                    name=f"cut_eta_state_link[{server},{scenario_name}]",
                )
                local_count += 1
        increment_cut_count(cut_stats, "state_link", local_count)

    if cut_flags.get("phi_scenario_mass"):
        local_count = 0
        for server in servers:
            for time_value in times:
                model.addConstr(
                    gp.quicksum(scenario_prob[scenario_name] * phi[server, time_value, scenario_name] for scenario_name in scenarios)
                    <= epsilon_od,
                    name=f"cut_phi_scenario_mass[{server},{time_value}]",
                )
                local_count += 1
        increment_cut_count(cut_stats, "phi_scenario_mass", local_count)

    if cut_flags.get("eta_aggregate_load"):
        local_count = 0
        time_count = len(times)
        for server in servers:
            for scenario_name in scenarios:
                model.addConstr(
                    gp.quicksum(load[server, time_value, scenario_name] for time_value in times)
                    <= capacity * gp.quicksum(u[server, time_value] for time_value in times)
                    + big_m * time_count * eta[server, scenario_name],
                    name=f"cut_eta_aggregate_load[{server},{scenario_name}]",
                )
                local_count += 1
        increment_cut_count(cut_stats, "eta_aggregate_load", local_count)

    if cut_flags.get("eta_cover_general"):
        local_count = 0
        for time_value in times:
            for scenario_name in scenarios:
                cover_items = build_greedy_minimal_cover(
                    data=data,
                    time_value=time_value,
                    scenario_name=scenario_name,
                    capacity=capacity,
                    include_types=("od", "sp", "bj"),
                )
                if not cover_items:
                    continue
                for server in servers:
                    lhs = gp.quicksum(
                        build_cover_expr(variables, item, server, time_value, scenario_name) for item in cover_items
                    )
                    item_tokens = ",".join(f"{item_type}:{item_id}" for item_type, item_id, _ in cover_items)
                    model.addConstr(
                        lhs <= len(cover_items) - 1 + eta[server, scenario_name],
                        name=f"cut_eta_cover_general[{server},{time_value},{scenario_name},{item_tokens}]",
                    )
                    local_count += 1
        increment_cut_count(cut_stats, "eta_cover_general", local_count)

    if cut_flags.get("eta_cover_fixed"):
        local_count = 0
        for time_value in times:
            for scenario_name in scenarios:
                cover_items = build_greedy_minimal_cover(
                    data=data,
                    time_value=time_value,
                    scenario_name=scenario_name,
                    capacity=capacity,
                    include_types=("od", "bj"),
                )
                if not cover_items:
                    continue
                for server in servers:
                    lhs = gp.quicksum(
                        build_cover_expr(variables, item, server, time_value, scenario_name) for item in cover_items
                    )
                    item_tokens = ",".join(f"{item_type}:{item_id}" for item_type, item_id, _ in cover_items)
                    model.addConstr(
                        lhs <= len(cover_items) - 1 + eta[server, scenario_name],
                        name=f"cut_eta_cover_fixed[{server},{time_value},{scenario_name},{item_tokens}]",
                    )
                    local_count += 1
        increment_cut_count(cut_stats, "eta_cover_fixed", local_count)

    if cut_flags.get("minimal_cover_general"):
        local_count = 0
        for time_value in times:
            for scenario_name in scenarios:
                cover_items = build_greedy_minimal_cover(
                    data=data,
                    time_value=time_value,
                    scenario_name=scenario_name,
                    capacity=capacity,
                    include_types=("od", "sp", "bj"),
                )
                if not cover_items:
                    continue
                for server in servers:
                    lhs = gp.quicksum(
                        build_cover_expr(variables, item, server, time_value, scenario_name) for item in cover_items
                    )
                    item_tokens = ",".join(f"{item_type}:{item_id}" for item_type, item_id, _ in cover_items)
                    model.addConstr(
                        lhs <= len(cover_items) - 1 + phi[server, time_value, scenario_name],
                        name=f"cut_minimal_cover_general[{server},{time_value},{scenario_name},{item_tokens}]",
                    )
                    local_count += 1
        increment_cut_count(cut_stats, "minimal_cover_general", local_count)

    if cut_flags.get("minimal_cover_fixed"):
        local_count = 0
        for time_value in times:
            for scenario_name in scenarios:
                cover_items = build_greedy_minimal_cover(
                    data=data,
                    time_value=time_value,
                    scenario_name=scenario_name,
                    capacity=capacity,
                    include_types=("od", "bj"),
                )
                if not cover_items:
                    continue
                for server in servers:
                    lhs = gp.quicksum(
                        build_cover_expr(variables, item, server, time_value, scenario_name) for item in cover_items
                    )
                    item_tokens = ",".join(f"{item_type}:{item_id}" for item_type, item_id, _ in cover_items)
                    model.addConstr(
                        lhs <= len(cover_items) - 1 + phi[server, time_value, scenario_name],
                        name=f"cut_minimal_cover_fixed[{server},{time_value},{scenario_name},{item_tokens}]",
                    )
                    local_count += 1
        increment_cut_count(cut_stats, "minimal_cover_fixed", local_count)

    if cut_flags.get("aggregate_fixed_load"):
        local_count = 0
        for time_value in times:
            for scenario_name in scenarios:
                lhs = gp.quicksum(
                    d_od[workload_id, time_value, scenario_name] * x[workload_id, server, time_value]
                    for workload_id in data["on_demand_ids"]
                    if time_value in od_active[workload_id]
                    for server in servers
                ) + gp.quicksum(
                    d_batch[batch_job_id, scenario_name] * z[batch_job_id, server, time_value]
                    for batch_job_id in data["batch_ids"]
                    for server in servers
                )
                rhs = capacity * gp.quicksum(u[server, time_value] for server in servers) + big_m * gp.quicksum(
                    phi[server, time_value, scenario_name] for server in servers
                )
                model.addConstr(
                    lhs <= rhs,
                    name=f"cut_aggregate_fixed_load[{time_value},{scenario_name}]",
                )
                local_count += 1
        increment_cut_count(cut_stats, "aggregate_fixed_load", local_count)

    if cut_flags.get("gamma_probability_cap"):
        local_count = 0
        for workload_id in spot_ids:
            for server in servers:
                for time_value in spot_active[workload_id]:
                    model.addConstr(
                        gp.quicksum(
                            scenario_prob[scenario_name] * gamma[server, time_value, scenario_name]
                            for scenario_name in scenarios
                        )
                        <= epsilon_sp + 1 - y[workload_id, server],
                        name=f"cut_gamma_probability_cap[{workload_id},{server},{time_value}]",
                    )
                    local_count += 1
        increment_cut_count(cut_stats, "gamma_probability_cap", local_count)

    if cut_flags.get("pairwise_cover"):
        local_count = 0
        for time_value in times:
            for scenario_name in scenarios:
                pair_candidates = build_cover_candidates(data, time_value, scenario_name, threshold=capacity / 2.0)
                for item_left, item_right in itertools.combinations(pair_candidates, 2):
                    if item_left[2] + item_right[2] <= capacity + 1e-9:
                        continue
                    for server in servers:
                        lhs = (
                            build_cover_expr(variables, item_left, server, time_value, scenario_name)
                            + build_cover_expr(variables, item_right, server, time_value, scenario_name)
                        )
                        model.addConstr(
                            lhs <= 1 + phi[server, time_value, scenario_name],
                            name=f"cut_pair_cover[{server},{time_value},{scenario_name},{item_left[0]},{item_left[1]},{item_right[0]},{item_right[1]}]",
                        )
                        local_count += 1
        increment_cut_count(cut_stats, "pairwise_cover", local_count)

    if cut_flags.get("triple_cover"):
        local_count = 0
        for time_value in times:
            for scenario_name in scenarios:
                triple_candidates = build_cover_candidates(data, time_value, scenario_name, threshold=capacity / 3.0)[:12]
                triple_rows = []
                for triple_items in itertools.combinations(triple_candidates, 3):
                    total_demand = sum(item[2] for item in triple_items)
                    if total_demand <= capacity + 1e-9:
                        continue
                    if any(
                        triple_items[left][2] + triple_items[right][2] > capacity + 1e-9
                        for left, right in ((0, 1), (0, 2), (1, 2))
                    ):
                        continue
                    triple_rows.append((total_demand, triple_items))

                triple_rows.sort(key=lambda row: row[0], reverse=True)
                for _, triple_items in triple_rows[:50]:
                    for server in servers:
                        lhs = gp.quicksum(
                            build_cover_expr(variables, item, server, time_value, scenario_name)
                            for item in triple_items
                        )
                        model.addConstr(
                            lhs <= 2 + phi[server, time_value, scenario_name],
                            name=(
                                f"cut_triple_cover[{server},{time_value},{scenario_name},"
                                f"{triple_items[0][0]},{triple_items[0][1]},"
                                f"{triple_items[1][0]},{triple_items[1][1]},"
                                f"{triple_items[2][0]},{triple_items[2][1]}]"
                            ),
                        )
                        local_count += 1
        increment_cut_count(cut_stats, "triple_cover", local_count)


def add_symmetry_constraints(model, data, variables, cut_flags=None, cut_stats=None):
    servers = data["servers"]
    times = data["times"]
    scenarios = data["scenarios"]
    scenario_prob = data["scenario_prob"]
    capacity = data["capacity"]
    objective_type = data["objective_type"]
    energy_idle = data["energy_idle"]
    energy_cpu = data["energy_cpu"]
    u = variables["u"]
    u_used = variables["u_used"]
    power_load = variables["power_load"]
    server_energy = variables["server_energy"]
    cut_flags = cut_flags or {}
    cut_stats = cut_stats or {}

    model.addConstrs(u_used[s] >= u_used[s + 1] for s in servers[:-1])
    increment_cut_count(cut_stats, "base_server_order", max(0, len(servers) - 1))

    if cut_flags.get("uptime_symmetry"):
        local_count = 0
        for server in servers[:-1]:
            model.addConstr(
                gp.quicksum(u[server, time_value] for time_value in times)
                >= gp.quicksum(u[server + 1, time_value] for time_value in times),
                name=f"cut_uptime_order[{server}]",
            )
            local_count += 1
        increment_cut_count(cut_stats, "uptime_symmetry", local_count)

    if objective_type == "energy":
        for s in servers:
            model.addConstr(
                server_energy[s]
                == energy_idle * gp.quicksum(u[s, t] for t in times)
                + (energy_cpu / capacity)
                * gp.quicksum(
                    scenario_prob[xi] * power_load[s, t, xi]
                    for t in times
                    for xi in scenarios
                ),
                name=f"server_energy_balance[{s}]",
            )
        model.addConstrs(server_energy[s] >= server_energy[s + 1] for s in servers[:-1])
        increment_cut_count(cut_stats, "energy_order", max(0, len(servers) - 1))


def set_objective(model, data, variables):
    objective_type = data["objective_type"]
    on_demand_ids = data["on_demand_ids"]
    od_transitions = data["od_transitions"]
    lambda_migration = data["lambda_migration"]
    servers = data["servers"]
    times = data["times"]
    scenarios = data["scenarios"]
    scenario_prob = data["scenario_prob"]
    capacity = data["capacity"]
    energy_idle = data["energy_idle"]
    energy_cpu = data["energy_cpu"]
    energy_migration = data["energy_migration"]

    u_used = variables["u_used"]
    u = variables["u"]
    m = variables["m"]
    power_load = variables["power_load"]

    if objective_type == "energy":
        idle_energy = energy_idle * gp.quicksum(u[s, t] for s in servers for t in times)
        cpu_energy = (energy_cpu / capacity) * gp.quicksum(
            scenario_prob[xi] * power_load[s, t, xi]
            for s in servers
            for t in times
            for xi in scenarios
        )
        migration_energy = energy_migration * gp.quicksum(
            m[i, prev_t, curr_t]
            for i in on_demand_ids
            for prev_t, curr_t in od_transitions[i]
        )
        model.setObjective(idle_energy + cpu_energy + migration_energy, GRB.MINIMIZE)
        return

    if on_demand_ids:
        migration_penalty = (
            lambda_migration
            * (1.0 / len(on_demand_ids))
            * gp.quicksum(m[i, prev_t, curr_t] for i in on_demand_ids for prev_t, curr_t in od_transitions[i])
        )
    else:
        migration_penalty = 0.0

    model.setObjective(gp.quicksum(u_used[s] for s in servers) + migration_penalty, GRB.MINIMIZE)


def build_model(
    data,
    results_dir,
    time_limit,
    mip_gap,
    log_name="solver.log",
    threads=None,
    server_limit=None,
    no_rel_heur_time=0.0,
    cut_profile="baseline",
):
    model = gp.Model("chance_constrained_2sp_toy")
    cut_flags = resolve_cut_profile(cut_profile)
    cut_stats = {}
    if time_limit is not None and float(time_limit) > 0.0:
        model.setParam("TimeLimit", float(time_limit))
    model.setParam("MIPGap", mip_gap)
    model.setParam("LogFile", str(results_dir / log_name))
    model.setParam("LogToConsole", 0)
    if threads is not None:
        model.setParam("Threads", threads)
    if no_rel_heur_time and no_rel_heur_time > 0.0:
        model.setParam("NoRelHeurTime", float(no_rel_heur_time))
    solver_cut_params = configure_solver_cut_parameters(model, cut_flags)

    variables = build_variables(model, data)
    add_first_stage_constraints(model, data, variables)
    add_second_stage_constraints(model, data, variables)
    add_experimental_cuts(model, data, variables, cut_flags, cut_stats)
    add_symmetry_constraints(model, data, variables, cut_flags=cut_flags, cut_stats=cut_stats)
    if server_limit is not None:
        model.addConstr(
            gp.quicksum(variables["u_used"][s] for s in data["servers"]) <= int(server_limit),
            name="server_limit",
        )
    set_objective(model, data, variables)
    model._cut_profile = cut_profile
    model._cut_profile_category = cut_profile
    model._cut_profile_description = json.dumps(cut_flags, ensure_ascii=False, sort_keys=True)
    model._cut_counts = cut_stats
    model._solver_cut_params = solver_cut_params
    return model, variables


def build_summary(model, data):
    has_solution = model.SolCount > 0
    return {
        "status_code": int(model.Status),
        "status_name": status_name(model.Status),
        "runtime_seconds": float(model.Runtime),
        "solution_count": int(model.SolCount),
        "has_solution": bool(has_solution),
        "objective_value": float(model.ObjVal) if has_solution else None,
        "objective_bound": float(model.ObjBound) if model.IsMIP else None,
        "mip_gap": float(model.MIPGap) if has_solution and model.IsMIP else None,
        "node_count": float(model.NodeCount) if model.IsMIP else 0.0,
        "iter_count": float(model.IterCount),
        "work_units": float(model.Work),
        "instance_name": data["instance"]["instance_name"],
        "server_capacity": data["capacity"],
        "num_servers": len(data["servers"]),
        "epsilon_od": data["epsilon_od"],
        "epsilon_sp": data["epsilon_sp"],
        "rho": data["rho"],
        "objective_type": data["objective_type"],
        "lambda_migration": data["lambda_migration"],
        "energy_idle": data["energy_idle"],
        "energy_cpu": data["energy_cpu"],
        "energy_migration": data["energy_migration"],
        "cut_profile": getattr(model, "_cut_profile", "baseline"),
        "cut_profile_category": getattr(model, "_cut_profile_category", "baseline"),
        "cut_profile_description": getattr(model, "_cut_profile_description", ""),
        "solver_cut_params": getattr(model, "_solver_cut_params", {}),
        "cut_counts": getattr(model, "_cut_counts", {}),
    }


def count_actual_migration_events(on_demand_frame):
    if on_demand_frame.empty:
        return 0

    event_count = 0
    for _, group in on_demand_frame.groupby("workload_id"):
        ordered = group.sort_values("time")
        previous_server = None
        for row in ordered.itertuples(index=False):
            if previous_server is not None and int(row.server) != int(previous_server):
                event_count += 1
            previous_server = int(row.server)
    return int(event_count)


def extract_solution_tables(data, variables):
    servers = data["servers"]
    times = data["times"]
    scenarios = data["scenarios"]
    scenario_prob = data["scenario_prob"]
    on_demand_ids = data["on_demand_ids"]
    spot_ids = data["spot_ids"]
    batch_ids = data["batch_ids"]
    batch_parents = data["batch_parents"]
    batch_source_time = data["batch_source_time"]
    od_active = data["od_active"]
    od_transitions = data["od_transitions"]
    spot_active = data["spot_active"]
    capacity = data["capacity"]
    lambda_migration = data["lambda_migration"]
    objective_type = data["objective_type"]
    energy_idle = data["energy_idle"]
    energy_cpu = data["energy_cpu"]
    energy_migration = data["energy_migration"]

    u_used = variables["u_used"]
    u = variables["u"]
    x = variables["x"]
    y = variables["y"]
    z = variables["z"]
    m = variables["m"]
    a = variables["a"]
    gamma = variables["gamma"]
    phi = variables["phi"]
    eta = variables["eta"]
    delta = variables["delta"]
    load = variables["load"]
    power_load = variables["power_load"]
    server_energy = variables["server_energy"]

    used_servers = [s for s in servers if u_used[s].X > 0.5]
    migration_count = (
        sum(m[i, prev_t, curr_t].X for i in on_demand_ids for prev_t, curr_t in od_transitions[i])
        if on_demand_ids
        else 0.0
    )

    on_demand_rows = []
    for i in on_demand_ids:
        for t in od_active[i]:
            for s in servers:
                if x[i, s, t].X > 0.5:
                    on_demand_rows.append({"workload_id": i, "time": t, "server": s})

    spot_rows = []
    for j in spot_ids:
        for s in servers:
            if y[j, s].X > 0.5:
                spot_rows.append({"workload_id": j, "server": s})

    batch_rows = []
    for k in batch_ids:
        for s in servers:
            for t in times:
                if z[k, s, t].X > 0.5:
                    batch_rows.append(
                        {
                            "batch_job_id": k,
                            "parent_workload_id": batch_parents.get(k),
                            "source_time": int(batch_source_time.get(k, -1)),
                            "server": s,
                            "time": t,
                        }
                    )

    on_demand_frame = pd.DataFrame(on_demand_rows, columns=["workload_id", "time", "server"])
    spot_placement = pd.DataFrame(spot_rows, columns=["workload_id", "server"])
    batch_schedule = pd.DataFrame(
        batch_rows,
        columns=["batch_job_id", "parent_workload_id", "source_time", "server", "time"],
    )

    od_lookup = {
        (row.workload_id, int(row.time), row.scenario): float(row.demand)
        for row in data["scenario_time_series"].loc[data["scenario_time_series"]["workload_type"] == "on_demand"].itertuples(index=False)
    }
    sp_lookup = {
        (row.workload_id, int(row.time), row.scenario): float(row.demand)
        for row in data["scenario_time_series"].loc[data["scenario_time_series"]["workload_type"] == "spot"].itertuples(index=False)
    }
    batch_lookup = {
        (row.batch_job_id, row.scenario): float(row.demand)
        for row in data["batch_job_demands"].itertuples(index=False)
    }

    spot_assignment = spot_placement.set_index("workload_id")["server"].to_dict() if not spot_placement.empty else {}

    scenario_rows = []
    spot_activity_rows = []

    for xi in scenarios:
        for s in servers:
            for t in times:
                on_demand_load = sum(
                    od_lookup.get((i, t, xi), 0.0) * x[i, s, t].X
                    for i in on_demand_ids
                    if t in od_active[i]
                )
                committed_spot_load = sum(
                    sp_lookup.get((j, t, xi), 0.0) * y[j, s].X
                    for j in spot_ids
                    if t in spot_active[j]
                )
                realized_spot_load = sum(
                    sp_lookup.get((j, t, xi), 0.0) * a[j, s, t, xi].X
                    for j in spot_ids
                    if t in spot_active[j]
                )
                batch_load = sum(batch_lookup.get((k, xi), 0.0) * z[k, s, t].X for k in batch_ids)
                committed_load = on_demand_load + committed_spot_load + batch_load
                realized_load = float(load[s, t, xi].X)

                scenario_rows.append(
                    {
                        "server": s,
                        "time": t,
                        "scenario": xi,
                        "load": realized_load,
                        "committed_load": committed_load,
                        "gamma": int(round(gamma[s, t, xi].X)),
                        "phi": int(round(phi[s, t, xi].X)),
                        "u": int(round(u[s, t].X)),
                        "realized_utilization": realized_load / capacity if capacity > 0 else 0.0,
                        "overbooking_ratio": committed_load / capacity if capacity > 0 else 0.0,
                    }
                )

        for j in spot_ids:
            assigned_server = int(spot_assignment[j]) if j in spot_assignment else None
            for t in spot_active[j]:
                active_value = 0
                server_value = None
                if assigned_server is not None:
                    server_value = assigned_server
                    active_value = int(round(a[j, assigned_server, t, xi].X))
                spot_activity_rows.append(
                    {
                        "workload_id": j,
                        "scenario": xi,
                        "time": t,
                        "server": server_value,
                        "active": active_value,
                        "suspended": 1 - active_value,
                    }
                )

    scenario_state = pd.DataFrame(
        scenario_rows,
        columns=[
            "server",
            "time",
            "scenario",
            "load",
            "committed_load",
            "gamma",
            "phi",
            "u",
            "realized_utilization",
            "overbooking_ratio",
        ],
    )
    spot_activity_state = pd.DataFrame(
        spot_activity_rows,
        columns=["workload_id", "scenario", "time", "server", "active", "suspended"],
    )

    if not scenario_state.empty:
        cluster_metrics = (
            scenario_state.groupby(["scenario", "time"], as_index=False)
            .agg(
                cluster_load=("load", "sum"),
                cluster_committed_load=("committed_load", "sum"),
                active_servers=("u", "sum"),
            )
        )
        cluster_metrics["cluster_overbooking_ratio"] = cluster_metrics.apply(
            lambda row: row["cluster_committed_load"] / (row["active_servers"] * capacity)
            if row["active_servers"] > 0
            else 0.0,
            axis=1,
        )
    else:
        cluster_metrics = pd.DataFrame(
            columns=["scenario", "time", "cluster_load", "cluster_committed_load", "active_servers", "cluster_overbooking_ratio"]
        )

    scenario_metrics = (
        scenario_state.groupby("scenario", as_index=False)
        .agg(
            peak_server_load=("load", "max"),
            peak_committed_server_load=("committed_load", "max"),
            peak_server_overbooking_ratio=("overbooking_ratio", "max"),
            gamma_count=("gamma", "sum"),
            phi_count=("phi", "sum"),
            active_server_periods=("u", "sum"),
        )
        .sort_values("scenario")
        .reset_index(drop=True)
    )
    if not cluster_metrics.empty:
        cluster_summary = (
            cluster_metrics.groupby("scenario", as_index=False)
            .agg(
                peak_cluster_load=("cluster_load", "max"),
                peak_cluster_committed_load=("cluster_committed_load", "max"),
                peak_cluster_overbooking_ratio=("cluster_overbooking_ratio", "max"),
            )
        )
        scenario_metrics = scenario_metrics.merge(cluster_summary, on="scenario", how="left")
    else:
        scenario_metrics["peak_cluster_load"] = 0.0
        scenario_metrics["peak_cluster_committed_load"] = 0.0
        scenario_metrics["peak_cluster_overbooking_ratio"] = 0.0
    scenario_metrics["peak_server_utilization"] = scenario_metrics["peak_server_load"] / capacity

    spot_metric_rows = []
    for j in spot_ids:
        suspension_probability = sum(scenario_prob[xi] * delta[j, xi].X for xi in scenarios)
        completion_ratio = sum(
            scenario_prob[xi] * sum(a[j, s, t, xi].X for s in servers for t in spot_active[j])
            for xi in scenarios
        ) / len(spot_active[j])
        spot_metric_rows.append(
            {
                "workload_id": j,
                "suspension_probability": float(suspension_probability),
                "completion_ratio": float(completion_ratio),
            }
        )

    server_violation_prob = {s: sum(scenario_prob[xi] * eta[s, xi].X for xi in scenarios) for s in servers}
    actual_migration_event_count = count_actual_migration_events(on_demand_frame)
    migration_count_value = float(migration_count)

    idle_energy_total = float(
        energy_idle * sum(u[s, t].X for s in servers for t in times)
    )
    cpu_energy_total = float(
        (energy_cpu / capacity)
        * sum(
            scenario_prob[xi] * power_load[s, t, xi].X
            for s in servers
            for t in times
            for xi in scenarios
        )
    )
    migration_energy_total = float(
        energy_migration
        * sum(m[i, prev_t, curr_t].X for i in on_demand_ids for prev_t, curr_t in od_transitions[i])
    )
    total_energy = idle_energy_total + cpu_energy_total + migration_energy_total

    server_energy_rows = []
    for s in servers:
        idle_energy = float(energy_idle * sum(u[s, t].X for t in times))
        cpu_energy = float(
            (energy_cpu / capacity)
            * sum(scenario_prob[xi] * power_load[s, t, xi].X for t in times for xi in scenarios)
        )
        server_energy_rows.append(
            {
                "server": s,
                "idle_energy": idle_energy,
                "cpu_energy": cpu_energy,
                "total_energy": float(server_energy[s].X) if objective_type == "energy" else idle_energy + cpu_energy,
            }
        )

    overbooking_peak_row = scenario_state.sort_values(
        ["overbooking_ratio", "committed_load", "scenario", "server", "time"],
        ascending=[False, False, True, True, True],
    ).iloc[0]
    realized_peak_row = scenario_state.sort_values(
        ["realized_utilization", "load", "scenario", "server", "time"],
        ascending=[False, False, True, True, True],
    ).iloc[0]

    normalization = len(on_demand_ids) if on_demand_ids else 1
    if objective_type == "energy":
        primary_objective_value = total_energy
    else:
        primary_objective_value = float(len(used_servers) + lambda_migration * migration_count_value / normalization)

    summary_updates = {
        "used_server_count": len(used_servers),
        "used_servers": used_servers,
        "primary_objective_value": primary_objective_value,
        "migration_count": migration_count_value,
        "actual_migration_event_count": actual_migration_event_count,
        "max_server_violation_probability": max(server_violation_prob.values(), default=0.0),
        "max_spot_suspension_probability": max((row["suspension_probability"] for row in spot_metric_rows), default=0.0),
        "min_spot_completion_ratio": min((row["completion_ratio"] for row in spot_metric_rows), default=1.0),
        "peak_realized_server_load": float(scenario_metrics["peak_server_load"].max()),
        "peak_realized_server_utilization": float(scenario_metrics["peak_server_utilization"].max()),
        "peak_overbooking_ratio": float(scenario_metrics["peak_server_overbooking_ratio"].max()),
        "peak_cluster_overbooking_ratio": float(scenario_metrics["peak_cluster_overbooking_ratio"].max()),
        "total_gamma_activations": int(scenario_metrics["gamma_count"].sum()),
        "total_phi_activations": int(scenario_metrics["phi_count"].sum()),
        "worst_realized_scenario": realized_peak_row["scenario"],
        "worst_realized_server": int(realized_peak_row["server"]),
        "worst_realized_time": int(realized_peak_row["time"]),
        "worst_overbooking_scenario": overbooking_peak_row["scenario"],
        "worst_overbooking_server": int(overbooking_peak_row["server"]),
        "worst_overbooking_time": int(overbooking_peak_row["time"]),
        "total_energy": total_energy,
        "idle_energy": idle_energy_total,
        "cpu_energy": cpu_energy_total,
        "migration_energy": migration_energy_total,
    }

    outputs = {
        "on_demand_placement": on_demand_frame,
        "spot_placement": spot_placement,
        "batch_schedule": batch_schedule,
        "scenario_server_state": scenario_state,
        "scenario_cluster_state": cluster_metrics,
        "spot_metrics": pd.DataFrame(spot_metric_rows, columns=["workload_id", "suspension_probability", "completion_ratio"]),
        "spot_activity_state": spot_activity_state,
        "scenario_metrics": scenario_metrics,
        "server_energy_summary": pd.DataFrame(server_energy_rows, columns=["server", "idle_energy", "cpu_energy", "total_energy"]),
    }
    return summary_updates, outputs


def write_solution_outputs(results_dir, summary, outputs):
    for name, frame in outputs.items():
        frame.to_csv(results_dir / f"{name}.csv", index=False)

    with open(results_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def write_summary_only(results_dir, summary):
    with open(results_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def capture_start_values(variables):
    start_values = {}
    for name, var_dict in variables.items():
        start_values[name] = {key: float(var.X) for key, var in var_dict.items()}
    return start_values


def apply_start_values(variables, start_values):
    if not start_values:
        return

    for name, value_map in start_values.items():
        if name not in variables:
            continue
        for key, value in value_map.items():
            if key in variables[name]:
                variables[name][key].Start = value


def apply_var_hints(variables, hint_values, hint_priority=10):
    if not hint_values:
        return

    for name, value_map in hint_values.items():
        if name not in variables:
            continue
        for key, value in value_map.items():
            if key in variables[name]:
                variables[name][key].VarHintVal = value
                variables[name][key].VarHintPri = int(hint_priority)


def apply_fixed_values(variables, fixed_values):
    if not fixed_values:
        return

    for name, value_map in fixed_values.items():
        if name not in variables:
            continue
        for key, value in value_map.items():
            if key in variables[name]:
                variables[name][key].LB = value
                variables[name][key].UB = value


def is_better_objective(candidate, incumbent, tolerance=1e-9):
    if candidate is None:
        return False
    if incumbent is None:
        return True
    return float(candidate) < float(incumbent) - tolerance


def solve_with_alternating_phases(
    data,
    results_dir,
    time_limit,
    mip_gap,
    threads=None,
    server_limit=None,
    alternate_norel_time=0.0,
    alternate_main_time=0.0,
    alternate_max_rounds=6,
    cut_profile="baseline",
    start_values=None,
    hint_values=None,
    hint_priority=10,
    fixed_values=None,
):
    cumulative_runtime = 0.0
    phase_rows = []
    incumbent_start_values = None
    incumbent_objective = None
    best_record = None
    incumbent_updates = 0
    last_summary = None
    norel_runtime_total = 0.0
    main_runtime_total = 0.0
    best_bound_seen = None

    if alternate_max_rounds is None or int(alternate_max_rounds) <= 0:
        alternate_max_rounds = 1

    for round_index in range(1, int(alternate_max_rounds) + 1):
        for phase_kind, phase_budget in (
            ("norel", float(alternate_norel_time)),
            ("main", float(alternate_main_time)),
        ):
            if phase_budget <= 0.0:
                continue

            if time_limit is not None:
                remaining_time = float(time_limit) - cumulative_runtime
                if remaining_time <= 1e-9:
                    break
                phase_budget = min(phase_budget, remaining_time)

            if phase_budget <= 1e-9:
                continue

            log_name = f"solver_round{round_index:02d}_{phase_kind}.log"
            model, variables = build_model(
                data=data,
                results_dir=results_dir,
                time_limit=phase_budget,
                mip_gap=mip_gap,
                log_name=log_name,
                threads=threads,
                server_limit=server_limit,
                no_rel_heur_time=phase_budget if phase_kind == "norel" else 0.0,
                cut_profile=cut_profile,
            )

            apply_fixed_values(variables, fixed_values)
            apply_start_values(variables, start_values)
            apply_start_values(variables, incumbent_start_values)
            apply_var_hints(variables, hint_values, hint_priority=hint_priority)
            if incumbent_objective is not None:
                cutoff_margin = max(1e-6, abs(float(incumbent_objective)) * 1e-6)
                model.setParam("Cutoff", float(incumbent_objective) + cutoff_margin)

            model.optimize()
            phase_summary = build_summary(model, data)
            phase_runtime = float(model.Runtime)
            cumulative_runtime += phase_runtime
            last_summary = phase_summary

            if phase_kind == "norel":
                norel_runtime_total += phase_runtime
            else:
                main_runtime_total += phase_runtime

            if phase_summary.get("objective_bound") is not None:
                current_bound = float(phase_summary["objective_bound"])
                if best_bound_seen is None or current_bound > best_bound_seen:
                    best_bound_seen = current_bound

            phase_row = {
                "round": round_index,
                "phase": phase_kind,
                "time_budget_seconds": float(phase_budget),
                "runtime_seconds": phase_runtime,
                "status_name": phase_summary["status_name"],
                "has_solution": phase_summary["has_solution"],
                "solution_count": phase_summary["solution_count"],
                "objective_value": phase_summary["objective_value"],
                "objective_bound": phase_summary["objective_bound"],
                "mip_gap": phase_summary["mip_gap"],
                "incumbent_objective_before_phase": incumbent_objective,
            }

            if model.SolCount > 0:
                candidate_objective = float(model.ObjVal)
                incumbent_start_values = capture_start_values(variables)
                phase_row["incumbent_objective_after_phase"] = candidate_objective

                if is_better_objective(candidate_objective, incumbent_objective):
                    incumbent_objective = candidate_objective
                    incumbent_updates += 1
                    summary_updates, outputs = extract_solution_tables(data, variables)
                    candidate_summary = dict(phase_summary)
                    candidate_summary.update(summary_updates)
                    best_record = {"summary": candidate_summary, "outputs": outputs}
            else:
                phase_row["incumbent_objective_after_phase"] = incumbent_objective

            phase_rows.append(phase_row)

            if model.Status == GRB.OPTIMAL:
                break

        if last_summary is not None and last_summary["status_name"] == "OPTIMAL":
            break
        if time_limit is not None and cumulative_runtime >= float(time_limit) - 1e-9:
            break

    phase_history = pd.DataFrame(
        phase_rows,
        columns=[
            "round",
            "phase",
            "time_budget_seconds",
            "runtime_seconds",
            "status_name",
            "has_solution",
            "solution_count",
            "objective_value",
            "objective_bound",
            "mip_gap",
            "incumbent_objective_before_phase",
            "incumbent_objective_after_phase",
        ],
    )
    phase_history.to_csv(results_dir / "solver_phase_history.csv", index=False)

    if best_record is not None:
        summary = best_record["summary"]
        outputs = best_record["outputs"]
    else:
        summary = dict(last_summary) if last_summary is not None else {
            "status_code": None,
            "status_name": "NOT_RUN",
            "runtime_seconds": 0.0,
            "solution_count": 0,
            "has_solution": False,
            "objective_value": None,
            "objective_bound": None,
            "mip_gap": None,
            "instance_name": data["instance"]["instance_name"],
            "server_capacity": data["capacity"],
            "num_servers": len(data["servers"]),
            "epsilon_od": data["epsilon_od"],
            "epsilon_sp": data["epsilon_sp"],
            "rho": data["rho"],
            "objective_type": data["objective_type"],
            "lambda_migration": data["lambda_migration"],
            "energy_idle": data["energy_idle"],
            "energy_cpu": data["energy_cpu"],
            "energy_migration": data["energy_migration"],
        }
        outputs = None

    summary["runtime_seconds"] = cumulative_runtime
    summary["threads"] = threads
    summary["server_limit"] = server_limit
    summary["no_rel_heur_time"] = 0.0
    summary["log_file"] = "solver_phase_history.csv"
    summary["norel_phase_attempted"] = True
    summary["norel_phase_time_limit"] = float(alternate_norel_time)
    summary["norel_phase_runtime_seconds"] = norel_runtime_total
    summary["main_phase_runtime_seconds"] = main_runtime_total
    norel_statuses = [row["status_name"] for row in phase_rows if row["phase"] == "norel"]
    summary["norel_phase_status_name"] = norel_statuses[-1] if norel_statuses else None
    summary["norel_phase_has_solution"] = any(row["phase"] == "norel" and row["has_solution"] for row in phase_rows)
    summary["norel_phase_solution_count"] = sum(
        int(row["solution_count"]) for row in phase_rows if row["phase"] == "norel"
    )
    summary["solve_strategy"] = "alternating_norel_main"
    summary["alternate_norel_time"] = float(alternate_norel_time)
    summary["alternate_main_time"] = float(alternate_main_time)
    summary["alternate_max_rounds"] = int(alternate_max_rounds)
    summary["phase_count"] = len(phase_rows)
    summary["incumbent_updates"] = int(incumbent_updates)
    summary["best_bound_seen"] = best_bound_seen

    return {"summary": summary, "outputs": outputs}


def solve_once(
    data,
    results_dir,
    time_limit,
    mip_gap,
    log_name,
    threads=None,
    server_limit=None,
    no_rel_heur_time=0.0,
    norel_pre_time=0.0,
    alternate_norel_time=0.0,
    alternate_main_time=0.0,
    alternate_max_rounds=6,
    cut_profile="baseline",
    start_values=None,
    hint_values=None,
    hint_priority=10,
    fixed_values=None,
):
    if alternate_norel_time and alternate_norel_time > 0.0 and alternate_main_time and alternate_main_time > 0.0:
        return solve_with_alternating_phases(
            data=data,
            results_dir=results_dir,
            time_limit=time_limit,
            mip_gap=mip_gap,
            threads=threads,
            server_limit=server_limit,
            alternate_norel_time=alternate_norel_time,
            alternate_main_time=alternate_main_time,
            alternate_max_rounds=alternate_max_rounds,
            cut_profile=cut_profile,
            start_values=start_values,
            hint_values=hint_values,
            hint_priority=hint_priority,
            fixed_values=fixed_values,
        )

    model, variables = build_model(
        data=data,
        results_dir=results_dir,
        time_limit=time_limit,
        mip_gap=mip_gap,
        log_name=log_name,
        threads=threads,
        server_limit=server_limit,
        no_rel_heur_time=no_rel_heur_time,
        cut_profile=cut_profile,
    )
    norel_summary = None
    norel_runtime = 0.0
    main_runtime = 0.0

    apply_fixed_values(variables, fixed_values)
    apply_var_hints(variables, hint_values, hint_priority=hint_priority)

    if norel_pre_time and norel_pre_time > 0.0:
        model.setParam("NoRelHeurTime", float(norel_pre_time))
        model.setParam("TimeLimit", float(norel_pre_time))
        apply_start_values(variables, start_values)
        model.optimize()

        norel_summary = build_summary(model, data)
        norel_runtime = float(model.Runtime)

        if model.Status != GRB.OPTIMAL:
            model.setParam("NoRelHeurTime", 0.0)
            if time_limit is not None and float(time_limit) > 0.0:
                model.setParam("TimeLimit", float(time_limit))
            else:
                model.setParam("TimeLimit", GRB.INFINITY)
            apply_start_values(variables, start_values)
            model.optimize()
            main_runtime = float(model.Runtime)
    else:
        apply_start_values(variables, start_values)
        model.optimize()
        main_runtime = float(model.Runtime)

    summary = build_summary(model, data)
    summary["runtime_seconds"] = norel_runtime + main_runtime
    summary["threads"] = threads
    summary["server_limit"] = server_limit
    summary["no_rel_heur_time"] = no_rel_heur_time
    summary["log_file"] = log_name
    summary["norel_phase_attempted"] = bool(norel_pre_time and norel_pre_time > 0.0)
    summary["norel_phase_time_limit"] = float(norel_pre_time) if norel_pre_time and norel_pre_time > 0.0 else 0.0
    summary["norel_phase_runtime_seconds"] = norel_runtime
    summary["main_phase_runtime_seconds"] = main_runtime

    if norel_summary is not None:
        summary["norel_phase_status_name"] = norel_summary["status_name"]
        summary["norel_phase_has_solution"] = norel_summary["has_solution"]
        summary["norel_phase_solution_count"] = norel_summary["solution_count"]
    else:
        summary["norel_phase_status_name"] = None
        summary["norel_phase_has_solution"] = False
        summary["norel_phase_solution_count"] = 0

    outputs = None
    if model.SolCount > 0:
        summary_updates, outputs = extract_solution_tables(data, variables)
        summary.update(summary_updates)

    return {"summary": summary, "outputs": outputs}


def choose_solution_record(base_record, fallback_record):
    if fallback_record is None or not fallback_record["summary"].get("has_solution"):
        return "base", base_record
    if not base_record["summary"].get("has_solution"):
        return "fallback", fallback_record

    base_summary = base_record["summary"]
    fallback_summary = fallback_record["summary"]
    base_servers = base_summary.get("used_server_count", float("inf"))
    fallback_servers = fallback_summary.get("used_server_count", float("inf"))

    if fallback_servers < base_servers:
        return "fallback", fallback_record
    if fallback_servers == base_servers:
        base_objective = base_summary.get("primary_objective_value", float("inf"))
        fallback_objective = fallback_summary.get("primary_objective_value", float("inf"))
        if fallback_objective < base_objective - 1e-9:
            return "fallback", fallback_record
    return "base", base_record


def solve_instance(
    instance_dir,
    results_dir,
    time_limit=1800,
    mip_gap=0.001,
    threads=None,
    server_limit=None,
    no_rel_heur_time=0.0,
    norel_pre_time=0.0,
    alternate_norel_time=0.0,
    alternate_main_time=0.0,
    alternate_max_rounds=6,
    cut_profile="baseline",
    use_fallback=False,
    start_values=None,
    hint_values=None,
    hint_priority=10,
    fixed_values=None,
):
    instance_dir = instance_dir.resolve()
    results_dir = results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    normalized_time_limit = None
    if time_limit is not None and float(time_limit) > 0.0:
        normalized_time_limit = float(time_limit)

    data = load_instance_data(instance_dir)
    base_record = solve_once(
        data=data,
        results_dir=results_dir,
        time_limit=normalized_time_limit,
        mip_gap=mip_gap,
        log_name="solver.log",
        threads=threads,
        server_limit=server_limit,
        no_rel_heur_time=no_rel_heur_time,
        norel_pre_time=norel_pre_time,
        alternate_norel_time=alternate_norel_time,
        alternate_main_time=alternate_main_time,
        alternate_max_rounds=alternate_max_rounds,
        cut_profile=cut_profile,
        start_values=start_values,
        hint_values=hint_values,
        hint_priority=hint_priority,
        fixed_values=fixed_values,
    )

    fallback_record = None
    fallback_server_limit = None
    base_summary = base_record["summary"]

    if (
        use_fallback
        and base_summary["status_name"] == "TIME_LIMIT"
        and base_summary.get("has_solution")
        and base_summary.get("used_server_count") is not None
        and int(base_summary["used_server_count"]) > 1
        and server_limit is None
    ):
        fallback_server_limit = int(base_summary["used_server_count"]) - 1
        fallback_no_rel_heur_time = no_rel_heur_time if no_rel_heur_time > 0.0 else float(normalized_time_limit)
        fallback_record = solve_once(
            data=data,
            results_dir=results_dir,
            time_limit=normalized_time_limit,
            mip_gap=mip_gap,
            log_name="solver_fallback.log",
            threads=threads,
            server_limit=fallback_server_limit,
            no_rel_heur_time=fallback_no_rel_heur_time,
            norel_pre_time=0.0,
            alternate_norel_time=0.0,
            alternate_main_time=0.0,
            alternate_max_rounds=1,
            cut_profile=cut_profile,
            start_values=start_values,
            hint_values=hint_values,
            hint_priority=hint_priority,
            fixed_values=fixed_values,
        )

    selected_source, selected_record = choose_solution_record(base_record, fallback_record)
    summary = selected_record["summary"]
    summary["selected_solution_source"] = selected_source
    summary["fallback_attempted"] = fallback_record is not None
    summary["base_status_name"] = base_summary["status_name"]
    summary["base_has_solution"] = base_summary["has_solution"]
    summary["base_used_server_count"] = base_summary.get("used_server_count")
    summary["time_limit"] = normalized_time_limit
    summary["total_runtime_seconds"] = float(
        base_summary.get("runtime_seconds", 0.0)
        + (fallback_record["summary"].get("runtime_seconds", 0.0) if fallback_record is not None else 0.0)
    )

    if fallback_record is not None:
        fallback_summary = fallback_record["summary"]
        summary["fallback_status_name"] = fallback_summary["status_name"]
        summary["fallback_has_solution"] = fallback_summary["has_solution"]
        summary["fallback_used_server_count"] = fallback_summary.get("used_server_count")
        summary["fallback_server_limit"] = fallback_server_limit
        summary["fallback_selected"] = selected_source == "fallback"
    else:
        summary["fallback_status_name"] = None
        summary["fallback_has_solution"] = False
        summary["fallback_used_server_count"] = None
        summary["fallback_server_limit"] = None
        summary["fallback_selected"] = False

    if selected_record["outputs"] is not None:
        write_solution_outputs(results_dir, summary, selected_record["outputs"])
    else:
        write_summary_only(results_dir, summary)

    if fallback_record is not None:
        with open(results_dir / "base_summary.json", "w", encoding="utf-8") as file:
            json.dump(base_record["summary"], file, indent=2)
        with open(results_dir / "fallback_summary.json", "w", encoding="utf-8") as file:
            json.dump(fallback_record["summary"], file, indent=2)

    return summary


def main():
    args = parse_args()
    instance_dir = args.instance_dir.resolve()
    results_dir = args.results_dir.resolve() if args.results_dir else (EXPERIMENT_DIR / "results" / "runs" / instance_dir.name)
    summary = solve_instance(
        instance_dir=instance_dir,
        results_dir=results_dir,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        threads=args.threads,
        server_limit=args.server_limit,
        no_rel_heur_time=args.no_rel_heur_time,
        norel_pre_time=args.norel_pre_time,
        alternate_norel_time=args.alternate_norel_time,
        alternate_main_time=args.alternate_main_time,
        alternate_max_rounds=args.alternate_max_rounds,
        cut_profile=args.cut_profile,
        use_fallback=args.use_fallback,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
