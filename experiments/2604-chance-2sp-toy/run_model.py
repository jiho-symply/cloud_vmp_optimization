import argparse
import json
from pathlib import Path

import gurobipy as gp
import pandas as pd
from gurobipy import GRB


EXPERIMENT_DIR = Path(__file__).resolve().parent
EXPERIMENT_NAME = EXPERIMENT_DIR.name
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DEFAULT_INSTANCE_DIR = REPO_ROOT / "data" / "processed" / EXPERIMENT_NAME / "chance_2sp_toy_30vm_od10_sp10_bj10_sc10_cap4"


def parse_args():
    parser = argparse.ArgumentParser(description="chance-constrained 2SP toy model을 풉니다.")
    parser.add_argument("--instance-dir", type=Path, default=DEFAULT_INSTANCE_DIR)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--time-limit", type=int, default=1800)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--server-limit", type=int, default=None)
    parser.add_argument("--no-rel-heur-time", type=float, default=0.0)
    parser.add_argument("--use-fallback", action="store_true")
    return parser.parse_args()


def status_name(status_code):
    status_names = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    return status_names.get(status_code, str(status_code))


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
        "lambda_migration": float(instance["objective"]["lambda_migration"]),
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
    model.addConstrs(u_used[s] >= u_used[s + 1] for s in servers[:-1])


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
    model.addConstrs(
        gamma[s, t, xi] <= gp.quicksum(y[j, s] for j in spot_ids if t in spot_active[j])
        for s in servers
        for t in times
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
                model.addConstr(phi[s, t, xi] <= gamma[s, t, xi])
                model.addConstr(eta[s, xi] >= phi[s, t, xi])

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


def set_objective(model, data, variables):
    servers = data["servers"]
    on_demand_ids = data["on_demand_ids"]
    od_transitions = data["od_transitions"]
    lambda_migration = data["lambda_migration"]

    u_used = variables["u_used"]
    m = variables["m"]

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
):
    model = gp.Model("chance_constrained_2sp_toy")
    model.setParam("TimeLimit", time_limit)
    model.setParam("MIPGap", mip_gap)
    model.setParam("LogFile", str(results_dir / log_name))
    if threads is not None:
        model.setParam("Threads", threads)
    if no_rel_heur_time and no_rel_heur_time > 0.0:
        model.setParam("NoRelHeurTime", float(no_rel_heur_time))

    variables = build_variables(model, data)
    add_first_stage_constraints(model, data, variables)
    add_second_stage_constraints(model, data, variables)
    if server_limit is not None:
        model.addConstr(
            gp.quicksum(variables["u_used"][s] for s in data["servers"]) <= int(server_limit),
            name="server_limit",
        )
    set_objective(model, data, variables)
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
        "instance_name": data["instance"]["instance_name"],
        "server_capacity": data["capacity"],
        "num_servers": len(data["servers"]),
        "epsilon_od": data["epsilon_od"],
        "epsilon_sp": data["epsilon_sp"],
        "rho": data["rho"],
        "lambda_migration": data["lambda_migration"],
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

    overbooking_peak_row = scenario_state.sort_values(
        ["overbooking_ratio", "committed_load", "scenario", "server", "time"],
        ascending=[False, False, True, True, True],
    ).iloc[0]
    realized_peak_row = scenario_state.sort_values(
        ["realized_utilization", "load", "scenario", "server", "time"],
        ascending=[False, False, True, True, True],
    ).iloc[0]

    normalization = len(on_demand_ids) if on_demand_ids else 1
    primary_objective_value = float(len(used_servers) + lambda_migration * migration_count / normalization)

    summary_updates = {
        "used_server_count": len(used_servers),
        "used_servers": used_servers,
        "primary_objective_value": primary_objective_value,
        "migration_count": float(migration_count),
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


def solve_once(
    data,
    results_dir,
    time_limit,
    mip_gap,
    log_name,
    threads=None,
    server_limit=None,
    no_rel_heur_time=0.0,
):
    model, variables = build_model(
        data=data,
        results_dir=results_dir,
        time_limit=time_limit,
        mip_gap=mip_gap,
        log_name=log_name,
        threads=threads,
        server_limit=server_limit,
        no_rel_heur_time=no_rel_heur_time,
    )
    model.optimize()

    summary = build_summary(model, data)
    summary["threads"] = threads
    summary["server_limit"] = server_limit
    summary["no_rel_heur_time"] = no_rel_heur_time
    summary["log_file"] = log_name

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
    use_fallback=False,
):
    instance_dir = instance_dir.resolve()
    results_dir = results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    data = load_instance_data(instance_dir)
    base_record = solve_once(
        data=data,
        results_dir=results_dir,
        time_limit=time_limit,
        mip_gap=mip_gap,
        log_name="solver.log",
        threads=threads,
        server_limit=server_limit,
        no_rel_heur_time=no_rel_heur_time,
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
        fallback_no_rel_heur_time = no_rel_heur_time if no_rel_heur_time > 0.0 else float(time_limit)
        fallback_record = solve_once(
            data=data,
            results_dir=results_dir,
            time_limit=time_limit,
            mip_gap=mip_gap,
            log_name="solver_fallback.log",
            threads=threads,
            server_limit=fallback_server_limit,
            no_rel_heur_time=fallback_no_rel_heur_time,
        )

    selected_source, selected_record = choose_solution_record(base_record, fallback_record)
    summary = selected_record["summary"]
    summary["selected_solution_source"] = selected_source
    summary["fallback_attempted"] = fallback_record is not None
    summary["base_status_name"] = base_summary["status_name"]
    summary["base_has_solution"] = base_summary["has_solution"]
    summary["base_used_server_count"] = base_summary.get("used_server_count")
    summary["time_limit"] = time_limit
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
        use_fallback=args.use_fallback,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
