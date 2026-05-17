import copy
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


CUT_EXPERIMENT_DIR = Path(__file__).resolve().parents[1] / "2604-cut-experiment"
if str(CUT_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(CUT_EXPERIMENT_DIR))

from run_model import (  # noqa: E402
    apply_fixed_values,
    apply_var_hints,
    build_model,
    build_summary,
    capture_start_values,
)


FIRST_STAGE_NAMES = ("u", "u_used", "x", "m", "y", "z")


def filter_first_stage_values(start_values):
    return {name: start_values.get(name, {}) for name in FIRST_STAGE_NAMES}


def filter_named_values(value_map, names):
    if value_map is None:
        return None
    return {name: value_map.get(name, {}) for name in names}


def clone_data(data):
    return copy.deepcopy(data)


def split_equal_time_budget(total_time_limit, num_parts):
    if total_time_limit is None:
        return None
    num_parts = max(1, int(num_parts))
    return max(0.1, float(total_time_limit) / float(num_parts))


def make_aggregated_scenario_data(data, mode):
    reduced = clone_data(data)
    scenario_name = f"agg_{mode}"
    scenarios = list(data["scenarios"])

    if mode not in {"mean", "peak"}:
        raise ValueError(f"Unsupported aggregation mode: {mode}")

    def aggregate(values):
        if mode == "mean":
            return sum(values) / len(values)
        return max(values)

    reduced["scenarios"] = [scenario_name]
    reduced["scenario_prob"] = {scenario_name: 1.0}
    reduced["d_od"] = {
        (workload_id, time_value, scenario_name): aggregate(
            [data["d_od"][workload_id, time_value, original_scenario] for original_scenario in scenarios]
        )
        for workload_id in data["on_demand_ids"]
        for time_value in data["od_active"][workload_id]
    }
    reduced["d_sp"] = {
        (workload_id, time_value, scenario_name): aggregate(
            [data["d_sp"][workload_id, time_value, original_scenario] for original_scenario in scenarios]
        )
        for workload_id in data["spot_ids"]
        for time_value in data["spot_active"][workload_id]
    }
    reduced["d_batch"] = {
        (batch_job_id, scenario_name): aggregate(
            [data["d_batch"][batch_job_id, original_scenario] for original_scenario in scenarios]
        )
        for batch_job_id in data["batch_ids"]
    }
    reduced["instance"]["instance_name"] = f"{data['instance']['instance_name']}_{mode}"
    return reduced


def make_single_scenario_data(data, scenario_name):
    reduced = clone_data(data)
    reduced["scenarios"] = [scenario_name]
    reduced["scenario_prob"] = {scenario_name: 1.0}
    reduced["d_od"] = {
        (workload_id, time_value, scenario_name): data["d_od"][workload_id, time_value, scenario_name]
        for workload_id in data["on_demand_ids"]
        for time_value in data["od_active"][workload_id]
    }
    reduced["d_sp"] = {
        (workload_id, time_value, scenario_name): data["d_sp"][workload_id, time_value, scenario_name]
        for workload_id in data["spot_ids"]
        for time_value in data["spot_active"][workload_id]
    }
    reduced["d_batch"] = {
        (batch_job_id, scenario_name): data["d_batch"][batch_job_id, scenario_name]
        for batch_job_id in data["batch_ids"]
    }
    reduced["instance"]["instance_name"] = f"{data['instance']['instance_name']}_{scenario_name}"
    return reduced


def make_window_data(data, window_times):
    reduced = clone_data(data)
    window_times = sorted(int(time_value) for time_value in window_times)
    window_set = set(window_times)

    reduced["times"] = window_times
    reduced["od_active"] = {
        workload_id: [time_value for time_value in data["od_active"].get(workload_id, []) if time_value in window_set]
        for workload_id in data["on_demand_ids"]
    }
    reduced["od_transitions"] = {
        workload_id: list(zip(active_times[:-1], active_times[1:]))
        for workload_id, active_times in reduced["od_active"].items()
    }
    reduced["spot_active"] = {
        workload_id: [time_value for time_value in data["spot_active"].get(workload_id, []) if time_value in window_set]
        for workload_id in data["spot_ids"]
    }

    reduced["d_od"] = {
        (workload_id, time_value, scenario_name): demand
        for (workload_id, time_value, scenario_name), demand in data["d_od"].items()
        if time_value in window_set
    }
    reduced["d_sp"] = {
        (workload_id, time_value, scenario_name): demand
        for (workload_id, time_value, scenario_name), demand in data["d_sp"].items()
        if time_value in window_set
    }

    reduced["batch_ids"] = [
        batch_job_id
        for batch_job_id in data["batch_ids"]
        if int(data["batch_source_time"].get(batch_job_id, window_times[0])) in window_set
    ]
    reduced["d_batch"] = {
        (batch_job_id, scenario_name): demand
        for (batch_job_id, scenario_name), demand in data["d_batch"].items()
        if batch_job_id in reduced["batch_ids"]
    }
    reduced["batch_parents"] = {
        batch_job_id: parent_id
        for batch_job_id, parent_id in data["batch_parents"].items()
        if batch_job_id in reduced["batch_ids"]
    }
    reduced["batch_source_time"] = {
        batch_job_id: source_time
        for batch_job_id, source_time in data["batch_source_time"].items()
        if batch_job_id in reduced["batch_ids"]
    }
    reduced["instance"]["instance_name"] = (
        f"{data['instance']['instance_name']}_window_{window_times[0]}_{window_times[-1]}"
    )
    return reduced


def solve_data_for_start(
    data,
    results_dir,
    cut_profile,
    time_limit,
    threads,
    server_limit=None,
    mip_gap=0.01,
    log_name="start_solver.log",
    fixed_values=None,
    hint_values=None,
    hint_priority=10,
):
    results_dir.mkdir(parents=True, exist_ok=True)
    model, variables = build_model(
        data=data,
        results_dir=results_dir,
        time_limit=time_limit,
        mip_gap=mip_gap,
        log_name=log_name,
        threads=threads,
        server_limit=server_limit,
        cut_profile=cut_profile,
    )
    apply_fixed_values(variables, fixed_values)
    apply_var_hints(variables, hint_values, hint_priority=hint_priority)
    model.optimize()
    summary = build_summary(model, data)
    if model.SolCount <= 0:
        return None, summary
    return filter_first_stage_values(capture_start_values(variables)), summary


def representative_start(data, results_dir, cut_profile, mode, time_limit, threads):
    reduced = make_aggregated_scenario_data(data, mode)
    return solve_data_for_start(
        data=reduced,
        results_dir=results_dir / f"representative_{mode}",
        cut_profile=cut_profile,
        time_limit=time_limit,
        threads=threads,
    )


def single_scenario_best_start(data, results_dir, cut_profile, time_limit, threads):
    best_start = None
    best_summary = None
    scenario_summaries = []
    per_scenario_time = split_equal_time_budget(time_limit, len(data["scenarios"]))

    for scenario_name in data["scenarios"]:
        reduced = make_single_scenario_data(data, scenario_name)
        start_values, summary = solve_data_for_start(
            data=reduced,
            results_dir=results_dir / "single_scenario_best" / scenario_name,
            cut_profile=cut_profile,
            time_limit=per_scenario_time,
            threads=threads,
        )
        summary_row = {
            "scenario": scenario_name,
            "status_name": summary["status_name"],
            "has_solution": summary["has_solution"],
            "objective_value": summary["objective_value"],
            "used_server_count": summary.get("used_server_count"),
        }
        scenario_summaries.append(summary_row)
        if start_values is None:
            continue
        if best_summary is None:
            best_start = start_values
            best_summary = summary
            continue

        candidate_servers = summary.get("used_server_count", math.inf)
        incumbent_servers = best_summary.get("used_server_count", math.inf)
        candidate_objective = summary.get("objective_value", math.inf)
        incumbent_objective = best_summary.get("objective_value", math.inf)
        if candidate_servers < incumbent_servers or (
            candidate_servers == incumbent_servers and candidate_objective < incumbent_objective
        ):
            best_start = start_values
            best_summary = summary

    metadata = {
        "strategy": "single_scenario_best",
        "best_summary": best_summary,
        "scenario_summaries": scenario_summaries,
        "per_scenario_time_limit": per_scenario_time,
    }
    return best_start, metadata


def scenario_consensus_start(data, results_dir, cut_profile, time_limit, threads):
    scenario_starts = []
    scenario_summaries = []
    per_scenario_time = split_equal_time_budget(time_limit, len(data["scenarios"]))

    for scenario_name in data["scenarios"]:
        reduced = make_single_scenario_data(data, scenario_name)
        start_values, summary = solve_data_for_start(
            data=reduced,
            results_dir=results_dir / "scenario_consensus" / scenario_name,
            cut_profile=cut_profile,
            time_limit=per_scenario_time,
            threads=threads,
        )
        scenario_summaries.append(
            {
                "scenario": scenario_name,
                "status_name": summary["status_name"],
                "has_solution": summary["has_solution"],
                "objective_value": summary["objective_value"],
                "used_server_count": summary.get("used_server_count"),
            }
        )
        if start_values is not None:
            scenario_starts.append(start_values)

    if not scenario_starts:
        return None, {"strategy": "scenario_consensus", "scenario_summaries": scenario_summaries}

    start_values = {name: {} for name in FIRST_STAGE_NAMES}

    x_votes = defaultdict(list)
    y_votes = defaultdict(list)
    z_votes = defaultdict(list)

    for start in scenario_starts:
        for (workload_id, server, time_value), value in start["x"].items():
            if value > 0.5:
                x_votes[(workload_id, time_value)].append(server)
        for (workload_id, server), value in start["y"].items():
            if value > 0.5:
                y_votes[workload_id].append(server)
        for (batch_job_id, server, time_value), value in start["z"].items():
            if value > 0.5:
                z_votes[batch_job_id].append((server, time_value))

    x_choice = {}
    for workload_id in data["on_demand_ids"]:
        for time_value in data["od_active"][workload_id]:
            votes = x_votes.get((workload_id, time_value), [])
            chosen_server = Counter(votes).most_common(1)[0][0] if votes else 0
            x_choice[(workload_id, time_value)] = chosen_server

    y_choice = {}
    for workload_id in data["spot_ids"]:
        votes = y_votes.get(workload_id, [])
        chosen_server = Counter(votes).most_common(1)[0][0] if votes else 0
        y_choice[workload_id] = chosen_server

    z_choice = {}
    for batch_job_id in data["batch_ids"]:
        votes = z_votes.get(batch_job_id, [])
        chosen_pair = Counter(votes).most_common(1)[0][0] if votes else (0, data["times"][0])
        z_choice[batch_job_id] = chosen_pair

    occupied = {(server, time_value): 0 for server in data["servers"] for time_value in data["times"]}

    for workload_id in data["on_demand_ids"]:
        for time_value in data["od_active"][workload_id]:
            chosen_server = x_choice[(workload_id, time_value)]
            for server in data["servers"]:
                start_values["x"][workload_id, server, time_value] = 1.0 if server == chosen_server else 0.0
            occupied[chosen_server, time_value] = 1

    for workload_id in data["spot_ids"]:
        chosen_server = y_choice[workload_id]
        for server in data["servers"]:
            start_values["y"][workload_id, server] = 1.0 if server == chosen_server else 0.0
        for time_value in data["spot_active"][workload_id]:
            occupied[chosen_server, time_value] = 1

    for batch_job_id in data["batch_ids"]:
        chosen_server, chosen_time = z_choice[batch_job_id]
        for server in data["servers"]:
            for time_value in data["times"]:
                start_values["z"][batch_job_id, server, time_value] = (
                    1.0 if (server == chosen_server and time_value == chosen_time) else 0.0
                )
        occupied[chosen_server, chosen_time] = 1

    for server in data["servers"]:
        used = False
        for time_value in data["times"]:
            current = 1.0 if occupied[server, time_value] else 0.0
            start_values["u"][server, time_value] = current
            used = used or bool(current)
        start_values["u_used"][server] = 1.0 if used else 0.0

    for workload_id in data["on_demand_ids"]:
        for previous_time, current_time in data["od_transitions"][workload_id]:
            previous_server = x_choice[(workload_id, previous_time)]
            current_server = x_choice[(workload_id, current_time)]
            start_values["m"][workload_id, previous_time, current_time] = (
                1.0 if previous_server != current_server else 0.0
            )

    metadata = {
        "strategy": "scenario_consensus",
        "scenario_summaries": scenario_summaries,
        "consensus_scenario_count": len(scenario_starts),
        "per_scenario_time_limit": per_scenario_time,
    }
    return start_values, metadata


def estimate_server_lower_bound(data):
    scenario_name = data["scenarios"][0]
    peak_fixed_load = 0.0
    for time_value in data["times"]:
        time_load = 0.0
        for workload_id in data["on_demand_ids"]:
            if time_value in data["od_active"][workload_id]:
                time_load += data["d_od"][workload_id, time_value, scenario_name]
        for workload_id in data["spot_ids"]:
            if time_value in data["spot_active"][workload_id]:
                time_load += data["d_sp"][workload_id, time_value, scenario_name]
        peak_fixed_load = max(peak_fixed_load, time_load)

    average_batch_load = 0.0
    if data["batch_ids"]:
        average_batch_load = (
            sum(data["d_batch"][batch_job_id, scenario_name] for batch_job_id in data["batch_ids"]) / len(data["times"])
        )

    total_required = peak_fixed_load + average_batch_load
    return max(1, math.ceil(total_required / data["capacity"]))


def progressive_server_limit_start(data, results_dir, cut_profile, mode, time_limit, threads):
    reduced = make_aggregated_scenario_data(data, mode)
    lower_bound = estimate_server_lower_bound(reduced)
    attempts = []
    server_limits = list(range(lower_bound, len(reduced["servers"]) + 1))
    per_attempt_time = split_equal_time_budget(time_limit, len(server_limits))

    for server_limit in server_limits:
        start_values, summary = solve_data_for_start(
            data=reduced,
            results_dir=results_dir / f"progressive_{mode}" / f"limit_{server_limit}",
            cut_profile=cut_profile,
            time_limit=per_attempt_time,
            threads=threads,
            server_limit=server_limit,
        )
        attempts.append(
            {
                "server_limit": server_limit,
                "status_name": summary["status_name"],
                "has_solution": summary["has_solution"],
                "objective_value": summary["objective_value"],
                "used_server_count": summary.get("used_server_count"),
            }
        )
        if start_values is not None:
            return start_values, {
                "strategy": f"progressive_{mode}",
                "attempts": attempts,
                "per_attempt_time_limit": per_attempt_time,
            }

    return None, {
        "strategy": f"progressive_{mode}",
        "attempts": attempts,
        "per_attempt_time_limit": per_attempt_time,
    }


def finalize_first_stage_start(data, start_values, x_choice=None):
    servers = data["servers"]
    times = data["times"]

    if x_choice is None:
        x_choice = {}
        for workload_id in data["on_demand_ids"]:
            for time_value in data["od_active"][workload_id]:
                chosen_server = next(
                    (
                        server
                        for server in servers
                        if start_values["x"].get((workload_id, server, time_value), 0.0) > 0.5
                    ),
                    servers[0],
                )
                x_choice[(workload_id, time_value)] = chosen_server

    for server in servers:
        used = False
        for time_value in times:
            occupied = 0.0
            if any(
                start_values["x"].get((workload_id, server, time_value), 0.0) > 0.5
                for workload_id in data["on_demand_ids"]
                if time_value in data["od_active"][workload_id]
            ):
                occupied = 1.0
            if any(
                start_values["y"].get((workload_id, server), 0.0) > 0.5
                for workload_id in data["spot_ids"]
                if time_value in data["spot_active"][workload_id]
            ):
                occupied = 1.0
            if any(
                start_values["z"].get((batch_job_id, server, time_value), 0.0) > 0.5
                for batch_job_id in data["batch_ids"]
            ):
                occupied = 1.0
            start_values["u"][server, time_value] = occupied
            used = used or occupied > 0.5
        start_values["u_used"][server] = 1.0 if used else 0.0

    for workload_id in data["on_demand_ids"]:
        for previous_time, current_time in data["od_transitions"][workload_id]:
            previous_host = x_choice[(workload_id, previous_time)]
            current_host = x_choice[(workload_id, current_time)]
            start_values["m"][workload_id, previous_time, current_time] = 1.0 if previous_host != current_host else 0.0

    return start_values


def greedy_rule_start(data, mode):
    reduced = make_aggregated_scenario_data(data, mode)
    scenario_name = reduced["scenarios"][0]
    servers = reduced["servers"]
    times = reduced["times"]
    capacity = reduced["capacity"]

    start_values = {name: {} for name in FIRST_STAGE_NAMES}
    load_map = {(server, time_value): 0.0 for server in servers for time_value in times}
    x_choice = {}

    spot_workloads = sorted(
        reduced["spot_ids"],
        key=lambda workload_id: max(reduced["d_sp"][workload_id, time_value, scenario_name] for time_value in reduced["spot_active"][workload_id]),
        reverse=True,
    )
    for workload_id in spot_workloads:
        best_server = None
        best_score = None
        for server in servers:
            candidate_score = max(
                (load_map[server, time_value] + reduced["d_sp"][workload_id, time_value, scenario_name]) / capacity
                for time_value in reduced["spot_active"][workload_id]
            )
            if best_score is None or candidate_score < best_score - 1e-9:
                best_server = server
                best_score = candidate_score
        for server in servers:
            start_values["y"][workload_id, server] = 1.0 if server == best_server else 0.0
        for time_value in reduced["spot_active"][workload_id]:
            load_map[best_server, time_value] += reduced["d_sp"][workload_id, time_value, scenario_name]

    previous_server = {}
    for time_value in times:
        active_on_demand = [
            workload_id for workload_id in reduced["on_demand_ids"] if time_value in reduced["od_active"][workload_id]
        ]
        active_on_demand.sort(
            key=lambda workload_id: reduced["d_od"][workload_id, time_value, scenario_name],
            reverse=True,
        )

        for workload_id in active_on_demand:
            demand = reduced["d_od"][workload_id, time_value, scenario_name]
            chosen_server = None
            preferred_server = previous_server.get(workload_id)
            if preferred_server is not None and load_map[preferred_server, time_value] + demand <= capacity + 1e-9:
                chosen_server = preferred_server
            else:
                feasible_servers = [
                    server for server in servers if load_map[server, time_value] + demand <= capacity + 1e-9
                ]
                if feasible_servers:
                    chosen_server = min(feasible_servers, key=lambda server: load_map[server, time_value])
                else:
                    chosen_server = min(servers, key=lambda server: load_map[server, time_value])

            x_choice[(workload_id, time_value)] = chosen_server
            previous_server[workload_id] = chosen_server
            load_map[chosen_server, time_value] += demand

    for workload_id in reduced["on_demand_ids"]:
        for time_value in reduced["od_active"][workload_id]:
            chosen_server = x_choice[(workload_id, time_value)]
            for server in servers:
                start_values["x"][workload_id, server, time_value] = 1.0 if server == chosen_server else 0.0

    batch_jobs = sorted(
        reduced["batch_ids"],
        key=lambda batch_job_id: reduced["d_batch"][batch_job_id, scenario_name],
        reverse=True,
    )
    for batch_job_id in batch_jobs:
        demand = reduced["d_batch"][batch_job_id, scenario_name]
        source_time = int(reduced["batch_source_time"].get(batch_job_id, times[0]))
        candidate_slots = sorted(times, key=lambda time_value: (abs(time_value - source_time), time_value))
        chosen_server = None
        chosen_time = None

        for time_value in candidate_slots:
            feasible_servers = [server for server in servers if load_map[server, time_value] + demand <= capacity + 1e-9]
            if feasible_servers:
                chosen_time = time_value
                chosen_server = min(feasible_servers, key=lambda server: load_map[server, time_value])
                break

        if chosen_server is None:
            chosen_time = min(times, key=lambda time_value: min(load_map[server, time_value] for server in servers))
            chosen_server = min(servers, key=lambda server: load_map[server, chosen_time])

        load_map[chosen_server, chosen_time] += demand
        for server in servers:
            for time_value in times:
                start_values["z"][batch_job_id, server, time_value] = (
                    1.0 if (server == chosen_server and time_value == chosen_time) else 0.0
                )

    finalize_first_stage_start(reduced, start_values, x_choice=x_choice)

    metadata = {
        "strategy": f"greedy_rule_{mode}",
        "max_utilization": max(load_map.values()) / capacity if load_map else 0.0,
    }
    return start_values, metadata


def kwon_threshold_start(data, mode):
    reduced = make_aggregated_scenario_data(data, mode)
    scenario_name = reduced["scenarios"][0]
    servers = reduced["servers"]
    times = reduced["times"]
    capacity = reduced["capacity"]

    start_values = {name: {} for name in FIRST_STAGE_NAMES}
    load_map = {(server, time_value): 0.0 for server in servers for time_value in times}
    x_choice = {}

    threshold_servers = {}
    for time_value in times:
        expected_load = 0.0
        for workload_id in reduced["on_demand_ids"]:
            if time_value in reduced["od_active"][workload_id]:
                expected_load += reduced["d_od"][workload_id, time_value, scenario_name]
        for workload_id in reduced["spot_ids"]:
            if time_value in reduced["spot_active"][workload_id]:
                expected_load += reduced["d_sp"][workload_id, time_value, scenario_name]
        threshold_servers[time_value] = max(1, math.ceil(expected_load / capacity))

    od_order = sorted(
        reduced["on_demand_ids"],
        key=lambda workload_id: (
            -sum(reduced["d_od"][workload_id, time_value, scenario_name] for time_value in reduced["od_active"][workload_id]),
            len(reduced["od_active"][workload_id]),
        ),
    )
    current_server = {}
    for workload_id in od_order:
        active_times = reduced["od_active"][workload_id]
        for time_value in active_times:
            demand = reduced["d_od"][workload_id, time_value, scenario_name]
            allowed_servers = servers[: threshold_servers[time_value]]
            chosen_server = current_server.get(workload_id)
            if (
                chosen_server is not None
                and chosen_server in allowed_servers
                and load_map[chosen_server, time_value] + demand <= capacity + 1e-9
            ):
                pass
            else:
                feasible_servers = [
                    server for server in allowed_servers if load_map[server, time_value] + demand <= capacity + 1e-9
                ]
                if feasible_servers:
                    chosen_server = min(feasible_servers, key=lambda server: load_map[server, time_value])
                else:
                    chosen_server = min(allowed_servers, key=lambda server: load_map[server, time_value])
            x_choice[(workload_id, time_value)] = chosen_server
            current_server[workload_id] = chosen_server
            load_map[chosen_server, time_value] += demand

    for workload_id in reduced["on_demand_ids"]:
        for time_value in reduced["od_active"][workload_id]:
            chosen_server = x_choice[(workload_id, time_value)]
            for server in servers:
                start_values["x"][workload_id, server, time_value] = 1.0 if server == chosen_server else 0.0

    spot_order = sorted(
        reduced["spot_ids"],
        key=lambda workload_id: max(
            reduced["d_sp"][workload_id, time_value, scenario_name] for time_value in reduced["spot_active"][workload_id]
        ),
        reverse=True,
    )
    for workload_id in spot_order:
        best_server = None
        best_score = None
        for server in servers:
            score = 0.0
            for time_value in reduced["spot_active"][workload_id]:
                allowed_servers = servers[: threshold_servers[time_value]]
                open_penalty = 0.0 if server in allowed_servers else 1.0
                score = max(
                    score,
                    (load_map[server, time_value] + reduced["d_sp"][workload_id, time_value, scenario_name]) / capacity
                    + open_penalty,
                )
            if best_score is None or score < best_score - 1e-9:
                best_server = server
                best_score = score
        for server in servers:
            start_values["y"][workload_id, server] = 1.0 if server == best_server else 0.0
        for time_value in reduced["spot_active"][workload_id]:
            load_map[best_server, time_value] += reduced["d_sp"][workload_id, time_value, scenario_name]

    batch_jobs = sorted(
        reduced["batch_ids"],
        key=lambda batch_job_id: (
            int(reduced["batch_source_time"].get(batch_job_id, times[0])),
            -reduced["d_batch"][batch_job_id, scenario_name],
        ),
    )
    for batch_job_id in batch_jobs:
        demand = reduced["d_batch"][batch_job_id, scenario_name]
        source_time = int(reduced["batch_source_time"].get(batch_job_id, times[0]))
        candidate_times = sorted(times, key=lambda time_value: (abs(time_value - source_time), time_value))
        chosen_server = None
        chosen_time = None
        best_score = None
        for time_value in candidate_times:
            allowed_servers = servers[: threshold_servers[time_value]]
            for server in allowed_servers:
                score = load_map[server, time_value] + demand
                if best_score is None or score < best_score - 1e-9:
                    chosen_server = server
                    chosen_time = time_value
                    best_score = score
        load_map[chosen_server, chosen_time] += demand
        for server in servers:
            for time_value in times:
                start_values["z"][batch_job_id, server, time_value] = (
                    1.0 if (server == chosen_server and time_value == chosen_time) else 0.0
                )

    finalize_first_stage_start(reduced, start_values, x_choice=x_choice)
    metadata = {
        "strategy": f"kwon_threshold_{mode}",
        "threshold_servers": threshold_servers,
        "max_utilization": max(load_map.values()) / capacity if load_map else 0.0,
    }
    return start_values, metadata


def rolling_horizon_start(data, results_dir, cut_profile, mode, time_limit, threads, window_size=2):
    reduced = make_aggregated_scenario_data(data, mode)
    times = reduced["times"]
    window_count = max(1, math.ceil(len(times) / int(window_size)))
    per_stage_time = split_equal_time_budget(time_limit, 1 + window_count)
    guide_start, guide_summary = representative_start(
        data=data,
        results_dir=results_dir / f"rolling_horizon_{mode}" / "guide",
        cut_profile=cut_profile,
        mode=mode,
        time_limit=per_stage_time,
        threads=threads,
    )
    if guide_start is None:
        return None, {
            "strategy": f"rolling_horizon_{mode}",
            "guide_summary": guide_summary,
            "windows": [],
            "per_stage_time_limit": per_stage_time,
        }

    start_values = {name: {} for name in FIRST_STAGE_NAMES}
    for workload_id in reduced["spot_ids"]:
        for server in reduced["servers"]:
            start_values["y"][workload_id, server] = guide_start["y"].get((workload_id, server), 0.0)

    window_rows = []
    x_choice = {}
    for start_index in range(0, len(times), int(window_size)):
        window_times = times[start_index : start_index + int(window_size)]
        window_data = make_window_data(reduced, window_times)
        fixed_values = filter_named_values(guide_start, ("y",))
        window_start, window_summary = solve_data_for_start(
            data=window_data,
            results_dir=results_dir / f"rolling_horizon_{mode}" / f"window_{window_times[0]}_{window_times[-1]}",
            cut_profile=cut_profile,
            time_limit=per_stage_time,
            threads=threads,
            fixed_values=fixed_values,
        )
        window_rows.append(
            {
                "window_start": window_times[0],
                "window_end": window_times[-1],
                "status_name": window_summary["status_name"],
                "has_solution": window_summary["has_solution"],
                "objective_value": window_summary["objective_value"],
            }
        )
        if window_start is None:
            continue

        for workload_id in window_data["on_demand_ids"]:
            for time_value in window_data["od_active"][workload_id]:
                for server in window_data["servers"]:
                    value = window_start["x"].get((workload_id, server, time_value), 0.0)
                    start_values["x"][workload_id, server, time_value] = value
                    if value > 0.5:
                        x_choice[(workload_id, time_value)] = server

        for batch_job_id in window_data["batch_ids"]:
            for server in window_data["servers"]:
                for time_value in window_data["times"]:
                    start_values["z"][batch_job_id, server, time_value] = window_start["z"].get(
                        (batch_job_id, server, time_value),
                        0.0,
                    )

    for workload_id in reduced["spot_ids"]:
        chosen_server = next(
            (server for server in reduced["servers"] if start_values["y"].get((workload_id, server), 0.0) > 0.5),
            reduced["servers"][0],
        )
        for server in reduced["servers"]:
            start_values["y"][workload_id, server] = 1.0 if server == chosen_server else 0.0

    for workload_id in reduced["on_demand_ids"]:
        active_times = reduced["od_active"][workload_id]
        if not active_times:
            continue
        fallback_server = next(
            (
                server
                for server in reduced["servers"]
                if guide_start["x"].get((workload_id, server, active_times[0]), 0.0) > 0.5
            ),
            reduced["servers"][0],
        )
        for time_value in active_times:
            if (workload_id, time_value) not in x_choice:
                x_choice[(workload_id, time_value)] = fallback_server
                for server in reduced["servers"]:
                    start_values["x"][workload_id, server, time_value] = 1.0 if server == fallback_server else 0.0

    for batch_job_id in reduced["batch_ids"]:
        if any(
            start_values["z"].get((batch_job_id, server, time_value), 0.0) > 0.5
            for server in reduced["servers"]
            for time_value in reduced["times"]
        ):
            continue
        source_time = int(reduced["batch_source_time"].get(batch_job_id, reduced["times"][0]))
        chosen_server = min(reduced["servers"])
        for server in reduced["servers"]:
            for time_value in reduced["times"]:
                start_values["z"][batch_job_id, server, time_value] = (
                    1.0 if (server == chosen_server and time_value == source_time) else 0.0
                )

    finalize_first_stage_start(reduced, start_values, x_choice=x_choice)
    metadata = {
        "strategy": f"rolling_horizon_{mode}",
        "guide_summary": guide_summary,
        "windows": window_rows,
        "window_size": int(window_size),
        "per_stage_time_limit": per_stage_time,
    }
    return start_values, metadata


def build_start_strategy(strategy_name, data, results_dir, cut_profile, strategy_time_limit, strategy_threads):
    if strategy_name == "representative_mean":
        return representative_start(data, results_dir, cut_profile, "mean", strategy_time_limit, strategy_threads)
    if strategy_name == "representative_peak":
        return representative_start(data, results_dir, cut_profile, "peak", strategy_time_limit, strategy_threads)
    if strategy_name == "single_scenario_best":
        return single_scenario_best_start(data, results_dir, cut_profile, strategy_time_limit, strategy_threads)
    if strategy_name == "scenario_consensus":
        return scenario_consensus_start(data, results_dir, cut_profile, strategy_time_limit, strategy_threads)
    if strategy_name == "progressive_mean":
        return progressive_server_limit_start(data, results_dir, cut_profile, "mean", strategy_time_limit, strategy_threads)
    if strategy_name == "greedy_rule_mean":
        return greedy_rule_start(data, "mean")
    if strategy_name == "kwon_threshold_mean":
        return kwon_threshold_start(data, "mean")
    if strategy_name == "rolling_horizon_mean":
        return rolling_horizon_start(
            data,
            results_dir,
            cut_profile,
            "mean",
            strategy_time_limit,
            strategy_threads,
            window_size=4,
        )
    if strategy_name == "rolling_horizon_peak":
        return rolling_horizon_start(
            data,
            results_dir,
            cut_profile,
            "peak",
            strategy_time_limit,
            strategy_threads,
            window_size=4,
        )
    raise ValueError(f"Unknown start strategy: {strategy_name}")
