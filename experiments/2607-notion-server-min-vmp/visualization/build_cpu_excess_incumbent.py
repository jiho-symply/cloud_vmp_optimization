#!/usr/bin/env python3
"""Build a compact, self-contained visualization data payload for one incumbent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


CPU = "cpu"
MEM = "mem"


def _round(value: Any, digits: int = 6) -> Any:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _series(values: list[float], digits: int = 4) -> list[float]:
    return [round(float(value), digits) for value in values]


def _short_ids(values: list[str], prefix: str) -> dict[str, str]:
    return {value: f"{prefix}{position + 1:02d}" for position, value in enumerate(values)}


def _load(run_dir: Path) -> dict[str, pd.DataFrame]:
    prepared = run_dir / "prepared_data"
    return {
        "load": pd.read_csv(run_dir / "server_scenario_load.csv"),
        "scenario": pd.read_csv(run_dir / "scenario_metrics.csv"),
        "server": pd.read_csv(run_dir / "server_schedule.csv"),
        "od": pd.read_csv(run_dir / "od_schedule.csv"),
        "od_initial": pd.read_csv(run_dir / "od_initial_placement.csv"),
        "spot": pd.read_csv(run_dir / "spot_schedule.csv"),
        "batch": pd.read_csv(run_dir / "batch_schedule.csv"),
        "od_params": pd.read_csv(prepared / "on_demand_parameters.csv"),
        "spot_params": pd.read_csv(prepared / "spot_parameters.csv"),
        "spot_usage": pd.read_csv(prepared / "spot_usage.csv"),
        "batch_params": pd.read_csv(prepared / "batch_parameters.csv"),
        "server_params": pd.read_csv(prepared / "server_parameters.csv"),
        "price": pd.read_csv(prepared / "rt_prices.csv"),
    }


def _build_payload(run_dir: Path) -> dict[str, Any]:
    tables = _load(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    data_audit = json.loads(
        (run_dir / "prepared_data" / "data_audit.json").read_text(encoding="utf-8")
    )
    config = json.loads(json.dumps({}))
    # PyYAML is already an experiment dependency, but import it lazily so this
    # builder remains usable for inspecting an already materialized run.
    import yaml

    config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))

    servers = sorted(tables["load"]["server_id"].unique().tolist())
    scenarios = sorted(int(value) for value in tables["scenario"]["scenario_id"].unique())
    periods = sorted(int(value) for value in tables["load"]["t"].unique())
    server_index = {server: position for position, server in enumerate(servers)}
    scenario_index = {scenario: position for position, scenario in enumerate(scenarios)}
    horizon = len(periods)

    capacities = (
        tables["server_params"].set_index("server_id")[["C_cpu", "C_mem"]].to_dict("index")
    )
    batch_parameters = tables["batch_params"].set_index("family_id").to_dict("index")
    batch_family_metadata = {
        str(row["family_id"]): row
        for row in data_audit["metadata"]["batch"]["families"]
    }
    total_batch_work = float(tables["batch_params"]["workload"].sum())

    spot = tables["spot"].merge(
        tables["spot_usage"],
        on=["scenario_id", "vm_id", "t"],
        how="left",
        validate="one_to_one",
    )
    active_spot = spot.loc[spot["active_server"].notna()].copy()
    preempted_spot = spot.loc[
        spot["preempted"].eq(1) & spot["initial_server"].notna()
    ].copy()

    spot_components = (
        active_spot.groupby(["scenario_id", "active_server", "t"], as_index=False)[
            ["cpu_usage", "mem_usage"]
        ]
        .sum()
        .rename(columns={"active_server": "server_id"})
    )
    preempted_spot_components = (
        preempted_spot.groupby(
            ["scenario_id", "initial_server", "t"], as_index=False
        )[["cpu_usage", "mem_usage"]]
        .sum()
        .rename(columns={"initial_server": "server_id"})
    )

    batch = tables["batch"].copy()
    batch["cpu_component"] = batch["cpu_workload"] + batch.apply(
        lambda row: batch_parameters[row["family_id"]]["startup_cpu"] * row["started"],
        axis=1,
    )
    batch["mem_component"] = batch["mem_workload"] + batch.apply(
        lambda row: batch_parameters[row["family_id"]]["base_mem"] * row["prepared"],
        axis=1,
    )
    batch_components = (
        batch.groupby(["scenario_id", "server_id", "t"], as_index=False)[
            ["cpu_component", "mem_component", "workload"]
        ].sum()
    )

    component_keys = pd.MultiIndex.from_product(
        [scenarios, servers, periods],
        names=["scenario_id", "server_id", "t"],
    )
    ledger = tables["load"].set_index(["scenario_id", "server_id", "t"]).reindex(component_keys)
    ledger = ledger.join(
        spot_components.set_index(["scenario_id", "server_id", "t"]).rename(
            columns={"cpu_usage": "spot_cpu", "mem_usage": "spot_mem"}
        )
    )
    ledger = ledger.join(
        preempted_spot_components.set_index(
            ["scenario_id", "server_id", "t"]
        ).rename(
            columns={
                "cpu_usage": "preempted_spot_cpu",
                "mem_usage": "preempted_spot_mem",
            }
        )
    )
    ledger = ledger.join(
        batch_components.set_index(["scenario_id", "server_id", "t"])
    )
    for column in (
        "spot_cpu",
        "spot_mem",
        "preempted_spot_cpu",
        "preempted_spot_mem",
        "cpu_component",
        "mem_component",
        "workload",
    ):
        ledger[column] = ledger[column].fillna(0.0)
    ledger["post_preemption_cpu_demand"] = (
        ledger["od_cpu_demand"] + ledger["spot_cpu"] + ledger["cpu_component"]
    )
    ledger["post_preemption_mem_load"] = (
        ledger["od_mem_served"] + ledger["spot_mem"] + ledger["mem_component"]
    )
    ledger["pre_preemption_cpu_demand"] = (
        ledger["post_preemption_cpu_demand"] + ledger["preempted_spot_cpu"]
    )
    ledger["pre_preemption_mem_load"] = (
        ledger["post_preemption_mem_load"] + ledger["preempted_spot_mem"]
    )

    cpu_reconstructed = (
        ledger["od_cpu_served"] + ledger["spot_cpu"] + ledger["cpu_component"]
    )
    mem_reconstructed = ledger["od_mem_served"] + ledger["spot_mem"] + ledger["mem_component"]
    cpu_residual = float((cpu_reconstructed - ledger["total_cpu_load"]).abs().max())
    mem_residual = float((mem_reconstructed - ledger["total_mem_load"]).abs().max())
    post_cpu_identity_residual = float(
        (
            ledger["post_preemption_cpu_demand"]
            - ledger["total_cpu_load"]
            - ledger["cpu_excess"]
        )
        .abs()
        .max()
    )

    spot_event_rows: list[dict[str, Any]] = []
    for (scenario_id, vm_id), group in spot.groupby(["scenario_id", "vm_id"], sort=False):
        ordered = group.sort_values("t")
        previous = 0
        for row in ordered.itertuples(index=False):
            state = int(row.preempted)
            event = int(state == 1 and previous == 0)
            if event:
                spot_event_rows.append(
                    {
                        "scenario_id": int(scenario_id),
                        "vm_id": str(vm_id),
                        "t": int(row.t),
                    }
                )
            previous = state
    spot_events = pd.DataFrame(
        spot_event_rows, columns=["scenario_id", "vm_id", "t"]
    )
    event_counts = (
        spot_events.groupby(["scenario_id", "t"]).size().to_dict()
        if not spot_events.empty
        else {}
    )
    event_vms = (
        spot_events.groupby(["scenario_id", "t"])["vm_id"].apply(list).to_dict()
        if not spot_events.empty
        else {}
    )
    event_totals = (
        spot_events.groupby("scenario_id").size().to_dict()
        if not spot_events.empty
        else {}
    )
    state_totals = spot.groupby("scenario_id")["preempted"].sum().to_dict()

    atlas: list[dict[str, Any]] = []
    for scenario_id in scenarios:
        scenario_ledger = ledger.xs(scenario_id, level="scenario_id")
        post_cpu: list[float] = []
        post_mem: list[float] = []
        served_cpu_at_post_peak: list[float] = []
        excess_at_post_peak: list[float] = []
        pre_cpu: list[float] = []
        post_cpu_at_pre_peak: list[float] = []
        preempted_spot_cpu_at_pre_peak: list[float] = []
        pre_mem: list[float] = []
        post_mem_at_pre_peak: list[float] = []
        preempted_spot_mem_at_pre_peak: list[float] = []
        excess: list[float] = []
        post_cpu_server: list[str] = []
        post_mem_server: list[str] = []
        pre_cpu_server: list[str] = []
        pre_mem_server: list[str] = []
        excess_server: list[str] = []
        batch_share: list[float] = []
        events: list[int] = []
        event_vm_ids: list[list[str]] = []
        for t in periods:
            frame = scenario_ledger.xs(t, level="t")
            cpu_ratios = {
                server: (
                    frame.loc[server, "post_preemption_cpu_demand"]
                    / capacities[server]["C_cpu"]
                )
                for server in servers
            }
            mem_ratios = {
                server: (
                    frame.loc[server, "post_preemption_mem_load"]
                    / capacities[server]["C_mem"]
                )
                for server in servers
            }
            pre_cpu_ratios = {
                server: (
                    frame.loc[server, "pre_preemption_cpu_demand"]
                    / capacities[server]["C_cpu"]
                )
                for server in servers
            }
            pre_mem_ratios = {
                server: (
                    frame.loc[server, "pre_preemption_mem_load"]
                    / capacities[server]["C_mem"]
                )
                for server in servers
            }
            served_cpu_ratios = {
                server: frame.loc[server, "total_cpu_load"] / capacities[server]["C_cpu"]
                for server in servers
            }
            excess_ratios = {
                server: frame.loc[server, "cpu_excess"] / capacities[server]["C_cpu"]
                for server in servers
            }
            cpu_server = max(cpu_ratios, key=cpu_ratios.get)
            mem_server = max(mem_ratios, key=mem_ratios.get)
            pre_cpu_host = max(pre_cpu_ratios, key=pre_cpu_ratios.get)
            pre_mem_host = max(pre_mem_ratios, key=pre_mem_ratios.get)
            excess_host = max(excess_ratios, key=excess_ratios.get)
            post_cpu.append(cpu_ratios[cpu_server])
            post_mem.append(mem_ratios[mem_server])
            served_cpu_at_post_peak.append(served_cpu_ratios[cpu_server])
            excess_at_post_peak.append(excess_ratios[cpu_server])
            pre_cpu.append(pre_cpu_ratios[pre_cpu_host])
            post_cpu_at_pre_peak.append(cpu_ratios[pre_cpu_host])
            preempted_spot_cpu_at_pre_peak.append(
                frame.loc[pre_cpu_host, "preempted_spot_cpu"]
                / capacities[pre_cpu_host]["C_cpu"]
            )
            pre_mem.append(pre_mem_ratios[pre_mem_host])
            post_mem_at_pre_peak.append(mem_ratios[pre_mem_host])
            preempted_spot_mem_at_pre_peak.append(
                frame.loc[pre_mem_host, "preempted_spot_mem"]
                / capacities[pre_mem_host]["C_mem"]
            )
            excess.append(excess_ratios[excess_host])
            post_cpu_server.append(cpu_server)
            post_mem_server.append(mem_server)
            pre_cpu_server.append(pre_cpu_host)
            pre_mem_server.append(pre_mem_host)
            excess_server.append(excess_host)
            batch_share.append(
                float(frame["workload"].sum()) / total_batch_work
                if total_batch_work
                else 0.0
            )
            events.append(int(event_counts.get((scenario_id, t), 0)))
            event_vm_ids.append(list(event_vms.get((scenario_id, t), [])))
        metric = tables["scenario"].loc[
            tables["scenario"]["scenario_id"].eq(scenario_id)
        ].iloc[0]
        atlas.append(
            {
                "id": scenario_id,
                "probability": _round(metric["probability"], 4),
                "profit": _round(metric["profit_usd"], 4),
                "energy": _round(metric["energy_kwh"], 4),
                "excessTotal": _round(metric["od_cpu_excess"], 4),
                "excessRate": _round(metric["od_cpu_excess_rate"], 6),
                "spotService": _round(metric["spot_service_rate"], 6),
                "preemptionEvents": int(event_totals.get(scenario_id, 0)),
                "preemptedSlots": int(state_totals.get(scenario_id, 0)),
                "postCpu": _series(post_cpu),
                "postMem": _series(post_mem),
                "servedCpuAtPostPeak": _series(served_cpu_at_post_peak),
                "excessAtPostPeak": _series(excess_at_post_peak),
                "preCpu": _series(pre_cpu),
                "postCpuAtPrePeak": _series(post_cpu_at_pre_peak),
                "preemptedSpotCpuAtPrePeak": _series(
                    preempted_spot_cpu_at_pre_peak
                ),
                "preMem": _series(pre_mem),
                "postMemAtPrePeak": _series(post_mem_at_pre_peak),
                "preemptedSpotMemAtPrePeak": _series(
                    preempted_spot_mem_at_pre_peak
                ),
                "excess": _series(excess),
                "postCpuServer": post_cpu_server,
                "postMemServer": post_mem_server,
                "preCpuServer": pre_cpu_server,
                "preMemServer": pre_mem_server,
                "excessServer": excess_server,
                "batchShare": _series(batch_share),
                "events": events,
                "eventVmIds": event_vm_ids,
            }
        )

    od_params = tables["od_params"].merge(
        tables["od_initial"], on="vm_id", how="left", validate="one_to_one"
    )
    od_ids = sorted(od_params["vm_id"].tolist())
    spot_ids = sorted(tables["spot_params"]["vm_id"].tolist())
    od_short = _short_ids(od_ids, "OD")
    spot_short = _short_ids(spot_ids, "SP")

    configured: dict[str, dict[str, Any]] = {
        server: {
            "cpu": {
                "od": [0.0] * horizon,
                "spot": [0.0] * horizon,
                "total": [0.0] * horizon,
            },
            "mem": {
                "od": [0.0] * horizon,
                "spot": [0.0] * horizon,
                "total": [0.0] * horizon,
            },
            "on": [0] * horizon,
        }
        for server in servers
    }
    for row in tables["server"].itertuples(index=False):
        configured[str(row.server_id)]["on"][int(row.t)] = int(row.on)
    for row in od_params.itertuples(index=False):
        server = str(row.server_id)
        for t in range(int(row.arrival_t), int(row.departure_t) + 1):
            configured[server]["cpu"]["od"][t] += (
                float(row.q_cpu) / capacities[server]["C_cpu"]
            )
            configured[server]["mem"]["od"][t] += (
                float(row.q_mem) / capacities[server]["C_mem"]
            )
    accepted_spot_hosts = (
        spot.groupby("vm_id", as_index=False)
        .agg(accepted=("accepted", "max"), initial_server=("initial_server", "first"))
    )
    spot_params = tables["spot_params"].merge(
        accepted_spot_hosts, on="vm_id", how="left", validate="one_to_one"
    )
    for row in spot_params.itertuples(index=False):
        if int(row.accepted) != 1:
            continue
        server = str(row.initial_server)
        for t in range(int(row.arrival_t), int(row.departure_t) + 1):
            configured[server]["cpu"]["spot"][t] += (
                float(row.q_cpu) / capacities[server]["C_cpu"]
            )
            configured[server]["mem"]["spot"][t] += (
                float(row.q_mem) / capacities[server]["C_mem"]
            )
    for server in servers:
        for resource in (CPU, MEM):
            configured[server][resource]["total"] = [
                od + spot
                for od, spot in zip(
                    configured[server][resource]["od"],
                    configured[server][resource]["spot"],
                    strict=True,
                )
            ]
            for workload_class in ("od", "spot", "total"):
                configured[server][resource][workload_class] = _series(
                    configured[server][resource][workload_class]
                )

    preparation = (
        batch.groupby(["family_id", "server_id", "t"], as_index=False)
        .agg(
            prepared=("prepared", "max"),
            logical_start=("logical_start", "max"),
        )
    )
    prepared_pairs = sorted(
        {
            (str(row.family_id), str(row.server_id))
            for row in preparation.loc[preparation["prepared"].gt(0)].itertuples(index=False)
        }
    )

    detail: dict[str, Any] = {}
    for scenario_id in scenarios:
        server_details: dict[str, Any] = {}
        for server in servers:
            frame = ledger.xs((scenario_id, server), level=("scenario_id", "server_id"))
            cpu_capacity = capacities[server]["C_cpu"]
            memory_capacity = capacities[server]["C_mem"]
            server_details[server] = {
                "on": [int(round(value)) for value in frame["server_on"].tolist()],
                "cpu": {
                    "od": _series((frame["od_cpu_served"] / cpu_capacity).tolist()),
                    "spot": _series((frame["spot_cpu"] / cpu_capacity).tolist()),
                    "batch": _series((frame["cpu_component"] / cpu_capacity).tolist()),
                    "excess": _series((frame["cpu_excess"] / cpu_capacity).tolist()),
                },
                "mem": {
                    "od": _series((frame["od_mem_served"] / memory_capacity).tolist()),
                    "spot": _series((frame["spot_mem"] / memory_capacity).tolist()),
                    "batch": _series(
                        (frame["mem_component"] / memory_capacity).tolist()
                    ),
                },
            }

        od_rows: list[dict[str, Any]] = []
        scenario_od = tables["od"].loc[tables["od"]["scenario_id"].eq(scenario_id)]
        for vm_id in od_ids:
            states = [-1] * horizon
            cpu = [0.0] * horizon
            mem = [0.0] * horizon
            rows = scenario_od.loc[scenario_od["vm_id"].eq(vm_id)].sort_values("t")
            for row in rows.itertuples(index=False):
                t = int(row.t)
                states[t] = server_index[str(row.server_id)]
                cpu[t] = float(row.actual_cpu_demand)
                mem[t] = float(row.actual_mem_demand)
            parameter = od_params.loc[od_params["vm_id"].eq(vm_id)].iloc[0]
            od_rows.append(
                {
                    "id": od_short[vm_id],
                    "fullId": vm_id,
                    "server": str(parameter["server_id"]),
                    "qCpu": _round(parameter["q_cpu"], 4),
                    "qMem": _round(parameter["q_mem"], 4),
                    "state": states,
                    "cpu": _series(cpu),
                    "mem": _series(mem),
                }
            )

        spot_rows: list[dict[str, Any]] = []
        scenario_spot = spot.loc[spot["scenario_id"].eq(scenario_id)]
        rejected_ids: list[str] = []
        for vm_id in spot_ids:
            rows = scenario_spot.loc[scenario_spot["vm_id"].eq(vm_id)].sort_values("t")
            accepted = int(rows["accepted"].max()) if not rows.empty else 0
            parameter = spot_params.loc[spot_params["vm_id"].eq(vm_id)].iloc[0]
            arrival_t = int(parameter["arrival_t"])
            departure_t = int(parameter["departure_t"])
            states = [-1] * horizon
            events = [0] * horizon
            cpu = [0.0] * horizon
            mem = [0.0] * horizon
            if not accepted:
                rejected_ids.append(spot_short[vm_id])
                for t in range(arrival_t, departure_t + 1):
                    states[t] = -3
            previous = 0
            for row in rows.itertuples(index=False):
                t = int(row.t)
                cpu[t] = float(row.cpu_usage)
                mem[t] = float(row.mem_usage)
                preempted = int(row.preempted)
                if not accepted:
                    continue
                if preempted:
                    states[t] = -2
                    events[t] = int(previous == 0)
                elif pd.notna(row.active_server):
                    states[t] = server_index[str(row.active_server)]
                previous = preempted
            initial_server = (
                str(parameter["initial_server"])
                if accepted and pd.notna(parameter["initial_server"])
                else None
            )
            spot_rows.append(
                {
                    "id": spot_short[vm_id],
                    "fullId": vm_id,
                    "accepted": bool(accepted),
                    "server": initial_server,
                    "arrivalT": arrival_t,
                    "departureT": departure_t,
                    "activeSlots": int(parameter["active_slots"]),
                    "lifespanHours": _round(
                        int(parameter["active_slots"])
                        * int(config["experiment"]["slot_minutes"])
                        / 60.0,
                        2,
                    ),
                    "qCpu": _round(parameter["q_cpu"], 4),
                    "qMem": _round(parameter["q_mem"], 4),
                    "state": states,
                    "events": events,
                    "cpu": _series(cpu),
                    "mem": _series(mem),
                }
            )

        batch_rows: list[dict[str, Any]] = []
        scenario_batch = batch.loc[batch["scenario_id"].eq(scenario_id)]
        for family_id, server in prepared_pairs:
            prep_values = [0] * horizon
            start_values = [0] * horizon
            work = [0.0] * horizon
            intensity = [0.0] * horizon
            prep_rows = preparation.loc[
                preparation["family_id"].eq(family_id)
                & preparation["server_id"].eq(server)
            ]
            for row in prep_rows.itertuples(index=False):
                prep_values[int(row.t)] = int(row.prepared)
                start_values[int(row.t)] = int(row.logical_start)
            work_rows = scenario_batch.loc[
                scenario_batch["family_id"].eq(family_id)
                & scenario_batch["server_id"].eq(server)
            ]
            parameter = batch_parameters[family_id]
            theta = min(
                parameter["q_cpu"] / parameter["rho_cpu"],
                parameter["q_mem"] / parameter["rho_mem"],
            )
            for row in work_rows.itertuples(index=False):
                t = int(row.t)
                work[t] = float(row.workload)
                intensity[t] = min(1.0, work[t] / theta)
            family_metadata = batch_family_metadata[family_id]
            batch_rows.append(
                {
                    "id": f"{family_id.replace('batch', 'B')}@{server.replace('s00', 's')}",
                    "family": family_id,
                    "server": server,
                    "jobs": int(family_metadata["jobs"]),
                    "totalWork": _round(parameter["workload"], 4),
                    "slotMaxWork": _round(theta, 4),
                    "qCpu": _round(parameter["q_cpu"], 4),
                    "qMem": _round(parameter["q_mem"], 4),
                    "prepared": prep_values,
                    "start": start_values,
                    "work": _series(work),
                    "intensity": _series(intensity),
                }
            )

        detail[str(scenario_id)] = {
            "servers": server_details,
            "od": od_rows,
            "spot": spot_rows,
            "rejectedSpotIds": rejected_ids,
            "batch": batch_rows,
        }

    # Default to the scenario with the longest preempted-state duration so the
    # first render demonstrates event onset versus cumulative state. The atlas
    # still exposes the largest integrated/instantaneous CPU-excess scenarios.
    default_scenario = int(
        max(
            atlas,
            key=lambda row: (
                row["preemptedSlots"],
                row["preemptionEvents"],
                row["excessTotal"],
                -row["id"],
            ),
        )["id"]
    )
    solver = summary["solver"]
    payload = {
        "meta": {
            "source": str(run_dir),
            "dataset": Path(config["data"]["google_dir"]).name,
            "workloadVariant": "Full On-demand + Spot + Batch",
            "modelVariant": "first-stage baseline",
            "status": solver["status"],
            "gap": _round(solver["mip_gap"], 6),
            "runtime": _round(solver["runtime_seconds"], 2),
            "bestBound": _round(solver["best_bound"], 6),
            "totalProfit": _round(solver["total_profit_including_constant_revenue"], 6),
            "alpha": _round(config["risk"]["alpha"], 4),
            "epsilon": _round(config["risk"]["epsilon"], 4),
            "expectedExcess": _round(
                summary["service"]["expected_od_cpu_excess"], 6
            ),
            "expectedExcessRate": _round(
                summary["service"]["expected_od_cpu_excess_rate"], 8
            ),
            "empiricalCvarMax": _round(
                summary["service"]["maximum_empirical_server_time_cvar"], 8
            ),
            "modelCvarMax": _round(
                summary["service"]["maximum_model_server_time_cvar_expression"], 8
            ),
            "acceptedSpotCount": int(summary["service"]["spot_acceptance_count"]),
            "spotCandidateCount": int(summary["service"]["spot_candidate_count"]),
            "classCounts": {
                "onDemand": int(config["experiment"]["class_counts"]["on_demand"]),
                "spot": int(config["experiment"]["class_counts"]["spot"]),
                "batchJobs": int(config["experiment"]["class_counts"]["batch_jobs"]),
                "batchFamilies": int(len(batch_parameters)),
            },
            "selectionPolicy": (
                "CPU-only eligible On-demand VM을 전체 active 구간의 coverage-weighted 평균 CPU로 "
                "정렬해 상위 20개 선택; configured size, usage, lifecycle은 변환하지 않음"
            ),
            "numScenarios": int(config["experiment"]["num_scenarios"]),
            "horizonHours": int(config["experiment"]["horizon_hours"]),
            "numServers": int(config["experiment"]["num_servers"]),
            "cpuSigma": _round(config["workload_scenarios"]["cpu_lognormal_sigma"], 4),
            "memorySigma": _round(
                config["workload_scenarios"]["memory_lognormal_sigma"], 4
            ),
            "syntheticRngMode": str(
                config["workload_scenarios"]["synthetic_rng_mode"]
            ),
            "solverThreads": int(config["solver"]["threads"]),
            "timeLimitSeconds": _round(config["solver"]["time_limit_seconds"], 2),
            "mipGapTarget": _round(config["solver"]["mip_gap"], 6),
            "noRelSeconds": _round(
                config["solver"]["no_rel_heur_time_seconds"], 2
            ),
            "seed": int(config["solver"]["seed"]),
            "migrationFixedZero": bool(config["migration"]["fixed_zero"]),
            "spotPreemptionAllowed": not bool(
                config.get("spot_preemption", {}).get("fixed_zero", False)
            ),
            "odCpuExcessAllowed": not bool(
                config.get("excess", {}).get("fixed_zero", False)
            ),
            "serverCpuCapacity": _round(
                tables["server_params"]["C_cpu"].iloc[0], 4
            ),
            "serverMemoryCapacity": _round(
                tables["server_params"]["C_mem"].iloc[0], 4
            ),
            "minimumOnSlots": int(config["server"]["min_on_slots"]),
            "minimumOffSlots": int(config["server"]["min_off_slots"]),
            "idleEnergyPerSlot": _round(
                tables["server_params"]["E_idle_kwh_per_slot"].iloc[0], 4
            ),
            "cpuEnergyPerUnitSlot": _round(
                tables["server_params"]["E_cpu_kwh_per_cpu_unit_slot"].iloc[0],
                4,
            ),
            "kappaOnDemand": _round(config["economics"]["kappa_on_demand"], 4),
            "spotDiscountRatio": _round(
                config["economics"]["spot_discount_ratio"], 4
            ),
            "batchDiscountRatio": _round(
                config["economics"]["batch_discount_ratio"], 4
            ),
            "electricitySite": str(config["electricity_price"]["site_id"]),
            "electricityMultiplier": _round(
                config["electricity_price"]["multiplier"], 4
            ),
            "electricityFloor": _round(
                config["electricity_price"]["floor_usd_per_kwh"], 4
            ),
            "numericFocus": int(config["solver"]["numeric_focus"]),
            "presolve": int(config["solver"]["presolve"]),
            "migration": "prohibited",
            "servers": servers,
            "periods": periods,
            "slotMinutes": int(config["experiment"]["slot_minutes"]),
            "defaultScenario": default_scenario,
            "cpuResidual": cpu_residual,
            "memResidual": mem_residual,
            "postCpuIdentityResidual": post_cpu_identity_residual,
            "configuredMax": _round(
                max(
                    max(configured[server][resource]["total"])
                    for server in servers
                    for resource in (CPU, MEM)
                ),
                4,
            ),
            "postCpuMax": _round(
                max(max(row["postCpu"]) for row in atlas), 4
            ),
            "postCpuMin": _round(
                min(min(row["postCpu"]) for row in atlas), 4
            ),
            "postMemMax": _round(
                max(max(row["postMem"]) for row in atlas), 4
            ),
            "postMemMin": _round(
                min(min(row["postMem"]) for row in atlas), 4
            ),
            "prePreemptionCpuMax": _round(
                max(
                    float(
                        row["pre_preemption_cpu_demand"]
                        / capacities[row.name[1]]["C_cpu"]
                    )
                    for _, row in ledger.iterrows()
                ),
                4,
            ),
            "prePreemptionMemMax": _round(
                max(
                    float(
                        row["pre_preemption_mem_load"]
                        / capacities[row.name[1]]["C_mem"]
                    )
                    for _, row in ledger.iterrows()
                ),
                4,
            ),
            "excessMax": _round(max(max(row["excess"]) for row in atlas), 4),
            "batchShareMax": _round(
                max(max(row["batchShare"]) for row in atlas), 4
            ),
            "preemptionEvents": int(len(spot_events)),
            "preemptedSlots": int(spot["preempted"].sum()),
        },
        "servers": servers,
        "scenarios": atlas,
        "configured": configured,
        "detail": detail,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = _build_payload(args.run_dir.resolve())
    template = args.template.read_text(encoding="utf-8")
    marker = "__CPU_EXCESS_DATA__"
    if template.count(marker) != 1:
        raise ValueError(f"Template must contain exactly one {marker} marker")
    rendered = template.replace(
        marker,
        json.dumps(payload, separators=(",", ":"), ensure_ascii=True, allow_nan=False),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "bytes": args.output.stat().st_size,
                "default_scenario": payload["meta"]["defaultScenario"],
                "cpu_residual": payload["meta"]["cpuResidual"],
                "mem_residual": payload["meta"]["memResidual"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
