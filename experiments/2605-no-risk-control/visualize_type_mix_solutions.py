"""
Visualize no-risk-control type-mix benchmark solutions.

This intentionally excludes SLA violation, CVaR, and spot suspension views.
It focuses on power state, realized load, placement/migration, batch processing,
and energy decomposition.
"""

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "experiments" / "2605-no-risk-control" / "type_mix_results_interactive_avg_wk"
EPS = 1e-7

COLORS = {
    "od": "#4C78A8",
    "spot": "#F58518",
    "batch": "#54A24B",
    "powered": "#D6DDE8",
    "off": "#F3F4F6",
    "migration": "#111827",
}
SERVER_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2", "#9D755D", "#BAB0AC"]

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.grid"] = False


def parse_case_name(case_name):
    match = re.match(
        r"^mix(?P<mask>\d+)_\d+vm_od(?P<od>\d+)_sp(?P<sp>\d+)_bj(?P<bj>\d+)(?:_k(?P<kappa>.+))?$",
        case_name,
    )
    if not match:
        return {"mask": case_name, "od": "", "sp": "", "bj": "", "kappa": ""}
    data = match.groupdict()
    if data.get("kappa"):
        data["kappa"] = data["kappa"].replace("p", ".")
    else:
        data["kappa"] = ""
    return data


def compact_case_label(case_name):
    data = parse_case_name(case_name)
    mix = f"mix{data['mask']}"
    counts = f"OD{data['od']}/SP{data['sp']}/BJ{data['bj']}"
    if data["kappa"]:
        return f"{mix} ({counts}, k={data['kappa']})"
    return f"{mix} ({counts})"


def case_tick_label(case_name):
    data = parse_case_name(case_name)
    label = f"{data['mask']}\nOD{data['od']} SP{data['sp']} BJ{data['bj']}"
    if data["kappa"]:
        label += f"\nk={data['kappa']}"
    return label


def add_bottom_legend(ax, handles, ncol):
    ax.legend(
        handles=handles,
        ncol=ncol,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.8,
        columnspacing=1.4,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize no-risk-control type-mix solutions without SLA/suspension plots.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--scenario", type=str, default=None)
    return parser.parse_args()


def parse_var_name(var_name):
    match = re.match(r"^([^\[]+)\[(.*)\]$", str(var_name))
    if not match:
        return str(var_name), ()
    return match.group(1), tuple(item.strip() for item in match.group(2).split(","))


def load_solution(results_dir):
    path = results_dir / "solution_nonzero_variables.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing solution file: {path}")
    lookup = {}
    frame = pd.read_csv(path)
    for row in frame.itertuples(index=False):
        name, indices = parse_var_name(row.variable)
        lookup[(name, indices)] = float(row.value)
    return lookup


def get_value(lookup, name, *indices):
    return float(lookup.get((name, tuple(str(index) for index in indices)), 0.0))


def load_instance(path):
    with open(path, "r", encoding="utf-8") as file:
        raw = json.load(file)
    prob = {row["id"]: float(row["probability"]) for row in raw["scenarios"]}
    return {
        "raw": raw,
        "servers": [int(value) for value in raw["sets"]["servers"]],
        "times": [int(value) for value in raw["sets"]["times"]],
        "scenarios": raw["sets"]["scenarios"],
        "on_demand": raw["sets"]["on_demand"],
        "spot": raw["sets"]["spot"],
        "batch": raw["sets"]["batch"],
        "prob": prob,
        "capacity": float(raw["parameters"]["capacity"]),
        "energy_idle": float(raw["parameters"]["energy_idle"]),
        "energy_cpu": float(raw["parameters"]["energy_cpu"]),
        "energy_migration": float(raw["parameters"]["energy_migration"]),
        "od_active": {key: [int(value) for value in values] for key, values in raw["active_periods"]["on_demand"].items()},
        "spot_active": {key: [int(value) for value in values] for key, values in raw["active_periods"]["spot"].items()},
        "d_od": {(row["id"], int(row["time"]), row["scenario"]): float(row["demand"]) for row in raw["demands"]["on_demand"]},
        "d_sp": {(row["id"], int(row["time"]), row["scenario"]): float(row["demand"]) for row in raw["demands"]["spot"]},
        "batch_info": {row["id"]: row for row in raw["batch_jobs"]},
    }


def build_tables(data, lookup):
    S = data["servers"]
    T = data["times"]
    Xi = data["scenarios"]
    I = data["on_demand"]
    J = data["spot"]
    K = data["batch"]
    C = data["capacity"]

    server_time = pd.DataFrame(
        [{"server": s, "time": t, "u": get_value(lookup, "u", s, t)} for s in S for t in T]
    )

    od_initial = []
    for i in I:
        for s in S:
            value = get_value(lookup, "x", i, s)
            if value > EPS:
                od_initial.append({"workload_id": i, "server": s, "value": value})
    od_initial = pd.DataFrame(od_initial, columns=["workload_id", "server", "value"])

    spot_initial = []
    for j in J:
        for s in S:
            value = get_value(lookup, "y", j, s)
            if value > EPS:
                spot_initial.append({"workload_id": j, "server": s, "value": value})
    spot_initial = pd.DataFrame(spot_initial, columns=["workload_id", "server", "value"])

    batch_reservation = []
    for k in K:
        for s in S:
            for t in T:
                value = get_value(lookup, "b", k, s, t)
                if value > EPS:
                    reserved_cpu = float(data["batch_info"][k]["reserved_cpu"])
                    batch_reservation.append(
                        {"batch_id": k, "server": s, "time": t, "reserved": value, "reserved_cpu": reserved_cpu}
                    )
    batch_reservation = pd.DataFrame(
        batch_reservation, columns=["batch_id", "server", "time", "reserved", "reserved_cpu"]
    )

    od_realized = []
    for i in I:
        for s in S:
            for t in data["od_active"][i]:
                for xi in Xi:
                    value = get_value(lookup, "xR", i, s, t, xi)
                    if value > EPS:
                        od_realized.append({"workload_id": i, "server": s, "time": t, "scenario": xi, "active": value})
    od_realized = pd.DataFrame(od_realized, columns=["workload_id", "server", "time", "scenario", "active"])

    od_server = {}
    if not od_realized.empty:
        for row in od_realized.itertuples(index=False):
            od_server[(row.workload_id, row.scenario, int(row.time))] = int(row.server)

    migrations = []
    for i in I:
        for t in data["od_active"][i][1:]:
            previous_time = data["od_active"][i][data["od_active"][i].index(t) - 1]
            for xi in Xi:
                value = get_value(lookup, "m", i, t, xi)
                if value <= EPS:
                    continue
                migrations.append(
                    {
                        "workload_id": i,
                        "time": int(t),
                        "previous_time": int(previous_time),
                        "scenario": xi,
                        "from_server": od_server.get((i, xi, previous_time)),
                        "to_server": od_server.get((i, xi, t)),
                        "value": value,
                        "probability": data["prob"][xi],
                        "weighted_value": data["prob"][xi] * value,
                    }
                )
    migrations = pd.DataFrame(
        migrations,
        columns=["workload_id", "time", "previous_time", "scenario", "from_server", "to_server", "value", "probability", "weighted_value"],
    )

    batch_processing = []
    for k in K:
        for s in S:
            for t in T:
                for xi in Xi:
                    value = get_value(lookup, "z", k, s, t, xi)
                    if value > EPS:
                        batch_processing.append({"batch_id": k, "server": s, "time": t, "scenario": xi, "processed": value})
    batch_processing = pd.DataFrame(batch_processing, columns=["batch_id", "server", "time", "scenario", "processed"])

    state_rows = []
    type_rows = []
    for s in S:
        for t in T:
            u_value = get_value(lookup, "u", s, t)
            for xi in Xi:
                od_load = get_value(lookup, "od_load", s, t, xi)
                spot_load = sum(
                    data["d_sp"].get((j, t, xi), 0.0) * get_value(lookup, "y", j, s)
                    for j in J
                    if t in data["spot_active"][j]
                )
                batch_load = sum(get_value(lookup, "z", k, s, t, xi) for k in K)
                total_load = get_value(lookup, "total_load", s, t, xi)
                if total_load <= EPS:
                    total_load = od_load + spot_load + batch_load
                state_rows.append(
                    {
                        "server": s,
                        "time": t,
                        "scenario": xi,
                        "probability": data["prob"][xi],
                        "u": u_value,
                        "od_load": od_load,
                        "spot_load": spot_load,
                        "batch_load": batch_load,
                        "total_load": total_load,
                        "utilization": total_load / C if C else 0.0,
                    }
                )
                type_rows.append({"server": s, "time": t, "scenario": xi, "kind": "od", "load": od_load})
                type_rows.append({"server": s, "time": t, "scenario": xi, "kind": "spot", "load": spot_load})
                type_rows.append({"server": s, "time": t, "scenario": xi, "kind": "batch", "load": batch_load})
    server_state = pd.DataFrame(state_rows)
    type_load = pd.DataFrame(type_rows)

    expected_state = (
        server_state.assign(
            weighted_total_load=server_state["probability"] * server_state["total_load"],
            weighted_od_load=server_state["probability"] * server_state["od_load"],
            weighted_spot_load=server_state["probability"] * server_state["spot_load"],
            weighted_batch_load=server_state["probability"] * server_state["batch_load"],
        )
        .groupby(["server", "time"], as_index=False)
        .agg(
            u=("u", "max"),
            expected_total_load=("weighted_total_load", "sum"),
            expected_od_load=("weighted_od_load", "sum"),
            expected_spot_load=("weighted_spot_load", "sum"),
            expected_batch_load=("weighted_batch_load", "sum"),
            max_total_load=("total_load", "max"),
            max_utilization=("utilization", "max"),
        )
    )
    expected_state["expected_utilization"] = expected_state["expected_total_load"] / C

    scenario_metrics = (
        server_state.groupby("scenario", as_index=False)
        .agg(
            peak_load=("total_load", "max"),
            peak_utilization=("utilization", "max"),
            active_server_slots=("u", "sum"),
            total_load=("total_load", "sum"),
            od_load=("od_load", "sum"),
            spot_load=("spot_load", "sum"),
            batch_load=("batch_load", "sum"),
        )
        .sort_values(["peak_utilization", "total_load"], ascending=False)
    )

    energy_rows = []
    for s in S:
        idle_energy = data["energy_idle"] * server_time.loc[server_time["server"] == s, "u"].sum()
        state = server_state.loc[server_state["server"] == s]
        cpu_energy = data["energy_cpu"] / C * (state["probability"] * state["total_load"]).sum()
        energy_rows.append({"server": s, "idle_energy": idle_energy, "expected_cpu_energy": cpu_energy, "migration_energy": 0.0})
    weighted_migrations = float(migrations["weighted_value"].sum()) if not migrations.empty else 0.0
    energy_rows.append(
        {
            "server": "migration_total",
            "idle_energy": 0.0,
            "expected_cpu_energy": 0.0,
            "migration_energy": data["energy_migration"] * weighted_migrations,
        }
    )
    energy_summary = pd.DataFrame(energy_rows)
    energy_summary["total_energy"] = (
        energy_summary["idle_energy"] + energy_summary["expected_cpu_energy"] + energy_summary["migration_energy"]
    )

    return {
        "server_time": server_time,
        "server_state": server_state,
        "expected_state": expected_state,
        "scenario_metrics": scenario_metrics,
        "type_load": type_load,
        "od_initial": od_initial,
        "spot_initial": spot_initial,
        "od_realized": od_realized,
        "migrations": migrations,
        "batch_reservation": batch_reservation,
        "batch_processing": batch_processing,
        "energy_summary": energy_summary,
    }


def write_tables(tables, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    names = {
        "server_time": "server_time_power.csv",
        "server_state": "server_time_state.csv",
        "expected_state": "expected_server_time_state.csv",
        "scenario_metrics": "scenario_metrics.csv",
        "type_load": "server_time_type_load.csv",
        "od_initial": "on_demand_initial_placement.csv",
        "spot_initial": "spot_initial_placement.csv",
        "od_realized": "on_demand_realized_placement.csv",
        "migrations": "migration_events.csv",
        "batch_reservation": "batch_reservation.csv",
        "batch_processing": "batch_processing.csv",
        "energy_summary": "energy_summary.csv",
    }
    for key, filename in names.items():
        tables[key].to_csv(output_dir / filename, index=False)


def save_fig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.2)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def used_servers(tables):
    expected = tables["expected_state"]
    servers = expected.loc[(expected["u"] > EPS) | (expected["expected_total_load"] > EPS), "server"].unique()
    return sorted(int(server) for server in servers)


def choose_scenario(tables, requested):
    if requested:
        return requested
    metrics = tables["scenario_metrics"].copy()
    if metrics.empty:
        return None
    return metrics.iloc[0]["scenario"]


def heatmap(ax, frame, value, title, cmap, vmin=None, vmax=None):
    pivot = frame.pivot(index="server", columns="time", values=value).sort_index()
    if pivot.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title(title)
        return None
    if vmin is None:
        vmin = float(np.nanmin(pivot.values))
    if vmax is None:
        vmax = float(np.nanmax(pivot.values))
    if math.isclose(vmin, vmax):
        vmax = vmin + 1.0
    image = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel("server")
    tick_positions = [index for index, column in enumerate(pivot.columns) if int(column) % 2 == 0]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(pivot.columns[index]) for index in tick_positions], fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(value) for value in pivot.index])
    return image


def plot_expected_heatmaps(data, tables, plot_dir, case_name):
    short_label = compact_case_label(case_name)
    expected = tables["expected_state"]
    servers = used_servers(tables)
    if servers:
        expected = expected.loc[expected["server"].isin(servers)].copy()
    fig, axes = plt.subplots(3, 1, figsize=(14, 9.5), sharex=False)
    images = [
        heatmap(axes[0], expected, "u", f"{short_label}: powered server slots", "Greys", 0, 1),
        heatmap(axes[1], expected, "expected_utilization", "expected utilization", "YlOrRd", 0, max(1.0, expected["expected_utilization"].max())),
        heatmap(axes[2], expected, "max_utilization", "max utilization across scenarios", "PuBuGn", 0, max(1.0, expected["max_utilization"].max())),
    ]
    for ax, image in zip(axes, images):
        if image is not None:
            fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    save_fig(fig, plot_dir / "power_and_load_heatmaps.png")


def plot_workload_gantt(data, tables, scenario, plot_dir, case_name):
    short_label = compact_case_label(case_name)
    type_load = tables["type_load"].loc[tables["type_load"]["scenario"] == scenario].copy()
    server_time = tables["server_time"]
    servers = used_servers(tables)
    if not servers:
        servers = data["servers"]
    times = data["times"]

    fig_height = max(4.9, 0.58 * len(servers) + 2.4)
    fig, ax = plt.subplots(figsize=(15, fig_height))
    row_height = 0.82
    server_to_y = {server: idx for idx, server in enumerate(servers)}

    for server in servers:
        y = server_to_y[server]
        for time in times:
            u_value = float(server_time.loc[(server_time["server"] == server) & (server_time["time"] == time), "u"].max())
            ax.add_patch(
                Rectangle(
                    (time, y - row_height / 2),
                    1.0,
                    row_height,
                    facecolor=COLORS["powered"] if u_value > 0.5 else COLORS["off"],
                    edgecolor="#CBD5E1",
                    linewidth=0.35,
                    alpha=0.55,
                )
            )
            bottom = y - row_height / 2
            cell = type_load.loc[(type_load["server"] == server) & (type_load["time"] == time)]
            for kind in ["od", "spot", "batch"]:
                amount = float(cell.loc[cell["kind"] == kind, "load"].sum())
                if amount <= EPS:
                    continue
                height = min(row_height, amount / data["capacity"] * row_height)
                ax.add_patch(
                    Rectangle(
                        (time + 0.08, bottom),
                        0.84,
                        height,
                        facecolor=COLORS[kind],
                        edgecolor="none",
                        alpha=0.88,
                    )
                )
                bottom += height

    ax.set_xlim(min(times), max(times) + 1)
    ax.set_ylim(-0.8, len(servers) - 0.2)
    ax.set_yticks(range(len(servers)))
    ax.set_yticklabels([f"S{server}" for server in servers])
    ax.set_xticks(times)
    ax.set_xlabel("time")
    ax.set_title(f"{short_label}: server workload ({scenario})", pad=14)
    add_bottom_legend(
        ax,
        [
            Patch(color=COLORS["od"], label="on-demand"),
            Patch(color=COLORS["spot"], label="spot"),
            Patch(color=COLORS["batch"], label="batch"),
            Patch(color=COLORS["powered"], label="powered slot"),
        ],
        ncol=4,
    )
    save_fig(fig, plot_dir / f"server_workload_gantt_{scenario}.png")


def plot_workload_stack(data, tables, scenario, plot_dir, case_name):
    short_label = compact_case_label(case_name)
    state = tables["server_state"].loc[tables["server_state"]["scenario"] == scenario]
    by_time = state.groupby("time", as_index=False).agg(
        od_load=("od_load", "sum"),
        spot_load=("spot_load", "sum"),
        batch_load=("batch_load", "sum"),
        total_load=("total_load", "sum"),
    )
    fig, ax = plt.subplots(figsize=(14, 5.3))
    times = by_time["time"].to_numpy()
    od = by_time["od_load"].to_numpy()
    spot = by_time["spot_load"].to_numpy()
    batch = by_time["batch_load"].to_numpy()
    ax.stackplot(times, od, spot, batch, colors=[COLORS["od"], COLORS["spot"], COLORS["batch"]], labels=["OD", "Spot", "Batch"], alpha=0.86)
    ax.plot(times, by_time["total_load"], color="#111827", linewidth=1.5, label="total")
    ax.set_title(f"{short_label}: aggregate workload stack ({scenario})", pad=14)
    ax.set_xlabel("time")
    ax.set_ylabel("CPU load")
    handles, labels = ax.get_legend_handles_labels()
    add_bottom_legend(ax, handles, ncol=4)
    ax.grid(axis="y", alpha=0.25)
    save_fig(fig, plot_dir / f"workload_stack_{scenario}.png")


def plot_batch_processing(data, tables, scenario, plot_dir, case_name):
    short_label = compact_case_label(case_name)
    if not data["batch"]:
        return None
    servers = used_servers(tables)
    times = data["times"]
    reservation = tables["batch_reservation"].copy()
    processing = tables["batch_processing"].loc[tables["batch_processing"]["scenario"] == scenario].copy()
    if reservation.empty and processing.empty:
        return None

    frames = []
    for label, frame, column in [
        ("reserved_capacity", reservation.assign(amount=reservation["reserved"] * reservation["reserved_cpu"]) if not reservation.empty else reservation, "amount"),
        ("processed", processing.rename(columns={"processed": "amount"}) if not processing.empty else processing, "amount"),
    ]:
        if frame.empty:
            values = pd.DataFrame({"server": servers, **{time: 0.0 for time in times}}).set_index("server")
        else:
            pivot = frame.groupby(["server", "time"], as_index=False)[column].sum().pivot(index="server", columns="time", values=column)
            pivot = pivot.reindex(index=servers, columns=times, fill_value=0.0).fillna(0.0)
            values = pivot
        frames.append((label, values))

    fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True)
    vmax = max(float(values.values.max()) for _, values in frames)
    vmax = max(vmax, 1.0)
    for ax, (label, values) in zip(axes, frames):
        image = ax.imshow(values.values, aspect="auto", cmap="Greens", vmin=0, vmax=vmax)
        ax.set_title(label.replace("_", " "))
        ax.set_ylabel("server")
        ax.set_yticks(range(len(values.index)))
        ax.set_yticklabels([str(value) for value in values.index])
        ax.set_xticks(range(len(values.columns)))
        tick_positions = [index for index, column in enumerate(values.columns) if int(column) % 2 == 0]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(values.columns[index]) for index in tick_positions], fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    axes[-1].set_xlabel("time")
    fig.suptitle(f"{short_label}: batch reservation and processing ({scenario})", y=0.995)
    save_fig(fig, plot_dir / f"batch_processing_{scenario}.png")
    return True


def plot_migration_timeline(data, tables, scenario, plot_dir, case_name):
    short_label = compact_case_label(case_name)
    od = tables["od_realized"].loc[tables["od_realized"]["scenario"] == scenario].copy()
    if od.empty:
        return None
    workloads = sorted(data["on_demand"])
    workload_to_y = {workload: idx for idx, workload in enumerate(workloads)}
    fig, ax = plt.subplots(figsize=(14, max(4.2, 0.42 * len(workloads) + 2.2)))
    for row in od.itertuples(index=False):
        y = workload_to_y[row.workload_id]
        color = SERVER_COLORS[int(row.server) % len(SERVER_COLORS)]
        ax.scatter(row.time, y, marker="s", s=34, color=color, edgecolor="white", linewidth=0.25)
    migrations = tables["migrations"].loc[tables["migrations"]["scenario"] == scenario]
    for row in migrations.itertuples(index=False):
        y = workload_to_y.get(row.workload_id)
        if y is None:
            continue
        ax.scatter(row.time, y, marker="*", s=95, color=COLORS["migration"], edgecolor="white", linewidth=0.35)
    ax.set_yticks(range(len(workloads)))
    ax.set_yticklabels(workloads)
    ax.set_xticks(data["times"])
    ax.set_xlabel("time")
    ax.set_title(f"{short_label}: on-demand placement and migrations ({scenario})", pad=14)
    server_handles = [
        Patch(color=SERVER_COLORS[server % len(SERVER_COLORS)], label=f"S{server}")
        for server in sorted(od["server"].dropna().astype(int).unique())
    ]
    if server_handles:
        add_bottom_legend(ax, server_handles, ncol=min(6, len(server_handles)))
    ax.grid(axis="x", alpha=0.25)
    save_fig(fig, plot_dir / f"migration_timeline_{scenario}.png")
    return True


def summarize_case(case_name, summary_row, tables):
    energy = tables["energy_summary"]
    server_energy = energy.loc[energy["server"] != "migration_total"]
    migration_energy = float(energy.loc[energy["server"] == "migration_total", "migration_energy"].sum())
    idle_energy = float(server_energy["idle_energy"].sum())
    cpu_energy = float(server_energy["expected_cpu_energy"].sum())
    expected = tables["expected_state"]
    powered_slots = float(expected["u"].sum())
    expected_work = float(expected["expected_total_load"].sum())
    capacity = expected_work / powered_slots if powered_slots > EPS else 0.0
    return {
        "case": case_name,
        "mask": summary_row.get("mask"),
        "kappa": summary_row.get("kappa"),
        "status": summary_row.get("status"),
        "objective": float(summary_row.get("objective") or 0.0),
        "gap": float(summary_row.get("gap") or 0.0),
        "idle_energy": idle_energy,
        "cpu_energy": cpu_energy,
        "migration_energy": migration_energy,
        "total_energy_from_solution": idle_energy + cpu_energy + migration_energy,
        "powered_slots": powered_slots,
        "expected_work": expected_work,
        "expected_load_per_powered_slot": capacity,
    }


def plot_comparison(case_summary, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(case_summary)
    frame.to_csv(output_dir / "case_energy_summary.csv", index=False)
    labels = [case_tick_label(case_name) for case_name in frame["case"]]
    x = np.arange(len(frame))

    fig, ax = plt.subplots(figsize=(15, 6.2))
    idle = frame["idle_energy"].to_numpy()
    cpu = frame["cpu_energy"].to_numpy()
    mig = frame["migration_energy"].to_numpy()
    ax.bar(x, idle, label="idle", color="#B8C4D9")
    ax.bar(x, cpu, bottom=idle, label="CPU", color="#4C78A8")
    ax.bar(x, mig, bottom=idle + cpu, label="migration", color="#111827")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("energy")
    ax.set_title("Energy decomposition by type mix", pad=14)
    add_bottom_legend(ax, ax.get_legend_handles_labels()[0], ncol=3)
    ax.grid(axis="y", alpha=0.25)
    save_fig(fig, output_dir / "comparison_energy_decomposition.png")

    fig, ax1 = plt.subplots(figsize=(15, 5.8))
    ax1.bar(x - 0.18, frame["powered_slots"], width=0.36, color="#72B7B2", label="powered slots")
    ax1.set_ylabel("powered server-time slots")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.5)
    ax2 = ax1.twinx()
    ax2.plot(x + 0.18, frame["expected_load_per_powered_slot"], color="#E45756", marker="o", label="load / powered slot")
    ax2.set_ylabel("expected load per powered slot")
    ax1.set_title("Powered slots and packing density", pad=14)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    add_bottom_legend(ax1, handles1 + handles2, ncol=2)
    ax1.grid(axis="y", alpha=0.25)
    save_fig(fig, output_dir / "comparison_powered_slots.png")


def write_index(results_root, case_outputs, comparison_dir):
    path = results_root / "visualization_index_no_sla.md"
    lines = [
        "# No-Risk-Control Solution Visualizations",
        "",
        "Excluded: SLA violation plots, spot suspension plots, and recourse-risk heatmaps.",
        "",
        f"Comparison plots: `{comparison_dir.relative_to(results_root)}`",
        "",
        "| case | scenario | plots | analysis |",
        "|---|---|---|---|",
    ]
    for item in case_outputs:
        lines.append(
            f"| {item['case']} | {item['scenario']} | `{item['plot_dir'].relative_to(results_root)}` | `{item['analysis_dir'].relative_to(results_root)}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    args = parse_args()
    results_root = args.results_root.resolve()
    summary = pd.read_csv(results_root / "summary.csv")
    case_summaries = []
    case_outputs = []

    for summary_row in summary.to_dict("records"):
        case_name = Path(summary_row["results_dir"]).name
        results_dir = Path(summary_row["results_dir"])
        if not results_dir.is_absolute():
            results_dir = (REPO_ROOT / results_dir).resolve()
        instance_path = Path(summary_row["instance_path"])
        if not instance_path.is_absolute():
            instance_path = (REPO_ROOT / instance_path).resolve()

        data = load_instance(instance_path)
        lookup = load_solution(results_dir)
        tables = build_tables(data, lookup)
        scenario = choose_scenario(tables, args.scenario)

        analysis_dir = results_dir / "analysis_no_sla"
        plot_dir = results_dir / "plots_no_sla"
        write_tables(tables, analysis_dir)
        plot_expected_heatmaps(data, tables, plot_dir, case_name)
        if scenario:
            plot_workload_gantt(data, tables, scenario, plot_dir, case_name)
            plot_workload_stack(data, tables, scenario, plot_dir, case_name)
            plot_batch_processing(data, tables, scenario, plot_dir, case_name)
            plot_migration_timeline(data, tables, scenario, plot_dir, case_name)
        case_summaries.append(summarize_case(case_name, summary_row, tables))
        case_outputs.append({"case": case_name, "scenario": scenario, "analysis_dir": analysis_dir, "plot_dir": plot_dir})

    comparison_dir = results_root / "visualizations_no_sla"
    plot_comparison(case_summaries, comparison_dir)
    index_path = write_index(results_root, case_outputs, comparison_dir)

    print(
        json.dumps(
            {
                "cases": len(case_outputs),
                "index": str(index_path),
                "comparison_dir": str(comparison_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
