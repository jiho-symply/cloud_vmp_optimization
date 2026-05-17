"""
Visualize a solved VM type modeling (1) instance.

The solver writes a compact solution file:

- solution_nonzero_variables.csv: nonzero Gurobi variables
- summary.json: solver status/objective metadata

This script reconstructs full analysis tables from those nonzero variables and
the prepared JSON instance, then writes both CSV summaries and PNG figures.
It is intentionally self-contained so the modeling code stays readable.

Outputs:

- analysis/server_time_state.csv: server-time-scenario load, barL, gamma, phi
- analysis/on_demand_realized_placement.csv: xR placement by scenario
- analysis/migration_events.csv: nonzero migration arcs
- analysis/spot_activity.csv and analysis/spot_metrics.csv
- analysis/batch_reservation.csv and analysis/batch_processing.csv
- analysis/energy_summary.csv and analysis/scenario_metrics.csv
- plots/*.png: load charts, nominal/actual workload Gantt charts,
  all-scenario recourse heatmaps, workload timelines, batch views
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, Patch, Rectangle


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INSTANCE = (
    REPO_ROOT
    / "data"
    / "processed"
    / "2605-vm-type-modeling-1"
    / "notion_vm_type_24vm_od8_sp8_bj8_sc10_cap8"
    / "vm_type_instance.json"
)
DEFAULT_RESULTS_DIR = REPO_ROOT / "experiments" / "2605-vm-type-modeling-1" / "results"

PALETTE_OD = ["#4E79A7", "#6BAED6", "#1F77B4", "#9ECAE1", "#08519C", "#BDD7E7"]
PALETTE_SP = ["#F2C94C", "#F2994A", "#F5C542", "#C99700", "#FFE680", "#E6B800"]
PALETTE_BJ = ["#59A14F", "#74C476", "#2CA25F", "#006D2C", "#A1D99B", "#41AB5D"]
SERVER_COLORS = ["#4E79A7", "#F28E2B", "#59A14F", "#B07AA1", "#E15759", "#76B7B2", "#9C755F", "#BAB0AC"]
FIG_BG = "#F6F3EE"
AX_BG = "#FFFCF7"
GRID_COLOR = "#D9D1C3"
TEXT_COLOR = "#2F2A24"
CAPACITY_COLOR = "#C94F4F"
OFF_COLOR = "#111827"
ALERT_COLOR = "#C94F4F"
MUTED_TEXT = "#6F675D"
EPS = 1e-7

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = FIG_BG
plt.rcParams["axes.facecolor"] = AX_BG
plt.rcParams["axes.edgecolor"] = "#C5BBAA"
plt.rcParams["axes.labelcolor"] = TEXT_COLOR
plt.rcParams["xtick.color"] = TEXT_COLOR
plt.rcParams["ytick.color"] = TEXT_COLOR
plt.rcParams["text.color"] = TEXT_COLOR
plt.rcParams["grid.color"] = GRID_COLOR


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize VM type modeling (1) solution.")
    parser.add_argument("--instance", type=Path, default=DEFAULT_INSTANCE)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--scenario", type=str, default=None, help="Scenario to emphasize in scenario-specific plots.")
    return parser.parse_args()


def parse_var_name(var_name):
    match = re.match(r"^([^\[]+)\[(.*)\]$", str(var_name))
    if not match:
        return str(var_name), []
    name = match.group(1)
    indices = [item.strip() for item in match.group(2).split(",")]
    return name, indices


def load_instance(path):
    with open(path, "r", encoding="utf-8") as file:
        instance = json.load(file)

    scenario_prob = {row["id"]: float(row["probability"]) for row in instance["scenarios"]}
    d_od = {(row["id"], int(row["time"]), row["scenario"]): float(row["demand"]) for row in instance["demands"]["on_demand"]}
    d_sp = {(row["id"], int(row["time"]), row["scenario"]): float(row["demand"]) for row in instance["demands"]["spot"]}
    batch_info = {row["id"]: row for row in instance["batch_jobs"]}

    return {
        "raw": instance,
        "servers": [int(value) for value in instance["sets"]["servers"]],
        "times": [int(value) for value in instance["sets"]["times"]],
        "on_demand": instance["sets"]["on_demand"],
        "spot": instance["sets"]["spot"],
        "batch": instance["sets"]["batch"],
        "scenarios": instance["sets"]["scenarios"],
        "prob": scenario_prob,
        "capacity": float(instance["parameters"]["capacity"]),
        "energy_idle": float(instance["parameters"]["energy_idle"]),
        "energy_cpu": float(instance["parameters"]["energy_cpu"]),
        "energy_migration": float(instance["parameters"]["energy_migration"]),
        "od_active": {key: [int(v) for v in values] for key, values in instance["active_periods"]["on_demand"].items()},
        "spot_active": {key: [int(v) for v in values] for key, values in instance["active_periods"]["spot"].items()},
        "d_od": d_od,
        "d_sp": d_sp,
        "batch_info": batch_info,
    }


def load_solution(results_dir):
    path = results_dir / "solution_nonzero_variables.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing solution file: {path}")

    frame = pd.read_csv(path)
    rows = []
    for row in frame.itertuples(index=False):
        name, indices = parse_var_name(row.variable)
        rows.append({"name": name, "indices": indices, "value": float(row.value)})
    return rows


def value_lookup(solution_rows):
    lookup = {}
    for row in solution_rows:
        lookup[(row["name"], tuple(row["indices"]))] = row["value"]
    return lookup


def get_value(lookup, name, *indices):
    return float(lookup.get((name, tuple(str(index) for index in indices)), 0.0))


def pick_color(label, labels, palette):
    labels = sorted(labels)
    index = labels.index(label) if label in labels else 0
    return palette[index % len(palette)]


def style_axis(ax, grid_axis="both"):
    if grid_axis:
        ax.grid(True, axis=grid_axis, linewidth=0.8, alpha=0.45)
    for spine in ax.spines.values():
        spine.set_color("#C5BBAA")
    return ax


def choose_scenario(data, server_state, requested):
    if requested:
        return requested
    ranked = (
        server_state.groupby("scenario", as_index=False)
        .agg(
            max_util=("utilization", "max"),
            gamma_count=("gamma", "sum"),
            phi_count=("phi", "sum"),
            total_load=("load", "sum"),
        )
        .sort_values(["phi_count", "gamma_count", "max_util", "total_load"], ascending=False)
    )
    return ranked.iloc[0]["scenario"]


def choose_migration_scenario(tables, fallback):
    migrations = tables["migrations"]
    if migrations.empty:
        return fallback
    ranked = migrations.groupby("scenario").size().sort_values(ascending=False)
    return ranked.index[0]


def build_tables(data, solution_rows):
    lookup = value_lookup(solution_rows)
    S = data["servers"]
    T = data["times"]
    Xi = data["scenarios"]
    I = data["on_demand"]
    J = data["spot"]
    K = data["batch"]
    C = data["capacity"]

    u_rows = []
    for s in S:
        for t in T:
            u_rows.append({"server": s, "time": t, "u": get_value(lookup, "u", s, t)})
    server_time = pd.DataFrame(u_rows)

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

    od_realized = []
    for i in I:
        for s in S:
            for t in data["od_active"][i]:
                for xi in Xi:
                    value = get_value(lookup, "xR", i, s, t, xi)
                    if value > EPS:
                        od_realized.append({"workload_id": i, "server": s, "time": t, "scenario": xi, "active": value})
    od_realized = pd.DataFrame(od_realized, columns=["workload_id", "server", "time", "scenario", "active"])

    migrations = []
    for row in solution_rows:
        if row["name"] == "m" and row["value"] > EPS:
            i, s, sp, t, xi = row["indices"]
            migrations.append(
                {
                    "workload_id": i,
                    "from_server": int(s),
                    "to_server": int(sp),
                    "time": int(t),
                    "scenario": xi,
                    "value": row["value"],
                }
            )
    migrations = pd.DataFrame(migrations, columns=["workload_id", "from_server", "to_server", "time", "scenario", "value"])

    spot_activity = []
    for j in J:
        for s in S:
            for t in data["spot_active"][j]:
                demand_by_scenario = {xi: data["d_sp"].get((j, t, xi), 0.0) for xi in Xi}
                for xi in Xi:
                    active = get_value(lookup, "yR", j, s, t, xi)
                    if active > EPS:
                        spot_activity.append(
                            {
                                "workload_id": j,
                                "server": s,
                                "time": t,
                                "scenario": xi,
                                "active": active,
                                "demand": demand_by_scenario[xi],
                            }
                        )
    spot_activity = pd.DataFrame(spot_activity, columns=["workload_id", "server", "time", "scenario", "active", "demand"])

    spot_metrics = []
    for j in J:
        active_len = max(1, len(data["spot_active"][j]))
        for xi in Xi:
            active_slots = 0.0
            for t in data["spot_active"][j]:
                active_slots += sum(get_value(lookup, "yR", j, s, t, xi) for s in S)
            spot_metrics.append(
                {
                    "workload_id": j,
                    "scenario": xi,
                    "completion_ratio": active_slots / active_len,
                    "delta": get_value(lookup, "delta", j, xi),
                }
            )
    spot_metrics = pd.DataFrame(spot_metrics, columns=["workload_id", "scenario", "completion_ratio", "delta"])
    if not spot_metrics.empty:
        spot_metrics["weighted_delta"] = spot_metrics.apply(lambda row: data["prob"][row["scenario"]] * row["delta"], axis=1)

    batch_reservation = []
    for k in K:
        for s in S:
            for t in T:
                value = get_value(lookup, "b", k, s, t)
                if value > EPS:
                    batch_reservation.append({"batch_id": k, "server": s, "time": t, "reserved": value})
    batch_reservation = pd.DataFrame(batch_reservation, columns=["batch_id", "server", "time", "reserved"])

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
    for s in S:
        for t in T:
            for xi in Xi:
                u_value = get_value(lookup, "u", s, t)
                load = get_value(lookup, "total_load", s, t, xi)
                bar_load = get_value(lookup, "barL", s, t, xi)
                state_rows.append(
                    {
                        "server": s,
                        "time": t,
                        "scenario": xi,
                        "probability": data["prob"][xi],
                        "u": u_value,
                        "load": load,
                        "bar_load": bar_load,
                        "utilization": load / C if C else 0.0,
                        "capped_utilization": bar_load / C if C else 0.0,
                        "gamma": get_value(lookup, "gamma", s, t, xi),
                        "phi": get_value(lookup, "phi", s, t, xi),
                        "eta": get_value(lookup, "eta", s, xi),
                    }
                )
    server_state = pd.DataFrame(state_rows)

    expected_state = (
        server_state.assign(
            weighted_load=server_state["probability"] * server_state["load"],
            weighted_bar_load=server_state["probability"] * server_state["bar_load"],
            weighted_gamma=server_state["probability"] * server_state["gamma"],
            weighted_phi=server_state["probability"] * server_state["phi"],
        )
        .groupby(["server", "time"], as_index=False)
        .agg(
            u=("u", "max"),
            expected_load=("weighted_load", "sum"),
            expected_bar_load=("weighted_bar_load", "sum"),
            gamma_probability=("weighted_gamma", "sum"),
            phi_probability=("weighted_phi", "sum"),
        )
    )
    expected_state["expected_utilization"] = expected_state["expected_load"] / C
    expected_state["expected_capped_utilization"] = expected_state["expected_bar_load"] / C

    scenario_metrics = (
        server_state.groupby("scenario", as_index=False)
        .agg(
            peak_load=("load", "max"),
            peak_bar_load=("bar_load", "max"),
            peak_utilization=("utilization", "max"),
            gamma_count=("gamma", "sum"),
            phi_count=("phi", "sum"),
            active_server_slots=("u", "sum"),
            total_load=("load", "sum"),
            total_bar_load=("bar_load", "sum"),
        )
        .sort_values(["phi_count", "gamma_count", "peak_utilization"], ascending=False)
    )

    recourse_rows = []
    for xi in Xi:
        for t in T:
            state_at_time = server_state.loc[(server_state["scenario"] == xi) & (server_state["time"] == t)]
            effective_gamma = 0
            for s in S:
                planned_spot = sum(
                    data["d_sp"].get((j, t, xi), 0.0) * get_value(lookup, "y", j, s)
                    for j in J
                    if t in data["spot_active"][j]
                )
                planned_batch = sum(
                    get_value(lookup, "b", k, s, t) * float(data["batch_info"][k]["reserved_cpu"])
                    for k in K
                )
                gamma_value = get_value(lookup, "gamma", s, t, xi)
                if gamma_value > 0.5 and planned_spot + planned_batch > EPS:
                    effective_gamma += 1
            suspended_spots = 0
            active_spots = 0
            for j in J:
                if t not in data["spot_active"][j]:
                    continue
                active_value = sum(get_value(lookup, "yR", j, s, t, xi) for s in S)
                active_spots += active_value
                if active_value < 0.5:
                    suspended_spots += 1
            recourse_rows.append(
                {
                    "scenario": xi,
                    "time": t,
                    "raw_gamma_server_count": state_at_time["gamma"].sum(),
                    "effective_gamma_server_count": effective_gamma,
                    "phi_server_count": state_at_time["phi"].sum(),
                    "suspended_spot_vm_count": suspended_spots,
                    "active_spot_vm_count": active_spots,
                    "max_utilization": state_at_time["utilization"].max(),
                }
            )
    scenario_time_recourse = pd.DataFrame(recourse_rows)

    energy_rows = []
    migration_total = float(migrations["value"].sum()) if not migrations.empty else 0.0
    for s in S:
        idle_energy = data["energy_idle"] * server_time.loc[server_time["server"] == s, "u"].sum()
        cpu_energy = (
            data["energy_cpu"]
            / C
            * server_state.loc[server_state["server"] == s].eval("probability * bar_load").sum()
        )
        energy_rows.append(
            {
                "server": s,
                "idle_energy": idle_energy,
                "expected_cpu_energy": cpu_energy,
                "server_energy_excluding_migration": idle_energy + cpu_energy,
            }
        )
    energy_summary = pd.DataFrame(energy_rows)
    energy_summary.loc[len(energy_summary)] = {
        "server": "migration_total",
        "idle_energy": 0.0,
        "expected_cpu_energy": data["energy_migration"] * migration_total,
        "server_energy_excluding_migration": data["energy_migration"] * migration_total,
    }

    return {
        "server_time": server_time,
        "server_state": server_state,
        "expected_state": expected_state,
        "scenario_time_recourse": scenario_time_recourse,
        "scenario_metrics": scenario_metrics,
        "energy_summary": energy_summary,
        "od_initial": od_initial,
        "od_realized": od_realized,
        "migrations": migrations,
        "spot_initial": spot_initial,
        "spot_activity": spot_activity,
        "spot_metrics": spot_metrics,
        "batch_reservation": batch_reservation,
        "batch_processing": batch_processing,
    }


def write_tables(tables, analysis_dir):
    analysis_dir.mkdir(parents=True, exist_ok=True)
    file_names = {
        "server_time": "server_time_power.csv",
        "server_state": "server_time_state.csv",
        "expected_state": "expected_server_time_state.csv",
        "scenario_time_recourse": "scenario_time_recourse.csv",
        "scenario_metrics": "scenario_metrics.csv",
        "energy_summary": "energy_summary.csv",
        "od_initial": "on_demand_initial_placement.csv",
        "od_realized": "on_demand_realized_placement.csv",
        "migrations": "migration_events.csv",
        "spot_initial": "spot_initial_placement.csv",
        "spot_activity": "spot_activity.csv",
        "spot_metrics": "spot_metrics.csv",
        "batch_reservation": "batch_reservation.csv",
        "batch_processing": "batch_processing.csv",
    }
    for key, filename in file_names.items():
        tables[key].to_csv(analysis_dir / filename, index=False)


def save_fig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def servers_with_workload(data, tables, scenario=None):
    """Servers that carry actual/planned workload, ignoring idle-on u artifacts."""

    servers = set()
    for key, server_col, scenario_col in [
        ("od_realized", "server", "scenario"),
        ("spot_activity", "server", "scenario"),
        ("batch_processing", "server", "scenario"),
    ]:
        frame = tables[key]
        if frame.empty:
            continue
        if scenario is not None:
            frame = frame.loc[frame[scenario_col] == scenario]
        servers.update(int(value) for value in frame[server_col].unique())

    for key, server_col in [
        ("od_initial", "server"),
        ("spot_initial", "server"),
        ("batch_reservation", "server"),
    ]:
        frame = tables[key]
        if not frame.empty:
            servers.update(int(value) for value in frame[server_col].unique())

    migrations = tables["migrations"]
    if not migrations.empty:
        if scenario is not None:
            migrations = migrations.loc[migrations["scenario"] == scenario]
        servers.update(int(value) for value in migrations["from_server"].unique())
        servers.update(int(value) for value in migrations["to_server"].unique())

    return sorted(server for server in data["servers"] if server in servers)


def contrast_text_color(value, vmin, vmax):
    if vmax is None or vmax <= vmin:
        return "#2F2A24"
    position = (float(value) - vmin) / (vmax - vmin)
    return "#FFFFFF" if position >= 0.58 else "#2F2A24"


def annotate_heatmap(ax, pivot, vmin, vmax, number_format, skip_zero=False):
    for row_idx, row_label in enumerate(pivot.index):
        for col_idx, col_label in enumerate(pivot.columns):
            value = float(pivot.loc[row_label, col_label])
            if skip_zero and abs(value) <= EPS:
                continue
            ax.text(
                col_idx,
                row_idx,
                number_format.format(value),
                ha="center",
                va="center",
                fontsize=6.5,
                color=contrast_text_color(value, vmin, vmax),
            )


def heatmap(ax, frame, value_col, title, cmap, vmin=None, vmax=None, servers=None, number_format="{:.2f}", skip_zero=False):
    if servers is not None:
        frame = frame.loc[frame["server"].isin(servers)].copy()
    pivot = frame.pivot(index="server", columns="time", values=value_col).sort_index()
    if vmax is None:
        vmax = float(pivot.values.max()) if pivot.size else 1.0
    if vmin is None:
        vmin = float(pivot.values.min()) if pivot.size else 0.0
    image = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel("server")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    annotate_heatmap(ax, pivot, vmin, vmax, number_format, skip_zero=skip_zero)
    return image


def plot_expected_heatmaps(data, tables, plot_dir):
    expected = tables["expected_state"]
    used_servers = servers_with_workload(data, tables)
    if used_servers:
        expected = expected.loc[expected["server"].isin(used_servers)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    images = [
        heatmap(axes[0, 0], expected, "expected_utilization", "Expected total utilization", "YlOrRd", 0, max(1.0, expected["expected_utilization"].max()), number_format="{:.2f}"),
        heatmap(axes[0, 1], expected, "expected_capped_utilization", "Expected capped utilization", "YlGnBu", 0, 1, number_format="{:.2f}"),
        heatmap(axes[1, 0], expected, "gamma_probability", "Spot/batch suspension probability", "Oranges", 0, 1, number_format="{:.2f}", skip_zero=True),
        heatmap(axes[1, 1], expected, "phi_probability", "On-demand violation probability", "Reds", 0, 1, number_format="{:.2f}", skip_zero=True),
    ]
    for ax, image in zip(axes.flat, images):
        fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    save_fig(fig, plot_dir / "expected_server_heatmaps.png")


def plot_scenario_loads(data, tables, scenario, plot_dir):
    state = tables["server_state"].loc[tables["server_state"]["scenario"] == scenario]
    used_servers = servers_with_workload(data, tables, scenario)
    if used_servers:
        state = state.loc[state["server"].isin(used_servers)].copy()
    fig, ax = plt.subplots(figsize=(15, 6))
    for server, group in state.groupby("server"):
        ordered = group.sort_values("time")
        ax.plot(ordered["time"], ordered["load"], color=SERVER_COLORS[int(server) % len(SERVER_COLORS)], linewidth=1.6, label=f"S{server} load")
        ax.plot(ordered["time"], ordered["bar_load"], color=SERVER_COLORS[int(server) % len(SERVER_COLORS)], linewidth=1.0, linestyle="--", alpha=0.8)
    ax.axhline(data["capacity"], color=CAPACITY_COLOR, linestyle=":", linewidth=1.8, label="capacity")
    ax.set_title(f"Server load and capped load by time - {scenario}")
    ax.set_xlabel("time")
    ax.set_ylabel("CPU load")
    ax.legend(ncol=4, fontsize=8)
    style_axis(ax)
    save_fig(fig, plot_dir / f"scenario_loads_{scenario}.png")


def plot_scenario_stack(data, tables, scenario, plot_dir):
    S = servers_with_workload(data, tables, scenario) or data["servers"]
    T = data["times"]
    I = data["on_demand"]
    J = data["spot"]
    K = data["batch"]
    od = tables["od_realized"].loc[tables["od_realized"]["scenario"] == scenario]
    sp = tables["spot_activity"].loc[tables["spot_activity"]["scenario"] == scenario]
    batch = tables["batch_processing"].loc[tables["batch_processing"]["scenario"] == scenario]

    fig, axes = plt.subplots(len(S), 1, figsize=(16, max(8, 2.0 * len(S))), sharex=True)
    if len(S) == 1:
        axes = [axes]

    for ax, server in zip(axes, S):
        bottom = np.zeros(len(T))
        for workload in I:
            heights = []
            for time in T:
                active = od.loc[(od["server"] == server) & (od["workload_id"] == workload) & (od["time"] == time)]
                heights.append(data["d_od"].get((workload, time, scenario), 0.0) * active["active"].sum())
            ax.bar(T, heights, bottom=bottom, width=0.85, color=pick_color(workload, I, PALETTE_OD), edgecolor="white", linewidth=0.2)
            bottom += np.array(heights)
        for workload in J:
            heights = []
            for time in T:
                active = sp.loc[(sp["server"] == server) & (sp["workload_id"] == workload) & (sp["time"] == time)]
                heights.append(active["demand"].sum())
            ax.bar(T, heights, bottom=bottom, width=0.85, color=pick_color(workload, J, PALETTE_SP), edgecolor="white", linewidth=0.2)
            bottom += np.array(heights)
        for workload in K:
            heights = []
            for time in T:
                active = batch.loc[(batch["server"] == server) & (batch["batch_id"] == workload) & (batch["time"] == time)]
                heights.append(active["processed"].sum())
            ax.bar(T, heights, bottom=bottom, width=0.85, color=pick_color(workload, K, PALETTE_BJ), edgecolor="white", linewidth=0.2)
            bottom += np.array(heights)
        ax.axhline(data["capacity"], color=CAPACITY_COLOR, linestyle="--", linewidth=1.0)
        ax.set_ylabel(f"S{server}")
        style_axis(ax, "y")

    axes[-1].set_xlabel("time")
    axes[0].set_title(f"Stacked realized workload by server - {scenario}")
    save_fig(fig, plot_dir / f"scenario_workload_stack_{scenario}.png")


def contiguous_on_ranges(server_rows, times):
    ranges = []
    start = None
    u_by_time = {int(row.time): float(row.u) for row in server_rows.itertuples(index=False)}
    for time in times:
        is_on = u_by_time.get(time, 0.0) > 0.5
        if is_on and start is None:
            start = time
        if not is_on and start is not None:
            ranges.append((start, time))
            start = None
    if start is not None:
        ranges.append((start, times[-1] + 1))
    return ranges


def complement_ranges(on_ranges, start_time, end_time):
    gaps = []
    cursor = start_time
    for left, right in on_ranges:
        if cursor < left:
            gaps.append((cursor, left))
        cursor = max(cursor, right)
    if cursor < end_time:
        gaps.append((cursor, end_time))
    return gaps


def gantt_segments(data, tables, scenario, mode):
    """Return VM-type workload segments for a Gantt chart.

    mode="nominal" shows the workload that would occupy the reserved/placed
    resources before spot/batch suspension.  mode="actual" shows the realized
    workload after gamma/yR/z recourse decisions.
    """

    S = data["servers"]
    T = data["times"]
    I = data["on_demand"]
    J = data["spot"]
    K = data["batch"]
    segments = []

    od = tables["od_realized"].loc[tables["od_realized"]["scenario"] == scenario].copy()
    if not od.empty:
        for row in od.sort_values(["server", "time", "workload_id"]).itertuples(index=False):
            amount = data["d_od"].get((row.workload_id, int(row.time), scenario), 0.0) * float(row.active)
            if amount > EPS:
                segments.append(
                    {
                        "kind": "od",
                        "label": row.workload_id,
                        "server": int(row.server),
                        "time": int(row.time),
                        "amount": amount,
                    }
                )

    if mode == "nominal":
        initial_spot = tables["spot_initial"]
        if not initial_spot.empty:
            for row in initial_spot.sort_values(["server", "workload_id"]).itertuples(index=False):
                for time in data["spot_active"][row.workload_id]:
                    amount = data["d_sp"].get((row.workload_id, time, scenario), 0.0) * float(row.value)
                    if amount > EPS:
                        segments.append(
                            {
                                "kind": "spot",
                                "label": row.workload_id,
                                "server": int(row.server),
                                "time": int(time),
                                "amount": amount,
                            }
                        )

        reservation = tables["batch_reservation"]
        if not reservation.empty:
            for row in reservation.sort_values(["server", "time", "batch_id"]).itertuples(index=False):
                amount = float(row.reserved) * float(data["batch_info"][row.batch_id]["reserved_cpu"])
                if amount > EPS:
                    segments.append(
                        {
                            "kind": "batch",
                            "label": row.batch_id,
                            "server": int(row.server),
                            "time": int(row.time),
                            "amount": amount,
                        }
                    )
    else:
        sp = tables["spot_activity"].loc[tables["spot_activity"]["scenario"] == scenario].copy()
        if not sp.empty:
            for row in sp.sort_values(["server", "time", "workload_id"]).itertuples(index=False):
                amount = float(row.demand) * float(row.active)
                if amount > EPS:
                    segments.append(
                        {
                            "kind": "spot",
                            "label": row.workload_id,
                            "server": int(row.server),
                            "time": int(row.time),
                            "amount": amount,
                        }
                    )

        batch = tables["batch_processing"].loc[tables["batch_processing"]["scenario"] == scenario].copy()
        if not batch.empty:
            for row in batch.sort_values(["server", "time", "batch_id"]).itertuples(index=False):
                amount = float(row.processed)
                if amount > EPS:
                    segments.append(
                        {
                            "kind": "batch",
                            "label": row.batch_id,
                            "server": int(row.server),
                            "time": int(row.time),
                            "amount": amount,
                        }
                    )

    frame = pd.DataFrame(segments, columns=["kind", "label", "server", "time", "amount"])
    if frame.empty:
        return frame
    frame = frame.loc[frame["server"].isin(S) & frame["time"].isin(T)].copy()
    return frame


def plot_server_type_gantt(data, tables, scenario, plot_dir, mode):
    """Faceted server Gantt chart: OD/SP/Batch stacked by CPU usage."""

    S = data["servers"]
    T = data["times"]
    I = data["on_demand"]
    J = data["spot"]
    K = data["batch"]
    C = data["capacity"]

    state = tables["server_state"].loc[tables["server_state"]["scenario"] == scenario].copy()
    segments = gantt_segments(data, tables, scenario, mode)
    migrations = tables["migrations"].loc[tables["migrations"]["scenario"] == scenario].copy()
    suspension_cells = set()
    if mode == "actual":
        nominal_segments = gantt_segments(data, tables, scenario, "nominal")
        if not nominal_segments.empty:
            affected = nominal_segments.loc[nominal_segments["kind"].isin(["spot", "batch"])]
            suspension_cells = set((int(row.server), int(row.time)) for row in affected.itertuples(index=False))

    segment_servers = set(segments["server"].unique().tolist()) if not segments.empty else set()
    migration_servers = set()
    if not migrations.empty:
        migration_servers.update(int(value) for value in migrations["from_server"].unique())
        migration_servers.update(int(value) for value in migrations["to_server"].unique())
    used_servers = sorted(server for server in S if server in segment_servers or server in migration_servers)
    if not used_servers:
        used_servers = sorted(state.loc[state["load"] > EPS, "server"].unique().tolist())
    if not used_servers:
        used_servers = S[:1]

    if segments.empty:
        workload_by_server_time = pd.DataFrame(columns=["server", "time", "amount"])
    else:
        workload_by_server_time = segments.groupby(["server", "time"], as_index=False)["amount"].sum()
    peak_by_server = {server: 0.0 for server in used_servers}
    for row in workload_by_server_time.itertuples(index=False):
        peak_by_server[int(row.server)] = max(peak_by_server[int(row.server)], float(row.amount) / C if C else 0.0)
    max_cpu = float(workload_by_server_time["amount"].max()) if not workload_by_server_time.empty else C
    y_max = max(C * 1.15, max_cpu * 1.12, 1.0)
    fig_height = max(5.5, 2.35 * len(used_servers) + 1.8)
    fig, axes = plt.subplots(len(used_servers), 1, figsize=(18, fig_height), sharex=True)
    if len(used_servers) == 1:
        axes = [axes]
    axes_by_server = {server: ax for server, ax in zip(used_servers, axes)}

    mode_title = {
        "nominal": "Nominal workload before spot/batch suspension",
        "actual": "Realized workload after spot/batch suspension",
    }[mode]
    mode_filename = {
        "nominal": "server_vm_type_gantt_nominal",
        "actual": "server_vm_type_gantt_actual",
    }[mode]

    od_midpoints = {}
    for row_index, server in enumerate(used_servers):
        ax = axes_by_server[server]
        ax.set_facecolor("#FFFCF7" if row_index % 2 == 0 else "#F3EEE5")
        style_axis(ax, "y")
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

        server_state = state.loc[state["server"] == server].sort_values("time")
        on_ranges = contiguous_on_ranges(server_state, T)

        for left, right in complement_ranges(on_ranges, T[0], T[-1] + 1):
            ax.axvspan(
                left,
                right,
                facecolor=OFF_COLOR,
                edgecolor=OFF_COLOR,
                hatch="///",
                alpha=0.08,
                linewidth=0.0,
                zorder=0,
            )

        for time in T:
            y_cursor = 0.0
            state_row = server_state.loc[server_state["time"] == time]
            if (
                mode == "actual"
                and not state_row.empty
                and float(state_row.iloc[0]["gamma"]) > 0.5
                and (server, time) in suspension_cells
            ):
                ax.add_patch(
                    Rectangle(
                        (time + 0.07, 0.0),
                        0.86,
                        y_max,
                        facecolor="#FDE2E2",
                        edgecolor=ALERT_COLOR,
                        hatch="////",
                        linewidth=1.2,
                        alpha=0.34,
                        zorder=1,
                    )
                )

            cell_segments = segments.loc[(segments["server"] == server) & (segments["time"] == time)].copy()
            if not cell_segments.empty:
                cell_segments["kind_order"] = cell_segments["kind"].map({"od": 0, "spot": 1, "batch": 2})
                cell_segments = cell_segments.sort_values(["kind_order", "label"])
            for row in cell_segments.itertuples(index=False):
                amount = float(row.amount)
                ax.bar(
                    time + 0.5,
                    amount,
                    bottom=y_cursor,
                    width=0.86,
                    align="center",
                    color={
                        "od": pick_color(row.label, I, PALETTE_OD),
                        "spot": pick_color(row.label, J, PALETTE_SP),
                        "batch": pick_color(row.label, K, PALETTE_BJ),
                    }[row.kind],
                    edgecolor="#111111",
                    linewidth=0.35,
                    alpha={"od": 0.92, "spot": 0.90, "batch": 0.88}[row.kind],
                    zorder=3,
                )
                if row.kind == "od":
                    od_midpoints[(row.label, server, time)] = (ax, time + 0.5, y_cursor + 0.5 * amount)
                y_cursor += amount

            if not state_row.empty and float(state_row.iloc[0]["phi"]) > 0.5:
                ax.add_patch(
                    Rectangle(
                        (time + 0.07, 0.0),
                        0.86,
                        y_max * 0.98,
                        facecolor="none",
                        edgecolor=ALERT_COLOR,
                        linewidth=1.1,
                        alpha=0.95,
                        zorder=4,
                    )
                )

        peak = peak_by_server.get(server, 0.0)
        ax.axhline(C, color=CAPACITY_COLOR, linestyle="--", linewidth=1.0, zorder=5)
        ax.set_ylim(0, y_max)
        ax.set_ylabel(f"S{server}\nCPU", rotation=0, labelpad=30, va="center", fontsize=11, fontweight="bold")
        ax.text(
            1.005,
            0.5,
            f"{peak * 100:.0f}%",
            transform=ax.transAxes,
            va="center",
            ha="left",
            fontsize=9,
            color=MUTED_TEXT,
        )
        ax.text(
            -0.015,
            C / y_max,
            "capacity",
            transform=ax.transAxes,
            va="center",
            ha="right",
            fontsize=8,
            color=CAPACITY_COLOR,
        )
        cpu_ticks = [0.0, C / 2.0, C]
        if y_max > C * 1.2:
            cpu_ticks.append(y_max)
        ax.set_yticks(cpu_ticks)

    for event in migrations.itertuples(index=False):
        if int(event.from_server) not in axes_by_server or int(event.to_server) not in axes_by_server:
            continue
        active_times = [time for time in data["od_active"][event.workload_id] if time < int(event.time)]
        prev_time = max(active_times) if active_times else int(event.time) - 1
        source = od_midpoints.get((event.workload_id, int(event.from_server), prev_time))
        target = od_midpoints.get((event.workload_id, int(event.to_server), int(event.time)))
        if source is None:
            source = (axes_by_server[int(event.from_server)], prev_time + 0.5, 0.72 * C)
        if target is None:
            target = (axes_by_server[int(event.to_server)], int(event.time) + 0.5, 0.72 * C)

        source_ax, source_x, source_y = source
        target_ax, target_x, target_y = target
        patch = ConnectionPatch(
            xyA=(source_x, source_y),
            coordsA="data",
            axesA=source_ax,
            xyB=(target_x, target_y),
            coordsB="data",
            axesB=target_ax,
            arrowstyle="->",
            mutation_scale=11,
            linewidth=1.25,
            color="#111827",
            alpha=0.8,
            connectionstyle="arc3,rad=0.12",
            zorder=8,
            clip_on=False,
        )
        fig.add_artist(patch)
        target_ax.scatter([target_x], [target_y], marker="*", s=65, color="#111827", zorder=9)
        target_ax.text(target_x + 0.12, target_y, f"{event.workload_id}", fontsize=7, va="center", color="#111827")

    for ax in axes:
        ax.set_xlim(T[0], T[-1] + 1)
        ax.set_xticks([time + 0.5 for time in T])
        ax.set_xticklabels(T, fontsize=8)

    axes[-1].set_xlabel("time")
    fig.suptitle(
        f"{mode_title} - {scenario}\n"
        "stacked bar height is CPU load; server off and suspension are drawn behind workload bars",
        y=0.995,
    )
    axes[0].text(
        1.018,
        1.02,
        "peak util.",
        transform=axes[0].transAxes,
        rotation=90,
        va="bottom",
        ha="left",
        fontsize=10,
        color=MUTED_TEXT,
    )
    axes[0].legend(
        handles=[
            Patch(facecolor=PALETTE_OD[0], edgecolor="#111111", label="On-demand"),
            Patch(facecolor=PALETTE_SP[0], edgecolor="#111111", label="Spot"),
            Patch(facecolor=PALETTE_BJ[0], edgecolor="#111111", label="Batch"),
            Patch(facecolor=OFF_COLOR, alpha=0.09, hatch="///", label="Server off"),
            Patch(facecolor="#FDE2E2", edgecolor=ALERT_COLOR, hatch="////", label="Spot/batch suspended"),
            FancyArrowPatch((0, 0), (1, 0), arrowstyle="->", color="#111827", label="OD migration"),
        ],
        loc="upper left",
        ncol=6,
        frameon=True,
        edgecolor="#C5BBAA",
        bbox_to_anchor=(0.0, 1.28),
    )
    save_fig(fig, plot_dir / f"{mode_filename}_{scenario}.png")


def plot_workload_timelines(data, tables, scenario, plot_dir):
    od = tables["od_realized"].loc[tables["od_realized"]["scenario"] == scenario].copy()
    sp = tables["spot_activity"].loc[tables["spot_activity"]["scenario"] == scenario].copy()
    migrations = tables["migrations"].loc[tables["migrations"]["scenario"] == scenario].copy()

    workloads = data["on_demand"] + data["spot"]
    y_pos = {workload: idx for idx, workload in enumerate(workloads)}

    fig, ax = plt.subplots(figsize=(16, max(5, 0.45 * len(workloads) + 2)))
    for row in od.itertuples(index=False):
        ax.scatter(row.time, y_pos[row.workload_id], s=70, marker="s", color=SERVER_COLORS[int(row.server) % len(SERVER_COLORS)], edgecolor="white", linewidth=0.4)
    for row in sp.itertuples(index=False):
        ax.scatter(row.time, y_pos[row.workload_id], s=70, marker="o", color=SERVER_COLORS[int(row.server) % len(SERVER_COLORS)], edgecolor="black", linewidth=0.3)

    for row in migrations.itertuples(index=False):
        ax.scatter(row.time, y_pos[row.workload_id], s=150, marker="*", color="#111827", zorder=5)
        ax.text(row.time + 0.15, y_pos[row.workload_id], f"{row.from_server}->{row.to_server}", fontsize=8, va="center")

    ax.set_yticks(range(len(workloads)))
    ax.set_yticklabels(workloads)
    ax.set_xticks(data["times"])
    ax.set_xlabel("time")
    ax.set_title(f"Workload server timeline - {scenario} (square=OD, circle=Spot, star=migration)")
    style_axis(ax)
    save_fig(fig, plot_dir / f"workload_timeline_{scenario}.png")


def plot_batch_views(data, tables, scenario, plot_dir):
    reservation = tables["batch_reservation"]
    processing = tables["batch_processing"].loc[tables["batch_processing"]["scenario"] == scenario]

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    if not reservation.empty:
        pivot_res = reservation.groupby(["batch_id", "time"])["reserved"].sum().unstack(fill_value=0)
        vmax = max(1.0, float(pivot_res.values.max()))
        im = axes[0].imshow(pivot_res.values, aspect="auto", cmap="Greens", vmin=0, vmax=vmax)
        axes[0].set_yticks(range(len(pivot_res.index)))
        axes[0].set_yticklabels(pivot_res.index)
        axes[0].set_title("Batch reserved slots by job")
        axes[0].set_xticks(range(len(pivot_res.columns)))
        axes[0].set_xticklabels(pivot_res.columns, rotation=90, fontsize=7)
        annotate_heatmap(axes[0], pivot_res, 0, vmax, "{:.0f}", skip_zero=True)
        fig.colorbar(im, ax=axes[0], fraction=0.02, pad=0.01)
    if not processing.empty:
        pivot_proc = processing.groupby(["batch_id", "time"])["processed"].sum().unstack(fill_value=0)
        vmax = max(1.0, float(pivot_proc.values.max()))
        im = axes[1].imshow(pivot_proc.values, aspect="auto", cmap="YlGn", vmin=0, vmax=vmax)
        axes[1].set_yticks(range(len(pivot_proc.index)))
        axes[1].set_yticklabels(pivot_proc.index)
        axes[1].set_title(f"Batch processed workload by job - {scenario}")
        axes[1].set_xticks(range(len(pivot_proc.columns)))
        axes[1].set_xticklabels(pivot_proc.columns, rotation=90, fontsize=7)
        annotate_heatmap(axes[1], pivot_proc, 0, vmax, "{:.1f}", skip_zero=True)
        fig.colorbar(im, ax=axes[1], fraction=0.02, pad=0.01)
    axes[1].set_xlabel("time")
    save_fig(fig, plot_dir / f"batch_views_{scenario}.png")


def plot_energy_and_scenarios(data, tables, plot_dir):
    energy = tables["energy_summary"].copy()
    server_energy = energy.loc[energy["server"] != "migration_total"].copy()
    server_energy["server"] = server_energy["server"].astype(int)
    used_servers = servers_with_workload(data, tables)
    if used_servers:
        server_energy = server_energy.loc[server_energy["server"].isin(used_servers)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].bar(server_energy["server"], server_energy["idle_energy"], label="idle", color="#A7C7E7")
    axes[0].bar(
        server_energy["server"],
        server_energy["expected_cpu_energy"],
        bottom=server_energy["idle_energy"],
        label="expected CPU",
        color="#F2C94C",
    )
    axes[0].set_title("Energy by server")
    axes[0].set_xlabel("server")
    axes[0].set_ylabel("energy")
    axes[0].legend()
    style_axis(axes[0], "y")

    metrics = tables["scenario_metrics"].copy()
    axes[1].bar(metrics["scenario"], metrics["peak_utilization"], color="#D37244")
    axes[1].axhline(1.0, color=CAPACITY_COLOR, linestyle="--", linewidth=1.0)
    axes[1].set_title("Peak utilization by scenario")
    axes[1].set_xlabel("scenario")
    axes[1].set_ylabel("peak util")
    axes[1].tick_params(axis="x", rotation=45)
    style_axis(axes[1], "y")
    save_fig(fig, plot_dir / "energy_and_scenario_summary.png")


def plot_spot_metrics(tables, plot_dir):
    spot = tables["spot_metrics"]
    if spot.empty:
        return
    summary = (
        spot.groupby("workload_id", as_index=False)
        .agg(
            mean_completion=("completion_ratio", "mean"),
            min_completion=("completion_ratio", "min"),
            suspension_probability=("weighted_delta", "sum"),
        )
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(summary))
    ax.bar(x - 0.2, summary["mean_completion"], width=0.4, label="mean completion", color="#F2C94C")
    ax.bar(x + 0.2, summary["suspension_probability"], width=0.4, label="suspension probability", color="#E15759")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["workload_id"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Spot completion and suspension risk")
    ax.legend()
    style_axis(ax, "y")
    save_fig(fig, plot_dir / "spot_metrics.png")


def plot_all_scenario_recourse(data, tables, plot_dir):
    """Scenario-time overview of SLA violations and spot/batch suspension."""

    recourse = tables["scenario_time_recourse"]
    if recourse.empty:
        return

    scenarios = data["scenarios"]
    times = data["times"]
    configs = [
        ("phi_server_count", "OD SLA violation server count (phi)", "Reds", 0, None, "{:.0f}"),
        ("effective_gamma_server_count", "Effective spot/batch suspension server count", "Oranges", 0, None, "{:.0f}"),
        ("suspended_spot_vm_count", "Suspended spot VM count", "YlOrBr", 0, None, "{:.0f}"),
        ("max_utilization", "Max server utilization", "YlGnBu", 0, max(1.0, recourse["max_utilization"].max()), "{:.1f}"),
    ]

    fig, axes = plt.subplots(len(configs), 1, figsize=(17, max(9, 1.1 * len(configs) * len(scenarios) / 2)), sharex=True)
    if len(configs) == 1:
        axes = [axes]

    for ax, (column, title, cmap, vmin, vmax, number_format) in zip(axes, configs):
        pivot = (
            recourse.pivot(index="scenario", columns="time", values=column)
            .reindex(index=scenarios, columns=times)
            .fillna(0.0)
        )
        actual_vmax = vmax
        if actual_vmax is None:
            actual_vmax = max(1.0, float(pivot.values.max()) if pivot.size else 1.0)
        image = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=actual_vmax)
        ax.set_title(title)
        ax.set_ylabel("scenario")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=90, fontsize=7)
        annotate_heatmap(ax, pivot, vmin, actual_vmax, number_format, skip_zero=(column != "max_utilization"))
        fig.colorbar(image, ax=ax, fraction=0.015, pad=0.01)

    axes[-1].set_xlabel("time")
    save_fig(fig, plot_dir / "all_scenarios_recourse_overview.png")


def describe_plot_file(filename):
    if filename == "expected_server_heatmaps.png":
        return (
            "사용된 서버만 대상으로 한 기대값 heatmap입니다. "
            "`Expected total utilization`은 scenario probability로 가중한 총 workload/C, "
            "`Expected capped utilization`은 에너지 계산에 들어가는 capped workload/C, "
            "`Spot/batch suspension probability`는 gamma 확률, "
            "`On-demand violation probability`는 phi 확률입니다. 각 cell 숫자가 실제 값입니다."
        )
    if filename == "all_scenarios_recourse_overview.png":
        return (
            "모든 scenario와 time을 한 화면에서 보는 recourse 요약입니다. "
            "위에서부터 OD SLA violation server count(phi), effective spot/batch suspension server count, "
            "suspended spot VM count, max server utilization을 보여줍니다. "
            "count 계열은 0을 생략하고, 짙은 cell의 숫자는 흰색으로 표시합니다."
        )
    if filename.startswith("scenario_loads_"):
        return (
            "선택 scenario의 서버별 load line plot입니다. 실선은 realized total load, 점선은 capped load(barL), "
            "붉은 점선은 capacity입니다. 실제 workload가 없는 서버는 제외합니다."
        )
    if filename.startswith("scenario_workload_stack_"):
        return (
            "선택 scenario에서 suspension 이후 실제 workload를 서버별 stacked bar로 보여줍니다. "
            "파란색은 on-demand, 노란색은 spot, 초록색은 batch이며, 붉은 선은 capacity입니다."
        )
    if filename.startswith("server_vm_type_gantt_nominal_"):
        return (
            "spot/batch suspension을 적용하기 전의 nominal workload Gantt입니다. "
            "OD는 migration 이후 실제 위치 기준, spot은 초기 placement 기준 demand, batch는 reservation capacity 기준으로 표시합니다. "
            "서버별 panel로 분리되어 있고 bar 높이는 CPU load입니다. server off 표시는 CPU bar보다 먼저 그려져 workload를 가리지 않습니다."
        )
    if filename.startswith("server_vm_type_gantt_actual_"):
        return (
            "spot/batch suspension과 batch processing decision을 반영한 realized workload Gantt입니다. "
            "붉은 hatched background는 해당 server-time에서 spot/batch suspension이 실제로 의미 있게 발생한 곳입니다. "
            "server off와 suspension 배경은 CPU bar보다 먼저 그려져 workload를 가리지 않습니다. "
            "붉은 outline은 OD SLA violation(phi)입니다. nominal Gantt와 비교하면 suspension으로 사라진 spot/batch workload를 볼 수 있습니다."
        )
    if filename.startswith("workload_timeline_"):
        return (
            "VM별 server assignment timeline입니다. 사각형은 on-demand, 원은 spot, 별표는 migration event입니다. "
            "색은 배정된 서버를 의미합니다."
        )
    if filename.startswith("batch_views_"):
        return (
            "batch job별 reservation과 processing heatmap입니다. 위 panel은 1st-stage reserved slot 수, "
            "아래 panel은 선택 scenario에서 실제 처리된 workload volume입니다. cell 숫자가 예약/처리량입니다."
        )
    if filename == "energy_and_scenario_summary.png":
        return (
            "왼쪽은 사용된 서버별 idle energy와 expected CPU energy 구성, 오른쪽은 scenario별 peak utilization입니다. "
            "서버 에너지 plot에서는 실제 workload/reservation이 없는 서버를 제외합니다."
        )
    if filename == "spot_metrics.png":
        return (
            "spot VM별 평균 completion ratio와 suspension probability입니다. "
            "completion ratio는 scenario별 active slot 비율, suspension probability는 probability-weighted delta 합입니다."
        )
    return "시각화 보조 그래프입니다. 축 제목과 legend를 기준으로 해석합니다."


def write_plot_descriptions(results_dir, plot_outputs, scenario, migration_scenario):
    lines = [
        "# Visualization Guide",
        "",
        f"- Detail scenario: `{scenario}`",
        f"- Migration-focused scenario: `{migration_scenario}`",
        "- 서버별 그래프는 실제 workload/reservation/migration이 없는 서버를 가능한 한 제외합니다.",
        "- Heatmap의 숫자는 cell 값입니다. 0 count는 대부분 생략하고, 어두운 cell은 흰색 숫자로 표시합니다.",
        "",
        "## Plot Descriptions",
        "",
    ]
    for filename in plot_outputs:
        lines.extend([f"### `{filename}`", describe_plot_file(filename), ""])

    path = results_dir / "plot_descriptions.md"
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines).rstrip() + "\n")
    return path


def main():
    args = parse_args()
    data = load_instance(args.instance)
    solution_rows = load_solution(args.results_dir)
    tables = build_tables(data, solution_rows)

    analysis_dir = args.results_dir / "analysis"
    plot_dir = args.results_dir / "plots"
    write_tables(tables, analysis_dir)
    for stale_gantt in plot_dir.glob("server_vm_type_gantt_*.png"):
        if "_nominal_" not in stale_gantt.name and "_actual_" not in stale_gantt.name:
            stale_gantt.unlink()

    scenario = choose_scenario(data, tables["server_state"], args.scenario)
    migration_scenario = choose_migration_scenario(tables, scenario)
    plot_expected_heatmaps(data, tables, plot_dir)
    plot_all_scenario_recourse(data, tables, plot_dir)
    plot_scenario_loads(data, tables, scenario, plot_dir)
    plot_scenario_stack(data, tables, scenario, plot_dir)
    plot_server_type_gantt(data, tables, scenario, plot_dir, mode="nominal")
    plot_server_type_gantt(data, tables, scenario, plot_dir, mode="actual")
    plot_workload_timelines(data, tables, scenario, plot_dir)
    if migration_scenario != scenario:
        plot_server_type_gantt(data, tables, migration_scenario, plot_dir, mode="nominal")
        plot_server_type_gantt(data, tables, migration_scenario, plot_dir, mode="actual")
        plot_workload_timelines(data, tables, migration_scenario, plot_dir)
    plot_batch_views(data, tables, scenario, plot_dir)
    plot_energy_and_scenarios(data, tables, plot_dir)
    plot_spot_metrics(tables, plot_dir)

    plot_outputs = sorted(path.name for path in plot_dir.glob("*.png"))
    description_path = write_plot_descriptions(args.results_dir, plot_outputs, scenario, migration_scenario)

    manifest = {
        "scenario_used_for_detail_plots": scenario,
        "scenario_used_for_migration_plots": migration_scenario,
        "analysis_dir": str(analysis_dir),
        "plot_dir": str(plot_dir),
        "plot_descriptions": str(description_path),
        "csv_outputs": sorted(path.name for path in analysis_dir.glob("*.csv")),
        "plot_outputs": plot_outputs,
    }
    with open(args.results_dir / "visualization_manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
