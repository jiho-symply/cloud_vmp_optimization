import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle


EXPERIMENT_DIR = Path(__file__).resolve().parent
EXPERIMENT_NAME = EXPERIMENT_DIR.name
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DEFAULT_INSTANCE_DIR = REPO_ROOT / "data" / "processed" / EXPERIMENT_NAME / "chance_2sp_toy_24vm_ratio_balanced_od8_sp8_bj8_sc10_cap8_avg20_lam010"

PALETTE_OD = ["#4E79A7", "#6B95C2", "#35618D", "#8CB2D9", "#264B6A", "#A3C2E3", "#5B84B1"]
PALETTE_BJ = ["#5B8E7D", "#78A99A", "#3F6F60", "#91BCB0", "#2F5A4D", "#A8CCC2", "#6E9F8E"]
SERVER_COLORS = ["#4E79A7", "#F28E2B", "#59A14F", "#B07AA1", "#E15759", "#76B7B2", "#9C755F", "#BAB0AC"]
COLOR_SPOT = "#E3A008"
COLOR_OFF = "#404040"
COLOR_PROXY = "#A07BEF"
COLOR_REALIZED = "#D37244"
COLOR_ALERT = "#C94F4F"
COLOR_INACTIVE = "#F4F1EA"
FIG_BG = "#F6F3EE"
AX_BG = "#FFFCF7"
GRID_COLOR = "#D9D1C3"
SPINE_COLOR = "#C5BBAA"
TEXT_COLOR = "#2F2A24"
MUTED_TEXT = "#6F675D"

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "util_heat",
    ["#FFF8E8", "#F5C98B", "#E38B5B", "#BC4B51"],
)
SPOT_RISK_CMAP = LinearSegmentedColormap.from_list(
    "spot_risk",
    ["#9ED9CC", "#F3C567", "#D37244", "#BC4B51"],
)
SPOT_SHARE_CMAP = LinearSegmentedColormap.from_list(
    "spot_share",
    ["#DCE7F3", "#8CB2D9", "#4E79A7", "#264B6A"],
)

ROW_HEIGHT = 2.0
ROW_GAP = 0.35
EPS = 1e-9
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = FIG_BG
plt.rcParams["axes.facecolor"] = AX_BG
plt.rcParams["axes.edgecolor"] = SPINE_COLOR
plt.rcParams["axes.labelcolor"] = TEXT_COLOR
plt.rcParams["xtick.color"] = TEXT_COLOR
plt.rcParams["ytick.color"] = TEXT_COLOR
plt.rcParams["text.color"] = TEXT_COLOR
plt.rcParams["grid.color"] = GRID_COLOR
plt.rcParams["grid.alpha"] = 0.45
plt.rcParams["axes.titleweight"] = "bold"


class HandlerLegendArrow(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        y = ydescent + 0.5 * height
        patch = FancyArrowPatch(
            (xdescent, y),
            (xdescent + width, y),
            arrowstyle="->",
            mutation_scale=12,
            linewidth=1.1,
            color="#111827",
            connectionstyle="arc3,rad=0.0",
        )
        patch.set_transform(trans)
        return [patch]


def parse_args():
    parser = argparse.ArgumentParser(description="chance-constrained 2SP toy 결과를 시각화합니다.")
    parser.add_argument("--instance-dir", type=Path, default=DEFAULT_INSTANCE_DIR)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--scenario", type=str, default=None)
    return parser.parse_args()


def pick_color(index_value, palette):
    return palette[int(index_value) % len(palette)]


def sort_workload_ids(workload_ids):
    return sorted(workload_ids, key=lambda value: (value.split("_")[0], int(value.split("_")[1])))


def get_server_color(server):
    return SERVER_COLORS[int(server) % len(SERVER_COLORS)]


def style_axis(ax, grid_axis="y"):
    ax.set_facecolor(AX_BG)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(0.9)

    if grid_axis == "both":
        ax.grid(True, axis="both", linewidth=0.8)
    elif grid_axis in {"x", "y"}:
        ax.grid(True, axis=grid_axis, linewidth=0.8)
    else:
        ax.grid(False)

    ax.tick_params(labelsize=10)
    return ax


def safe_read_csv(path, columns):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=columns)


def load_case_files(instance_dir, results_dir):
    with open(instance_dir / "instance.json", "r", encoding="utf-8") as file:
        instance = json.load(file)
    with open(results_dir / "summary.json", "r", encoding="utf-8") as file:
        summary = json.load(file)

    return {
        "instance": instance,
        "summary": summary,
        "metadata": safe_read_csv(instance_dir / "workload_metadata.csv", ["workload_id", "vm_id", "workload_type", "vCPU"]),
        "scenario_time_series": safe_read_csv(instance_dir / "scenario_time_series.csv", ["workload_id", "workload_type", "time", "scenario", "demand"]),
        "batch_jobs": safe_read_csv(instance_dir / "batch_jobs.csv", ["batch_job_id", "parent_workload_id", "source_time", "vm_id", "vCPU", "min_core_usage", "avg_core_usage", "max_core_usage"]),
        "batch_job_demands": safe_read_csv(instance_dir / "batch_job_demands.csv", ["batch_job_id", "scenario", "demand"]),
        "scenario_summary": safe_read_csv(instance_dir / "scenario_summary.csv", ["scenario", "probability"]),
        "on_demand_placement": safe_read_csv(results_dir / "on_demand_placement.csv", ["workload_id", "time", "server"]),
        "spot_placement": safe_read_csv(results_dir / "spot_placement.csv", ["workload_id", "server"]),
        "batch_schedule": safe_read_csv(results_dir / "batch_schedule.csv", ["batch_job_id", "parent_workload_id", "source_time", "server", "time"]),
        "scenario_server_state": safe_read_csv(results_dir / "scenario_server_state.csv", ["server", "time", "scenario", "load", "committed_load", "gamma", "phi", "u", "realized_utilization", "overbooking_ratio"]),
        "scenario_cluster_state": safe_read_csv(results_dir / "scenario_cluster_state.csv", ["scenario", "time", "cluster_load", "cluster_committed_load", "active_servers", "cluster_overbooking_ratio"]),
        "spot_metrics": safe_read_csv(results_dir / "spot_metrics.csv", ["workload_id", "suspension_probability", "completion_ratio"]),
        "spot_activity_state": safe_read_csv(results_dir / "spot_activity_state.csv", ["workload_id", "scenario", "time", "server", "active", "suspended"]),
        "scenario_metrics": safe_read_csv(results_dir / "scenario_metrics.csv", ["scenario", "peak_server_load", "peak_committed_server_load", "peak_server_overbooking_ratio", "gamma_count", "phi_count", "active_server_periods", "peak_cluster_load", "peak_cluster_committed_load", "peak_cluster_overbooking_ratio", "peak_server_utilization"]),
    }


def choose_scenario(case_data, requested_scenario=None):
    if requested_scenario:
        return requested_scenario

    if case_data["summary"].get("worst_realized_scenario"):
        return case_data["summary"]["worst_realized_scenario"]

    scenario_metrics = case_data["scenario_metrics"]
    return scenario_metrics.sort_values(
        ["peak_server_load", "gamma_count", "phi_count"],
        ascending=[False, False, False],
    )["scenario"].iloc[0]


def on_intervals(values, times):
    intervals = []
    start = None

    for time, value in zip(times, values):
        is_on = int(round(value)) == 1
        if start is None and is_on:
            start = time
        if start is not None and not is_on:
            intervals.append((start, time))
            start = None

    if start is not None:
        intervals.append((start, times[-1] + 1))

    return intervals


def off_intervals(on_ranges, start, end):
    gaps = []
    cursor = start

    for left, right in on_ranges:
        if cursor < left:
            gaps.append((cursor, left))
        cursor = max(cursor, right)

    if cursor < end:
        gaps.append((cursor, end))

    return gaps


def build_plot_frames(case_data, scenario_name):
    capacity = float(case_data["instance"]["server_capacity"])
    scenario_state = case_data["scenario_server_state"]
    scenario_state = scenario_state.loc[scenario_state["scenario"] == scenario_name].copy()

    od_demand = case_data["scenario_time_series"].loc[
        (case_data["scenario_time_series"]["scenario"] == scenario_name)
        & (case_data["scenario_time_series"]["workload_type"] == "on_demand"),
        ["workload_id", "time", "demand"],
    ].rename(columns={"demand": "od_demand"})

    batch_demand = case_data["batch_job_demands"].loc[
        case_data["batch_job_demands"]["scenario"] == scenario_name,
        ["batch_job_id", "demand"],
    ].rename(columns={"demand": "batch_demand"})

    od_stack = case_data["on_demand_placement"].merge(od_demand, on=["workload_id", "time"], how="left")
    batch_stack = case_data["batch_schedule"].merge(batch_demand, on="batch_job_id", how="left")

    od_total = od_stack.groupby(["server", "time"], as_index=False)["od_demand"].sum()
    batch_total = batch_stack.groupby(["server", "time"], as_index=False)["batch_demand"].sum()

    spot_total = scenario_state.merge(od_total, on=["server", "time"], how="left").merge(
        batch_total, on=["server", "time"], how="left"
    )
    spot_total["od_demand"] = spot_total["od_demand"].fillna(0.0)
    spot_total["batch_demand"] = spot_total["batch_demand"].fillna(0.0)
    spot_total["spot_demand"] = (spot_total["load"] - spot_total["od_demand"] - spot_total["batch_demand"]).clip(lower=0.0)
    spot_total["utilization"] = spot_total["load"] / capacity

    return scenario_state, od_stack, batch_stack, spot_total


def build_spot_activity_tables(case_data):
    activity = case_data["spot_activity_state"].copy()
    if activity.empty:
        return (
            pd.DataFrame(columns=["workload_id", "scenario", "time", "server", "active", "suspended"]),
            pd.DataFrame(columns=["workload_id", "scenario", "server", "start_time", "end_time", "duration_hours"]),
        )

    activity["time"] = activity["time"].astype(int)
    if "server" in activity:
        activity["server"] = activity["server"].fillna(-1).astype(int)

    suspension_rows = []
    for (workload_id, scenario_name), group in activity.groupby(["workload_id", "scenario"]):
        ordered = group.sort_values("time")
        suspended_start = None
        suspended_last = None
        suspended_server = None

        for row in ordered.itertuples(index=False):
            is_suspended = int(row.suspended) == 1
            if is_suspended and suspended_start is None:
                suspended_start = int(row.time)
                suspended_server = int(row.server)
            if is_suspended:
                suspended_last = int(row.time)
            if not is_suspended and suspended_start is not None:
                suspension_rows.append(
                    {
                        "workload_id": workload_id,
                        "scenario": scenario_name,
                        "server": suspended_server,
                        "start_time": suspended_start,
                        "end_time": suspended_last,
                        "duration_hours": suspended_last - suspended_start + 1,
                    }
                )
                suspended_start = None
                suspended_last = None
                suspended_server = None

        if suspended_start is not None:
            suspension_rows.append(
                {
                    "workload_id": workload_id,
                    "scenario": scenario_name,
                    "server": suspended_server,
                    "start_time": suspended_start,
                    "end_time": suspended_last,
                    "duration_hours": suspended_last - suspended_start + 1,
                }
            )

    suspensions = pd.DataFrame(suspension_rows)
    if suspensions.empty:
        suspensions = pd.DataFrame(columns=["workload_id", "scenario", "server", "start_time", "end_time", "duration_hours"])

    return activity, suspensions


def build_on_demand_migration_events(case_data):
    migration_rows = []
    placement = case_data["on_demand_placement"].copy()
    placement["time"] = placement["time"].astype(int)
    placement["server"] = placement["server"].astype(int)

    for workload_id, group in placement.groupby("workload_id"):
        ordered = group.sort_values("time")
        previous_server = None
        previous_time = None

        for row in ordered.itertuples(index=False):
            if previous_server is not None and row.server != previous_server:
                migration_rows.append(
                    {
                        "workload_id": workload_id,
                        "from_time": int(previous_time),
                        "time": int(row.time),
                        "to_time": int(row.time),
                        "from_server": int(previous_server),
                        "to_server": int(row.server),
                    }
                )
            previous_server = int(row.server)
            previous_time = int(row.time)

    migrations = pd.DataFrame(migration_rows)
    if migrations.empty:
        migrations = pd.DataFrame(columns=["workload_id", "from_time", "time", "to_time", "from_server", "to_server"])
    return migrations


def choose_spot_comparison_scenarios(case_data, spot_activity, default_scenario):
    scenario_order = [scenario["scenario"] for scenario in case_data["instance"]["scenarios"]]
    if spot_activity.empty:
        return None, None

    suspended_hours = (
        spot_activity.groupby("scenario", as_index=False)["suspended"]
        .sum()
        .sort_values(["suspended", "scenario"], ascending=[False, True])
        .reset_index(drop=True)
    )
    suspended_scenario = suspended_hours.loc[suspended_hours["suspended"] > 0, "scenario"]
    clean_scenario = suspended_hours.loc[suspended_hours["suspended"] == 0, "scenario"]

    suspended_choice = suspended_scenario.iloc[0] if not suspended_scenario.empty else None
    if suspended_choice is None and default_scenario in scenario_order:
        suspended_choice = default_scenario

    if not clean_scenario.empty:
        clean_choice = clean_scenario.iloc[0]
    elif default_scenario in scenario_order:
        clean_choice = default_scenario
    else:
        clean_choice = scenario_order[0]

    return suspended_choice, clean_choice


def create_gantt_figure(case_data, scenario_name, output_path, migration_events=None):
    times = [int(time) for time in case_data["instance"]["time_periods"]]
    used_servers = case_data["summary"].get("used_servers", list(range(case_data["instance"]["num_servers"])))
    capacity = float(case_data["instance"]["server_capacity"])

    scenario_state, od_stack, batch_stack, spot_total = build_plot_frames(case_data, scenario_name)
    scenario_metrics = case_data["scenario_metrics"].set_index("scenario")
    peak_utilization = float(scenario_metrics.loc[scenario_name, "peak_server_utilization"])
    gamma_count = int(scenario_metrics.loc[scenario_name, "gamma_count"])
    phi_count = int(scenario_metrics.loc[scenario_name, "phi_count"])
    if migration_events is None:
        migration_events = build_on_demand_migration_events(case_data)
    migration_count = int(len(migration_events))

    fig_height = max(4.8, 1.0 * len(used_servers) + 1.8)
    fig, ax = plt.subplots(figsize=(17, fig_height), facecolor=FIG_BG)
    style_axis(ax, grid_axis="x")

    row_base = {server: index * (ROW_HEIGHT + ROW_GAP) for index, server in enumerate(used_servers)}
    max_y = len(used_servers) * (ROW_HEIGHT + ROW_GAP)
    od_midpoint = {}

    for server in used_servers:
        base = row_base[server]
        server_state = scenario_state.loc[scenario_state["server"] == server].sort_values("time")
        on_ranges = on_intervals(server_state["u"].tolist(), times)

        ax.plot([times[0], times[-1] + 1], [base, base], color=SPINE_COLOR, linewidth=0.8, alpha=0.55)
        ax.plot([times[0], times[-1] + 1], [base + ROW_HEIGHT, base + ROW_HEIGHT], color=SPINE_COLOR, linewidth=0.9, alpha=0.65)

        for left, right in off_intervals(on_ranges, times[0], times[-1] + 1):
            ax.broken_barh(
                [(left, right - left)],
                (base, ROW_HEIGHT),
                facecolors=COLOR_OFF,
                edgecolor=COLOR_OFF,
                alpha=0.08,
                linewidth=0.0,
                hatch="///",
            )

        for time in times:
            y = base

            od_rows = od_stack.loc[(od_stack["server"] == server) & (od_stack["time"] == time)].sort_values("workload_id")
            for row in od_rows.itertuples(index=False):
                if row.od_demand <= EPS:
                    continue
                height = (row.od_demand / capacity) * ROW_HEIGHT
                ax.broken_barh(
                    [(time, 1.0)],
                    (y, height),
                    facecolors=pick_color(int(row.workload_id.split("_")[1]), PALETTE_OD),
                    edgecolor=AX_BG,
                    linewidth=0.45,
                    alpha=0.97,
                )
                od_midpoint[(row.workload_id, time)] = (time + 0.5, y + 0.5 * height)
                y += height

            spot_value = float(
                spot_total.loc[
                    (spot_total["server"] == server) & (spot_total["time"] == time),
                    "spot_demand",
                ].sum()
            )
            if spot_value > EPS:
                height = (spot_value / capacity) * ROW_HEIGHT
                ax.broken_barh(
                    [(time, 1.0)],
                    (y, height),
                    facecolors=COLOR_SPOT,
                    edgecolor=AX_BG,
                    linewidth=0.45,
                    alpha=0.94,
                )
                y += height

            batch_rows = batch_stack.loc[(batch_stack["server"] == server) & (batch_stack["time"] == time)].sort_values("batch_job_id")
            for row in batch_rows.itertuples(index=False):
                if row.batch_demand <= EPS:
                    continue
                height = (row.batch_demand / capacity) * ROW_HEIGHT
                ax.broken_barh(
                    [(time, 1.0)],
                    (y, height),
                    facecolors=pick_color(int(str(row.parent_workload_id).split("_")[1]), PALETTE_BJ),
                    edgecolor=AX_BG,
                    linewidth=0.45,
                    alpha=0.95,
                )
                y += height

        server_peak = server_state["load"].max() / capacity if not server_state.empty else 0.0
        ax.text(
            times[-1] + 1.18,
            base + 0.5 * ROW_HEIGHT,
            f"{server_peak * 100:.0f}%",
            va="center",
            ha="left",
            fontsize=9,
            color=MUTED_TEXT,
            bbox=dict(boxstyle="round,pad=0.18", facecolor=FIG_BG, edgecolor="none", alpha=0.95),
        )

    arrow_style = dict(
        arrowstyle="->",
        linewidth=1.1,
        alpha=0.88 if migration_count <= 40 else 0.48,
        color="#111827",
        connectionstyle="angle3,angleA=90,angleB=0",
        mutation_scale=10,
    )
    for event in migration_events.itertuples(index=False):
        source = od_midpoint.get((event.workload_id, int(event.from_time)))
        target = od_midpoint.get((event.workload_id, int(event.to_time)))
        if source is None or target is None:
            continue
        x_from = int(event.from_time) + 0.5
        x_to = int(event.to_time) + 0.5
        _, y_from = source
        _, y_to = target
        ax.annotate("", xy=(x_to, y_to), xytext=(x_from, y_from), arrowprops=arrow_style)

    ax.set_xlim(times[0], times[-1] + 2.2)
    ax.set_ylim(-ROW_GAP, max_y)
    ax.set_xlabel("시간")
    ax.set_ylabel("서버")
    ax.set_yticks([row_base[server] + 0.5 * ROW_HEIGHT for server in used_servers])
    ax.set_yticklabels([f"S{server}" for server in used_servers])
    ax.set_title(
        f"{case_data['instance']['instance_name']}\n"
        f"실제 worst-case scenario {scenario_name} | 최대 이용률 {peak_utilization * 100:.1f}%",
        pad=28,
    )
    insight_text = (
        f"사용 서버 {len(used_servers)}대  |  migration {migration_count}회  |  "
        f"spot suspension flag γ {gamma_count}  |  SLA flag φ {phi_count}"
    )
    ax.text(
        0.01,
        1.02,
        insight_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color=TEXT_COLOR,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#F1ECE1", edgecolor="none", alpha=0.98),
    )
    ax.legend(
        handles=[
            Patch(facecolor=COLOR_OFF, alpha=0.10, hatch="///", label="서버 OFF"),
            Patch(facecolor=PALETTE_OD[0], label="On-demand"),
            Patch(facecolor=COLOR_SPOT, label="Spot"),
            Patch(facecolor=PALETTE_BJ[0], label="Batch"),
            FancyArrowPatch((0, 0), (1, 0), arrowstyle="->", label="Migration"),
        ],
        loc="upper left",
        frameon=True,
        edgecolor=SPINE_COLOR,
        handler_map={FancyArrowPatch: HandlerLegendArrow()},
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def create_diagnostics_figure(case_data, scenario_name, output_path):
    capacity = float(case_data["instance"]["server_capacity"])
    scenario_order = [scenario["scenario"] for scenario in case_data["instance"]["scenarios"]]
    used_servers = case_data["summary"].get("used_servers", list(range(case_data["instance"]["num_servers"])))

    scenario_metrics = case_data["scenario_metrics"].copy()
    scenario_metrics = scenario_metrics.set_index("scenario").reindex(scenario_order).reset_index()

    peak_utilization = (
        case_data["scenario_server_state"]
        .groupby(["scenario", "server"])["load"]
        .max()
        .unstack(fill_value=0.0)
        .reindex(index=scenario_order, columns=used_servers, fill_value=0.0)
        / capacity
        * 100.0
    )
    scenario_summary = case_data["scenario_summary"].set_index("scenario").reindex(scenario_order).reset_index()
    cluster_state = case_data["scenario_cluster_state"].copy()
    if cluster_state.empty:
        cluster_ratio = pd.Series(0.0, index=scenario_order)
    else:
        cluster_ratio = (
            cluster_state.groupby("scenario")["cluster_overbooking_ratio"]
            .max()
            .reindex(scenario_order)
            .fillna(0.0)
            * 100.0
        )
    selected_index = scenario_order.index(scenario_name)

    fig, axes = plt.subplots(2, 2, figsize=(17.5, 10.5), facecolor=FIG_BG)
    for row in axes:
        for ax in row:
            style_axis(ax, grid_axis="y")

    heatmap_ax = axes[0, 0]
    heatmap = heatmap_ax.imshow(
        peak_utilization.values,
        cmap=HEATMAP_CMAP,
        aspect="auto",
        vmin=0,
        vmax=max(100.0, peak_utilization.values.max() * 1.05),
    )
    heatmap_ax.grid(False)
    heatmap_ax.set_title("시나리오별 서버 최대 이용률", loc="left")
    heatmap_ax.set_xticks(range(len(used_servers)))
    heatmap_ax.set_xticklabels([f"S{server}" for server in used_servers])
    heatmap_ax.set_yticks(range(len(scenario_order)))
    heatmap_ax.set_yticklabels(scenario_order)
    heatmap_ax.add_patch(
        Rectangle(
            (-0.5, selected_index - 0.5),
            len(used_servers),
            1.0,
            fill=False,
            edgecolor=COLOR_ALERT,
            linewidth=2.0,
        )
    )
    plt.colorbar(heatmap, ax=heatmap_ax, fraction=0.046, pad=0.04, label="최대 부하 / 용량 (%)")

    x = np.arange(len(scenario_order))
    stress_ax = axes[0, 1]
    stress_ax.bar(
        x,
        cluster_ratio.values,
        color="#D7E6D1",
        edgecolor="none",
        label="클러스터 총수요 / 사용 서버 총용량",
    )
    stress_ax.plot(
        x,
        scenario_metrics["peak_server_utilization"] * 100.0,
        color=COLOR_REALIZED,
        linewidth=2.4,
        marker="o",
        label="가장 바쁜 서버 이용률",
    )
    stress_ax.scatter(
        [selected_index],
        [scenario_metrics.loc[selected_index, "peak_server_utilization"] * 100.0],
        s=120,
        color=COLOR_ALERT,
        edgecolors="white",
        linewidth=1.0,
        zorder=3,
    )
    stress_ax.axhline(100.0, color=SPINE_COLOR, linestyle="--", linewidth=1.0)
    stress_ax.set_title("시나리오별 전체 압박과 단일 서버 병목", loc="left")
    stress_ax.set_xticks(x)
    stress_ax.set_xticklabels(scenario_order, rotation=35)
    stress_ax.set_ylabel("사용 비율 (%)")
    stress_ax.legend(frameon=True, edgecolor=SPINE_COLOR, loc="upper left")

    recourse_ax = axes[1, 0]
    recourse_ax.bar(x, scenario_metrics["gamma_count"], color=COLOR_SPOT, label="spot suspension flag γ")
    recourse_ax.bar(
        x,
        scenario_metrics["phi_count"],
        bottom=scenario_metrics["gamma_count"],
        color=COLOR_ALERT,
        label="SLA violation flag φ",
    )
    recourse_ax.axvspan(selected_index - 0.45, selected_index + 0.45, color="#F1ECE1", alpha=0.95, zorder=0)
    recourse_ax.set_title("시나리오별 recourse 사용량", loc="left")
    recourse_ax.set_xticks(x)
    recourse_ax.set_xticklabels(scenario_order, rotation=35)
    recourse_ax.set_ylabel("flag count")
    recourse_ax.legend(frameon=True, edgecolor=SPINE_COLOR, loc="upper left")
    if float(scenario_metrics["gamma_count"].sum()) == 0.0 and float(scenario_metrics["phi_count"].sum()) == 0.0:
        recourse_ax.set_ylim(0, 1)
        recourse_ax.text(
            0.5,
            0.5,
            "활성화된 recourse flag 없음",
            transform=recourse_ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color=MUTED_TEXT,
        )

    spot_ax = axes[1, 1]
    spot_metrics = case_data["spot_metrics"].copy()
    if spot_metrics.empty:
        spot_ax.axis("off")
        spot_ax.text(
            0.5,
            0.5,
            "이 케이스에는 spot VM이 없습니다.",
            ha="center",
            va="center",
            fontsize=12,
            color=MUTED_TEXT,
        )
    else:
        spot_metrics["risk_score"] = 0.65 * spot_metrics["suspension_probability"] + 0.35 * (1.0 - spot_metrics["completion_ratio"])
        spot_metrics = spot_metrics.sort_values(
            ["risk_score", "suspension_probability", "completion_ratio"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        colors = SPOT_RISK_CMAP(np.linspace(0.2, 0.95, len(spot_metrics)))
        spot_ax.scatter(
            spot_metrics["suspension_probability"],
            spot_metrics["completion_ratio"],
            s=95,
            c=colors,
            edgecolors="white",
            linewidth=1.0,
        )
        for row in spot_metrics.head(min(4, len(spot_metrics))).itertuples(index=False):
            spot_ax.annotate(
                row.workload_id,
                (row.suspension_probability, row.completion_ratio),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=9,
                color=TEXT_COLOR,
            )
        epsilon_sp = case_data["instance"]["chance_constraints"]["epsilon_sp"]
        rho = case_data["instance"]["chance_constraints"]["rho"]
        spot_ax.axvline(epsilon_sp, color=SPINE_COLOR, linestyle=":", linewidth=1.2, label="epsilon_sp")
        spot_ax.axhline(rho, color=MUTED_TEXT, linestyle="--", linewidth=1.2, label="rho")
        spot_ax.set_xlim(-0.01, max(0.25, spot_metrics["suspension_probability"].max() + 0.04))
        spot_ax.set_ylim(min(0.75, spot_metrics["completion_ratio"].min() - 0.05), 1.02)
        spot_ax.set_xlabel("중단 확률")
        spot_ax.set_ylabel("완료 비율")
        spot_ax.set_title("Spot VM SLA 분포", loc="left")
        spot_ax.legend(frameon=True, edgecolor=SPINE_COLOR, loc="lower left")

    fig.suptitle(
        f"{case_data['instance']['instance_name']}  |  선택 scenario: {scenario_name}",
        fontsize=15,
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def create_spot_activity_figure(case_data, spot_activity, scenario_name, output_path):
    scenario_order = [scenario["scenario"] for scenario in case_data["instance"]["scenarios"]]
    times = [int(time) for time in case_data["instance"]["time_periods"]]

    if spot_activity.empty:
        fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=FIG_BG)
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "이 케이스에는 spot VM이 없습니다.\n따라서 시나리오별 실행/중단 그림이 비어 있습니다.",
            ha="center",
            va="center",
            fontsize=12,
            color=MUTED_TEXT,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return

    total_suspended_by_vm = (
        spot_activity.groupby("workload_id", as_index=False)["suspended"]
        .sum()
        .sort_values(["suspended", "workload_id"], ascending=[False, True])
    )
    spot_ids = total_suspended_by_vm["workload_id"].tolist()
    time_index = {time: idx for idx, time in enumerate(times)}
    spot_index = {workload_id: idx for idx, workload_id in enumerate(spot_ids)}

    scenario_count = len(scenario_order)
    ncols = min(5, scenario_count)
    nrows = int(math.ceil(scenario_count / ncols))
    fig_height = max(6.0, 1.2 + 0.34 * len(spot_ids) * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, fig_height), squeeze=False, facecolor=FIG_BG)

    cmap = ListedColormap([COLOR_OFF] + [get_server_color(server) for server in range(case_data["instance"]["num_servers"])])
    cmap.set_bad(COLOR_INACTIVE)
    selected_border_color = COLOR_ALERT
    total_suspended_vm_hours = int(spot_activity["suspended"].sum())
    suspended_lookup = total_suspended_by_vm.set_index("workload_id")["suspended"].to_dict()

    for scenario_index, scenario_label in enumerate(scenario_order):
        row = scenario_index // ncols
        col = scenario_index % ncols
        ax = axes[row][col]
        style_axis(ax, grid_axis="none")

        matrix = np.full((len(spot_ids), len(times)), np.nan)
        scenario_rows = spot_activity.loc[spot_activity["scenario"] == scenario_label]
        for activity_row in scenario_rows.itertuples(index=False):
            value = int(activity_row.server) + 1 if int(activity_row.active) == 1 else 0
            matrix[spot_index[activity_row.workload_id], time_index[int(activity_row.time)]] = value

        ax.imshow(
            matrix,
            aspect="auto",
            interpolation="none",
            cmap=cmap,
            vmin=-0.5,
            vmax=case_data["instance"]["num_servers"] + 0.5,
        )
        suspended_hours = int(scenario_rows["suspended"].sum())
        ax.set_title(f"{scenario_label}\n중단 VM-hour {suspended_hours}", fontsize=10)
        ax.set_xticks(range(0, len(times), 4))
        ax.set_xticklabels([times[idx] for idx in range(0, len(times), 4)])
        if col == 0:
            ytick_labels = [f"{workload_id} · {int(suspended_lookup[workload_id])}" for workload_id in spot_ids]
            ax.set_yticks(range(len(spot_ids)))
            ax.set_yticklabels(ytick_labels, fontsize=8)
            ax.set_ylabel("Spot VM · 총 중단 VM-hour")
        else:
            ax.set_yticks(range(len(spot_ids)))
            ax.set_yticklabels([])
        ax.set_xlabel("시간")
        ax.set_xticks(np.arange(-0.5, len(times), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(spot_ids), 1), minor=True)
        ax.grid(which="minor", color=AX_BG, linewidth=0.45)
        ax.tick_params(which="minor", bottom=False, left=False)

        if scenario_label == scenario_name:
            for spine in ax.spines.values():
                spine.set_edgecolor(selected_border_color)
                spine.set_linewidth(2.0)

    total_axes = nrows * ncols
    for empty_index in range(scenario_count, total_axes):
        row = empty_index // ncols
        col = empty_index % ncols
        axes[row][col].axis("off")

    legend_handles = [Patch(facecolor=COLOR_OFF, label="중단")] + [
        Patch(facecolor=get_server_color(server), label=f"S{server}")
        for server in range(case_data["instance"]["num_servers"])
    ]
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=min(5, len(legend_handles)),
        frameon=True,
        edgecolor=SPINE_COLOR,
    )
    fig.suptitle(
        f"Spot VM 시나리오별 실행/중단 상태  |  전체 중단 VM-hour {total_suspended_vm_hours}  |  "
        f"빨간 테두리는 상세 scenario ({scenario_name})",
        y=0.995,
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def create_on_demand_migration_figure(case_data, migration_events, output_path):
    times = [int(time) for time in case_data["instance"]["time_periods"]]
    placement = case_data["on_demand_placement"].copy()

    if placement.empty:
        fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=FIG_BG)
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "이 케이스에는 on-demand VM이 없습니다.",
            ha="center",
            va="center",
            fontsize=12,
            color=MUTED_TEXT,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return

    placement["time"] = placement["time"].astype(int)
    placement["server"] = placement["server"].astype(int)

    migration_counts = migration_events.groupby("workload_id").size().to_dict()
    od_ids = sorted(
        placement["workload_id"].unique().tolist(),
        key=lambda workload_id: (-migration_counts.get(workload_id, 0), workload_id),
    )
    time_index = {time: idx for idx, time in enumerate(times)}
    od_index = {workload_id: idx for idx, workload_id in enumerate(od_ids)}

    matrix = np.full((len(od_ids), len(times)), np.nan)
    for row in placement.itertuples(index=False):
        matrix[od_index[row.workload_id], time_index[row.time]] = row.server

    hourly_counts = migration_events.groupby("time").size().reindex(times, fill_value=0)

    fig_height = max(4.8, 1.5 + 0.42 * len(od_ids))
    fig = plt.figure(figsize=(18, fig_height), facecolor=FIG_BG)
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, max(2.8, 0.35 * len(od_ids))], hspace=0.08)
    top_ax = fig.add_subplot(grid[0, 0])
    heat_ax = fig.add_subplot(grid[1, 0], sharex=top_ax)

    style_axis(top_ax, grid_axis="y")
    style_axis(heat_ax, grid_axis="none")

    top_ax.bar(times, hourly_counts.values, color="#8E6CBE", width=0.82, edgecolor="none")
    top_ax.set_ylabel("건수")
    top_ax.set_title("시간대별 on-demand migration event", loc="left")
    top_ax.tick_params(axis="x", labelbottom=False)

    cmap = ListedColormap([get_server_color(server) for server in range(case_data["instance"]["num_servers"])])
    cmap.set_bad(COLOR_INACTIVE)
    heat_ax.imshow(
        matrix,
        aspect="auto",
        interpolation="none",
        cmap=cmap,
        vmin=-0.5,
        vmax=case_data["instance"]["num_servers"] - 0.5,
    )
    heat_ax.set_title("On-demand VM 서버 배치 변화", loc="left")
    heat_ax.set_xlabel("시간")
    heat_ax.set_ylabel("On-demand VM")
    heat_ax.set_xticks(range(len(times)))
    heat_ax.set_xticklabels(times)
    heat_ax.set_yticks(range(len(od_ids)))
    heat_ax.set_yticklabels([f"{workload_id} · {migration_counts.get(workload_id, 0)}" for workload_id in od_ids], fontsize=9)
    heat_ax.set_xticks(np.arange(-0.5, len(times), 1), minor=True)
    heat_ax.set_yticks(np.arange(-0.5, len(od_ids), 1), minor=True)
    heat_ax.grid(which="minor", color=AX_BG, linewidth=0.55)
    heat_ax.tick_params(which="minor", bottom=False, left=False)

    for event in migration_events.itertuples(index=False):
        heat_ax.scatter(
            event.time - 0.5,
            od_index[event.workload_id],
            marker="D",
            s=28,
            color="white",
            edgecolor="#111111",
            linewidth=0.8,
            zorder=3,
        )

    legend_handles = [Patch(facecolor=get_server_color(server), label=f"S{server}") for server in range(case_data["instance"]["num_servers"])]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="white",
            markeredgecolor="#111111",
            markersize=6,
            linewidth=0,
            label="migration 발생 시점",
        )
    )
    heat_ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=min(5, len(legend_handles)),
        frameon=True,
        edgecolor=SPINE_COLOR,
    )

    fig.suptitle(
        f"색은 각 시간의 배치 서버를 뜻합니다. y축 숫자는 VM별 총 migration 횟수입니다. 총 migration 수: {int(len(migration_events))}",
        y=0.99,
        fontsize=13,
    )
    fig.subplots_adjust(top=0.88, bottom=0.16, left=0.07, right=0.98, hspace=0.10)
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def create_case_plots(instance_dir, results_dir, scenario=None):
    instance_dir = instance_dir.resolve()
    results_dir = results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    case_data = load_case_files(instance_dir, results_dir)
    scenario_name = choose_scenario(case_data, requested_scenario=scenario)
    spot_activity, spot_suspensions = build_spot_activity_tables(case_data)
    migration_events = build_on_demand_migration_events(case_data)
    suspended_scenario, clean_scenario = choose_spot_comparison_scenarios(case_data, spot_activity, scenario_name)

    gantt_path = results_dir / f"server_workload_gantt_{scenario_name}.png"
    diagnostics_path = results_dir / "scenario_diagnostics.png"
    spot_activity_path = results_dir / "spot_vm_activity_by_scenario.png"
    migration_path = results_dir / "on_demand_migration_timeline.png"
    suspended_gantt_path = results_dir / "server_workload_gantt_suspended.png"
    clean_gantt_path = results_dir / "server_workload_gantt_clean.png"

    create_gantt_figure(case_data, scenario_name, gantt_path, migration_events=migration_events)
    if suspended_scenario is not None:
        create_gantt_figure(case_data, suspended_scenario, suspended_gantt_path, migration_events=migration_events)
    if clean_scenario is not None:
        create_gantt_figure(case_data, clean_scenario, clean_gantt_path, migration_events=migration_events)
    create_diagnostics_figure(case_data, scenario_name, diagnostics_path)
    create_spot_activity_figure(case_data, spot_activity, scenario_name, spot_activity_path)
    create_on_demand_migration_figure(case_data, migration_events, migration_path)

    spot_activity.to_csv(results_dir / "spot_activity.csv", index=False)
    spot_suspensions.to_csv(results_dir / "spot_suspension_events.csv", index=False)
    migration_events.to_csv(results_dir / "on_demand_migration_events.csv", index=False)

    return {
        "scenario": scenario_name,
        "gantt_path": gantt_path,
        "diagnostics_path": diagnostics_path,
        "spot_activity_path": spot_activity_path,
        "migration_path": migration_path,
        "suspended_gantt_path": suspended_gantt_path if suspended_scenario is not None else None,
        "clean_gantt_path": clean_gantt_path if clean_scenario is not None else None,
        "suspended_scenario": suspended_scenario,
        "clean_scenario": clean_scenario,
        "spot_activity_csv": results_dir / "spot_activity.csv",
        "spot_suspension_events_csv": results_dir / "spot_suspension_events.csv",
        "migration_events_csv": results_dir / "on_demand_migration_events.csv",
    }


def create_ratio_comparison_figure(comparison_df, output_path):
    if comparison_df.empty:
        raise ValueError("comparison_df is empty.")

    comparison_df = comparison_df.copy().reset_index(drop=True)
    if "actual_migration_event_count" not in comparison_df.columns:
        comparison_df["actual_migration_event_count"] = comparison_df.get("migration_count", 0.0)
    if "total_gamma_activations" not in comparison_df.columns:
        comparison_df["total_gamma_activations"] = 0.0
    if "total_phi_activations" not in comparison_df.columns:
        comparison_df["total_phi_activations"] = 0.0
    if "peak_overbooking_ratio" not in comparison_df.columns:
        comparison_df["peak_overbooking_ratio"] = 0.0
    comparison_df["display_case"] = comparison_df["case"]
    comparison_df["mix_label"] = comparison_df.apply(
        lambda row: f"OD {int(row['on_demand'])} / SP {int(row['spot'])} / BJ {int(row['batch'])}",
        axis=1,
    )
    x = np.arange(len(comparison_df))

    fig, axes = plt.subplots(2, 2, figsize=(17.5, 11), facecolor=FIG_BG)
    for row in axes:
        for ax in row:
            style_axis(ax, grid_axis="y")

    mix_ax = axes[0, 0]
    mix_ax.barh(x, comparison_df["on_demand_share"], color=PALETTE_OD[0], edgecolor="none", label="On-demand")
    mix_ax.barh(
        x,
        comparison_df["spot_share"],
        left=comparison_df["on_demand_share"],
        color=COLOR_SPOT,
        edgecolor="none",
        label="Spot",
    )
    mix_ax.barh(
        x,
        comparison_df["batch_share"],
        left=comparison_df["on_demand_share"] + comparison_df["spot_share"],
        color=PALETTE_BJ[0],
        edgecolor="none",
        label="Batch",
    )
    mix_ax.set_xlim(0, 1.0)
    mix_ax.set_yticks(x)
    mix_ax.set_yticklabels(comparison_df["display_case"])
    mix_ax.invert_yaxis()
    mix_ax.set_xlabel("전체 VM 중 비율")
    mix_ax.set_title("Workload mix 구성", loc="left")
    for index, row in comparison_df.iterrows():
        mix_ax.text(
            1.01,
            index,
            f"{row['status_name']} · {row['mix_label']}",
            va="center",
            ha="left",
            fontsize=9,
            color=MUTED_TEXT,
        )
    mix_ax.legend(frameon=True, edgecolor=SPINE_COLOR, loc="lower right")

    frontier_ax = axes[0, 1]
    for row in comparison_df.itertuples(index=False):
        marker = "s" if row.status_name == "TIME_LIMIT" else "o"
        size = 120 + min(row.runtime_seconds, 600.0) * 0.6
        color = get_server_color(int(row.used_server_count))
        frontier_ax.scatter(
            row.peak_overbooking_ratio * 100.0,
            row.peak_realized_server_utilization * 100.0,
            s=size,
            color=color,
            marker=marker,
            edgecolors="white",
            linewidth=1.1,
            zorder=3,
        )
        frontier_ax.annotate(
            row.case,
            (row.peak_overbooking_ratio * 100.0, row.peak_realized_server_utilization * 100.0),
            xytext=(7, 5),
            textcoords="offset points",
            fontsize=9,
        )
    frontier_ax.axhline(100.0, color=SPINE_COLOR, linestyle="--", linewidth=1.0)
    frontier_ax.axvline(100.0, color=SPINE_COLOR, linestyle=":", linewidth=1.0)
    frontier_ax.set_xlabel("사용 서버 수")
    frontier_ax.set_ylabel("최대 단일 서버 이용률 (%)")
    frontier_ax.set_title("효율 프론티어: 서버 수 vs 병목 강도", loc="left")
    status_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#808080", markeredgecolor="white", markersize=9, label="OPTIMAL"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#808080", markeredgecolor="white", markersize=9, label="TIME_LIMIT"),
    ]
    frontier_ax.legend(handles=status_handles, frameon=True, edgecolor=SPINE_COLOR, loc="lower left")

    disruption_ax = axes[1, 0]
    width = 0.34
    disruption_ax.bar(x - width / 2, comparison_df["actual_migration_event_count"], width=width, color="#8E6CBE", label="실제 migration event")
    disruption_ax.bar(x + width / 2, comparison_df["total_gamma_activations"], width=width, color=COLOR_SPOT, label="spot suspension flag γ")
    disruption_ax.plot(x, comparison_df["total_phi_activations"], color=COLOR_ALERT, linewidth=2.0, marker="o", label="SLA flag φ")
    disruption_ax.set_xticks(x)
    disruption_ax.set_xticklabels(comparison_df["display_case"], rotation=20)
    disruption_ax.set_ylabel("count")
    disruption_ax.set_title("운영 불안정성: 이동과 recourse 사용", loc="left")
    disruption_ax.legend(frameon=True, edgecolor=SPINE_COLOR, loc="upper left")

    sla_ax = axes[1, 1]
    for row in comparison_df.itertuples(index=False):
        marker = "s" if row.status_name == "TIME_LIMIT" else "o"
        size = 110 + min(row.actual_migration_event_count, 200) * 1.2
        color = get_server_color(int(row.used_server_count))
        sla_ax.scatter(
            row.max_spot_suspension_probability,
            row.min_spot_completion_ratio,
            s=size,
            color=color,
            marker=marker,
            edgecolors="white",
            linewidth=1.0,
            zorder=3,
        )
        sla_ax.annotate(
            row.case,
            (row.max_spot_suspension_probability, row.min_spot_completion_ratio),
            xytext=(7, 5),
            textcoords="offset points",
            fontsize=9,
        )
    epsilon_sp = 0.20
    rho = 0.80
    sla_ax.axvline(epsilon_sp, color=SPINE_COLOR, linestyle=":", linewidth=1.1, label="epsilon_sp")
    sla_ax.axhline(rho, color=MUTED_TEXT, linestyle="--", linewidth=1.1, label="rho")
    sla_ax.set_xlim(-0.01, max(0.25, comparison_df["max_spot_suspension_probability"].max() + 0.05))
    sla_ax.set_ylim(min(0.75, comparison_df["min_spot_completion_ratio"].min() - 0.05), 1.02)
    sla_ax.set_xlabel("최대 spot 중단 확률")
    sla_ax.set_ylabel("최소 spot 완료 비율")
    sla_ax.set_title("Spot SLA 리스크 분포", loc="left")
    sla_ax.legend(frameon=True, edgecolor=SPINE_COLOR, loc="lower left")

    fig.suptitle(
        "비교 요약: workload mix가 서버 수, 병목, 이동, SLA에 미치는 영향",
        fontsize=15,
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    instance_dir = args.instance_dir.resolve()
    results_dir = args.results_dir.resolve() if args.results_dir else (EXPERIMENT_DIR / "results" / "runs" / instance_dir.name)
    outputs = create_case_plots(instance_dir, results_dir, scenario=args.scenario)
    print(f"Saved: {outputs['gantt_path']}")
    print(f"Saved: {outputs['diagnostics_path']}")
    print(f"Saved: {outputs['spot_activity_path']}")
    print(f"Saved: {outputs['migration_path']}")
    print(f"Saved: {outputs['spot_activity_csv']}")
    print(f"Saved: {outputs['spot_suspension_events_csv']}")
    print(f"Saved: {outputs['migration_events_csv']}")


if __name__ == "__main__":
    main()
