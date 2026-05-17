from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


START_RE = re.compile(
    r"Gurobi .* logging started (?P<timestamp>[A-Za-z]{3} [A-Za-z]{3} \d{1,2} \d{2}:\d{2}:\d{2} \d{4})"
)
PHASE1_RE = re.compile(r"Found phase-1 solution: relaxation (?P<value>[-+0-9.eE]+)")
HEURISTIC_RE = re.compile(r"Found heuristic solution: objective (?P<value>[-+0-9.eE]+)")
NOREL_ELAPSED_RE = re.compile(
    r"Elapsed time for NoRel heuristic: (?P<time>\d+)s \(best bound (?P<bound>[-+0-9.eE]+)\)"
)
ROOT_RE = re.compile(
    r"Root relaxation: objective (?P<objective>[-+0-9.eE]+), (?P<iterations>\d+) iterations, "
    r"(?P<seconds>[-+0-9.eE]+) seconds \((?P<work_units>[-+0-9.eE]+) work units\)"
)
INCUMBENT_RE = re.compile(
    r"^(?P<marker>[H*])\s+(?P<explored>\d+)\s+(?P<unexplored>\d+)\s+"
    r"(?P<incumbent>[-+0-9.eE]+)\s+(?P<best_bound>[-+0-9.eE]+)\s+"
    r"(?P<gap>[-+0-9.]+)%\s+(?P<it_per_node>-|[-+0-9.eE]+)\s+(?P<time>\d+)s$"
)
BEST_LINE_RE = re.compile(
    r"Best objective (?P<objective>[-+0-9.eE]+), best bound (?P<bound>[-+0-9.eE]+), gap (?P<gap>[-+0-9.eE]+)%"
)


BACKGROUND = "#FBF8F3"
INK = "#1E2430"
GRID = "#D7D2C8"
INCUMBENT_COLOR = "#B6503A"
BOUND_COLOR = "#2F5D8A"
GAP_COLOR = "#26867A"
NODE_COLOR = "#6B7280"
NOREL_COLOR = "#C68A2F"
PHASE1_COLOR = "#A38B6D"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="중단된 Gurobi 실행 로그에서 요약과 시각화를 복원합니다.")
    parser.add_argument("--manifest", required=True, help="run_manifest.json 경로")
    return parser.parse_args()


def parse_float(token: str | None) -> float | None:
    if token is None or token in {"-", ""}:
        return None
    try:
        return float(token)
    except ValueError:
        return None


def parse_progress_row(line: str) -> dict[str, Any] | None:
    tokens = line.split()
    if len(tokens) < 8:
        return None
    if not tokens[0].isdigit() or not tokens[1].isdigit() or not tokens[-1].endswith("s"):
        return None

    explored = int(tokens[0])
    unexplored = int(tokens[1])
    time_seconds = parse_float(tokens[-1][:-1])
    it_per_node = parse_float(tokens[-2])
    gap_percent = parse_float(tokens[-3].rstrip("%"))
    best_bound = parse_float(tokens[-4])
    incumbent = parse_float(tokens[-5])
    prefix = tokens[:-5]

    current_obj = prefix[2] if len(prefix) >= 3 else None
    depth = int(prefix[3]) if len(prefix) >= 4 and prefix[3].isdigit() else None
    intinf = int(prefix[4]) if len(prefix) >= 5 and prefix[4].isdigit() else None

    return {
        "explored_nodes": explored,
        "open_nodes": unexplored,
        "current_obj_token": current_obj,
        "depth": depth,
        "intinf": intinf,
        "incumbent": incumbent,
        "best_bound": best_bound,
        "gap_percent": gap_percent,
        "it_per_node": it_per_node,
        "time_seconds": time_seconds,
    }


def distribute_pending_events(
    pending: list[dict[str, Any]],
    start_time: float,
    end_time: float,
) -> None:
    if not pending:
        return
    if end_time < start_time:
        end_time = start_time
    interval = end_time - start_time
    event_count = len(pending)
    for index, event in enumerate(pending, start=1):
        if event_count == 1:
            fraction = 1.0
        else:
            fraction = index / event_count
        event["time_seconds"] = start_time + interval * fraction


def parse_log(log_path: Path) -> dict[str, Any]:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()

    start_timestamp: datetime | None = None
    in_norel = False
    phase = "phase1"
    last_norel_checkpoint_time = 0.0
    pending_phase1: list[dict[str, Any]] = []
    pending_phase2: list[dict[str, Any]] = []
    phase1_events: list[dict[str, Any]] = []
    incumbent_events: list[dict[str, Any]] = []
    norel_checkpoints: list[dict[str, Any]] = []
    progress_rows: list[dict[str, Any]] = []
    root_info: dict[str, Any] | None = None
    first_progress_time: float | None = None
    final_line_metrics: dict[str, Any] | None = None

    best_phase1_value: float | None = None
    best_norel_objective: float | None = None

    for line_number, line in enumerate(lines, start=1):
        if start_timestamp is None:
            start_match = START_RE.search(line)
            if start_match:
                start_timestamp = datetime.strptime(start_match.group("timestamp"), "%a %b %d %H:%M:%S %Y")

        if "Starting NoRel heuristic" in line:
            in_norel = True
            phase = "phase1"
            continue

        if "Transition to phase 2" in line:
            phase = "phase2"
            continue

        phase1_match = PHASE1_RE.search(line)
        if phase1_match:
            value = float(phase1_match.group("value"))
            best_phase1_value = value
            pending_phase1.append(
                {
                    "event_type": "phase1_relaxation",
                    "stage": "norel_phase1",
                    "line_number": line_number,
                    "relaxation_value": value,
                }
            )
            continue

        heuristic_match = HEURISTIC_RE.search(line)
        if heuristic_match:
            value = float(heuristic_match.group("value"))
            best_norel_objective = value
            pending_phase2.append(
                {
                    "event_type": "heuristic_solution",
                    "stage": "norel_phase2" if in_norel else "heuristic",
                    "line_number": line_number,
                    "objective": value,
                }
            )
            continue

        norel_elapsed_match = NOREL_ELAPSED_RE.search(line)
        if norel_elapsed_match:
            checkpoint_time = float(norel_elapsed_match.group("time"))
            best_bound = float(norel_elapsed_match.group("bound"))
            distribute_pending_events(pending_phase1, last_norel_checkpoint_time, checkpoint_time)
            distribute_pending_events(pending_phase2, last_norel_checkpoint_time, checkpoint_time)
            phase1_events.extend(pending_phase1)
            incumbent_events.extend(pending_phase2)
            pending_phase1 = []
            pending_phase2 = []
            norel_checkpoints.append(
                {
                    "line_number": line_number,
                    "time_seconds": checkpoint_time,
                    "stage": phase,
                    "best_bound": best_bound,
                    "best_phase1_relaxation": best_phase1_value,
                    "best_objective": best_norel_objective,
                }
            )
            last_norel_checkpoint_time = checkpoint_time
            continue

        root_match = ROOT_RE.search(line)
        if root_match:
            root_info = {
                "line_number": line_number,
                "root_objective": float(root_match.group("objective")),
                "root_iterations": int(root_match.group("iterations")),
                "root_seconds": float(root_match.group("seconds")),
                "root_work_units": float(root_match.group("work_units")),
            }
            continue

        incumbent_match = INCUMBENT_RE.match(line.strip())
        if incumbent_match:
            incumbent_events.append(
                {
                    "event_type": "incumbent_update",
                    "stage": "branch_search",
                    "line_number": line_number,
                    "marker": incumbent_match.group("marker"),
                    "explored_nodes": int(incumbent_match.group("explored")),
                    "open_nodes": int(incumbent_match.group("unexplored")),
                    "objective": float(incumbent_match.group("incumbent")),
                    "best_bound": float(incumbent_match.group("best_bound")),
                    "gap_percent": float(incumbent_match.group("gap")),
                    "it_per_node": parse_float(incumbent_match.group("it_per_node")),
                    "time_seconds": float(incumbent_match.group("time")),
                }
            )
            continue

        progress_row = parse_progress_row(line)
        if progress_row is not None:
            progress_row["line_number"] = line_number
            progress_rows.append(progress_row)
            if first_progress_time is None:
                first_progress_time = float(progress_row["time_seconds"])
            continue

        best_match = BEST_LINE_RE.search(line)
        if best_match:
            final_line_metrics = {
                "objective": float(best_match.group("objective")),
                "best_bound": float(best_match.group("bound")),
                "gap_percent": float(best_match.group("gap")),
                "line_number": line_number,
            }

    if first_progress_time is not None:
        distribute_pending_phase_end = first_progress_time
    elif root_info is not None:
        distribute_pending_phase_end = last_norel_checkpoint_time + root_info["root_seconds"]
    else:
        distribute_pending_phase_end = last_norel_checkpoint_time

    distribute_pending_events(pending_phase1, last_norel_checkpoint_time, distribute_pending_phase_end)
    distribute_pending_events(pending_phase2, last_norel_checkpoint_time, distribute_pending_phase_end)
    phase1_events.extend(pending_phase1)
    incumbent_events.extend(pending_phase2)

    incumbent_events.sort(key=lambda row: (float(row.get("time_seconds", math.inf)), row["line_number"]))
    phase1_events.sort(key=lambda row: (float(row.get("time_seconds", math.inf)), row["line_number"]))
    progress_rows.sort(key=lambda row: (float(row["time_seconds"]), row["line_number"]))
    norel_checkpoints.sort(key=lambda row: (float(row["time_seconds"]), row["line_number"]))

    combined_incumbent_candidates = list(incumbent_events)
    for row in progress_rows:
        combined_incumbent_candidates.append(
            {
                "event_type": "progress_incumbent",
                "stage": "branch_search",
                "line_number": row["line_number"],
                "marker": "P",
                "explored_nodes": row["explored_nodes"],
                "open_nodes": row["open_nodes"],
                "objective": row["incumbent"],
                "best_bound": row["best_bound"],
                "gap_percent": row["gap_percent"],
                "it_per_node": row["it_per_node"],
                "time_seconds": row["time_seconds"],
            }
        )

    combined_incumbent_candidates.sort(
        key=lambda row: (float(row.get("time_seconds", math.inf)), row["line_number"])
    )

    incumbent_events = []
    best_seen_objective = math.inf
    tolerance = 1e-6
    for event in combined_incumbent_candidates:
        objective = event.get("objective")
        if objective is None:
            continue
        if objective + tolerance < best_seen_objective:
            incumbent_events.append(event)
            best_seen_objective = objective

    final_progress = progress_rows[-1] if progress_rows else None
    final_incumbent_event = incumbent_events[-1] if incumbent_events else None

    if final_line_metrics is not None:
        objective_value = final_line_metrics["objective"]
        objective_bound = final_line_metrics["best_bound"]
        gap_percent = final_line_metrics["gap_percent"]
    else:
        objective_value = None
        objective_bound = None
        gap_percent = None
        if final_progress is not None:
            objective_value = final_progress["incumbent"]
            objective_bound = final_progress["best_bound"]
            gap_percent = final_progress["gap_percent"]
        elif final_incumbent_event is not None:
            objective_value = final_incumbent_event.get("objective")
            objective_bound = final_incumbent_event.get("best_bound")
            gap_percent = final_incumbent_event.get("gap_percent")

    if gap_percent is None and objective_value is not None and objective_bound is not None and objective_value != 0:
        gap_percent = 100.0 * (objective_value - objective_bound) / abs(objective_value)

    runtime_seconds = None
    if final_progress is not None:
        runtime_seconds = final_progress["time_seconds"]
    elif final_incumbent_event is not None:
        runtime_seconds = final_incumbent_event.get("time_seconds")
    elif norel_checkpoints:
        runtime_seconds = norel_checkpoints[-1]["time_seconds"]

    stop_timestamp = None
    if start_timestamp is not None and runtime_seconds is not None:
        stop_timestamp = start_timestamp + timedelta(seconds=float(runtime_seconds))

    norel_objectives = [row["objective"] for row in incumbent_events if row["stage"] == "norel_phase2"]
    branch_objectives = [row["objective"] for row in incumbent_events if row["stage"] == "branch_search"]

    return {
        "start_timestamp": start_timestamp,
        "stop_timestamp": stop_timestamp,
        "runtime_seconds": runtime_seconds,
        "root_info": root_info,
        "phase1_events": phase1_events,
        "incumbent_events": incumbent_events,
        "norel_checkpoints": norel_checkpoints,
        "progress_rows": progress_rows,
        "final_progress": final_progress,
        "objective_value": objective_value,
        "objective_bound": objective_bound,
        "gap_percent": gap_percent,
        "best_norel_objective": min(norel_objectives) if norel_objectives else None,
        "best_branch_objective": min(branch_objectives) if branch_objectives else None,
        "first_feasible_objective": incumbent_events[0]["objective"] if incumbent_events else None,
        "first_feasible_time_seconds": incumbent_events[0].get("time_seconds") if incumbent_events else None,
        "last_norel_checkpoint_time": norel_checkpoints[-1]["time_seconds"] if norel_checkpoints else None,
    }


def to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if not df.empty and "time_seconds" in df.columns:
        df["time_hours"] = df["time_seconds"] / 3600.0
    return df


def save_tables(parsed: dict[str, Any], manifest: dict[str, Any], analysis_dir: Path, run_dir: Path) -> None:
    phase1_df = to_dataframe(parsed["phase1_events"])
    incumbent_df = to_dataframe(parsed["incumbent_events"])
    checkpoint_df = to_dataframe(parsed["norel_checkpoints"])
    progress_df = to_dataframe(parsed["progress_rows"])

    phase1_df.to_csv(analysis_dir / "phase1_relaxation_events.csv", index=False)
    incumbent_df.to_csv(analysis_dir / "incumbent_updates.csv", index=False)
    checkpoint_df.to_csv(analysis_dir / "norel_checkpoints.csv", index=False)
    progress_df.to_csv(analysis_dir / "progress_trace.csv", index=False)

    final_progress = parsed["final_progress"] or {}
    root_info = parsed["root_info"] or {}

    summary = {
        "status_code": None,
        "status_name": "MANUAL_STOP_LOG_RECOVERY",
        "runtime_seconds": parsed["runtime_seconds"],
        "solution_count": int(len(parsed["incumbent_events"])),
        "has_solution": bool(parsed["objective_value"] is not None),
        "objective_value": parsed["objective_value"],
        "objective_bound": parsed["objective_bound"],
        "mip_gap": None if parsed["gap_percent"] is None else parsed["gap_percent"] / 100.0,
        "node_count": final_progress.get("explored_nodes"),
        "iter_count": None,
        "work_units": None,
        "instance_name": manifest["case_name"],
        "server_capacity": manifest.get("server_capacity"),
        "objective_type": "energy" if "objenergy" in manifest["case_name"] else None,
        "cut_profile": manifest["profile"],
        "cut_profile_category": manifest["profile"],
        "solver_cut_params": {},
        "cut_counts": {},
        "wall_clock_solve_seconds": parsed["runtime_seconds"],
        "solve_time_limit_seconds": manifest.get("time_limit"),
        "experiment_name": manifest["profile"],
        "experiment_notes": "manual stop recovered from solver.log",
        "branch_scheme": None,
        "branch_priorities": {},
        "callback_metadata": None,
        "callback_cut_counts": {},
        "used_server_count": None,
        "start_timestamp": None if parsed["start_timestamp"] is None else parsed["start_timestamp"].isoformat(),
        "stop_timestamp_estimated": None if parsed["stop_timestamp"] is None else parsed["stop_timestamp"].isoformat(),
        "results_dir": str(run_dir),
        "analysis_dir": str(analysis_dir),
        "first_feasible_objective": parsed["first_feasible_objective"],
        "first_feasible_time_seconds": parsed["first_feasible_time_seconds"],
        "best_norel_objective": parsed["best_norel_objective"],
        "best_branch_objective": parsed["best_branch_objective"],
        "last_norel_checkpoint_time": parsed["last_norel_checkpoint_time"],
        "open_nodes": final_progress.get("open_nodes"),
        "best_bound_at_stop": parsed["objective_bound"],
    }

    with (analysis_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    summary_row = {
        "profile": manifest["profile"],
        "instance_name": manifest["case_name"],
        "cut_profile": manifest["profile"],
        "status_name": summary["status_name"],
        "has_solution": summary["has_solution"],
        "runtime_seconds": summary["runtime_seconds"],
        "objective_value": summary["objective_value"],
        "objective_bound": summary["objective_bound"],
        "mip_gap": summary["mip_gap"],
        "node_count": summary["node_count"],
        "iter_count": summary["iter_count"],
        "work_units": summary["work_units"],
        "used_server_count": summary["used_server_count"],
        "branch_scheme": summary["branch_scheme"],
        "families": None,
        "schedule_mode": None,
        "branch_priorities": json.dumps(summary["branch_priorities"], ensure_ascii=False),
        "callback_metadata": summary["callback_metadata"],
        "callback_runtime_metadata": None,
        "callback_cut_counts": json.dumps(summary["callback_cut_counts"], ensure_ascii=False),
        "results_dir": summary["results_dir"],
        "notes": summary["experiment_notes"],
    }
    pd.DataFrame([summary_row]).to_csv(analysis_dir / "final_summary.csv", index=False)


def apply_axes_style(ax: plt.Axes) -> None:
    ax.set_facecolor(BACKGROUND)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.85)
    ax.tick_params(colors=INK, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#A79D91")
        spine.set_linewidth(0.9)


def plot_overview(parsed: dict[str, Any], manifest: dict[str, Any], analysis_dir: Path) -> None:
    progress_df = to_dataframe(parsed["progress_rows"])
    incumbent_df = to_dataframe(parsed["incumbent_events"])
    checkpoint_df = to_dataframe(parsed["norel_checkpoints"])

    if progress_df.empty:
        return

    no_rel_limit = manifest.get("no_rel_heur_time") or 0.0

    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), facecolor=BACKGROUND)
    ax_obj, ax_gap, ax_nodes, ax_inc = axes.flatten()

    for ax in axes.flatten():
        apply_axes_style(ax)
        if no_rel_limit > 0:
            ax.axvspan(0.0, no_rel_limit / 3600.0, color=NOREL_COLOR, alpha=0.10, linewidth=0)

    ax_obj.step(
        progress_df["time_hours"],
        progress_df["incumbent"],
        where="post",
        color=INCUMBENT_COLOR,
        linewidth=2.2,
        label="Incumbent",
    )
    ax_obj.step(
        progress_df["time_hours"],
        progress_df["best_bound"],
        where="post",
        color=BOUND_COLOR,
        linewidth=2.0,
        label="Best bound",
    )
    ax_obj.set_title("Objective Progress", color=INK, fontsize=12, pad=12)
    ax_obj.set_xlabel("경과 시간 (hours)", color=INK)
    ax_obj.set_ylabel("Objective", color=INK)
    ax_obj.legend(frameon=False, fontsize=9, loc="upper right")

    ax_gap.step(
        progress_df["time_hours"],
        progress_df["gap_percent"],
        where="post",
        color=GAP_COLOR,
        linewidth=2.0,
    )
    ax_gap.set_title("Gap Progress", color=INK, fontsize=12, pad=12)
    ax_gap.set_xlabel("경과 시간 (hours)", color=INK)
    ax_gap.set_ylabel("Gap (%)", color=INK)

    ax_nodes.plot(
        progress_df["time_hours"],
        progress_df["explored_nodes"],
        color=NODE_COLOR,
        linewidth=2.0,
        label="Explored nodes",
    )
    ax_nodes.plot(
        progress_df["time_hours"],
        progress_df["open_nodes"],
        color="#A38B6D",
        linewidth=1.8,
        label="Open nodes",
    )
    ax_nodes.set_yscale("log")
    ax_nodes.set_title("Search Tree Size", color=INK, fontsize=12, pad=12)
    ax_nodes.set_xlabel("경과 시간 (hours)", color=INK)
    ax_nodes.set_ylabel("노드 수 (log scale)", color=INK)
    ax_nodes.legend(frameon=False, fontsize=9, loc="upper left")

    branch_df = incumbent_df[incumbent_df["stage"] == "branch_search"].copy()
    norel_df = incumbent_df[incumbent_df["stage"] == "norel_phase2"].copy()

    if not norel_df.empty:
        ax_inc.step(
            norel_df["time_hours"],
            norel_df["objective"],
            where="post",
            color=NOREL_COLOR,
            linewidth=2.0,
            label="NoRel incumbent",
        )
    if not branch_df.empty:
        ax_inc.step(
            branch_df["time_hours"],
            branch_df["objective"],
            where="post",
            color=INCUMBENT_COLOR,
            linewidth=2.0,
            label="Branch incumbent",
        )
        ax_inc.scatter(
            branch_df["time_hours"],
            branch_df["objective"],
            color=INCUMBENT_COLOR,
            s=14,
            alpha=0.7,
            zorder=3,
        )
    ax_inc.set_title("Incumbent Improvement Events", color=INK, fontsize=12, pad=12)
    ax_inc.set_xlabel("경과 시간 (hours)", color=INK)
    ax_inc.set_ylabel("Objective", color=INK)
    ax_inc.legend(frameon=False, fontsize=9, loc="upper right")

    case_label = manifest["case_name"]
    fig.suptitle(
        f"Stopped Run Recovery: {case_label}\nprofile={manifest['profile']}, threads={manifest['threads']}, "
        f"cap={manifest['server_capacity']}, max_vcpu={manifest['max_vcpu']}",
        color=INK,
        fontsize=14,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(analysis_dir / "progress_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    if checkpoint_df.empty:
        return

    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), facecolor=BACKGROUND, sharex=True)
    ax_phase1, ax_norel = axes
    apply_axes_style(ax_phase1)
    apply_axes_style(ax_norel)

    phase1_df = to_dataframe(parsed["phase1_events"])
    if not phase1_df.empty:
        ax_phase1.plot(
            phase1_df["time_hours"],
            phase1_df["relaxation_value"],
            color=PHASE1_COLOR,
            linewidth=1.6,
            alpha=0.95,
        )
        ax_phase1.scatter(
            phase1_df["time_hours"],
            phase1_df["relaxation_value"],
            color=PHASE1_COLOR,
            s=12,
            alpha=0.75,
        )
    ax_phase1.set_title("NoRel Phase-1 Relaxation Score", color=INK, fontsize=12, pad=12)
    ax_phase1.set_ylabel("Relaxation score", color=INK)

    norel_df = incumbent_df[incumbent_df["stage"] == "norel_phase2"].copy()
    if not norel_df.empty:
        ax_norel.step(
            norel_df["time_hours"],
            norel_df["objective"],
            where="post",
            color=NOREL_COLOR,
            linewidth=2.0,
            label="NoRel incumbent",
        )
    ax_norel.plot(
        checkpoint_df["time_hours"],
        checkpoint_df["best_bound"],
        color=BOUND_COLOR,
        linewidth=1.8,
        linestyle="--",
        label="Best bound checkpoint",
    )
    ax_norel.set_title("NoRel Phase-2 Feasible Improvement", color=INK, fontsize=12, pad=12)
    ax_norel.set_xlabel("경과 시간 (hours)", color=INK)
    ax_norel.set_ylabel("Objective / bound", color=INK)
    ax_norel.legend(frameon=False, fontsize=9, loc="upper right")

    fig.suptitle("NoRel Heuristic Recovery", color=INK, fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(analysis_dir / "norel_progress.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(parsed: dict[str, Any], manifest: dict[str, Any], analysis_dir: Path, run_dir: Path) -> None:
    gap_percent = parsed["gap_percent"]
    start_text = "-" if parsed["start_timestamp"] is None else parsed["start_timestamp"].strftime("%Y-%m-%d %H:%M:%S")
    stop_text = "-" if parsed["stop_timestamp"] is None else parsed["stop_timestamp"].strftime("%Y-%m-%d %H:%M:%S")

    branch_gain = None
    if parsed["best_norel_objective"] is not None and parsed["objective_value"] is not None:
        branch_gain = parsed["best_norel_objective"] - parsed["objective_value"]

    lines = [
        "# 중단 실행 로그 복원 요약",
        "",
        "## 실행 정보",
        f"- 실행명: `{manifest['case_name']}`",
        f"- 프로파일: `{manifest['profile']}`",
        f"- 스레드 수: `{manifest['threads']}`",
        f"- 서버 용량: `{manifest['server_capacity']}`",
        f"- 최대 VM 코어 수: `{manifest['max_vcpu']}`",
        f"- NoRel heuristic 시간 제한: `{manifest['no_rel_heur_time']}`초",
        f"- 원본 로그: `{run_dir / 'solver.log'}`",
        "",
        "## 복원 결과",
        "- 상태: 수동 중단 후 `solver.log` 기반 복원",
        f"- 시작 시각: `{start_text}`",
        f"- 마지막 로그 시각 추정: `{stop_text}`",
        f"- 누적 실행 시간: `{parsed['runtime_seconds']:.0f}`초" if parsed["runtime_seconds"] is not None else "- 누적 실행 시간: `-`",
        f"- 최종 incumbent: `{parsed['objective_value']:.6f}`" if parsed["objective_value"] is not None else "- 최종 incumbent: `-`",
        f"- 최종 best bound: `{parsed['objective_bound']:.6f}`" if parsed["objective_bound"] is not None else "- 최종 best bound: `-`",
        f"- 최종 gap: `{gap_percent:.2f}%`" if gap_percent is not None else "- 최종 gap: `-`",
        f"- 마지막 explored nodes: `{parsed['final_progress']['explored_nodes']}`" if parsed["final_progress"] is not None else "- 마지막 explored nodes: `-`",
        f"- 마지막 open nodes: `{parsed['final_progress']['open_nodes']}`" if parsed["final_progress"] is not None else "- 마지막 open nodes: `-`",
        "",
        "## Heuristic / Root 요약",
        f"- 첫 feasible objective: `{parsed['first_feasible_objective']:.6f}`" if parsed["first_feasible_objective"] is not None else "- 첫 feasible objective: `-`",
        f"- 첫 feasible 시각 추정: `{parsed['first_feasible_time_seconds']:.1f}`초" if parsed["first_feasible_time_seconds"] is not None else "- 첫 feasible 시각 추정: `-`",
        f"- NoRel 종료 시점 최고 incumbent: `{parsed['best_norel_objective']:.6f}`" if parsed["best_norel_objective"] is not None else "- NoRel 종료 시점 최고 incumbent: `-`",
        f"- Branch-and-bound 이후 추가 개선: `{branch_gain:.6f}`" if branch_gain is not None else "- Branch-and-bound 이후 추가 개선: `-`",
    ]

    root_info = parsed["root_info"]
    if root_info is not None:
        lines.extend(
            [
                f"- Root relaxation objective: `{root_info['root_objective']:.6f}`",
                f"- Root relaxation iterations: `{root_info['root_iterations']}`",
                f"- Root relaxation time: `{root_info['root_seconds']:.2f}`초",
                f"- Root relaxation work units: `{root_info['root_work_units']:.2f}`",
            ]
        )

    lines.extend(
        [
            "",
            "## 생성 파일",
            "- `summary.json`: 복원된 핵심 지표",
            "- `final_summary.csv`: 실험 요약 1행 표",
            "- `progress_trace.csv`: branch-and-bound 진행 로그 표",
            "- `incumbent_updates.csv`: incumbent 갱신 이벤트",
            "- `norel_checkpoints.csv`: NoRel 체크포인트",
            "- `progress_overview.png`: 전체 탐색 진행 그래프",
            "- `norel_progress.png`: NoRel 진행 그래프",
            "",
            "## 한계",
            "- 이번 실행은 정상 종료 전 강제 중단되어 `summary.json`, placement CSV, scenario CSV가 생성되지 않았습니다.",
            "- 따라서 서버 배치 간트차트나 migration 시각화는 복원할 수 없고, solver 진행 이력 기반 시각화만 제공됩니다.",
        ]
    )

    (analysis_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    analysis_dir = Path(manifest["analysis_dir"])
    run_dir = Path(manifest["runs_group_dir"]) / manifest["case_name"] / manifest["profile"]
    log_path = run_dir / "solver.log"

    parsed = parse_log(log_path)
    save_tables(parsed, manifest, analysis_dir, run_dir)
    plot_overview(parsed, manifest, analysis_dir)
    write_report(parsed, manifest, analysis_dir, run_dir)


if __name__ == "__main__":
    main()
