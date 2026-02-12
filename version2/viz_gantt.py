import json
from pathlib import Path
from collections import defaultdict, OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase


class HandlerLegendArrow(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        y = ydescent + 0.5 * height
        p = FancyArrowPatch(
            (xdescent, y), (xdescent + width, y),
            arrowstyle="->",                 # gantt와 동일
            mutation_scale=12,               # gantt와 동일하게 맞춤
            lw=1.1,                          # gantt와 동일
            color="black",
            connectionstyle="arc3,rad=0.0"   # 직선 (굽지 않음)
        )
        p.set_transform(trans)
        return [p]

# ============================================================
# Config
# ============================================================
BASE_PATH = Path("results")

OUT_GANTT_NAME = "server_workload_gantt.png"
OUT_OVB_NAME   = "server_requested_stack.png"   # keep filename (per your existing pipeline)

EPS = 1e-12

# row heights (match both figures)
H_USED = 2.0
H_UNUSED = 0.35
GAP = 0.25

# OFF styling
OFF_FACE = "black"
OFF_ALPHA = 0.10
OFF_HATCH = "///"

# workload styling
EDGE_COLOR = "black"
EDGE_LW = 0.45
ALPHA = 0.90

# reference line in overbooking figure
REF100_COLOR = "red"

# palettes
PALETTE_OD = ["#4292C6","#6BAED6","#2171B5","#9ECAE1","#08519C","#BDD7E7","#4F81BD"]
PALETTE_SP = ["#FFD34D","#F2C400","#FFDD66","#E6B800","#FFE680","#C99700","#D4A400"]
PALETTE_BJ = ["#41AB5D","#74C476","#238B45","#A1D99B","#2CA25F","#66C2A4","#006D2C"]


def pick_color(key_int: int, palette):
    return palette[int(key_int) % len(palette)]


# ============================================================
# IO helpers
# ============================================================
def find_latest_result_dir(base_path: Path) -> Path:
    if not base_path.exists():
        raise FileNotFoundError(f"Missing base results directory: {base_path}")
    cand = [p for p in base_path.iterdir() if p.is_dir() and (p / "result.json").exists()]
    if not cand:
        raise FileNotFoundError(f"No subdirectory under {base_path} contains result.json")
    return max(cand, key=lambda p: p.stat().st_mtime)


def list_server_ids(servers_dict):
    out = []
    for k in servers_dict.keys():
        try:
            out.append(int(k))
        except Exception:
            pass
    return sorted(out)


def first_timeseries_server_id(servers_dict):
    for k, v in servers_dict.items():
        if isinstance(v, dict) and isinstance(v.get("timeseries", None), list) and len(v["timeseries"]) > 0:
            try:
                return int(k)
            except Exception:
                continue
    return None


# ============================================================
# Interval helpers (ON/OFF shading)
# ============================================================
def on_intervals(u_series, times):
    ints, on, s = [], False, None
    for idx, t in enumerate(times):
        v = int(u_series[idx])
        if (not on) and v == 1:
            on, s = True, t
        elif on and v == 0:
            ints.append((s, t))
            on, s = False, None
    if on:
        ints.append((s, times[-1] + 1))
    return ints


def complement_intervals(on_ints, start, end):
    off, cur = [], start
    for (s, e) in sorted(on_ints):
        if cur < s:
            off.append((cur, s))
        cur = max(cur, e)
    if cur < end:
        off.append((cur, end))
    return off


# ============================================================
# Parse result.json into minimal reusable structures
# ============================================================
def parse_result(res):
    servers_root = res.get("servers", {})
    servers = servers_root.get("servers", {})
    if not isinstance(servers, dict) or len(servers) == 0:
        raise ValueError("Invalid result.json: res['servers']['servers'] missing or empty")

    cap = float(servers_root.get("capacity", 1.0))

    I_all = list_server_ids(servers)
    sid0 = first_timeseries_server_id(servers)
    if sid0 is None:
        raise RuntimeError("No server has non-empty timeseries in result.json")

    T = sorted(int(r["t"]) for r in servers[str(sid0)]["timeseries"])
    if not T:
        raise RuntimeError("Empty time index T detected from timeseries")
    T0, T_last = T[0], T[-1]

    # used flag (robust)
    used = set()
    for i in I_all:
        sdata = servers.get(str(i), {})
        if bool(sdata.get("used_flag", False)) and isinstance(sdata.get("timeseries", None), list):
            used.add(i)

    # plot order: bottom=used (ascending), top=unused (ascending)
    I_plot = sorted(used) + sorted([i for i in I_all if i not in used])

    # row heights + bases
    row_h = {i: (H_USED if i in used else H_UNUSED) for i in I_plot}
    row_base, cur = {}, 0.0
    for i in I_plot:
        row_base[i] = cur
        cur += row_h[i] + GAP
    y_max = cur

    # per-(i,t): u, overbooking, usage contrib lists, requested entity sums
    u_by_it, over_by_it = {}, {}
    od_by_it, sp_by_it, bj_by_it = defaultdict(list), defaultdict(list), defaultdict(list)
    req_by_it = defaultdict(lambda: defaultdict(float))  # (i,t)->{entity_key: requested_sum}

    for i in I_all:
        ts = servers.get(str(i), {}).get("timeseries", [])
        if not isinstance(ts, list) or len(ts) == 0:
            # still ensure u_by_it defaults later
            continue

        for r in ts:
            t = int(r["t"])
            u = int(r.get("u", 0))
            u_by_it[(i, t)] = u
            over_by_it[(i, t)] = float(r.get("overbooking_ratio", 0.0))

            contrib = r.get("contrib", [])
            if not isinstance(contrib, list) or len(contrib) == 0:
                continue

            # NOTE: usage stacks should not appear when server is OFF
            if u == 0:
                continue

            for c in contrib:
                typ = c.get("type", None)
                if typ not in ("on_demand", "spot", "batch"):
                    continue
                try:
                    vid = int(c.get("id"))
                except Exception:
                    continue

                usage = c.get("usage", None)
                if usage is not None:
                    try:
                        usage_f = float(usage)
                    except Exception:
                        usage_f = None
                else:
                    usage_f = None

                if usage_f is not None:
                    if typ == "on_demand":
                        od_by_it[(i, t)].append((vid, usage_f))
                    elif typ == "spot":
                        sp_by_it[(i, t)].append((vid, usage_f))
                    else:
                        bj_by_it[(i, t)].append((vid, usage_f))

                req = c.get("requested", None)
                if req is not None:
                    try:
                        req_f = float(req)
                    except Exception:
                        req_f = None
                    if req_f is not None and req_f > EPS:
                        # entity key keeps palette mapping consistent with Fig1 legend
                        if typ == "on_demand":
                            ek = f"od_{vid}"
                        elif typ == "spot":
                            ek = f"sp_{vid}"
                        else:
                            ek = f"bj_{vid}"
                        req_by_it[(i, t)][ek] += req_f

    for key in od_by_it:
        od_by_it[key].sort(key=lambda x: x[0])
    for key in sp_by_it:
        sp_by_it[key].sort(key=lambda x: x[0])
    for key in bj_by_it:
        bj_by_it[key].sort(key=lambda x: x[0])

    # migrations
    mig_rows = []
    od_vms = res.get("on_demand", {}).get("VMs", {})
    if isinstance(od_vms, dict):
        for j_str, jdata in od_vms.items():
            try:
                j = int(j_str)
            except Exception:
                continue
            if not isinstance(jdata, dict):
                continue
            for ev in jdata.get("migrations", []):
                try:
                    mig_rows.append({"j": j, "t": int(ev["t"]), "from": int(ev["from"]), "to": int(ev["to"])})
                except Exception:
                    continue

    return {
        "servers": servers,
        "cap": cap,
        "I_all": I_all,
        "I_plot": I_plot,
        "T": T,
        "T0": T0,
        "T_last": T_last,
        "used": used,
        "row_h": row_h,
        "row_base": row_base,
        "y_max": y_max,
        "u_by_it": u_by_it,
        "over_by_it": over_by_it,
        "od_by_it": od_by_it,
        "sp_by_it": sp_by_it,
        "bj_by_it": bj_by_it,
        "req_by_it": req_by_it,
        "mig_rows": mig_rows,
    }


def entity_color(ek: str):
    typ, vid = ek.split("_", 1)
    vid = int(vid)
    if typ == "od":
        return pick_color(vid, PALETTE_OD)
    if typ == "sp":
        return pick_color(vid, PALETTE_SP)
    return pick_color(vid, PALETTE_BJ)


# ============================================================
# Plot 1: Server VM/Job Workload Gantt
# ============================================================
def plot_workload_gantt(ctx, out_path: Path):
    cap = ctx["cap"]
    I_plot = ctx["I_plot"]
    T = ctx["T"]
    T0, T_last = ctx["T0"], ctx["T_last"]
    row_h, row_base, y_max = ctx["row_h"], ctx["row_base"], ctx["y_max"]
    used = ctx["used"]
    u_by_it = ctx["u_by_it"]
    od_by_it, sp_by_it, bj_by_it = ctx["od_by_it"], ctx["sp_by_it"], ctx["bj_by_it"]
    mig_rows = ctx["mig_rows"]

    fig_h = max(5.0, 0.70 * len(I_plot) + 1.8)
    fig, ax = plt.subplots(figsize=(16, fig_h))

    # (j,t) -> (i, y_mid) for arrow endpoints
    od_mid = {}

    for i in I_plot:
        base = row_base[i]
        H = row_h[i]

        # row borders
        ax.plot([T0, T_last + 1], [base, base], color="black", linewidth=0.6, alpha=0.35)
        ax.plot([T0, T_last + 1], [base + H, base + H], color="black", linewidth=0.6, alpha=0.35)

        # OFF shading from u
        u_series = [u_by_it.get((i, t), 0) for t in T]
        on_ints = on_intervals(u_series, T)
        off_ints = complement_intervals(on_ints, T0, T_last + 1)
        for (s, e) in off_ints:
            ax.broken_barh(
                [(s, e - s)],
                (base, H),
                facecolors=OFF_FACE,
                alpha=OFF_ALPHA,
                edgecolor=OFF_FACE,
                linewidth=0.0,
                hatch=OFF_HATCH
            )

        # stacks per t
        for t in T:
            if u_by_it.get((i, t), 0) == 0:
                continue

            od_list = od_by_it.get((i, t), [])
            sp_list = sp_by_it.get((i, t), [])
            bj_list = bj_by_it.get((i, t), [])

            total = sum(v for _, v in od_list) + sum(v for _, v in sp_list) + sum(v for _, v in bj_list)
            if total <= EPS:
                continue

            scale = 1.0
            if total > cap + 1e-9 and cap > 0:
                scale = cap / total

            y0 = base

            # OD
            for (j, usage) in od_list:
                if usage <= EPS:
                    continue
                h = (usage * scale / cap) * H if cap > 0 else 0.0
                if h <= EPS:
                    continue

                od_mid[(int(j), int(t))] = (int(i), float(y0 + 0.5 * h))
                ax.broken_barh(
                    [(t, 1.0)],
                    (y0, h),
                    facecolors=pick_color(j, PALETTE_OD),
                    edgecolor=EDGE_COLOR,
                    linewidth=EDGE_LW,
                    alpha=ALPHA
                )
                y0 += h

            # SP
            for (k, usage) in sp_list:
                if usage <= EPS:
                    continue
                h = (usage * scale / cap) * H if cap > 0 else 0.0
                if h <= EPS:
                    continue
                ax.broken_barh(
                    [(t, 1.0)],
                    (y0, h),
                    facecolors=pick_color(k, PALETTE_SP),
                    edgecolor=EDGE_COLOR,
                    linewidth=EDGE_LW,
                    alpha=ALPHA
                )
                y0 += h

            # BJ
            for (l, usage) in bj_list:
                if usage <= EPS:
                    continue
                h = (usage * scale / cap) * H if cap > 0 else 0.0
                if h <= EPS:
                    continue
                ax.broken_barh(
                    [(t, 1.0)],
                    (y0, h),
                    facecolors=pick_color(l, PALETTE_BJ),
                    edgecolor=EDGE_COLOR,
                    linewidth=EDGE_LW,
                    alpha=ALPHA
                )
                y0 += h

    # migration arrows
    arrow_kw = dict(
        arrowstyle="->",
        linewidth=1.1,
        alpha=0.9,
        connectionstyle="angle3,angleA=90,angleB=0"
    )
    for r in mig_rows:
        t, j = int(r["t"]), int(r["j"])
        src = od_mid.get((j, t - 1), None)
        dst = od_mid.get((j, t), None)
        if src is None or dst is None:
            continue
        _, y_fr = src
        _, y_to = dst
        x_fr = (t - 1) + 0.5
        x_to = t + 0.5
        ax.annotate("", xy=(x_to, y_to), xytext=(x_fr, y_fr), arrowprops=arrow_kw)

    # axes
    ax.set_xlim(T0, T_last + 1)
    ax.set_ylim(-GAP, y_max)
    ax.set_xlabel("Time (hour)")
    ax.set_ylabel("Server")
    ax.set_title("Server VM/Job Workload Gantt")

    centers = [row_base[i] + 0.5 * row_h[i] for i in I_plot]
    ax.set_yticks(centers)
    ax.set_yticklabels([f"S{i}" for i in I_plot])

    # Right-side Core Usage axis (single centered label)
    axr = ax.twinx()
    axr.set_ylim(ax.get_ylim())  # share geometry

    ticks, labels = [], []
    for i in I_plot:
        if i not in used:
            continue
        base = row_base[i]
        H = row_h[i]
        for pct in (0, 50, 100):
            y = base + (pct / 100.0) * H
            ticks.append(y)
            labels.append(str(pct))

    axr.set_yticks(ticks)
    axr.set_yticklabels(labels)
    axr.set_ylabel("Core Usage (%)")

    # legend
    legend_handles = [
        Patch(facecolor="black", alpha=OFF_ALPHA, hatch=OFF_HATCH, label="OFF (server off)"),
        Patch(facecolor=PALETTE_OD[0], edgecolor="black", linewidth=EDGE_LW, alpha=ALPHA, label="On-demand VM workload"),
        Patch(facecolor=PALETTE_SP[0], edgecolor="black", linewidth=EDGE_LW, alpha=ALPHA, label="Spot VM workload"),
        Patch(facecolor=PALETTE_BJ[0], edgecolor="black", linewidth=EDGE_LW, alpha=ALPHA, label="Batch job workload"),
        FancyArrowPatch((0, 0), (1, 0), arrowstyle="->"),
    ]
    legend_labels = [
        "OFF (server off)",
        "On-demand VM workload",
        "Spot VM workload",
        "Batch job workload",
        "Migration",
    ]
    leg = ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        borderaxespad=0.0,
        handlelength=2.0,
        handler_map={FancyArrowPatch: HandlerLegendArrow()},  # 핵심
    )
    leg.get_frame().set_alpha(0.85)

    ax.grid(True, axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


# ============================================================
# Plot 2: Server Overbooking Gantt (requested stacks)
# ============================================================
def plot_overbooking_gantt(ctx, out_path: Path):
    cap = ctx["cap"]
    I_plot = ctx["I_plot"]
    T = ctx["T"]
    T0, T_last = ctx["T0"], ctx["T_last"]
    row_h, row_base, y_max = ctx["row_h"], ctx["row_base"], ctx["y_max"]
    used = ctx["used"]
    u_by_it = ctx["u_by_it"]
    req_by_it = ctx["req_by_it"]

    # server-wise max overbooking ratio (requested_sum / CAP), only when server is ON
    max_ratio_by_i = {}
    for i in I_plot:
        mr = 1.0
        for t in T:
            if u_by_it.get((i, t), 0) == 0:
                continue
            ent_map = req_by_it.get((i, t), {})
            tot = float(sum(ent_map.values())) if ent_map else 0.0
            if cap > 0:
                mr = max(mr, tot / cap)
        max_ratio_by_i[i] = mr

    # global entity list for consistent stacking (OD -> SP -> BJ, by id)
    all_entities = set()
    for i in I_plot:
        for t in T:
            all_entities.update(req_by_it.get((i, t), {}).keys())

    def ent_sort(k):
        typ, vid = k.split("_", 1)
        typ_ord = {"od": 0, "sp": 1, "bj": 2}[typ]
        return (typ_ord, int(vid))

    entities = sorted(all_entities, key=ent_sort)

    fig_h = max(5.0, 0.70 * len(I_plot) + 1.8)
    fig, ax = plt.subplots(figsize=(16, fig_h))

    # draw each server row (same y layout as fig1)
    for i in I_plot:
        base = row_base[i]
        H = row_h[i]

        # row borders
        ax.plot([T0, T_last + 1], [base, base], color="black", linewidth=0.6, alpha=0.35)
        ax.plot([T0, T_last + 1], [base + H, base + H], color="black", linewidth=0.6, alpha=0.35)

        # OFF shading
        u_series = [u_by_it.get((i, t), 0) for t in T]
        on_ints = on_intervals(u_series, T)
        off_ints = complement_intervals(on_ints, T0, T_last + 1)
        for (s, e) in off_ints:
            ax.broken_barh(
                [(s, e - s)],
                (base, H),
                facecolors=OFF_FACE,
                alpha=OFF_ALPHA,
                edgecolor=OFF_FACE,
                linewidth=0.0,
                hatch=OFF_HATCH
            )

        # requested stacks per t (normalized into row height by CAP)
        for t in T:
            if u_by_it.get((i, t), 0) == 0:
                continue

            ent_map = req_by_it.get((i, t), {})
            if not ent_map:
                continue

            total_req = float(sum(ent_map.values()))
            if total_req <= EPS:
                continue

            mr = float(max_ratio_by_i.get(i, 1.0))
            if mr <= 0:
                mr = 1.0

            y0 = base
            for ek in entities:
                v = float(ent_map.get(ek, 0.0))
                if v <= EPS:
                    continue
                # ratio = v / CAP, then map by (ratio / max_ratio) into row height H
                h = ((v / cap) / mr) * H if cap > 0 else 0.0
                if h <= EPS:
                    continue

                ax.broken_barh(
                    [(t, 1.0)],
                    (y0, h),
                    facecolors=entity_color(ek),
                    edgecolor="black",
                    linewidth=0.25,
                    alpha=0.92
                )
                y0 += h

    # axes
    ax.set_xlim(T0, T_last + 1)
    ax.set_ylim(-GAP, y_max)
    ax.set_xlabel("Time (hour)")
    ax.set_ylabel("Server")
    ax.set_title("Server Overbooking Gantt")

    centers = [row_base[i] + 0.5 * row_h[i] for i in I_plot]
    ax.set_yticks(centers)
    ax.set_yticklabels([f"S{i}" for i in I_plot])

    # Right-side overbooking ratio axis (single centered label)
    axr = ax.twinx()
    axr.set_ylim(ax.get_ylim())  # share geometry

    ticks, labels = [], []
    for i in I_plot:
        if i not in used:
            continue
        base = row_base[i]
        H = row_h[i]
        mr = float(max_ratio_by_i.get(i, 1.0))
        if mr <= 0:
            mr = 1.0

        # ticks: 0,100,200,... up to ceil(max_ratio*100)
        max_pct = int(np.ceil(mr * 100.0 / 100.0) * 100.0)
        for pct in range(0, max_pct + 1, 100):
            y = base + ((pct / 100.0) / mr) * H
            if y <= base + H + 1e-9:
                ticks.append(y)
                labels.append(str(pct))

        # 100% reference line inside this server row
        y100 = base + (1.0 / mr) * H
        ax.hlines(
            y100, xmin=T0, xmax=T_last + 1,
            colors=REF100_COLOR, linestyles=":", linewidth=0.9, alpha=0.9
        )

    axr.set_yticks(ticks if ticks else [])
    axr.set_yticklabels(labels if labels else [])
    axr.set_ylabel("Overbooking Ratio (%)")


    # legends: unify with fig1 excluding migration + add 100% reference line
    legend_handles = [
        Patch(facecolor="black", alpha=OFF_ALPHA, hatch=OFF_HATCH, label="OFF (server off)"),
        Patch(facecolor=PALETTE_OD[0], edgecolor="black", linewidth=EDGE_LW, alpha=ALPHA, label="On-demand VM workload"),
        Patch(facecolor=PALETTE_SP[0], edgecolor="black", linewidth=EDGE_LW, alpha=ALPHA, label="Spot VM workload"),
        Patch(facecolor=PALETTE_BJ[0], edgecolor="black", linewidth=EDGE_LW, alpha=ALPHA, label="Batch job workload"),
        Line2D([0], [0], color=REF100_COLOR, linestyle=":", linewidth=0.9, label="100% (physical capacity)"),
    ]
    leg = ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        borderaxespad=0.0,
    )
    leg.get_frame().set_alpha(0.85)

    ax.grid(True, axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close(fig)


# ============================================================
# Main
# ============================================================
LATEST_DIR = find_latest_result_dir(BASE_PATH)
RESULT_PATH = LATEST_DIR / "result.json"
OUT_GANTT = LATEST_DIR / OUT_GANTT_NAME
OUT_OVB  = LATEST_DIR / OUT_OVB_NAME

with open(RESULT_PATH, "r") as f:
    res = json.load(f)

ctx = parse_result(res)

plot_workload_gantt(ctx, OUT_GANTT)
print(f"Saved: {OUT_GANTT}")

plot_overbooking_gantt(ctx, OUT_OVB)
print(f"Saved: {OUT_OVB}")
