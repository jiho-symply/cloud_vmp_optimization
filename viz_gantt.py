import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from collections import defaultdict

# -----------------------
# Config
# -----------------------
BASE_PATH = Path("results")
LATEST_DIR = max((p for p in BASE_PATH.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)

SERVER_PATH = LATEST_DIR / "server_timeseries.csv"
OD_PATH     = LATEST_DIR / "place_on_demand.csv"
SP_PATH     = LATEST_DIR / "place_spot.csv"
BJ_PATH     = LATEST_DIR / "place_batch.csv"
MIG_PATH    = LATEST_DIR / "migration.csv"
OUT_PATH    = LATEST_DIR / "server_workload_gantt.png"

for p in [SERVER_PATH, OD_PATH, SP_PATH, BJ_PATH, MIG_PATH]:
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")

# model capacity is normalized to 1.0
CAP = 1.0
EPS = 1e-12

# server row heights (visual)
H_USED = 2.0    # used servers ~2x height
H_UNUSED = 0.35 # unused servers compressed

# spacing
GAP = 0.25

# OFF styling
OFF_FACE = "black"
OFF_ALPHA = 0.10
OFF_HATCH = "///"

# outlines for all workloads
EDGE_COLOR = "black"
EDGE_LW = 0.45
ALPHA = 0.90

# -----------------------
# Hard-coded palettes (7 each)
# -----------------------
# On-demand: Blues (medium → slightly dark → light)
PALETTE_OD = [
    "#4292C6",
    "#6BAED6",
    "#2171B5",
    "#9ECAE1",
    "#08519C",
    "#BDD7E7",
    "#4F81BD",
]

# Spot: Yellows (clear golden tones, avoid brownish darks)
PALETTE_SP = [
    "#FFD34D",
    "#F2C400",
    "#FFDD66",
    "#E6B800",
    "#FFE680",
    "#C99700",
    "#D4A400",
]

# Batch: Greens (vivid mid-greens first)
PALETTE_BJ = [
    "#41AB5D",
    "#74C476",
    "#238B45",
    "#A1D99B",
    "#2CA25F",
    "#66C2A4",
    "#006D2C",
]


def pick_color(key_int: int, palette):
    return palette[int(key_int) % len(palette)]


# -----------------------
# Load data
# -----------------------
server_df = pd.read_csv(SERVER_PATH)
od_df = pd.read_csv(OD_PATH)
sp_df = pd.read_csv(SP_PATH)
bj_df = pd.read_csv(BJ_PATH)
mig_df = pd.read_csv(MIG_PATH)

I = sorted(server_df["i"].unique().tolist())
T = sorted(server_df["t"].unique().tolist())
T0, T_last = T[0], T[-1]

# which servers are ever used (u==1 at any time)
used_servers = set(
    server_df.groupby("i")["u"].max().loc[lambda s: s > 0].index.astype(int).tolist()
)

# row heights per server
row_h = {i: (H_USED if i in used_servers else H_UNUSED) for i in I}

# compute row base positions with variable heights
row_base = {}
cur = 0.0
for i in I:
    row_base[i] = cur
    cur += row_h[i] + GAP

y_max = cur

# -----------------------
# Helper: intervals for ON/OFF
# -----------------------
def on_intervals(u_series, times):
    """Return (start,end) intervals where u==1; end exclusive."""
    ints = []
    on = False
    s = None
    for k, t in enumerate(times):
        v = int(u_series[k])
        if (not on) and v == 1:
            on = True
            s = t
        elif on and v == 0:
            ints.append((s, t))
            on = False
            s = None
    if on:
        ints.append((s, times[-1] + 1))
    return ints


def complement_intervals(on_ints, start, end):
    off = []
    cur = start
    for (s, e) in sorted(on_ints):
        if cur < s:
            off.append((cur, s))
        cur = max(cur, e)
    if cur < end:
        off.append((cur, end))
    return off


# -----------------------
# Build lookup lists per (i,t)
#   od: (j, usage)
#   sp: (k, usage)  <-- same k same color (ignore n)
#   bj: (l, usage)  <-- same l same color (ignore n)
# -----------------------
od_by_it = defaultdict(list)
for r in od_df.itertuples(index=False):
    od_by_it[(int(r.i), int(r.t))].append((int(r.j), float(r.usage)))

sp_by_it = defaultdict(list)
for r in sp_df.itertuples(index=False):
    sp_by_it[(int(r.i), int(r.t))].append((int(r.k), float(r.usage)))

bj_by_it = defaultdict(list)
for r in bj_df.itertuples(index=False):
    bj_by_it[(int(r.i), int(r.t))].append((int(r.l), float(r.usage)))

# deterministic order
for key in od_by_it:
    od_by_it[key].sort(key=lambda x: x[0])
for key in sp_by_it:
    sp_by_it[key].sort(key=lambda x: x[0])
for key in bj_by_it:
    bj_by_it[key].sort(key=lambda x: x[0])

# -----------------------
# Plot
# -----------------------
fig_h = max(5.5, 0.75 * len(I) + 2.0)
fig, ax = plt.subplots(figsize=(16, fig_h))

# (j,t) -> (i, y_mid) for on-demand VM block midpoints
od_mid = {}

for i in I:
    base = row_base[i]
    H = row_h[i]

    # row borders
    ax.plot([T0, T_last + 1], [base, base], color="black", linewidth=0.6, alpha=0.35)
    ax.plot([T0, T_last + 1], [base + H, base + H], color="black", linewidth=0.6, alpha=0.35)

    # OFF shading from u
    u_series = server_df.loc[server_df["i"].eq(i)].sort_values("t")["u"].values.tolist()
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

    # workload stacks per (i,t)
    for t in T:
        # collect by type to keep stable stacking and to record on-demand midpoints correctly
        od_list = od_by_it.get((i, t), [])
        sp_list = sp_by_it.get((i, t), [])
        bj_list = bj_by_it.get((i, t), [])

        total = sum(u for (_, u) in od_list) + sum(u for (_, u) in sp_list) + sum(u for (_, u) in bj_list)
        if total <= EPS:
            continue

        # safety: if total slightly exceeds capacity, scale down to CAP (prevents overflow in plot)
        scale = 1.0
        if total > CAP + 1e-9:
            scale = CAP / total

        y0 = base

        # ---- On-demand first (record midpoint) ----
        for (j, usage) in od_list:
            if usage <= EPS:
                continue
            u_scaled = usage * scale
            h = u_scaled / CAP * H

            y_mid = y0 + 0.5 * h
            od_mid[(int(j), int(t))] = (int(i), float(y_mid))

            color = pick_color(j, PALETTE_OD)
            ax.broken_barh(
                [(t, 1.0)],
                (y0, h),
                facecolors=color,
                edgecolor=EDGE_COLOR,
                linewidth=EDGE_LW,
                alpha=ALPHA
            )
            y0 += h

        # ---- Spot ----
        for (k, usage) in sp_list:
            if usage <= EPS:
                continue
            u_scaled = usage * scale
            h = u_scaled / CAP * H
            color = pick_color(k, PALETTE_SP)
            ax.broken_barh(
                [(t, 1.0)],
                (y0, h),
                facecolors=color,
                edgecolor=EDGE_COLOR,
                linewidth=EDGE_LW,
                alpha=ALPHA
            )
            y0 += h

        # ---- Batch ----
        for (l, usage) in bj_list:
            if usage <= EPS:
                continue
            u_scaled = usage * scale
            h = u_scaled / CAP * H
            color = pick_color(l, PALETTE_BJ)
            ax.broken_barh(
                [(t, 1.0)],
                (y0, h),
                facecolors=color,
                edgecolor=EDGE_COLOR,
                linewidth=EDGE_LW,
                alpha=ALPHA
            )
            y0 += h

# -----------------------
# Migration arrows (orthogonal): workload center -> workload center
# -----------------------
for r in mig_df.itertuples(index=False):
    t = int(r.t)
    j = int(r.j)

    src = od_mid.get((j, t - 1), None)
    dst = od_mid.get((j, t), None)
    if (src is None) or (dst is None):
        continue

    _, y_fr = src
    _, y_to = dst

    x_fr = (t - 1) + 0.5   # center of (t-1) workload block
    x_to = t + 0.5         # center of t workload block

    ax.annotate(
        "",
        xy=(x_to, y_to),
        xytext=(x_fr, y_fr),
        arrowprops=dict(
            arrowstyle="->",
            linewidth=1.1,
            alpha=0.9,
            connectionstyle="angle3"
        )
    )

# -----------------------
# Axes + legend
# -----------------------
ax.set_xlim(T0, T_last + 1)
ax.set_ylim(-GAP, y_max)
ax.set_xlabel("Time (hour)")
ax.set_ylabel("Server")
ax.set_title("Server OFF (hatched) + Workloads (outlined) + Orthogonal Migrations")

centers = [row_base[i] + 0.5 * row_h[i] for i in I]
ax.set_yticks(centers)
ax.set_yticklabels([f"S{i}" for i in I])

legend_handles = [
    Patch(
        facecolor="black",
        alpha=OFF_ALPHA,
        hatch=OFF_HATCH,
        label="OFF (server off)"
    ),
    Patch(
        facecolor="#4292C6",
        edgecolor="black",
        linewidth=EDGE_LW,
        alpha=ALPHA,
        label="On-demand VM workload"
    ),
    Patch(
        facecolor="#FFD34D",
        edgecolor="black",
        linewidth=EDGE_LW,
        alpha=ALPHA,
        label="Spot VM workload"
    ),
    Patch(
        facecolor="#41AB5D",
        edgecolor="black",
        linewidth=EDGE_LW,
        alpha=ALPHA,
        label="Batch job workload"
    ),
]

leg = ax.legend(
    handles=legend_handles,
    loc="upper left",
    bbox_to_anchor=(0.01, 0.99),
    frameon=True,
    borderaxespad=0.0
)
leg.get_frame().set_alpha(0.85)

ax.grid(True, axis="x", alpha=0.2)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=220)
plt.close(fig)

print(f"Saved: {OUT_PATH}")
