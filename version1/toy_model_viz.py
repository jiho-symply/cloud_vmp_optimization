import os, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.patches import Patch

# -----------------------
# Config
# -----------------------
BASE_PATH = Path("results")
LATEST_DIR = max(
    (p for p in BASE_PATH.iterdir() if p.is_dir()),
    key=lambda p: p.stat().st_mtime
)
VAR_PATH = LATEST_DIR / "variables.csv"
OUT_PATH = LATEST_DIR / "server_stack_gantt.png"

pCPU = 24
eps_bin = 0.5

COLOR_OD    = "#1f77b4"
COLOR_BATCH = "#ff7f0e"
COLOR_BAND  = "#000000"

# OFF styling (requested)
OFF_FACE = "black"
OFF_ALPHA = 0.1
OFF_HATCH = "///"

if not os.path.exists(VAR_PATH):
    raise FileNotFoundError(f"Missing: {VAR_PATH}")

df = pd.read_csv(VAR_PATH)
df = df[df["value"].notna()].copy()

pat = re.compile(r"^(\w+)\[(.*)\]$")

def parse_var(name: str):
    m = pat.match(name)
    if not m:
        return None, None
    vname = m.group(1)
    idx = tuple(int(x) for x in m.group(2).split(","))
    return vname, idx

u = defaultdict(float)
L = defaultdict(float)
w = defaultdict(float)
x = defaultdict(float)
q = defaultdict(float)

for name, val in zip(df["var_name"], df["value"]):
    vname, idx = parse_var(name)
    if vname is None:
        continue
    val = float(val)
    if vname == "u": u[idx] = val
    elif vname == "L": L[idx] = val
    elif vname == "w": w[idx] = val
    elif vname == "x": x[idx] = val
    elif vname == "q": q[idx] = val

I = sorted({i for (i,t) in u.keys()})
T = sorted({t for (i,t) in u.keys()})

# -----------------------
# Batch + On-demand decomposition
# -----------------------
batch_it = defaultdict(float)
for (i, req_t, run_t), val in w.items():
    if val > 1e-10:
        batch_it[(i, run_t)] += val

od_it = defaultdict(float)
for i in I:
    for t in T:
        od_it[(i,t)] = max(0.0, L.get((i,t), 0.0) - batch_it.get((i,t), 0.0))

# -----------------------
# Migration events
# -----------------------
vm_server = {}
for (i,j,t), val in x.items():
    if val > eps_bin:
        vm_server[(j,t)] = i

mig_events = []
for (j,t), val in q.items():
    if val > eps_bin:
        i_to = vm_server.get((j,t), None)
        i_fr = vm_server.get((j,t-1), None)
        if (i_fr is not None) and (i_to is not None) and (i_fr != i_to):
            mig_events.append((t, j, i_fr, i_to))

# -----------------------
# Helpers: intervals
# -----------------------
def on_intervals(u_series, times):
    """Return (start,end) intervals where u==1; end exclusive."""
    ints = []
    on = False
    s = None
    for k, t in enumerate(times):
        v = 1 if u_series[k] > eps_bin else 0
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
    """Given ON intervals within [start,end), return OFF intervals (complement)."""
    off = []
    cur = start
    for (s,e) in sorted(on_ints):
        if cur < s:
            off.append((cur, s))
        cur = max(cur, e)
    if cur < end:
        off.append((cur, end))
    return off

# Build bin edges so the last t draws over [T_last, T_last+1)
T0 = T[0]
T_last = T[-1]
edges = np.array(T + [T_last + 1], dtype=float)  # len = len(T)+1

# -----------------------
# Plot
# -----------------------
gap = max(2.0, 0.15 * pCPU)
row_base = {i: k*(pCPU + gap) for k, i in enumerate(I)}

fig_h = max(4.0, 0.55*len(I) + 2.0)
fig, ax = plt.subplots(figsize=(16, fig_h))

for i in I:
    base = row_base[i]

    # band borders
    ax.plot([T0, T_last+1], [base, base], color=COLOR_BAND, linewidth=0.5, alpha=0.5)
    ax.plot([T0, T_last+1], [base+pCPU, base+pCPU], color=COLOR_BAND, linewidth=0.5, alpha=0.5)

    # ON/OFF intervals from u
    u_series = [u.get((i,t), 0.0) for t in T]
    on_ints = on_intervals(u_series, T)
    off_ints = complement_intervals(on_ints, T0, T_last+1)

    # OFF shading + hatch (requested)
    # Draw OFF first so loads appear on top
    for (s,e) in off_ints:
        ax.broken_barh(
            [(s, e-s)],
            (base, pCPU),
            facecolors=OFF_FACE,
            alpha=OFF_ALPHA,
            edgecolor=OFF_FACE,
            linewidth=0.0,
            hatch=OFF_HATCH
        )

    # Loads (do NOT mask by u anymore; OFF region is visible anyway)
    od = np.array([od_it[(i,t)] for t in T], dtype=float)
    bt = np.array([batch_it.get((i,t),0.0) for t in T], dtype=float)

    # step="post": y needs len(edges)
    od_ext = np.r_[od, od[-1] if len(od) else 0.0]
    bt_ext = np.r_[bt, bt[-1] if len(bt) else 0.0]

    y0 = base
    ax.fill_between(edges, y0, y0 + od_ext, step="post",
                    facecolor=COLOR_OD, alpha=0.55, linewidth=0.0)
    ax.fill_between(edges, y0 + od_ext, y0 + od_ext + bt_ext, step="post",
                    facecolor=COLOR_BATCH, alpha=0.55, linewidth=0.0)

# migration arrows (diagonal t-1 -> t)
for (t, j, i_fr, i_to) in mig_events:
    y_fr = row_base[i_fr] + 0.85*pCPU
    y_to = row_base[i_to] + 0.85*pCPU
    ax.annotate(
        "",
        xy=(t + 0.05, y_to),
        xytext=(t - 0.95, y_fr),
        arrowprops=dict(arrowstyle="->", linewidth=1.1, alpha=0.9)
    )

# axes formatting
ax.set_xlim(T0, T_last + 1)
ax.set_ylim(-gap, row_base[I[-1]] + pCPU + gap)
ax.set_xlabel("Time (hour)")
ax.set_title("Server OFF (hatched) + Load + Migration")

centers = [row_base[i] + 0.5*pCPU for i in I]
ax.set_yticks(centers)
ax.set_yticklabels([f"S{i}" for i in I])
ax.set_ylabel("Server")

legend_handles = [
    Patch(facecolor="black", alpha=OFF_ALPHA, label="OFF (hatched)"),
    Patch(facecolor=COLOR_OD, alpha=0.55, label="On-demand"),
    Patch(facecolor=COLOR_BATCH, alpha=0.55, label="Batch"),
]
ax.legend(handles=legend_handles, loc="upper left", frameon=True)

ax.grid(True, axis="x", alpha=0.2)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200)
plt.close(fig)

print(f"Saved: {OUT_PATH}")
