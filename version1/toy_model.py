import math
import pandas as pd
import gurobipy as gp
from gurobipy import GRB


# ----------------------------
# Parameters
# ----------------------------
pCPU = 24       # 각 서버의 pCPU 개수
D = 4           # batch + spot Deadline 설정
# E_idle, E_cpu, E_mig = 100, 300, 50
E_idle, E_cpu, E_mig = 100, 300, 0
M_on, M_off = 6, 6


# ----------------------------
# Load Dataset
# ----------------------------
vm_data = pd.read_csv("sample_vm_data.csv")
vm_data["vm_id"] = vm_data["vm_id"].str.slice(0, 6)

# workload 분리
is_on_demand = vm_data["vm_category"].str.lower().eq("interactive")
on_demand_df = vm_data.loc[is_on_demand].reset_index(drop=True)
batch_spot_df = vm_data.loc[~is_on_demand].reset_index(drop=True)

# on-demand VMs
vm_hash = on_demand_df["vm_id"].unique()
vm_id_map = dict(zip(vm_hash, range(len(vm_hash))))
on_demand_df["vm_id"] = on_demand_df["vm_id"].map(vm_id_map)


# ----------------------------
# Data Driven Parameters
# ----------------------------
agg_col = "avg_core_usage"

# server index
hourly_required_pCPU = vm_data.groupby("hour")[agg_col].sum()
max_required_server = math.ceil((hourly_required_pCPU / pCPU).max())
I = list(range(max_required_server))

# VM index
J = sorted(list(set(on_demand_df["vm_id"])))

# Time horizon
T_lim = vm_data["hour"].max()
T = list(range(T_lim + 1))

# W_t: time t에서 요청된 batch + spot workload
# (데이터는 실행된 시점이지만, 요청된 시점으로 간주함)
batch_spot_workload = batch_spot_df.groupby("hour")[agg_col].sum()
W = {t:batch_spot_workload[t].item() for t in T}

# a_j, d_j: on-demand VM j의 시작 및 종료 time period
vm_time_table = (
    on_demand_df
    .groupby("vm_id", as_index=False)
    .agg(a_j=("hour", "min"), d_j=("hour", "max"))
)
active_hours = {}
for j in J:
    a, d = vm_time_table.loc[j, "a_j":"d_j"].values
    active_hours[j] = list(range(a, d+1))

# c_jt: t시점에서 on-demand VM j의 실제 pCPU 사용량
_on_demand_df = on_demand_df.set_index(["vm_id", "hour"])
c = {(j, t): _on_demand_df.loc[(j, t), agg_col].item()
     for j in J for t in active_hours[j]}


# ----------------------------
# Init Gurobi Model
# ----------------------------
m = gp.Model("Cloud Operation")


# ----------------------------
# Define Decision Variables
# ----------------------------
# u_it: Server on/off
u = m.addVars(I, T, vtype=GRB.BINARY, name="u")
u_on = m.addVars(I, T, vtype=GRB.BINARY, name="u_on")
u_off = m.addVars(I, T, vtype=GRB.BINARY, name="u_off")

# x_ijt: On-demand VM placement
x_indices = [(i, j, t) for i in I for j in J for t in active_hours[j]]
x = m.addVars(x_indices, vtype=GRB.BINARY, name="x")

# w_itl: Amount of batch + spot workload requested at time t, allocated to server i at time l
w_indices = [(i, t, l) for i in I for t in T for l in range(t, t+D+1)]
w = m.addVars(w_indices, lb=0.0, vtype=GRB.CONTINUOUS, name="w")

# L_it: Load of server i at time t
L = m.addVars(I, T, lb=0.0, vtype=GRB.CONTINUOUS, name="L")

# q_jt: On-demand VM j migrates at time t
q_indices = [(j, t) for j in J for t in active_hours[j][1:]]
q = m.addVars(q_indices, vtype=GRB.BINARY, name="q")


# ----------------------------
# Add Constraints
# ----------------------------
# Server Provisioning
for i in I:
    for t in T[:-1]:
        m.addConstr(u[i,t] + u_on[i,t] - u_off[i,t] - u[i,t+1] == 0)

    for t in T:
        m.addConstrs(u_on[i,t] - u[i,l] <= 0 for l in range(t+1, min(T[-1],t+M_on)+1))

    for t in T:
        m.addConstrs(u_off[i,t] + u[i,l] <= 1 for l in range(t+1, min(T[-1],t+M_off)+1))

# On-demand VM placement
m.addConstrs(gp.quicksum(x[i,j,t] for i in I) == 1 for j in J for t in active_hours[j])

# Batch + spot workload allocation with deadline
for t in T:
    m.addConstr(gp.quicksum(w[i,t,l] for i in I for l in range(t, t+D+1)) == W[t])

# Server Capacity constraint
for i in I:
    for t in T:
        on_demand_workload = gp.quicksum(c[j,t]*x[i,j,t] for j in J if t in active_hours[j])
        # batch_spot_workload = gp.quicksum(w[i,l,t] for l in range(max(t-D, 0), t+1) if l < t and l <= T[-1])
        batch_spot_workload = gp.quicksum(w[i,l,t] for l in range(max(t-D, 0), t+1))
        m.addConstr(L[i,t] == on_demand_workload + batch_spot_workload)
        m.addConstr(L[i,t] <= pCPU * u[i,t])

# VM Migration model
m.addConstrs(q[j,t] >= x[i,j,t] - x[i,j,t-1] for i in I for j in J for t in active_hours[j][1:])
m.addConstrs(q[j,t] >= x[i,j,t-1] - x[i,j,t] for i in I for j in J for t in active_hours[j][1:])

# Special Constraint
# m.addConstrs(x[i,j,t] == 0 for i in I[5:] for j in J for t in active_hours[j])
# m.addConstr(gp.quicksum(q[j,t] for j in J for t in active_hours[j][1:]) <= 20)
# m.addConstr(gp.quicksum(q[j,t] for j in J for t in active_hours[j][1:]) >= 10)

# ----------------------------
# Set Objective
# ----------------------------
energy_consumption_idle = E_idle * gp.quicksum(u[i,t] for i in I for t in T)
energy_consumption_cpu = E_cpu / pCPU * gp.quicksum(L[i,t] for i in I for t in T)
energy_consumption_mig = E_mig * gp.quicksum(q[j,t] for j in J for t in active_hours[j][1:])

m.setObjective(energy_consumption_idle + energy_consumption_cpu + energy_consumption_mig)

# m.setParam("Method", 3)  # concurrent LP를 명시적으로 선택
# m.setParam("MIPFocus", 1)
# m.setParam("Heuristics", 0.2)
# m.setParam("CutPasses", 10)  # Cutting Plane을 명시적으로 제한
m.setParam("CliqueCuts", 1)
# m.setParam("Cuts", 1)

# ----------------------------
# Solve Model and Save
# ----------------------------
import os
import json
import csv

def optimize_and_save(m, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)

    m.optimize()

    has_sol = (m.SolCount is not None) and (m.SolCount > 0)

    # ------------------
    # Collect summary
    # ------------------
    summary = {
        "status": int(m.Status),
        "runtime": m.Runtime,
        "sol_count": int(m.SolCount),
        "has_solution": bool(has_sol),
        "obj_val": m.ObjVal if has_sol else None,
        "is_mip": bool(m.IsMIP),
    }

    if m.IsMIP:
        summary.update({
            "obj_bound": m.ObjBound,
            "mip_gap": m.MIPGap if has_sol else None,
            "node_count": m.NodeCount,
        })

    # ------------------
    # Collect variables
    # ------------------
    variables = []
    for v in m.getVars():
        variables.append({
            "name": v.VarName,
            "value": float(v.X) if has_sol else None
        })

    # ------------------
    # Collect linear constraints (Constr)
    # Slack is only available if a solution exists
    # ------------------
    constraints = []
    for c in m.getConstrs():
        constraints.append({
            "name": c.ConstrName,
            "rhs": float(c.RHS),
            "slack": float(c.Slack) if has_sol else None
        })

    # ------------------
    # Write JSON
    # ------------------
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(out_dir, "variables.json"), "w") as f:
        json.dump(variables, f, indent=2)

    with open(os.path.join(out_dir, "constraints.json"), "w") as f:
        json.dump(constraints, f, indent=2)

    # ------------------
    # Write CSV
    # ------------------
    with open(os.path.join(out_dir, "variables.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["var_name", "value"])
        for row in variables:
            writer.writerow([row["name"], row["value"]])

    with open(os.path.join(out_dir, "constraints.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["constr_name", "rhs", "slack"])
        for row in constraints:
            writer.writerow([row["name"], row["rhs"], row["slack"]])

    # ------------------
    # Save native Gurobi files
    # ------------------
    m.write(os.path.join(out_dir, "model.lp"))
    if has_sol:
        m.write(os.path.join(out_dir, "solution.sol"))

    return summary

if __name__ == "__main__":
    from datetime import datetime
    
    # 실행 시점 timestamp 생성
    curr_timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    
    # Solve
    optimize_and_save(m, out_dir=f"results/{curr_timestamp}")
