import os
import json
import csv
from datetime import datetime

import math
from itertools import product

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB


# ----------------------------
# Parameters
# ----------------------------
T_max = 23  # from 0 to T_max
usage_target = "max_avg_core_usage"
spot_buffer_ratio = 0.25  # Beta
# objective = "energy"
objective = "server"

E_idle, E_cpu, E_mig = 100, 300, 50
# E_idle, E_cpu, E_mig = 100, 300, 0
M_on, M_off = 6, 6


# ----------------------------
# Load Dataset
# ----------------------------
vm_data = pd.read_csv("sample_vm_data.csv")
vm_data["vm_id"] = vm_data["vm_id"].str.slice(0, 8)

# Normalize core_usage values (0-1)
max_vCPU = vm_data["vCPU"].max()
for col_prefix in ["min", "avg", "max_avg", "max"]:
    colname = col_prefix+"_core_usage"
    vm_data[colname] /= max_vCPU
vm_data["vCPU"] /= max_vCPU

# on-demand, else 분리
is_on_demand = vm_data["vm_category"].str.lower().eq("interactive")
on_demand_df = vm_data.loc[is_on_demand].reset_index(drop=True)
batch_spot_df = vm_data.loc[~is_on_demand].reset_index(drop=True)

# spot vm, batch job 분리 
batch_spot_vm_id = batch_spot_df["vm_id"].unique()
rng = np.random.default_rng(42)
spot_vm_id = rng.choice(batch_spot_vm_id, size=len(batch_spot_vm_id)//2, replace=False)

spot_vm_df = batch_spot_df.loc[batch_spot_df["vm_id"].isin(spot_vm_id)].reset_index(drop=True)
batch_job_df = batch_spot_df.loc[~batch_spot_df["vm_id"].isin(spot_vm_id)].reset_index(drop=True)

# vm index 부여
on_demand_vm_map = dict(zip(on_demand_df["vm_id"].unique(), range(on_demand_df["vm_id"].nunique())))
on_demand_df["vm_id"] = on_demand_df["vm_id"].map(on_demand_vm_map)

spot_vm_map = dict(zip(spot_vm_df["vm_id"].unique(), range(spot_vm_df["vm_id"].nunique())))
spot_vm_df["vm_id"] = spot_vm_df["vm_id"].map(spot_vm_map)

batch_job_vm_map = dict(zip(batch_job_df["vm_id"].unique(), range(batch_job_df["vm_id"].nunique())))
batch_job_df["vm_id"] = batch_job_df["vm_id"].map(batch_job_vm_map)

print("# of on-demand VMs:", on_demand_df["vm_id"].nunique())
print("# of spot VMs:", spot_vm_df["vm_id"].nunique())
print("# of batch jobs (VM):", batch_job_df["vm_id"].nunique())


# ----------------------------
# Data Preparation
# ----------------------------
# on-demand VM data
on_demand_vm_data = dict()
for vm_id, g in on_demand_df.groupby("vm_id"):
    timestamp = g["timestamp"].values
    usage = g[usage_target].values
    on_demand_vm_data[vm_id] = {
        "required_vCPU": g["vCPU"].unique()[0],
        "usage_data": dict(zip(timestamp, usage)),
        "a_j": timestamp.min(),
        "d_j": timestamp.max(),
    }

# spot VM data
spot_vm_data = dict()
for vm_id, g in spot_vm_df.groupby("vm_id"):
    timestamp = g["timestamp"].values
    usage = g[usage_target].values
    
    # preemption buffer를 넣기 위해 workload 수를 제한
    max_workload = math.floor((T_max - timestamp.min()) / (1 + spot_buffer_ratio))
    num_workload = min(len(usage), max_workload)
    
    spot_vm_data[vm_id] = {
        "required_vCPU": g["vCPU"].unique()[0],
        "workloads": dict(zip(range(num_workload), usage[:num_workload])),
        "a_k": timestamp.min(),
        "d_k": timestamp.min() + num_workload + math.ceil(num_workload * spot_buffer_ratio),
    }

# batch job data
batch_job_data = dict()
for vm_id, g in batch_job_df.groupby("vm_id"):
    timestamp = g["timestamp"].values
    usage = g[usage_target].values
    batch_job_data[vm_id] = {
        "required_vCPU": g["vCPU"].unique()[0],
        "workloads": dict(zip(range(len(usage)), usage)),
    }


# ----------------------------
# Data Driven Parameters
# ----------------------------
# server index
required_pCPU_per_time = vm_data.groupby("timestamp")["vCPU"].sum()
max_required_server = math.ceil(required_pCPU_per_time.max())
I = list(range(max_required_server))

# VM index
J = list(sorted(on_demand_vm_data.keys()))
K = list(sorted(spot_vm_data.keys()))
L = list(sorted(batch_job_data.keys()))

# Workload index
W_sp = dict()
for k in K:
    W_sp[k] = list(spot_vm_data[k]["workloads"].keys())

W_bj = dict()
for l in L:
    W_bj[l] = list(batch_job_data[l]["workloads"].keys())

# Planning horizon
T = list(range(T_max+1))

# VM Activation periods
T_od = dict()
for j in J:
    a_j = on_demand_vm_data[j]["a_j"]
    d_j = on_demand_vm_data[j]["d_j"]
    T_od[j] = list(range(a_j, d_j+1))

T_sp = dict()
for k in K:
    a_k = spot_vm_data[k]["a_k"]
    d_k = spot_vm_data[k]["d_k"]
    T_sp[k] = list(range(a_k, d_k+1))

# vCPU utilization
c_od = dict()
for j in J:
    for t, c_t in on_demand_vm_data[j]["usage_data"].items():
        c_od[j, t] = c_t

c_sp = dict()
for k in K:
    for n, c_t in spot_vm_data[k]["workloads"].items():
        c_sp[k, n] = c_t

c_bj = dict()
for l in L:
    for n, c_t in batch_job_data[l]["workloads"].items():
        c_bj[l, n] = c_t


# ----------------------------
# Init Gurobi Model
# ----------------------------
model = gp.Model("Cloud Operation")


# ----------------------------
# Define Decision Variables
# ----------------------------
# Server Provisioning
u = model.addVars(I, T, vtype=GRB.BINARY, name="u")
u_on = model.addVars(I, T, vtype=GRB.BINARY, name="u_on")
u_off = model.addVars(I, T, vtype=GRB.BINARY, name="u_off")
u_flag = model.addVars(I, vtype=GRB.BINARY, name="u_flag")

# VM Placement
x_indices = [(i, j, t) for i in I for j in J for t in T_od[j]]
x = model.addVars(x_indices, vtype=GRB.BINARY, name="x")

m_indices = [(j, t) for j in J for t in T_od[j][1:]]
m = model.addVars(m_indices, vtype=GRB.BINARY, name="m")

y_indices = [(i,k,n,t) for i in I for k in K for n in W_sp[k] for t in T_sp[k]]
y = model.addVars(y_indices, vtype=GRB.BINARY, name="y")

z_indices = [(i,l,n,t) for i in I for l in L for n in W_bj[l] for t in T]
z = model.addVars(z_indices, vtype=GRB.BINARY, name="z")

# Server Load
load_indices = [(i, t) for i in I for t in T]
load = model.addVars(load_indices, lb=0, vtype=GRB.CONTINUOUS, name="load")


# ----------------------------
# Add Constraints
# ----------------------------
# Server Provisioning
for i in I:
    for t in T[:-1]:
        model.addConstr(u[i,t] + u_on[i,t] - u_off[i,t] - u[i,t+1] == 0)

    for t in T:
        model.addConstrs(u_on[i,t] - u[i,l] <= 0 for l in range(t+1, min(T[-1],t+M_on)+1))
        model.addConstrs(u_off[i,t] + u[i,l] <= 1 for l in range(t+1, min(T[-1],t+M_off)+1))

# On-demand VM placement & Migration
model.addConstrs(gp.quicksum(x[i,j,t] for i in I) == 1 for j in J for t in T_od[j])
model.addConstrs(m[j,t] >= x[i,j,t] - x[i,j,t-1] for i in I for j in J for t in T_od[j][1:])
model.addConstrs(m[j,t] >= x[i,j,t-1] - x[i,j,t] for i in I for j in J for t in T_od[j][1:])

# Spot VM workload Placement (Preemption)
model.addConstrs(gp.quicksum(y[i,k,n,t] for i in I for n in W_sp[k]) <= 1 for k in K for t in T_sp[k])
model.addConstrs(gp.quicksum(y[i,k,n,t] for i in I for n in W_sp[k] for t in T_sp[k]) == len(W_sp[k]) for k in K)

# Batch Job workload Allocation (Splitting)
model.addConstrs(gp.quicksum(z[i,l,n,t] for i in I for n in W_bj[l] for t in T) == len(W_bj[l]) for l in L)

# Server Capacity
for i, t in product(I, T):
    on_demand_load = gp.quicksum(c_od[j,t] * x[i,j,t] for j in J)
    spot_load = gp.quicksum(c_sp[k,n] * y[i,k,n,t] for k in K for n in W_sp[k])
    batch_load = gp.quicksum(c_bj[l,n] * z[i,l,n,t] for l in L for n in W_bj[l])
    
    model.addConstr(load[i,t] == on_demand_load + spot_load + batch_load)
    model.addConstr(load[i,t] <= u[i,t])

# Special Constraint
# model.addConstrs(x[i,j,t] == 0 for i in I[5:] for j in J for t in active_hours[j])
# model.addConstr(gp.quicksum(q[j,t] for j in J for t in active_hours[j][1:]) <= 20)
# model.addConstr(gp.quicksum(q[j,t] for j in J for t in active_hours[j][1:]) >= 10)


# ----------------------------
# Set Objective
# ----------------------------
if objective == "energy":
    # Energy Consumption Minimization
    energy_consumption_idle = E_idle * gp.quicksum(u[i,t] for i in I for t in T)
    energy_consumption_cpu = E_cpu * gp.quicksum(load[i,t] for i in I for t in T)
    energy_consumption_mig = E_mig * gp.quicksum(m[j,t] for j in J for t in T_od[j][1:])
    
    model.setObjective(energy_consumption_idle + energy_consumption_cpu + energy_consumption_mig)
    
else:
    # Server Count Minimization
    model.addConstrs(u_flag[i] >= u[i,t] for i in I for t in T)
    model.setObjective(gp.quicksum(u_flag[i] for i in I))

# model.setParam("Method", 3)  # concurrent LP를 명시적으로 선택
# model.setParam("MIPFocus", 1)
# model.setParam("Heuristics", 0.2)
# model.setParam("CutPasses", 10)  # Cutting Plane을 명시적으로 제한
model.setParam("CliqueCuts", 1)
# model.setParam("Cuts", 1)

# ----------------------------
# Solve Model and Save
# ----------------------------
def optimize_and_save(
    m,
    out_dir="results",
    # --- 아래는 그래프용 tidy 저장을 위해 추가로 받는 인자들 ---
    I=None, T=None, J=None, K=None, L=None,
    T_od=None, T_sp=None,
    W_sp=None, W_bj=None,
    c_od=None, c_sp=None, c_bj=None,
    u=None, x=None, y=None, z=None, load=None,
):
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

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ------------------
    # (옵션) 원본 variables dump 유지
    # ------------------
    variables = []
    for v in m.getVars():
        variables.append({"name": v.VarName, "value": float(v.X) if has_sol else None})

    with open(os.path.join(out_dir, "variables.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["var_name", "value"])
        for row in variables:
            writer.writerow([row["name"], row["value"]])

    # ------------------
    # 그래프용 tidy CSV 저장 (해결 시에만)
    # ------------------
    if has_sol and all(arg is not None for arg in [I,T,J,K,L,T_od,T_sp,W_sp,W_bj,c_od,c_sp,c_bj,u,x,y,z,load]):

        # 1) server_timeseries.csv
        server_rows = []
        for i in I:
            for t in T:
                server_rows.append({
                    "i": int(i),
                    "t": int(t),
                    "u": int(round(u[i,t].X)),
                    "load": float(load[i,t].X),
                })
        pd.DataFrame(server_rows).to_csv(os.path.join(out_dir, "server_timeseries.csv"), index=False)

        # 2) place_on_demand.csv
        od_rows = []
        for j in J:
            for t in T_od[j]:
                # find i with x[i,j,t] = 1
                i_assigned = None
                for i in I:
                    if x[i,j,t].X > 0.5:
                        i_assigned = i
                        break
                if i_assigned is None:
                    continue
                od_rows.append({
                    "j": int(j),
                    "t": int(t),
                    "i": int(i_assigned),
                    "usage": float(c_od[j,t]),
                })
        pd.DataFrame(od_rows).to_csv(os.path.join(out_dir, "place_on_demand.csv"), index=False)

        # 3) place_spot.csv
        sp_rows = []
        for k in K:
            for t in T_sp[k]:
                for n in W_sp[k]:
                    for i in I:
                        if y[i,k,n,t].X > 0.5:
                            sp_rows.append({
                                "k": int(k),
                                "n": int(n),
                                "t": int(t),
                                "i": int(i),
                                "usage": float(c_sp[k,n]),
                            })
        pd.DataFrame(sp_rows).to_csv(os.path.join(out_dir, "place_spot.csv"), index=False)

        # 4) place_batch.csv
        bj_rows = []
        for l in L:
            for n in W_bj[l]:
                for t in T:
                    for i in I:
                        if z[i,l,n,t].X > 0.5:
                            bj_rows.append({
                                "l": int(l),
                                "n": int(n),
                                "t": int(t),
                                "i": int(i),
                                "usage": float(c_bj[l,n]),
                            })
        pd.DataFrame(bj_rows).to_csv(os.path.join(out_dir, "place_batch.csv"), index=False)

        # 5) migration.csv (x로 from/to 복원)
        mig_rows = []
        # 미리 (j,t)->i 맵
        jt_i = {(r["j"], r["t"]): r["i"] for r in od_rows}
        for j in J:
            Ts = T_od[j]
            for idx in range(1, len(Ts)):
                t = Ts[idx]
                t_prev = Ts[idx-1]
                i_fr = jt_i.get((int(j), int(t_prev)))
                i_to = jt_i.get((int(j), int(t)))
                if (i_fr is not None) and (i_to is not None) and (i_fr != i_to):
                    mig_rows.append({"j": int(j), "t": int(t), "i_from": int(i_fr), "i_to": int(i_to)})
        pd.DataFrame(mig_rows).to_csv(os.path.join(out_dir, "migration.csv"), index=False)

    # ------------------
    # Save native Gurobi files
    # ------------------
    m.write(os.path.join(out_dir, "model.lp"))
    if has_sol:
        m.write(os.path.join(out_dir, "solution.sol"))

    return summary

if __name__ == "__main__":
    # 실행 시점 timestamp 생성
    curr_timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    
    # Solve
    summary = optimize_and_save(
        model, out_dir=f"results/{curr_timestamp}",
        I=I, T=T, J=J, K=K, L=L,
        T_od=T_od, T_sp=T_sp,
        W_sp=W_sp, W_bj=W_bj,
        c_od=c_od, c_sp=c_sp, c_bj=c_bj,
        u=u, x=x, y=y, z=z, load=load
    )
