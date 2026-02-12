import os
import re
import json
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
objective = "energy"
# objective = "server"

E_idle, E_cpu, E_mig = 100, 300, 50
# E_idle, E_cpu, E_mig = 100, 300, 0
M_on, M_off = 6, 6

random_seed = 42
round_result = True

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
rng = np.random.default_rng(random_seed)
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
model.addConstrs(gp.quicksum(y[i,k,n,t] for i in I for t in T_sp[k]) <= 1 for k in K for n in W_sp[k])
model.addConstrs(gp.quicksum(y[i,k,n,t] for i in I for n in W_sp[k]) <= 1 for k in K for t in T_sp[k])
model.addConstrs(gp.quicksum(y[i,k,n,t] for i in I for n in W_sp[k] for t in T_sp[k]) == len(W_sp[k]) for k in K)

# Batch Job workload Allocation (Splitting)
model.addConstrs(gp.quicksum(z[i,l,n,t] for i in I for t in T) <= 1 for l in L for n in W_bj[l])
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
def _json_default(o):
    # numpy scalar
    try:
        if isinstance(o, np.generic):
            return o.item()
    except Exception:
        pass

    # python "type" objects (e.g., int, float, ...)
    if isinstance(o, type):
        return o.__name__

    # fallback
    return str(o)

def json_dump_compact(obj, fp, indent=4, default=None, sort_keys=False):
    """
    json.dump를 대체하는 함수.
    기본적인 indent를 유지하되, 하위 리스트(primitive list)는 한 줄로 가로 정렬하여 파일에 씁니다.
    """
    # 1. 일단 문자열로 변환 (기본 Infinity가 포함된 상태로 생성됨)
    json_str = json.dumps(
        obj, 
        indent=indent, 
        default=default, 
        sort_keys=sort_keys, 
        ensure_ascii=False
    )

    # 2. [Infinity/NaN 처리] 
    # 정규식: 콜론(:), 콤마(,), 여는대괄호([) 뒤에 오는 Infinity나 NaN을 찾아서 null로 치환
    # (문자열 내부에 "Infinity"라고 적힌 것은 건드리지 않기 위해 문맥을 확인합니다)
    def replace_inf(match):
        # match.group(1): 구분자 (예: ": ", ", ")
        # match.group(2): 값 (Infinity, NaN 등) -> 이걸 'null'로 바꿈
        return match.group(1) + "null"
        
    # 패턴 설명: (구분자+공백)(Infinity 혹은 -Infinity 혹은 NaN)(단어끝)
    json_str = re.sub(r'([:\[,]\s*)(-?Infinity|NaN)\b', replace_inf, json_str)

    # 3. [가로 정렬 처리] 하위 리스트 압축
    def collapse_list(match):
        txt = match.group(0)
        return re.sub(r'\s+', ' ', txt).replace('[ ', '[').replace(' ]', ']')

    compact_json = re.sub(r'\[\s+([^\[\]\{\}]*?)\s+\]', collapse_list, json_str)

    # 4. 파일 쓰기
    fp.write(compact_json)

def optimize_and_save(
    m,
    out_dir="results",
    # sets
    I=None, T=None, J=None, K=None, L=None,
    T_od=None, T_sp=None,
    W_sp=None, W_bj=None,
    # usage (actual)
    c_od=None, c_sp=None, c_bj=None,
    # decision vars
    u=None, x=None, y=None, z=None, load=None, u_flag=None,
    # required resource
    on_demand_vm_data=None, spot_vm_data=None, batch_job_data=None,
    # run meta / parameters
    run_params=None,
    # capacity (normalized)
    server_capacity=1.0,
):
    os.makedirs(out_dir, exist_ok=True)

    # ------------------
    # Solve
    # ------------------
    m.optimize()
    has_sol = (m.SolCount is not None) and (m.SolCount > 0)
    
    # ---- Persist solution / variable values
    if has_sol:
        # Gurobi 표준 솔루션 파일
        try:
            m.write(os.path.join(out_dir, "solution.sol"))
        except Exception:
            pass

    else:
        # infeasible이면 IIS도 남겨서 직접 확인 가능하게
        if m.Status == GRB.INFEASIBLE:
            try:
                m.computeIIS()
                m.write(os.path.join(out_dir, "iis.ilp"))
            except Exception:
                pass
    
    # ------------------
    # Summary / solver info
    # ------------------
    summary = {
        "status": int(m.Status),
        "runtime": float(m.Runtime) if m.Runtime is not None else None,
        "sol_count": int(m.SolCount) if m.SolCount is not None else 0,
        "has_solution": bool(has_sol),
        "obj_val": float(m.ObjVal) if has_sol else None,
        "is_mip": bool(m.IsMIP),
    }
    if m.IsMIP:
        summary.update({
            "obj_bound": float(m.ObjBound) if m.ObjBound is not None else None,
            "mip_gap": float(m.MIPGap) if (has_sol and m.MIPGap is not None) else None,
            "node_count": int(m.NodeCount) if m.NodeCount is not None else None,
        })

    # ------------------
    # Collect gurobi parameter snapshot (best-effort)
    # ------------------
    gurobi_params = {}
    # NOTE: getParamInfo / getParam are available; we only record a small safe subset unless you want full dump.
    for p in ["CliqueCuts", "Cuts", "MIPFocus", "Heuristics", "CutPasses", "Method", "TimeLimit", "Threads"]:
        try:
            # getParamInfo는 (name, type, value, min, max, default)를 반환
            gurobi_params[p] = m.getParamInfo(p)[2]
        except Exception:
            pass

    # ------------------
    # If no solution, still write minimal JSON
    # ------------------
    result = {
        "run_params": run_params or {},
        "gurobi_params": gurobi_params,
        "summary": summary,
    }

    if not has_sol:
        with open(os.path.join(out_dir, "result.json"), "w") as f:
            # json.dump(result, f, indent=4, sort_keys=False)
            json_dump_compact(result, f, indent=4, sort_keys=False)
        return summary

    # ------------------
    # Build per-VM timelines + events
    # ------------------
    # ---- On-demand: placement timeline & migrations
    on_demand = {
        "VMs": {}, 
        "total_migrations": 0,
    }
    
    # od_place: precompute (j,t) -> i
    od_place = {}
    for j in J:
        for t in T_od[j]:
            assigned_i = None
            for i in I:
                if x[i, j, t].X > 0.5:
                    assigned_i = i
                    break
            od_place[(j, t)] = assigned_i

    for j in J:
        timeline = {
            "t": [],
            "server": [],
            "usage": [],
        }
        mig_events = []
        for idx, t in enumerate(sorted(T_od[j])):
            i_assigned = od_place[j, t]
            timeline["t"].append(t)
            timeline["server"].append(i_assigned)
            timeline["usage"].append(c_od[j, t])

            if idx >= 1:
                if i_prev != i_assigned:
                    mig_events.append({
                        "t": t, 
                        "from": i_prev,
                        "to": i_assigned,
                    })
            
            i_prev = i_assigned

        on_demand["VMs"][j] = {
            "a_j": on_demand_vm_data[j]["a_j"],
            "d_j": on_demand_vm_data[j]["d_j"],
            "required_vCPU": on_demand_vm_data[j]["required_vCPU"],
            "timeline": timeline,               # 시간순
            "migrations": mig_events,           # 이벤트 리스트
            "migration_count": len(mig_events),
        }
        on_demand["total_migrations"] += len(mig_events)

    # ---- Spot: workload placements (robust extraction from y tupledict)
    spot = {
        "VMs": {}, 
        "total_preemptions": 0, 
        "total_workloads": 0, 
        "preemption_per_workload": None,
    }

    # sp_state: (k,t)별 spot VM 실행 상태 추적
    sp_state = {}
    for k in K:
        for t in T_sp[k]:
            sp_state[k, t] = {"n": None, "server": None, "usage": None}

    # sp_workload_map: (k,n)별 배치
    sp_workload_map = {}
    
    for (i, k, n, t), var in y.items():
        xval = float(var.X)
        if xval <= 0.5:
            continue
        sp_state[k, t] = {"n": n, "server": i, "usage": c_sp[k, n]}
        sp_workload_map[k, n] = {"t": t, "server": i}

    # VM별 timeline + preemption 계산은 기존 로직 유지 (sp_state 기반)
    for k in K:
        # timeline
        timeline = {
            "t": [],
            "server": [],
            "workload": [],
            "usage": []
        }
        for t in sorted(T_sp[k]):
            item = sp_state[k, t]
            timeline["t"].append(t)
            timeline["server"].append(item["server"])
            timeline["workload"].append(item["n"])
            timeline["usage"].append(item["usage"])
        
        # preemptions
        preempt_events = []
        run_times = sorted([t for t in T_sp[k] if sp_state[k, t]["n"] is not None])
        if len(run_times) > 1:
            t_prev = run_times[0]
            svr_prev = sp_state[k, t_prev]["server"]
            for t_cur in run_times[1:]:
                svr_cur = sp_state[k, t_cur]["server"]
                
                # 건너뜀이 발생하면 idle_gap preemption
                if t_cur - t_prev > 1:
                    preempt_events.append({
                        "t": t_cur,
                        "reason": "idle_gap",
                        "gap": {"start": t_prev+1, "end": t_cur-1},
                        "server": {"from": svr_prev, "to": svr_cur},
                    })
                    
                # 서버 변경이 발생하면 server_change preemption
                elif svr_cur != svr_prev:
                    preempt_events.append({
                        "t": t_cur,
                        "reason": "server_change",
                        "gap": None,
                        "server": {"from": svr_prev, "to": svr_cur},
                    })
                
                t_prev, svr_prev = t_cur, svr_cur

        # workload_map_k = {str(n): sp_workload_map.get((k, n), None) for n in W_sp[k]}
        
        spot["VMs"][k] = {
            "a_k": spot_vm_data[k]["a_k"],
            "d_k": spot_vm_data[k]["d_k"],
            "required_vCPU": spot_vm_data[k]["required_vCPU"],
            "total_workloads": len(W_sp[k]),
            # "workload_map": workload_map_k,
            "timeline": timeline,
            "preemptions": preempt_events,
            "preemption_count": len(preempt_events),
        }
        
        spot["total_workloads"] += len(W_sp[k])
        spot["total_preemptions"] += len(preempt_events)

    spot["preemption_per_workload"] = (
        spot["total_preemptions"] / spot["total_workloads"]
        if spot["total_workloads"] > 0 else None
    )

    # ---- Batch: robust extraction from z tupledict
    batch = {
        "jobs": {},
        "jobs_running_per_time": {t: 0 for t in sorted(T)},
        "parallel_workload_ratio": None,
        "total_workloads": 0,
        "parallel_workloads": 0,
    }

    # bj_lt: (l,t)별 실행 workload 리스트
    bj_lt = {(l, t): [] for l in L for t in T}

    # bj_workload_map: (l,n)별 배치
    bj_workload_map = {}
    
    for (i, l, n, t), var in z.items():
        xval = float(var.X)
        if xval <= 0.5:
            continue
        bj_lt[l, t].append({"n": n, "server": i, "usage": c_bj[l, n]})
        bj_workload_map[l, n] = {"t": t, "server": i}

    # bj_lt 내부 workload 순서 정렬
    for key in bj_lt:
        bj_lt[key].sort(key=lambda a: a["n"])

    # per t: 실행 중인 job 수
    for t in T:
        batch["jobs_running_per_time"][t] = sum(bool(bj_lt[l, t]) for l in L)

    # timeline + parallel ratio
    for l in L:
        # timeline
        timeline = {
            "t": [],
            "server": [],
            "workload": [],
            "usage": [],
        }
        for t in sorted(T):
            items = bj_lt[l, t]
            for it in items:
                timeline["t"].append(t)
                timeline["server"].append(it["server"])
                timeline["workload"].append(it["n"])
                timeline["usage"].append(it["usage"])
            
            batch["total_workloads"] += len(items)
            if len(items) >= 2:
                batch["parallel_workloads"] += len(items)
        
        # workload_map_l = {n: bj_workload_map.get((l, n), None) for n in W_bj[l]}
        
        batch["jobs"][l] = {
            "required_vCPU": batch_job_data[l]["required_vCPU"],
            # "workload_map": workload_map_l,
            "timeline": timeline,
        }
    
    batch["parallel_workload_ratio"] = (
        batch["total_workloads"] / batch["parallel_workloads"]
        if batch["parallel_workloads"] > 0 else None
    )
    
    # ------------------
    # Server post-analysis (requested load / overbooking / contributions)
    # ------------------
    # Precompute on-demand required placement: (i,t) -> list of j
    od_it = {(i, t): [] for i in I for t in T}
    for j in J:
        req = on_demand_vm_data[j]["required_vCPU"]
        for t in T_od[j]:
            i_assigned = od_place[j, t]
            od_it[i_assigned, t].append(("on_demand", j, None, req, c_od[j, t]))

    # Precompute spot: (i,t) -> list of spot workloads (supports assignments list; no st["n"])
    sp_it = {(i, t): [] for i in I for t in T}
    for k in K:
        req = spot_vm_data[k]["required_vCPU"]
        for t in T_sp[k]:
            st = sp_state[k, t]
            if st["n"] is None:
                continue
            i, n = st["server"], st["n"]
            sp_it[i, t].append(("spot", k, n, req, st["usage"]))

    # Precompute batch: (i,t) -> possibly many (l,n)
    bj_it = {(i, t): [] for i in I for t in T}
    for l in L:
        req = batch_job_data[l]["required_vCPU"]
        for t in T:
            items = bj_lt[l, t]
            for it in items:
                i, n = it["server"], it["n"]
                bj_it[i, t].append(("batch", l, n, req, it["usage"]))

    servers = {
        "capacity": float(server_capacity),
        "servers": {}
    }

    for i in I:
        # used_flag는 u(i,t)=1이 한 번이라도 있으면 True로 정의 (u_flag는 energy 목적이면 unconstrained)
        used_flag = any((u[i, t].X > 0.5) for t in T)
        if not used_flag:
            # used가 아니면 저장 최소화
            servers["servers"][i] = {"used_flag": False}
            continue

        timeseries = []
        for t in T:
            u_val = 1 if u[i, t].X > 0.5 else 0

            contrib = []
            requested_load = 0.0
            actual_load = load[i, t].X

            # on-demand contributions
            for (typ, vid, wn, req, usage_val) in od_it[i, t]:
                contrib.append({
                    "type": typ,
                    "id": vid,
                    # "workload": wn,
                    "requested": req,
                    "usage": usage_val,
                })
                requested_load += req

            # spot contributions
            for (typ, vid, wn, req, usage_val) in sp_it[i, t]:
                contrib.append({
                    "type": typ,
                    "id": vid,
                    "workload": wn,
                    "requested": req,
                    "usage": usage_val,
                })
                requested_load += req

            # batch contributions
            for (typ, vid, wn, req, usage_val) in bj_it[i, t]:
                contrib.append({
                    "type": typ,
                    "id": vid,
                    "workload": wn,
                    "requested": req,
                    "usage": usage_val,
                })
                requested_load += req

            overbooking_ratio = (requested_load / server_capacity) if server_capacity > 0 else None

            timeseries.append({
                "t": t,
                "u": u_val,
                "actual_load": actual_load,
                "requested_load": requested_load,
                "overbooking_ratio": overbooking_ratio,
                "contrib": contrib,
            })

        servers["servers"][i] = {
            "used_flag": True,
            "timeseries": timeseries,
        }

    # ------------------
    # Server KPI aggregation (per-server + overall)
    # ------------------
    servers_kpi = {"per_server": {}, "overall": {}}

    cap = float(server_capacity) if server_capacity is not None else 1.0

    def safe_mean(vals):
        vals = [v for v in vals if v is not None]
        return (float(sum(vals)) / len(vals)) if len(vals) > 0 else None

    def safe_max(vals):
        vals = [v for v in vals if v is not None]
        return float(max(vals)) if len(vals) > 0 else None

    used_server_count = 0
    all_on_frac = []
    all_peak_over = []
    all_avg_over_on = []
    all_peak_util = []
    all_avg_util_on = []

    for i, sdata in servers["servers"].items():
        if not sdata.get("used_flag", False):
            servers_kpi["per_server"][i] = {"used_flag": False}
            continue
        
        ts = sdata["timeseries"]

        u_series = [int(r["u"]) for r in ts]
        on_times = [r for r in ts if int(r["u"]) == 1]

        # used_flag는 이미 time-series 기반으로 True 가능, 여기선 그대로 사용
        used = bool(sdata["used_flag"])
        if used:
            used_server_count += 1

        # on/off transitions
        on_transitions = 0
        off_transitions = 0
        for idx in range(1, len(u_series)):
            if u_series[idx-1] == 0 and u_series[idx] == 1:
                on_transitions += 1
            if u_series[idx-1] == 1 and u_series[idx] == 0:
                off_transitions += 1

        # on fraction
        on_frac = (float(sum(u_series)) / float(len(u_series))) if len(u_series) > 0 else None

        # overbooking / loads / utilization
        over_all = [r.get("overbooking_ratio", None) for r in ts]
        over_on  = [r.get("overbooking_ratio", None) for r in on_times]

        req_all = [r.get("requested_load", None) for r in ts]
        req_on  = [r.get("requested_load", None) for r in on_times]

        act_all = [r.get("actual_load", None) for r in ts]
        act_on  = [r.get("actual_load", None) for r in on_times]

        util_all = [(r["actual_load"] / cap) if (r.get("actual_load", None) is not None and cap > 0) else None for r in ts]
        util_on  = [(r["actual_load"] / cap) if (r.get("actual_load", None) is not None and cap > 0) else None for r in on_times]

        kpi_i = {
            "used_flag": used,
            "on_fraction": on_frac,

            # switching/cycles
            "on_transitions": int(on_transitions),
            "off_transitions": int(off_transitions),
            "num_cycles_on": int(on_transitions),  # 0->1을 on-cycle 시작으로 정의

            # requested load
            "requested_load_peak": safe_max(req_all),
            "requested_load_avg_on": safe_mean(req_on),

            # actual load
            "actual_load_peak": safe_max(act_all),
            "actual_load_avg_on": safe_mean(act_on),

            # overbooking
            "overbooking_ratio_peak": safe_max(over_all),
            "overbooking_ratio_avg_on": safe_mean(over_on),

            # utilization
            "utilization_peak": safe_max(util_all),
            "utilization_avg_on": safe_mean(util_on),

            # optional: idle-on waste proxy (on인데 actual=0 비율)
            "idle_on_fraction": (
                float(sum(1 for r in on_times if (r.get("actual_load", 0.0) is not None and r.get("actual_load", 0.0) <= 1e-12)))
                / float(len(on_times))
            ) if len(on_times) > 0 else None,
        }

        servers_kpi["per_server"][i] = kpi_i

        if on_frac is not None:
            all_on_frac.append(on_frac)
        if kpi_i["overbooking_ratio_peak"] is not None:
            all_peak_over.append(kpi_i["overbooking_ratio_peak"])
        if kpi_i["overbooking_ratio_avg_on"] is not None:
            all_avg_over_on.append(kpi_i["overbooking_ratio_avg_on"])
        if kpi_i["utilization_peak"] is not None:
            all_peak_util.append(kpi_i["utilization_peak"])
        if kpi_i["utilization_avg_on"] is not None:
            all_avg_util_on.append(kpi_i["utilization_avg_on"])

    servers_kpi["overall"] = {
        "num_servers_total": int(len(I)),
        "num_servers_used": int(used_server_count),

        "avg_on_fraction_across_servers": safe_mean(all_on_frac),
        "peak_overbooking_across_servers": safe_max(all_peak_over),
        "avg_overbooking_on_across_servers": safe_mean(all_avg_over_on),

        "peak_utilization_across_servers": safe_max(all_peak_util),
        "avg_utilization_on_across_servers": safe_mean(all_avg_util_on),
    }

    # result에 KPI를 포함
    result.update({"servers_kpi": servers_kpi})

    # ------------------
    # Final JSON
    # ------------------
    result.update({
        "on_demand": on_demand,
        "spot": spot,
        "batch": batch,
        "servers": servers,
    })

    def _round_floats(obj, ndigits: int):
        # float -> round
        if isinstance(obj, float):
            return round(obj, ndigits)

        # numpy scalar
        try:
            if isinstance(obj, np.generic):
                v = obj.item()
                return round(v, ndigits) if isinstance(v, float) else v
        except Exception:
            pass

        # containers
        if isinstance(obj, dict):
            return {k: _round_floats(v, ndigits) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [ _round_floats(v, ndigits) for v in obj ]

        return obj
    
    if round_result:
        result = _round_floats(result, ndigits=4)
    
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        # json.dump(result, f, indent=4, default=_json_default, sort_keys=False)
        json_dump_compact(result, f, indent=4, default=_json_default, sort_keys=False)

    return summary


if __name__ == "__main__":
    curr_timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

    # 원하는 “파라미터/실험 설정”을 json에 박제
    run_params = {
        "T_max": T_max,
        "usage_target": usage_target,
        "spot_buffer_ratio": spot_buffer_ratio,
        "objective": objective,
        "E_idle": E_idle,
        "E_cpu": E_cpu,
        "E_mig": E_mig,
        "M_on": M_on,
        "M_off": M_off,
        "dataset": {
            "max_vCPU": float(max_vCPU),
            "num_rows": len(vm_data),
            "num_on_demand_vms": len(J),
            "num_spot_vms": len(K),
            "num_batch_jobs": len(L),
        },
        "num_servers": len(I),
        "random_seed": random_seed,
    }

    summary = optimize_and_save(
        model,
        out_dir=f"results/{curr_timestamp}",

        I=I, T=T, J=J, K=K, L=L,
        T_od=T_od, T_sp=T_sp,
        W_sp=W_sp, W_bj=W_bj,
        c_od=c_od, c_sp=c_sp, c_bj=c_bj,

        u=u, x=x, y=y, z=z, load=load, u_flag=u_flag,

        on_demand_vm_data=on_demand_vm_data,
        spot_vm_data=spot_vm_data,
        batch_job_data=batch_job_data,

        run_params=run_params,
        server_capacity=1.0,  # 현재 모델이 load <= u 로 capacity=1로 동작
    )