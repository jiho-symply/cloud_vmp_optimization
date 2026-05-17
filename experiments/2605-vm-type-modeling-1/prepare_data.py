"""
Prepare a compact JSON instance for the Notion "VM type modeling (1)" model.

This script deliberately reuses the 2604 chance-2SP toy data instead of
resampling Azure traces again.  The existing folder already contains:

- workload_metadata.csv: selected on-demand, spot, and batch workloads
- scenario_time_series.csv: scenario demand for on-demand and spot VMs
- batch_jobs.csv / batch_job_demands.csv: one-slot batch fragments
- instance.json: server capacity, scenarios, chance-constraint settings

The Notion model treats a batch job k as a larger workload W_k with reserved
batch-VM slots b[k,s,t] and realized processing volume z[k,s,t,xi].  The 2604
data stores each original batch workload as many one-slot fragments, so this
script folds those fragments back to their parent workload id:

- K is the set of parent_workload_id values such as bj_00.
- W_k is the worst-case total batch demand across scenarios.
- r_B[k] is the selected VM's vCPU, capped by the homogeneous server capacity.
- N_B[k] = ceil((1 + kappa) * W_k / r_B[k]).

The output JSON is intentionally simple and verbose.  model.py reads it
directly, so there is no hidden preprocessing inside the optimization code.
"""

import argparse
import json
import math
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = (
    REPO_ROOT
    / "data"
    / "processed"
    / "2604-chance-2sp-toy"
    / "chance_2sp_toy_24vm_combination_od_sp_bj_od8_sp8_bj8_sc10_cap8_avg20_lam010"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "data"
    / "processed"
    / "2605-vm-type-modeling-1"
    / "notion_vm_type_24vm_od8_sp8_bj8_sc10_cap8"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build a compact JSON instance for VM type modeling (1).")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--kappa", type=float, default=0.20, help="Batch reservation slack ratio.")
    parser.add_argument("--objective-type", choices=["server_count", "energy"], default="energy")
    parser.add_argument("--lambda-migration", type=float, default=0.10)
    parser.add_argument("--energy-idle", type=float, default=100.0)
    parser.add_argument("--energy-cpu", type=float, default=300.0)
    parser.add_argument("--energy-migration", type=float, default=50.0)
    return parser.parse_args()


def active_periods(frame, workload_type):
    filtered = frame.loc[frame["workload_type"] == workload_type]
    return {
        workload_id: [int(value) for value in sorted(group["time"].unique())]
        for workload_id, group in filtered.groupby("workload_id")
    }


def demand_records(frame, workload_type):
    filtered = frame.loc[frame["workload_type"] == workload_type]
    return [
        {
            "id": row.workload_id,
            "time": int(row.time),
            "scenario": row.scenario,
            "demand": float(row.demand),
        }
        for row in filtered.itertuples(index=False)
    ]


def build_batch_jobs(batch_jobs, batch_demands, capacity, kappa):
    rows = []
    for batch_id, group in batch_jobs.groupby("parent_workload_id"):
        demand_by_scenario = (
            batch_demands.loc[batch_demands["parent_workload_id"] == batch_id]
            .groupby("scenario")["demand"]
            .sum()
        )
        reserved_cpu = min(float(group["vCPU"].max()), capacity)
        workload = float(demand_by_scenario.max()) if len(demand_by_scenario) else 0.0
        required_slots = int(math.ceil((1.0 + kappa) * workload / reserved_cpu)) if reserved_cpu > 0 else 0
        rows.append(
            {
                "id": batch_id,
                "reserved_cpu": reserved_cpu,
                "workload": workload,
                "processing_slots": required_slots,
                "source_slots": int(group["source_time"].nunique()),
            }
        )
    return sorted(rows, key=lambda row: row["id"])


def build_instance(source_dir, kappa, objective_type, lambda_migration, energy_idle, energy_cpu, energy_migration):
    with open(source_dir / "instance.json", "r", encoding="utf-8") as file:
        source_instance = json.load(file)

    metadata = pd.read_csv(source_dir / "workload_metadata.csv")
    scenario_time_series = pd.read_csv(source_dir / "scenario_time_series.csv")
    batch_jobs = pd.read_csv(source_dir / "batch_jobs.csv")
    batch_demands = pd.read_csv(source_dir / "batch_job_demands.csv")

    capacity = float(source_instance["server_capacity"])
    scenarios = [
        {"id": row["scenario"], "probability": float(row["probability"])}
        for row in source_instance["scenarios"]
    ]

    return {
        "name": "notion_vm_type_modeling_1_from_" + source_instance["instance_name"],
        "source_instance": source_instance["instance_name"],
        "sets": {
            "servers": list(range(int(source_instance["num_servers"]))),
            "times": [int(value) for value in source_instance["time_periods"]],
            "on_demand": metadata.loc[metadata["workload_type"] == "on_demand", "workload_id"].tolist(),
            "spot": metadata.loc[metadata["workload_type"] == "spot", "workload_id"].tolist(),
            "batch": sorted(batch_jobs["parent_workload_id"].unique().tolist()),
            "scenarios": [scenario["id"] for scenario in scenarios],
        },
        "parameters": {
            "capacity": capacity,
            "big_m": float(source_instance["big_m"]),
            "epsilon_od": float(source_instance["chance_constraints"]["epsilon_od"]),
            "epsilon_sp": float(source_instance["chance_constraints"]["epsilon_sp"]),
            "rho": float(source_instance["chance_constraints"]["rho"]),
            "kappa": float(kappa),
            "lambda_migration": float(lambda_migration),
            "objective_type": objective_type,
            "energy_idle": float(energy_idle),
            "energy_cpu": float(energy_cpu),
            "energy_migration": float(energy_migration),
        },
        "scenarios": scenarios,
        "active_periods": {
            "on_demand": active_periods(scenario_time_series, "on_demand"),
            "spot": active_periods(scenario_time_series, "spot"),
        },
        "demands": {
            "on_demand": demand_records(scenario_time_series, "on_demand"),
            "spot": demand_records(scenario_time_series, "spot"),
        },
        "batch_jobs": build_batch_jobs(batch_jobs, batch_demands, capacity, kappa),
        "notes": [
            "On-demand and spot scenario demands are inherited from scenario_time_series.csv.",
            "Batch one-slot fragments are aggregated back to parent workload ids.",
            "W_k is the maximum total parent batch demand over scenarios.",
        ],
    }


def main():
    args = parse_args()
    instance = build_instance(
        source_dir=args.source_dir,
        kappa=args.kappa,
        objective_type=args.objective_type,
        lambda_migration=args.lambda_migration,
        energy_idle=args.energy_idle,
        energy_cpu=args.energy_cpu,
        energy_migration=args.energy_migration,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "vm_type_instance.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(instance, file, indent=2)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
