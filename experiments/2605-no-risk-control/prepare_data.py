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
- W_k is the total original-trace average core usage for the parent workload.
- r_B[k] is the selected VM's vCPU, capped by the homogeneous server capacity.
- N_B[k] = ceil((1 + kappa) * W_k / r_B[k]).

The output JSON is intentionally simple and verbose.  model.py reads it
directly, so there is no hidden preprocessing inside the optimization code.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = EXPERIMENT_DIR.parent
REPO_ROOT = EXPERIMENTS_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from config_2605 import (  # noqa: E402
    default_output_instance_name,
    default_source_instance_name,
    first_not_none,
    load_2604_builder,
    load_config,
    model_parameters,
    resolve_path,
    resolve_workload_counts,
    write_config,
)
DEFAULT_CONFIG = EXPERIMENT_DIR / "config.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Build a compact JSON instance for VM type modeling (1).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--build-source", dest="build_source", action="store_true")
    parser.add_argument("--no-build-source", dest="build_source", action="store_false")
    parser.set_defaults(build_source=None)
    parser.add_argument("--kappa", type=float, default=None, help="Batch reservation slack ratio.")
    parser.add_argument("--objective-type", choices=["server_count", "energy"], default=None)
    parser.add_argument("--lambda-migration", type=float, default=None)
    parser.add_argument("--energy-idle", type=float, default=None)
    parser.add_argument("--energy-cpu", type=float, default=None)
    parser.add_argument("--energy-migration", type=float, default=None)
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
        reserved_cpu = min(float(group["vCPU"].max()), capacity)
        workload = float(group["avg_core_usage"].sum())
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


def validate_source_counts(source_dir, expected_counts):
    metadata = pd.read_csv(source_dir / "workload_metadata.csv")
    actual = metadata.groupby("workload_type").size().to_dict()
    for workload_type, expected in expected_counts.items():
        observed = int(actual.get(workload_type, 0))
        if observed != expected:
            raise ValueError(
                f"{source_dir} has {observed} {workload_type} workloads, "
                f"but config expects {expected}."
            )


def build_source_instance(config, source_dir, source_instance_name, counts, params):
    data_config = config.get("data", {})
    builder = load_2604_builder(REPO_ROOT)
    source_csv = resolve_path(data_config.get("source_csv"), REPO_ROOT)
    if source_csv is None:
        raise ValueError("data.source_csv is required when data.build_source is true.")

    builder.build_instance(
        source_csv=source_csv.resolve(),
        output_dir=source_dir.resolve(),
        instance_name=source_instance_name,
        seed=int(data_config.get("random_seed", 42)),
        scenario_count=int(data_config.get("scenario_count", 10)),
        on_demand_count=counts["on_demand"],
        spot_count=counts["spot"],
        batch_count=counts["batch"],
        max_vcpu=int(data_config.get("max_vcpu", 8)),
        min_avg_cpu=float(data_config.get("min_avg_cpu", 0.0)),
        server_capacity=float(data_config.get("server_capacity", params["server_capacity"] or 8.0)),
        epsilon_od=params["epsilon_od"],
        epsilon_sp=params["epsilon_sp"],
        rho=params["rho"],
        objective_type=params["objective_type"],
        lambda_migration=params["lambda_migration"],
        energy_idle=params["energy_idle"],
        energy_cpu=params["energy_cpu"],
        energy_migration=params["energy_migration"],
        scenario_seed=int(data_config.get("scenario_seed", 42)),
    )


def build_instance(
    source_dir,
    kappa,
    objective_type,
    lambda_migration,
    energy_idle,
    energy_cpu,
    energy_migration,
    epsilon_od=None,
    epsilon_sp=None,
    rho=None,
    capacity=None,
    big_m=None,
    server_count=None,
    config_metadata=None,
):
    with open(source_dir / "instance.json", "r", encoding="utf-8") as file:
        source_instance = json.load(file)

    metadata = pd.read_csv(source_dir / "workload_metadata.csv")
    scenario_time_series = pd.read_csv(source_dir / "scenario_time_series.csv")
    batch_jobs = pd.read_csv(source_dir / "batch_jobs.csv")
    batch_demands = pd.read_csv(source_dir / "batch_job_demands.csv")

    capacity = float(first_not_none(capacity, source_instance["server_capacity"]))
    scenarios = [
        {"id": row["scenario"], "probability": float(row["probability"])}
        for row in source_instance["scenarios"]
    ]

    return {
        "name": "notion_vm_type_modeling_1_from_" + source_instance["instance_name"],
        "source_instance": source_instance["instance_name"],
        "sets": {
            "servers": list(range(int(first_not_none(server_count, source_instance["num_servers"])))),
            "times": [int(value) for value in source_instance["time_periods"]],
            "on_demand": metadata.loc[metadata["workload_type"] == "on_demand", "workload_id"].tolist(),
            "spot": metadata.loc[metadata["workload_type"] == "spot", "workload_id"].tolist(),
            "batch": sorted(batch_jobs["parent_workload_id"].unique().tolist()),
            "scenarios": [scenario["id"] for scenario in scenarios],
        },
        "parameters": {
            "capacity": capacity,
            "big_m": float(first_not_none(big_m, source_instance["big_m"])),
            "epsilon_od": float(first_not_none(epsilon_od, source_instance["chance_constraints"]["epsilon_od"])),
            "epsilon_sp": float(first_not_none(epsilon_sp, source_instance["chance_constraints"]["epsilon_sp"])),
            "rho": float(first_not_none(rho, source_instance["chance_constraints"]["rho"])),
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
        "config": config_metadata or {},
        "notes": [
            "On-demand and spot scenario demands are inherited from scenario_time_series.csv.",
            "Batch one-slot fragments are aggregated back to parent workload ids.",
            "W_k is the sum of original average core usage over the parent batch trace.",
        ],
    }


def main():
    args = parse_args()
    config = load_config(args.config)
    data_config = config.get("data", {})
    params = model_parameters(config)
    counts = resolve_workload_counts(data_config)

    scenario_count = int(data_config.get("scenario_count", 10))
    server_capacity = float(data_config.get("server_capacity", params["server_capacity"] or 8.0))
    source_instance_name = data_config.get("source_instance_name") or default_source_instance_name(
        counts,
        scenario_count,
        server_capacity,
    )
    output_instance_name = data_config.get("output_instance_name") or default_output_instance_name(
        counts,
        server_capacity,
    )
    source_dir = args.source_dir or resolve_path(
        data_config.get("source_dir"),
        REPO_ROOT,
        REPO_ROOT / "data" / "processed" / "2604-chance-2sp-toy" / source_instance_name,
    )
    output_dir = args.output_dir or resolve_path(
        data_config.get("output_dir"),
        REPO_ROOT,
        REPO_ROOT / "data" / "processed" / EXPERIMENT_DIR.name / output_instance_name,
    )
    build_source = first_not_none(args.build_source, data_config.get("build_source", False))

    if build_source:
        build_source_instance(config, source_dir, source_instance_name, counts, params)
        print(f"Prepared source 2604 instance at {source_dir}")
    elif not source_dir.exists():
        raise FileNotFoundError(f"{source_dir} does not exist. Set data.build_source=true or pass --build-source.")

    if data_config.get("strict_source_match", True):
        validate_source_counts(source_dir, counts)

    instance = build_instance(
        source_dir=source_dir,
        kappa=first_not_none(args.kappa, params["kappa"]),
        objective_type=first_not_none(args.objective_type, params["objective_type"]),
        lambda_migration=first_not_none(args.lambda_migration, params["lambda_migration"]),
        energy_idle=first_not_none(args.energy_idle, params["energy_idle"]),
        energy_cpu=first_not_none(args.energy_cpu, params["energy_cpu"]),
        energy_migration=first_not_none(args.energy_migration, params["energy_migration"]),
        epsilon_od=params["epsilon_od"],
        epsilon_sp=params["epsilon_sp"],
        rho=params["rho"],
        capacity=params["server_capacity"],
        big_m=params["big_m"],
        server_count=params["server_count"],
        config_metadata={
            "config_file": str(args.config),
            "source_instance_name": source_instance_name,
            "output_instance_name": output_instance_name,
            "workload_counts": counts,
        },
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "vm_type_instance.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(instance, file, indent=2)
    write_config(output_dir / "config_used.json", config)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
