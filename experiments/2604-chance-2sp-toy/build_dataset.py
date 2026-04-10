import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
EXPERIMENT_NAME = EXPERIMENT_DIR.name
REPO_ROOT = EXPERIMENT_DIR.parents[1]
SOURCE_CSV = REPO_ROOT / "data" / "processed" / "2601-initial-toy-model" / "sample_vm_data.csv"
DEFAULT_COUNTS = {"on_demand": 10, "spot": 10, "batch": 10}

SERVER_CAPACITY = 8.0
MAX_VCPU = 8
MIN_AVG_CPU = 0.0
EPSILON_OD = 0.10
EPSILON_SP = 0.20
RHO = 0.80
LAMBDA_MIGRATION = 0.50


def default_instance_name(on_demand_count, spot_count, batch_count, scenario_count, server_capacity):
    total_vm = on_demand_count + spot_count + batch_count
    return (
        f"chance_2sp_toy_{total_vm}vm_"
        f"od{on_demand_count}_sp{spot_count}_bj{batch_count}_"
        f"sc{scenario_count}_cap{int(server_capacity)}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="chance-constrained 2SP toy 실험용 instance를 생성합니다.")
    parser.add_argument("--source-csv", type=Path, default=SOURCE_CSV)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--instance-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario-count", type=int, default=10)
    parser.add_argument("--on-demand-count", type=int, default=DEFAULT_COUNTS["on_demand"])
    parser.add_argument("--spot-count", type=int, default=DEFAULT_COUNTS["spot"])
    parser.add_argument("--batch-count", type=int, default=DEFAULT_COUNTS["batch"])
    parser.add_argument("--max-vcpu", type=int, default=MAX_VCPU)
    parser.add_argument("--min-avg-cpu", type=float, default=MIN_AVG_CPU)
    parser.add_argument("--server-capacity", type=float, default=SERVER_CAPACITY)
    parser.add_argument("--epsilon-od", type=float, default=EPSILON_OD)
    parser.add_argument("--epsilon-sp", type=float, default=EPSILON_SP)
    parser.add_argument("--rho", type=float, default=RHO)
    parser.add_argument("--lambda-migration", type=float, default=LAMBDA_MIGRATION)
    parser.add_argument("--scenario-seed", type=int, default=42)
    return parser.parse_args()


def make_scenarios(scenario_count):
    probability = 1.0 / scenario_count
    return [
        {"scenario": f"scen_{index:02d}", "probability": probability}
        for index in range(1, scenario_count + 1)
    ]


def triangular_sample(min_value, avg_value, max_value, rng):
    if abs(max_value - min_value) <= 1e-12:
        return float(min_value)
    mode = 3.0 * avg_value - min_value - max_value
    mode = min(max(mode, min_value), max_value)
    return float(rng.triangular(min_value, mode, max_value))


def deterministic_triangular_sample(min_value, avg_value, max_value, scenario_seed, vm_id, time_value, scenario_name):
    if abs(max_value - min_value) <= 1e-12:
        return float(min_value)

    mode = 3.0 * avg_value - min_value - max_value
    mode = min(max(mode, min_value), max_value)
    low = float(min_value)
    high = float(max_value)
    if abs(high - low) <= 1e-12:
        return low

    key = f"{scenario_seed}|{vm_id}|{int(time_value)}|{scenario_name}"
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    u = (int.from_bytes(digest, "big") + 0.5) / float(2**64)

    if abs(mode - low) <= 1e-12:
        return high - math.sqrt((1.0 - u) * (high - low) * (high - mode))
    if abs(mode - high) <= 1e-12:
        return low + math.sqrt(u * (high - low) * (mode - low))

    split = (mode - low) / (high - low)
    if u < split:
        return low + math.sqrt(u * (high - low) * (mode - low))
    return high - math.sqrt((1.0 - u) * (high - low) * (high - mode))


def label_workloads(frame, workload_type, prefix):
    labeled = frame.copy().reset_index(drop=True)
    labeled["workload_type"] = workload_type
    labeled["workload_id"] = [f"{prefix}_{index:02d}" for index in range(len(labeled))]
    return labeled


def load_vm_pool(source_csv, max_vcpu, min_avg_cpu):
    vm_trace = pd.read_csv(source_csv)
    vm_trace = vm_trace.loc[vm_trace["hour"] < 24].copy()

    vm_summary = (
        vm_trace.groupby("vm_id", as_index=False)
        .agg(
            hours=("hour", "count"),
            start_time=("hour", "min"),
            end_time=("hour", "max"),
            vm_category=("vm_category", "first"),
            vCPU=("vCPU", "first"),
            avg_cpu_mean=("avg_cpu", "mean"),
            avg_mean=("avg_core_usage", "mean"),
            avg_peak=("avg_core_usage", "max"),
            max_peak=("max_core_usage", "max"),
        )
    )

    vm_summary = vm_summary.loc[
        vm_summary["vm_category"].isin(["Interactive", "Delay-insensitive"])
        & (vm_summary["vCPU"] <= max_vcpu)
        & (vm_summary["avg_cpu_mean"] >= min_avg_cpu)
    ].copy()
    vm_summary["selection_score"] = vm_summary["avg_peak"] + 0.25 * vm_summary["max_peak"]
    return vm_trace, vm_summary


def sample_category(frame, count, rng, category):
    candidates = frame.loc[frame["vm_category"] == category].copy()
    if len(candidates) < count:
        raise ValueError(f"Not enough {category} VMs. Needed {count}, found {len(candidates)}.")
    if count == 0:
        return candidates.iloc[0:0].copy()
    picked = rng.choice(candidates.index.to_numpy(), size=count, replace=False)
    return candidates.loc[picked].sort_values("vm_id").reset_index(drop=True)


def select_workload_metadata(vm_summary, on_demand_count, spot_count, batch_count, seed):
    rng = np.random.default_rng(seed)
    selected_on_demand = sample_category(vm_summary, on_demand_count, rng, "Interactive")
    selected_delay = sample_category(vm_summary, spot_count + batch_count, rng, "Delay-insensitive")

    selected_delay = selected_delay.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    selected_spot = selected_delay.iloc[:spot_count].copy()
    selected_batch = selected_delay.iloc[spot_count : spot_count + batch_count].copy()

    workload_metadata = pd.concat(
        [
            label_workloads(selected_on_demand, "on_demand", "od"),
            label_workloads(selected_spot, "spot", "sp"),
            label_workloads(selected_batch, "batch", "bj"),
        ],
        ignore_index=True,
    )
    return workload_metadata.sort_values(["workload_type", "workload_id"]).reset_index(drop=True)


def build_selected_traces(vm_trace, workload_metadata):
    selected_trace = vm_trace.merge(
        workload_metadata[
            ["vm_id", "workload_id", "workload_type", "selection_score", "start_time", "end_time"]
        ],
        on="vm_id",
        how="inner",
    ).rename(columns={"hour": "time", "vm_category": "source_vm_category"})
    return selected_trace.sort_values(["workload_type", "workload_id", "time"]).reset_index(drop=True)


def build_time_series_demands(selected_trace, scenarios, scenario_seed):
    rows = []
    time_series_trace = selected_trace.loc[selected_trace["workload_type"].isin(["on_demand", "spot"])].copy()

    for row in time_series_trace.itertuples(index=False):
        for scenario in scenarios:
            rows.append(
                {
                    "scenario": scenario["scenario"],
                    "probability": scenario["probability"],
                    "workload_id": row.workload_id,
                    "workload_type": row.workload_type,
                    "time": int(row.time),
                    "demand": deterministic_triangular_sample(
                        row.min_core_usage,
                        row.avg_core_usage,
                        row.max_core_usage,
                        scenario_seed,
                        row.vm_id,
                        row.time,
                        scenario["scenario"],
                    ),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["scenario", "probability", "workload_id", "workload_type", "time", "demand"]
        )
    return pd.DataFrame(rows)


def build_batch_jobs(selected_trace):
    batch_trace = selected_trace.loc[selected_trace["workload_type"] == "batch"].copy()
    if batch_trace.empty:
        return pd.DataFrame(
            columns=[
                "batch_job_id",
                "parent_workload_id",
                "source_time",
                "vm_id",
                "vCPU",
                "min_core_usage",
                "avg_core_usage",
                "max_core_usage",
            ]
        )

    batch_trace["batch_job_id"] = batch_trace.apply(
        lambda row: f"{row['workload_id']}_t{int(row['time']):02d}",
        axis=1,
    )
    return batch_trace[
        [
            "batch_job_id",
            "workload_id",
            "time",
            "vm_id",
            "vCPU",
            "min_core_usage",
            "avg_core_usage",
            "max_core_usage",
        ]
    ].rename(columns={"workload_id": "parent_workload_id", "time": "source_time"}).reset_index(drop=True)


def build_batch_demands(batch_jobs, scenarios, scenario_seed):
    rows = []
    for row in batch_jobs.itertuples(index=False):
        for scenario in scenarios:
            rows.append(
                {
                    "scenario": scenario["scenario"],
                    "probability": scenario["probability"],
                    "batch_job_id": row.batch_job_id,
                    "parent_workload_id": row.parent_workload_id,
                    "source_time": int(row.source_time),
                    "demand": deterministic_triangular_sample(
                        row.min_core_usage,
                        row.avg_core_usage,
                        row.max_core_usage,
                        scenario_seed,
                        row.vm_id,
                        row.source_time,
                        scenario["scenario"],
                    ),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["scenario", "probability", "batch_job_id", "parent_workload_id", "source_time", "demand"]
        )
    return pd.DataFrame(rows)


def summarize_scenarios(scenario_time_series, batch_job_demands):
    scenario_names = (
        pd.concat(
            [
                scenario_time_series["scenario"] if not scenario_time_series.empty else pd.Series(dtype=str),
                batch_job_demands["scenario"] if not batch_job_demands.empty else pd.Series(dtype=str),
            ],
            ignore_index=True,
        )
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    base = pd.MultiIndex.from_product([scenario_names, list(range(24))], names=["scenario", "time"]).to_frame(index=False)

    if scenario_time_series.empty:
        pivot = base.copy()
        pivot["on_demand"] = 0.0
        pivot["spot"] = 0.0
    else:
        trace_totals = (
            scenario_time_series.groupby(["scenario", "time", "workload_type"], as_index=False)["demand"].sum()
        )
        pivot = trace_totals.pivot_table(
            index=["scenario", "time"],
            columns="workload_type",
            values="demand",
            fill_value=0.0,
        ).reset_index()
        pivot = base.merge(pivot, on=["scenario", "time"], how="left")
        if "on_demand" not in pivot.columns:
            pivot["on_demand"] = 0.0
        else:
            pivot["on_demand"] = pivot["on_demand"].fillna(0.0)
        if "spot" not in pivot.columns:
            pivot["spot"] = 0.0
        else:
            pivot["spot"] = pivot["spot"].fillna(0.0)

    if batch_job_demands.empty:
        batch_totals = pd.DataFrame({"scenario": scenario_names, "batch_total_work": 0.0})
    else:
        batch_totals = batch_job_demands.groupby("scenario", as_index=False)["demand"].sum()
        batch_totals = batch_totals.rename(columns={"demand": "batch_total_work"})

    pivot = pivot.merge(batch_totals, on="scenario", how="left")
    pivot["batch_total_work"] = pivot["batch_total_work"].fillna(0.0)
    pivot["batch_average_per_slot"] = pivot["batch_total_work"] / 24.0
    pivot["equivalent_total_demand"] = pivot["on_demand"] + pivot["spot"] + pivot["batch_average_per_slot"]

    summary = (
        pivot.groupby("scenario", as_index=False)
        .agg(
            min_equivalent_demand=("equivalent_total_demand", "min"),
            mean_equivalent_demand=("equivalent_total_demand", "mean"),
            peak_equivalent_demand=("equivalent_total_demand", "max"),
            peak_time=("equivalent_total_demand", "idxmax"),
            peak_on_demand=("on_demand", "max"),
            peak_spot=("spot", "max"),
            batch_total_work=("batch_total_work", "first"),
            batch_average_per_slot=("batch_average_per_slot", "first"),
        )
    )
    summary["peak_time"] = summary["peak_time"].map(lambda index_value: int(pivot.loc[index_value, "time"]))
    return pivot, summary


def estimate_num_servers(scenario_summary, server_capacity):
    peak_trace_demand = float((scenario_summary["peak_on_demand"] + scenario_summary["peak_spot"]).max())
    average_batch_slot = float(scenario_summary["batch_average_per_slot"].max())
    return max(6, math.ceil((peak_trace_demand + average_batch_slot) / server_capacity) + 2)


def build_instance(
    source_csv,
    output_dir,
    instance_name,
    seed,
    scenario_count,
    on_demand_count,
    spot_count,
    batch_count,
    max_vcpu,
    min_avg_cpu,
    server_capacity,
    epsilon_od,
    epsilon_sp,
    rho,
    lambda_migration,
    scenario_seed,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    vm_trace, vm_summary = load_vm_pool(source_csv, max_vcpu=max_vcpu, min_avg_cpu=min_avg_cpu)
    workload_metadata = select_workload_metadata(
        vm_summary=vm_summary,
        on_demand_count=on_demand_count,
        spot_count=spot_count,
        batch_count=batch_count,
        seed=seed,
    )
    selected_trace = build_selected_traces(vm_trace, workload_metadata)

    scenarios = make_scenarios(scenario_count)
    scenario_time_series = build_time_series_demands(selected_trace, scenarios, scenario_seed)
    batch_jobs = build_batch_jobs(selected_trace)
    batch_job_demands = build_batch_demands(batch_jobs, scenarios, scenario_seed)
    scenario_detail, scenario_summary = summarize_scenarios(scenario_time_series, batch_job_demands)

    num_servers = estimate_num_servers(scenario_summary, server_capacity)
    peak_trace_demand = float((scenario_summary["peak_on_demand"] + scenario_summary["peak_spot"]).max())
    average_batch_slot = float(scenario_summary["batch_average_per_slot"].max())
    big_m = float(peak_trace_demand + average_batch_slot)

    mix_summary = (
        workload_metadata.groupby(["workload_type", "vCPU"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["workload_type", "vCPU"])
        .reset_index(drop=True)
    )

    instance = {
        "instance_name": instance_name,
        "seed": seed,
        "scenario_seed": scenario_seed,
        "source_csv": str(source_csv),
        "time_periods": list(range(24)),
        "server_capacity": server_capacity,
        "num_servers": num_servers,
        "big_m": big_m,
        "chance_constraints": {
            "epsilon_od": epsilon_od,
            "epsilon_sp": epsilon_sp,
            "rho": rho,
        },
        "objective": {"lambda_migration": lambda_migration},
        "scenario_sampling": "deterministic triangular(min, avg, max) keyed by (vm_id, time, scenario)",
        "batch_modeling": "split each original batch VM trace into one-slot batch jobs",
        "sampling_policy": {
            "pool_filter": f"vm_category in {{Interactive, Delay-insensitive}}, vCPU <= {max_vcpu}, avg_cpu_mean >= {min_avg_cpu}, hour < 24",
            "toy_sampling": "uniform over candidate VMs regardless of lifetime length within the horizon",
        },
        "counts": {
            "on_demand": on_demand_count,
            "spot": spot_count,
            "batch": batch_count,
        },
        "scenarios": scenarios,
        "files": {
            "workload_metadata": "workload_metadata.csv",
            "selected_vm_traces": "selected_vm_traces.csv",
            "scenario_time_series": "scenario_time_series.csv",
            "batch_jobs": "batch_jobs.csv",
            "batch_job_demands": "batch_job_demands.csv",
            "scenario_detail": "scenario_detail.csv",
            "scenario_summary": "scenario_summary.csv",
            "scenario_probabilities": "scenario_probabilities.csv",
            "mix_summary": "mix_summary.csv",
        },
    }

    workload_metadata.to_csv(output_dir / "workload_metadata.csv", index=False)
    selected_trace.to_csv(output_dir / "selected_vm_traces.csv", index=False)
    scenario_time_series.to_csv(output_dir / "scenario_time_series.csv", index=False)
    batch_jobs.to_csv(output_dir / "batch_jobs.csv", index=False)
    batch_job_demands.to_csv(output_dir / "batch_job_demands.csv", index=False)
    scenario_detail.to_csv(output_dir / "scenario_detail.csv", index=False)
    scenario_summary.to_csv(output_dir / "scenario_summary.csv", index=False)
    pd.DataFrame(scenarios).to_csv(output_dir / "scenario_probabilities.csv", index=False)
    mix_summary.to_csv(output_dir / "mix_summary.csv", index=False)

    with open(output_dir / "instance.json", "w", encoding="utf-8") as file:
        json.dump(instance, file, indent=2)

    return instance, workload_metadata, scenario_summary


def main():
    args = parse_args()
    total_vm = args.on_demand_count + args.spot_count + args.batch_count
    if total_vm <= 0:
        raise ValueError("Total VM count must be positive.")

    instance_name = args.instance_name or default_instance_name(
        args.on_demand_count,
        args.spot_count,
        args.batch_count,
        args.scenario_count,
        args.server_capacity,
    )
    output_dir = args.output_dir or (REPO_ROOT / "data" / "processed" / EXPERIMENT_NAME / instance_name)

    instance, workload_metadata, scenario_summary = build_instance(
        source_csv=args.source_csv.resolve(),
        output_dir=output_dir.resolve(),
        instance_name=instance_name,
        seed=args.seed,
        scenario_count=args.scenario_count,
        on_demand_count=args.on_demand_count,
        spot_count=args.spot_count,
        batch_count=args.batch_count,
        max_vcpu=args.max_vcpu,
        min_avg_cpu=args.min_avg_cpu,
        server_capacity=args.server_capacity,
        epsilon_od=args.epsilon_od,
        epsilon_sp=args.epsilon_sp,
        rho=args.rho,
        lambda_migration=args.lambda_migration,
        scenario_seed=args.scenario_seed,
    )

    print(f"instance 생성 위치: {output_dir}")
    print(f"instance 이름: {instance_name}")
    print(f"workload 개수: {instance['counts']}")
    print(f"서버 용량: {instance['server_capacity']}")
    print(f"서버 상한: {instance['num_servers']}")
    print(f"batch job 개수: {len(pd.read_csv(output_dir / 'batch_jobs.csv'))}")
    print("vCPU별 workload 구성:")
    print(workload_metadata.groupby(["workload_type", "vCPU"]).size().to_string())
    print("scenario별 등가 총수요 요약:")
    print(
        scenario_summary[["scenario", "mean_equivalent_demand", "peak_equivalent_demand", "batch_total_work"]]
        .round(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
