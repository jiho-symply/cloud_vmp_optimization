from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


T5_US = 300 * 1_000_000
CPU_P100_AUDIT_STATUS = "cpu_p100_with_representative_capacity_vm_drop_no_clipping"

REQUIRED_FILES = [
    "metadata.json",
    "scaling_diagnostics.json",
    "model_params.json",
    "servers.csv",
    "vm_requests.csv",
    "vm_usage_5min_scenarios.csv",
    "vm_usage_hourly_scenarios.csv",
    "spot_preemption_scenarios.csv",
    "batch_families.csv",
    "batch_workload.csv",
    "energy_scenarios.csv",
    "scenario_probabilities.csv",
]


def _read_csv(output_dir: Path, name: str) -> pd.DataFrame:
    path = output_dir / name
    if not path.exists():
        raise FileNotFoundError(f"missing required output file: {path}")
    return pd.read_csv(path)


def _assert_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _finite_numeric(df: pd.DataFrame, columns: list[str], name: str) -> None:
    values = df[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values in {columns}")


def _validated_boolean(series: pd.Series, label: str) -> pd.Series:
    """Parse a CSV boolean without treating arbitrary non-empty strings as true."""

    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.map(
        lambda value: value.strip().lower() if isinstance(value, str) else value
    )
    mapping = {
        True: True,
        False: False,
        1: True,
        0: False,
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    parsed = normalized.map(mapping)
    if parsed.isna().any():
        raise ValueError(f"{label} must contain only boolean values")
    return parsed.astype(bool)


def _validated_coverage_us(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    values = numeric.to_numpy(dtype=float)
    if (
        not np.isfinite(values).all()
        or not np.equal(values, np.floor(values)).all()
        or (values <= 0).any()
        or (values > T5_US).any()
    ):
        raise ValueError(
            f"vm_usage_5min_scenarios.csv coverage_us must contain finite integer "
            f"values in (0, {T5_US}]"
        )
    return numeric.astype(np.int64)


def _nested(mapping: dict[str, object], path: tuple[str, ...]) -> object:
    value: object = mapping
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(
                "scaling_diagnostics.json is missing " + ".".join(path)
            )
        value = value[key]
    return value


def _finite_positive_json_number(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite and positive") from exc
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return number


def _summary_markdown(summary: dict[str, object]) -> str:
    class_counts = summary["vm_count_by_class"]
    class_rows = "\n".join(f"| {klass} | {count} |" for klass, count in class_counts.items())
    return f"""# Google 2019 VM-like Toy Instance Summary

This is a VM-like toy dataset derived from Google ClusterData 2019 Borg instance traces, not a real public-cloud VM trace.

## VM Counts by Class

| class | count |
|---|---:|
{class_rows}

## Aggregate Checks

| metric | value |
|---|---:|
| usage rows | {summary["num_usage_rows"]} |
| average lifetime (5-minute periods) | {summary["avg_lifetime_t5"]:.2f} |
| p50 q_cpu | {summary["p50_q_cpu"]:.6f} |
| p95 q_cpu | {summary["p95_q_cpu"]:.6f} |
| p50 q_mem | {summary["p50_q_mem"]:.6f} |
| p95 q_mem | {summary["p95_q_mem"]:.6f} |
| batch families | {summary["num_batch_families"]} |
| servers | {summary["num_servers"]} |
| scenarios | {summary["num_scenarios"]} |
| peak hourly CPU demand / server CPU capacity | {summary["peak_hourly_cpu_demand_over_capacity"]:.6f} |
| peak hourly MEM demand / server MEM capacity | {summary["peak_hourly_mem_demand_over_capacity"]:.6f} |
"""


def validate_output_dir(output_dir: str | Path, write_summary: bool = False) -> dict[str, object]:
    output_dir = Path(output_dir)
    for name in REQUIRED_FILES:
        if not (output_dir / name).exists():
            raise FileNotFoundError(f"missing required output file: {output_dir / name}")

    vm_requests = _read_csv(output_dir, "vm_requests.csv")
    usage_5min = _read_csv(output_dir, "vm_usage_5min_scenarios.csv")
    usage_hourly = _read_csv(output_dir, "vm_usage_hourly_scenarios.csv")
    spot = _read_csv(output_dir, "spot_preemption_scenarios.csv")
    batch_families = _read_csv(output_dir, "batch_families.csv")
    servers = _read_csv(output_dir, "servers.csv")
    energy = _read_csv(output_dir, "energy_scenarios.csv")
    scenario_probabilities = _read_csv(output_dir, "scenario_probabilities.csv")
    _ = _read_csv(output_dir, "batch_workload.csv")
    with (output_dir / "metadata.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    with (output_dir / "scaling_diagnostics.json").open("r", encoding="utf-8") as f:
        scaling_diagnostics = json.load(f)
    with (output_dir / "model_params.json").open("r", encoding="utf-8") as f:
        model_params = json.load(f)

    _assert_columns(
        vm_requests,
        [
            "vm_id",
            "stable_episode_key",
            "collection_id",
            "instance_index",
            "episode_index",
            "episode_start_time_us",
            "episode_end_time_us",
            "termination_reason",
            "horizon_censored",
            "class",
            "arrival_t5",
            "departure_t5",
            "q_cpu",
            "q_mem",
            "resource_request_cpu",
            "max_cpu_usage",
            "arrival_state_ambiguous",
            "scheduler_state_ambiguous",
            "cpu_p100_gt_one",
        ],
        "vm_requests.csv",
    )
    _assert_columns(
        usage_5min,
        [
            "scenario_id",
            "vm_id",
            "t5_day",
            "t_hour",
            "cpu_usage",
            "mem_usage",
            "coverage_us",
        ],
        "vm_usage_5min_scenarios.csv",
    )
    _assert_columns(usage_hourly, ["scenario_id", "vm_id", "t_hour", "cpu_usage", "mem_usage"], "vm_usage_hourly_scenarios.csv")
    _assert_columns(servers, ["server_id", "C_cpu", "C_mem"], "servers.csv")
    _assert_columns(
        energy,
        [
            "scenario_id",
            "t_hour",
            "day_ahead_price",
            "real_time_price",
            "sell_price",
            "renewable_generation",
            "ess_charge_efficiency",
            "ess_discharge_efficiency",
        ],
        "energy_scenarios.csv",
    )
    _assert_columns(scenario_probabilities, ["scenario_id", "probability"], "scenario_probabilities.csv")

    _finite_numeric(usage_5min, ["cpu_usage", "mem_usage"], "vm_usage_5min_scenarios.csv")
    _finite_numeric(usage_hourly, ["cpu_usage", "mem_usage"], "vm_usage_hourly_scenarios.csv")
    _finite_numeric(
        vm_requests,
        [
            "q_cpu",
            "q_mem",
            "resource_request_cpu",
            "resource_request_mem",
            "max_cpu_usage",
            "max_mem_usage",
        ],
        "vm_requests.csv",
    )
    _finite_numeric(servers, ["C_cpu", "C_mem"], "servers.csv")
    if (usage_5min[["cpu_usage", "mem_usage"]] < 0).any().any():
        raise ValueError("vm_usage_5min_scenarios.csv contains negative CPU or memory values")
    if (usage_hourly[["cpu_usage", "mem_usage"]] < 0).any().any():
        raise ValueError("vm_usage_hourly_scenarios.csv contains negative CPU or memory values")
    if (vm_requests[["q_cpu", "q_mem"]] <= 0).any().any():
        raise ValueError("vm_requests.csv contains non-positive q_cpu or q_mem")
    if (vm_requests["q_cpu"] + 1e-12 < vm_requests["resource_request_cpu"]).any():
        raise ValueError("vm_requests.csv violates q_cpu >= resource_request_cpu")
    if (vm_requests["q_cpu"] + 1e-12 < vm_requests["max_cpu_usage"]).any():
        raise ValueError("vm_requests.csv violates q_cpu >= max_cpu_usage")
    if (vm_requests["q_mem"] + 1e-12 < vm_requests["resource_request_mem"]).any():
        raise ValueError("vm_requests.csv violates q_mem >= resource_request_mem")
    if (vm_requests["q_mem"] + 1e-12 < vm_requests["max_mem_usage"]).any():
        raise ValueError("vm_requests.csv violates q_mem >= max_mem_usage")

    configured = vm_requests.set_index("vm_id")[["q_cpu", "q_mem"]]
    for frame, label in (
        (usage_5min, "vm_usage_5min_scenarios.csv"),
        (usage_hourly, "vm_usage_hourly_scenarios.csv"),
    ):
        synthetic = frame.loc[frame["scenario_id"].gt(0)].join(
            configured,
            on="vm_id",
            how="left",
            validate="many_to_one",
        )
        if synthetic[["q_cpu", "q_mem"]].isna().any().any():
            raise ValueError(f"{label} contains synthetic usage for an unknown VM")
        if (
            synthetic["cpu_usage"].gt(synthetic["q_cpu"] + 1e-12).any()
            or synthetic["mem_usage"].gt(synthetic["q_mem"] + 1e-12).any()
        ):
            raise ValueError(f"{label} contains synthetic CPU or memory usage above q")

    usage_5min["coverage_us"] = _validated_coverage_us(usage_5min["coverage_us"])
    if "coverage_ratio" in usage_5min.columns:
        coverage_ratio = pd.to_numeric(usage_5min["coverage_ratio"], errors="coerce").to_numpy(float)
        expected_ratio = usage_5min["coverage_us"].to_numpy(float) / T5_US
        if (
            not np.isfinite(coverage_ratio).all()
            or not np.allclose(coverage_ratio, expected_ratio, rtol=0.0, atol=1e-12)
        ):
            raise ValueError(
                "vm_usage_5min_scenarios.csv coverage_ratio is inconsistent with coverage_us"
            )

    scenario_zero = usage_5min.loc[usage_5min["scenario_id"].eq(0)]
    if scenario_zero.empty:
        raise ValueError("vm_usage_5min_scenarios.csv is missing canonical scenario 0")
    duplicate_scenario_zero = scenario_zero.duplicated(["vm_id", "t5_day"], keep=False)
    if duplicate_scenario_zero.any():
        examples = (
            scenario_zero.loc[duplicate_scenario_zero, ["vm_id", "t5_day"]]
            .drop_duplicates()
            .head(5)
            .to_dict("records")
        )
        raise ValueError(
            "canonical scenario 0 must have one row per (vm_id, t5_day); "
            f"duplicate examples={examples}"
        )

    arrival_ambiguous = _validated_boolean(
        vm_requests["arrival_state_ambiguous"],
        "vm_requests.csv arrival_state_ambiguous",
    )
    scheduler_ambiguous = _validated_boolean(
        vm_requests["scheduler_state_ambiguous"],
        "vm_requests.csv scheduler_state_ambiguous",
    )
    if arrival_ambiguous.any() or scheduler_ambiguous.any():
        bad_ids = vm_requests.loc[
            arrival_ambiguous | scheduler_ambiguous, "vm_id"
        ].astype(str).head(10).tolist()
        raise ValueError(
            "strict preprocessing output contains unresolved arrival/scheduler state "
            f"ambiguity; vm_ids={bad_ids}"
        )

    p100_outlier_flags = _validated_boolean(
        vm_requests["cpu_p100_gt_one"],
        "vm_requests.csv cpu_p100_gt_one",
    )

    if vm_requests["stable_episode_key"].duplicated().any():
        raise ValueError("vm_requests.csv contains duplicate stable_episode_key values")
    if vm_requests[["collection_id", "instance_index", "episode_index"]].duplicated().any():
        raise ValueError("vm_requests.csv contains duplicate lifecycle episode keys")
    allowed_termination_reasons = {
        "EVICT",
        "FAIL",
        "FINISH",
        "KILL",
        "LOST",
        "HORIZON_CENSORED",
    }
    invalid_termination = ~vm_requests["termination_reason"].astype(str).isin(
        allowed_termination_reasons
    )
    if invalid_termination.any():
        raise ValueError("vm_requests.csv contains an unsupported termination_reason")
    horizon_censored = _validated_boolean(
        vm_requests["horizon_censored"], "vm_requests.csv horizon_censored"
    )
    if not horizon_censored.eq(
        vm_requests["termination_reason"].astype(str).eq("HORIZON_CENSORED")
    ).all():
        raise ValueError("horizon_censored is inconsistent with termination_reason")
    episode_start = pd.to_numeric(
        vm_requests["episode_start_time_us"], errors="coerce"
    )
    episode_end = pd.to_numeric(vm_requests["episode_end_time_us"], errors="coerce")
    if episode_start.isna().any() or episode_end.isna().any() or not episode_start.lt(
        episode_end
    ).all():
        raise ValueError("vm_requests.csv contains invalid episode time bounds")

    server_cpu = pd.to_numeric(servers["C_cpu"], errors="coerce").to_numpy(float)
    server_mem = pd.to_numeric(servers["C_mem"], errors="coerce").to_numpy(float)
    if not np.equal(server_cpu, 1.0).all() or not np.equal(server_mem, 1.0).all():
        raise ValueError("servers.csv must have exactly C_cpu=1.0 and C_mem=1.0 for every server")

    if scaling_diagnostics.get("automatic_utilization_calibration_applied") is not False:
        raise ValueError(
            "scaling_diagnostics.json must set automatic_utilization_calibration_applied=false"
        )
    cpu_divisor = _finite_positive_json_number(
        _nested(scaling_diagnostics, ("unit_conversion_factors", "cpu_divisor")),
        "unit_conversion_factors.cpu_divisor",
    )
    mem_divisor = _finite_positive_json_number(
        _nested(scaling_diagnostics, ("unit_conversion_factors", "mem_divisor")),
        "unit_conversion_factors.mem_divisor",
    )
    representative_cpu = _finite_positive_json_number(
        _nested(scaling_diagnostics, ("representative_machine_capacity_raw", "cpu")),
        "representative_machine_capacity_raw.cpu",
    )
    representative_mem = _finite_positive_json_number(
        _nested(scaling_diagnostics, ("representative_machine_capacity_raw", "mem")),
        "representative_machine_capacity_raw.mem",
    )
    if cpu_divisor != representative_cpu or mem_divisor != representative_mem:
        raise ValueError(
            "unit conversion divisors must equal the representative joint machine capacity"
        )

    p100_audit = _nested(
        scaling_diagnostics, ("cpu_p100_audit",)
    )
    if not isinstance(p100_audit, dict):
        raise ValueError("cpu_p100_audit must be a JSON object")
    expected_status = CPU_P100_AUDIT_STATUS
    if p100_audit.get("status") != expected_status:
        raise ValueError(
            "CPU maximum must use distribution p100 and drop representative-capacity-oversized "
            "VMs without clipping"
        )
    outlier_count = int(p100_outlier_flags.sum())
    if p100_audit.get("selected_vm_count") != outlier_count:
        raise ValueError(
            "cpu_p100_audit.selected_vm_count does not match vm_requests.csv"
        )
    converted_q_gt_one_count = int(vm_requests["q_cpu"].gt(1.0 + 1e-12).sum())
    if p100_audit.get("converted_q_cpu_gt_one_count") != converted_q_gt_one_count:
        raise ValueError(
            "cpu_p100_audit.converted_q_cpu_gt_one_count does not match "
            "vm_requests.csv"
        )
    converted_q_mem_gt_one_count = int(vm_requests["q_mem"].gt(1.0 + 1e-12).sum())
    if converted_q_gt_one_count or converted_q_mem_gt_one_count:
        raise ValueError(
            "vm_requests.csv contains q above one representative server; oversized VMs must be dropped"
        )
    capacity_filter = scaling_diagnostics.get("representative_capacity_vm_filter")
    if not isinstance(capacity_filter, dict):
        raise ValueError("scaling_diagnostics.json must record representative_capacity_vm_filter")
    if capacity_filter.get("configured_resource_clipping_applied") is not False:
        raise ValueError("representative-capacity filtering must not clip configured resources")
    if capacity_filter.get("sampling_replacement_applied") is not False:
        raise ValueError("representative-capacity filtering must not replacement-sample VMs")
    try:
        filter_source_count = int(capacity_filter["source_vm_count"])
        filter_retained_count = int(capacity_filter["retained_vm_count"])
        filter_dropped_count = int(capacity_filter["dropped_vm_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "representative_capacity_vm_filter must record source/retained/dropped VM counts"
        ) from exc
    if filter_retained_count != len(vm_requests):
        raise ValueError(
            "representative_capacity_vm_filter.retained_vm_count does not match vm_requests.csv"
        )
    if filter_source_count != filter_retained_count + filter_dropped_count:
        raise ValueError("representative-capacity filter VM counts are inconsistent")
    if outlier_count:
        raw_outlier_max = _finite_positive_json_number(
            _nested(
                scaling_diagnostics,
                (
                    "cpu_p100_audit",
                    "raw_cpu_p100_summary",
                    "max",
                ),
            ),
            "cpu_p100_audit.raw_cpu_p100_summary.max",
        )
        if raw_outlier_max <= 1.0:
            raise ValueError("cpu_p100 audit maximum must exceed raw threshold 1.0")
    if not usage_5min["t5_day"].between(0, 287).all():
        raise ValueError("vm_usage_5min_scenarios.csv contains t5_day outside 0..287")
    if not usage_5min["t_hour"].between(0, 23).all():
        raise ValueError("vm_usage_5min_scenarios.csv contains t_hour outside 0..23")
    if not usage_hourly["t_hour"].between(0, 23).all():
        raise ValueError("vm_usage_hourly_scenarios.csv contains t_hour outside 0..23")
    if usage_hourly[["scenario_id", "vm_id", "t_hour"]].duplicated().any():
        raise ValueError("vm_usage_hourly_scenarios.csv must have one row per scenario_id, vm_id, t_hour")
    if not energy["t_hour"].between(0, 23).all():
        raise ValueError("energy_scenarios.csv contains t_hour outside 0..23")
    if not (vm_requests["arrival_t5"] < vm_requests["departure_t5"]).all():
        raise ValueError("vm_requests.csv contains VM rows with arrival_t5 >= departure_t5")

    request_ids = set(vm_requests["vm_id"])
    missing_usage_ids = set(usage_5min["vm_id"]) - request_ids
    if missing_usage_ids:
        raise ValueError(f"vm_usage_5min_scenarios.csv references unknown vm_id values: {sorted(missing_usage_ids)[:5]}")
    missing_hourly_usage_ids = set(usage_hourly["vm_id"]) - request_ids
    if missing_hourly_usage_ids:
        raise ValueError(
            f"vm_usage_hourly_scenarios.csv references unknown vm_id values: {sorted(missing_hourly_usage_ids)[:5]}"
        )

    scenario_ids = sorted(usage_hourly["scenario_id"].unique().tolist())
    probability_ids = sorted(scenario_probabilities["scenario_id"].unique().tolist())
    if probability_ids != scenario_ids:
        raise ValueError("scenario_probabilities.csv scenario IDs must match hourly usage scenario IDs")
    if abs(float(scenario_probabilities["probability"].sum()) - 1.0) > 1e-9:
        raise ValueError("scenario probabilities must sum to 1")
    for key in ["alpha", "epsilon", "soc_init"]:
        if key not in model_params:
            raise ValueError(f"model_params.json missing {key}")

    spot_ids = vm_requests.loc[vm_requests["class"] == "spot", "vm_id"].tolist()
    if spot_ids:
        _assert_columns(spot, ["scenario_id", "vm_id", "t5_day", "t_hour", "active", "preempted"], "spot_preemption_scenarios.csv")
        spot_pairs = spot[["scenario_id", "vm_id"]].drop_duplicates()
        expected_pairs = {(scenario_id, vm_id) for scenario_id in scenario_ids for vm_id in spot_ids}
        actual_pairs = set(map(tuple, spot_pairs.to_numpy()))
        missing_pairs = expected_pairs - actual_pairs
        if missing_pairs:
            raise ValueError(f"spot_preemption_scenarios.csv is missing spot VM scenario paths: {sorted(missing_pairs)[:5]}")
        if not spot["t5_day"].between(0, 287).all():
            raise ValueError("spot_preemption_scenarios.csv contains t5_day outside 0..287")
        if not spot["active"].isin([0, 1]).all():
            raise ValueError("spot_preemption_scenarios.csv active must be 0 or 1")
    elif not spot.empty:
        raise ValueError("spot_preemption_scenarios.csv has rows but vm_requests.csv has no spot VMs")

    energy_counts = energy.groupby("scenario_id")["t_hour"].nunique()
    if not energy_counts.eq(24).all():
        raise ValueError("every energy scenario must have exactly 24 hourly rows")
    if (energy[["day_ahead_price", "sell_price", "renewable_generation"]] < 0).any().any():
        raise ValueError("energy_scenarios.csv contains negative day-ahead/sell/renewable values")

    total_server_cpu_capacity = float(servers["C_cpu"].sum())
    total_server_mem_capacity = float(servers["C_mem"].sum())
    peak_hourly_cpu = float(usage_hourly.groupby(["scenario_id", "t_hour"])["cpu_usage"].sum().max())
    peak_hourly_mem = float(usage_hourly.groupby(["scenario_id", "t_hour"])["mem_usage"].sum().max())
    peak_cpu_over_capacity = peak_hourly_cpu / total_server_cpu_capacity if total_server_cpu_capacity > 0 else float("inf")
    peak_mem_over_capacity = peak_hourly_mem / total_server_mem_capacity if total_server_mem_capacity > 0 else float("inf")

    class_counts = vm_requests["class"].value_counts().sort_index().to_dict()
    summary = {
        "vm_count_by_class": {str(k): int(v) for k, v in class_counts.items()},
        "num_usage_rows": int(len(usage_5min)),
        "num_hourly_usage_rows": int(len(usage_hourly)),
        "avg_lifetime_t5": float((vm_requests["departure_t5"] - vm_requests["arrival_t5"]).mean()),
        "p50_q_cpu": float(vm_requests["q_cpu"].quantile(0.50)),
        "p95_q_cpu": float(vm_requests["q_cpu"].quantile(0.95)),
        "p50_q_mem": float(vm_requests["q_mem"].quantile(0.50)),
        "p95_q_mem": float(vm_requests["q_mem"].quantile(0.95)),
        "num_batch_families": int(len(batch_families)),
        "num_servers": int(len(servers)),
        "num_scenarios": int(len(energy_counts)),
        "metadata_num_scenarios": int(metadata.get("num_scenarios", len(energy_counts))),
        "peak_hourly_cpu_demand_over_capacity": float(peak_cpu_over_capacity),
        "peak_hourly_mem_demand_over_capacity": float(peak_mem_over_capacity),
        "coverage_us": {
            "minimum": int(usage_5min["coverage_us"].min()),
            "maximum": int(usage_5min["coverage_us"].max()),
            "partial_row_count": int(usage_5min["coverage_us"].lt(T5_US).sum()),
        },
        "canonical_scenario_zero_key_count": int(len(scenario_zero)),
        "strict_event_state": {
            "arrival_state_ambiguous_count": int(arrival_ambiguous.sum()),
            "scheduler_state_ambiguous_count": int(scheduler_ambiguous.sum()),
        },
        "cpu_p100_audit": {
            "status": expected_status,
            "selected_vm_count": outlier_count,
            "converted_q_cpu_gt_one_count": converted_q_gt_one_count,
        },
        "automatic_utilization_calibration_applied": False,
        "unit_conversion_divisors": {
            "cpu": cpu_divisor,
            "mem": mem_divisor,
        },
    }

    if write_summary:
        (output_dir / "toy_instance_summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Google 2019 toy instance outputs.")
    parser.add_argument("--output_dir", type=Path, default=Path("data/processed/notion_toy_google2019_v1"))
    parser.add_argument("--write_summary", action="store_true", help="Write toy_instance_summary.md after validation.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = validate_output_dir(args.output_dir, write_summary=args.write_summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
