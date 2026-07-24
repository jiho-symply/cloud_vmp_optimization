from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from google2019_toy.build_toy_instance import (
    build_batch_outputs,
    build_energy_scenarios,
    build_hourly_usage_scenarios,
    build_model_params,
    build_scenario_probabilities,
    build_servers,
    build_spot_preemption_scenarios,
    build_usage_scenarios,
    build_vm_requests,
)
from google2019_toy.extract_bigquery import (
    build_collection_events_query,
    build_events_query,
    build_machine_events_query,
    build_usage_query,
    day_bounds_us,
)
from google2019_toy.validate_outputs import validate_output_dir


STEP_US = 300 * 1_000_000


def _usage_rows(
    collection_id: int,
    instance_index: int,
    cpu: float,
    mem: float,
    rows: int = 12,
    start_bucket: int = 0,
) -> list[dict[str, float | int]]:
    out = []
    for offset in range(rows):
        t5 = start_bucket + offset
        out.append(
            {
                "collection_id": collection_id,
                "instance_index": instance_index,
                "start_time": t5 * STEP_US,
                "end_time": (t5 + 1) * STEP_US,
                "cpu_usage": cpu + offset * 0.001,
                "mem_usage": mem + offset * 0.0005,
            }
        )
    return out


def test_build_vm_requests_computes_nominal_resources_and_class_rules() -> None:
    usage = pd.DataFrame(
        _usage_rows(100, 0, 0.10, 0.20, rows=16)
        + _usage_rows(200, 0, 0.05, 0.08, rows=12)
        + _usage_rows(300, 0, 0.04, 0.06, rows=12)
    )
    events = pd.DataFrame(
        [
            {
                "collection_id": 100,
                "instance_index": 0,
                "priority": 200,
                "resource_request_cpu": 0.25,
                "resource_request_mem": 0.30,
                "event_type": "SCHEDULE",
                "time": 0,
            },
            {
                "collection_id": 200,
                "instance_index": 0,
                "priority": 10,
                "resource_request_cpu": np.nan,
                "resource_request_mem": np.nan,
                "event_type": "SCHEDULE",
                "time": 0,
            },
            {
                "collection_id": 300,
                "instance_index": 0,
                "priority": 5,
                "resource_request_cpu": 0.03,
                "resource_request_mem": 0.02,
                "event_type": "SCHEDULE",
                "time": 0,
            },
        ]
    )
    collection_events = pd.DataFrame(
        [{"collection_id": 300, "scheduler": "batch", "collection_type": "batch"}]
    )

    vm_requests, observed = build_vm_requests(
        usage,
        events=events,
        collection_events=collection_events,
        max_instances=10,
        seed=7,
    )

    assert set(vm_requests["class"]) == {"on_demand", "spot", "batch_candidate"}
    assert (vm_requests["q_cpu"] > 0).all()
    assert (vm_requests["q_mem"] > 0).all()
    assert observed["t5_day"].between(0, 287).all()

    high = vm_requests.loc[vm_requests["collection_id"] == 100].iloc[0]
    assert high["class"] == "on_demand"
    assert high["q_cpu"] == pytest.approx(0.25)
    assert high["q_mem"] == pytest.approx(0.30)
    assert "priority" in high["class_rule"]

    low = vm_requests.loc[vm_requests["collection_id"] == 200].iloc[0]
    assert low["class"] == "spot"
    assert low["q_cpu"] == pytest.approx(low["max_cpu_usage"])

    batch = vm_requests.loc[vm_requests["collection_id"] == 300].iloc[0]
    assert batch["class"] == "batch_candidate"
    assert "scheduler" in batch["class_rule"]


def test_usage_scenarios_keep_observed_scenario_exact_and_bound_all_synthetic_usage() -> None:
    usage = pd.DataFrame(
        _usage_rows(100, 0, 0.10, 0.20, rows=12)
        + _usage_rows(200, 0, 0.05, 0.08, rows=12)
    )
    events = pd.DataFrame(
        [
            {"collection_id": 100, "instance_index": 0, "priority": 200},
            {"collection_id": 200, "instance_index": 0, "priority": 1},
        ]
    )
    vm_requests, observed = build_vm_requests(usage, events=events)

    scenarios = build_usage_scenarios(observed, vm_requests, num_scenarios=5, seed=42)

    scenario0 = scenarios.loc[scenarios["scenario_id"] == 0].sort_values(["vm_id", "t5_day"])
    base = observed.sort_values(["vm_id", "t5_day"])
    pd.testing.assert_series_equal(
        scenario0["cpu_usage"].reset_index(drop=True),
        base["cpu_usage"].reset_index(drop=True),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        scenario0["mem_usage"].reset_index(drop=True),
        base["mem_usage"].reset_index(drop=True),
        check_names=False,
    )

    merged = scenarios.merge(vm_requests[["vm_id", "class", "q_cpu", "q_mem"]], on="vm_id")
    synthetic = merged["scenario_id"].gt(0)
    assert (merged.loc[synthetic, "cpu_usage"] >= 0).all()
    assert (merged.loc[synthetic, "mem_usage"] >= 0).all()
    assert (merged.loc[synthetic, "cpu_usage"] <= merged.loc[synthetic, "q_cpu"] + 1e-12).all()
    assert (merged.loc[synthetic, "mem_usage"] <= merged.loc[synthetic, "q_mem"] + 1e-12).all()
    assert scenarios["scenario_id"].nunique() == 5


def test_on_demand_lognormal_usage_is_capped_without_changing_scenario_zero() -> None:
    observed = pd.DataFrame(
        [
            {
                "vm_id": "od0",
                "t5_day": 0,
                "t5": 0,
                "t_hour": 0,
                "cpu_usage": 0.9,
                "mem_usage": 0.8,
            }
        ]
    )
    requests = pd.DataFrame(
        [{"vm_id": "od0", "class": "on_demand", "q_cpu": 0.1, "q_mem": 0.2}]
    )

    scenarios = build_usage_scenarios(observed, requests, num_scenarios=2, seed=42)

    scenario0 = scenarios.loc[scenarios["scenario_id"].eq(0)].iloc[0]
    scenario1 = scenarios.loc[scenarios["scenario_id"].eq(1)].iloc[0]
    assert scenario0["cpu_usage"] == pytest.approx(0.9)
    assert scenario0["mem_usage"] == pytest.approx(0.8)
    assert 0.0 <= scenario1["cpu_usage"] <= 0.1
    assert 0.0 <= scenario1["mem_usage"] <= 0.2


def test_spot_preemption_paths_are_scenario_complete_and_monotone() -> None:
    usage = pd.DataFrame(
        _usage_rows(100, 0, 0.10, 0.20, rows=12)
        + _usage_rows(200, 0, 0.05, 0.08, rows=12)
    )
    events = pd.DataFrame(
        [
            {"collection_id": 100, "instance_index": 0, "priority": 200},
            {"collection_id": 200, "instance_index": 0, "priority": 1, "event_type": "EVICT", "time": 7 * STEP_US},
        ]
    )
    vm_requests, observed = build_vm_requests(usage, events=events)

    preemptions = build_spot_preemption_scenarios(vm_requests, observed, events=events, num_scenarios=5, seed=42)
    spot_vm = vm_requests.loc[vm_requests["class"] == "spot", "vm_id"].iloc[0]
    path = preemptions.loc[preemptions["vm_id"] == spot_vm]

    assert path.groupby("scenario_id")["t5_day"].nunique().eq(12).all()
    for _, group in path.groupby("scenario_id"):
        active = group.sort_values("t5_day")["active"].to_numpy()
        assert np.all(np.diff(active) <= 0)
    assert path.loc[(path["scenario_id"] == 0) & (path["t5_day"] >= 7), "active"].eq(0).all()


def test_numeric_evict_event_type_4_preempts_spot_vm() -> None:
    usage = pd.DataFrame(
        _usage_rows(200, 0, 0.05, 0.08, rows=12)
    )
    events = pd.DataFrame(
        [
            {"collection_id": 200, "instance_index": 0, "priority": 1, "event_type": 4, "time": 7 * STEP_US},
        ]
    )
    vm_requests, observed = build_vm_requests(usage, events=events)

    preemptions = build_spot_preemption_scenarios(vm_requests, observed, events=events, num_scenarios=5, seed=42)
    spot_vm = vm_requests.loc[vm_requests["class"] == "spot", "vm_id"].iloc[0]

    assert preemptions.loc[
        (preemptions["scenario_id"] == 0) & (preemptions["vm_id"] == spot_vm) & (preemptions["t5_day"] >= 7),
        "active",
    ].eq(0).all()


def test_numeric_scheduler_1_assigns_batch_candidate() -> None:
    usage = pd.DataFrame(_usage_rows(300, 0, 0.04, 0.06, rows=12))
    events = pd.DataFrame([{"collection_id": 300, "instance_index": 0, "priority": 130}])
    collection_events = pd.DataFrame([{"collection_id": 300, "scheduler": 1}])

    vm_requests, _ = build_vm_requests(usage, events=events, collection_events=collection_events)
    vm = vm_requests.iloc[0]

    assert vm["class"] == "batch_candidate"
    assert "SCHEDULER_BATCH" in vm["class_rule"]


def test_priority_proxy_mapping_uses_google_2019_tiers() -> None:
    usage = pd.DataFrame(_usage_rows(115, 0, 0.05, 0.08, rows=12) + _usage_rows(120, 0, 0.05, 0.08, rows=12))
    events = pd.DataFrame(
        [
            {"collection_id": 115, "instance_index": 0, "priority": 115},
            {"collection_id": 120, "instance_index": 0, "priority": 120},
        ]
    )

    vm_requests, _ = build_vm_requests(usage, events=events)

    spot = vm_requests.loc[vm_requests["collection_id"] == 115].iloc[0]
    od = vm_requests.loc[vm_requests["collection_id"] == 120].iloc[0]
    assert spot["class"] == "spot"
    assert "priority < 120" in spot["class_rule"]
    assert od["class"] == "on_demand"
    assert "priority >= 120" in od["class_rule"]


def test_t5_day_preserves_absolute_day_bucket() -> None:
    rows = _usage_rows(400, 0, 0.05, 0.08, rows=12, start_bucket=300)
    for offset, row in enumerate(rows):
        row["t5_day"] = 12 + offset
    usage = pd.DataFrame(rows)
    events = pd.DataFrame([{"collection_id": 400, "instance_index": 0, "priority": 120}])

    vm_requests, observed = build_vm_requests(usage, events=events, day_start_us=288 * STEP_US)

    assert observed["t5_day"].min() == 12
    assert observed["t5_day"].max() == 23
    assert vm_requests.iloc[0]["arrival_t5"] == 12


def test_hourly_usage_scenarios_are_unique_and_aggregate_5min_usage() -> None:
    usage = pd.DataFrame(_usage_rows(100, 0, 0.10, 0.20, rows=24))
    events = pd.DataFrame([{"collection_id": 100, "instance_index": 0, "priority": 120}])
    vm_requests, observed = build_vm_requests(usage, events=events)
    usage_5min = build_usage_scenarios(observed, vm_requests, num_scenarios=1, seed=42)

    hourly = build_hourly_usage_scenarios(usage_5min)

    assert hourly[["scenario_id", "vm_id", "t_hour"]].duplicated().sum() == 0
    first_hour = usage_5min.loc[usage_5min["t_hour"] == 0]
    hourly_first = hourly.loc[hourly["t_hour"] == 0].iloc[0]
    assert hourly_first["cpu_usage"] == pytest.approx(first_hour["cpu_usage"].mean())
    assert hourly_first["mem_usage"] == pytest.approx(first_hour["mem_usage"].mean())


def test_scenario_probabilities_and_model_params_are_model_ready() -> None:
    probabilities = build_scenario_probabilities(5)
    params = build_model_params()

    assert probabilities["probability"].sum() == pytest.approx(1.0)
    assert set(probabilities["scenario_id"]) == set(range(5))
    for key in ["alpha", "epsilon", "soc_init"]:
        assert key in params


def _write_valid_output_dir(output_dir) -> None:
    usage = pd.DataFrame(_usage_rows(100, 0, 0.10, 0.20, rows=12) + _usage_rows(200, 0, 0.05, 0.08, rows=12))
    events = pd.DataFrame(
        [
            {"collection_id": 100, "instance_index": 0, "priority": 120},
            {"collection_id": 200, "instance_index": 0, "priority": 1},
        ]
    )
    vm_requests, observed = build_vm_requests(usage, events=events)
    usage_5min = build_usage_scenarios(observed, vm_requests, num_scenarios=5, seed=42)
    usage_hourly = build_hourly_usage_scenarios(usage_5min)
    preemptions = build_spot_preemption_scenarios(vm_requests, observed, events=events, num_scenarios=5, seed=42)
    families, workload = build_batch_outputs(vm_requests, observed)

    vm_requests.to_csv(output_dir / "vm_requests.csv", index=False)
    usage_5min.to_csv(output_dir / "vm_usage_5min_scenarios.csv", index=False)
    usage_hourly.to_csv(output_dir / "vm_usage_hourly_scenarios.csv", index=False)
    preemptions.to_csv(output_dir / "spot_preemption_scenarios.csv", index=False)
    families.to_csv(output_dir / "batch_families.csv", index=False)
    workload.to_csv(output_dir / "batch_workload.csv", index=False)
    build_servers(10).to_csv(output_dir / "servers.csv", index=False)
    build_energy_scenarios(5, seed=42).to_csv(output_dir / "energy_scenarios.csv", index=False)
    build_scenario_probabilities(5).to_csv(output_dir / "scenario_probabilities.csv", index=False)
    (output_dir / "model_params.json").write_text(json.dumps(build_model_params()), encoding="utf-8")
    (output_dir / "metadata.json").write_text(json.dumps({"num_scenarios": 5}), encoding="utf-8")
    outlier_count = int(vm_requests["cpu_p100_gt_one"].astype(bool).sum())
    max_raw_cpu = float(vm_requests["max_cpu_usage"].max())
    (output_dir / "scaling_diagnostics.json").write_text(
        json.dumps(
            {
                "automatic_utilization_calibration_applied": False,
                "representative_machine_capacity_raw": {"cpu": 1.0, "mem": 1.0},
                "unit_conversion_factors": {"cpu_divisor": 1.0, "mem_divisor": 1.0},
                "representative_capacity_vm_filter": {
                    "configured_resource_clipping_applied": False,
                    "sampling_replacement_applied": False,
                    "source_vm_count": len(vm_requests),
                    "retained_vm_count": len(vm_requests),
                    "dropped_vm_count": 0,
                },
                "cpu_p100_audit": {
                    "status": "cpu_p100_with_representative_capacity_vm_drop_no_clipping",
                    "selected_vm_count": outlier_count,
                    "converted_q_cpu_gt_one_count": int(vm_requests["q_cpu"].gt(1.0).sum()),
                    "raw_cpu_p100_summary": {"max": max_raw_cpu},
                },
            }
        ),
        encoding="utf-8",
    )


def test_validate_output_dir_checks_probabilities_params_and_feasibility(tmp_path) -> None:
    _write_valid_output_dir(tmp_path)

    summary = validate_output_dir(tmp_path, write_summary=True)
    assert summary["peak_hourly_cpu_demand_over_capacity"] >= 0
    assert (tmp_path / "toy_instance_summary.md").exists()

    pd.DataFrame([{"scenario_id": scenario, "probability": 0.1} for scenario in range(5)]).to_csv(
        tmp_path / "scenario_probabilities.csv", index=False
    )
    with pytest.raises(ValueError, match="sum to 1"):
        validate_output_dir(tmp_path)

    build_scenario_probabilities(5).to_csv(tmp_path / "scenario_probabilities.csv", index=False)
    (tmp_path / "model_params.json").write_text(json.dumps({"alpha": 0.95, "soc_init": 0.5}), encoding="utf-8")
    with pytest.raises(ValueError, match="epsilon"):
        validate_output_dir(tmp_path)


def test_batch_and_energy_outputs_have_expected_shape() -> None:
    usage = pd.DataFrame(
        sum((_usage_rows(300 + idx, 0, 0.02 + idx * 0.002, 0.05, rows=12) for idx in range(14)), [])
    )
    events = pd.DataFrame(
        [{"collection_id": 300 + idx, "instance_index": 0, "priority": 1} for idx in range(14)]
    )
    collection_events = pd.DataFrame(
        [{"collection_id": 300 + idx, "scheduler": "batch"} for idx in range(14)]
    )
    vm_requests, observed = build_vm_requests(usage, events=events, collection_events=collection_events)

    families, workload = build_batch_outputs(vm_requests, observed, max_families=10)
    energy = build_energy_scenarios(num_scenarios=5, seed=42)

    assert 1 <= len(families) <= 10
    assert set(workload["family_id"]).issubset(set(families["family_id"]))
    synthetic_batch = workload.loc[workload["scenario_id"].gt(0)].merge(
        families[["family_id", "q_cpu_B", "q_mem_B"]],
        on="family_id",
        validate="many_to_one",
    )
    assert (synthetic_batch["cpu_workload"] >= 0).all()
    assert (synthetic_batch["mem_workload"] >= 0).all()
    assert (
        synthetic_batch["cpu_workload"]
        <= synthetic_batch["workload_volume"] * synthetic_batch["q_cpu_B"] + 1e-12
    ).all()
    assert (
        synthetic_batch["mem_workload"]
        <= synthetic_batch["workload_volume"] * synthetic_batch["q_mem_B"] + 1e-12
    ).all()
    assert energy.groupby("scenario_id")["t_hour"].nunique().eq(24).all()
    assert (energy["renewable_generation"] >= 0).all()
    assert (energy["sell_price"] <= energy["day_ahead_price"]).all()
    assert {"ess_charge_efficiency", "ess_discharge_efficiency"}.issubset(energy.columns)


def test_validate_output_dir_rejects_negative_usage(tmp_path) -> None:
    _write_valid_output_dir(tmp_path)
    bad_usage = pd.read_csv(tmp_path / "vm_usage_5min_scenarios.csv")
    bad_usage.loc[0, "cpu_usage"] = -0.1
    bad_usage.to_csv(tmp_path / "vm_usage_5min_scenarios.csv", index=False)

    with pytest.raises(ValueError, match="negative"):
        validate_output_dir(tmp_path)


def test_validate_output_dir_rejects_synthetic_usage_above_q(tmp_path) -> None:
    _write_valid_output_dir(tmp_path)
    requests = pd.read_csv(tmp_path / "vm_requests.csv").set_index("vm_id")
    usage = pd.read_csv(tmp_path / "vm_usage_5min_scenarios.csv")
    row_index = usage.index[usage["scenario_id"].gt(0)][0]
    vm_id = usage.loc[row_index, "vm_id"]
    usage.loc[row_index, "cpu_usage"] = float(requests.loc[vm_id, "q_cpu"]) + 0.1
    usage.to_csv(tmp_path / "vm_usage_5min_scenarios.csv", index=False)

    with pytest.raises(ValueError, match="synthetic CPU or memory usage above q"):
        validate_output_dir(tmp_path)


@pytest.mark.parametrize("coverage_us", [0, STEP_US + 1, 1.5, np.nan, np.inf])
def test_validate_output_dir_rejects_invalid_coverage_us(tmp_path, coverage_us) -> None:
    _write_valid_output_dir(tmp_path)
    usage = pd.read_csv(tmp_path / "vm_usage_5min_scenarios.csv")
    usage["coverage_us"] = usage["coverage_us"].astype(object)
    usage.loc[0, "coverage_us"] = coverage_us
    usage.to_csv(tmp_path / "vm_usage_5min_scenarios.csv", index=False)

    with pytest.raises(ValueError, match="coverage_us.*finite integer"):
        validate_output_dir(tmp_path)


def test_validate_output_dir_requires_coverage_us(tmp_path) -> None:
    _write_valid_output_dir(tmp_path)
    usage = pd.read_csv(tmp_path / "vm_usage_5min_scenarios.csv").drop(columns="coverage_us")
    usage.to_csv(tmp_path / "vm_usage_5min_scenarios.csv", index=False)

    with pytest.raises(ValueError, match="coverage_us"):
        validate_output_dir(tmp_path)


def test_validate_output_dir_rejects_duplicate_canonical_scenario_zero_key(tmp_path) -> None:
    _write_valid_output_dir(tmp_path)
    usage = pd.read_csv(tmp_path / "vm_usage_5min_scenarios.csv")
    scenario_zero_row = usage.loc[usage["scenario_id"].eq(0)].iloc[[0]]
    usage = pd.concat([usage, scenario_zero_row], ignore_index=True)
    usage.to_csv(tmp_path / "vm_usage_5min_scenarios.csv", index=False)

    with pytest.raises(ValueError, match="canonical scenario 0.*one row"):
        validate_output_dir(tmp_path)


@pytest.mark.parametrize(
    "ambiguity_column", ["arrival_state_ambiguous", "scheduler_state_ambiguous"]
)
def test_validate_output_dir_rejects_strict_event_state_ambiguity(
    tmp_path, ambiguity_column
) -> None:
    _write_valid_output_dir(tmp_path)
    requests = pd.read_csv(tmp_path / "vm_requests.csv")
    requests.loc[0, ambiguity_column] = True
    requests.to_csv(tmp_path / "vm_requests.csv", index=False)

    with pytest.raises(ValueError, match="strict preprocessing.*ambiguity"):
        validate_output_dir(tmp_path)


def test_validate_output_dir_rejects_vm_q_above_representative_capacity(tmp_path) -> None:
    _write_valid_output_dir(tmp_path)
    requests = pd.read_csv(tmp_path / "vm_requests.csv")
    requests.loc[0, "cpu_p100_gt_one"] = True
    requests.loc[0, "max_cpu_usage"] = 1.5
    requests.loc[0, "q_cpu"] = 1.5
    requests.to_csv(tmp_path / "vm_requests.csv", index=False)
    diagnostics = json.loads(
        (tmp_path / "scaling_diagnostics.json").read_text(encoding="utf-8")
    )
    audit = diagnostics["cpu_p100_audit"]
    audit["selected_vm_count"] = 1
    audit["converted_q_cpu_gt_one_count"] = 1
    audit["raw_cpu_p100_summary"]["max"] = 1.5
    (tmp_path / "scaling_diagnostics.json").write_text(
        json.dumps(diagnostics), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="q above one representative server"):
        validate_output_dir(tmp_path)


def test_validate_output_dir_requires_exact_unit_server_capacities(tmp_path) -> None:
    _write_valid_output_dir(tmp_path)
    servers = pd.read_csv(tmp_path / "servers.csv")
    servers.loc[0, "C_cpu"] = np.nextafter(1.0, 2.0)
    servers.to_csv(tmp_path / "servers.csv", index=False, float_format="%.17g")

    with pytest.raises(ValueError, match="exactly C_cpu=1.0 and C_mem=1.0"):
        validate_output_dir(tmp_path)


def test_validate_output_dir_rejects_automatic_utilization_calibration(tmp_path) -> None:
    _write_valid_output_dir(tmp_path)
    diagnostics = json.loads(
        (tmp_path / "scaling_diagnostics.json").read_text(encoding="utf-8")
    )
    diagnostics["automatic_utilization_calibration_applied"] = True
    (tmp_path / "scaling_diagnostics.json").write_text(
        json.dumps(diagnostics), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="automatic_utilization_calibration_applied=false"):
        validate_output_dir(tmp_path)


def test_bigquery_queries_are_bounded_and_parameterized() -> None:
    assert day_bounds_us(0) == (600_000_000, 87_000_000_000)
    assert day_bounds_us(1) == (87_000_000_000, 173_400_000_000)

    usage_query = build_usage_query("a")
    events_query = build_events_query("a")
    collection_events_query = build_collection_events_query("a")
    machine_events_query = build_machine_events_query("a")

    assert "`google.com:google-cluster-data`.clusterdata_2019_a.instance_usage" in usage_query
    assert "start_time < @day_end_us" in usage_query
    assert "end_time > @day_start_us" in usage_query
    assert "end_time > start_time" in usage_query
    assert "GREATEST(start_time, @day_start_us) AS clipped_start_time" in usage_query
    assert "LEAST(end_time, @day_end_us) AS clipped_end_time" in usage_query
    assert "GREATEST(start_time, @day_start_us) - @day_start_us" in usage_query
    assert "MIN(u.t5)" not in usage_query
    assert "start_time >= @day_start_us" not in usage_query
    assert "end_time - start_time >=" not in usage_query
    assert "alloc_collection_id IS NULL OR alloc_collection_id = 0" in usage_query
    assert "HAVING usage_rows >= 12" in usage_query
    assert "FARM_FINGERPRINT" in usage_query
    assert "LIMIT @max_instances" in usage_query
    assert "maximum_usage.cpus" in usage_query
    assert "maximum_usage.memory" in usage_query
    assert "cpu_usage_distribution" in usage_query
    assert "machine_id" in usage_query

    assert "`google.com:google-cluster-data`.clusterdata_2019_a.instance_events" in events_query
    assert "JOIN candidates" in events_query
    assert "resource_request.cpus" in events_query
    assert "WHERE e.time < @day_end_us" in events_query
    assert "e.time >= @day_start_us" not in events_query

    assert "`google.com:google-cluster-data`.clusterdata_2019_a.collection_events" in collection_events_query
    assert "CASE e.scheduler" in collection_events_query
    assert "WHEN 1 THEN 'SCHEDULER_BATCH'" in collection_events_query
    assert "CAST(NULL AS STRING) AS scheduler" not in collection_events_query
    assert "WHERE e.time < @day_end_us" in collection_events_query
    assert "e.time >= @day_start_us" not in collection_events_query

    assert "`google.com:google-cluster-data`.clusterdata_2019_a.machine_events" in machine_events_query
    assert "capacity.cpus" in machine_events_query
    assert "capacity.memory" in machine_events_query
    assert "WHERE e.time < @day_end_us" in machine_events_query
    assert "JOIN candidates" not in machine_events_query
