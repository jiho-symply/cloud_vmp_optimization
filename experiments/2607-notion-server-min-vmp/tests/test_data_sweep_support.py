from __future__ import annotations

import copy
import json

import pandas as pd
import pytest

from conftest import EXPERIMENT_ROOT

from notion_server_min_vmp.data import (
    CPU,
    MEM,
    T5_US,
    _build_batch_families,
    _build_service_scenarios,
    _eligible_service_ids,
    _read_scenario_zero_usage,
    _sampling_metadata,
    _validated_configured_resources,
    build_instance,
    load_config,
)


def test_service_trace_requires_one_observation_in_every_active_slot() -> None:
    requests = pd.DataFrame(
        [
            {
                "vm_id": "spot0",
                "class": "spot",
                "arrival_t5": 1,
                "departure_t5": 11,
            }
        ]
    )
    usage = pd.DataFrame(
        {
            "vm_id": ["spot0", "spot0"],
            "t5_day": [1, 9],
        }
    )

    assert _eligible_service_ids(
        requests,
        usage,
        "spot",
        slots_per_day=48,
        t5_per_slot=6,
    ) == ["spot0"]

    # t5=0 is in the same 30-minute slot but precedes this VM's arrival, so it
    # cannot satisfy the active-slot observation requirement.
    usage.loc[0, "t5_day"] = 0
    assert _eligible_service_ids(
        requests,
        usage,
        "spot",
        slots_per_day=48,
        t5_per_slot=6,
    ) == []


def test_partial_slot_cpu_and_memory_use_coverage_weighted_means() -> None:
    requests = pd.DataFrame(
        [
            {
                "vm_id": "od0",
                "class": "on_demand",
                "arrival_t5": 1,
                "departure_t5": 6,
                "q_cpu": 0.5,
                "q_mem": 0.8,
                "resource_request_cpu": 0.5,
                "resource_request_mem": 0.8,
            }
        ]
    )
    usage = pd.DataFrame(
        {
            "vm_id": ["od0", "od0", "od0"],
            # The first row precedes arrival and must not enter the aggregate.
            "t5_day": [0, 1, 4],
            "cpu_usage": [9.0, 0.2, 0.4],
            "mem_usage": [9.0, 0.3, 0.7],
            "coverage_us": [T5_US, T5_US, 1_000_000],
        }
    )

    _, _, load_od, _, metadata = _build_service_scenarios(
        requests,
        usage,
        ["od0"],
        [],
        {"od0": [0]},
        [0],
        t5_per_slot=6,
        seed=42,
        cpu_sigma=0.22,
        mem_sigma=0.08,
    )

    cpu = load_od[("od0", CPU, 0, 0)]
    mem = load_od[("od0", MEM, 0, 0)]
    assert isinstance(cpu, float)
    assert isinstance(mem, float)
    assert cpu == pytest.approx((0.2 * T5_US + 0.4 * 1_000_000) / (T5_US + 1_000_000))
    assert mem == pytest.approx((0.3 * T5_US + 0.7 * 1_000_000) / (T5_US + 1_000_000))
    assert "coverage_us-weighted mean" in metadata["scenario_zero_policy"]


def test_full_coverage_rows_preserve_ordinary_cpu_and_memory_means() -> None:
    requests = pd.DataFrame(
        [
            {
                "vm_id": "od0",
                "class": "on_demand",
                "arrival_t5": 0,
                "departure_t5": 6,
                "q_cpu": 1.0,
                "q_mem": 1.0,
            }
        ]
    )
    cpu_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    mem_values = [0.2, 0.4, 0.3, 0.7, 0.1, 0.6]
    usage = pd.DataFrame(
        {
            "vm_id": ["od0"] * 6,
            "t5_day": list(range(6)),
            "cpu_usage": cpu_values,
            "mem_usage": mem_values,
            "coverage_us": [T5_US] * 6,
        }
    )

    _, _, load_od, _, _ = _build_service_scenarios(
        requests,
        usage,
        ["od0"],
        [],
        {"od0": [0]},
        [0],
        t5_per_slot=6,
        seed=42,
        cpu_sigma=0.22,
        mem_sigma=0.08,
    )

    assert load_od[("od0", CPU, 0, 0)] == pytest.approx(sum(cpu_values) / 6)
    assert load_od[("od0", MEM, 0, 0)] == pytest.approx(sum(mem_values) / 6)


def test_synthetic_service_draws_are_capped_for_every_class_and_scenario_zero_is_exact() -> None:
    requests = pd.DataFrame(
        [
            {
                "vm_id": "od0",
                "class": "on_demand",
                "arrival_t5": 0,
                "departure_t5": 1,
                "q_cpu": 0.2,
                "q_mem": 0.3,
            },
            {
                "vm_id": "spot0",
                "class": "spot",
                "arrival_t5": 0,
                "departure_t5": 1,
                "q_cpu": 0.4,
                "q_mem": 0.5,
            },
        ]
    )
    usage = pd.DataFrame(
        {
            "vm_id": ["od0", "spot0"],
            "t5_day": [0, 0],
            "cpu_usage": [0.2, 0.4],
            "mem_usage": [0.3, 0.5],
            "coverage_us": [T5_US, T5_US],
        }
    )

    q_od, q_spot, load_od, load_spot, metadata = _build_service_scenarios(
        requests,
        usage,
        ["od0"],
        ["spot0"],
        {"od0": [0], "spot0": [0]},
        [0, 1, 2],
        t5_per_slot=6,
        seed=42,
        cpu_sigma=0.22,
        mem_sigma=0.08,
    )

    assert load_od[("od0", CPU, 0, 0)] == pytest.approx(0.2)
    assert load_od[("od0", MEM, 0, 0)] == pytest.approx(0.3)
    assert load_spot[("spot0", CPU, 0, 0)] == pytest.approx(0.4)
    assert load_spot[("spot0", MEM, 0, 0)] == pytest.approx(0.5)
    for loads, q in ((load_od, q_od), (load_spot, q_spot)):
        for (vm_id, resource, _slot, scenario_id), value in loads.items():
            if scenario_id > 0:
                assert 0.0 <= value <= q[(vm_id, resource)]
    # Seed 42's first CPU multiplier is greater than one, so this checks that
    # on-demand CPU is actually capped rather than passing vacuously.
    assert load_od[("od0", CPU, 0, 1)] == pytest.approx(q_od[("od0", CPU)])
    assert metadata["synthetic_caps"] == (
        "all service CPU and memory lognormal draws capped to [0, q] in synthetic scenarios"
    )


def test_scenario_zero_reader_collapses_duplicate_keys_with_coverage_audit(
    tmp_path,
) -> None:
    path = tmp_path / "usage.csv"
    rows = pd.DataFrame(
        [
            {
                "scenario_id": 0,
                "vm_id": "vm0",
                "t5_day": 0,
                "cpu_usage": 0.2,
                "mem_usage": 0.3,
                "coverage_us": T5_US,
            },
            {
                "scenario_id": 0,
                "vm_id": "vm0",
                "t5_day": 0,
                "cpu_usage": 0.8,
                "mem_usage": 0.7,
                "coverage_us": 1_000_000,
            },
            # Exact duplicates must not distort the weighted CPU value.
            {
                "scenario_id": 0,
                "vm_id": "vm0",
                "t5_day": 0,
                "cpu_usage": 0.2,
                "mem_usage": 0.3,
                "coverage_us": T5_US,
            },
        ]
    )
    rows.to_csv(path, index=False)

    usage = _read_scenario_zero_usage(
        path,
        {"vm0"},
        chunksize=2,
        sorted_by_scenario=True,
    )

    assert len(usage) == 1
    assert usage.loc[0, "cpu_usage"] == pytest.approx(
        (0.2 * T5_US + 0.8 * 1_000_000) / (T5_US + 1_000_000)
    )
    assert usage.loc[0, "mem_usage"] == pytest.approx(
        (0.3 * T5_US + 0.7 * 1_000_000) / (T5_US + 1_000_000)
    )
    assert usage.loc[0, "coverage_us"] == T5_US
    assert usage.attrs["coverage_audit"] == {
        "source_scenario_zero_rows": 3,
        "canonical_scenario_zero_rows": 1,
        "duplicate_key_count": 1,
        "duplicate_extra_row_count": 2,
        "exact_duplicate_row_count": 1,
        "duplicate_coverage_policy": (
            "CPU and memory coverage-weighted means, and max coverage_us per duplicate VM/bucket key"
        ),
    }


@pytest.mark.parametrize("column", ["q_cpu", "q_mem"])
def test_loader_rejects_vm_larger_than_one_representative_server(column: str) -> None:
    requests = pd.DataFrame(
        [{"vm_id": "vm0", "q_cpu": 0.5, "q_mem": 0.5}]
    )
    requests.loc[0, column] = 1.0001

    with pytest.raises(ValueError, match=r"finite values in \(0, 1\]"):
        _validated_configured_resources(requests)


@pytest.mark.parametrize("coverage_us", [0, T5_US + 1, 1.5, float("inf")])
def test_scenario_zero_reader_rejects_invalid_coverage(tmp_path, coverage_us: float) -> None:
    path = tmp_path / "usage.csv"
    pd.DataFrame(
        [
            {
                "scenario_id": 0,
                "vm_id": "vm0",
                "t5_day": 0,
                "cpu_usage": 0.2,
                "mem_usage": 0.3,
                "coverage_us": coverage_us,
            }
        ]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match=r"coverage_us.*finite integer values"):
        _read_scenario_zero_usage(
            path,
            {"vm0"},
            chunksize=2,
            sorted_by_scenario=True,
        )


def test_sampling_metadata_has_no_observation_count_or_coverage_audit() -> None:
    metadata = _sampling_metadata(
        seed=42,
        source_counts={"on_demand": 10, "spot": 10, "batch_candidate": 10},
        eligible_od=["od0"],
        eligible_spot=["spot0"],
        selected_od=["od0"],
        selected_spot=["spot0"],
        selected_batch=["batch0"],
    )
    serialized = json.dumps(metadata, sort_keys=True)

    for removed_term in (
        "coverage",
        "incomplete",
        "rows_scanned",
        "scenario_zero_rows_seen",
        "candidate_rows_after_duplicate_collapse",
        "duplicate_extra_rows_collapsed",
        "duplicate_keys_collapsed",
    ):
        assert removed_term not in serialized


@pytest.mark.parametrize(
    "deprecated_key", ["minimum_5min_coverage_ratio", "require_contiguous_5min"]
)
def test_deprecated_trace_coverage_options_are_rejected(deprecated_key: str) -> None:
    baseline_path = EXPERIMENT_ROOT / "configs" / "baseline.yaml"
    config = copy.deepcopy(load_config(baseline_path))
    config["data"][deprecated_key] = 0.8

    with pytest.raises(ValueError, match="Deprecated trace-coverage configuration"):
        build_instance(config, config_path=baseline_path)


@pytest.mark.parametrize("invalid_count", [True, 100.5])
def test_vm_class_counts_must_be_explicit_integers(invalid_count: object) -> None:
    baseline_path = EXPERIMENT_ROOT / "configs" / "baseline.yaml"
    config = copy.deepcopy(load_config(baseline_path))
    config["experiment"]["class_counts"]["on_demand"] = invalid_count

    with pytest.raises(ValueError, match="class_counts.on_demand must be an integer"):
        build_instance(config, config_path=baseline_path)


@pytest.mark.parametrize("invalid_count", [True, 6.5])
def test_server_count_must_be_an_explicit_integer(invalid_count: object) -> None:
    baseline_path = EXPERIMENT_ROOT / "configs" / "baseline.yaml"
    config = copy.deepcopy(load_config(baseline_path))
    config["experiment"]["num_servers"] = invalid_count

    with pytest.raises(ValueError, match="experiment.num_servers must be an integer"):
        build_instance(config, config_path=baseline_path)


def test_empty_batch_class_builds_an_empty_family_set() -> None:
    K, q, rho, workload, audit = _build_batch_families(
        pd.DataFrame(),
        pd.DataFrame(),
        [],
        t5_per_slot=6,
        max_families=100,
        pair_round_digits=12,
    )

    assert (K, q, rho, workload) == ([], {}, {}, {})
    assert audit["selected_batch_jobs"] == 0
    assert audit["constructed_family_count"] == 0


def test_batch_workload_and_rho_use_observed_coverage_duration() -> None:
    requests = pd.DataFrame(
        [
            {
                "vm_id": "batch0",
                "class": "batch_candidate",
                "q_cpu": 0.9,
                "q_mem": 0.8,
            }
        ]
    )
    usage = pd.DataFrame(
        {
            "vm_id": ["batch0", "batch0"],
            "t5_day": [0, 1],
            "cpu_usage": [0.2, 0.8],
            "mem_usage": [0.4, 0.1],
            "coverage_us": [T5_US, 1_000_000],
        }
    )

    _, q, rho, workload, audit = _build_batch_families(
        requests,
        usage,
        ["batch0"],
        t5_per_slot=6,
        max_families=100,
        pair_round_digits=12,
    )

    slot_us = 6 * T5_US
    expected_coverage = T5_US + 1_000_000
    expected_cpu_volume = (0.2 * T5_US + 0.8 * 1_000_000) / slot_us
    expected_mem_volume = (0.4 * T5_US + 0.1 * 1_000_000) / slot_us
    assert q[("batch000", CPU)] == pytest.approx(0.9)
    assert q[("batch000", MEM)] == pytest.approx(0.8)
    assert workload["batch000"] == pytest.approx(expected_coverage / slot_us)
    assert rho[("batch000", CPU)] == pytest.approx(
        expected_cpu_volume / (expected_coverage / slot_us)
    )
    assert rho[("batch000", MEM)] == pytest.approx(
        expected_mem_volume / (expected_coverage / slot_us)
    )
    assert audit["total_cpu_resource_slot_volume"] == pytest.approx(expected_cpu_volume)
    assert audit["total_memory_resource_slot_volume"] == pytest.approx(expected_mem_volume)
    assert "coverage_us" in audit["workload_definition"]


def test_full_window_batch_fixture_is_equivalent_to_row_count_formula() -> None:
    requests = pd.DataFrame(
        [{"vm_id": "batch0", "class": "batch_candidate", "q_cpu": 0.7, "q_mem": 0.6}]
    )
    usage = pd.DataFrame(
        {
            "vm_id": ["batch0"] * 6,
            "t5_day": list(range(6)),
            "cpu_usage": [0.3] * 6,
            "mem_usage": [0.4] * 6,
            "coverage_us": [T5_US] * 6,
        }
    )

    _, _, rho, workload, _ = _build_batch_families(
        requests,
        usage,
        ["batch0"],
        t5_per_slot=6,
        max_families=100,
        pair_round_digits=12,
    )

    assert workload["batch000"] == pytest.approx(1.0)
    assert rho[("batch000", CPU)] == pytest.approx(0.3)
    assert rho[("batch000", MEM)] == pytest.approx(0.4)
