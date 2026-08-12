from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


def _load_builder_module():
    path = next(
        parent / "scripts/build_micro_top10_unscaled_dataset.py"
        for parent in Path(__file__).resolve().parents
        if (parent / "scripts/build_micro_top10_unscaled_dataset.py").is_file()
    )
    spec = importlib.util.spec_from_file_location(
        "build_micro_top10_unscaled_dataset", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_cpu_top20_builder_module():
    path = next(
        parent
        / "scripts/build_micro_cpu_top20_od_sp10_bj10_unscaled_dataset.py"
        for parent in Path(__file__).resolve().parents
        if (
            parent
            / "scripts/build_micro_cpu_top20_od_sp10_bj10_unscaled_dataset.py"
        ).is_file()
    )
    spec = importlib.util.spec_from_file_location(
        "build_micro_cpu_top20_od_sp10_bj10_unscaled_dataset",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_canonical_fixture(root: Path) -> None:
    request_rows: list[dict[str, object]] = []
    usage_rows: list[dict[str, object]] = []
    prefixes = {
        "on_demand": "od",
        "spot": "sp",
        "batch_candidate": "ba",
    }
    for vm_class, prefix in prefixes.items():
        for number in range(25):
            vm_id = f"{prefix}_{number:02d}"
            q_cpu = 0.24
            q_mem = 0.31
            cpu = 0.050 + 0.001 * (11 - number)
            mem = 0.180 - 0.004 * number
            # Highest score, but it has no observation in active slot 1.
            # Service classes must exclude it; batch candidates need not.
            missing_active_slot = number == 10 and vm_class != "batch_candidate"
            if number == 10:
                cpu = 0.19
                mem = 0.19
            request_rows.append(
                {
                    "vm_id": vm_id,
                    "collection_id": 100 + number,
                    "instance_index": number,
                    "class": vm_class,
                    "scheduler": (
                        "SCHEDULER_BATCH"
                        if vm_class == "batch_candidate"
                        else "SCHEDULER_DEFAULT"
                    ),
                    "arrival_t5": 0,
                    "departure_t5": 12,
                    "lifetime_t5": 12,
                    "q_cpu": q_cpu,
                    "q_mem": q_mem,
                    "resource_request_cpu": q_cpu / 2,
                    "resource_request_mem": q_mem / 2,
                    "priority": 120 if vm_class == "on_demand" else 100,
                }
            )
            buckets = [0] if missing_active_slot else [0, 6]
            for t5 in buckets:
                usage_rows.append(
                    {
                        "scenario_id": 0,
                        "vm_id": vm_id,
                        "t5_day": t5,
                        "t_hour": t5 // 12,
                        "cpu_usage": cpu,
                        "mem_usage": mem,
                        "coverage_us": 300_000_000,
                        "coverage_ratio": 1.0,
                        "max_cpu_usage": cpu,
                        "max_mem_usage": mem,
                        "assigned_memory": q_mem,
                    }
                )

    later = dict(usage_rows[0])
    later["scenario_id"] = 1
    usage_rows.append(later)

    pd.DataFrame(request_rows).to_csv(root / "vm_requests.csv", index=False)
    pd.DataFrame(usage_rows).sort_values(
        ["scenario_id", "vm_id", "t5_day"]
    ).to_csv(root / "vm_usage_5min_scenarios.csv", index=False)
    pd.DataFrame(
        {
            "server_id": [f"s{i:03d}" for i in range(6)],
            "C_cpu": [1.0] * 6,
            "C_mem": [1.0] * 6,
            "E_idle": [0.35] * 6,
            "E_cpu": [0.65] * 6,
            "min_on_time": [1] * 6,
            "min_off_time": [1] * 6,
        }
    ).to_csv(root / "servers.csv", index=False)
    (root / "metadata.json").write_text(
        '{"fixture": true}\n', encoding="utf-8"
    )


def test_builder_selects_top_ten_and_copies_q_and_usage_exactly(
    tmp_path: Path,
) -> None:
    module = _load_builder_module()
    source = tmp_path / "canonical"
    output = tmp_path / "derived"
    source.mkdir()
    _write_canonical_fixture(source)
    source_hashes = {path.name: _digest(path) for path in source.iterdir()}

    report = module.build_micro_top10_unscaled_dataset(source, output)

    assert {path.name: _digest(path) for path in source.iterdir()} == source_hashes
    source_requests = pd.read_csv(
        source / "vm_requests.csv", dtype={"vm_id": str}
    )
    requests = pd.read_csv(output / "vm_requests.csv", dtype={"vm_id": str})
    source_usage = pd.read_csv(
        source / "vm_usage_5min_scenarios.csv", dtype={"vm_id": str}
    )
    usage = pd.read_csv(
        output / "vm_usage_5min_scenarios.csv", dtype={"vm_id": str}
    )
    manifest = pd.read_csv(
        output / "selection_manifest.csv", dtype={"vm_id": str}
    )

    assert requests["class"].value_counts().to_dict() == {
        "on_demand": 10,
        "spot": 10,
        "batch_candidate": 10,
    }
    assert set(requests.loc[requests["class"].eq("on_demand"), "vm_id"]) == {
        f"od_{number:02d}" for number in range(10)
    }
    assert "od_10" not in set(requests["vm_id"])
    assert "ba_10" in set(requests["vm_id"])
    for vm_class in ("on_demand", "spot", "batch_candidate"):
        ranks = manifest.loc[
            manifest["class"].eq(vm_class), "eligible_rank_within_class"
        ].sort_values()
        assert ranks.tolist() == list(range(1, 11))

    selected_ids = set(requests["vm_id"])
    expected_requests = source_requests.loc[
        source_requests["vm_id"].isin(selected_ids)
    ].sort_values("vm_id").reset_index(drop=True)
    actual_requests = requests.sort_values("vm_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(
        actual_requests[expected_requests.columns],
        expected_requests,
        check_dtype=False,
        check_exact=True,
    )

    expected_usage = source_usage.loc[
        source_usage["scenario_id"].eq(0)
        & source_usage["vm_id"].isin(selected_ids)
    ].sort_values(["vm_id", "t5_day"]).reset_index(drop=True)
    actual_usage = usage.sort_values(["vm_id", "t5_day"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        actual_usage[expected_usage.columns],
        expected_usage,
        check_dtype=False,
        check_exact=True,
    )
    assert usage["scenario_id"].unique().tolist() == [0]

    policy = json.loads((output / "selection_policy.json").read_text())
    assert policy["transformation"]["q_cpu"] == "copied unchanged"
    assert policy["transformation"]["q_mem"] == "copied unchanged"
    assert policy["transformation"]["actual_cpu_usage"] == "copied unchanged"
    assert policy["transformation"]["actual_mem_usage"] == "copied unchanged"
    assert policy["transformation"]["target_q"] is None
    assert policy["transformation"]["memory_pressure_transform"] is None
    assert report["canonical_source_integrity"]["unchanged"] is True
    assert report["invariants"]["top_average_usage_selection"] is True
    assert report["invariants"]["no_resource_scaling"] is True
    assert report["invariants"]["no_memory_pressure_transform"] is True

    probabilities = pd.read_csv(output / "scenario_probabilities.csv")
    assert probabilities["scenario_id"].tolist() == list(range(10))
    assert np.isclose(probabilities["probability"].sum(), 1.0)


def test_builder_output_is_deterministic(tmp_path: Path) -> None:
    module = _load_builder_module()
    source = tmp_path / "canonical"
    first = tmp_path / "first"
    second = tmp_path / "second"
    source.mkdir()
    _write_canonical_fixture(source)

    module.build_micro_top10_unscaled_dataset(source, first)
    module.build_micro_top10_unscaled_dataset(source, second)

    for name in (
        "vm_requests.csv",
        "vm_usage_5min_scenarios.csv",
        "selection_manifest.csv",
        "scenario_probabilities.csv",
        "selection_policy.json",
        "validation_report.json",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_cpu_top20_builder_uses_cpu_only_and_freezes_spot_batch(
    tmp_path: Path,
) -> None:
    base_module = _load_builder_module()
    module = _load_cpu_top20_builder_module()
    source = tmp_path / "canonical"
    reference = tmp_path / "reference"
    output = tmp_path / "derived"
    source.mkdir()
    _write_canonical_fixture(source)

    # This low-CPU/high-memory decoy would rank near the top under CPU+MEM,
    # but must not enter a CPU-only top-20 selection.
    source_usage = pd.read_csv(
        source / "vm_usage_5min_scenarios.csv",
        dtype={"vm_id": str},
    )
    decoy = source_usage["vm_id"].eq("od_24")
    source_usage.loc[decoy, ["mem_usage", "max_mem_usage"]] = 0.95
    source_usage.to_csv(
        source / "vm_usage_5min_scenarios.csv",
        index=False,
    )
    source_hashes = {path.name: _digest(path) for path in source.iterdir()}

    base_module.build_micro_top10_unscaled_dataset(source, reference)
    reference_hashes = {
        path.name: _digest(path) for path in reference.iterdir()
    }
    report = module.build_micro_cpu_top20_od_sp10_bj10_unscaled_dataset(
        source,
        reference,
        output,
    )

    assert {path.name: _digest(path) for path in source.iterdir()} == (
        source_hashes
    )
    assert {path.name: _digest(path) for path in reference.iterdir()} == (
        reference_hashes
    )

    source_requests = pd.read_csv(
        source / "vm_requests.csv",
        dtype={"vm_id": str},
    )
    source_usage = pd.read_csv(
        source / "vm_usage_5min_scenarios.csv",
        dtype={"vm_id": str},
    )
    requests = pd.read_csv(
        output / "vm_requests.csv",
        dtype={"vm_id": str},
    )
    usage = pd.read_csv(
        output / "vm_usage_5min_scenarios.csv",
        dtype={"vm_id": str},
    )
    manifest = pd.read_csv(
        output / "selection_manifest.csv",
        dtype={"vm_id": str},
    )
    reference_manifest = pd.read_csv(
        reference / "selection_manifest.csv",
        dtype={"vm_id": str},
    )

    assert requests["class"].value_counts().to_dict() == {
        "on_demand": 20,
        "spot": 10,
        "batch_candidate": 10,
    }
    expected_od = [
        *(f"od_{number:02d}" for number in range(10)),
        *(f"od_{number:02d}" for number in range(11, 21)),
    ]
    actual_od = requests.loc[
        requests["class"].eq("on_demand"), "vm_id"
    ].tolist()
    assert actual_od == expected_od
    assert "od_10" not in actual_od
    assert "od_24" not in actual_od

    for vm_class in ("spot", "batch_candidate"):
        reference_ids = (
            reference_manifest.loc[
                reference_manifest["class"].eq(vm_class)
            ]
            .sort_values("selection_rank_within_class")["vm_id"]
            .tolist()
        )
        output_ids = requests.loc[
            requests["class"].eq(vm_class), "vm_id"
        ].tolist()
        manifest_ids = manifest.loc[
            manifest["class"].eq(vm_class), "vm_id"
        ].tolist()
        assert output_ids == reference_ids
        assert manifest_ids == reference_ids

    selected_ids = set(requests["vm_id"])
    expected_requests = source_requests.loc[
        source_requests["vm_id"].isin(selected_ids)
    ].sort_values("vm_id").reset_index(drop=True)
    actual_requests = requests.sort_values("vm_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(
        actual_requests[expected_requests.columns],
        expected_requests,
        check_dtype=False,
        check_exact=True,
    )

    expected_usage = source_usage.loc[
        source_usage["scenario_id"].eq(0)
        & source_usage["vm_id"].isin(selected_ids)
    ].sort_values(["vm_id", "t5_day"]).reset_index(drop=True)
    actual_usage = usage.sort_values(
        ["vm_id", "t5_day"]
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        actual_usage[expected_usage.columns],
        expected_usage,
        check_dtype=False,
        check_exact=True,
    )
    assert usage["scenario_id"].unique().tolist() == [0]

    od_manifest = manifest.loc[
        manifest["class"].eq("on_demand")
    ]
    assert od_manifest["eligible_rank_within_class"].tolist() == list(
        range(1, 21)
    )
    assert od_manifest["selection_score_name"].unique().tolist() == [
        "coverage_weighted_avg_cpu_usage"
    ]

    policy = json.loads((output / "selection_policy.json").read_text())
    assert policy["selection"]["on_demand"]["count"] == 20
    assert policy["selection"]["on_demand"]["score"] == (
        "coverage-weighted mean(cpu_usage)"
    )
    assert policy["transformation"]["usage_rescaling"] is None
    assert policy["transformation"]["memory_pressure_transform"] is None
    assert policy["transformation"]["target_q"] is None
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["dataset_name"] == (
        "notion_toy_google2019_micro_cpu_top20_od_sp10_bj10_unscaled_v1"
    )
    assert metadata["num_vms"] == 40
    assert metadata["class_counts"] == {
        "on_demand": 20,
        "spot": 10,
        "batch_candidate": 10,
    }
    assert metadata["num_scenarios"] == 10
    assert metadata["stored_num_scenarios"] == 1
    assert metadata["stored_scenario_ids"] == [0]
    assert metadata["default_model_usage_file"] == (
        "vm_usage_5min_scenarios.csv"
    )
    assert metadata["canonical_source_metadata"]["fixture"] is True
    assert report["canonical_source_integrity"]["unchanged"] is True
    assert report["reference_fixture_integrity"]["unchanged"] is True
    assert report["builder_integrity"]["unchanged"] is True
    assert report["builder_integrity"]["sha256_before"] == (
        report["builder_integrity"]["sha256_after"]
    )
    assert report["invariants"][
        "on_demand_is_eligible_cpu_only_top20"
    ] is True
    assert report["invariants"][
        "spot_ids_and_order_match_reference_exactly"
    ] is True
    assert report["invariants"][
        "batch_ids_and_order_match_reference_exactly"
    ] is True


def test_cpu_top20_builder_output_is_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_module = _load_builder_module()
    module = _load_cpu_top20_builder_module()
    source = tmp_path / "canonical"
    reference = tmp_path / "reference"
    first = tmp_path / "first"
    second = tmp_path / "second"
    source.mkdir()
    _write_canonical_fixture(source)
    base_module.build_micro_top10_unscaled_dataset(source, reference)

    module.build_micro_cpu_top20_od_sp10_bj10_unscaled_dataset(
        source,
        reference,
        first,
    )
    monkeypatch.chdir(tmp_path)
    module.build_micro_cpu_top20_od_sp10_bj10_unscaled_dataset(
        source,
        reference,
        second,
    )

    for name in (
        "vm_requests.csv",
        "vm_usage_5min_scenarios.csv",
        "selection_manifest.csv",
        "scenario_probabilities.csv",
        "selection_policy.json",
        "validation_report.json",
        "metadata.json",
        "README.md",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_cpu_top20_builder_rejects_overlapping_output_roots(
    tmp_path: Path,
) -> None:
    base_module = _load_builder_module()
    module = _load_cpu_top20_builder_module()
    source = tmp_path / "canonical"
    reference = tmp_path / "reference"
    source.mkdir()
    _write_canonical_fixture(source)
    base_module.build_micro_top10_unscaled_dataset(source, reference)
    source_hashes = {path.name: _digest(path) for path in source.iterdir()}
    reference_hashes = {
        path.name: _digest(path) for path in reference.iterdir()
    }

    with pytest.raises(ValueError, match="must be disjoint paths"):
        module.build_micro_cpu_top20_od_sp10_bj10_unscaled_dataset(
            source,
            reference,
            tmp_path,
            overwrite=True,
        )
    with pytest.raises(ValueError, match="must be disjoint paths"):
        module.build_micro_cpu_top20_od_sp10_bj10_unscaled_dataset(
            source,
            reference,
            source / "derived",
            overwrite=True,
        )

    assert {path.name: _digest(path) for path in source.iterdir()} == (
        source_hashes
    )
    assert {path.name: _digest(path) for path in reference.iterdir()} == (
        reference_hashes
    )


def test_unscaled_config_uses_requested_workload_sigmas() -> None:
    config_path = next(
        parent
        / "experiments/2607-notion-server-min-vmp/configs"
        / "micro_top10_unscaled_baseline.yaml"
        for parent in Path(__file__).resolve().parents
        if (
            parent
            / "experiments/2607-notion-server-min-vmp/configs"
            / "micro_top10_unscaled_baseline.yaml"
        ).is_file()
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["data"]["google_dir"].endswith(
        "notion_toy_google2019_micro_top10_unscaled_v1"
    )
    assert config["workload_scenarios"] == {
        "cpu_lognormal_sigma": 0.24,
        "memory_lognormal_sigma": 0.12,
    }
