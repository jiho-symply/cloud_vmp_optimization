from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_builder_module():
    path = next(
        parent / "scripts/build_micro_stress_dataset.py"
        for parent in Path(__file__).resolve().parents
        if (parent / "scripts/build_micro_stress_dataset.py").is_file()
    )
    spec = importlib.util.spec_from_file_location("build_micro_stress_dataset", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_canonical_fixture(root: Path) -> None:
    request_rows = []
    usage_rows = []
    prefixes = {
        "on_demand": "od",
        "spot": "sp",
        "batch_candidate": "ba",
    }
    for vm_class, prefix in prefixes.items():
        for number in range(12):
            vm_id = f"{prefix}_{number:02d}"
            q_cpu = 0.2
            q_mem = 0.2
            cpu = 0.05 + 0.001 * (11 - number)
            mem = 0.18 - 0.004 * number
            # This is the highest-scoring OD candidate, but it lacks all
            # observations in active slot 1 and must be excluded by eligibility.
            missing_active_slot = vm_class == "on_demand" and number == 10
            if missing_active_slot:
                cpu = 0.19
                mem = 0.19
            request_rows.append(
                {
                    "vm_id": vm_id,
                    "collection_id": 100 + number,
                    "instance_index": number,
                    "class": vm_class,
                    "scheduler": "SCHEDULER_BATCH" if vm_class == "batch_candidate" else "SCHEDULER_DEFAULT",
                    "arrival_t5": 0,
                    "departure_t5": 12,
                    "lifetime_t5": 12,
                    "q_cpu": q_cpu,
                    "q_mem": q_mem,
                    "resource_request_cpu": 0.1,
                    "resource_request_mem": 0.1,
                    "p95_cpu_usage": cpu,
                    "p95_mem_usage": mem,
                    "avg_cpu_usage": cpu,
                    "avg_mem_usage": mem,
                    "max_cpu_usage": cpu,
                    "max_mem_usage": mem,
                    "priority": 120 if vm_class == "on_demand" else 100,
                }
            )
            buckets = [0] if missing_active_slot else [0, 6]
            for t5 in buckets:
                row_cpu = 0.0 if vm_id == "od_00" and t5 == 0 else cpu
                usage_rows.append(
                    {
                        "scenario_id": 0,
                        "vm_id": vm_id,
                        "t5_day": t5,
                        "t_hour": t5 // 12,
                        "cpu_usage": row_cpu,
                        "mem_usage": mem,
                        "coverage_us": 300_000_000,
                        "coverage_ratio": 1.0,
                        "max_cpu_usage": cpu,
                        "max_mem_usage": mem,
                        "assigned_memory": q_mem,
                    }
                )

    # Later scenarios are deliberately present.  The derived pool must store
    # only scenario zero because the experiment loader creates scenarios 1..9.
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
    (root / "metadata.json").write_text('{"fixture": true}\n', encoding="utf-8")


def test_builder_selects_average_usage_top_ten_and_proves_five_servers(
    tmp_path: Path,
) -> None:
    module = _load_builder_module()
    source = tmp_path / "canonical"
    output = tmp_path / "derived"
    source.mkdir()
    _write_canonical_fixture(source)
    source_hashes = {path.name: _digest(path) for path in source.iterdir()}

    report = module.build_micro_stress_dataset(source, output)

    assert {path.name: _digest(path) for path in source.iterdir()} == source_hashes
    requests = pd.read_csv(output / "vm_requests.csv", dtype={"vm_id": str})
    usage = pd.read_csv(output / "vm_usage_5min_scenarios.csv", dtype={"vm_id": str})
    manifest = pd.read_csv(output / "selection_manifest.csv", dtype={"vm_id": str})

    assert requests["class"].value_counts().to_dict() == {
        "on_demand": 10,
        "spot": 10,
        "batch_candidate": 10,
    }
    assert set(requests.loc[requests["class"].eq("on_demand"), "vm_id"]) == {
        f"od_{number:02d}" for number in range(10)
    }
    assert "od_10" not in set(requests["vm_id"])
    assert manifest.sort_values(
        ["class", "selection_rank_within_class"]
    )["selection_score_avg_cpu_plus_mem"].notna().all()

    assert usage["scenario_id"].unique().tolist() == [0]
    assert not usage.duplicated(["vm_id", "t5_day"]).any()
    indexed_q_cpu = requests.set_index("vm_id")["q_cpu"]
    indexed_q_mem = requests.set_index("vm_id")["q_mem"]
    assert (usage["cpu_usage"] <= usage["vm_id"].map(indexed_q_cpu) + 1e-12).all()
    assert (usage["mem_usage"] <= usage["vm_id"].map(indexed_q_mem) + 1e-12).all()
    assert usage.loc[
        usage["vm_id"].eq("od_00") & usage["t5_day"].eq(0), "cpu_usage"
    ].item() == 0.0

    batch_pairs = requests.loc[
        requests["class"].eq("batch_candidate"), ["q_cpu", "q_mem"]
    ].drop_duplicates()
    assert sorted(map(tuple, batch_pairs.to_numpy(float))) == [
        (0.47, 0.47),
        (0.48, 0.48),
        (0.49, 0.49),
    ]
    assert report["od_30min_memory"]["aggregate_server_lower_bound"] == 5
    assert report["od_30min_memory"]["selection_used_peak_slot"] is False
    assert report["scenario_zero_od_two_dimensional_packing"][
        "maximum_minimum_bins"
    ] == 5
    assert report["canonical_source_integrity"]["unchanged"] is True
    probabilities = pd.read_csv(output / "scenario_probabilities.csv")
    assert probabilities["scenario_id"].tolist() == list(range(10))
    assert np.isclose(probabilities["probability"].sum(), 1.0)


def test_builder_is_deterministic_and_batch_groups_follow_average_rank(
    tmp_path: Path,
) -> None:
    module = _load_builder_module()
    source = tmp_path / "canonical"
    first = tmp_path / "first"
    second = tmp_path / "second"
    source.mkdir()
    _write_canonical_fixture(source)

    module.build_micro_stress_dataset(source, first)
    module.build_micro_stress_dataset(source, second)

    for name in (
        "vm_requests.csv",
        "vm_usage_5min_scenarios.csv",
        "selection_manifest.csv",
        "scenario_probabilities.csv",
        "validation_report.json",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()

    manifest = pd.read_csv(first / "selection_manifest.csv")
    batch = manifest.loc[manifest["class"].eq("batch_candidate")].sort_values(
        "selection_rank_within_class"
    )
    assert batch.iloc[:4][["target_q_cpu", "target_q_mem"]].drop_duplicates().values.tolist() == [
        [0.49, 0.49]
    ]
    assert batch.iloc[4:7][["target_q_cpu", "target_q_mem"]].drop_duplicates().values.tolist() == [
        [0.48, 0.48]
    ]
    assert batch.iloc[7:][["target_q_cpu", "target_q_mem"]].drop_duplicates().values.tolist() == [
        [0.47, 0.47]
    ]


def test_exact_two_dimensional_bin_check_handles_joint_resources() -> None:
    module = _load_builder_module()
    items = [(0.49, 0.49)] * 9

    assert not module._fits_two_dimensional_bins(items, 4)
    assert module._fits_two_dimensional_bins(items, 5)
    assert module._minimum_two_dimensional_bins(items) == 5
