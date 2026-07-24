from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from conftest import EXPERIMENT_ROOT


def _load_launcher():
    path = EXPERIMENT_ROOT / "run_micro_stress_sweep.py"
    spec = importlib.util.spec_from_file_location("notion_micro_stress_sweep", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_yaml(relative_path: str) -> dict:
    return yaml.safe_load((EXPERIMENT_ROOT / relative_path).read_text(encoding="utf-8"))


def test_micro_stress_baseline_has_requested_dimensions_and_solver_limits() -> None:
    config = _read_yaml("configs/micro_stress_baseline.yaml")

    assert config["experiment"]["num_scenarios"] == 10
    assert config["experiment"]["class_counts"] == {
        "on_demand": 10,
        "spot": 10,
        "batch_jobs": 10,
    }
    assert config["experiment"]["num_servers"] == 6
    assert config["batch"]["max_families"] == 3
    assert config["solver"]["mip_gap"] == 0.001
    assert config["solver"]["threads"] == 16
    assert config["solver"]["time_limit_seconds"] == 3600
    assert config["solver"]["soft_mem_limit_gb"] == 80
    assert config["data"]["google_dir"].startswith("__REQUIRED_VIA_")


def test_micro_stress_plan_is_deduplicated_first_stage_ofat() -> None:
    launcher = _load_launcher()
    plan = _read_yaml("plans/micro_stress_first_stage.yaml")
    base = _read_yaml("configs/micro_stress_baseline.yaml")

    candidates = launcher.SHARED._build_candidate_specs(
        base, plan["stages"]["first_stage"]
    )

    assert [candidate["run_id"] for candidate in candidates] == [
        "baseline",
        "spot_discount_ratio/gamma_0",
        "spot_discount_ratio/gamma_0_1",
        "spot_discount_ratio/gamma_0_5",
        "migration_coefficient/c_mig_0",
        "migration_coefficient/c_mig_0_1",
        "migration_coefficient/c_mig_1",
        "batch_startup_cpu/startup_cpu_0",
        "batch_startup_cpu/startup_cpu_0_1",
        "batch_base_mem/base_mem_0",
        "batch_base_mem/base_mem_0_1",
    ]
    assert len(candidates) == plan["generation"]["exact_unique_candidate_configs"]["first_stage"]


def test_symbreak_v2_plan_preserves_design_and_pins_formulation_source() -> None:
    launcher = _load_launcher()
    plan_path = EXPERIMENT_ROOT / "plans/micro_stress_first_stage_symbreak_v2.yaml"
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    base_path = EXPERIMENT_ROOT / plan["base_config"]
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))

    candidates = launcher.SHARED._build_candidate_specs(
        base, plan["stages"]["first_stage"]
    )
    formulation = launcher._resolve_formulation(plan)
    assert formulation is not None
    source_hashes = launcher._capture_source_hashes(
        plan_path=plan_path,
        base_path=base_path,
        formulation=formulation,
    )

    assert launcher.DEFAULT_PLAN == plan_path
    assert plan["suite_name"] == "notion_server_min_micro_stress_first_stage_symbreak_v2"
    assert plan["output_root"] == "runs/notion_server_min_micro_stress_first_stage_symbreak_v2"
    assert len(candidates) == 11
    assert formulation == {
        "id": "notion_server_min_symbreak_v2_20260722",
        "spec_sha256": "1dd1dd55b38833f1d229ed018179e349a25e70e28edb66e80e97b9b7d28fd607",
        "spec_verified": False,
        "change_summary_path": formulation["change_summary_path"],
        "change_summary_sha256": formulation["change_summary_sha256"],
    }
    assert plan["formulation"]["spec_path"] is None
    assert str((EXPERIMENT_ROOT / "run_sweep.py").resolve()) in source_hashes
    assert str((EXPERIMENT_ROOT / "run_micro_stress_sweep.py").resolve()) in source_hashes
    assert str((EXPERIMENT_ROOT / "src/notion_server_min_vmp/model.py").resolve()) in source_hashes
    assert formulation["change_summary_path"] in source_hashes
    legacy_plan = _read_yaml("plans/micro_stress_first_stage.yaml")
    assert launcher._resolve_formulation(legacy_plan) is None


def test_external_formulation_spec_is_optional_but_hash_verified(
    tmp_path: Path,
) -> None:
    launcher = _load_launcher()
    plan_path = EXPERIMENT_ROOT / "plans/micro_stress_first_stage_symbreak_v2.yaml"
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    base_path = EXPERIMENT_ROOT / plan["base_config"]
    external_spec = tmp_path / "formulation.md"
    external_spec.write_text("portable external formulation\n", encoding="utf-8")
    plan["formulation"]["spec_sha256"] = launcher.SHARED._sha256(external_spec)

    formulation = launcher._resolve_formulation(plan, external_spec)
    assert formulation is not None
    assert formulation["spec_path"] == str(external_spec.resolve())
    assert formulation["spec_verified"] is True
    source_hashes = launcher._capture_source_hashes(
        plan_path=plan_path,
        base_path=base_path,
        formulation=formulation,
    )
    assert source_hashes[str(external_spec.resolve())] == formulation["spec_sha256"]

    plan["formulation"]["spec_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        launcher._resolve_formulation(plan, external_spec)


def test_micro_stress_policy_uses_five_disjoint_sixteen_core_jobs_and_memory_preflight() -> None:
    plan = _read_yaml("plans/micro_stress_first_stage.yaml")
    policy = plan["execution_policy"]

    assert policy["max_parallel_jobs"] == 5
    assert policy["threads_per_job"] == 16
    assert policy["total_thread_limit"] == 80
    assert policy["time_limit_seconds"] == 3600
    assert policy["mip_gap"] == 0.001
    assert policy["soft_mem_limit_gb_per_job"] == 80
    assert (
        policy["memory_limit_enforcement"]
        == "gurobi_softmemlimit_plus_aggregate_preflight"
    )
    assert policy["safety"]["memory_reserve_gb"] == 64


def test_google_dir_is_required_and_materialized_as_absolute_path(tmp_path: Path) -> None:
    launcher = _load_launcher()
    with pytest.raises(FileNotFoundError, match="not a directory"):
        launcher._resolve_google_dir(tmp_path / "absent")

    data_dir = tmp_path / "micro"
    data_dir.mkdir()
    for name in launcher.REQUIRED_DATA_FILES:
        (data_dir / name).touch()
    (data_dir / "metadata.json").write_text("{}", encoding="utf-8")

    resolved = launcher._resolve_google_dir(data_dir)
    config = launcher._effective_base_config(
        _read_yaml("configs/micro_stress_baseline.yaml"), resolved
    )

    assert config["data"]["google_dir"] == str(data_dir.resolve())
    assert str((data_dir / "metadata.json").resolve()) in config["data"]["additional_source_files"]


def test_dataset_shape_preserves_builder_selection_without_loader_reselection(tmp_path: Path) -> None:
    launcher = _load_launcher()
    data_dir = tmp_path / "micro"
    data_dir.mkdir()
    request_lines = ["vm_id,class"]
    request_lines.extend(f"od-{index},on_demand" for index in range(10))
    request_lines.extend(f"sp-{index},spot" for index in range(10))
    request_lines.extend(f"ba-{index},batch_candidate" for index in range(10))
    (data_dir / "vm_requests.csv").write_text("\n".join(request_lines) + "\n", encoding="utf-8")
    (data_dir / "servers.csv").write_text(
        "server_id\n" + "".join(f"server-{index}\n" for index in range(6)),
        encoding="utf-8",
    )

    audit = launcher._validate_dataset_shape(data_dir)

    assert audit["vm_request_rows"] == 30
    assert audit["class_counts"] == {
        "on_demand": 10,
        "spot": 10,
        "batch_candidate": 10,
    }

    with (data_dir / "vm_requests.csv").open("a", encoding="utf-8") as stream:
        stream.write("od-extra,on_demand\n")
    with pytest.raises(ValueError, match="exactly the selected 10/10/10"):
        launcher._validate_dataset_shape(data_dir)


def test_dataset_hash_manifest_covers_every_materialized_input_file(tmp_path: Path) -> None:
    launcher = _load_launcher()
    data_dir = tmp_path / "micro"
    data_dir.mkdir()
    first = data_dir / "vm_requests.csv"
    second = data_dir / "vm_usage_5min_scenarios.csv"
    first.write_text("first\n", encoding="utf-8")
    second.write_text("second\n", encoding="utf-8")

    hashes = launcher._capture_dataset_hashes(data_dir)

    assert set(hashes) == {str(first.resolve()), str(second.resolve())}
    assert hashes[str(first.resolve())] == launcher.SHARED._sha256(first)
