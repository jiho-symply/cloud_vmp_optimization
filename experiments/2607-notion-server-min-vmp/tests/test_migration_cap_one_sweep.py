from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = (
    EXPERIMENT_ROOT / "run_micro_stress_migration_cap_one_sweep.py"
)
ENTRYPOINT_PATH = EXPERIMENT_ROOT / "run_experiment_migration_cap_one.py"
CONFIG_PATH = EXPERIMENT_ROOT / "configs/micro_stress_migration_cap_one.yaml"
PLAN_PATH = (
    EXPERIMENT_ROOT
    / "plans/micro_stress_first_stage_migration_cap_one_3h.yaml"
)


def _load_launcher():
    spec = importlib.util.spec_from_file_location(
        "notion_server_min_migration_cap_one_sweep_test", LAUNCHER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_plan_reuses_exact_first_stage_ofat_design_with_cap_one() -> None:
    launcher = _load_launcher()
    config = _read_yaml(CONFIG_PATH)
    plan = _read_yaml(PLAN_PATH)

    launcher._validate_contract(config, plan)
    candidates = launcher.BASE.SHARED._build_candidate_specs(
        config, plan["stages"]["first_stage"]
    )

    assert launcher.DEFAULT_CONFIG == CONFIG_PATH
    assert launcher.DEFAULT_PLAN == PLAN_PATH
    assert plan["status"] == "immutable_launch_contract"
    assert plan["formulation"]["id"] == (
        "notion_server_min_migration_cap_one_v4_20260724"
    )
    assert plan["base_config"] == "configs/micro_stress_migration_cap_one.yaml"
    assert config["migration"]["max_count_per_vm_per_scenario"] == 1
    assert type(config["migration"]["max_count_per_vm_per_scenario"]) is int
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
    assert len(candidates) == 11
    assert all(
        candidate["config"]["migration"]["max_count_per_vm_per_scenario"]
        == 1
        for candidate in candidates
    )
    assert all(
        factor["config_path"] != "migration.max_count_per_vm_per_scenario"
        for factor in plan["stages"]["first_stage"]["factors"]
    )


def test_plan_pins_six_by_sixteen_three_hour_resource_policy() -> None:
    launcher = _load_launcher()
    config = _read_yaml(CONFIG_PATH)
    plan = _read_yaml(PLAN_PATH)

    launcher._validate_contract(config, plan)

    assert config["solver"] == {
        "mip_gap": 0.001,
        "threads": 16,
        "time_limit_seconds": 10800,
        "seed": 42,
        "numeric_focus": 1,
        "presolve": 2,
        "soft_mem_limit_gb": 64,
        "nodefile_start_gb": 0.5,
        "nodefile_dir": "nodefiles",
        "write_lp": False,
        "write_mps": False,
    }
    assert plan["execution_policy"] == {
        "max_parallel_jobs": 6,
        "threads_per_job": 16,
        "total_thread_limit": 96,
        "time_limit_seconds": 10800,
        "mip_gap": 0.001,
        "soft_mem_limit_gb_per_job": 64,
        "memory_limit_enforcement": (
            "gurobi_softmemlimit_plus_aggregate_preflight"
        ),
        "nodefile_start_gb": 0.5,
        "safety": {
            "memory_reserve_gb": 64,
            "disk_reserve_gb": 60,
            "write_lp": False,
            "write_mps": False,
            "require_disjoint_cpu_affinity": True,
        },
    }
    assert (
        plan["execution_policy"]["max_parallel_jobs"]
        * plan["execution_policy"]["threads_per_job"]
        == 96
    )


@pytest.mark.parametrize("invalid", [True, False, 0, 2, 1.0, "1", None])
def test_launcher_rejects_nonliteral_cap_one(invalid: object) -> None:
    launcher = _load_launcher()
    config = _read_yaml(CONFIG_PATH)
    plan = _read_yaml(PLAN_PATH)
    config["migration"]["max_count_per_vm_per_scenario"] = invalid

    with pytest.raises(
        ValueError, match=r"migration\.max_count_per_vm_per_scenario.*integer 1"
    ):
        launcher._validate_contract(config, plan)


def test_launcher_rejects_worker_or_solver_policy_drift() -> None:
    launcher = _load_launcher()
    config = _read_yaml(CONFIG_PATH)
    plan = _read_yaml(PLAN_PATH)

    worker_drift = copy.deepcopy(plan)
    worker_drift["execution_policy"]["max_parallel_jobs"] = 5
    with pytest.raises(ValueError, match="execution policy drifted"):
        launcher._validate_contract(config, worker_drift)

    solver_drift = copy.deepcopy(config)
    solver_drift["solver"]["soft_mem_limit_gb"] = 80
    with pytest.raises(ValueError, match="base solver contract drifted"):
        launcher._validate_contract(solver_drift, plan)


def test_adapter_patches_both_entrypoint_globals_and_hashes_every_source() -> None:
    launcher = _load_launcher()
    plan = _read_yaml(PLAN_PATH)
    base_path = EXPERIMENT_ROOT / plan["base_config"]
    formulation = launcher.BASE._resolve_formulation(plan)

    hashes = launcher._capture_source_hashes(
        plan_path=PLAN_PATH,
        base_path=base_path,
        formulation=formulation,
    )

    assert launcher.BASE.SINGLE_RUN_ENTRYPOINT == ENTRYPOINT_PATH
    assert launcher.BASE.SHARED.SINGLE_RUN_ENTRYPOINT == ENTRYPOINT_PATH
    required_sources = {
        LAUNCHER_PATH.resolve(),
        ENTRYPOINT_PATH.resolve(),
        launcher.BASE_LAUNCHER_PATH.resolve(),
        (EXPERIMENT_ROOT / "run_sweep.py").resolve(),
        CONFIG_PATH.resolve(),
        PLAN_PATH.resolve(),
        (EXPERIMENT_ROOT / "src/notion_server_min_vmp/model.py").resolve(),
        Path(formulation["change_summary_path"]).resolve(),
        launcher._resolve_nyiso_price_path(base_path),
    }
    assert required_sources.issubset({Path(path).resolve() for path in hashes})
    assert all(len(digest) == 64 for digest in hashes.values())
    price_path = launcher._resolve_nyiso_price_path(base_path)
    assert hashes[str(price_path)] == launcher.BASE.SHARED._sha256(price_path)


def test_materialized_candidates_keep_cap_and_sixteen_thread_limits(
    tmp_path: Path,
) -> None:
    launcher = _load_launcher()
    config = _read_yaml(CONFIG_PATH)
    plan = _read_yaml(PLAN_PATH)
    specs = launcher.BASE.SHARED._build_candidate_specs(
        config, plan["stages"]["first_stage"]
    )

    rows = launcher.BASE.SHARED._materialize_candidates(
        specs=specs,
        plan=plan,
        plan_path=PLAN_PATH,
        base_path=CONFIG_PATH,
        stage_name="first_stage",
        suite_root=tmp_path,
    )

    assert len(rows) == 11
    for row in rows:
        derived = _read_yaml(Path(row["config_path"]))
        assert derived["migration"]["max_count_per_vm_per_scenario"] == 1
        assert type(
            derived["migration"]["max_count_per_vm_per_scenario"]
        ) is int
        assert row["threads"] == derived["solver"]["threads"] == 16
        assert row["time_limit_seconds"] == pytest.approx(10800.0)
        assert derived["solver"]["time_limit_seconds"] == pytest.approx(10800.0)
        assert row["soft_mem_limit_gb"] == pytest.approx(64.0)
        assert derived["solver"]["soft_mem_limit_gb"] == pytest.approx(64.0)


def test_resource_preflight_requires_all_96_cpus_and_448_gib(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    launcher = _load_launcher()
    policy = _read_yaml(PLAN_PATH)["execution_policy"]
    shared = launcher.BASE.SHARED
    gib = 1024**3
    monkeypatch.setattr(shared, "_available_cpus", lambda: list(range(96)))
    monkeypatch.setattr(shared, "_memory_available_gib", lambda: 500.0)
    monkeypatch.setattr(
        shared.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=200 * gib, used=100 * gib, free=100 * gib),
    )

    audit = shared._resource_preflight(policy, tmp_path)

    assert audit["required_solver_threads"] == 96
    assert audit["total_thread_limit"] == 96
    assert audit["required_memory_including_reserve_gib"] == pytest.approx(448.0)

    monkeypatch.setattr(shared, "_available_cpus", lambda: list(range(95)))
    with pytest.raises(ValueError, match="96 solver threads.*95 CPUs"):
        shared._resource_preflight(policy, tmp_path)


def test_runtime_hash_wrapper_records_and_passes_combined_launch_hashes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    launcher = _load_launcher()
    suite_root = tmp_path / "suite"
    suite_root.mkdir()
    source_path = tmp_path / "source.py"
    dataset_path = tmp_path / "dataset.csv"
    config_path = tmp_path / "candidate.yaml"
    source_path.write_text("source\n", encoding="utf-8")
    dataset_path.write_text("dataset\n", encoding="utf-8")
    config_path.write_text("config\n", encoding="utf-8")
    source_hashes = {
        str(source_path.resolve()): launcher.BASE.SHARED._sha256(source_path)
    }
    dataset_hashes = {
        str(dataset_path.resolve()): launcher.BASE.SHARED._sha256(dataset_path)
    }
    rows = [
        {
            "config_path": str(config_path.resolve()),
            "config_sha256": launcher.BASE.SHARED._sha256(config_path),
        }
    ]
    metadata_path = suite_root / "suite_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "candidate_count": 1,
                "dataset_files_sha256": dataset_hashes,
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, str] = {}

    def fake_execute(
        _rows: list[dict],
        *,
        policy: dict,
        suite_root: Path,
        required_source_hashes: dict[str, str],
    ) -> list[dict]:
        del policy, suite_root
        captured.update(required_source_hashes)
        return []

    monkeypatch.setattr(launcher, "_ORIGINAL_EXECUTE_CANDIDATES", fake_execute)

    result = launcher._execute_candidates_with_runtime_hashes(
        rows,
        policy={},
        suite_root=suite_root,
        required_source_hashes=source_hashes,
    )

    assert result == []
    expected_paths = {
        str(source_path.resolve()),
        str(dataset_path.resolve()),
        str(config_path.resolve()),
    }
    assert set(captured) == expected_paths
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["required_launch_files_sha256"] == captured


@pytest.mark.parametrize("drift_target", ["dataset", "config"])
def test_runtime_hash_wrapper_detects_input_and_config_drift_before_popen(
    drift_target: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    launcher = _load_launcher()
    suite_root = tmp_path / "suite"
    suite_root.mkdir()
    source_path = tmp_path / "source.py"
    dataset_path = tmp_path / "dataset.csv"
    config_path = tmp_path / "candidate.yaml"
    for path, value in (
        (source_path, "source\n"),
        (dataset_path, "dataset\n"),
        (config_path, "config\n"),
    ):
        path.write_text(value, encoding="utf-8")
    source_hashes = {
        str(source_path.resolve()): launcher.BASE.SHARED._sha256(source_path)
    }
    dataset_hashes = {
        str(dataset_path.resolve()): launcher.BASE.SHARED._sha256(dataset_path)
    }
    rows = [
        {
            "config_path": str(config_path.resolve()),
            "config_sha256": launcher.BASE.SHARED._sha256(config_path),
        }
    ]
    (suite_root / "suite_metadata.json").write_text(
        json.dumps(
            {
                "candidate_count": 1,
                "dataset_files_sha256": dataset_hashes,
            }
        ),
        encoding="utf-8",
    )
    drift_path = dataset_path if drift_target == "dataset" else config_path

    def fake_execute(
        _rows: list[dict],
        *,
        policy: dict,
        suite_root: Path,
        required_source_hashes: dict[str, str],
    ) -> list[dict]:
        del policy, suite_root
        drift_path.write_text("drifted\n", encoding="utf-8")
        launcher.BASE.SHARED._verify_required_source_hashes(
            required_source_hashes
        )
        return []

    monkeypatch.setattr(launcher, "_ORIGINAL_EXECUTE_CANDIDATES", fake_execute)

    with pytest.raises(RuntimeError, match="Captured source hash changed"):
        launcher._execute_candidates_with_runtime_hashes(
            rows,
            policy={},
            suite_root=suite_root,
            required_source_hashes=source_hashes,
        )
