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
    EXPERIMENT_ROOT
    / "run_micro_stress_fixed_zero_migration_excess_preemption_sweep.py"
)
ENTRYPOINT_PATH = (
    EXPERIMENT_ROOT
    / "run_experiment_fixed_zero_migration_excess_preemption.py"
)
V5_ENTRYPOINT_PATH = (
    EXPERIMENT_ROOT / "run_experiment_fixed_zero_migration_excess.py"
)
CONFIG_PATH = (
    EXPERIMENT_ROOT
    / "configs"
    / "micro_stress_fixed_zero_migration_excess_preemption.yaml"
)
PLAN_PATH = (
    EXPERIMENT_ROOT
    / "plans"
    / "micro_stress_first_six_fixed_zero_migration_excess_preemption_3h.yaml"
)
CHANGE_SUMMARY_PATH = (
    EXPERIMENT_ROOT
    / "FORMULATION_CHANGES_FIXED_ZERO_MIGRATION_EXCESS_PREEMPTION_V6.md"
)
EXPECTED_RUN_IDS = [
    "baseline",
    "spot_discount_ratio/gamma_0",
    "spot_discount_ratio/gamma_0_1",
    "spot_discount_ratio/gamma_0_5",
    "migration_coefficient/c_mig_0",
    "migration_coefficient/c_mig_0_1",
]
GUARD_SECTIONS = (
    "migration",
    "excess_load_indicator",
    "spot_preemption",
)


def _load_launcher():
    spec = importlib.util.spec_from_file_location(
        "notion_server_min_cumulative_fixed_zero_sweep_test",
        LAUNCHER_PATH,
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


def _runtime_hash_fixture(tmp_path: Path, launcher):
    suite_root = tmp_path / "suite"
    suite_root.mkdir()
    source_path = tmp_path / "source.py"
    dataset_path = tmp_path / "dataset.csv"
    source_path.write_text("source\n", encoding="utf-8")
    dataset_path.write_text("dataset\n", encoding="utf-8")
    source_hashes = {
        str(source_path.resolve()): launcher.BASE.SHARED._sha256(source_path)
    }
    dataset_hashes = {
        str(dataset_path.resolve()): launcher.BASE.SHARED._sha256(dataset_path)
    }

    rows = []
    config_paths = []
    for index, run_id in enumerate(EXPECTED_RUN_IDS):
        config_path = tmp_path / f"candidate-{index}.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "migration": {"fixed_zero": True},
                    "excess_load_indicator": {"fixed_zero": True},
                    "spot_preemption": {"fixed_zero": True},
                    "solver": {
                        "threads": 16,
                        "time_limit_seconds": 10800,
                        "mip_gap": 0.001,
                        "soft_mem_limit_gb": 64,
                        "nodefile_start_gb": 0.5,
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        config_paths.append(config_path)
        rows.append(
            {
                "run_id": run_id,
                "config_path": str(config_path.resolve()),
                "config_sha256": launcher.BASE.SHARED._sha256(config_path),
            }
        )

    metadata_path = suite_root / "suite_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "candidate_count": len(EXPECTED_RUN_IDS),
                "dataset_files_sha256": dataset_hashes,
            }
        ),
        encoding="utf-8",
    )
    return {
        "suite_root": suite_root,
        "source_path": source_path,
        "dataset_path": dataset_path,
        "config_paths": config_paths,
        "source_hashes": source_hashes,
        "rows": rows,
        "metadata_path": metadata_path,
    }


def test_plan_and_config_pin_cumulative_v6_and_exact_prior_six() -> None:
    launcher = _load_launcher()
    config = _read_yaml(CONFIG_PATH)
    plan = _read_yaml(PLAN_PATH)

    launcher._validate_contract(config, plan)
    candidates = launcher.BASE.SHARED._build_candidate_specs(
        config,
        plan["stages"]["first_stage"],
    )

    assert launcher.DEFAULT_CONFIG == CONFIG_PATH
    assert launcher.DEFAULT_PLAN == PLAN_PATH
    assert launcher.SINGLE_RUN_ENTRYPOINT == ENTRYPOINT_PATH
    assert launcher.CHANGE_SUMMARY == CHANGE_SUMMARY_PATH
    assert plan["schema_version"] == 6
    assert plan["status"] == "immutable_launch_contract"
    assert plan["formulation"] == {
        "id": (
            "notion_server_min_fixed_zero_migration_excess_"
            "preemption_v6_20260725"
        ),
        "spec_path": None,
        "spec_sha256": (
            "bf2b4dc77d9fd3cf26b6d7f3314113454103bd50767e9f72b9f4dc3d3c2e26d7"
        ),
        "change_summary_path": CHANGE_SUMMARY_PATH.name,
    }
    assert plan["generation"]["exact_unique_candidate_configs"] == {
        "first_stage": 6
    }
    assert [candidate["run_id"] for candidate in candidates] == EXPECTED_RUN_IDS
    assert len(candidates) == 6
    for section in GUARD_SECTIONS:
        assert config[section]["fixed_zero"] is True
        assert type(config[section]["fixed_zero"]) is bool
        assert all(
            candidate["config"][section]["fixed_zero"] is True
            for candidate in candidates
        )

    fixed_paths = {
        "migration.fixed_zero",
        "excess_load_indicator.fixed_zero",
        "spot_preemption.fixed_zero",
    }
    assert all(
        factor["config_path"] not in fixed_paths
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


@pytest.mark.parametrize("section", GUARD_SECTIONS)
@pytest.mark.parametrize("invalid", [False, 1, "true", None])
def test_launcher_rejects_nonliteral_fixed_zero_guards(
    section: str,
    invalid: object,
) -> None:
    launcher = _load_launcher()
    config = _read_yaml(CONFIG_PATH)
    plan = _read_yaml(PLAN_PATH)
    config[section]["fixed_zero"] = invalid

    with pytest.raises(ValueError, match=rf"{section}\.fixed_zero"):
        launcher._validate_contract(config, plan)


def test_launcher_rejects_resource_candidate_and_validation_drift() -> None:
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

    candidate_drift = copy.deepcopy(plan)
    candidate_drift["stages"]["first_stage"]["factors"][1]["levels"].append(
        {"id": "c_mig_1", "value": 1.0}
    )
    with pytest.raises(ValueError, match="factor contract drifted"):
        launcher._validate_contract(config, candidate_drift)

    validation_drift = copy.deepcopy(plan)
    validation_drift["validation"]["spot_preemption_fixed_zero_config_path"] = (
        "spot_preemption.disabled"
    )
    with pytest.raises(ValueError, match="validation policy drifted"):
        launcher._validate_contract(config, validation_drift)


def test_plan_validation_names_all_three_literal_boolean_guards() -> None:
    launcher = _load_launcher()
    config = _read_yaml(CONFIG_PATH)
    plan = _read_yaml(PLAN_PATH)

    launcher._validate_contract(config, plan)

    assert plan["validation"]["migration_fixed_zero_config_path"] == (
        "migration.fixed_zero"
    )
    assert plan["validation"][
        "excess_load_indicator_fixed_zero_config_path"
    ] == "excess_load_indicator.fixed_zero"
    assert plan["validation"]["spot_preemption_fixed_zero_config_path"] == (
        "spot_preemption.fixed_zero"
    )
    assert plan["validation"]["require_literal_boolean_true"] is True
    assert plan["validation"]["exact_candidate_run_ids"] == EXPECTED_RUN_IDS


def test_adapter_patches_entrypoint_and_hashes_all_declared_inputs() -> None:
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
        V5_ENTRYPOINT_PATH.resolve(),
        launcher.BASE_LAUNCHER_PATH.resolve(),
        (EXPERIMENT_ROOT / "run_sweep.py").resolve(),
        CONFIG_PATH.resolve(),
        PLAN_PATH.resolve(),
        CHANGE_SUMMARY_PATH.resolve(),
        (EXPERIMENT_ROOT / "src/notion_server_min_vmp/model.py").resolve(),
        *launcher._resolve_declared_data_inputs(base_path),
    }
    hashed_paths = {Path(path).resolve() for path in hashes}
    assert required_sources.issubset(hashed_paths)
    assert all(len(digest) == 64 for digest in hashes.values())
    for path in launcher._resolve_declared_data_inputs(base_path):
        assert hashes[str(path)] == launcher.BASE.SHARED._sha256(path)


def test_materialized_candidates_keep_all_guards_and_limits(tmp_path: Path) -> None:
    launcher = _load_launcher()
    config = _read_yaml(CONFIG_PATH)
    plan = _read_yaml(PLAN_PATH)
    specs = launcher.BASE.SHARED._build_candidate_specs(
        config,
        plan["stages"]["first_stage"],
    )

    rows = launcher.BASE.SHARED._materialize_candidates(
        specs=specs,
        plan=plan,
        plan_path=PLAN_PATH,
        base_path=CONFIG_PATH,
        stage_name="first_stage",
        suite_root=tmp_path,
    )

    assert [row["run_id"] for row in rows] == EXPECTED_RUN_IDS
    for row in rows:
        derived = _read_yaml(Path(row["config_path"]))
        for section in GUARD_SECTIONS:
            assert derived[section]["fixed_zero"] is True
            assert type(derived[section]["fixed_zero"]) is bool
        assert row["threads"] == derived["solver"]["threads"] == 16
        assert row["time_limit_seconds"] == pytest.approx(10800.0)
        assert derived["solver"]["time_limit_seconds"] == pytest.approx(10800.0)
        assert row["mip_gap"] == pytest.approx(0.001)
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
        lambda _path: SimpleNamespace(
            total=200 * gib,
            used=100 * gib,
            free=100 * gib,
        ),
    )

    audit = shared._resource_preflight(policy, tmp_path)

    assert audit["required_solver_threads"] == 96
    assert audit["total_thread_limit"] == 96
    assert audit["required_memory_including_reserve_gib"] == pytest.approx(448.0)

    monkeypatch.setattr(shared, "_available_cpus", lambda: list(range(95)))
    with pytest.raises(ValueError, match="96 solver threads.*95 CPUs"):
        shared._resource_preflight(policy, tmp_path)


def test_runtime_hash_wrapper_records_source_dataset_and_six_configs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    launcher = _load_launcher()
    fixture = _runtime_hash_fixture(tmp_path, launcher)
    captured: dict[str, str] = {}

    def fake_execute(
        _rows: list[dict],
        *,
        policy: dict,
        suite_root: Path,
        required_source_hashes: dict[str, str],
    ) -> list[dict]:
        del policy, suite_root
        assert [row["run_id"] for row in _rows] == EXPECTED_RUN_IDS
        captured.update(required_source_hashes)
        return []

    monkeypatch.setattr(launcher, "_ORIGINAL_EXECUTE_CANDIDATES", fake_execute)

    result = launcher._execute_candidates_with_runtime_hashes(
        fixture["rows"],
        policy={},
        suite_root=fixture["suite_root"],
        required_source_hashes=fixture["source_hashes"],
    )

    assert result == []
    expected_paths = {
        str(fixture["source_path"].resolve()),
        str(fixture["dataset_path"].resolve()),
        *(str(path.resolve()) for path in fixture["config_paths"]),
    }
    assert set(captured) == expected_paths
    metadata = json.loads(fixture["metadata_path"].read_text(encoding="utf-8"))
    assert metadata["required_launch_files_sha256"] == captured


def test_runtime_hash_wrapper_rejects_order_or_guard_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    launcher = _load_launcher()
    fixture = _runtime_hash_fixture(tmp_path, launcher)
    called = False

    def unexpected_execute(*_args: object, **_kwargs: object) -> list[dict]:
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(
        launcher,
        "_ORIGINAL_EXECUTE_CANDIDATES",
        unexpected_execute,
    )
    reordered = list(fixture["rows"])
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(ValueError, match="runtime candidate order drifted"):
        launcher._execute_candidates_with_runtime_hashes(
            reordered,
            policy={},
            suite_root=fixture["suite_root"],
            required_source_hashes=fixture["source_hashes"],
        )
    assert called is False

    config_path = fixture["config_paths"][0]
    derived = _read_yaml(config_path)
    derived["spot_preemption"]["fixed_zero"] = False
    config_path.write_text(yaml.safe_dump(derived), encoding="utf-8")
    with pytest.raises(ValueError, match=r"spot_preemption\.fixed_zero"):
        launcher._execute_candidates_with_runtime_hashes(
            fixture["rows"],
            policy={},
            suite_root=fixture["suite_root"],
            required_source_hashes=fixture["source_hashes"],
        )
    assert called is False


@pytest.mark.parametrize("drift_target", ["source", "dataset", "config"])
def test_complete_runtime_hash_map_detects_drift_before_popen(
    drift_target: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    launcher = _load_launcher()
    fixture = _runtime_hash_fixture(tmp_path, launcher)
    drift_path = {
        "source": fixture["source_path"],
        "dataset": fixture["dataset_path"],
        "config": fixture["config_paths"][0],
    }[drift_target]

    def fake_execute(
        _rows: list[dict],
        *,
        policy: dict,
        suite_root: Path,
        required_source_hashes: dict[str, str],
    ) -> list[dict]:
        del _rows, policy, suite_root
        drift_path.write_text("drifted\n", encoding="utf-8")
        launcher.BASE.SHARED._verify_required_source_hashes(
            required_source_hashes
        )
        return []

    monkeypatch.setattr(launcher, "_ORIGINAL_EXECUTE_CANDIDATES", fake_execute)

    with pytest.raises(RuntimeError, match="Captured source hash changed"):
        launcher._execute_candidates_with_runtime_hashes(
            fixture["rows"],
            policy={},
            suite_root=fixture["suite_root"],
            required_source_hashes=fixture["source_hashes"],
        )
