from __future__ import annotations

import copy
import importlib.util
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest
import yaml

from conftest import EXPERIMENT_ROOT


LAUNCHER_PATH = EXPERIMENT_ROOT / "run_micro_stress_paired_migration_sweep.py"
PLAN_PATH = (
    EXPERIMENT_ROOT / "plans/micro_stress_first_stage_3h_migration_pair.yaml"
)


def _load_launcher():
    spec = importlib.util.spec_from_file_location(
        "notion_paired_migration_sweep_test", LAUNCHER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _plan_base_and_candidates():
    launcher = _load_launcher()
    plan = _read_yaml(PLAN_PATH)
    base_path = (EXPERIMENT_ROOT / plan["base_config"]).resolve()
    base = _read_yaml(base_path)
    variants, specs = launcher._validate_plan_contract(plan, base)
    return launcher, plan, base_path, variants, specs


def _without_policy_only_differences(config: dict) -> dict:
    comparable = copy.deepcopy(config)
    del comparable["experiment"]["name"]
    del comparable["migration"]["fixed_zero"]
    return comparable


def test_materialization_has_exactly_22_interleaved_matched_rows(
    tmp_path: Path,
) -> None:
    launcher, plan, base_path, variants, specs = _plan_base_and_candidates()
    suite_root = tmp_path / "paired-suite"
    suite_root.mkdir()

    rows = launcher._materialize_paired_candidates(
        plan=plan,
        plan_path=PLAN_PATH,
        base_path=base_path,
        variants=variants,
        specs=specs,
        suite_root=suite_root,
    )

    assert len(specs) == 11
    assert len(rows) == 22
    assert Counter(row["variant_id"] for row in rows) == {
        "migration_allowed": 11,
        "migration_prohibited": 11,
    }
    assert [row["run_id"] for row in rows] == [
        f"{variant_id}/{spec['run_id']}"
        for spec in specs
        for variant_id in ("migration_allowed", "migration_prohibited")
    ]
    assert [row["comparison_pair_id"] for row in rows] == [
        spec["run_id"] for spec in specs for _variant in range(2)
    ]
    assert (suite_root / "manifest.csv").is_file()


def test_every_pair_differs_only_by_migration_policy_name_and_entrypoint(
    tmp_path: Path,
) -> None:
    launcher, plan, base_path, variants, specs = _plan_base_and_candidates()
    suite_root = tmp_path / "paired-suite"
    suite_root.mkdir()
    rows = launcher._materialize_paired_candidates(
        plan=plan,
        plan_path=PLAN_PATH,
        base_path=base_path,
        variants=variants,
        specs=specs,
        suite_root=suite_root,
    )
    by_pair: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_pair[row["comparison_pair_id"]].append(row)

    assert set(by_pair) == {str(spec["run_id"]) for spec in specs}
    for pair_id, pair in by_pair.items():
        assert [row["variant_id"] for row in pair] == [
            "migration_allowed",
            "migration_prohibited",
        ]
        allowed_row, prohibited_row = pair
        allowed = _read_yaml(Path(allowed_row["config_path"]))
        prohibited = _read_yaml(Path(prohibited_row["config_path"]))

        assert allowed["migration"]["fixed_zero"] is False
        assert prohibited["migration"]["fixed_zero"] is True
        assert allowed["experiment"]["name"] != prohibited["experiment"]["name"]
        assert allowed["experiment"]["name"].endswith(
            f"__migration_allowed__{pair_id.replace('/', '__')}"
        )
        assert prohibited["experiment"]["name"].endswith(
            f"__migration_prohibited__{pair_id.replace('/', '__')}"
        )
        assert Path(allowed_row["entrypoint"]).name == "run_experiment.py"
        assert (
            Path(prohibited_row["entrypoint"]).name
            == "run_experiment_no_migration.py"
        )
        assert _without_policy_only_differences(allowed) == (
            _without_policy_only_differences(prohibited)
        )
        for field in (
            "factor_id",
            "level_id",
            "override_path",
            "override_value_json",
            "threads",
            "time_limit_seconds",
            "mip_gap",
            "soft_mem_limit_gb",
        ):
            assert allowed_row[field] == prohibited_row[field]


def test_plan_pins_three_hour_five_by_sixteen_execution_contract() -> None:
    launcher, plan, _base_path, variants, specs = _plan_base_and_candidates()
    policy = plan["execution_policy"]

    assert launcher.DEFAULT_PLAN == PLAN_PATH
    assert len(variants) == 2
    assert len(specs) == 11
    assert policy["time_limit_seconds"] == 10_800
    assert policy["max_parallel_jobs"] == 5
    assert policy["threads_per_job"] == 16
    assert policy["total_thread_limit"] == 80
    assert policy["soft_mem_limit_gb_per_job"] == 80
    assert policy["mip_gap"] == pytest.approx(0.001)
    assert policy["safety"]["require_disjoint_cpu_affinity"] is True


def test_no_migration_variant_retains_all_migration_coefficient_controls(
    tmp_path: Path,
) -> None:
    launcher, plan, base_path, variants, specs = _plan_base_and_candidates()
    suite_root = tmp_path / "paired-suite"
    suite_root.mkdir()
    rows = launcher._materialize_paired_candidates(
        plan=plan,
        plan_path=PLAN_PATH,
        base_path=base_path,
        variants=variants,
        specs=specs,
        suite_root=suite_root,
    )

    prohibited = [
        row for row in rows if row["variant_id"] == "migration_prohibited"
    ]
    migration_controls = [
        row for row in prohibited if row["factor_id"] == "migration_coefficient"
    ]
    assert [row["comparison_pair_id"] for row in migration_controls] == [
        "migration_coefficient/c_mig_0",
        "migration_coefficient/c_mig_0_1",
        "migration_coefficient/c_mig_1",
    ]
    assert [yaml.safe_load(row["override_value_json"]) for row in migration_controls] == [
        0.0,
        0.1,
        1.0,
    ]
    assert all(row["migration_fixed_zero"] is True for row in migration_controls)


def test_source_hash_manifest_includes_pair_launcher_and_both_entrypoints() -> None:
    launcher, plan, base_path, _variants, _specs = _plan_base_and_candidates()

    source_hashes = launcher._capture_source_hashes(
        plan=plan,
        plan_path=PLAN_PATH,
        base_path=base_path,
    )

    required = {
        LAUNCHER_PATH.resolve(),
        (EXPERIMENT_ROOT / "run_micro_stress_sweep.py").resolve(),
        (EXPERIMENT_ROOT / "run_sweep.py").resolve(),
        (EXPERIMENT_ROOT / "run_experiment.py").resolve(),
        (EXPERIMENT_ROOT / "run_experiment_no_migration.py").resolve(),
    }
    assert {str(path) for path in required} <= set(source_hashes)
    for path in required:
        assert source_hashes[str(path)] == launcher.SHARED._sha256(path)


def test_predecessor_wait_rejects_recycled_pid_with_mismatched_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = _load_launcher()
    predecessor = tmp_path / "prior-suite"
    predecessor.mkdir()
    (predecessor / "suite.pid").write_text("12345\n", encoding="utf-8")
    monkeypatch.setattr(
        launcher,
        "_process_command",
        lambda _pid: "python unrelated_worker.py --still-alive",
    )

    with pytest.raises(RuntimeError, match="command no longer matches"):
        launcher._wait_for_predecessor(
            predecessor,
            queue_state_path=tmp_path / "queue_state.json",
            poll_seconds=0.0,
        )
