from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


def _experiment_root() -> Path:
    return next(
        parent / "experiments/2607-notion-server-min-vmp"
        for parent in Path(__file__).resolve().parents
        if (parent / "experiments/2607-notion-server-min-vmp").is_dir()
    )


def test_adapter_uses_no_migration_entrypoint() -> None:
    root = _experiment_root()
    path = root / "run_no_migration_sweep.py"
    spec = importlib.util.spec_from_file_location(
        "run_no_migration_sweep_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    expected = (root / "run_experiment_no_migration.py").resolve()
    assert module.SINGLE_RUN_ENTRYPOINT.resolve() == expected
    assert module.BASE.SINGLE_RUN_ENTRYPOINT.resolve() == expected


def test_no_migration_pair_config_is_explicit() -> None:
    root = _experiment_root()
    config = yaml.safe_load(
        (
            root
            / "configs/micro_top10_unscaled_no_migration_norel0_baseline.yaml"
        ).read_text(encoding="utf-8")
    )
    plan = yaml.safe_load(
        (
            root
            / "plans/micro_top10_unscaled_no_migration_norel0_baseline_gamma_0_1_3h.yaml"
        ).read_text(encoding="utf-8")
    )

    assert config["migration"]["fixed_zero"] is True
    assert config["solver"]["no_rel_heur_time_seconds"] == 0
    assert config["solver"]["threads"] == 16
    assert config["solver"]["time_limit_seconds"] == 10_800
    assert plan["generation"]["exact_unique_candidate_configs"]["first_stage"] == 2
    assert plan["execution_policy"]["max_parallel_jobs"] == 2


def test_no_migration_cvar2_pair_changes_only_the_risk_bound() -> None:
    root = _experiment_root()
    baseline = yaml.safe_load(
        (
            root
            / "configs/micro_top10_unscaled_no_migration_norel0_baseline.yaml"
        ).read_text(encoding="utf-8")
    )
    cvar2 = yaml.safe_load(
        (
            root
            / "configs/micro_top10_unscaled_no_migration_norel0_cvar2_baseline.yaml"
        ).read_text(encoding="utf-8")
    )
    plan = yaml.safe_load(
        (
            root
            / "plans/micro_top10_unscaled_no_migration_norel0_cvar2_baseline_gamma_0_1_3h.yaml"
        ).read_text(encoding="utf-8")
    )

    expected = dict(baseline)
    expected["experiment"] = dict(baseline["experiment"])
    expected["experiment"]["name"] = cvar2["experiment"]["name"]
    expected["risk"] = dict(baseline["risk"])
    expected["risk"]["epsilon"] = 2.0

    assert cvar2 == expected
    assert cvar2["migration"]["fixed_zero"] is True
    assert cvar2["solver"]["no_rel_heur_time_seconds"] == 0
    assert plan["base_config"].endswith(
        "micro_top10_unscaled_no_migration_norel0_cvar2_baseline.yaml"
    )
    assert plan["generation"]["exact_unique_candidate_configs"]["first_stage"] == 2
    assert plan["validation"]["risk"] == {"alpha": 0.95, "epsilon": 2.0}


def test_no_migration_alpha0_5_cvar2_pair_changes_only_alpha() -> None:
    root = _experiment_root()
    cvar2 = yaml.safe_load(
        (
            root
            / "configs/micro_top10_unscaled_no_migration_norel0_cvar2_baseline.yaml"
        ).read_text(encoding="utf-8")
    )
    alpha0_5 = yaml.safe_load(
        (
            root
            / "configs/micro_top10_unscaled_no_migration_norel0_alpha0_5_cvar2_baseline.yaml"
        ).read_text(encoding="utf-8")
    )
    plan = yaml.safe_load(
        (
            root
            / "plans/micro_top10_unscaled_no_migration_norel0_alpha0_5_cvar2_baseline_gamma_0_1_3h.yaml"
        ).read_text(encoding="utf-8")
    )

    expected = dict(cvar2)
    expected["experiment"] = dict(cvar2["experiment"])
    expected["experiment"]["name"] = alpha0_5["experiment"]["name"]
    expected["risk"] = dict(cvar2["risk"])
    expected["risk"]["alpha"] = 0.5

    assert alpha0_5 == expected
    assert alpha0_5["risk"] == {"alpha": 0.5, "epsilon": 2.0}
    assert alpha0_5["migration"]["fixed_zero"] is True
    assert alpha0_5["solver"]["no_rel_heur_time_seconds"] == 0
    assert plan["base_config"].endswith(
        "micro_top10_unscaled_no_migration_norel0_alpha0_5_cvar2_baseline.yaml"
    )
    assert plan["generation"]["exact_unique_candidate_configs"]["first_stage"] == 2
    assert plan["validation"]["risk"] == {"alpha": 0.5, "epsilon": 2.0}


def test_cpu_top20_full_vs_od_only_three_server_plan_is_explicit() -> None:
    root = _experiment_root()
    config = yaml.safe_load(
        (
            root
            / "configs/micro_cpu_top20_no_migration_norel0_alpha0_5_cvar2_3servers_full.yaml"
        ).read_text(encoding="utf-8")
    )
    plan = yaml.safe_load(
        (
            root
            / "plans/micro_cpu_top20_no_migration_alpha0_5_cvar2_full_vs_od_only_3servers_3h.yaml"
        ).read_text(encoding="utf-8")
    )

    assert config["experiment"]["class_counts"] == {
        "on_demand": 20,
        "spot": 10,
        "batch_jobs": 10,
    }
    assert config["experiment"]["num_servers"] == 3
    assert config["data"]["google_dir"].endswith(
        "notion_toy_google2019_micro_cpu_top20_od_sp10_bj10_unscaled_v1"
    )
    assert config["risk"] == {"alpha": 0.5, "epsilon": 2.0}
    assert config["workload_scenarios"]["synthetic_rng_mode"] == (
        "per_vm_stable"
    )
    assert config["migration"]["fixed_zero"] is True
    assert config["solver"]["no_rel_heur_time_seconds"] == 0
    assert config["solver"]["threads"] == 16
    assert config["solver"]["time_limit_seconds"] == 10_800

    assert plan["generation"]["exact_unique_candidate_configs"][
        "first_stage"
    ] == 2
    assert plan["execution_policy"]["max_parallel_jobs"] == 2
    assert plan["execution_policy"]["threads_per_job"] == 16
    assert plan["execution_policy"]["time_limit_seconds"] == 10_800
    factor = plan["stages"]["first_stage"]["factors"][0]
    assert factor["config_path"] == "experiment.class_counts"
    assert factor["levels"] == [
        {
            "id": "od_only",
            "value": {
                "on_demand": 20,
                "spot": 0,
                "batch_jobs": 0,
            },
        }
    ]
    assert plan["validation"]["risk"] == {"alpha": 0.5, "epsilon": 2.0}
