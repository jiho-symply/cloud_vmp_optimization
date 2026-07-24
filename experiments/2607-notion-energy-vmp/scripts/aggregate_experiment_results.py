#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


TERMINAL_FAILURE_STATES = {"FAILED", "LAUNCH_ERROR"}
COMPLETED_STATES = {"COMPLETED", "SKIPPED_COMPLETE"}

LHS_FEATURES = {
    "kappa_sla": "config.kappa_sla",
    "spot_discount": "config.spot_discount_ratio",
    "kappa_migration": "config.kappa_migration",
    "alpha": "config.alpha",
    "epsilon": "config.epsilon",
    "solar": "config.solar_capacity_multiplier",
    "wind": "config.wind_capacity_multiplier",
    "ess": "config.ess_capacity_ratio",
}
LHS_TARGETS = {
    "objective": "result.solver.objective",
    "cpu_excess": "result.service.expected_od_cpu_excess",
    "empirical_cvar": "result.service.max_time_empirical_cvar",
    "actual_migrations": "result.operations.expected_actual_migrations",
    "grid_kwh": "analysis.expected_grid_kwh",
    "renewable_coverage": "result.energy.renewable_coverage_ratio",
}


def _latest_suite_states(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    latest: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
        run_id = event.get("run_id")
        if run_id:
            latest[str(run_id)] = event
    return latest


def _missing_summary_bucket(state: str) -> str:
    if state in TERMINAL_FAILURE_STATES or state in COMPLETED_STATES:
        return "failure"
    return "incomplete"


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten(child, child_prefix))
    elif isinstance(value, (list, tuple)):
        out[prefix] = json.dumps(value, ensure_ascii=False)
    else:
        out[prefix] = value
    return out


def _standardize(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    std = float(numeric.std(ddof=0))
    if not np.isfinite(std) or std == 0:
        return pd.Series(np.nan, index=values.index, dtype=float)
    return (numeric - float(numeric.mean())) / std


def _lhs_effect_tables(lhs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    columns = [*LHS_FEATURES.values(), *LHS_TARGETS.values()]
    data = lhs.loc[:, columns].apply(pd.to_numeric, errors="coerce").dropna()
    spearman_rows: list[dict[str, Any]] = []
    main_rows: list[dict[str, Any]] = []
    interaction_rows: list[dict[str, Any]] = []
    if data.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    feature_frame = pd.DataFrame(
        {name: _standardize(data[column]) for name, column in LHS_FEATURES.items()},
        index=data.index,
    )
    valid_features = [column for column in feature_frame if feature_frame[column].notna().all()]
    feature_frame = feature_frame[valid_features]
    if feature_frame.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for target_name, target_column in LHS_TARGETS.items():
        target = _standardize(data[target_column])
        if target.isna().any():
            continue
        for feature_name in valid_features:
            spearman_rows.append(
                {
                    "target": target_name,
                    "feature": feature_name,
                    "spearman": data[LHS_FEATURES[feature_name]].corr(data[target_column], method="spearman"),
                    "n": len(data),
                }
            )

        main_design = feature_frame.copy()
        main_matrix = np.column_stack([np.ones(len(main_design)), main_design.to_numpy(dtype=float)])
        main_beta, *_ = np.linalg.lstsq(main_matrix, target.to_numpy(dtype=float), rcond=None)
        main_prediction = main_matrix @ main_beta
        main_r2 = 1.0 - float(np.square(target.to_numpy() - main_prediction).sum()) / float(
            np.square(target.to_numpy() - target.mean()).sum()
        )
        for feature_name, coefficient in zip(valid_features, main_beta[1:], strict=True):
            main_rows.append(
                {
                    "target": target_name,
                    "feature": feature_name,
                    "standardized_coefficient": coefficient,
                    "absolute_coefficient": abs(coefficient),
                    "model_r2": main_r2,
                    "n": len(data),
                }
            )

        interaction_design = main_design.copy()
        interaction_names: list[str] = []
        for left_index, left in enumerate(valid_features):
            for right in valid_features[left_index + 1 :]:
                name = f"{left}:{right}"
                interaction_design[name] = _standardize(feature_frame[left] * feature_frame[right])
                interaction_names.append(name)
        interaction_design = interaction_design.dropna(axis=1)
        interaction_matrix = np.column_stack([np.ones(len(interaction_design)), interaction_design.to_numpy(dtype=float)])
        interaction_beta, *_ = np.linalg.lstsq(interaction_matrix, target.to_numpy(dtype=float), rcond=None)
        interaction_prediction = interaction_matrix @ interaction_beta
        interaction_r2 = 1.0 - float(np.square(target.to_numpy() - interaction_prediction).sum()) / float(
            np.square(target.to_numpy() - target.mean()).sum()
        )
        coefficients = dict(zip(interaction_design.columns, interaction_beta[1:], strict=True))
        for term in interaction_names:
            if term not in coefficients:
                continue
            coefficient = coefficients[term]
            interaction_rows.append(
                {
                    "target": target_name,
                    "interaction": term,
                    "standardized_coefficient": coefficient,
                    "absolute_coefficient": abs(coefficient),
                    "model_r2": interaction_r2,
                    "n": len(data),
                }
            )

    return pd.DataFrame(spearman_rows), pd.DataFrame(main_rows), pd.DataFrame(interaction_rows)


def _write_tuning_analysis(result_df: pd.DataFrame, output: Path, expected_lhs_runs: int) -> None:
    analysis = result_df.copy()
    grid_columns = ["result.energy.expected_grid_da_kwh", "result.energy.expected_grid_rt_kwh"]
    if all(column in analysis for column in grid_columns):
        analysis["analysis.expected_grid_kwh"] = sum(
            (pd.to_numeric(analysis[column], errors="coerce") for column in grid_columns),
            start=pd.Series(0.0, index=analysis.index),
        )
    else:
        analysis["analysis.expected_grid_kwh"] = np.nan

    lhs = analysis.loc[
        (analysis["phase"] == "lhs") & (analysis["result.solver.status"] == "OPTIMAL")
    ].copy()
    spearman, main_effects, interactions = _lhs_effect_tables(lhs)
    spearman.to_csv(output / "lhs_spearman.csv", index=False)
    main_effects.to_csv(output / "lhs_main_effects.csv", index=False)
    interactions.to_csv(output / "lhs_pairwise_interactions.csv", index=False)

    selected_metrics = [
        "run_id",
        "phase",
        *LHS_FEATURES.values(),
        "result.solver.objective",
        "result.solver.best_bound",
        "result.solver.mip_gap",
        "result.service.expected_od_cpu_excess",
        "result.service.max_time_empirical_cvar",
        "result.operations.expected_actual_migrations",
        "result.operations.expected_nonmovement_migration_indicators",
        "analysis.expected_grid_kwh",
        "result.energy.renewable_coverage_ratio",
    ]
    analysis.loc[analysis["phase"] != "lhs", selected_metrics].to_csv(
        output / "designed_experiment_metrics.csv", index=False
    )

    mix = analysis.loc[analysis["phase"] == "renewable_mix"]
    if not mix.empty:
        mix.pivot(
            index="config.solar_capacity_multiplier",
            columns="config.wind_capacity_multiplier",
            values="result.solver.objective",
        ).sort_index().sort_index(axis=1).to_csv(output / "renewable_mix_objective_grid.csv")

    complete = len(lhs) == expected_lhs_runs
    lines = [
        "# Tuning analysis",
        "",
        f"- LHS OPTIMAL results: {len(lhs)}/{expected_lhs_runs}",
        f"- analysis status: {'complete' if complete else 'provisional; the LHS design is incomplete'}",
        "- coefficients are standardized and indicate association within the sampled ranges, not causal effects",
        "",
        "## Main effects",
        "",
    ]
    if not main_effects.empty:
        for target in LHS_TARGETS:
            subset = main_effects.loc[main_effects["target"] == target].sort_values(
                "absolute_coefficient", ascending=False
            )
            if subset.empty:
                continue
            r2 = float(subset["model_r2"].iloc[0])
            terms = ", ".join(
                f"{row.feature}={row.standardized_coefficient:+.3f}" for row in subset.itertuples()
            )
            lines.append(f"- `{target}` (R2={r2:.3f}): {terms}")
    lines += ["", "## Pairwise interaction screen", ""]
    if not interactions.empty:
        for target in LHS_TARGETS:
            subset = interactions.loc[interactions["target"] == target].nlargest(5, "absolute_coefficient")
            if subset.empty:
                continue
            r2 = float(subset["model_r2"].iloc[0])
            terms = ", ".join(
                f"{row.interaction}={row.standardized_coefficient:+.3f}" for row in subset.itertuples()
            )
            lines.append(f"- `{target}` (main+pairwise R2={r2:.3f}): {terms}")
    lines += [
        "",
        "## Interpretation guards",
        "",
        "1. Compare raw objectives only among runs with the same VM and scenario counts.",
        "2. Treat spot discount as a business input, not a free operational tuning parameter.",
        "3. Use actual migrations rather than migration indicators when discussing movement.",
        "4. Use empirical CVaR for realized-risk interpretation; auxiliary CVaR variables need not be tight.",
        "5. Read the sampling replicates before treating effects smaller than sample variability as structural.",
    ]
    (output / "tuning_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    manifest = Path(args.manifest).resolve()
    output = Path(args.output_dir).resolve() if args.output_dir else manifest.parent / "aggregated"
    output.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(manifest.open("r", newline="", encoding="utf-8")))
    latest_states = _latest_suite_states(manifest.parent / "suite_status.jsonl")

    results, failures, incomplete = [], [], []
    for entry in rows:
        run_dir = Path(entry["run_dir"])
        summary_path = run_dir / "summary.json"
        audit_path = run_dir / "prepared_data/data_audit.json"
        config_path = Path(entry["config_path"])
        if not summary_path.is_file():
            state = str(latest_states.get(entry["run_id"], {}).get("state", "PENDING"))
            record = {**entry, "state": state, "reason": "missing summary.json"}
            if _missing_summary_bucket(state) == "failure":
                failures.append(record)
            else:
                incomplete.append(record)
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {}
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        selected_cfg = {
            "seed": cfg["experiment"]["seed"],
            "target_total_vms": cfg["experiment"]["target_total_vms"],
            "num_scenarios": cfg["experiment"]["num_scenarios"],
            "num_servers": cfg["experiment"]["num_servers"],
            "kappa_sla": cfg["economics"]["kappa_sla"],
            "spot_discount_ratio": cfg["economics"]["spot_discount_ratio"],
            "kappa_migration": cfg["economics"]["kappa_migration"],
            "alpha": cfg["risk"]["alpha"],
            "epsilon": cfg["risk"]["epsilon"],
            "renewable_ratio": cfg["energy_data"]["renewable_to_reference_demand_ratio"],
            "solar_capacity_multiplier": cfg["energy_data"].get("solar_capacity_multiplier", 1.0),
            "wind_capacity_multiplier": cfg["energy_data"].get("wind_capacity_multiplier", 1.0),
            "ess_capacity_ratio": cfg["ess"]["capacity_ratio_to_full_server_hour"],
            "scenario_dates": cfg["energy_data"]["scenario_dates"],
        }
        results.append(
            {
                "run_id": entry["run_id"],
                "phase": entry["phase"],
                **_flatten(selected_cfg, "config"),
                **_flatten(summary, "result"),
                **_flatten(audit.get("sampling", {}), "audit.sampling"),
                **_flatten(audit.get("energy_alignment", {}), "audit.energy"),
            }
        )

    result_df = pd.DataFrame(results)
    result_df.to_csv(output / "suite_results.csv", index=False)
    nonresult_columns = [*rows[0].keys(), "state", "reason"] if rows else ["run_id", "state", "reason"]
    pd.DataFrame(failures, columns=nonresult_columns).to_csv(output / "suite_failures.csv", index=False)
    pd.DataFrame(incomplete, columns=nonresult_columns).to_csv(output / "suite_incomplete.csv", index=False)
    if not result_df.empty:
        expected_lhs_runs = sum(entry["phase"] == "lhs" for entry in rows)
        _write_tuning_analysis(result_df, output, expected_lhs_runs)

    diagnostic_cols = [
        "result.model_validation_diagnostics.false_batch_startup_count",
        "result.model_validation_diagnostics.simultaneous_ess_charge_discharge_slot_count",
        "result.model_validation_diagnostics.nonmovement_migration_indicator_count",
    ]
    lines = [
        "# Model validation experiment review",
        "",
        f"- completed summaries: {len(results)}",
        f"- terminal failures or inconsistent completed runs: {len(failures)}",
        f"- running/pending summaries: {len(incomplete)}",
        "",
        "## Immediate formulation signals",
        "",
    ]
    if not result_df.empty:
        for column in diagnostic_cols:
            if column in result_df:
                positive = int((pd.to_numeric(result_df[column], errors="coerce").fillna(0) > 0).sum())
                lines.append(f"- `{column}`: positive in {positive}/{len(result_df)} runs")
        if "result.solver.status" in result_df:
            lines += ["", "## Solver status", ""]
            for status, count in result_df["result.solver.status"].value_counts(dropna=False).items():
                lines.append(f"- {status}: {count}")
        lines += [
            "",
            "## Interpretation order",
            "",
            "1. Check data-audit and feasibility diagnostics before comparing objectives.",
            "2. Separate OPTIMAL results from time-limited incumbents; use objective and bound together.",
            "3. Use sampling replicates to distinguish structural effects from VM-sample noise.",
            "4. Read OFAT curves before interpreting the Latin-hypercube interactions.",
            "5. Treat any formulation-signal count above zero as evidence for a later, explicitly separated ablation—not an automatic model change.",
            "6. With 10 equiprobable scenarios, alpha=0.95 CVaR is effectively driven by the worst sampled scenario; do not over-interpret fine alpha changes.",
            "7. Google interruption traces are external references only; differences from endogenous spot preemption are not feasibility violations.",
        ]
    (output / "review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output / "suite_results.csv")
    print(output / "review.md")
    if not result_df.empty:
        print(output / "tuning_analysis.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
