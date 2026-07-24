# 2607 Notion Energy-Aware VMP

This directory is the canonical home of the July 2026 experiments that connect
the Notion two-stage VM placement formulation to Google ClusterData 2019,
NYISO 2019 prices, and NREL 2019 renewable traces.

Run commands from the repository root. Compatibility links at the old root
locations keep the existing commands, imports, manifests, and run paths valid.
The source and processed datasets remain under the repository-level `data/`
directory and are intentionally not duplicated here.

## Layout

```text
configs/       data-pipeline and optimization configurations
docs/          data mapping, handoff, and detailed model/result reviews
generated/     immutable suite configs, manifests, controller state, aggregates
plans/         v1 and v2 sweep definitions
runs/          prepare-only audits and per-run solver outputs
scripts/       data builders, suite runner, solver entry point, aggregator
src/           Python packages for data preparation and the optimization model
tests/         pipeline, suite, aggregation, and solver configuration tests
```

`MODEL_FIDELITY_REVIEW.md` records the equation-level comparison with the
Notion formulation and the user-directed parameter changes.

## Experiment Sets

- `notion_model_validation_v1`: initial 58-run design retained for provenance.
- `notion_model_validation_v2`: 178-run design with 128 LHS samples, a 4x4
  solar/wind cross sweep, scaling, economics, risk, energy, and seasonal runs.
- `notion_energy_prepare_v40_xi2` and `notion_energy_prepare_v200_xi10`:
  prepare-only data audits with checksum manifests.

The v2 suite uses a rolling pool of 12 processes, eight Gurobi threads per
process, `MIPGap=0.001`, `SoftMemLimit=32 GiB`, `NodefileStart=4 GiB`, and no
solver time limit.

## Reproduce

Generate the v2 suite:

```bash
.venv/bin/python scripts/generate_experiment_suite.py \
  --plan experiments/sweep_plan_v2.yaml
```

Run it with the established rolling worker settings:

```bash
.venv/bin/python -u scripts/run_experiment_suite.py \
  --manifest experiments/generated/notion_model_validation_v2/manifest.csv \
  --max-workers 12 \
  --threads-per-run 8
```

Aggregate all terminal runs:

```bash
.venv/bin/python scripts/aggregate_experiment_results.py \
  --manifest experiments/generated/notion_model_validation_v2/manifest.csv
```

The main aggregate reports are
`generated/notion_model_validation_v2/aggregated/review.md` and
`generated/notion_model_validation_v2/aggregated/tuning_analysis.md`.

## Compatibility Paths

The repository-level `src/`, `configs/`, `tests/`, `docs/`, and `runs/` paths
resolve into this directory. The relevant files under repository-level
`scripts/`, plus `experiments/generated` and the two sweep-plan files, are also
links to this canonical copy. This preserves all commands already recorded in
logs and documentation.
