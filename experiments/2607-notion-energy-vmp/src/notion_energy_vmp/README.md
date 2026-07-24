# Notion Energy-Aware Two-Stage VM Placement (Gurobi)

The canonical experiment bundle is `experiments/2607-notion-energy-vmp/`.
Commands below are run from the repository root through compatibility links.

This bundle implements the Notion model **“OD+SP+BC with Energy Procurement & Server Provisioning”** as a deterministic-equivalent two-stage stochastic MILP.

It is designed to be used from the root of any clone of this repository.

The implementation uses:

- Google ClusterData 2019 VM-like hourly scenarios for on-demand/spot CPU and memory load.
- Google-derived batch families, scaled to the sampled toy population.
- NYISO 2019 30-minute zonal DA/RT LBMP, aggregated hourly and floored at 0.01 USD/kWh for model purchases without an upper cap.
- NREL 2019 30-minute renewable energy traces, aggregated to hourly generation and scaled to the toy data-center demand.
- 10 equiprobable scenarios and a default total population of about 200 source VM-like entities.

## Files

```text
configs/toy_200_real.yaml                experiment configuration
src/notion_energy_vmp/data.py            data validation, sampling, alignment
src/notion_energy_vmp/model.py           Gurobi deterministic-equivalent MILP
src/notion_energy_vmp/reporting.py       solution extraction and metrics
src/notion_energy_vmp/cli.py             command-line entry point
scripts/run_notion_energy_experiment.py  thin executable wrapper
scripts/run_background.sh                nohup background runner
CODEX_CONNECT_AND_RUN_PROMPT.md           prompt for the server-side Codex agent
```

## Model-faithful validation policy

This bundle deliberately does **not** repair or strengthen the Notion model. Its purpose is to test the current formulation against carefully prepared data.

- Server-transition indexing follows the displayed Notion equations; no initial-state constraint is added.
- Batch startup keeps the displayed equality at the first period and lower bound thereafter; no exact-difference upper bounds are added.
- Spot preemption remains an endogenous scenario-dependent decision. The Google preemption path is retained only as validation data and is not imposed as a constraint.
- ESS charging and discharging keep the displayed separate limits; no mutually-exclusive mode binary is added.
- Migration energy is proportional to realized memory usage. `kappa_migration` scales one full-server-hour of energy per full normalized memory unit, so the resulting coefficient has units of kWh per normalized memory unit migrated.

The report detects rather than prevents fake batch startups, simultaneous ESS charge/discharge, and nonmovement migration indicators. Google interruption paths are external references for comparing endogenous preemption decisions, not feasibility constraints. NREL solar and wind traces receive one common reference-demand scale and then independent capacity multipliers.

## Default toy interpretation

`target_total_vms: 200` is allocated across the source classes in proportion to the 10,000-entity pool. On-demand and spot entities remain individual placement decisions. The sampled number of batch candidates is represented by the existing 10 batch families, with each family workload multiplied by:

```text
target_batch_candidates / source_batch_candidates
```

This keeps the total source population near 200 without pretending that the aggregate batch families are individual VMs.

## Install in the repository

From the repository root:

```bash
cp -r <this-bundle>/src/notion_energy_vmp src/
cp <this-bundle>/scripts/run_notion_energy_experiment.py scripts/
cp <this-bundle>/scripts/run_background.sh scripts/
mkdir -p configs
cp <this-bundle>/configs/toy_200_real.yaml configs/
.venv/bin/python -m pip install gurobipy PyYAML pandas numpy pyarrow
```

## Foreground validation

Prepare and audit the canonical instance without solving:

```bash
.venv/bin/python -u scripts/run_notion_energy_experiment.py \
  --config configs/toy_200_real.yaml \
  --prepare-only
```

Run a short foreground solve:

```bash
.venv/bin/python -u scripts/run_notion_energy_experiment.py \
  --config configs/toy_200_real.yaml \
  --time-limit 300 \
  --run-dir runs/notion_energy_smoke
```

## Background execution and live log

```bash
bash scripts/run_background.sh configs/toy_200_real.yaml runs/notion_energy_toy_200
tail -F runs/notion_energy_toy_200/solver.log
```

The wrapper also writes `console.log` and `run.pid`. Gurobi writes `solver.log` during optimization with `DisplayInterval=1`.

## Main outputs

Each run directory contains:

- `solver.log`, `console.log`, `run.pid`
- `resolved_config.yaml`, `instance_summary.json`, `input_manifest.json`
- `prepared_data/` canonical CSV/JSON inputs plus SHA-256 checksums
- `model.lp`, `model.mps`, `solution.sol`
- `summary.json`, `scenario_metrics.csv`, `time_scenario_metrics.csv`
- `server_schedule.csv`, `energy_dispatch.csv`, `excess_load.csv`
- `od_initial_placement.csv`, `migrations.csv`, `spot_schedule.csv`
- `batch_schedule.csv`, `model_validation_diagnostics.json`
- `iis.ilp` if the model is infeasible

`summary.json` records status, incumbent objective, best bound, MIP gap, runtime, node count, model size, profit components, risk/service metrics, solar/wind availability, server utilization, migration indicators and actual server changes, spot acceptance/preemption, batch completion, and model-validation diagnostics.

## Before the full run

Inspect the generated `instance_summary.json`. In particular, verify:

- selected class counts and batch scale;
- scenario/date mapping;
- renewable capacity scale;
- reference demand and peak aggregate CPU/MEM;
- price units (`USD/MWh -> USD/kWh`);
- estimated lower bound on required servers;
- ESS capacity and charge/discharge limits;
- `lambda_exc` and spot revenue scale.

The server-side Codex prompt in this bundle instructs Codex to perform those checks, run a 2-scenario/40-VM smoke solve first, and only then launch the 10-scenario/200-VM experiment in the background.

## Parallel model-dynamics experiment suite

The v2 suite contains 178 runs: 50 smoke/OFAT/scaling/seasonal/grid runs plus a 128-run Latin-hypercube design. It uses `MIPGap=0.001`, no solver time limit, a rolling pool of 12 workers with eight Gurobi threads each, and a baseline SLA excess multiplier of `0.2` times normalized on-demand revenue.

Generate the immutable run configurations:

```bash
.venv/bin/python scripts/generate_experiment_suite.py \
  --plan experiments/sweep_plan_v2.yaml
```

First run only the data preparation for all designs if desired:

```bash
nohup .venv/bin/python -u scripts/run_experiment_suite.py \
  --manifest experiments/generated/notion_model_validation_v2/manifest.csv \
  --max-workers 12 \
  --threads-per-run 8 \
  --prepare-only \
  > experiments/generated/notion_model_validation_v2/prepare_controller.log 2>&1 &
```

Run the optimization suite in the background:

```bash
nohup .venv/bin/python -u scripts/run_experiment_suite.py \
  --manifest experiments/generated/notion_model_validation_v2/manifest.csv \
  --max-workers 12 \
  --threads-per-run 8 \
  > experiments/generated/notion_model_validation_v2/suite_controller.log 2>&1 &
```

The runner gives each of the 12 active runs a disjoint eight-CPU set, launches the next queued run as soon as any worker finishes, sets BLAS threads to 1, and records live status in `suite_status.jsonl`. `TimeLimit` is left at Gurobi's default infinity, so each run ends only after reaching the configured `MIPGap=0.001` or another solver termination condition.

Aggregate results:

```bash
.venv/bin/python scripts/aggregate_experiment_results.py \
  --manifest experiments/generated/notion_model_validation_v2/manifest.csv
```

Read `aggregated/review.md` first, then analyze `suite_results.csv`. Do not compare objective values from different VM counts as if they were normalized; use service rates, excess rates, renewable coverage, active-server ratios, and per-VM/per-slot normalized measures.
