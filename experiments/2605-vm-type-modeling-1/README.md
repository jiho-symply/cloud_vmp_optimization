# VM Type Modeling 1

This folder implements the Notion page `VM type modeling (1)` as a small,
readable Gurobi prototype.

## Files

- `prepare_data.py`: converts an existing `2604-chance-2sp-toy` instance into a compact JSON file for this model.
- `model.py`: builds and solves the gurobipy model from that JSON.
- `config.json`: controls source data generation, VM mix, chance-constraint ratios, model constants, formulation switches, and Gurobi parameters.
- `visualize_solution.py`: reconstructs solution tables and writes PNG charts, including nominal/actual workload Gantt charts and all-scenario recourse heatmaps.

## Run

```powershell
.\.venv\Scripts\python.exe .\experiments\2605-vm-type-modeling-1\prepare_data.py
.\.venv\Scripts\python.exe .\experiments\2605-vm-type-modeling-1\model.py --time-limit 600 --threads 8
.\.venv\Scripts\python.exe .\experiments\2605-vm-type-modeling-1\visualize_solution.py
```

By default both scripts read `config.json`.  CLI arguments such as
`--time-limit`, `--threads`, `--instance`, and `--output-dir` override the
config for a single run.

Edit `config.json` to change:

- data generation: random seed, scenario seed/count, total VM count, OD/SP/BJ counts or ratios, source/output paths, vCPU and average CPU filters
- risk and constants: `epsilon_od`, `epsilon_sp`, `rho`, `kappa`, server capacity/count, big-M override, energy and migration costs
- formulation and solver: `bar_load_mode`, `server_symmetry`, Gurobi method, node method, seed, MIP gap, time limit, threads

The default config builds the 24VM OD/SP/BJ source instance
`chance_2sp_toy_24vm_combination_od_sp_bj_od8_sp8_bj8_sc10_cap8_avg20_lam010`
under `data/processed/2604-chance-2sp-toy`, then writes the 2605 JSON under
`data/processed/2605-vm-type-modeling-1`.

The migration variable `m[i,t,xi]` is a one-sided change detector in the
current formulation: a placement change forces it to 1, while an unchanged
placement does not force it to 0.  With a positive migration coefficient the
objective removes such slack.  For zero-cost sensitivity runs, reconstruct
actual migration events from consecutive `xR` placements rather than treating
every nonzero `m` as a physical move.

If Gurobi finds any incumbent solution, `model.py` writes:

- `summary.json`
- `solution.sol`
- `solution_nonzero_variables.csv`

`visualize_solution.py` reads `solution_nonzero_variables.csv`, writes derived
analysis CSV files under `analysis/`, and writes PNG figures under `plots/`.
The Gantt charts are split into pre-suspension nominal workload and
post-suspension realized workload; idle servers with no workload are omitted.
