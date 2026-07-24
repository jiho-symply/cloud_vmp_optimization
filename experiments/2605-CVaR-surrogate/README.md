# VM Type Modeling 1 - CVaR Surrogate

This folder implements a CVaR-surrogate variant of the Notion page
`VM type modeling (1)`.  The OD and spot chance constraints are replaced by
CVaR constraints over scenario losses.

## Files

- `prepare_data.py`: converts an existing `2604-chance-2sp-toy` instance into a compact JSON file for this model.
- `model.py`: builds and solves the gurobipy model from that JSON.
- `config.json`: controls source data generation, VM mix, chance/CVaR risk ratios, model constants, formulation switches, and Gurobi parameters.

## Run

```powershell
.\.venv\Scripts\python.exe .\experiments\2605-CVaR-surrogate\prepare_data.py
.\.venv\Scripts\python.exe .\experiments\2605-CVaR-surrogate\model.py --time-limit 600 --threads 8
```

By default both scripts read `config.json`.  CLI arguments such as
`--time-limit`, `--threads`, `--instance`, and `--output-dir` override the
config for a single run.

Edit `config.json` to change:

- data generation: random seed, scenario seed/count, total VM count, OD/SP/BJ counts or ratios, source/output paths, vCPU and average CPU filters
- risk and constants: `epsilon_od`, `epsilon_sp`, `rho`, `kappa`, server capacity/count, big-M override, energy and migration costs
- formulation and solver: `bar_load_mode`, `server_symmetry`, Gurobi method, node method, crossover, seed, MIP gap, time limit, threads

The default config builds the 24VM OD/SP/BJ source instance
`chance_2sp_toy_24vm_combination_od_sp_bj_od8_sp8_bj8_sc10_cap8_avg20_lam010`
under `data/processed/2604-chance-2sp-toy`, then writes the 2605 CVaR JSON
under `data/processed/2605-CVaR-surrogate`.

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
