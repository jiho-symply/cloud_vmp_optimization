# VM Type Modeling 1 - No Risk Control

This folder implements a robust no-risk-control variant of the 2605 VM type
model.  The chance-control components are removed and every scenario must
satisfy the physical server capacity constraints directly.

## Files

- `config.json`: controls data generation, VM mix, model constants, symmetry, and Gurobi parameters.
- `prepare_data.py`: builds the 2605 JSON instance using the shared config flow.
- `model.py`: builds and solves the no-risk-control gurobipy model.

## Model Changes

- Removed risk/suspension binary variables: `phi`, `eta`, `delta`, `gamma`.
- Removed spot recourse active-state variable `yR`; spot VMs remain active on their assigned server in every scenario.
- Removed capped-load selector `cap_select` and `barL`; the energy objective uses `total_load` directly because `total_load[s,t,xi] <= C u[s,t]` is enforced for every scenario.
- Replaced chance constraints with robust capacity satisfaction: all scenario-time server loads must fit.

The core placement/reservation/migration binary variables remain: `u`,
`u_used`, `x`, `y`, `b`, `xR`, and `m`.

The migration variable `m[i,t,xi]` is a one-sided change detector in the
current formulation: a placement change forces it to 1, while an unchanged
placement does not force it to 0.  With a positive migration coefficient the
objective removes such slack.  For zero-cost sensitivity runs, reconstruct
actual migration events from consecutive `xR` placements rather than treating
every nonzero `m` as a physical move.

## Run

```powershell
.\.venv\Scripts\python.exe .\experiments\2605-no-risk-control\prepare_data.py
.\.venv\Scripts\python.exe .\experiments\2605-no-risk-control\model.py --time-limit 600 --threads 8
```

By default both scripts read `config.json`.  CLI arguments such as
`--time-limit`, `--threads`, `--instance`, and `--output-dir` override the
config for a single run.
