# VM Type Modeling 1

This folder implements the Notion page `VM type modeling (1)` as a small,
readable Gurobi prototype.

## Files

- `prepare_data.py`: converts an existing `2604-chance-2sp-toy` instance into a compact JSON file for this model.
- `model.py`: builds and solves the gurobipy model from that JSON.
- `visualize_solution.py`: reconstructs solution tables and writes PNG charts, including nominal/actual workload Gantt charts and all-scenario recourse heatmaps.

## Run

```powershell
.\.venv\Scripts\python.exe .\experiments\2605-vm-type-modeling-1\prepare_data.py
.\.venv\Scripts\python.exe .\experiments\2605-vm-type-modeling-1\model.py --time-limit 600 --threads 8
.\.venv\Scripts\python.exe .\experiments\2605-vm-type-modeling-1\visualize_solution.py
```

The default data source is the 24VM OD/SP/BJ toy instance
`chance_2sp_toy_24vm_combination_od_sp_bj_od8_sp8_bj8_sc10_cap8_avg20_lam010`
under `data/processed/2604-chance-2sp-toy`.

If Gurobi finds any incumbent solution, `model.py` writes:

- `summary.json`
- `solution.sol`
- `solution_nonzero_variables.csv`

`visualize_solution.py` reads `solution_nonzero_variables.csv`, writes derived
analysis CSV files under `analysis/`, and writes PNG figures under `plots/`.
The Gantt charts are split into pre-suspension nominal workload and
post-suspension realized workload; idle servers with no workload are omitted.
