# Model Fidelity Review

Source Notion page fetched: `https://app.notion.com/p/395a14fa720380f5bbd8cda62f50fc93`

Fetch timestamp reported by Notion MCP: `2026-07-09T18:28:49.033Z`

## Equation Comparison

| Notion 식 | 현재 코드 | 영향 | 수정 여부 |
|---|---|---|---|
| Displayed server on/off transitions, minimum on/off constraints, and `u_on + u_off <= 1`. | `src/notion_energy_vmp/model.py` implements the same transition indexing with 0-based Python periods. | No material difference found. No initial server-state parameter is present. | 미수정 |
| On-demand initial placement, start linkage, active-period assignment, and powered-server requirement. | Implemented directly with `x_init`, `x`, and `u`. | No material difference found. | 미수정 |
| Migration indicator lower-bound inequalities and migration energy term using memory usage. | Implemented directly with `mig` and `c_mig * MEM usage`. | Indicator formulation is unchanged. The coefficient is now `kappa_migration * full_server_hour_kWh / C_MEM`, with units kWh per normalized memory unit. | 단위 수정 |
| Spot admission, fixed initial assignment, irreversible endogenous preemption, and active/preempted state equality. | Implemented directly with `w`, `y_init`, `y`, and `h`. Google observed availability is loaded only for diagnostics. | No observed preemption constraint is imposed, matching the current validation policy. | 미수정 |
| Batch preparation/startup with equality at first period and lower bound afterward. | Implemented directly with `z` and `z_on`. | Allows false-positive startup indicators because there is no exact upper bound. | 미수정, diagnostic only |
| Batch workload completion and resource-limit constraints. | Implemented directly with `b`, `rho_batch`, `q_batch`, and scaled `W_k`. | No material difference found. | 미수정 |
| Batch base/startup parameters are displayed as family-independent constants. | Source `batch_families.csv` has family-specific `base_cpu`, `base_mem`, `startup_cpu`, and `startup_mem`. The loader canonicalizes each to one scalar by taking the maximum source value and records the source range in `data_audit.json`. | This keeps the Notion scalar-parameter formulation unchanged, but is conservative relative to smaller source families. | loader canonicalization only |
| On-demand CPU excess split, memory exactness, total load construction, and hard CPU/MEM capacity. | Implemented directly with `load_od`, `excess`, and `load_total`. | Memory excess is represented by absence of a MEM excess variable and exact memory equality. | 미수정 |
| Time-indexed CVaR constraint for aggregate on-demand CPU excess. | Implemented directly with `var_eta` and `cvar_zeta`. | No material difference found. With 10 equiprobable scenarios and `alpha=0.95`, sampled worst-case behavior can dominate. | 미수정 |
| Energy definition, renewable dispatch, grid balance, ESS SoC dynamics, charge/discharge limits, and terminal SoC. | Implemented directly with nonnegative grid, renewable, ESS, and SoC variables. | No ESS charge/discharge exclusivity binary is present. Simultaneous charge/discharge is possible and is measured after solve. | 미수정, diagnostic only |
| Objective excludes constant on-demand revenue and maximizes expected spot revenue plus renewable sale revenue minus DA/RT grid cost and CPU-excess penalty. | Implemented directly as the Gurobi objective. | No material difference found. | 미수정 |

## User-Directed Parameter Overrides

These settings differ from the parameter notes at the bottom of the Notion page, but do not change the displayed optimization equations.

| Notion parameter note | Current configuration | Impact | Change status |
|---|---|---|---|
| The page still lists baseline `kappa_SLA=2`. | `configs/toy_200_real.yaml` uses `kappa_sla=0.2`, so `lambda_exc=0.2 * kappa_OD * reference CPU-energy cost`. | CPU excess is priced at 0.2 times normalized on-demand CPU revenue rather than 2 times. | User-directed parameter override |
| The page's parameter note writes `c_i^mig=kappa_mig*p_i^OD`, while the displayed energy equation requires `c_i^mig` to be an energy coefficient multiplied by memory usage. | The loader uses `c_mig=kappa_migration*(E_idle+E_cpu*C_CPU)/C_MEM`, in kWh per normalized memory unit migrated. | Migration contributes energy proportional to realized memory usage and therefore affects cost indirectly through the energy balance. | User-directed unit correction |
| The page does not prescribe a lower bound for observed NYISO purchase prices. | DA and RT purchase prices are floored at `0.01 USD/kWh`; raw values and adjusted counts remain in `data_audit.json`. There is no upper cap. | Prevents negative-price grid purchases while preserving the raw-price audit trail. | User-directed data assumption |

## Data-Source Note

| Notion 식 | 현재 코드 | 영향 | 수정 여부 |
|---|---|---|---|
| Parameter notes still mention AzurePublicDatasetV2 for VM CPU demand. | Current validation pipeline uses Google ClusterData 2019 VM-like proxy classes documented in `docs/MODEL_DATA_MAPPING.md`. | This is a data-source substitution for this validation run, not a change to the displayed optimization equations. | 미수정, documented in audit |

## Unresolved Formulation Warnings

- No ESS charge/discharge exclusivity binary was added.
- No exact batch-startup upper bound was added.
- Google interruption data is an external comparison trace; spot preemption remains an endogenous model decision over each VM's data-defined active period.
- No server initial-state constraint was added.
- No additional migration restrictions or grid bounds were added.
- Migration indicator lower bounds can still produce nonmovement indicators within MIP tolerance; reporting therefore separates indicator and actual server-change counts.
