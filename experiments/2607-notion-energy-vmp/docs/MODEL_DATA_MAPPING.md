# Model-to-data mapping and validation contract

## Scope

The code is a validation implementation of the current Notion formulation. Data preparation may be corrected when source schemas, units, timestamps, or completeness checks demand it. The optimization formulation must not be strengthened silently.

## Canonical mapping

| Notion item | Canonical source and transformation |
|---|---|
| `I`, `J` | `vm_requests.csv`; deterministic sampling from `on_demand` and `spot` proxy classes after requiring complete hourly usage in all selected scenarios |
| Batch population | `batch_candidate` source count determines the sampled toy share; existing family workloads are multiplied by one common ratio |
| `A_i^O`, `A_j^S` | `arrival_t5 // 12` through `(departure_t5 - 1) // 12`, inclusive |
| `q^O`, `q^S` | `q_cpu`, `q_mem` from `vm_requests.csv` |
| `ell^O`, `ell^S` | `vm_usage_hourly_scenarios.csv`; CPU hourly mean and memory hourly max are already constructed upstream |
| `q^B`, `rho^B`, `W`, base/startup | `batch_families.csv`; only `W` is scaled to the target batch-candidate population |
| Spot interruption trace | `spot_preemption_scenarios.csv`, hourly availability is the minimum 5-minute active flag; retained for post-solve comparison only |
| `C`, `E_idle`, `E_CPU`, minimum times | first configured rows of `servers.csv`; homogeneity is required because the Notion parameters are not server-indexed |
| `p_xi` | selected rows of `scenario_probabilities.csv`, renormalized only when a subset of scenarios is used |
| `P_t^da` | NYISO 30-minute DA LBMP, hourly mean, USD/MWh divided by 1000, floored at the configured 0.01 USD/kWh purchase-price minimum, then hour-wise mean over the selected historical dates |
| `P_t^rt(xi)` | NYISO 30-minute RT LBMP, hourly mean, USD/MWh divided by 1000, floored at 0.01 USD/kWh, one selected historical local date per workload scenario |
| `P_t^sell` | `max(0, sell_price_ratio * P_t^da)`; this is a model assumption, not an NYISO observation |
| `R_t(xi)` | NREL 30-minute solar/wind energy, hourly sum and MWh-to-kWh conversion, followed by one common reference-demand scale and independent solar/wind capacity multipliers |
| ESS | current Notion values: capacity ratio 1.0, charge/discharge limits 10%, efficiencies 0.95/0.895, initial SoC 50% |
| Spot revenue and excess penalty | current Notion reference-energy-cost formulas |
| Migration coefficient | `kappa_migration * (E_idle + E_CPU*C_CPU) / C_MEM`, yielding kWh per normalized memory unit migrated; realized migration energy is this coefficient times memory usage |

## Safe-data rules

1. Raw and existing processed files are read-only.
2. A generated instance is written to a temporary sibling directory, hashed, and atomically renamed.
3. Existing prepared output is never overwritten unless the caller explicitly passes the overwrite flag; failed replacement restores the prior directory.
4. Source and canonical files receive SHA-256 checksums.
5. Duplicate keys, missing active-period usage, incomplete spot paths, nonpositive capacities, scenario mismatch, probability mismatch, DST-short/long days, unit conversion errors, and heterogeneous server parameters are hard failures.
6. Raw NYISO prices remain unchanged and are audited. Model DA/RT inputs use a configurable 0.01 USD/kWh lower floor and no upper cap.
7. No interpolation, clipping, or silent fill is applied by this stage to Google hourly usage. Instead, only service VMs with complete observations over their Notion active interval in every selected scenario are eligible.
8. Dates are interpreted in `America/New_York`; configured dates must have exactly hours 0–23 in both NYISO and NREL after timezone conversion.

## Current-model behavior to diagnose, not prevent

- batch startup variable equal to 1 without a logical 0→1 preparation transition;
- simultaneous ESS charge and discharge;
- comparison of endogenous spot preemption against the external Google interruption reference trace;
- arbitrary first-period server state because the displayed formulation has no initial-state parameter;
- sensitivity to the migration-energy ratio and any nonmovement migration indicators left by the one-sided indicator formulation;
- sensitivity to the DA/RT price floor and uncapped high-price observations;
- CPU excess being driven by the relative magnitude of `lambda_exc` and energy/spot economics;
- with 10 equiprobable scenarios and `alpha=0.95`, time-wise CVaR being effectively worst-scenario dominated.

## Experiment interpretation

Always compare status, incumbent, bound, and gap before objectives. For different VM counts, compare normalized service/excess/energy/server metrics rather than raw profit. Use the three sampling seeds to estimate data-sample variability. Use OFAT runs for direction and thresholds, seasonal runs for trace dependence, and only then the Latin-hypercube runs for interactions.
