# 2605 Constraint Ablation Summary

Ran constraint-family ablations on smaller 2605 chance-model instances while
leaving the 2605-CVaR-surrogate run untouched.

## Instances

- `ablation_6vm`: OD 2, Spot 2, Batch 2, Scenario 3
- `ablation_9vm`: OD 3, Spot 3, Batch 3, Scenario 4
- `ablation_12vm`: OD 4, Spot 4, Batch 4, Scenario 5
- `ablation_15vm_long`: OD 5, Spot 5, Batch 5, Scenario 6

All runs used `MIPGap=0.01`, `Threads=4`; time limits were 30s for 6/9VM,
45s for 12VM, and 180s for 15VM.

## Main Finding

The dominant bottleneck is the capped-load definition:

```text
barL_stxi = min(total_load_stxi, C u_st)
```

as modeled in the ablation base variant with `cap_select` and four big-M
constraints per server-time-scenario. The current default config uses the
indicator formulation.

Evidence:

| Instance | Variant | Runtime | Status | Gap |
| --- | ---: | ---: | --- | ---: |
| 6VM | base | 2.548s | OPTIMAL | 0.008 |
| 6VM | barload_none | 0.223s | OPTIMAL | 0.000 |
| 6VM | barload_indicator | 0.276s | OPTIMAL | 0.000 |
| 9VM | base | 30.012s | TIME_LIMIT | 0.339 |
| 9VM | barload_none | 0.611s | OPTIMAL | 0.000 |
| 9VM | barload_indicator | 0.649s | OPTIMAL | 0.000 |
| 15VM | base | 180.021s | TIME_LIMIT | 1.000 |
| 15VM | barload_indicator | 180.015s | TIME_LIMIT | 1.000 |
| 15VM | barload_indicator_slot_symmetry | 180.030s | TIME_LIMIT | 0.757 |

The indicator formulation preserves capped-load semantics and removes the
observed slowdown on 6/9VM. On 15VM it is not enough by itself, but the
indicator formulation combined with powered-slot symmetry is the only tested
semantic-preserving variant that produced a useful root bound within 180s.

## Secondary Findings

- Adding powered-slot symmetry (`sum_t u_s,t >= sum_t u_s+1,t`) helps. On 9VM,
  the gap improved from `33.9%` to `5.6%` in the same 30s limit.
- This symmetry does not remove the optimal value under homogeneous servers:
  any feasible solution can be relabeled by sorting server indices in
  nonincreasing powered-slot count. All server-indexed variables are permuted
  with the server labels, and the constraints/objective are invariant because
  all server capacities/costs are identical.
- Removing batch completion also helps, especially as size grows, so batch
  workload completion/scheduling is a secondary bottleneck.
- Removing OD chance budget, spot chance budget, spot service ratio, or exact
  indicator upper bounds did not consistently improve solve time. These are
  not the primary bottleneck in the tested sizes.
- Removing migration helps on 6VM but is not enough on 9/12VM, so migration is
  not the dominant bottleneck relative to capped-load and batch scheduling.

## Artifacts

- `ablation_6vm/ablation_summary.csv`
- `ablation_9vm/ablation_summary.csv`
- `ablation_12vm/ablation_summary.csv`
- `ablation_15vm_long/ablation_summary.csv`
