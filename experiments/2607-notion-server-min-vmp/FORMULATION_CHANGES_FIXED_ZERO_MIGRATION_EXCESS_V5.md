# Fixed-zero migration and OD-excess formulation v5 changes

This is a user-directed experimental restriction of the destination-indexed
migration-v3 formulation. The migration-v3 model and all existing entry points
remain unchanged.

- policy/formulation id: `notion_server_min_fixed_zero_migration_excess_v5_20260724`
- base formulation id: `notion_server_min_exact_destination_migration_v3_20260723`
- extension revision: `fixed_zero_migration_excess_v5_20260724`

## Mathematical restriction

For every modeled destination-indexed migration variable, impose

```text
m[i,s,t,xi] = 0.
```

The primary model's three exact destination-entry inequalities then prohibit
every adjacent active-slot server change by an on-demand VM in every scenario.

For every server, time slot, and scenario, also impose

```text
excess_positive_branch[s,t,xi] = 0.
```

The primary formulation already has two Gurobi indicator constraints whose
trigger value is zero. Therefore fixing the branch to zero activates both

```text
OD_CPU_demand[s,t,xi] <= C_cpu * u[s,t]
od_cpu_excess[s,t,xi] = 0.
```

This is stronger than removing excess from the objective or merely fixing the
continuous excess value: it rules out an assignment whose OD CPU demand
exceeds online capacity. The existing hard CPU/MEM capacity, service, energy,
and CVaR constraints remain in force.

## Isolated implementation and opt-in

The restriction is implemented only by
`run_experiment_fixed_zero_migration_excess.py`. It builds the unchanged
primary migration-v3 model, then sets `LB=UB=0` on every variable named
`migration[...]` exposed as `artifacts.variables["migration"]` and every
variable named `excess_positive_branch[...]` exposed as
`artifacts.variables["excess_branch"]`.

Use requires a derived configuration containing exactly the two explicit
boolean opt-ins

```yaml
migration:
  fixed_zero: true
excess_load_indicator:
  fixed_zero: true
```

Both guards require the literal YAML boolean `true`. Integers such as `1`,
strings such as `"true"`, false, null, and missing sections are rejected. This
prevents the restricted model from being selected accidentally or partially.

## Audit and artifact contract

Immediately after construction, the variant verifies and records the count and
zero lower/upper bounds of both restricted variable groups. A solved incumbent
is then independently audited for:

- zero migration-variable values;
- zero adjacent on-demand assignment changes;
- zero excess-branch values;
- zero continuous OD CPU-excess values;
- zero probability-weighted expected OD CPU excess; and
- agreement with the primary report's expected excess, migration sum, and
  actual server-change aggregates.

Non-finite values or a residual above the `1e-5` hard-audit tolerance fail the
run. Bounds use a separate `1e-12` tolerance. The policy payload and instance
metadata include an explicit fidelity warning because this is an experimental
restriction, not the unrestricted source formulation.

For a solved incumbent, the direct audit runs before the primary reporter can
publish a completion marker. Finalization writes synchronized
`model_validation_diagnostics.json` and `fixed_zero_policy.json` first and
atomically publishes `summary.json` last. A failure clears all three final
artifacts so a partial report cannot look complete to the sweep launcher.

The no-incumbent path scans the built model by the canonical Gurobi variable
name prefixes, rechecks every fixed bound and construction-time variable count,
and records `incumbent_available: false`. It follows the same atomic ordering.
The shared CLI's `--build-only` branch intentionally stops before report hooks,
so persistence of these policy JSON files is part of solved and no-incumbent
runs rather than model construction itself.
