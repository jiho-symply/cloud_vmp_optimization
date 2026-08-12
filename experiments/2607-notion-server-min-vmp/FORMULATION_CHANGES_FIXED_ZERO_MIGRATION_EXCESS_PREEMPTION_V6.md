# Cumulative fixed-zero migration, excess, and Spot preemption (v6)

This isolated experimental adapter extends the exact destination-indexed
migration-v3 model without changing the primary implementation.

- policy/formulation id:
  `notion_server_min_fixed_zero_migration_excess_preemption_v6_20260725`
- base formulation id:
  `notion_server_min_exact_destination_migration_v3_20260723`
- entry point:
  `run_experiment_fixed_zero_migration_excess_preemption.py`

## Cumulative restriction

The entry point requires all three values below to be literal YAML booleans:

```yaml
migration:
  fixed_zero: true
excess_load_indicator:
  fixed_zero: true
spot_preemption:
  fixed_zero: true
```

After the ordinary model is constructed, it sets `LB=UB=0` for every variable
in these artifact groups:

1. `variables["migration"]`: destination-indexed on-demand migration;
2. `variables["excess_branch"]`: positive OD CPU-excess branch indicators;
3. `variables["h"]`: Spot cumulative preempted-state variables.

The first two restrictions retain the v5 semantics. In particular, a zero
excess branch activates the existing indicator rows requiring OD CPU demand to
remain within online capacity and the corresponding continuous excess to be
zero.

## Meaning of `h=0`

For Spot VM `j`, scenario `xi`, and active model slot `t`, the primary state
row is

```text
sum_s y[j,s,t,xi] + h[j,t,xi] = sum_s y_init[j,s].
```

`h` is a cumulative preempted state, not a one-slot event. It is binary,
starts at zero, and is nondecreasing. Fixing it to zero therefore has two
cases:

- rejected Spot VM: admission, placement, and `h` are all zero;
- accepted Spot VM: placement sums to one in every active slot and scenario.

Together with `y <= y_init` and `y <= u`, an accepted Spot VM remains active
on its initial server throughout its modeled active window, and that server
must remain online. There are no Spot variables outside the active window.

## Required audits

Construction fails unless every migration, excess-branch, and Spot-preemption
bound is exactly zero within `1e-12`. A solution is rejected unless the
following independent checks pass within `1e-5`:

- all migration values and actual adjacent OD assignment changes are zero;
- all excess indicators, continuous excess values, and expected excess are
  zero;
- all `h` values and derived preemption events are zero;
- every Spot state equation is satisfied;
- every accepted Spot VM is active in every modeled slot/scenario on its
  initial server, with no server-on violation;
- primary summary acceptance/service, migration, and excess aggregates agree
  with the direct variable audit.

The no-incumbent path scans Gurobi variable names and persists the three-group
bound audit. For an incumbent, `model_validation_diagnostics.json` and
`fixed_zero_policy.json` are written first; the fully audited `summary.json`
is atomically published last as the completion marker.

## Experiment isolation

This policy is a deliberately restricted sensitivity experiment. It is not a
replacement for the unrestricted migration-v3 formulation, and v5 files and
results remain unchanged.
