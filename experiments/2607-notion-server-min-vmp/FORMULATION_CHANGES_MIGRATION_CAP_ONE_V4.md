# Migration-cap-one formulation v4 changes

This is a user-directed experimental extension of the destination-indexed
migration-v3 formulation. The migration-v3 model, entry point, configuration,
plan, and historical outputs remain unchanged.

- policy/formulation id: `notion_server_min_migration_cap_one_v4_20260724`
- base formulation id: `notion_server_min_exact_destination_migration_v3_20260723`
- extension revision: `migration_cap_one_v4_20260724`

## Mathematical change

For every on-demand VM `i` and workload scenario `xi`, add

```text
sum_{s in S} sum_{t in T_i^transition} m[i,s,t,xi] <= 1,

T_i^transition = {t | t and t+1 are both in A_i^O}.
```

The existing `m[i,s,t,xi]` is the exact destination-entry indicator for the
transition from slot `t` to slot `t+1`. Because each on-demand VM occupies
exactly one server at each active slot, summing over destination servers gives
zero when the VM stays and one when its server changes. The new left-hand side
therefore counts that VM's realized migrations in that scenario.

If the consecutive discrete active-slot set is written as
`A_i^O = {a_i^O, ..., d_i^O}`, then
`T_i^transition = A_i^O \ {d_i^O}`. The excluded index is the last active slot,
not the first: a migration from the first slot to the second slot must be
counted. The data source's `departure_t5` is an exclusive timestamp for the
half-open interval `[arrival_t5, departure_t5)` and must not be confused with
the last discrete active-slot index `d_i^O` used in this shorthand.

Scenario `xi` is not summed. The model adds a separate cap for every `(i, xi)`;
different workload scenarios may each realize one migration for the same VM.

## Isolated implementation

The extension is implemented only by
`run_experiment_migration_cap_one.py`. That entry point builds the unchanged
primary migration-v3 model and then adds one constraint named
`migration_count_cap[i,xi]` for every on-demand VM/scenario pair.

Use requires the derived configuration to contain exactly

```yaml
migration:
  max_count_per_vm_per_scenario: 1
```

The guard accepts only the literal integer `1`; booleans, floating-point
values, strings, missing values, and other integer limits are rejected. This
fixed contract prevents an unintended configuration from silently selecting a
different experimental model.

## Audit contract

Before optimization, the variant verifies and records that:

- there is exactly one cap constraint per `(i, xi)`;
- each row has sense `<=` and right-hand side `1`;
- its support contains every destination server and every active outgoing
  transition for that VM/scenario, with coefficient one;
- it contains no variables belonging to another VM or scenario.

When an incumbent exists, the variant records for every `(i, xi)`:

- the summed destination-indexed migration value;
- the independently reconstructed number of adjacent placement changes;
- the definition residual, constraint slack, and cap violation.

Any count-definition residual, slack inconsistency, or cap violation above the
`1e-5` primary hard-audit tolerance fails the run. The policy payload and
solution audit both record this tolerance; the exact coefficient/support audit
separately records its `1e-12` structural tolerance.

For a solved incumbent, the cap solution audit runs before the primary reporter
can publish a completion marker. After primary reports have been generated,
the variant writes the cap-augmented standalone
`model_validation_diagnostics.json` and `migration_policy.json`, then writes the
final `summary.json` last. The standalone diagnostics are therefore identical
to `summary.json["model_validation_diagnostics"]`, and a failed cap audit or
failed finalization cannot leave a cap-free `summary.json`. The no-incumbent
path follows the same ordering and records an explicit
`incumbent_available: false` solution-audit status.

The construction audit is computed in memory whenever the model is built.
The shared CLI's `--build-only` branch intentionally stops before either report
hook, so it does not persist the policy JSON files. Persistence is part of the
ordinary solved-incumbent and no-incumbent report paths used by the sweep.
