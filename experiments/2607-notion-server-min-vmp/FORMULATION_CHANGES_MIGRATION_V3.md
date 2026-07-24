# Migration formulation v3 changes

This run is pinned to the 2026-07-23 attachment with SHA-256
`bf2b4dc77d9fd3cf26b6d7f3314113454103bd50767e9f72b9f4dc3d3c2e26d7`.

Relative to the preceding symmetry-breaking formulation, the only mathematical
change is the on-demand migration definition:

- replace one transition variable `m[i,t,xi]` with destination-indexed
  `m[i,s,t,xi]`;
- retain the documented continuous `[0,1]` relaxation;
- impose all three displayed constraints
  `m >= x_next - x_previous`, `m <= x_next`, and
  `m <= 1 - x_previous`;
- compute migration energy with `sum_s m[i,s,t,xi]`;
- aggregate destination variables once per `(i,t,xi)` in reports and audit the
  exact definition against the realized placement change.

The initial-assignment packing orbitope and every non-migration equation remain
unchanged.  The parameter table mentions `M_i`, but the attachment supplies no
migration-count constraint, value, or sweep setting.  No count cap is invented
for this run.
