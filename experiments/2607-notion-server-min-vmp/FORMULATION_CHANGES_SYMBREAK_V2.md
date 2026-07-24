# Symmetry-breaking formulation v2

Formulation ID: `notion_server_min_symbreak_v2_20260722`

Authoritative source: the attached Korean modeling specification recorded in
`plans/micro_stress_first_stage_symbreak_v2.yaml`, pinned by SHA-256.

The v2 rerun keeps the micro-stress experimental data and OFAT design fixed and
changes only the mathematical formulation and its reporting:

- remove the redundant Spot admission variable and derive admission from the
  initial Spot assignment;
- relax second-stage Spot placement and migration indicators to `[0, 1]` as
  specified; Spot placement remains structurally integral, while migration's
  lower-bound-only definition retains explicit false-positive/fractionality
  reporting when its objective coefficient is zero;
- add the excess-branch implication `v <= u`;
- add a packing-orbitope symmetry breaker to initial on-demand rows followed by
  initial Spot rows;
- optimize the nonconstant operating-margin expression while retaining
  on-demand and batch revenue as an audited constant offset in total-profit
  reporting.

Removing the constant revenue changes neither the feasible region nor the LP
relaxation's absolute bound distance. It does change the denominator used by
Gurobi's relative `MIPGap`, so v1 and v2 percentage gaps are not directly
comparable; every v2 summary reports both the raw solver objective/bound and the
constant-added total-profit objective/bound.

The comparison suite remains 11 globally deduplicated first-stage OFAT cases,
with 10 scenarios, 10 on-demand VMs, 10 Spot VMs, 10 batch jobs grouped into at
most 3 families, and 6 servers. Each solve receives 16 threads, a 3600-second
time limit, a 0.001 MIP gap target, and an 80 GiB Gurobi SoftMemLimit; at most
five solves run concurrently.
