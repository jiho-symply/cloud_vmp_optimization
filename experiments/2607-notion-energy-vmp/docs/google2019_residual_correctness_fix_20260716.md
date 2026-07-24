# Google ClusterData 2019 query/preprocessing residual correctness fix

Date: 2026-07-16

This report records the implementation and acceptance results for the three
residual correctness issues in the day-0 Google ClusterData 2019 pipeline:

1. same-timestamp event state depended on input row order;
2. overlapping source usage intervals double-weighted CPU and memory means;
3. `coverage_us` was dropped before service and batch aggregation.

The existing raw parquet files were reused. The extraction schema did not
change; only deterministic SQL ordering was added. No paid BigQuery extraction
was rerun.

## Changed files

Production/data validation:

- `src/google2019_toy/extract_bigquery.py`
- `src/google2019_toy/build_toy_instance.py`
- `src/google2019_toy/validate_outputs.py`
- `experiments/2607-notion-server-min-vmp/src/notion_server_min_vmp/data.py`
- `experiments/2607-notion-server-min-vmp/configs/baseline.yaml`

Tests and read-only diagnostics tooling:

- `tests/test_extract_bigquery.py`
- `tests/test_google2019_resource_semantics.py`
- `tests/test_google2019_toy_pipeline.py`
- `experiments/2607-notion-server-min-vmp/tests/test_data_sweep_support.py`
- `scripts/audit_google2019_residual_fix.py`
- `scripts/run_google2019_residual_acceptance.py`
- `tests/test_audit_google2019_residual_fix.py`
- `tests/test_run_google2019_residual_acceptance.py`

`experiments/2607-notion-server-min-vmp/src/notion_server_min_vmp/model.py`
was not modified.

The default baseline now points directly to
`data/processed/notion_toy_google2019_v3_deterministic_union_coverage`. Its
provenance list names the actual day-0 usage, instance-event,
collection-event, and machine-event parquet files used to build v3. This avoids
mixing the patched loader with the old v1 processed CSVs during an ordinary
baseline invocation.

## 1. Deterministic same-timestamp event state

The instance-event SQL now orders by collection, instance, time, type,
machine, CPU request, memory request, and the remaining emitted state fields.
The collection-event SQL similarly includes type, scheduler, priority,
scheduling class, and all remaining emitted state fields. SQL order is only an
audit aid; Python does not use the returned row position as event semantics.

Python maps the documented v3 enum to this explicit lifecycle order:

```text
EVICT, FAIL, FINISH, KILL, LOST,
SUBMIT, QUEUE, ENABLE, UPDATE_PENDING, SCHEDULE, UPDATE_RUNNING
```

For a same-time running transition, a `SCHEDULE` or `UPDATE_RUNNING` row whose
machine matches the first observed usage machine is preferred. Within the
remaining final cohort, a normal `missing_type` row is preferred. Sparse state
is forward-filled, while conflicts are tested only among distinct non-NULL
values. CPU and memory are never resolved with independent `MAX()` calls.

Unresolved same-type, same-machine conflicts receive an ambiguity flag, reason,
and candidate count. Strict preprocessing excludes those candidates before the
unchanged deterministic hash head.

Actual day-0 audit:

| metric | before audit | after policy |
|---|---:|---:|
| processed VMs with multiple rows at latest instance timestamp | 5,381 | explicitly resolved |
| latest-tie VMs with CPU request disagreement | 2,937 | explicitly resolved or flagged |
| latest-tie VMs with memory request disagreement | 1,750 | explicitly resolved or flagged |
| latest-tie VMs with machine disagreement | 2,119 | observed-machine preference applied |
| VMs with unresolved arrival-state ambiguity | not represented | 123 excluded |
| VMs with unresolved scheduler ambiguity | not represented | 0 excluded |

The 123 excluded candidates comprise 10 provisional on-demand, 2 spot, and 111
batch candidates. All emitted strict-output ambiguity flags are false. The full
candidate list is stored in `preprocessing_diagnostics.json`.

The resolver was also made linear-time in the event history. On the real input,
normalizing 274,925 raw instance-event rows took 3.173 seconds and resolving all
8,840 candidate histories took 7.760 seconds. The old and optimized resolvers
were value-equivalent over 500 randomized instance histories and 500 randomized
collection histories.

## 2. Union-weighted overlapping usage intervals

Rows identical across every raw field are removed first. Only affected
`(collection_id, instance_index, t5_day)` groups take the elementary-interval
path. Each elementary interval averages its simultaneous non-NULL observations
per resource, and its unique duration enters the bucket numerator and
denominator exactly once. CPU, memory, and assigned memory retain independent
valid-duration denominators. Source `maximum_usage` is not averaged or capped.

Raw day-0 audit:

| metric | before exact dedup | after exact dedup/current diagnostic |
|---|---:|---:|
| exact duplicate extra rows | 3 | removed |
| duplicate interval groups | 508 | 508 recorded |
| overlap-entering source rows | 8,772 | 8,769 |
| all overlap-participating source rows | 14,081 | 14,075 |
| affected raw VM keys | 1,037 | 1,037 |
| affected raw 5-minute buckets | 3,236 | 3,236 |
| duplicated timeline | 29,627 seconds | 29,620 seconds |
| maximum concurrent source rows | 3 | 3 |

The 7-second duplicated-timeline difference is exactly the contribution of the
three exact duplicate rows removed before union arithmetic.

Comparing the common strict v2/v3 output keys gives 930 VMs and 3,096 buckets
whose CPU or memory value changed, zero coverage changes, a maximum absolute CPU
change of 0.05074462097 server units, and a maximum absolute memory change of
0.004877430577 server units. Before strict ambiguity exclusion, the isolated
overlap fix affected 940 processed VMs and 3,112 buckets.

## 3. Coverage propagation

The experiment loader now requires scenario-0 `coverage_us`, validates it as a
finite integer in `(0, 300,000,000]`, and treats it as the source of truth.
Canonical preprocessing emits one row per `(vm_id, t5_day)`; duplicate discovery
is nevertheless audited.

Service aggregation is now:

```text
CPU_30min = sum(cpu_5min * coverage_us) / sum(coverage_us)
MEM_30min = max(mem_5min)
```

Synthetic lognormal multipliers, sigma, seed, and existing caps are unchanged.
The same observed coverage is retained for every synthetic scenario. Legacy
hourly CPU aggregation uses the same duration weighting and memory remains max.

Batch quantities are now:

```text
W_job          = sum(coverage_us) / slot_us
CPU volume_job = sum(cpu_usage * coverage_us) / slot_us
MEM volume_job = sum(mem_usage * coverage_us) / slot_us
slot_us        = 1,800,000,000
```

Using the pre-fix baseline-selected jobs to isolate only this loader correction:

| metric | result |
|---|---:|
| selected service VM-slot groups | 6,806 |
| groups with CPU change above 10% | 71 |
| groups with CPU change above 50% | 8 |
| maximum aggregate service CPU slot difference | 0.03749776968 |
| batch W, row-count method | 2,789.6667 |
| batch W, duration method | 2,767.3839 |
| total batch W overstatement | 0.805193% |
| maximum individual-job W overstatement | 14.302462% |
| maximum individual-job CPU-volume overstatement | 56.316178% |

In the regenerated pool, scenario 0 contains 1,535,188 canonical rows. Coverage
ranges from 1,000,000 to 300,000,000 microseconds, 40,424 rows are partial, and
there are no invalid or duplicate scenario-0 keys.

## Candidate and baseline selection changes

Strict filtering changes the processed pool from 8,840 to 8,717 VMs:

| class | before | after | delta |
|---|---:|---:|---:|
| on-demand | 3,027 | 3,017 | -10 |
| spot | 886 | 884 | -2 |
| batch candidate | 4,927 | 4,816 | -111 |

The experiment's `_stable_order` implementation and counts remain unchanged.
Because preprocessing assigns sequential `vm_id` values after strict filtering,
removing 123 candidates renumbers later VMs; `_stable_order` intentionally still
uses those VM IDs. Therefore the baseline stable-key selection changes are
large even though the selection algorithm itself was not changed:

| class | retained | removed | added | final count |
|---|---:|---:|---:|---:|
| on-demand | 5 | 95 | 95 | 100 |
| spot | 15 | 85 | 85 | 100 |
| batch jobs | 1 | 99 | 99 | 100 |

Examples of removed -> added stable keys:

- on-demand: `55653690268:3430`, `320177399264:2768`,
  `222482058726:441` -> `4982357443:3826`,
  `374570755765:2329`, `300128803935:251`
- spot: `344407582599:1349`, `360229350142:1700`,
  `360399818442:69` -> `373174661826:2583`, `39516980626:206`,
  `374781170852:0`
- batch: `374733344632:959`, `374675839188:572`,
  `374675853022:6430` -> `375230736006:206`, `374860558572:180`,
  `374780797776:1376`

The complete ordered before/after records and added/removed lists are in
`residual_fix_acceptance_audit.json`; VM IDs alone must not be used to compare
the two pools.

## Maximum-usage CPU outlier audit

No value was capped, removed, replaced, or rescaled:

| metric | value |
|---|---:|
| raw source rows with `maximum_usage.cpu > 1` | 892 |
| raw VM keys represented by those rows | 92 |
| raw maximum | 10.34375 |
| selected strict VMs flagged | 90 |
| converted `q_cpu > 1` VMs | 252 |
| converted `q_cpu > 1` by class | OD 125 / Spot 25 / Batch 102 |
| converted maximum `q_cpu` | 17.4785478548 |

This remains an unresolved source-data-quality issue. A per-usage machine
capacity lifecycle join was not added, so the audit explicitly records that
such a ratio was not computed.

## Scaling and invariant checks

- Representative joint raw shape is unchanged: CPU `0.591796875`, memory
  `0.33349609375`.
- CPU quantities are divided only by `0.591796875`; memory quantities are
  divided only by `0.33349609375`.
- Automatic utilization calibration is false.
- Every emitted server has exactly `C_cpu = C_mem = 1.0`.
- The q formula remains `max(arrival request, source maximum_usage)`.
- Classification rules, deterministic hash sampling, seed, minimum usage row
  count, maximum candidate count, synthetic distribution, and family grouping
  were not changed.

The actual-data shuffle audit used independent seeds for usage, instance events,
collection events, and machine events. It found zero stable-key, class,
arrival/departure, scheduler, ambiguity-flag, or coverage mismatches. Maximum
numeric differences were approximately `1e-16` to `1.78e-15`, attributable to
floating-point summation order. The selected-key SHA256 was
`26aacf9f205516e0b79fb309d3749926184c3b4c082fb9ff5ea6e0cd0083f4e0`.

## Tests and model acceptance

- Root test suite: `111 passed`, with two Google client Python 3.10 future
  support warnings.
- Server-min experiment suite: `36 passed`.
- Focused event/overlap/coverage set: `67 passed`.
- `model.py` SHA256 before and after:
  `9578aa46611db356fadee80644598aedf38ef7e8446fc6316b26872ecf3ede81`.

Baseline build with the v3 data succeeded:

| item | value |
|---|---:|
| OD / Spot / selected batch jobs | 100 / 100 / 100 |
| batch families | 97 |
| servers / periods / scenarios | 6 / 48 / 40 |
| variables / binary variables | 3,084,392 / 1,895,624 |
| linear / general constraints | 6,717,720 / 46,080 |

Reduced solve (OD 5, Spot 5, Batch 5, scenarios 2, servers 6) completed with
status `OPTIMAL`, four incumbents, objective `2.2141454867621118`, bound
`2.216916965322982`, relative gap `0.0012517147483939886`, and runtime 22.75
seconds.

The full baseline 120-second run used 119.19 seconds in presolve, explored zero
nodes, and returned `TIME_LIMIT` with no incumbent. It did not return
`INFEASIBLE` or `INF_OR_UNBD`; full feasibility is therefore inconclusive under
this short limit, while reduced-model feasibility is confirmed.

## Generated outputs

The regenerated directory is
`data/processed/notion_toy_google2019_v3_deterministic_union_coverage` and
contains all required legacy files plus:

- `preprocessing_diagnostics.json`
- `residual_fix_acceptance_audit.json`
- `toy_instance_summary.md`

The output directory is approximately 2.2 GiB. The pre-fix v2 directory and raw
parquet files were not overwritten.

## Reproduction commands

```bash
# Extraction SQL dry-run only. The actual residual-fix acceptance reused the
# existing parquet because no selected field changed.
PYTHONPATH=src .venv/bin/python -m google2019_toy.extract_bigquery \
  --project_id <GCP_PROJECT> --cell a --day_index 0 \
  --max_instances 10000 --seed 42 \
  --output_dir data/raw/google2019_cell_a_day0 --dry_run_only

# Regenerate every processed output.
PYTHONPATH=src .venv/bin/python -m google2019_toy.build_toy_instance \
  --raw_dir data/raw/google2019_cell_a_day0 \
  --output_dir data/processed/notion_toy_google2019_v3_deterministic_union_coverage \
  --max_instances 10000 --seed 42 --num_servers 200 --num_scenarios 5

# Independent output validation and before/after audit.
PYTHONPATH=src .venv/bin/python -m google2019_toy.validate_outputs \
  --output_dir data/processed/notion_toy_google2019_v3_deterministic_union_coverage \
  --write_summary
.venv/bin/python scripts/audit_google2019_residual_fix.py \
  --after-dir data/processed/notion_toy_google2019_v3_deterministic_union_coverage \
  --before-selection-json /tmp/google2019_pre_fix_baseline_selection.json \
  --output data/processed/notion_toy_google2019_v3_deterministic_union_coverage/residual_fix_acceptance_audit.json

# Tests.
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest -q experiments/2607-notion-server-min-vmp/tests

# Baseline model build only.
.venv/bin/python scripts/run_google2019_residual_acceptance.py \
  --processed-dir data/processed/notion_toy_google2019_v3_deterministic_union_coverage \
  --mode build \
  --run-dir experiments/2607-notion-server-min-vmp/runs/residual_fix_v3_build

# Reproduced reduced solve.
.venv/bin/python scripts/run_google2019_residual_acceptance.py \
  --processed-dir data/processed/notion_toy_google2019_v3_deterministic_union_coverage \
  --mode solve \
  --run-dir experiments/2607-notion-server-min-vmp/runs/residual_fix_v3_reduced_5x5x5 \
  --on-demand-count 5 --spot-count 5 --batch-job-count 5 \
  --num-scenarios 2 --num-servers 6 --time-limit 300 \
  --mip-gap 0.01 --threads 8

# Short full-baseline status run.
.venv/bin/python scripts/run_google2019_residual_acceptance.py \
  --processed-dir data/processed/notion_toy_google2019_v3_deterministic_union_coverage \
  --mode solve \
  --run-dir experiments/2607-notion-server-min-vmp/runs/residual_fix_v3_full_120s \
  --time-limit 120 --mip-gap 0.001 --threads 8
```
