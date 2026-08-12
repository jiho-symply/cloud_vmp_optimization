# 2607 Notion Server-Min VMP

이 디렉터리는 첨부된 Notion 모델 **“모델링3. OD+SP+BC with Serv. Prov.
(Server Min.)”**을 구현할 다음 실험의 독립 작업 공간이다. 이 문서가 보장하는 현재
범위는 명세-구현 계약, 데이터 준비, Gurobi 모델, reporting, 검증된 단일-run
runner와 sensitivity 설계 및 stage별 suite launcher이다.

권위 원본과 후속 지시는 다음 순서로 적용한다.

1. 2026-07-22 최신 formulation snapshot:
   `모델링3 OD+SP+BC with Serv Prov (Server Min )` (private workspace export),
   SHA-256 `1dd1dd55b38833f1d229ed018179e349a25e70e28edb66e80e97b9b7d28fd607`
2. 2026-07-21 micro-stress 재실행 지시: 평균 CPU+MEM usage 내림차순 선택,
   OD/spot/batch 각 10개, batch family 최대 3개, scenario 10개, server 6대
3. 2026-07-20 사용자 후속 지시: scenario 20개, OD/spot/batch 각 50개, batch family
   최대 10개, 모든 합성 CPU/MEM workload의 `[0,q]` cap, 30분 CPU/MEM average 집계,
   machine-time 최빈 joint server shape 기준 정규화와 변환 후 `q>1` VM 제외
4. 2026-07-20 사용자 후속 지시: lifecycle episode 단위 VM, CPU distribution p100
   maximum, empty distribution 및 assigned-machine history/capacity 불완전 episode의 strict drop
5. 2026-07-15 사용자 후속 지시: active slot별 최소 한 관측 정책, 기본 server 6대,
   그리고 private workspace에 첨부된 새 parameter-sweep 이미지
6. 이전 private workspace Markdown snapshot
7. [Notion 원문 페이지](https://app.notion.com/p/3-OD-SP-BC-with-Serv-Prov-Server-Min-39ca14fa720380ada86cc405031dddf0)
8. 이 디렉터리의 [SPEC_IMPLEMENTATION.md](./SPEC_IMPLEMENTATION.md)

Private workspace의 Markdown/image 원본은 이 public repository에 vendoring하지 않는다.
각 실행 plan은 대신 권위 Markdown의 SHA-256을 고정한다. 원본 없이 실행하면 이 checksum이
hash-only provenance로 manifest에 남고, 로컬 원본을 독립적으로 대조하려면 launcher에
`--spec-path /absolute/path/to/formulation.md`를 넘긴다. 이 경우 candidate materialization
전에 plan의 `formulation.spec_sha256`과 일치하는지 검증하고 해당 파일도 source hash
manifest에 포함한다. Destination-indexed migration-v3 원본 checksum은
`bf2b4dc77d9fd3cf26b6d7f3314113454103bd50767e9f72b9f4dc3d3c2e26d7`이다.

사용자 후속 지시가 기존 문서의 값과 충돌하면 후속 지시를 적용한다. 그 밖에 Notion 문서의
수식과 설명이 충돌할 때에는 수식을 literal baseline으로 구현하고,
충돌 및 대안 해석을 결과 manifest에 기록한다. 기존
`experiments/2607-notion-energy-vmp`는 비교 기준일 뿐이며 수정하거나 덮어쓰지 않는다.
repository root의 기존 compatibility symlink와 실행 중인 run 경로도 변경하지 않는다.

## 모델 범위

24시간 동안 서버 on/off 계획, OD 초기 배치, spot 초기 배치를 통한 admission,
batch VM 준비를 1st stage에서 결정한다. 독립적인 spot-admission 변수 `w`는 없고,
`sum_s y_init[j,s] <= 1`에서 0이면 reject, 1이면 accept로 해석한다. Scenario가
실현된 뒤 OD migration, spot preemption, batch
처리량, OD CPU excess와 실시간 전력 비용을 2nd stage에서 결정한다.

- OD CPU는 strict priority로 먼저 서비스한다.
- spot과 batch CPU는 남은 hard capacity만 사용한다.
- memory는 모든 workload class에 대해 hard capacity이다.
- OD CPU excess ratio의 CVaR를 **서버-시간별**로 제한한다.
- 전력은 NYISO real-time 시장에서 전량 구매한다.
- Gurobi는 `expected spot revenue - expected RT electricity cost`만 최대화한다.
  의사결정과 무관한 OD/batch 상수 수익은 solve 후 더해 total profit으로 보고한다.
- Day-ahead 구매, renewable generation/판매, ESS, CPU-excess monetary penalty는 범위 밖이다.

### 최신 formulation과 symmetry breaking

- Spot admission은 `sum_s y_init[j,s] <= 1`로 표현하고, scenario state equation의
  우변도 `sum_s y_init[j,s]`로 둔다. 별도 `w_j`를 두지 않는다.
- 2nd-stage spot placement `y[j,s,t,xi]`와 destination-indexed migration
  indicator `m[i,s,t,xi]`는 continuous `[0,1]`이다. Migration은 세 개의
  entry 제약으로 `s`로 실제 진입할 때만 정확히 1이 되고, Spot preemption
  `h[j,t,xi]`는 binary로 유지한다.
- OD excess 양의 branch `v[s,t,xi]`에 `v[s,t,xi] <= u[s,t]`를 추가해
  꺼진 서버에서 양의 branch를 선택하지 못하게 한다.
- Initial-assignment 행렬은 OD 행을 먼저, Spot 행을 나중에 두고 triangular
  fixing과 server-introduction inequalities를 포함한 packing orbitope를 적용한다.
- 상수 OD/batch revenue 제거는 feasible region이나 LP root relaxation bound를
  수학적으로 강화하지 않는다. 다만 solver objective의 크기가 바뀌므로
  relative `MIPGap`의 분모가 바뀌고, 기존 total-profit gap과 새 raw solver gap은
  숫자만 직접 비교하면 안 된다.

## Literal baseline

| 항목 | 값 |
| --- | --- |
| Planning horizon | 24시간 |
| Slot | 30분 |
| Periods | `H=48` |
| Scenarios | `20`, `p_xi=1/20` |
| Seed | `42` |
| Workloads | OD 50, spot 50, batch jobs 50 |
| Batch families | 50 job의 configured CPU/MEM pair를 최대 10개 deterministic rank bin으로 집계 |
| Servers | homogeneous server 6대, `C_CPU=C_MEM=1` |
| Minimum state duration | `tau_on=tau_off=2` slots |
| Batch overhead | startup CPU `0.05`, prepared MEM `0.05`; 교차 항은 `0` |
| Risk | `alpha=0.95`, `epsilon=0.05` |
| Price factors | `kappa_OD=10`, `gamma_SP=0.3`, `gamma_B=0.5` |
| Migration | `c_mig=0.05 kWh/normalized-MEM` |
| RT price | NYISO NYC, 2019-06-01부터 20개 연속 local date, 0.01 USD/kWh floor |
| Solver | Gurobi, `MIPGap=0.001`, threads 8, time limit 없음 |

최신 명세는 이전의 OD 가격 연동식을 제거하고 모든 OD VM에 공통인 scalar
`c_mig=0.05`를 사용한다. Migration energy는
`c_mig * actual_OD_memory * sum_s(migration_entry_indicator)`이며 RT price를 곱한
값이 migration 전력비가 된다.

## 직전 실험에서 바뀌는 점

| 영역 | `2607-notion-energy-vmp` | 이 실험 |
| --- | --- | --- |
| 시간 해상도 | hourly `T=24` | 30분 `T=48` |
| Scenario | 10 | 20, 균등 확률 |
| Workload/server | 비례 sampling, server 12 | OD/SP/batch=50/50/50, server 6 |
| OD CPU risk | 시간별 aggregate absolute excess CVaR | 서버-시간별 normalized excess CVaR |
| OD priority | split 변수와 penalty에 의존 | exact positive-part excess로 strict priority 강제 |
| Spot 시작 | 최초 slot 즉시 preempt 가능 | accepted spot은 최초 slot에 active |
| Energy market | DA+RT, renewable, sale, ESS | RT 전량 구매만 |
| Objective | spot−DA−RT+sale−excess penalty | solver: spot−expected RT cost; report: OD+spot+batch−expected RT cost |
| Sweep | energy system 중심 178-run | workload/capacity, risk, price, migration, batch overhead |

세부 수식-코드 계약, 데이터 lineage, 명세 충돌과 acceptance criteria는
[SPEC_IMPLEMENTATION.md](./SPEC_IMPLEMENTATION.md)에 있다. 계획된 sweep은
[plans/sweep_plan.yaml](./plans/sweep_plan.yaml)에 있다.

Baseline은 선택된 batch job 50개의 configured `(q_cpu,q_mem)` pair를 정렬한 뒤 최대
`max_families=10`개 rank bin으로 결정론적으로 묶는다. 같은 configured pair는 항상 같은
family에 남고, job-to-family mapping과 family parameter는 data audit에 기록한다. Sweep에서는
LP/MPS를 쓰지 않으며 각 후보는 실행 전에 resource 및 구조적 feasibility를 확인한다.

## 구현 layout

다음 파일들이 서로 격리된 새 패키지 `notion_server_min_vmp`를 구성한다.

```text
configs/baseline.yaml                 literal baseline configuration
configs/micro_stress_baseline.yaml    10/10/10 capacity-pressure configuration
configs/micro_top10_unscaled_baseline.yaml unscaled top-10 configuration
plans/sweep_plan.yaml                 bounded sensitivity design
plans/micro_stress_first_stage_symbreak_v2.yaml latest formulation rerun plan
run_experiment.py                     single-run entry point
run_sweep.py                          deduplicated OFAT stage launcher
run_micro_stress_sweep.py             five-worker micro-stress launcher
src/notion_server_min_vmp/data.py     30-minute model-facing loader
src/notion_server_min_vmp/model.py    Gurobi formulation
src/notion_server_min_vmp/reporting.py objective/risk/lineage reports
tests/                                data, formulation, and reporting tests
```

## 재현 명령

모든 Python 실행은 repository의 `.venv`를 사용한다. 아래 명령은 repository root에서
실행한다. 상대 config/run 경로는 이 실험 디렉터리를 기준으로 해석된다.

```bash
.venv/bin/python experiments/2607-notion-server-min-vmp/run_experiment.py \
  --config configs/baseline.yaml \
  --run-dir runs/baseline_seed42 \
  --prepare-only

.venv/bin/python experiments/2607-notion-server-min-vmp/run_experiment.py \
  --config configs/baseline.yaml \
  --run-dir runs/baseline_seed42
```

두 번째 명령은 6-server baseline MILP를 실제로 최적화한다. 현재 구현 검증에서는 전체
baseline solve는 시작하지 않았고,
`--num-scenarios 2 --on-demand-count 2 --spot-count 2 --batch-job-count 4
--num-servers 2 --threads 1` 축소형 solve를 사용했다.

[plans/sweep_plan.yaml](./plans/sweep_plan.yaml)은 sensitivity의 declarative 설계다.
`run_sweep.py`는 선택한 stage의 immutable derived YAML을 생성하고, baseline 중복을 제거한
뒤 단일-run interface를 서로 다른 run directory와 CPU affinity로 병렬 실행한다.

```bash
.venv/bin/python experiments/2607-notion-server-min-vmp/run_sweep.py \
  --plan plans/sweep_plan.yaml \
  --stage first_stage
```

Sweep은 baseline을 공통 anchor로 사용하는 two-stage OFAT(one factor at a time) 설계이며
전체 Cartesian product가 아니다. 여기서 sweep의 stage는 stochastic model의 1st/2nd stage와
별개의 실험 순서를 뜻한다.

- Sweep 1st stage는 각 경제·overhead parameter가 decision과 결과 metric을 실제로 바꾸는지
  screening한다. Spot 가격 비율은 `[0.0, 0.1, 0.3, 0.5]`, migration coefficient는
  `[0.0, 0.05, 0.1, 1.0]`, startup CPU와 base memory overhead는 각각
  `[0.0, 0.05, 0.1]`이다.
- Sweep 2nd stage는 fleet size `[3, 4, 5, 6, 7, 8]`, workload composition
  `OD150`, `OD75+SP75`, `OD50+SP50+Batch50`, risk
  `alpha=[0.5, 0.8, 0.9, 0.95]`, `epsilon=[0.5, 0.1, 0.05, 0.0]`을 각각 OFAT로
  평가한다.

각 stage 내부의 baseline 중복을 제거하면 1st stage 11개, 2nd stage 14개이고, stage 간
baseline도 다시 deduplicate하면 전체 24개 unique candidate config이다. 각 factor level은
같은 seed와 나머지 baseline 값을 유지한다. 구조적으로 불가능한 후보는 solve 전에
feasibility precheck로 분류하고, 임의의 대체 범위나 adaptive stopping은 적용하지 않는다.

### 즉시 micro-stress formulation-v2 재실행

최신 formulation의 before/after 비교는 기존 baseline 데이터를 바꾸지 않고 별도
`notion_toy_google2019_micro_stress_v1` artifact와
[`micro_stress_first_stage_symbreak_v2.yaml`](./plans/micro_stress_first_stage_symbreak_v2.yaml)을
사용한다. 이 suite는 특정 slot이 아니라 scenario-0의 coverage-weighted
평균 `CPU+MEM` usage가 큰 순서로 각 class를 선택한다. Stress transform은
trace 모집단 분포에 대한 주장이 아니라, 적은 VM으로 용량 압력이 있는 예제를
만들기 위한 감사 가능한 derived benchmark이다.

```bash
.venv/bin/python experiments/2607-notion-server-min-vmp/run_micro_stress_sweep.py \
  --plan plans/micro_stress_first_stage_symbreak_v2.yaml \
  --google-dir data/processed/notion_toy_google2019_micro_stress_v1
```

위 명령은 vendoring되지 않은 명세 파일 없이 실행할 수 있다. 로컬에 권위 Markdown 사본이
있다면 `--spec-path /absolute/path/to/formulation.md`를 추가해 checksum 검증까지 수행한다.
`run_micro_stress_paired_migration_sweep.py`와 migration-v3 adapter도 같은 옵션을 지원한다.

재실행은 10 scenario, OD/spot/batch 각 10개, batch family 최대 3개, server
6대를 고정한 11개 first-stage OFAT case다. 각 solve는 16 thread, `MIPGap=0.001`,
time limit 3600초, Gurobi `SoftMemLimit=80 GiB`를 사용하고 최대 5개를 병렬 실행한다.
모든 case는 같은 selected VM과 scenario seed를 사용하므로, v1과 v2의 차이는
최신 formulation과 solver objective accounting 변경으로 해석한다.

### 무변환 top-10 micro fixture

`notion_toy_google2019_micro_top10_unscaled_v1`은 동일한 service coverage 조건과
scenario-0 coverage-weighted 평균 `CPU+MEM` 순위를 사용해 각 class의 상위 10개를
선택한다. 과거 stress fixture와 달리 canonical의 `q_cpu`, `q_mem`, resource request,
scenario-0 CPU/MEM usage, lifecycle/provenance를 그대로 복사하며 target-q scaling,
usage rescaling, OD memory transform, minimum-server guarantee를 적용하지 않는다.

```bash
.venv/bin/python scripts/build_micro_top10_unscaled_dataset.py

.venv/bin/python experiments/2607-notion-server-min-vmp/run_experiment.py \
  --config configs/micro_top10_unscaled_baseline.yaml \
  --run-dir runs/micro_top10_unscaled_prepare \
  --prepare-only
```

새 config는 관측 scenario 0을 유지하고 OD/SP synthetic scenario 1–9에
CPU lognormal sigma `0.24`, memory sigma `0.12`를 사용한다. BJ는 scenario-0
resource volume에서 최대 3개 workload family로 변환되므로 이 sigma의 대상이 아니다.
선택 분포, configured-q/actual-usage packing, BJ volume, 2대/3대 full-model feasibility
결과는 [`MICRO_TOP10_UNSCALED_V1_ANALYSIS.md`](./MICRO_TOP10_UNSCALED_V1_ANALYSIS.md)에
분리해 기록한다.

### CPU-only OD top-20 micro fixture

`notion_toy_google2019_micro_cpu_top20_od_sp10_bj10_unscaled_v1`은 service coverage
조건을 만족하는 OD를 scenario-0 coverage-weighted 평균 CPU만으로 정렬해 상위 20개를
선택한다. SP와 BJ는 위 top-10 fixture의 ID와 순서를 그대로 유지한다. 모든 request,
`q`, lifecycle/provenance, scenario-0 CPU/MEM usage 값은 canonical source에서
변환 없이 복사한다.

```bash
.venv/bin/python scripts/build_micro_cpu_top20_od_sp10_bj10_unscaled_dataset.py
```

이 fixture를 모델에 넣는 config는 loader가 40개를 모두 사용하도록
`class_counts`를 `on_demand: 20`, `spot: 10`, `batch_jobs: 10`으로 지정해야 한다.
fixture 내부 행 순서는 loader의 seed 기반 순서로 바뀔 수 있지만, 정확한 개수를
요청하면 선택 집합은 보존된다.
여기서 SP 동일성은 저장된 request와 canonical scenario-0 usage 기준이다. 현재 loader는
선택된 OD+SP 전체에 하나의 RNG stream을 사용하므로 OD 집합이 달라지면 synthetic
scenario 1–9의 SP draw는 기존 top-10 instance와 달라질 수 있다.

## 데이터 요약

- Raw 입력은 `data/raw/google2019_cell_a_day0_cpu_distribution/`의
  `usage_5min.parquet`, `instance_events.parquet`, `collection_events.parquet`,
  `machine_events.parquet`, `metadata.json`이다. Model-facing processed 입력은
  `data/processed/notion_toy_google2019_v5_mode_capacity_bounded/`에 생성한다.
- Service/batch VM 식별자는 lifecycle execution 단위의
  `(collection_id, instance_index, episode_index)`이다. `SCHEDULE`이 episode를 시작하고,
  누락 시 `UPDATE_RUNNING`을 방어적 시작으로 허용한다. `EVICT`, `FAIL`, `FINISH`,
  `KILL`, `LOST`가 종료 reason이다. Terminal event가 없으면 planning-horizon 끝에서
  `HORIZON_CENSORED`로 표시하며, running transition 없이 usage만 있는 key는
  full-horizon usage-inferred censored episode로 기록한다.
- Exact duplicate usage row는 episode 분할 전에 제거한다. 서로 다른 interval이 겹치면
  elementary-interval union으로 시간을 한 번만 세고 동시 관측은 resource별 평균을 쓴다.
- CPU distribution은 11개 값 `p0,p10,...,p100`이어야 한다. CPU maximum은
  `cpu_usage_distribution[10]`인 p100이고, BigQuery `maximum_usage.cpus`나 5분 평균
  maximum으로 대체하지 않는다. CPU nominal request `q_cpu`는 episode arrival request와
  episode p100 maximum 중 큰 값이다. Memory는 arrival request와 source
  `maximum_usage.memory` maximum 중 큰 값을 계속 사용한다.
- Episode의 positive-duration assigned-usage fragment 중 distribution이 하나라도 비어
  있거나, assigned machine history가 없거나, positive CPU capacity history가 fragment
  전체를 덮지 못하면 해당 episode 전체를 deterministic sampling 전에 drop한다. Fully
  covered p100 capacity 초과는 값 변경 없이 audit한다.
- Horizon의 active machine-time을 가중치로 가장 자주 관측된 positive joint
  `(capacity_cpu,capacity_mem)` shape를 대표 서버로 선택한다. CPU와 memory는 그 동일 shape의
  각 capacity로 나누어 대표 서버 1대의 fraction으로 변환하고 `servers.csv`는 두 resource
  capacity를 모두 `1.0`으로 둔다. 변환 후 `q_cpu>1` 또는 `q_mem>1`인 VM은 model 입력에서
  제외하며 target-utilization calibration은 적용하지 않는다.
- Strict quality filter 뒤 episode key와 seed를 사용해 deterministic selection한다.
  `SCHEDULER_BATCH` 판정을 먼저 적용하고, 남은 episode 중 arrival priority `>=120`은 OD,
  나머지는 spot으로 분류한다.
- OD/spot VM은 모든 active 30분 slot에 5분 관측값이 하나 이상 있을 때만 선택한다. 각
  slot의 CPU와 memory workload는 존재하는 5분 관측값을 `coverage_us`로 가중한 평균으로
  집계한다. 누락된 5분 값을 보간하거나 0-fill하거나 `6/n` 배율로 보정하지 않는다.
  선택하는 VM 수는 정수이고 model-facing workload 값은 소수형이다.
- CPU percentile distribution은 configured maximum과 quality validation에 사용하지만,
  time-aligned scenario sample로 해석하지 않는다. Scenario 0은 관측 5분 평균을 쓰고,
  scenario 1~19는 평균-보존 lognormal perturbation
  (`CPU sigma=0.22`, `MEM sigma=0.08`)을 사용한다. 모든 합성 CPU와 memory 값은 VM별
  configured resource에 따라 `0 <= usage <= q`로 cap하고, 두 resource 모두 30분 slot에서
  coverage-weighted average로 집계한다.
- 20개 workload scenario와 20개 NYISO 날짜는 index 순서로 결합한다.
  이는 공동 관측이나 workload-price 상관관계를 뜻하지 않는다.
- NYISO RT 5분 raw grid의 내부 결측은 인접한 이전·다음 raw 관측치 두 점으로 time
  interpolation한 뒤 30분 평균한다. Endpoint fill은 별도로 표시하며 model-facing
  loader에서는 추가 보간하지 않는다.
- 각 input, 변환 설정, seed, 선택 ID, scenario-date mapping, checksum을 run manifest에
  보존한다.

## 현재 상태

- [x] 권위 명세 snapshot 분석
- [x] 구현 계약과 알려진 모순 기록
- [x] 제한된 sensitivity plan 작성
- [x] 30분 데이터 파이프라인과 구 baseline(50/50/K10/S10) prepare-only audit
- [x] Gurobi 모델과 reporting
- [x] scalar migration coefficient와 zero-count workload class 지원
- [x] 단위·수식 test (`74 passed`)
- [x] 축소형 real-data solve와 objective/risk/load audit
- [x] 직전 40-scenario baseline(100/100/K100/S6) prepare-only 재검증 (역사적 기록)
- [x] Episode/p100/assigned-machine strict preprocessing 정책 반영
- [x] `notion_toy_google2019_v5_mode_capacity_bounded` processed artifact 재생성 및 baseline/model-build 재검증
- [ ] 현재 20-scenario baseline(50/50/K10/S6) solve
- [x] sweep config generator와 stage별 background dispatcher
- [x] 10-scenario micro-stress artifact(10/10/10, K<=3, S6) 및 직전 formulation suite 생성
- [x] formulation-v2(`w` 제거, `y/m` 완화, `v<=u`, packing orbitope) 재실행 계획 고정
- [x] `notion_server_min_micro_stress_first_stage_symbreak_v2` 및 migration 허용/금지 비교 완료
- [x] destination-indexed exact migration-v3 구현·보고·build 검증
- [x] migration-v3 3시간 11-case OFAT 실행 완료 (원 실행 5개 + retry 6개;
  11/11 `TIME_LIMIT`, target `MIPGap=0.001` 미도달)
- [ ] 전체 sensitivity 실행 완료
