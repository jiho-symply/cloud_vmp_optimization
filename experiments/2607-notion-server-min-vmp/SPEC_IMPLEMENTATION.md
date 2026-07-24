# Notion Server-Min VMP — Specification-to-Implementation Contract

최종 명세 반영일: 2026-07-24
상태: migration-v3 11-case execution complete; full baseline and sensitivity pending

## 1. 권위, 목적, 범위

기본 권위 명세는 private workspace에서 export한 Markdown snapshot
`모델링3 OD+SP+BC with Serv Prov (Server Min )`이다.
이 snapshot의 SHA-256는
`1dd1dd55b38833f1d229ed018179e349a25e70e28edb66e80e97b9b7d28fd607`이다.
이 원본과 parameter-sweep 이미지는 public repository에 vendoring하지 않는다. 실행 plan은
원본 checksum을 유지하며, launcher의 선택적 `--spec-path`로 로컬 사본을 제공하면 실행 전
checksum을 검증해 source manifest에 포함한다. 경로를 생략하면 checksum만 provenance로
기록한다. Migration-v3 snapshot은 SHA-256
`bf2b4dc77d9fd3cf26b6d7f3314113454103bd50767e9f72b9f4dc3d3c2e26d7`로 별도 고정한다.
단, 2026-07-20 사용자가 직접 지정한 20-scenario/50-VM/10-family 축소 설정, 모든 합성
CPU/MEM usage의 `[0,q]` cap, CPU/MEM average 집계, lifecycle episode, distribution p100 및
strict assigned-machine quality 정책과, 2026-07-15 사용자가 지정한 active-slot 관측 정책,
기본 server 6대와 private workspace의 새 parameter-sweep 이미지 범위가
기존 snapshot과 충돌하는 경우 이 후속 지시를 우선한다.
이 계약은 해당 명세를 직전 실험
`experiments/2607-notion-energy-vmp`의 코드 구조 위에 구현할 때 필요한 변경과
검증 기준을 고정한다.

우선순위는 다음과 같다.

1. 2026-07-22 snapshot의 formulation, variable domain, symmetry-breaking, objective accounting
2. 사용자의 2026-07-21 micro-stress 재실행 지시
3. 사용자의 2026-07-20 scenario/workload/family 축소, synthetic cap 및 average 집계 후속 지시
4. 사용자의 2026-07-20 episode/p100/strict-quality 후속 지시
5. 사용자의 2026-07-15 후속 지시와 sweep 이미지
6. Notion에 표시된 수식과 후속 지시로 대체되지 않은 baseline parameter
7. Notion prose가 수식과 일치하는 경우의 의미
8. 이 문서에 명시한 충돌 해석과 mode
9. 직전 실험의 구현 관례

직전 실험의 실행 중 process, run output, generated manifest, repository root symlink는
어떤 구현 단계에서도 수정하지 않는다. 새 산출물은 이 실험 디렉터리 아래에만 쓴다.

### In scope

- 2-stage stochastic OD/spot/batch placement
- 1st-stage server provisioning과 minimum on/off time
- OD migration, endogenous spot preemption, divisible batch processing recourse
- CPU strict priority와 memory hard capacity
- 서버-시간별 OD CPU excess-ratio CVaR
- NYISO real-time 전력 가격과 linear server energy
- 명세의 OD/spot/batch pricing

### Out of scope

- Day-ahead procurement
- Renewable generation, curtailment, sale
- ESS charge/discharge/SoC
- 별도 CPU-excess monetary penalty
- Spot checkpoint/restart/re-execution
- Migration network, duration, downtime, dirty-page dynamics
- Batch task-level precedence/deadline/integer scheduling

## 2. 추출된 요구사항

### Functional requirements

- **MOD-01**: `u`, OD/spot initial assignment, batch preparation/startup은 scenario-independent
  1st-stage 변수이다. Spot admission은 독립 `w_j`가 아니라
  `sum_s y_init[j,s] <= 1`인 initial-assignment 행에서 직접 결정된다.
- **MOD-02**: OD placement, spot preempted state, batch processed volume, load/excess/energy는
  scenario-dependent 2nd-stage 변수이다. Destination-indexed OD migration
  `m[i,s,t,xi]`와 spot placement `y`는 continuous `[0,1]`, spot preemption `h`는
  binary이다. 세 migration 제약은 binary placement에서 destination entry를 exact하게
  정의한다.
- **MOD-03**: OD VM은 active period의 모든 slot에 정확히 한 대의 켜진 서버에 있다.
- **MOD-04**: accepted spot VM은 assigned server에서 첫 slot에 active이고, 한 번
  preempt되면 horizon 내에 재실행되지 않는다.
- **MOD-05**: batch workload는 모든 scenario에서 horizon 내 전량 처리한다.
- **MOD-06**: OD CPU excess는 서버 capacity를 넘는 OD actual CPU demand의 정확한
  positive part이다. 따라서 spot/batch CPU는 OD service 후 남은 capacity만 쓴다.
- **MOD-07**: 모든 class의 served memory load는 hard capacity를 넘지 않는다.
- **MOD-08**: CVaR loss는 `L_exc[s,t,xi]/C_CPU`이며 각 `(s,t)`에 독립적으로
  `CVaR_alpha <= epsilon`을 적용한다.
- **MOD-09**: energy는 server idle, served CPU load, migration term으로 정의한다.
- **MOD-10**: Gurobi objective는 expected spot revenue에서 expected RT electricity cost를
  뺀 nonconstant operating margin이다. OD와 batch constant revenue는 model objective에서
  제외하고 solve 후 더해 total profit으로 보고한다.
- **MOD-11**: initial-assignment packing orbitope는 OD 행을 앞에, Spot 행을
  뒤에 두고 triangular fixing과 server-introduction inequalities를 모두 적용한다.
- **MOD-12**: excess positive-branch binary `v[s,t,xi]`는
  `v[s,t,xi] <= u[s,t]`를 만족한다.

### Data and reproducibility requirements

- **DATA-01**: `H=48`, 30분 단위의 workload, price, energy coefficient를 사용한다.
- **DATA-02**: baseline은 20개 scenario와 균등 확률 `1/20`을 사용한다.
- **DATA-03**: OD 50, spot 50, batch job 50을 seed 42로 결정론적으로 선택하고,
  기본 fleet은 homogeneous server 6대로 둔다.
- **DATA-04**: 모든 선택 ID, random draw, bin mapping, source checksum과 변환 설정을
  manifest에 기록한다.
- **DATA-05**: Google workload와 NYISO price의 scenario 결합은 index pairing이며
  joint observation으로 표현하지 않는다.
- **DATA-06**: service VM은 모든 active 30분 slot에 5분 관측값이 하나 이상 있을 때만
  선택한다. 각 slot의 CPU와 memory workload는 존재하는 관측값의 `coverage_us` 가중
  평균이며, 누락된 5분 값은 보간, 0-fill 또는 `6/n` 배율 보정하지 않는다.
- **DATA-07**: model-facing VM unit은
  `(collection_id, instance_index, episode_index)`이다. Running episode는 `SCHEDULE`로
  시작하고, 누락 시 `UPDATE_RUNNING`을 방어적 시작으로 허용한다. `EVICT`, `FAIL`,
  `FINISH`, `KILL`, `LOST`가 종료 reason이며 terminal event가 없는 실행은
  `HORIZON_CENSORED`로 명시한다.
- **DATA-08**: exact duplicate usage row는 episode 분할 전에 제거한다. Episode 안의
  positive-duration assigned-usage fragment 중 CPU distribution이 비어 있거나, assigned
  machine history가 없거나, positive CPU capacity coverage가 불완전한 경우 해당 episode
  전체를 min-usage 및 deterministic sampling 전에 제외한다.
- **DATA-09**: CPU maximum은 11-value
  `cpu_usage_distribution[10]`의 p100이다. `maximum_usage.cpus` 또는 average usage
  maximum으로 대체하지 않는다. `q_cpu=max(resource_request_cpu_at_episode_arrival,
  episode_p100_max)`이다.
- **DATA-10**: scenario 1부터 19까지의 lognormal CPU/MEM draw는 모두 VM별 configured
  resource에 따라 `0 <= usage <= q`로 cap한다.
- **DATA-11**: 선택된 batch job 50개의 configured `(q_cpu,q_mem)` pair는 정렬 후 최대
  10개의 deterministic rank bin으로 묶으며 같은 pair는 분리하지 않는다.
- **DATA-12**: horizon active machine-time 가중 최빈 positive joint CPU/MEM capacity pair를
  대표 서버로 선택하고, 각 resource를 그 동일 pair의 capacity로 나누어 서버 capacity를
  `(1,1)`로 변환한다. 변환 후 `q_cpu>1` 또는 `q_mem>1`인 VM은 제외한다.
- **RUN-01**: Python은 repository `.venv`로 실행한다.
- **RUN-02**: baseline solver는 Gurobi, MIPGap 0.001, 8 threads, no time limit이다.
- **RUN-03**: immediate formulation-v2 rerun은 기존 micro-stress artifact를 고정하고
  scenario 10개, OD/spot/batch 각 10개, batch family 최대 3개, server 6대를
  사용한다. 각 solve는 16 threads, time limit 3600초, MIPGap 0.001,
  `SoftMemLimit=80 GiB`이고 최대 5개를 병렬 실행한다.

## 3. Literal baseline contract

```yaml
horizon_hours: 24
slot_minutes: 30
num_periods: 48
num_scenarios: 20
scenario_probability: 0.05
seed: 42
class_counts:
  on_demand: 50
  spot: 50
  batch_jobs: 50
num_servers: 6
min_on_slots: 2
min_off_slots: 2
batch_startup_cpu: 0.05
batch_prepared_mem: 0.05
max_batch_families: 10  # selected 50 jobs의 configured-pair rank bins
alpha: 0.95
epsilon: 0.05
kappa_on_demand: 10.0
spot_discount_ratio: 0.30
batch_discount_ratio: 0.50
migration_energy_coefficient: 0.05
google_raw_dir: data/raw/google2019_cell_a_day0_cpu_distribution
google_processed_dir: data/processed/notion_toy_google2019_v5_mode_capacity_bounded
google_vm_unit: [collection_id, instance_index, episode_index]
```

NYISO baseline은 NYC zone의 local date `2019-06-01`부터 `2019-06-20`까지 날짜당
하나의 scenario를 순서대로 매핑한다. 30분 RT price를 USD/kWh로 변환한 뒤
`0.01 USD/kWh` 하한을 적용한다.

### 3.1 Parameter sweep contract

Parameter sweep은 baseline을 공통 anchor로 쓰는 two-stage OFAT(one factor at a time)
설계이다. Sweep stage는 stochastic model의 1st/2nd stage와 별개이며, 각 실행은 명시된
factor 하나만 바꾸고 seed 42와 나머지 baseline 값을 유지한다. Factor 간 Cartesian product
또는 adaptive range 변경은 하지 않는다.

| Sweep stage | Factor | Levels |
| --- | --- | --- |
| 1st | Spot 가격 비율 `gamma_SP` | `0.0, 0.1, 0.3, 0.5` |
| 1st | Migration coefficient `c_mig` | `0.0, 0.05, 0.1, 1.0` |
| 1st | Batch startup CPU overhead | `0.0, 0.05, 0.1` |
| 1st | Batch base memory overhead | `0.0, 0.05, 0.1` |
| 2nd | Servers | `3, 4, 5, 6, 7, 8` |
| 2nd | Workloads | `OD150`; `OD75+SP75`; `OD50+SP50+Batch50` |
| 2nd | Risk `alpha` | `0.5, 0.8, 0.9, 0.95` |
| 2nd | Risk `epsilon` | `0.5, 0.1, 0.05, 0.0` |

1st stage는 각 경제·overhead parameter가 decision과 결과 metric에 의미 있는 변화를 만드는지
screening한다. 2nd stage는 다른 fleet, workload composition, risk budget 범위에서 model
response를 확인한다. Server sweep은 baseline workload를, workload sweep은 server 6대를,
risk sweep은 baseline workload와 server 6대를 사용하며 `alpha`와 `epsilon`도 서로 별도의
OFAT 축이다. Stage별 baseline 중복 제거 후 1st stage 11개, 2nd stage 14개이고, stage 간
baseline을 다시 deduplicate한 전체 수는 24개 unique candidate config이다. 구조적
feasibility precheck에서 불가능하다고 판정된 후보는 solver를 시작하지 않고 해당 상태로
기록한다.

### 3.2 Immediate micro-stress formulation-v2 rerun

Full literal baseline과 별개로, 최신 formulation의 비교 실험은
`data/processed/notion_toy_google2019_micro_stress_v1`을 그대로 사용한다. 선택은
특정 slot의 peak가 아니라 scenario-0 coverage-weighted 평균 `cpu_usage+mem_usage`
내림차순이며, VM ID로 deterministic tie-break한다. 모든 memory workload는 기존
average 집계 정책을 유지한다. Stress resource transform은 canonical processed
artifact를 수정하지 않는 derived benchmark이며, Google trace 모집단의 자원 분포로
해석하지 않는다.

```yaml
plan: plans/micro_stress_first_stage_symbreak_v2.yaml
formulation_id: notion_server_min_symbreak_v2_20260722
scenarios: 10
class_counts: {on_demand: 10, spot: 10, batch_jobs: 10}
max_batch_families: 3
servers: 6
first_stage_unique_ofat_cases: 11
max_parallel_jobs: 5
threads_per_job: 16
time_limit_seconds: 3600
mip_gap: 0.001
soft_mem_limit_gb_per_job: 80
```

이 rerun은 selected VM, stress transform, scenario seed, OFAT level을 직전 suite와 같게
유지하고 formulation만 바꾸어 비교한다. 계획은 latest snapshot hash와 실행에
사용한 source/data hash를 suite metadata에 고정한다.

## 4. 구현된 수식-코드 매핑

아래 경로와 symbol은 현재 구현 interface이다. 변경 시 이 표와 test를 함께 수정한다.

| 명세 블록 | 예정 코드 | 변수/constraint 계약 |
| --- | --- | --- |
| Sets, parameters | `src/notion_server_min_vmp/data.py::InstanceData` | `I,J,K,S,T,Xi,p,active_od,active_spot,C,q,load,rho,W,price,energy,risk` |
| 1st-stage variables | `model.py::build_model` | `x_init,y_init,z,z_on,u,u_on,u_off`; 별도 spot `w` 없음 |
| 2nd-stage variables | `model.py::build_model` | `x,mig,y,h,b,load_od,load_total,excess,excess_branch,eta,zeta,energy` |
| Server provisioning | `model.py`, `server_*`, `min_up`, `min_down` | binary state transition, exclusive on/off event, truncated terminal dwell |
| OD placement | `model.py`, `od_*` | one initial server, start link, one live placement, only on server |
| OD migration | `model.py`, `migration_entry_*` | destination별 `mig in [0,1]`; lower/next-upper/previous-upper 세 제약으로 `0->1` entry exact 정의 |
| Spot placement | `model.py`, `spot_*` | `sum(y_init)<=1`, fixed server, binary `h`, `h[a]=0`, monotone `h`, `sum(y)+h=sum(y_init)`; recourse `y in [0,1]` |
| Assignment symmetry | `model.py`, `symmetry_assignment_*` | OD-first/Spot-second packing orbitope triangular fixing + server-introduction inequalities |
| Batch preparation | `model.py`, `batch_start_*` | `z<=u`, first-slot equality, later positive-transition lower bound |
| Batch processing | `model.py`, `batch_limit`, `batch_complete` | resource limit별 processing bound, scenario별 `sum(b)=W` |
| Exact OD CPU excess | `model.py`, `od_excess_branch_*`, `excess_branch_server_on` | Gurobi indicator 4개 branch와 `v<=u`로 `max(0,D_OD-Cu)` exact 구현 |
| Served/total load | `model.py`, `*_load_definition` | OD served + spot + batch overhead/work, CPU/MEM 분리 |
| Hard capacity | `model.py`, `capacity` | `load_total[s,r,t,xi] <= C[r] u[s,t]` |
| Server-time CVaR | `model.py`, `cvar_*` | `eta[s,t]`, `zeta[s,t,xi]`, normalized server loss |
| Energy | `model.py`, `energy_definition` | idle + served CPU + migration |
| Solver objective | `model.py`, `expected_profit` | expected spot − expected RT cost; constant OD/batch revenue 제외 |
| Reports | `reporting.py` | raw solver objective와 constant-offset total profit, per-server-time empirical CVaR, event diagnostics, lineage |

### 4.1 Exact OD CPU priority

서버별 OD actual CPU demand를

`D_OD[s,t,xi] = sum_i load_OD[i,CPU,t,xi] * x[i,s,t,xi]`

로 둔다. `v[s,t,xi]`에 대해 Notion의 indicator를 그대로 구현한다.

- `v=0 => D_OD <= C_CPU*u`
- `v=0 => excess=0`
- `v=1 => D_OD >= C_CPU*u`
- `v=1 => excess=D_OD-C_CPU*u`
- `v<=u`

이후 `load_od_cpu = D_OD-excess`로 두고 spot, batch startup, batch work를 더한
`load_total_cpu`에 hard capacity를 적용한다. 단순히
`load_od_cpu+excess=D_OD`만 두는 기존 방식은 OD를 의도적으로 덜 서비스하고 spot을
살릴 수 있으므로 허용하지 않는다.

### 4.2 서버-시간 CVaR

각 `(s,t)`에서 loss는 `R[s,t,xi]=excess[s,t,xi]/C_CPU`이다.

```text
R[s,t,xi] <= eta[s,t] + zeta[s,t,xi]
eta[s,t] + 1/(1-alpha) * sum_xi p[xi] zeta[s,t,xi] <= epsilon
```

기존의 `CVaR(sum_s excess[s,t,xi])`는 새 명세가 아니며 사용하지 않는다. Report는
model auxiliary expression과 scenario loss에서 다시 계산한 empirical VaR/CVaR를 모두
`(server_id,t)` 단위로 저장한다.

### 4.3 Objective accounting

```text
constant_od_revenue = sum_i sum_t_in_A_i pi_OD[i]
constant_batch_revenue = sum_k pi_B[k] W[k]
expected_spot_revenue = sum_xi p[xi] sum_j,s,t pi_SP[j] y[j,s,t,xi]
expected_rt_cost = sum_xi p[xi] sum_t P_rt[t,xi] E[t,xi]
profit = constant_od_revenue + constant_batch_revenue
         + expected_spot_revenue - expected_rt_cost
```

Gurobi에 주는 objective는
`raw_solver_objective = expected_spot_revenue - expected_rt_cost`이다. Report는
`constant_revenue_offset = constant_od_revenue + constant_batch_revenue`를 독립적으로
계산하고, incumbent와 bound에 같은 offset을 더해 total-profit incumbent/bound를 보고한다.

상수항을 objective에서 빼는 것은 feasible region, LP relaxation, absolute incumbent-bound
차이를 수학적으로 바꾸지 않는다. 즉 root relaxation을 강화한다고 표현하지
않는다. 그러나 Gurobi relative `MIPGap`은 raw objective의 크기를 분모로 삼으므로,
상수 offset을 포함했던 이전 solver gap과 새 raw solver gap은 동일 지표가 아니다.

### 4.4 Integrality relaxation and assignment orbitope

`y[j,s,t,xi]` 및 `m[i,t,xi]`는 continuous `[0,1]`이고 `h[j,t,xi]`는 binary이다.
Initial spot row가 하나의 server만 선택하고 `h`가 binary이며
`sum_s y[j,s,t,xi]+h[j,t,xi]=sum_s y_init[j,s]`이므로 spot placement의 연속
완화는 linked state의 의미를 유지한다. Migration `m`은 양방향 placement-change
lower bound를 유지한다.

OD initial-assignment 행을 먼저, Spot 행을 나중에 두어
`A=(a[r,s])`, `R=|I|+|J|`를 구성한다. 다음을 모두 적용한다.

```text
a[r,s] = 0                                           for s > r
sum_(ell=s)^min(r,|S|) a[r,ell]
  <= sum_(q=1)^(r-1) a[q,s-1]                       for r=2..R,
                                                       s=2..min(r,|S|)
```

OD를 앞에 두는 것은 모든 OD 행이 반드시 하나의 server를 선택하므로 server
도입 순서를 안정적으로 고정하기 위해서다. 이 제약은 homogeneous-server
column permutation으로 생기는 대칭해만 제거하고, physical server별 parameter가
다른 heterogeneous fleet variant에는 그대로 적용하지 않는다.

## 5. 데이터 lineage

### 5.1 Google ClusterData 2019

```text
data/raw/google2019_cell_a_day0_cpu_distribution
  -> raw usage + instance/collection/machine event history + extraction metadata
  -> exact raw-row deduplication before interval or lifecycle splitting
  -> lifecycle reconstruction per parent key (collection_id, instance_index)
       -> SCHEDULE starts an episode
       -> missing SCHEDULE may use UPDATE_RUNNING as a defensive start
       -> EVICT / FAIL / FINISH / KILL / LOST closes the episode
       -> no terminal event: close at day_end_us as HORIZON_CENSORED
       -> usage without a usable running transition: auditable full-horizon
          usage-inferred HORIZON_CENSORED episode
  -> stable model VM key: (collection_id, instance_index, episode_index)
  -> split source intervals at episode boundaries and retain positive overlap only
  -> source CPU maximum = cpu_usage_distribution[10] (p100)
       -> ignore maximum_usage.cpus for q; keep it only as raw provenance
  -> strict episode quality gate before min-row/sampling
       -> any empty/unavailable CPU distribution: drop episode
       -> any assigned machine without active history: drop episode
       -> any fragment without complete positive CPU-capacity coverage: drop episode
  -> representative server unit conversion
       -> machine-time weighted modal positive joint (capacity_cpu, capacity_mem) pair
       -> CPU/MEM divided by the two capacities from that same physical shape
       -> converted server capacity = (1.0, 1.0); no utilization calibration
       -> converted q_CPU > 1 or q_MEM > 1: drop VM
  -> class precedence at episode arrival: SCHEDULER_BATCH first
       -> remaining priority >=120: on_demand
       -> remaining priority <120 or missing: spot
  -> seed-42 episode-key deterministic selection: OD 50, spot 50, batch jobs 50
  -> each service VM has >=1 observation in every active 30-minute slot
  -> observed 5-minute values grouped into 48 local 30-minute slots
       -> CPU = coverage_us-weighted mean of available observations
       -> MEM = coverage_us-weighted mean of available observations
       -> no interpolation, zero-fill, or 6/n scaling
  -> q_CPU = max(episode-arrival request, episode p100 maximum)
  -> q_MEM = max(episode-arrival request, source maximum_usage.memory maximum)
  -> scenario 0 observed trace; scenario 1..19 mean-preserving lognormal draws
       -> every synthetic CPU/MEM value clipped to [0, q]
  -> 20 scenario tables + probability table + selection/draw manifest
```

로컬 extract의 `cpu_usage_distribution`은 `p0,p10,...,p100`의 11개 summary이다. 이 중
p100은 configured CPU maximum과 strict availability validation에 사용한다. Percentile
summary를 time-aligned scenario observation으로 간주하지 않으므로 Notion의 inverse-CDF
scenario construction은 사용하지 않는다. Scenario 0은 관측 average-usage trace이며,
scenario 1~19는 고정 seed의 독립 평균-보존 lognormal multiplier(CPU sigma 0.22,
MEM sigma 0.08)로 만든 뒤 모든 CPU/MEM 값을 VM별 `[0,q]` 범위로 cap한다. 이 명세 이탈,
독립 draw 가정, seed와 cap을 data audit에 기록하고 historical joint day로 표현하지 않는다.

Episode quality gate는 source row에 대해 먼저 수행한다. Exact duplicate는 distribution을
포함한 모든 raw field가 같은 row를 한 번만 남긴다. 그 밖의 overlapping interval은
elementary-interval union으로 중복 시간을 한 번만 세고 동시 non-NULL observation을
resource별 평균한다. Distribution이 빈 row는 평균 CPU/MEM 값이 유효하더라도 episode를
drop한다. Assigned-machine coverage는 episode와 horizon으로 clip된 각 positive-duration
fragment 전체에 positive reconstructed CPU capacity가 존재해야 complete이다. Machine
capacity가 fragment 중간에 UPDATE되어도 coverage union이 전체를 덮으면 허용한다.
Fully covered p100이 capacity를 넘는 경우는 drop 조건이 아니며 cap, replacement, target-load
rescaling 없이 별도 audit에 남긴다.

선택 가능 조건은 각 active 30분 slot에 관측값이 하나 이상 존재하는지뿐이다. CPU와 memory
집계에는 실제로 존재하는 5분 관측값과 각 행의 `coverage_us`만 사용해 두 resource 모두
coverage-weighted average를 계산하고, 존재하지 않는 값을 생성하거나 기대 row 수에 맞춰
확대하지 않는다. VM 개수와 class count는 정수로 유지하고, 집계된 CPU/MEM
workload와 이후 scenario workload는 소수형으로 사용한다. Terminal event가 없는 episode는
종료 event를 임의로 합성하지 않고 `HORIZON_CENSORED` provenance를 유지한다. 이미 horizon
시작에 running인 episode만 full horizon을 차지하며, horizon 안에서 시작 근거가 있는 open
episode는 그 시작부터 `day_end_us`까지 active이다.

### 5.2 Batch jobs와 family

`SCHEDULER_BATCH` job 50개를 seed 42로 선택한다. 먼저 최종 configured `(q_cpu,q_mem)`의
exact pair를 만들고, 이를 정렬한 뒤 최대 10개의 deterministic rank bin으로 묶는다.

- `W_k`: family 내 job의 horizon-overlap duration을 30분 active job-slot로 환산한 합
- `rho[k,r]`: active job-slot당 평균 actual resource consumption
- `q_B[k,r]`: prepared VM의 configured resource limit
- startup/base overhead: CPU `(0.05,0)`, MEM `(0,0.05)` 구조를 loader가 검증

Exact configured pair 수가 10 이하면 pair별 family를 그대로 사용한다. 10을 넘으면 정렬된
pair rank를 `max_batch_families: 10`개 bin에 결정론적으로 배정하되 동일 pair는 분리하지
않는다. Bin의 `q_cpu/q_mem`은 구성 job의 최대 configured resource이고, `W`와 resource
volume은 합산하며 `rho`는 합산 volume을 합산 `W`로 나눈다. Job ID, exact pair, family ID,
workload와 family별 집계를 data audit에 기록한다.

### 5.3 NYISO RT price

```text
NYISO public 2019 NYC zonal RT LBMP ZIP (native 5-minute, local timestamp)
  -> timestamp를 interval end로 해석하고 5분을 빼 interval start로 변환
  -> America/New_York DST-aware 5-minute control grid
  -> duplicate timestamp/zone mean collapse
  -> interior missing grid point: nearest previous/next raw point의 2-point time interpolation
  -> boundary missing grid point: explicit forward/backward fill, 별도 audit flag
  -> six 5-minute points의 arithmetic mean으로 30-minute RT LBMP
  -> USD/MWh / 1000 = USD/kWh
  -> price sensitivity multiplier
  -> max(0.01, adjusted price)
  -> 2019-06-01..2019-07-10 date를 xi 순서로 pairing
```

“raw price”라는 표현은 보간 전 observation에만 쓴다. Regularized/30-minute price를 raw로
부르지 않는다. 각 30분 slot은 raw observation 수, interior-interpolated point 수,
endpoint-filled point 수를 audit에 저장한다. Model-facing loader는 이 processed artifact에
추가 interpolation, forward fill 또는 silent clipping을 하지 않는다.

### 5.4 30분 energy scaling

SPECpower 등의 power curve가 kW 또는 W로 주어질 때 30분 energy coefficient는 다음과
같이 만든다.

```text
E_idle[kWh/slot] = idle_power[kW] * 0.5 h
E_CPU[kWh/(normalized CPU*slot)] = load_slope[kW] * 0.5 h
```

W 입력은 먼저 kW로 `/1000` 변환한다. 이미 kWh/30min인 source를 다시 `0.5`배하지
않는다. Source unit, conversion expression, source hash를 manifest에 기록한다. Price가
USD/kWh이므로 `P_rt*E`만 USD가 되어야 한다.

Batch `W`, per-slot revenue, minimum on/off time, active interval도 같은 30분 grid를
사용해야 한다. 기존 hourly coefficient/table을 label만 바꾸어 재사용하지 않는다.

## 6. 명세 충돌과 결정

### DEC-01 — scalar migration energy coefficient

최신 명세는 이전의 가격 연동식과 두 calibration mode를 제거했다. 모든 OD VM에 공통인
`c_mig=0.05 kWh/normalized-MEM`을 사용하며 migration energy는
`c_mig * load_OD_MEM[i,t,xi] * sum_s m[i,s,t,xi]`이다. RT price를 곱한 값만
USD 비용이 된다.
Sensitivity는 mode나 kappa가 아니라 scalar `migration.coefficient`를 직접 변경한다.

### DEC-02 — exact destination migration와 lower-bound batch startup

최신 formulation은 migration을 destination server별 `m[i,s,t,xi]`로 두고
`m>=x_next-x_now`, `m<=x_next`, `m<=1-x_now`를 모두 적용한다. 따라서 binary
placement incumbent에서 `m`은 실제 destination entry와 정확히 일치한다. Optimization
절에 따라 `m` 자체는 continuous `[0,1]`로 완화한다. Batch `z_on`만 기존
lower-bound-only event 정의를 유지하며 false-positive 가능성을 별도 감사한다.

Parameter 표의 `M_i`는 값과 제약식이 없으므로 migration-count cap으로 구현하지 않는다.

### DEC-03 — OD constant revenue와 CPU excess

명세는 OD가 전부 서비스되므로 revenue가 상수라고 하지만, `L_exc>0`은 actual CPU
demand 일부가 served load에서 제외됨을 뜻한다. Baseline 해석은 다음과 같다.

- “전부 서비스”는 active slot 동안 VM availability/placement를 보장한다는 뜻이다.
- CPU performance degradation은 server-time CVaR QoS constraint로만 통제한다.
- 허용 범위 안 excess에 별도 monetary SLA penalty나 revenue rebate를 적용하지 않는다.
- Report는 full OD accounting revenue와 served/excess CPU를 동시에 공개한다.

### DEC-04 — “Server Min.”과 objective

제목의 “Server Min.”은 서버 수 최소화 objective가 아니라 server minimum on/off duration을
가리키는 것으로 해석한다. 권위 수식에 따라 objective는 expected profit maximization이다.
Active server count는 결과 metric이며 lexicographic/secondary objective가 아니다.

### DEC-05 — energy 구성 설명

개요는 energy를 idle과 CPU load 두 항이라고 쓰지만 energy 수식은 migration을 추가한다.
Displayed energy equation을 우선하여 migration term을 포함한다.

### DEC-06 — 초기 server state

명세는 horizon 이전 `u` 상태와 residual up/down time을 지정하지 않는다. Literal baseline은
`t=1`의 `u`를 자유 initial state로 두며 첫 slot 이전 startup cost/dwell을 소급하지 않는다.
이 가정을 manifest와 server report에 표시한다.

### DEC-07 — synthetic workload-price pairing

20개 workload sample과 20개 NYISO historical date는 같은 `xi` index로 순서 결합한다.
이는 합성 joint scenario이며 실제 동시 관측, 인과관계 또는 상관 calibration이 아니다.
Sensitivity 결과도 이 한계를 유지한다.

### DEC-08 — deterministic batch family 축약

최신 baseline은 선택된 50 job을 최대 10개 family로 축약한다. Exact configured pair를
정렬한 rank에 따라 결정론적으로 bin을 만들고 동일 pair는 분리하지 않는다. 이 K<=10
mapping은 현재 baseline identity의 일부이며 selection seed, pair 순서와 job-to-family
mapping을 audit에 보존한다.

### DEC-10 — risk sweep domain

Displayed CVaR 식의 `1/(1-alpha)` 때문에 `alpha=1.0`은 정의되지 않는다. Sweep은 유효한
`alpha=[0.5,0.8,0.9,0.95]`만 직접 사용하고 보정값으로 대체하지 않는다.
`epsilon=[0.5,0.1,0.05,0.0]`이며, `epsilon=0`은 nonnegative excess가 모든 scenario에서
0이어야 함을 뜻한다.

### DEC-11 — zero-count workload class

Workload block의 `OD150`은 spot/batch가 0이고 `OD75+SP75`는 batch가 0인 named empirical
composition으로 해석한다. Loader, model, report는 empty `J` 또는 `K`를 정식 지원한다.
세 block은 resource-time 등가 scale이 아니므로 class-mix 효과로 단정하지 않는다.

### DEC-12 — lifecycle episode와 strict source-quality gate

같은 `(collection_id,instance_index)`가 terminal event 뒤 다시 running이 되면 별도 VM
execution으로 취급한다. 따라서 deterministic model-input key와 hash key에는
`episode_index`를 포함한다. Episode index는 horizon 이전 event history도 순서에 포함해
인접 day extract 사이에서 같은 execution identity가 바뀌지 않도록 한다.

Terminal reason은 `EVICT`, `FAIL`, `FINISH`, `KILL`, `LOST`를 그대로 보존한다. Terminal
event가 없는 execution은 합성 종료 reason 대신 `HORIZON_CENSORED`를 쓴다. Source usage가
있지만 running transition이 전혀 복원되지 않는 parent key는 silently old VM unit으로
되돌리지 않고 full-horizon usage-inferred censored episode 하나로 표시한다.

CPU maximum은 source `maximum_usage.cpus`가 아니라 11-value distribution의 마지막 값
`cpu_usage_distribution[10]`이다. Empty distribution, missing assigned-machine history,
incomplete positive CPU-capacity coverage는 서로 중복 가능한 episode-level exclusion reason이다.
이 gate는 minimum usage row와 deterministic sampling보다 먼저 적용한다. Capacity exceedance
자체는 coverage 불완전과 다른 조건이며, complete coverage 아래의 p100 exceedance를
임의로 cap하거나 drop하지 않는다.

### DEC-13 — constant revenue와 solver MIP gap

OD와 batch revenue는 decision-independent constant이므로 Gurobi objective에서 제외한다.
Reporting은 이 offset을 incumbent와 bound에 동일하게 더해 accounting total profit을
제공하지만, solver가 제공한 raw `MIPGap`을 total-profit gap으로 재계산하지
않는다. Constant 제거는 root LP를 강화하지 않고 relative gap denominator만
변경하므로, 이전 total-objective run과 v2의 relative gap은 성능 개선의 동일 척도로
직접 비교하지 않는다.

### DEC-09 — NYISO 2-point interpolation

NYISO control-grid 내부 결측은 두 인접 raw 관측 사이에서만 선형 보간한다. Edge fill은
보간으로 위장하지 않고 별도 lineage 상태로 남긴다. 30분 집계 후 model loader는 추가
보간하지 않는다.

## 7. Reporting contract

각 solved run은 최소 다음을 남긴다.

- `summary.json`: solver 상태, raw solver objective/bound/gap, constant revenue offset,
  total-profit incumbent/bound, 모든 objective component, mode/warning
- `server_schedule.csv`: `u,u_on,u_off`, minimum-duration audit
- `od_initial_placement.csv`, `od_schedule.csv`, `migrations.csv`: placement와 actual server change
- `spot_schedule.csv`: acceptance, first-slot activity, irreversible preemption
- `batch_schedule.csv`: preparation/startup/work volume와 completion residual
- `server_scenario_load.csv`: OD demand/served/excess, total CPU/MEM과 capacity audit
- `cvar_by_server_time.csv`: normalized scenario loss, model eta/expression, empirical VaR/CVaR
- `energy_rt_cost.csv`: idle/load/migration energy, RT price, slot cost
- `prepared_data/data_audit.json`: source hashes, IDs, dates, seeds, interpolation/binning audit

Google data audit에는 episode key와 terminal reason, `HORIZON_CENSORED` 여부, exact duplicate
제거 수, strict gate 전후 episode 수, 서로 중복 가능한 drop reason별 수, termination reason별
retained/dropped 수, assigned-machine coverage 정의와 p100 provenance를 포함한다.

Objective report는 `expected_spot_revenue-expected_rt_cost`로 raw Gurobi objective를 독립
재구성하고 tolerance를 넘으면 run을 실패시킨다. Total profit은 재구성한
raw objective에 constant OD/batch revenue를 한 번만 더해 검증한다. Gurobi
`MIPGap`은 raw solver objective에 대한 값임을 표시한다.
Migration report는 scalar coefficient, actual memory, migration energy(kWh), RT cost(USD)를
분리한다. Destination별 indicator를 transition 단위로 합산하고 실제 server change와의
definition residual을 검증하므로 `c_mig=0`에서도 false-positive migration은 허용되지 않는다.

## 8. 구현 단계

### Phase 0 — 계약 고정

- [x] 명세 요구사항과 충돌 추출
- [x] 최신 literal baseline 고정
- [x] two-stage OFAT sweep 정의

### Phase 1 — 30분 data foundation

- [x] Google 30분 slot과 configurable scenario builder
- [x] Lifecycle episode key와 terminal/`HORIZON_CENSORED` provenance
- [x] Distribution p100 CPU maximum과 strict assigned-machine quality gate
- [x] Batch 50-job selection 및 deterministic K<=10 family mapping
- [x] NYISO RT 30분 lineage/audit adapter
- [x] Server energy unit conversion validation
- [x] Selection/scenario/date/source checksum audit

### Phase 2 — formulation

- [x] `InstanceData`와 strict validation
- [x] 1st/2nd-stage Gurobi variables
- [x] Exact positive-part OD excess와 strict priority
- [x] Per-server-time normalized CVaR
- [x] RT-only energy/objective와 scalar migration coefficient
- [x] Spot admission `w` 제거 및 `sum(y_init)` state link
- [x] Continuous `[0,1]` spot recourse `y`/migration `m`, binary preemption `h`
- [x] `v<=u`와 OD-first/Spot-second initial-assignment packing orbitope
- [x] Raw solver objective와 constant-offset total-profit accounting 분리

### Phase 3 — reports and tests

- [x] Objective reconstruction
- [x] Empirical per-server-time CVaR
- [x] Exact destination-migration definition과 startup false-positive diagnostics
- [x] Baseline shape/unit/lineage tests
- [x] Removed-feature absence tests

### Phase 4 — execution

- [x] 구 baseline(50/50/K10/S10) prepare-only audit
- [x] Reduced real-data solve (2 scenarios, OD/SP/batch=2/2/4, servers=2)
- [ ] Full baseline solve
- [x] 직전 40-scenario baseline(100/100/K100/S6) prepare-only 재검증 (역사적 기록)
- [x] v4 episode/p100/strict processed artifact 생성과 6-server baseline/model-build 재검증
- [x] Micro-stress artifact(10 scenarios, 10/10/10, K<=3, S6) 및 직전 suite 생성
- [x] Formulation-v2 전용 11-case OFAT 계획과 source/spec hash 고정
- [x] Formulation-v2 micro-stress background rerun 완료
- [x] Destination-indexed migration-v3 3-hour 11-case OFAT 실행 완료
  (원 실행 5개 + retry 6개; 11/11 `TIME_LIMIT`, target `MIPGap=0.001` 미도달)
- [ ] Two-stage OFAT sensitivity 실행

## 9. Acceptance criteria

- [x] Revised baseline has exactly `|T|=48`, `|Xi|=20`, OD 50, spot 50,
  batch jobs 50, at most 10 deterministic rank-binned batch families, servers 6, and scenario
  probability sum 1.
- [x] All server min-on/min-off parameters equal two 30-minute slots.
- [x] Accepted spot VM is active at its first active slot in every scenario.
- [x] Spot admission has no standalone `w`; acceptance equals `sum_s y_init[j,s]` and
  `sum_s y[j,s,t,xi]+h[j,t,xi]=sum_s y_init[j,s]`.
- [x] Recourse spot placement `y` and migration `m` are continuous `[0,1]`, while spot
  preemption `h` remains binary.
- [x] For every OD transition, `m[i,s,t,xi]` is exactly the destination-entry indicator
  and `sum_s m` equals the actual server-change indicator within numeric tolerance.
- [x] For every `(s,t,xi)`, solved excess equals
  `max(0,D_OD-C_CPU*u)` within numeric tolerance.
- [x] Every excess branch satisfies `v[s,t,xi] <= u[s,t]`.
- [x] Initial assignments use an OD-first/Spot-second packing orbitope with triangular
  fixings and all defined server-introduction inequalities.
- [x] No spot/batch CPU is served ahead of OD capacity under the strict-priority definition.
- [x] Memory and total served CPU never exceed on-server hard capacity.
- [x] CVaR variables and constraints are indexed by server and time; empirical normalized CVaR
  is at most `epsilon+tolerance` for every `(s,t)`.
- [x] Every batch family completes `W_k` in every scenario.
- [x] Energy and objective contain no DA, renewable, sale, ESS, or excess-penalty term.
- [x] Raw solver-objective reconstruction excludes constant OD/batch revenues; reporting adds the
  audited constant offset exactly once to incumbent and bound total profit.
- [x] Reported solver `MIPGap` remains the raw-objective gap and is not presented as directly
  comparable to a prior total-objective relative gap.
- [x] Migration uses one finite nonnegative scalar coefficient with kWh/MEM unit metadata.
- [x] Every selected OD/spot VM has at least one 5-minute observation in every active 30-minute
  slot; CPU and MEM are coverage-weighted means of available observations.
- [x] Every synthetic CPU and MEM workload satisfies `0 <= usage <= q`.
- [x] CPU/MEM unit conversion uses one machine-time modal joint server shape, every server has
  converted capacity `(1,1)`, and no selected VM has `q_cpu>1` or `q_mem>1`.
- [x] Every model-facing Google VM is keyed by
  `(collection_id,instance_index,episode_index)` and records one of
  `EVICT/FAIL/FINISH/KILL/LOST/HORIZON_CENSORED` as its termination reason.
- [x] CPU `max_cpu_usage` and `q_cpu` use `cpu_usage_distribution[10]` p100 and never
  `maximum_usage.cpus` or average-usage maximum as a fallback.
- [x] Any episode with an empty CPU distribution, missing assigned-machine history, or incomplete
  positive CPU-capacity coverage is removed before minimum-row filtering and deterministic sampling.
- [x] Exact duplicate source usage rows are removed before episode splitting; distinct overlapping
  observations use interval-union accounting rather than double-counted duration.
- [x] Missing 5-minute service observations are not interpolated, zero-filled, or rescaled by
  `6/n`; VM counts are integer and model-facing workloads are floating-point values.
- [x] OD-only and OD+spot named workload blocks prepare successfully with empty class sets.
- [x] Sweep uses the exact two-stage OFAT levels in §3.1 and excludes undefined `alpha=1.0`.
- [x] No model-facing service workload value is interpolated or filled; price preprocessing is
  limited to the separately documented NYISO policy.
- [x] All selected IDs, random seeds, scenario-date pairs and source hashes are reproducible.
- [x] Existing `2607-notion-energy-vmp` paths, outputs, processes and root symlinks remain unchanged.

## 10. 재현 interface

```bash
.venv/bin/python experiments/2607-notion-server-min-vmp/run_experiment.py \
  --config configs/baseline.yaml \
  --run-dir runs/baseline_seed42 \
  --prepare-only

.venv/bin/python experiments/2607-notion-server-min-vmp/run_experiment.py \
  --config configs/baseline.yaml \
  --run-dir runs/baseline_seed42

.venv/bin/python experiments/2607-notion-server-min-vmp/run_micro_stress_sweep.py \
  --plan plans/micro_stress_first_stage_symbreak_v2.yaml \
  --google-dir data/processed/notion_toy_google2019_micro_stress_v1
```

Runner는 단일 config를 준비/빌드/solve한다. `run_sweep.py`는
`plans/sweep_plan.yaml`에서 선택한 stage의 OFAT config를 전역 중복 제거하여 생성하고,
고유 run directory와 disjoint CPU affinity로 단일-run runner를 dispatch한다. Full
baseline solve는 sensitivity suite의 baseline candidate로 실행한다.
전용 micro-stress launcher는 최신 formulation snapshot과 code/data hash를 검증한 뒤
11개 OFAT candidate를 최대 5개씩 background로 dispatch한다.
