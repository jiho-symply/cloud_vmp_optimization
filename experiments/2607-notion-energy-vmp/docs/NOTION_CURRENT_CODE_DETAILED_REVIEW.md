# Notion 모델과 현재 코드 상세 비교

## 검토 범위

- Notion 문서: `2 OD SP BC with Energy Proc Serv Prov`
- URL: `https://app.notion.com/p/2-OD-SP-BC-with-Energy-Proc-Serv-Prov-395a14fa720380f5bbd8cda62f50fc93`
- connector가 반환한 Notion snapshot 시각: `2026-07-09T18:28:49.033Z`
- repository: clone한 repository root
- 검토 방식: Notion과 코드를 읽기만 했으며 Notion에는 아무것도 작성하지 않았다.

차이는 다음 다섯 종류로 구분한다.

1. **수식 불일치**: 코드의 feasible region 또는 objective가 Notion 표시식과 다름.
2. **사용자 지정 parameter override**: 수식은 같지만 Notion 작성 후 baseline이나 coefficient를 의도적으로 변경함.
3. **데이터 대체/변환**: 수학 기호는 같지만 empirical source 또는 생성 방법이 다름.
4. **실험/문서 누락**: 현재 실험에서 수행하는 내용을 Notion이 설명하지 않음.
5. **formulation warning**: 코드는 Notion 식을 그대로 구현했지만 그 식 자체가 의도하지 않은 해를 허용할 수 있음.

## 핵심 결론

Notion에 표시된 MILP과 `src/notion_energy_vmp/model.py`는 매우 유사하다. 핵심 목적함수 항이나 제약식 family가 코드에서 빠진 것은 발견하지 못했다. 큰 차이는 수식 바깥에 있다.

- Notion은 Azure VM data를 설명하지만 현재 실험은 Google Borg 1일 workload pool, proxy service class, synthetic workload scenario를 사용한다.
- hourly aggregation, workload-energy scenario 결합, 가격 하한, renewable demand scaling, batch scaling 등이 Notion에 없다.
- Notion은 `kappa_SLA=2`와 차원이 맞지 않는 migration coefficient 공식을 적고 있지만 현재 설정은 사용자 지시에 따라 `0.2`와 energy coefficient를 사용한다.
- 178-run suite, solver 정책, diagnostics, 결과 해석 방법이 Notion에 없다.

따라서 **현재 코드는 Notion의 수학 모델에는 충실하지만, Notion 문서만으로는 현재 데이터 생성 과정, 실제 baseline parameter, 실험 프로토콜을 재현할 수 없다.**

## 1. 수식 수준 비교

### 1.1 변수와 stage

| 항목 | 판정 | 근거와 영향 |
|---|---|---|
| 1st-stage 변수 | 일치 | `x_init`, `y_init`, `z`, `z_on`, `w`, `u`, `u_on`, `u_off`, `g_da`는 scenario index가 없다. `model.py:42-52`. |
| 2nd-stage 변수 | 일치 | placement, migration, spot state, batch work, load/excess, CVaR, energy, RT grid, renewable, ESS가 scenario별로 정의된다. `model.py:54-71`. |
| `x`, `y`, `h`, `mig` domain | 동치인 sparse 구현 | 전체 `T`가 아니라 데이터의 VM별 active period에서만 생성된다. 사용자가 의도한 정의와 같다. `model.py:28-37`. |
| SoC index | 동치인 index 변환 | Notion `1,...,H+1`을 Python `0,...,H`로 표현한다. `model.py:71`. |
| terminal server transition | 동치인 구현 | `u_on/u_off`를 `T[:-1]`에만 만든다. `t+1`이 없는 terminal transition은 만들지 않는다. `model.py:49-51`. |

### 1.2 제약식과 목적함수

| family | 판정 | 중요한 세부사항 |
|---|---|---|
| Server transition/minimum on-off | 일치 | 0-based loop로 transition, exclusivity, persistence를 구현한다. `model.py:72-80`. 양쪽 모두 horizon 이전 상태가 없다. |
| OD initial placement/active service | 일치 | initial server 하나, scenario 공통 start linkage, active period 동안 켜진 서버 하나에서 서비스한다. `model.py:82-96`. |
| OD migration | 수식 일치 | placement 차이에 대한 두 lower bound만 있다. 서버가 바뀌면 `mig=1`이지만 안 바뀌었을 때 `mig=0`은 논리적으로 강제되지 않는다. `model.py:93-96`. |
| Spot admission/preemption | 일치 | admission, initial/fixed server, powered-server service, 누적 preemption, `sum(y)+h=w`가 같다. `model.py:98-112`. Google interruption은 제약이 아니다. |
| Batch preparation/startup | 수식 일치 | 첫 period equality와 이후 lower bound가 같다. `z_on`을 정확한 transition indicator로 만드는 upper bound는 없다. `model.py:114-122`. |
| Batch processing/completion | 일치 | 모든 scenario에서 `W_k`를 전부 처리하고 CPU/MEM limit을 만족한다. `model.py:124-135`. |
| OD load/excess/total capacity | 일치 | OD CPU는 served+excess, memory는 excess 없이 exact, spot/batch는 hard capacity다. `model.py:138-168`. |
| Time-wise CPU CVaR | 일치 | aggregate OD CPU excess와 표시된 CVaR upper bound가 같다. `model.py:170-181`. |
| Energy | 구조 일치 | idle, served CPU, memory-proportional migration energy를 포함한다. `model.py:183-198`. coefficient 생성식 차이는 3절에서 설명한다. |
| Renewable/ESS | 일치 | renewable 사용/판매, balance, SoC dynamics, limits, terminal SoC가 같다. `model.py:200-213`. |
| Objective | 일치 | expected spot revenue와 renewable sale에서 DA/RT cost와 OD CPU excess penalty를 뺀다. 상수 OD revenue는 제외한다. `model.py:215-230`. |

### 1.3 오해하기 쉬운 의미

1. `h[j,t,xi]`는 그 slot에서 발생한 event가 아니라 **그 시점까지 preempt된 누적 상태**다.
2. 양쪽 수식 모두 `w_j=1`인 VM이 첫 active slot부터 `h=1`이 되는 것을 허용한다. 즉 accepted지만 한 slot도 서비스하지 않은 spot VM이 가능하다.
3. Migration energy는 실제 server change가 아니라 `mig` indicator에 부과된다. nonmovement indicator도 모델 안에서는 energy를 소비한다.
4. CVaR auxiliary는 risk estimator를 최소화하는 변수가 아니라 feasibility 변수다. 제약이 slack이면 auxiliary expression이 실제 empirical CVaR보다 클 수 있다. 결과 해석에는 `reporting.py:410-428`의 empirical CVaR를 사용해야 한다.
5. 이 실험의 Gurobi `OPTIMAL`은 `MIPGap=0.001` 허용오차 안의 optimal status이며 zero-gap 증명과 같지 않다.

## 2. 데이터 source와 생성 방법

### 2.1 Workload와 service class

| 항목 | Notion | 현재 pipeline | 영향 |
|---|---|---|---|
| Workload source | Azure Public Dataset V2 30일 public-cloud VM trace | Google ClusterData 2019 Borg 1일 pool; `(collection_id, instance_index)`를 VM-like unit으로 해석 | Notion의 empirical setup 재현이 아니다. |
| OD | Azure interactive category | batch가 아니고 Google priority `>=120` | 실제 on-demand 계약이 아니라 production-priority proxy다. |
| Spot | Azure delay-insensitive/unknown | batch가 아니고 priority `<120` 또는 missing | 실제 spot 구매형태가 아니라 lower-priority proxy다. |
| Batch | Batch VM category | `SCHEDULER_BATCH` collection | scheduler proxy를 10개 family로 집계한다. |
| Nominal resource `q` | vCPU bucket과 utilization 기반 | `max(resource_request, 1.2*p95 usage)` | VM 크기 의미와 단위가 다르다. |
| Realized load `ell` | Azure utilization sampling | Google scenario CPU hourly mean, memory hourly max | 실제 optimization 해상도는 1시간이다. |

Proxy rule은 `src/google2019_toy/build_toy_instance.py:320-349`에 있다. Source pool은 OD 3,476, spot 789, batch candidate 5,735개다. Baseline은 OD 69, spot 17개를 선택하고 batch candidate 114개 분량을 10개 family로 표현한다.

이는 `I`, `J`, `K`라는 수학적 set을 바꾸지는 않지만 외적 타당성을 크게 제한한다. Spot admission, SLA, migration 결과를 실제 public-cloud 고객 행동으로 표현해서는 안 되고 **normalized toy instance의 proxy class 결과**라고 해야 한다.

### 2.2 Active period와 missing-data 정책

Active interval은 계약/request 기간이 아니라 관측 범위로 만든다.

```text
arrival_t5   = min(observed t5 bucket)
departure_t5 = max(observed t5 bucket) + 1
```

Loader는 이를 `arrival_t5//12`부터 `(departure_t5-1)//12`까지의 inclusive hourly interval로 바꾼다. `data.py:166-174`. 따라서 one-day extraction 경계와 관측 gap이 active-period 의미에 영향을 줄 수 있다.

Model-facing loader는 엄격하다. 선택된 모든 scenario의 active hour가 완전한 VM만 sampling하며 보간이나 fill을 하지 않는다. `data.py:432-496`. 재생성한 200/10 audit는 다음을 확인한다.

- eligible complete VM: OD 3,257, spot 598;
- selected VM-scenario pair: 860;
- missing active hour: 0;
- duplicate active hour: 0.

그러나 upstream Google normalizer는 nonnumeric/missing CPU와 MEM을 `0.0`으로 바꾼다. `build_toy_instance.py:126-134`. 현재 audit는 downstream completeness는 증명하지만, 전체 lineage에 imputation이 없었다거나 upstream에서 몇 개를 0으로 바꿨는지는 증명하지 않는다.

### 2.3 Workload scenario는 historical day가 아니다

- Scenario 0만 observed Google trace다.
- Scenario 1-9는 각 5-minute row에 mean-preserving lognormal noise를 독립적으로 곱한다. CPU sigma `0.22`, memory sigma `0.08`. `build_toy_instance.py:371-407`.
- Synthetic scenario 1-9에서만 non-OD CPU를 nominal CPU로 cap하고 모든 class의 memory를 nominal memory로 cap한다. Scenario 0 observed usage는 nonnegative clip만 적용되고 nominal resource cap은 적용되지 않는다.
- CPU는 hourly mean, memory는 hourly max가 된다. `build_toy_instance.py:410-422`.
- Scenario probability는 모두 `0.1`이다.

이는 한 base trace의 shape에 controlled variation을 주는 방식이다. 여러 historical workload day에서 cross-day distribution, temporal tail dependence, probability를 추정한 것이 아니다.

### 2.4 Workload와 energy uncertainty는 공동 관측이 아니다

Workload scenario `0,...,9`를 NYISO/NREL date `2019-07-01,...,10`과 순서대로 연결한다. `toy_200_real.yaml:29-39`, `data.py:620-638`.

- Google workload와 NYISO price/NREL weather 사이에는 공동 관측 관계가 없다.
- 모델은 joint scenario처럼 사용하지만 dependence는 data에서 추정한 것이 아니라 index로 부여한 것이다.
- Seasonal run은 workload scenario는 그대로 두고 energy date block만 바꾼다.

따라서 현재 실험으로 실제 workload-price-renewable correlation 효과를 주장할 수 없다.

### 2.5 NYISO price

Notion은 DA hourly 값을 5-minute grid로 forward-fill하고 RT 5-minute 값을 사용한다고 적는다. 현재 optimization은 processed 30-minute NYISO file을 읽고 두 observation을 hourly mean으로 만들며 USD/MWh를 USD/kWh로 변환한다. `data.py:268-320`.

Baseline은 다음과 같다.

- zone: `NYC`;
- DA: 선택한 10일의 hour-wise mean인 1st-stage 값;
- RT: scenario별 하루;
- raw DA: `0.01494`-`0.05321 USD/kWh`;
- raw RT: `-0.02667`-`0.11854 USD/kWh`;
- model floor: `0.01 USD/kWh`, RT 240 slot 중 2개, DA 0개에 적용;
- upper cap: 없음;
- sell price: `0.85*P_DA`, NYISO 관측값이 아닌 model assumption.

Raw file은 변경하지 않았고 audit가 raw/applied 통계를 모두 보존한다.

### 2.6 NREL renewable과 지역 mismatch

Renewable input은 measured plant output이 아니다. NREL resource trace를 PV/wind model과 assumed capacity로 generation으로 바꾼 것이다. Upstream은 이미 site 평균수요 100 MW, renewable-to-demand ratio `2.0`, annual energy mix solar 20%/wind 80%를 가정한다. `configs/nrel_renewables_2019_v0.yaml:48-52`.

Experiment loader는 다시 model reference demand 평균에 맞추는 scale을 적용한 뒤 solar/wind multiplier를 독립 적용한다. `data.py:661-709`. Baseline audit는 다음과 같다.

- raw mean component energy: hourly period당 `98,527.42 kWh`(한 시간 평균 power로 쓰면 약 `98.53 MW`);
- loader scale: `1.623298e-5`;
- scaled model mean: hourly period당 `1.599394 kWh`;
- 선택한 10일 solar mean: hourly period당 `0.766588 kWh`;
- 선택한 10일 wind mean: hourly period당 `0.832806 kWh`.

즉 원래 MW magnitude는 대부분 사라지고 temporal profile shape와 relative capacity multiplier를 연구하게 된다.

가격은 NYISO `NYC`, renewable은 NREL `VA` site다. Clock은 둘 다 `America/New_York`로 맞지만 같은 지역 또는 같은 전력시장 자원은 아니다. 이는 Notion의 “전력가격과 매칭되는 한 지역”보다 느슨한 proxy 결합이다.

`solar=1, wind=1`은 equal capacity라는 뜻이 아니라 upstream capacity construction을 보존한다는 뜻이다. 현재 4x4 sweep은 renewable 조성과 total volume을 동시에 바꾼다.

### 2.7 Batch, server, interruption proxy

- Batch candidate는 최대 10개 family로 묶는다. 현재 model은 deterministic total `W_k`만 사용하고 upstream `batch_workload.csv`의 scenario별 시간 분포는 사용하지 않는다.
- Baseline batch scale은 `114/5735=0.01987794`이며 family별 `W_k`는 약 `11.39-11.41`이다. Historical workload unit이 아니라 experiment-size proxy다.
- Server는 homogeneous toy row다: normalized CPU/MEM capacity `1`, idle energy `.35`, CPU marginal energy `.65`, minimum on/off `1`. Notion에 예정된 SPECpower calibration은 아직 수행하지 않았다.
- Google spot availability는 diagnostic only다. 실제 EVICT가 있으면 사용하지만 없으면 priority-based synthetic hazard를 사용한다. `build_toy_instance.py:468-518`. Baseline의 36,860 selected row는 spot-interruption ground truth가 아니며 endogenous preemption을 제약하지 않는다.
- 이 정책 차이는 실제 결과에서도 크다. 200/10 baseline은 diagnostic availability가 0이 된 뒤에도 endogenous model에서 spot이 active인 VM-scenario-hour slot이 1,237개다. 이는 의도된 외부 비교이며 feasibility violation은 아니지만, 현재 model policy가 comparison trace를 재현한다고 해석해서는 안 된다.

### 2.8 데이터 위험 우선순위

1. Google priority/scheduler proxy가 OD/spot/batch를 대표하는지;
2. synthetic workload scenario와 임의적인 workload-energy scenario 결합;
3. renewable scale, ESS, revenue, SLA, migration energy까지 연쇄 영향을 주는 normalized toy server-power calibration;
4. upstream missing usage의 0 대체 가능성;
5. NYC price와 Virginia renewable의 geographic mismatch.

## 3. Parameter와 단위

### 3.1 Baseline 비교

| parameter | Notion | 현재 baseline | 판정과 영향 |
|---|---|---|---|
| Server capacity | `C_CPU`, `C_MEM`, 값 미기재 | 둘 다 normalized `1.0` | 물리 core/GB가 아니며 Notion calibration이 없다. |
| Server energy | SPECpower-based 선택 예정 | `E_idle=.35`, `E_CPU=.65`, full server-hour `1 kWh` | 문서화된 SPECpower fit이 아닌 synthetic calibration이다. |
| Minimum on/off | 기호만 있음 | 모든 서버 `1/1` hour | Notion baseline 누락. 한 시간 값이라 제약 효과도 작다. |
| Scenario probability | 기호만 있음 | 10개 각 `.1`, subset 선택 시 재정규화 | Notion baseline/정책 누락. |
| ESS | ratio `1`, limit `.1C`, efficiency `.95/.895`, initial `.5C` | 12-server baseline `C=12 kWh`, limit `1.2`, initial `6` | 일치. |
| `kappa_OD` | `10` | `10` | 일치. |
| Spot ratio | OD reference의 `.3` | `.3`, 선택 VM 평균 revenue `0.00186169 USD/slot` | 일치. 다만 sweep에서는 business assumption이다. |
| SLA multiplier | `kappa_SLA=2` | 사용자 지정 `.2` | 의도적 override. 현재 `lambda_exc=.0564909 USD/(normalized CPU excess-hour)`. |
| CVaR | 기호만 있고 baseline 불명확 | `alpha=.95`, `epsilon=.05` | Notion baseline 누락. 10개 equal-probability scenario에서는 5% tail보다 scenario atom 10%가 커 empirical CVaR가 사실상 worst sampled scenario에 지배된다. |
| Migration | `kappa_mig=.1` | `.1` | multiplier는 같지만 coefficient 생성식은 다르다. |
| Purchase price floor | 없음 | DA/RT `0.01 USD/kWh`, upper cap 없음 | 사용자 지정 assumption. |
| Sell price | 외생 `P_sell` | `max(0,.85*P_DA)` | Notion에 생성식이 없고 NYISO 관측값도 아니다. |
| Renewable capacity | workload scale을 정성적으로 언급 | reference demand ratio `1`로 공통 scale 후 solar/wind multiplier | 정확한 공식과 component multiplier가 Notion에 없다. |
| Solver | 없음 | `MIPGap=.001`, time limit 없음, suite run당 8 threads, SoftMemLimit 32 GiB | 실험 설정이며 수식 차이는 아니다. |

권위 있는 baseline은 `configs/toy_200_real.yaml:3-78`과 각 run의 `resolved_config.yaml`, 실제 계산값은 `prepared_data/data_audit.json`이다.

### 3.2 Migration coefficient의 실제 Notion 불일치

Notion parameter note는 다음과 같이 적는다.

```text
c_i^mig = kappa_mig * p_i^OD
```

`p_i^OD`는 money/slot이다. 이를 memory usage와 곱해 kWh 단위인 `E_t`에 더하면 차원이 맞지 않는다.

현재 loader는 다음을 사용한다. `data.py:722-735`.

```text
c_mig = kappa_migration * (E_idle + E_CPU*C_CPU) / C_MEM
      = 0.1 * (0.35 + 0.65*1) / 1
      = 0.1 kWh / normalized-memory
```

Memory usage를 곱하면 kWh가 되고 DA/RT procurement를 통해 간접 비용이 된다. 이는 displayed energy equation 변경이 아니라 **Notion parameter note에 대한 사용자 지정 단위 수정**이다.

Scale을 보면 normalized memory `1.0` 전체 migration은 baseline 평균 DA에서 약 `0.0028245 USD`다. 선택 VM의 최대 memory usage `0.0555908`에서는 약 `0.005559 kWh`, 평균 DA 기준 `0.000157 USD`다. 그래서 양의 coefficient가 임의 migration을 강하게 억제하지만 완전히 제거한다는 보장은 없다. 실제 200/10 baseline에도 nonmovement migration indicator 2개(동일확률 평균으로는 `0.2`)가 남았다.

### 3.3 Notion에 없는 batch parameter 생성

- 모든 `W_k`에 `selected batch candidates / 5735`를 곱한다. 200-VM baseline scale은 `0.01987794`. `data.py:581-589`.
- Source의 base/startup은 family별 값이지만 Notion은 공통 scalar다. Loader는 각 column의 최대값을 공통값으로 사용한다. `data.py:590-600`.
- 예를 들어 base CPU source range는 `0.0005454`-`0.0235803`이고 canonical value는 `0.0235803`이다. 대부분 family보다 보수적이며 capacity/energy에 영향을 줄 수 있다.

### 3.4 사용되지 않는 `model_params.json`

`configs/toy_200_real.yaml:21`은 `model_params.json`을 적지만 `data.py:394-404`의 실제 path map은 이 파일을 읽지 않는다. 이 파일에는 ESS efficiency `.92/.92`가 남아 있지만 실험은 YAML `.95/.895`를 쓴다.

현재 solve에는 YAML이 적용되므로 결과 오류는 아니다. 그러나 독자가 named file을 authority로 오해할 수 있는 실제 config/documentation defect다.

## 4. Notion에 없는 실험 프로토콜

### 4.1 Baseline

- 5-minute가 아니라 24 hourly period;
- OD 69, spot 17, batch candidate 114개 분량을 표현하는 10 family;
- homogeneous normalized server 12개;
- equal-probability synthetic workload scenario 10개;
- index로 연결한 NYISO/NREL historical date 10일;
- NYC price와 VA renewable;
- DA는 선택 date들의 hour-wise mean, RT/renewable은 scenario date별 값.

### 4.2 178-run suite

| phase | run | 목적 |
|---|---:|---|
| Smoke | 1 | 40 VM / 2 scenario end-to-end |
| Sampling | 3 | VM seed 42, 43, 44 |
| Scale | 4 | 100/200 VM과 5/10 scenario |
| Economics | 10 | SLA, spot ratio, migration OFAT |
| Risk | 6 | `alpha`, `epsilon` OFAT |
| Energy | 6 | aggregate renewable, ESS OFAT |
| Solar x wind | 16 | `{0,.5,1,2} x {0,.5,1,2}` full grid |
| Season | 4 | 계절별 10일 block |
| LHS | 128 | 8-factor interaction screen |

LHS range는 SLA `[0,.5]`, spot `[.1,.7]`, migration `[0,.5]`, alpha `[.8,.95]`, epsilon `[0,.2]`, solar/wind/ESS `[0,2]`다. 각 dimension의 stratum을 한 번씩 사용하지만 endpoint를 반드시 포함하지 않는다. `sweep_plan_v2.yaml:65-77`, `generate_experiment_suite.py:33-43`.

### 4.3 실행과 solver 의미

Rolling pool 12개를 사용한다. 각 run은 겹치지 않는 CPU 8개, Gurobi thread 8개, BLAS thread 1개, SoftMemLimit 32 GiB, NodefileStart 4 GiB를 받는다. 하나가 끝나면 그 slot이 다음 run을 즉시 시작한다. `run_experiment_suite.py:23-36,71-145`.

Time limit은 없고 `MIPGap=.001`이다. `OPTIMAL`은 relative gap 0.1% 허용오차 안의 status다.

### 4.4 말할 수 있는 것과 없는 것

말할 수 있는 것:

- baseline OFAT의 방향과 대략적인 threshold;
- 선택한 계절별 10일 block에서의 조건부 차이;
- 128 LHS 완료 후 sampled range 안의 association/interaction screen;
- false startup, simultaneous ESS, nonmovement migration, empirical CVaR 등 formulation diagnostic.

현재 설계만으로 말할 수 없는 것:

- Azure/public-cloud 일반화;
- 실제 interruption 정책의 causal effect;
- 실제 workload-price-renewable correlation effect;
- 4개의 10일 block으로 annual-average effect;
- fixed-total solar-vs-wind substitution. 현재 grid는 조성과 total capacity를 동시에 바꾼다;
- robust parameter effect. Seed는 3개뿐이고 parameter-by-seed 반복이 없다.

Partial LHS는 runtime-dependent 완료 순서 때문에 balanced Latin hypercube가 아니다. 현재 pairwise OLS도 in-sample R2만 있고 confidence interval, cross-validation, quadratic term이 없다. `aggregate_experiment_results.py:82-163`.

## 5. Notion과 코드에 공통으로 남은 formulation warning

이 항목은 code-vs-Notion 차이가 아니다. 코드는 표시식을 유지하고 결과를 진단한다.

| warning | 결과 | 현재 처리 |
|---|---|---|
| Pre-horizon server state 없음 | startup 없이 처음부터 켜지고 initial minimum-up obligation이 없는 서버가 가능 | initial on server를 report한다. `reporting.py:394`. |
| Migration lower bound only | 실제 server change 없이 `mig=1` 가능 | indicator, actual change, nonmovement를 분리 report한다. `reporting.py:88-113`. |
| Batch startup lower bound only | 0-to-1 transition 없이 `z_on=1` 가능 | false startup을 검출한다. `reporting.py:173-185`. |
| Accepted spot immediate preemption | admission count가 실제 서비스를 받은 수보다 클 수 있음 | schedule은 기록하지만 accepted-never-served 전용 summary metric은 아직 없다. |
| ESS exclusivity 없음 | simultaneous charge/discharge 가능 | 동시 slot을 검출한다. `reporting.py:370-378`. |
| CVaR auxiliary non-tight | auxiliary expression과 empirical CVaR가 다를 수 있음 | 둘 다 report하며 empirical 값을 해석에 사용한다. |
| Grid explicit upper bound 없음 | balance, positive price, terminal SoC, sink에 의해서만 간접 제한 | 추가 bound는 넣지 않았고 purchase price를 양수로 floor한다. |

현재 결과에서도 warning이 실제로 관측된다. 200/10 baseline에는 nonmovement migration indicator 2개와 comparison spot availability 이후 active slot 1,237개가 있다. 반면 partial aggregate 164개 시점에는 false batch startup과 simultaneous ESS charge/discharge가 발생한 run은 0개였고, nonmovement migration indicator는 37개 run에서 양수였다. 또한 10개 equal-probability scenario의 `alpha=.95` empirical CVaR는 사실상 worst sampled scenario 지표로 읽어야 한다.

## 6. 현재 무엇을 authority로 봐야 하는가

| 질문 | 현재 authority | 의존하면 안 되는 것 |
|---|---|---|
| 최적화 수식 | Notion 표시식 + `model.py`; 차이는 `MODEL_FIDELITY_REVIEW.md` | parameter 설명만 단독으로 사용 |
| 실제 baseline | run의 `resolved_config.yaml` + `prepared_data/data_audit.json` | 오래된 Notion baseline, `model_params.json` |
| VM/workload 생성 | `build_toy_instance.py` + `data.py` + audit | Notion Azure 설명 |
| Price/renewable alignment | `toy_200_real.yaml` + `data.py` + audit `energy_alignment` | Notion의 정성적인 5-minute 설명 |
| 실험 설계 | `sweep_plan_v2.yaml` + generated `manifest.csv` | 현재 실험을 설명하지 않는 Notion |
| 실제 solve 결과 | run `summary.json`, diagnostics, CSV, objective reconstruction, bound/gap | solver status 이름이나 raw objective 단독 |
| Migration | server change에는 `expected_actual_migrations`, nonmovement는 별도 | raw indicator count 단독 |
| CVaR | scenario excess에서 재구성한 empirical CVaR | slack auxiliary expression 단독 |

재생성한 40/2와 200/10 prepare-only output은 canonical checksum과 source SHA-256이 모두 일치한다. Baseline audit는 `kappa_sla=.2`, `c_mig=.1 kWh/normalized-memory`, price floor `.01 USD/kWh`, 현재 source file과 일관된다.

## 7. 문서와 향후 실험 권고

이 절은 권고일 뿐이며 Notion을 수정하지 않았다.

### Priority 0: 잘못된 해석 방지

1. Azure workload 절을 실제 Google Borg proxy pipeline으로 바꾸거나, 현재 실험을 별도 Google validation instance로 명시한다.
2. OD/spot/batch가 구매 class가 아니라 priority/scheduler proxy임을 크게 표시한다.
3. Scenario 1-9가 한 trace의 lognormal perturbation이고 energy date와 index로 결합됨을 기록한다.
4. `kappa_SLA=2`를 현재 `.2`로 고치거나 parameter version table을 둔다.
5. `c_i^mig=kappa_mig*p_i^OD`를 현재 energy-consistent coefficient와 단위로 바꾼다.
6. Suite `OPTIMAL`이 `MIPGap=.001` 기준임을 기록한다.

### Priority 1: 재현 가능한 empirical setup

1. Horizon, scenario, server, capacity, energy, minimum time, ESS, risk, price floor, sale ratio, renewable scale, economics를 포함한 baseline table을 만든다.
2. CPU mean, memory max, price mean, renewable energy sum의 hourly aggregation을 기록한다.
3. DA date mean과 scenario별 RT/renewable mapping을 기록한다.
4. NYC price와 VA renewable 결합이 의도적인지 설명하고 가능하면 동일 지역/market으로 맞춘다.
5. Batch `W_k` scaling과 base/startup max canonicalization을 기록한다.
6. 사용하지 않는 `data.model_params` config를 제거하거나 deprecated/non-authoritative로 표시한다.
7. Upstream CPU/MEM NaN의 zero replacement 전후 count를 audit한다.

### Priority 2: formulation 의미 명확화

1. `h`를 one-time event가 아니라 cumulative preempted state로 설명한다.
2. Accepted spot이 서비스 전에 preempt되어도 되는지 결정한다. 안 된다면 문서가 아니라 향후 수식 변경이 필요하다.
3. Migration과 batch-startup이 one-sided indicator임을 적는다.
4. CVaR auxiliary와 empirical CVaR를 구분한다.
5. Initial server state와 ESS exclusivity가 없음을 적는다.

### Priority 3: 실험 강화

1. Fixed-total renewable solar/wind composition sweep을 추가한다.
2. 중요한 parameter point를 workload seed와 season에 걸쳐 반복한다.
3. 실제 일반화가 필요하면 한 Google day perturbation보다 많은 independent historical workload period를 사용한다.
4. Absolute cost claim 전에 특정 SPECpower machine 또는 문헌으로 server energy를 calibrate한다.
5. LHS main/interaction에 uncertainty estimate 또는 cross-validation을 추가한다.
6. Spot discount는 operational control이 아니라 business/revenue scenario로 취급한다.

## 세 질문에 대한 직접 답변

1. **데이터와 실험 사용법은 Notion과 상당히 다르다.** 현재는 Google proxy class, hourly synthetic workload scenario, NYC/VA energy trace, demand-normalized renewable, 문서에 없는 여러 transformation을 사용한다.
2. **Parameter도 다르거나 Notion에 빠진 것이 있다.** 핵심은 SLA `2 -> .2`, migration energy 단위 수정, price floor, sell-price 공식, normalized server energy, batch scaling/canonicalization, solver 설정이다.
3. **핵심 최적화 수식은 대부분 같다.** 차이는 주로 sparse/index 구현과 parameter 생성이다. One-sided indicator, accepted-never-served spot, initial server state 부재, ESS nonexclusive, non-tight CVaR 같은 문제는 대부분 코드가 새로 만든 것이 아니라 Notion 표시식에도 있는 특성이다.

## 근거 파일

- `src/notion_energy_vmp/model.py`
- `src/notion_energy_vmp/data.py`
- `src/notion_energy_vmp/reporting.py`
- `src/google2019_toy/build_toy_instance.py`
- `configs/toy_200_real.yaml`
- `configs/nrel_renewables_2019_v0.yaml`
- `experiments/sweep_plan_v2.yaml`
- `scripts/generate_experiment_suite.py`
- `scripts/run_experiment_suite.py`
- `scripts/aggregate_experiment_results.py`
- `runs/notion_energy_prepare_v40_xi2/prepared_data/data_audit.json`
- `runs/notion_energy_prepare_v200_xi10/prepared_data/data_audit.json`
- `MODEL_FIDELITY_REVIEW.md`
