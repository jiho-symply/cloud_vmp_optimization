# 전처리 파이프라인 명세

`experiments/2607-notion-energy-vmp/src/google2019_toy/build_toy_instance.py`
(이하 **원본**)의 전처리를 9개 단계로 분해한 설계다. 스타일 계약은
[AGENTS.md](./AGENTS.md)에, 데이터 용어의 뜻은
[DATA_GLOSSARY.md](./DATA_GLOSSARY.md)에, 완성된 코드를 읽는 방법은
[CODE_WALKTHROUGH.md](./CODE_WALKTHROUGH.md)에 있고, 이 문서는 **무엇을 만들지**만
정의한다.

원본은 각 단계의 의미론에 대한 유일한 진실 공급원(source of truth)이다. 아래
표의 "원본 위치"를 읽고 그 안의 **결과 숫자를 결정하는 로직만** 옮긴다. 감사·
진단 코드는 옮기지 않는다(AGENTS.md 금지 규칙 1·2).

## 전체 흐름과 순서의 이유

단계 번호는 임의로 붙인 것이 아니라 **의존 관계가 강제하는 순서**다. 각 단계가
왜 그 자리에 있는지는 다음과 같다.

```text
              machine_events ──► s1  대표 머신 용량을 정한다
                                  │   = "서버 1대"의 정의
            instance_events ──► s2  실행 회차(episode)를 만든다
                                  │   = 앞으로 한 "행"이 무엇인지 정의
      usage_5min + s1 + s2 ──► s3  회차별 5분 사용량을 붙이고 정리한다
                                  │   = 측정값 확보
              s3 + 이벤트 ──► s4  25,351개 회차에서 5,000개를 고른다
                                  │   = 대상 확정
              s4 + 이벤트 ──► s5  회차 하나를 VM 한 행으로 요약한다 (q, class)
                                  │
                    s5 + s1 ──► s6  대표 머신 단위로 환산하고 q>1 VM을 버린다
                                  │   = 모델이 쓰는 단위로 진입
                         s6 ──► s7  workload scenario를 만든다
                         s6 ──► s8  spot 선점 / batch family를 만든다
                     (없음) ──► s9  합성 입력 (서버·전기요금·ESS)
```

읽는 순서로 풀면 이렇다. **먼저 "1대"가 무엇인지 정하지 않으면**(s1) 그 뒤의 모든
자원 숫자가 의미를 갖지 못한다. **다음으로 "한 행"이 무엇인지 정한다**(s2) —
instance 하나가 하루에 여러 번 실행되므로 실행 회차가 행이 된다. 행이 정해져야
측정값을 붙일 수 있고(s3), 측정값이 있어야 품질 기준으로 걸러낼 수 있다(s4).
걸러낸 뒤에야 VM 한 줄로 요약하는 것이 의미가 있고(s5), 그제서야 단위 환산과
용량 초과 제거를 한다(s6). s7·s8은 확정된 VM 집합 위에서 갈라지는 가지이고,
s9는 trace와 무관해서 아무 때나 돌아간다.

s1과 s2는 서로 독립이라 순서를 바꿔도 되지만, s1이 **단위 정의**라는 가장 근본적인
결정이라 앞에 뒀다.

## 디렉터리 배치

```text
experiments/2608-modeling-exp/
  preprocessing/                    전처리 구현 (패키지)
    s1_machine_capacity.py .. s9_static.py   단계별 모듈
    paths.py                        경로 상수만 (로직 없음)
    raw_tables.py                   raw parquet 로더 (단계 아님)
  build_dataset.py                  단일 진입점 (얇은 드라이버)
  verify_against_canonical.py       기존 데이터셋과 비교 (파이프라인과 분리)
  work/                             중간 산출물 (git 미추적)
  out/                              최종 산출물 (git 미추적)
```

전처리 구현을 `preprocessing/` 안에 모으는 이유는 이 디렉터리에 최적화 모델
코드가 함께 들어오기 때문이다. 밖에서는 `build_dataset.py` 하나만 부르면 되고,
최적화 코드는 `out/`의 산출물만 소비한다.

```bash
.venv/bin/python experiments/2608-modeling-exp/build_dataset.py
.venv/bin/python experiments/2608-modeling-exp/build_dataset.py --only s3
```

입력은 `data/raw/google2019_cell_a_day0_cpu_distribution/`,
검증 대상은 `data/processed/notion_toy_google2019_v5_mode_capacity_bounded/`
(이하 **canonical**)이다.

## 고정 상수

원본은 인자로 받지만 실제로 한 값만 쓰인다. 분기를 만들지 말고 상수로 박는다.

| 상수 | 값 | 출처 |
|---|---|---|
| `day_start_us` | 600000000 | raw `metadata.json` |
| `day_end_us` | 87000000000 | raw `metadata.json` |
| `max_instances` | 5000 | 추출 시 이미 적용됨 |
| `seed` | 42 | |
| `num_scenarios` | 5 | |
| `num_servers` | 6 | |
| `min_usage_rows` | 12 | |
| `slot` | 5분 = 300000000 us | `t5` 단위 |

## 단계 정의

### raw_tables (단계 아님)

**초판에서 이 자리에 `s1_load_raw` 단계를 두었으나 철회한다.** 실측 결과 raw
parquet 4개는 이미 필요한 컬럼을 정확히 그만큼 갖고 있고 dtype도 이후 단계가
기대하는 것과 같다. 컬럼 선택과 `astype`이 전부 no-op이었고, 남는 것은 26MB를
`work/`에 다시 쓰는 복사뿐이었다. AGENTS.md의 기준("지우면 출력 숫자가 바뀌는가")
에 따라 단계 자체를 없앤다.

대신 `raw_tables.py`에 로더 함수만 둔다. 각 함수는 `pd.read_parquet` 한 줄이고,
컬럼 선택도 dtype 캐스팅도 하지 않는다. 존재 이유는 입력 경로를 한 곳에 모으는
것뿐이다. 이후 단계는 이 모듈에서 직접 읽는다.

- 검증: usage 867906행, instance_events 141047행, machine_events 14205행,
  collection_events 4591행

### s1_machine_capacity

machine_events에서 대표 머신 용량을 뽑는다. 방법은 **joint (cpu, mem) shape의
active-duration 가중 최빈값**이고, 동점은 capacity_cpu 최소 → capacity_mem 최소.

- 원본 위치: `derive_representative_machine_capacity` (3256–3409),
  `_machine_cpu_capacity_intervals` (2248–2330), `_machine_event_code` (2155)
- 출력: `work/s1_representative_capacity.json`,
  `work/s1_machine_capacity_intervals.parquet`
- 검증: `capacity_cpu = 0.591796875`, `capacity_mem = 0.33349609375`,
  joint shape 13개, 선택된 shape 가중치 `283330432576447`,
  `total_active_machine_duration_us = 820416271910336`

### s2_episodes

instance_events에서 lifecycle episode를 만든다. `SCHEDULE`이 시작이고 없으면
`UPDATE_RUNNING`을 방어적 시작으로 쓴다. `EVICT/FAIL/FINISH/KILL/LOST`가 종료.
terminal event가 없으면 `day_end_us`까지 활성이며 `HORIZON_CENSORED`로 표시한다.
모델이 보는 단위는 `(collection_id, instance_index, episode_index)`다.

- 원본 위치: `_build_lifecycle_episodes` (2331–2500), `_instance_event_name` (226),
  `_instance_event_precedence` (243), `_is_evict_event` (217)
- 출력: `work/s2_episodes.parquet`
- 검증: horizon과 겹치는 episode 25351개, parent instance 5000개,
  usage-inferred episode 0개

### s3_episode_usage

usage 5분 행을 episode에 붙이고 정규화한다. 세 가지 trace 결함 대응이 들어간다:
exact duplicate 행 제거, 겹치는 interval의 elementary-interval union(중복 시간을
한 번만 세고 동시 관측값은 resource별 평균), episode 경계에서의 행 분할. CPU
maximum은 `cpu_usage_distribution[10]`(p100)이며 `maximum_usage.cpus`는 쓰지
않는다.

**이 단계가 가장 어렵고, 저자가 나중에 가장 많이 들여다볼 곳이다.** 결함 대응
로직 세 개를 각각 독립된 함수로 분리하고, 각 함수 위에 대응 대상 결함을 한 줄로
적는다.

**이 단계는 episode를 버리지 않는다.** 원본 `_prepare_strict_episode_usage`는
분할과 품질 게이트를 한 함수에서 같이 하지만, 여기서는 분리한다. s3는 품질
판정에 필요한 플래그(fragment별 CPU distribution 완전성, assigned machine의
positive CPU capacity 커버리지)를 **계산해서 컬럼으로 남기기만** 하고, 실제로
버리는 것은 s4가 한다. 저자가 품질 필터를 통째로 들어내고 결과를 보려면 두
관심사가 분리돼 있어야 한다. 따라서 s3 출력의 episode 수는 필터 이전 값인
24,883개다.

- 원본 위치: `_normalize_usage` (334–848), `_prepare_strict_episode_usage`
  (2501–2803), `_exact_usage_duplicate_mask` (292), `_cpu_distribution_max` (2195)
- 출력: `work/s3_episode_usage.parquet`
- 검증: exact duplicate 2행 제거 → 867904행, duplicate interval group 23개,
  overlap 행 1760개, 최대 동시 행 3개, 중복 timeline `9979126571` us,
  running episode와 겹치지 않는 행 27536개, 경계에서 분할된 행 7155개,
  usage를 가진 episode 24883개

### s4_select_episodes

episode를 걸러 최종 5000개를 고른다. 순서가 결과를 바꾸므로 그대로 지킨다.

1. quality filter — assigned machine의 positive CPU capacity가 fragment 전체를
   덮지 못하거나, CPU distribution이 한 번이라도 비어 있거나, machine history가
   없으면 episode 전체를 버린다
2. 재구성된 usage bucket이 12개 미만이면 버린다
3. arrival/scheduler state가 모호한 후보를 버린다
4. `(collection_id, instance_index)` 결정론적 hash 정렬 후 상위 5000개

- 원본 위치: `_prepare_strict_episode_usage`의 필터 부분, `_stable_key_hash` (187),
  `_resolve_sparse_state_at_time` (1016–1267), `_instance_states_at_arrival` (1281),
  `_collection_states_at_arrival` (1328)
- 출력: `work/s4_selected_episodes.parquet`
- 검증: quality filter 후 20065개 유지 / 4818개 제거(제거율
  `0.19362617047783628`), event-state 필터 대상 후보 5471개 중 모호 57개 제외,
  최종 선택 5000개

### s5_vm_requests

선택된 episode를 모델이 보는 VM 요청 행으로 만든다. `q_cpu`는 arrival request와
episode p100 maximum의 큰 값, `q_mem`은 arrival request와 source memory maximum의
큰 값. class는 proxy label이다: `priority >= 120` → `on_demand`,
`priority < 120` → `spot`, `collection_events.scheduler == SCHEDULER_BATCH`면
`batch_candidate`가 우선.

- 원본 위치: `build_vm_requests` (1362–1851), `_normalize_collection_events` (926),
  `_normalize_scheduler` (998), `_stable_episode_key` (205)
- 출력: `work/s5_vm_requests_raw.parquet` (raw trace 단위)
- 검증: canonical `vm_requests.csv`의 49개 컬럼 중 단위 변환 전 값들과 일치.
  `max_mem_usage_fallback_from_average` 7건, 나머지 fallback 카운트 0

### s6_units

대표 머신 용량으로 나눠 representative-server fraction으로 바꾸고, `q_cpu` 또는
`q_mem`이 1을 넘는 VM은 **자르지 않고 VM 전체를 제거**한다.

- 원본 위치: `convert_resource_units` (3410–3501),
  `drop_vms_exceeding_representative_capacity` (3502–3598)
- 출력: `out/vm_requests.csv`, `work/s6_observed_usage.parquet`
- 검증: cpu 나눗수 `0.591796875`, mem 나눗수 `0.33349609375`,
  5000개 → 5000개 유지 / 0개 제거,
  변환 후 request cpu p50 `0.011641398514851485`, mem p50 `0.019857247437774523`

### s7_scenarios

workload scenario 5개. scenario 0은 관측값 그대로, 1–4는 CPU/MEM에 mean-preserving
lognormal 승수를 곱한 뒤 각 resource를 `[0, q]`로 제한한다. hourly는 관측 duration
가중평균으로 집계한다.

- 원본 위치: `build_usage_scenarios` (1852–1901),
  `build_hourly_usage_scenarios` (1902–1936)
- 출력: `out/vm_usage_5min_scenarios.csv`, `out/vm_usage_hourly_scenarios.csv`
- 검증: scenario 0이 s6 관측값과 완전 일치, 모든 usage ≤ 해당 `q`

원본은 같은 내용을 `vm_usage_scenarios.csv`로 한 번 더 쓴다(466MB 중복). 이건
재현하지 않는다.

### s8_spot_batch

spot preemption scenario와 batch family/workload를 만든다. preemption은
`instance_events.type == EVICT`(numeric enum 4 또는 문자열)를 쓰고, 실제 EVICT가
없으면 low priority VM에 더 높은 synthetic hazard를 준다. preempted 이후는 horizon
끝까지 inactive. batch는 `batch_candidate` VM들을 최대 10개 family의 hourly
workload volume으로 추상화한다.

- 원본 위치: `build_spot_preemption_scenarios` (2010–2065), `_eviction_t5_by_vm`
  (1963–2009), `build_batch_outputs` (2066–2154)
- 출력: `out/spot_preemption_scenarios.csv`, `out/batch_families.csv`,
  `out/batch_workload.csv`

### s9_static

trace와 무관한 합성 입력. 전부 짧다.

- 원본 위치: `build_servers` (3599), `build_energy_scenarios` (3618),
  `build_scenario_probabilities` (1937), `build_model_params` (1947)
- 출력: `out/servers.csv`, `out/energy_scenarios.csv`,
  `out/scenario_probabilities.csv`, `out/model_params.json`
- 검증: servers 6행 전부 `C_cpu = C_mem = 1.0`, `E_idle = 0.35`, `E_cpu = 0.65`;
  model_params `alpha 0.95, epsilon 0.05, soc_init 0.5, ess 효율 0.92/0.92`

## 재현하지 않는 것

원본에서 아래는 결과 숫자에 영향이 없다. 옮기지 않는다.

| 원본 | 줄 수 | 성격 |
|---|---:|---|
| `audit_assigned_machine_cpu_capacity` | 393 | 감사 전용 (`status = audit_only_no_cap_drop_or_rescaling_applied`) |
| `_write_scaling_diagnostics` | 104 | 진단 JSON |
| `_write_preprocessing_diagnostics` | 55 | 진단 JSON |
| `_write_metadata` | 81 | 진단 JSON |
| `_episode_stage_summary`, `_numeric_summary`, `_resource_summary`, `_peak_30min_load`, `_cpu_p100_audit` | ~180 | 진단 집계 |
| `vm_usage_scenarios.csv` 쓰기 | 1 | 466MB 중복 파일 |
| `strict_event_state=False` 경로 | — | 실제로 쓰이지 않는 분기 |

위 표의 수치들은 canonical의 `scaling_diagnostics.json`에서 읽은 것이며, 이
문서의 검증 기준으로만 쓴다. 파이프라인이 그 JSON을 다시 만들지는 않는다.
