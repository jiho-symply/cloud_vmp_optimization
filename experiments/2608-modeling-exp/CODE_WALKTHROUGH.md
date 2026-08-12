# 코드 해설

코드를 옆에 띄워두고 함께 읽는 문서다. 무엇을 만들지는
[PIPELINE_SPEC.md](./PIPELINE_SPEC.md)에, 용어의 뜻은
[DATA_GLOSSARY.md](./DATA_GLOSSARY.md)에, 코드 스타일 규칙은
[AGENTS.md](./AGENTS.md)에 있다. 이 문서는 **실제로 짜인 코드를 읽는 방법**과
**어디를 의심해야 하는지**를 다룬다.

## 읽는 순서

처음 읽을 때는 단계 번호 순서대로 가면 된다. 다만 난이도가 고르지 않으니 시간을
어디에 쓸지 미리 알아두면 좋다.

| 단계 | 줄수 | 난이도 | 성격 |
|---|---:|---|---|
| s1 machine_capacity | 144 | 낮음 | 이벤트를 훑어 구간을 만드는 단순 상태 기계 |
| s2 episodes | 206 | 낮음 | 같은 패턴의 상태 기계 |
| s3 episode_usage | 339 | **가장 높음** | trace 결함 3종 + elementary-interval union |
| s4 select_episodes | 265 | **높음** | 4단계 필터 + sparse state 해석 |
| s5 vm_requests | 300 | 중간 | 집계와 라벨 붙이기, 분량이 많음 |
| s6 units | 102 | 낮음 | 나눗셈과 제거 |
| s7 scenarios | 121 | 낮음 | lognormal 변형과 시간 집계 |
| s8 spot_batch | 251 | 중간 | 선점 모델과 family 묶기 |
| s9 static | 116 | 낮음 | trace 무관 합성 입력 |

**s3와 s4에 시간의 대부분을 쓰게 된다.** 나머지 일곱 단계는 한 번 읽으면
넘어갈 수 있다.

## 데이터가 줄어드는 과정

전체를 한 장으로 보면 이렇다. 이 숫자들은 각 단계 스크립트가 실행할 때 직접
출력하므로, 코드를 고치면서 어디가 달라졌는지 바로 확인할 수 있다.

```text
usage 원본 행                                        867,906
  └ 완전 중복 2행 제거                     ────────► 867,904   (s3)

horizon과 겹치는 episode                              25,351   (s2)
  └ usage가 붙은 episode                   ────────►  24,883   (s3)
      └ 품질 필터 통과                     ────────►  20,065   (s4, 4,818개 제거)
          └ 5분 bucket 12개 이상            ────────►   5,471   (s4)
              └ 상태 모호성 제외            ────────►   5,414   (s4, 57개 제거)
                  └ hash 정렬 상위 5,000    ────────►   5,000   (s4)
                      └ q>1 제거            ────────►   5,000   (s6, 0개 제거)
```

**가장 많이 버리는 곳은 품질 필터(19.4%)이고, 가장 많이 남기는 곳은 hash
표본추출이다.** 5,414개 중 5,000개를 고르는 것이므로 hash 단계는 사실상 414개만
잘라낸다. 나중에 무엇을 들어낼지 고민할 때 이 비율이 판단 근거가 된다.

## 단계별 해설

### s1_machine_capacity — "서버 1대"를 정의한다

머신 13종이 섞인 클러스터에서 대표 머신 하나를 골라, 그것을 이후 모든 자원
숫자의 1.0으로 삼는다. 이 결정이 뒤의 모든 숫자의 의미를 정하므로 맨 앞에 있다.

읽는 순서는 `main()` → `_machine_cpu_capacity_intervals()` →
`_apply_machine_event()` → `_joint_shape_weights()` →
`_representative_capacity()`다.

핵심은 `_machine_cpu_capacity_intervals()`의 while 루프 두 개다. 첫 루프는
`day_start_us` 이전 이벤트를 모두 소화해 **관측 시작 시점의 머신 상태를
복원**한다. 두 번째 루프가 그 상태에서 출발해 이벤트마다 구간을 끊는다. 머신
이벤트는 상태 변화만 기록하는 sparse 형식이라 이 복원 과정이 필요하다.

`_apply_machine_event()`에서 REMOVE(코드 2)일 때 용량을 `None`으로 되돌리는
부분에 주석이 붙어 있다. 이걸 빼면 나중에 같은 머신이 다시 ADD될 때 제거된
머신의 용량을 물려받는다.

`_representative_capacity()`의 정렬 기준 세 개는 순서가 결과를 바꾼다. active
duration 내림차순이 본 기준이고, cpu·mem 오름차순은 동점 처리다.

**의심할 것 없음.** 이 단계는 원본과 대조했을 때 중복 구현을 하나로 합친 것
외에 군더더기가 없다. 원본은 같은 구간 재구성 루프를 두 곳에
(`_machine_cpu_capacity_intervals` 2248줄, `derive_representative_machine_capacity`
3256줄) 갖고 있었다.

### s2_episodes — "한 행"을 정의한다

instance 하나가 하루에 여러 번 실행되므로, 실행 회차(episode)가 모델의 행이
된다. 이 단계는 사건 이력만 보고 회차의 시작과 끝을 재구성한다.

`_episodes_for_key()`가 본체다. 사건을 시각순으로 훑으며 `running` 상태를
갱신하는데, 같은 시각에 여러 사건이 있을 수 있어서 timestamp 단위로 묶어
처리한다(cohort). 주석 두 개가 붙은 곳이 판단이 갈리는 지점이다.

- **같은 시각에 종료 사건이 여러 개**일 때 lifecycle precedence가 가장 높은 것을
  고른다. `EVICT=0 … LOST=4` 순서에서 `[-1]`을 취하므로 LOST가 EVICT를 이긴다.
  원본(`build_toy_instance.py` 2407–2416줄)과 같은 규칙이다.
- **SCHEDULE이 없을 때 UPDATE_RUNNING을 시작으로 삼는다.** UPDATE_RUNNING은
  "실행 중 속성 변경"이므로 실행 중이라는 증거가 되고, SCHEDULE 기록이 누락된
  경우의 방어적 시작점이다.

`episode_start_inferred_from_usage` 컬럼은 **항상 False**다. 원본에는 usage
관측만으로 episode를 유추하는 경로가 있는데, 이 데이터에서는 5,000개 parent
instance 전부가 SCHEDULE 또는 UPDATE_RUNNING을 갖고 있어서 그 경로가 실행되지
않는다. 컬럼은 canonical 출력에 있어야 하므로 상수로 남겼다.

### s3_episode_usage — 측정값을 붙인다 (가장 어려움)

**여기가 이 파이프라인의 핵심이자 가장 지저분한 곳이다.** Google trace의 결함
세 가지를 모두 여기서 처리한다. 각각 독립 함수로 분리해뒀으니 하나씩 지워보며
출력이 어떻게 변하는지 확인할 수 있다.

**결함 1 — 완전 중복 행.** `_prepare_usage_source()`. 똑같은 usage 행이 2개
있다. 그냥 `duplicated()`를 쓰면 안 되는데, `cpu_usage_distribution`이 배열
필드라서 비교가 안 된다. 그래서 튜플로 바꿔서 비교한다. **867,906 → 867,904.**

**결함 2 — 겹치는 관측 구간.** `_elementary_interval_union()`과
`_aggregate_overlap_bucket()`. 같은 VM의 5분 창이 서로 겹치는 경우가 있다
(최대 3중). 단순 합산하면 시간이 중복 계산되므로, 모든 구간 경계를 모아
elementary interval로 쪼갠 뒤 각 조각마다:

- 겹친 시간은 **한 번만** 센다 (`coverage_us`)
- 동시 관측값은 **평균**낸다 (같은 논리적 VM을 두 번 기록한 것이므로)
- 중복된 시간량을 `duration × (concurrency − 1)`로 누적한다

**23개 bucket이 영향받고, 1,760행이 겹치며, 중복 시간은 9,979,126,571 us다.**

**결함 3 — episode 경계를 가로지르는 창.** `_split_usage_at_episode_boundaries()`.
5분 창이 episode의 시작이나 끝을 넘어가면 잘라야 한다. **7,155행이 분할되고,
어떤 running episode와도 겹치지 않는 27,536행은 버려진다.**

이 단계에서 품질 플래그 `cpu_complete`(CPU 분포가 모든 조각에 있는지)와
`capacity_complete`(배치된 머신의 양의 CPU 용량이 조각 전체를 덮는지)를
계산하지만, **행을 버리지는 않는다.** 버리는 일은 s4가 한다. 원본은 이 두
관심사를 한 함수(`_prepare_strict_episode_usage`)에 섞어놨는데, 그러면 품질
필터만 들어내고 결과를 볼 수 없어서 분리했다.

> **의심할 곳 하나.** `_normalize_usage()`가 `_elementary_interval_union()`을
> **두 번** 호출한다. 첫 호출은 실제 출력을 만들고, 두 번째(`quality_buckets`)는
> 품질 통과 부분집합에 대해 다시 돌려서 검증 수치(중복 group 23개, overlap
> 1,760행 등)만 뽑아낸다. canonical의 진단 수치가 원본에서는 품질 필터 **이후**
> 모집단에 대해 계산됐기 때문에 생긴 군더더기다. 즉 s3가 행을 버리지 않게 만든
> 내 설계 결정의 대가다. **정합성 요구를 풀면 두 번째 호출과 그에 딸린 `attrs`
> 배관을 전부 지울 수 있다.** 출력 데이터는 한 글자도 바뀌지 않는다.

### s4_select_episodes — 24,883개에서 5,000개를 고른다

네 단계 필터를 순서대로 적용한다. **순서가 결과를 바꾸므로** 바꾸면 안 된다.
`_select_episodes()`에 주석으로 각 순서의 이유가 붙어 있다.

1. **품질 필터** — s3가 남긴 플래그로 episode 단위 제거. 20,065개 유지
2. **최소 관측 길이** — 재구성된 5분 bucket 12개 미만 제거. 5,471개 남음
3. **상태 모호성** — 57개 제거
4. **결정론적 표본추출** — `blake2b` hash 정렬 후 상위 5,000개

3번이 이 파일에서 가장 읽기 어려운 부분이다. `_resolve_sparse_state_at_time()`과
`_resolve_cohort()`가 하는 일은 이렇다. 사건 기록이 sparse해서 "도착 시점의
priority와 요청 자원이 무엇이었나"를 알려면 그 시점까지의 사건을 누적해야 한다.
그런데 같은 시각에 서로 다른 값을 주장하는 사건이 여러 개 있을 수 있고, 그때
어느 값이 맞는지 정할 근거가 없다. **임의로 하나를 고르는 대신 episode 전체를
버린다.** 그것이 이 필터의 존재 이유다.

`_value_key()`는 타입이 섞인 값들을 결정론적으로 비교·정렬하기 위한 문자열
변환이다. `0:<NULL>`, `1:bool`, `2:숫자`, `3:문자열` 접두사로 타입 순서를 강제한다.

> **읽기 어려운 곳.** `_resolve_cohort()`는 튜플을 위치 인덱스로 다룬다
> (`record[5][index]`, `row[6:]`, `item[0][6]`). 무엇이 몇 번째인지 위쪽
> `columns` 리스트와 맞춰봐야 알 수 있어서, 이 파이프라인에서 가장 읽기 힘든
> 코드다. `NamedTuple`이나 dataclass로 바꾸면 훨씬 나아진다. 다만 이 함수 자체가
> 삭제 후보이기도 하므로(아래 참조) 고치기 전에 지울지 먼저 결정하는 편이 낫다.

### s5_vm_requests — episode를 VM 한 행으로 요약한다

분량은 많지만(300줄) 하는 일은 단순하다. 집계하고 라벨을 붙인다.

읽는 순서는 `_aggregate_requests()` → `_apply_resource_requests()` →
`_finish_requests()` → `_add_stable_ids()`다.

`_apply_resource_requests()`가 `q_cpu`, `q_mem`을 정하는 곳이다. **요청량과 관측
최댓값 중 큰 값**을 쓴다. 요청보다 많이 쓴 VM을 요청량만큼만 잡으면 배치가 실제로
불가능해지기 때문이다.

클래스 proxy 라벨은 `_finish_requests()`에서 붙는다. 규칙은
`SCHEDULER_BATCH → batch_candidate` (최우선), `priority ≥ 120 → on_demand`,
나머지 `→ spot`. 이건 **이 저장소가 붙인 대리 라벨이고 실제 구매 유형이
아니다.** 각 행에 `class_rule` 문자열로 어느 규칙이 적용됐는지 남는다.

`max_mem_usage_fallback_from_average` 7건이 유일하게 발동하는 fallback이다. 나머지
fallback 카운터는 모두 0이므로, 그 처리 경로들은 이 데이터에서 실행되지 않는다.

> **중복 관심사.** s4와 s5가 **둘 다** sparse state를 해석한다. s4는 "모호한가"만
> 판정하고, s5는 실제 값을 꺼낸다. 같은 로직이 두 파일에 나뉘어 있는 셈이다.
> s4의 모호성 필터를 지우기로 결정하면 s5의 `_resolve_state()`만 남으므로 이
> 중복도 함께 사라진다.

### s6_units — 모델 단위로 바꾼다

102줄이고 어려운 것이 없다. 모든 CPU를 `0.591796875`로, 모든 메모리를
`0.33349609375`로 나눈다. 그 결과 `1.0`은 대표 서버 한 대분을 뜻한다.

`_filter_capacity()`가 `q_cpu` 또는 `q_mem`이 1을 넘는 VM을 **자르지 않고 통째로
제거**한다. 이 데이터에서는 **0개가 제거된다.** 즉 이 필터는 실제로 아무 일도
하지 않는다. 안전장치로서의 가치는 있으나, 지워도 지금 출력은 바뀌지 않는다.

### s7_scenarios — 수요 실현값을 만든다

121줄. `scenario 0`은 관측값 그대로 복사하고, `scenario 1–4`는 CPU/메모리에
평균보존 lognormal 승수를 곱한 뒤 각 자원을 `[0, q]`로 자른다.

`_hourly_frame()`의 집계가 단순 평균이 **아니라** 관측 duration 가중평균인 것에
주의. `overlap_us`가 온전한 5분이 아닌 조각들이 있기 때문이다(867,906행 중
721,839행만 온전한 5분).

원본은 같은 466MB 내용을 `vm_usage_scenarios.csv`로 한 번 더 쓴다. 재현하지
않았다.

### s8_spot_batch — 선점과 batch 추상화

`build_spot_preemption_scenarios()`는 spot VM이 언제 쫓겨나는지를 시나리오별로
만든다. 실제 `EVICT` 사건이 있으면 그 시점을 쓰고, 없으면 low priority VM에 더
높은 synthetic hazard를 준다. 선점 이후는 관측 구간 끝까지 비활성이다.

`build_batch_outputs()`는 `batch_candidate` VM들을 개별 배치 대상으로 두지 않고
**최대 10개 family의 시간별 작업량으로 추상화**한다. 개별 VM 배치 결정을 모델에서
빼서 문제 크기를 줄이는 것이 목적이다.

> **의심할 곳.** `_is_evict_event()`가 숫자 4와 문자열 `EVICT`를 모두 받는데,
> **문자열 분기는 죽은 코드다.** `instance_events.event_type`은 `Int64`이고
> 문자열 값이 하나도 없음을 확인했다. `pd.isna` 분기도 null이 0개라 마찬가지다.
> 함수 전체가 `value == 4` 한 줄로 줄어든다. s2에서 같은 함수를 같은 이유로
> 지웠는데 s8에 다시 들어왔다.

### s9_static — trace와 무관한 합성 입력

116줄. 서버 6대(`C_cpu = C_mem = 1.0`, `E_idle = 0.35`, `E_cpu = 0.65`), 24시간
전기요금·renewable, 균등 시나리오 확률, 모델 파라미터(`alpha 0.95`,
`epsilon 0.05`, `soc_init 0.5`, ESS 효율 0.92)를 만든다.

전부 합성값이라 trace를 이해하는 것과 무관하다. **가장 먼저 읽고 잊어도 되는
단계다.**

## 정합성 요구를 풀 때의 삭제 후보

현재는 canonical과 바이트 단위로 일치한다. 그 요구를 풀기로 하면 아래 순서로
검토하는 것을 권한다. 위쪽이 이득 대비 위험이 낮다.

| 후보 | 위치 | 출력 변화 | 근거 |
|---|---|---|---|
| 두 번째 union 호출과 `attrs` 배관 | s3 `_normalize_usage` | **없음** | 검증 수치 재현용 군더더기 |
| `q>1` 제거 필터 | s6 `_filter_capacity` | **없음** | 제거 대상 0개 |
| `_is_evict_event` 문자열·null 분기 | s8 | **없음** | dtype 확인 완료, 실행 불가 |
| 상태 모호성 필터 | s4 3번 + `_resolve_*` | 57개 episode 추가 | 가장 복잡한 코드가 사라진다 |
| elementary-interval union | s3 결함 2 | 23개 bucket의 값 변화 | 겹침이 23개 bucket에만 영향 |
| 품질 필터 | s4 1번 | 4,818개 episode 추가 | 가장 크게 바뀐다 |

**상태 모호성 필터가 비용 대비 효과가 가장 좋은 후보다.** 5,471개 중 57개
(1.0%)에만 영향을 주는데, 그 대가로 이 파이프라인에서 가장 읽기 어려운 코드
(`_resolve_sparse_state_at_time`, `_resolve_cohort`, `_value_key`)를 전부
안고 있다. 이걸 지우면 s4가 절반 이하로 줄고 s5의 중복도 정리된다.

반대로 **품질 필터는 신중해야 한다.** 4,818개를 되살리면 배치된 머신의 CPU 용량이
관측 구간을 덮지 못하는 episode가 섞여 들어온다. 자원 숫자의 신뢰도와 직결된다.

## 고치고 나서 확인하는 방법

로직을 지우거나 바꾼 뒤에는 두 가지를 본다.

```bash
.venv/bin/python experiments/2608-modeling-exp/run_all.py
```

각 단계가 자기 검증 수치를 출력하므로, **어느 단계에서 숫자가 달라졌는지 바로
보인다.** 어느 디렉터리에서 실행해도 동작한다.

```bash
.venv/bin/python experiments/2608-modeling-exp/verify_against_canonical.py
```

10개 산출물을 canonical과 파일별로 비교한다. 파이프라인과 분리된 스크립트이므로
파이프라인을 고쳐도 검증 기준은 흔들리지 않는다. canonical에만 있고 의도적으로
만들지 않는 파일 5개는 비교에서 제외되며 그 사실이 출력에 표시된다.

단계 하나만 다시 돌리고 싶으면 그 스크립트를 직접 실행하면 된다. 앞 단계 산출물이
`work/`에 남아 있으므로 처음부터 다시 돌릴 필요가 없다.

```bash
.venv/bin/python experiments/2608-modeling-exp/s3_episode_usage.py
```

전체 완주는 약 3분이다.
