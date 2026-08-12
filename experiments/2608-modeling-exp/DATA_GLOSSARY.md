# 데이터 용어집

파이프라인 코드에 나오는 모든 데이터 용어의 뜻을 정리한다. 출처는 두 가지다 —
Google ClusterData 2019 trace 자체의 스키마와, 이 저장소가 전처리 과정에서 새로
만든 개념. **둘을 섞어 쓰면 코드가 안 읽히므로 구분해서 적는다.**

값은 전부 `data/raw/google2019_cell_a_day0_cpu_distribution/`의 실제 데이터에서
확인한 것이다. 추출 쿼리는 같은 디렉터리의 `sql/*.sql`에 있고, 그것이 각 컬럼의
정의에 대한 최종 근거다.

---

## 1. 이 데이터가 무엇인가

Google이 2019년에 공개한 **Borg 클러스터 운영 로그**다. 공개 클라우드의 VM trace가
아니라 내부 스케줄러 로그이며, 이 저장소는 여기서 VM-like toy 데이터를 만든다.
따라서 "VM"이라는 말은 전부 proxy이지 실제 VM이 아니다.

범위는 **cell `a`의 하루치(day 0)**다. cell은 클러스터 하나를 가리킨다.

시간은 전부 **trace 시작 이후 마이크로초(us)** 다. 절대 시각이 아니다.

| 값 | 의미 |
|---|---|
| `day_start_us = 600_000_000` | 관측 구간 시작. trace 시작 후 600초. 앞 600초는 워밍업이라 Google이 버리라고 안내한다 |
| `day_end_us = 87_000_000_000` | 관측 구간 끝. 시작으로부터 정확히 24시간(86,400초) 뒤 |

---

## 2. Google trace의 원래 용어

### 무엇이 하나의 "작업"인가

Borg의 작업 단위는 3층이다.

- **collection** — 작업 묶음 하나. 보통 job이라 부른다. `collection_id`로 식별.
- **instance** — collection 안의 개별 실행 단위. Borg 용어로 task.
  `instance_index`가 collection 안에서의 번호다.
- **alloc** — 자원을 미리 잡아두는 예약 컨테이너. instance가 alloc 안에서 돌 수도
  있다.

**`(collection_id, instance_index)` 한 쌍이 이 파이프라인이 다루는 하나의 실행
주체**이고, 이것이 나중에 "VM"이 된다.

추출 쿼리는 `alloc_collection_id IS NULL OR = 0` 조건으로 **alloc 안에서 도는
instance를 전부 제외**했다. 그래서 데이터의 `alloc_collection_id`는 항상 0이고
`alloc_instance_index`는 -1이다. 두 컬럼은 실질적으로 죽은 값이다.

`collection_type`은 `0 = JOB`, `1 = ALLOC_SET`. usage 테이블에는 0이 782,620행,
1이 85,286행 있다.

### 네 개의 원본 테이블

| 파일 | 행 수 | 한 행이 뜻하는 것 |
|---|---:|---|
| `usage_5min.parquet` | 867,906 | 한 instance의 **5분치 자원 사용량 측정** |
| `instance_events.parquet` | 141,047 | 한 instance의 **상태 변화 사건** |
| `collection_events.parquet` | 4,591 | collection 수준의 상태 변화 사건 |
| `machine_events.parquet` | 14,205 | 물리 머신의 추가/제거/변경 사건 |

### usage_5min 컬럼

5분 창(window) 하나에 대한 측정 결과다.

| 컬럼 | 뜻 |
|---|---|
| `start_time`, `end_time` | 측정 창의 시작·끝 (us). 보통 길이 300,000,000 = 5분 |
| `clipped_start_time`, `clipped_end_time` | 관측 구간 `[day_start, day_end]`로 잘라낸 창 |
| `overlap_us` | 잘라낸 창의 길이. 300,000,000이면 온전한 5분이고, 그보다 작으면 구간 경계에 걸친 조각이다. 전체 867,906행 중 721,839행만 온전한 5분이다 |
| `machine_id` | 이 창 동안 이 instance가 올라가 있던 물리 머신 |
| `cpu_usage`, `mem_usage` | 창 **평균** 사용량 |
| `max_cpu_usage`, `max_mem_usage` | BigQuery의 `maximum_usage`. **CPU 쪽은 이 파이프라인에서 쓰지 않는다** (아래 참조) |
| `cpu_usage_distribution` | 창 안에서의 CPU 사용량 분포. **길이 11의 배열**이고 낮은 것부터 정렬돼 있다. 즉 `[0]`이 최저, `[10]`이 최고(p100) |
| `assigned_memory` | 이 instance에 할당된 메모리 |
| `t5` | 전역 5분 슬롯 번호 = `start_time // 300_000_000` |
| `t5_day` | 관측 구간 안에서의 5분 슬롯 번호. **0~287** (288 = 24시간 / 5분) |

**CPU 최댓값을 왜 `cpu_usage_distribution[10]`에서 가져오는가.** BigQuery의
`maximum_usage.cpus`도 최댓값이지만 이 파이프라인은 그것을 쓰지 않고 분포 배열의
마지막 값(p100)을 쓴다. `max_cpu_usage`는 감사 목적으로만 남긴다. 메모리는
반대로 `max_mem_usage`를 그대로 쓴다 (메모리에는 분포 배열이 없다).

### 자원 단위

CPU와 메모리 값은 **Google이 정규화한 무차원 값**이다. 코어 개수나 GB가 아니다.
trace 전체에서 가장 큰 머신의 용량을 1로 두고 각 자원을 독립적으로 나눈 값이다.
그래서 `capacity_cpu = 1.0`인 머신이 실제로 존재한다.

이 파이프라인은 여기서 한 번 더 단위를 바꾼다 (4장 참조).

### instance_events 컬럼과 사건 종류

`event_type`은 정수 코드다. 관측된 분포는 다음과 같다.

| 코드 | 이름 | 뜻 | 관측 행 수 |
|---:|---|---|---:|
| 0 | SUBMIT | 제출됨 | 26,526 |
| 1 | QUEUE | 큐에서 대기 | 944 |
| 2 | ENABLE | 스케줄 가능 상태가 됨 | 26,517 |
| 3 | SCHEDULE | **머신에 배치되어 실행 시작** | 27,519 |
| 4 | EVICT | **선점당해 쫓겨남** | 9,154 |
| 5 | FAIL | 실패로 종료 | 2,206 |
| 6 | FINISH | 정상 종료 | 4,963 |
| 7 | KILL | 강제 종료 | 7,684 |
| 8 | LOST | 기록 유실 | 980 |
| 9 | UPDATE_PENDING | 대기 중 속성 변경 | 3,980 |
| 10 | UPDATE_RUNNING | **실행 중 속성 변경** | 30,574 |

이 중 **3(SCHEDULE)이 실행의 시작**이고 **4·5·6·7·8이 실행의 끝**이다. 10은
실행 중이라는 증거가 되므로 SCHEDULE이 누락됐을 때 방어적 시작점으로 쓴다.

| 컬럼 | 뜻 |
|---|---|
| `priority` | 우선순위 숫자. 관측값은 0, 25, 101, 103, 105, 118, 119, 200 |
| `scheduling_class` | 지연 민감도 0~3. 3이 가장 민감하다. **class 라벨에는 쓰지 않는다** |
| `resource_request_cpu/mem` | 그 시점에 요청한 자원량 |
| `missing_type` | 기록이 유실·합성된 이유. 141,047행 중 139,753행이 NULL |

**priority 계층.** Google 2019 문서의 구간은 free 0–99, best-effort batch 100–115,
mid 116–119, **production 120–359**, monitoring 360+ 이다. 이 파이프라인은
`priority >= 120`을 production으로 보고 `on_demand` proxy를 붙인다.

### collection_events

| 컬럼 | 뜻 |
|---|---|
| `scheduler` | 이 collection을 처리한 스케줄러. `SCHEDULER_DEFAULT` 3,081행, **`SCHEDULER_BATCH` 1,483행**, NULL 27행 |

`SCHEDULER_BATCH`가 batch 작업 판별의 근거다.

### machine_events

| 컬럼 | 뜻 |
|---|---|
| `event_type` | `1 = ADD` (머신 투입, 11,857행), `2 = REMOVE` (제거, 2,336행), `3 = UPDATE` (용량 변경, 12행) |
| `capacity_cpu`, `capacity_mem` | 그 머신의 정규화 용량 |

capacity 쌍은 14종이고 그중 하나는 NULL이라 **유효한 머신 형태는 13종**이다.
`(1.0, 1.0)`짜리 최대 머신부터 `(0.386719, 0.166748)`까지 섞여 있다.

---

## 3. 이 저장소가 새로 만든 개념

trace에는 없고 전처리가 도입한 용어다. 코드에서 이 단어가 보이면 Google 문서를
찾지 말고 여기를 봐야 한다.

### episode (실행 회차)

**하나의 instance가 하루 안에 여러 번 실행될 수 있다.** 배치됐다가(SCHEDULE)
쫓겨나고(EVICT) 다시 배치되는 일이 흔하다. 이 각각의 실행 회차를 episode라 부르고
`episode_index`를 0부터 붙인다.

그래서 모델이 보는 최종 단위는 `(collection_id, instance_index)`가 아니라
**`(collection_id, instance_index, episode_index)`** 다. 5,000개 instance에서
관측 구간과 겹치는 episode가 25,351개 나온다.

- **`HORIZON_CENSORED`** — 관측 구간이 끝날 때까지 종료 사건이 없어서 언제
  끝났는지 모르는 episode. 구간 끝까지 살아 있는 것으로 처리한다.
- **lifecycle event history** — 한 parent instance의 사건을 시간순으로 읽어
  실행 중인 episode의 시작과 종료를 재구성하는 사건 이력이다.
- **usage-inferred episode** — usable running event가 없는 parent instance를 usage
  관측만으로 보충한 episode. 이 데이터에서는 해당 episode가 0개라 파이프라인에
  구현하지 않는다.
- **`stable_episode_key`** — `collection=2:...|instance=2:...|episode=0` 형태의
  사람이 읽을 수 있는 식별자.
- **`vm_id`** — 위 키를 해시한 최종 VM 이름. `ep_acb94c06fbc70712b9c6` 꼴.

### 대표 머신 (representative machine)

머신 13종이 섞여 있으면 "서버 한 대"의 의미가 흔들린다. 그래서 **활성 시간이
가장 긴 (cpu, mem) 용량 쌍 하나를 골라 그것을 서버 1대로 정의**한다.

선정된 값은 `capacity_cpu = 0.591796875`, `capacity_mem = 0.33349609375`이고, 이
형태로 돌아간 머신 시간이 총 283,330,432,576,447 us다.

"joint shape"는 cpu와 mem을 **쌍으로 묶어서** 센다는 뜻이다. 각각 따로 최빈값을
구하면 실존하지 않는 머신 형태가 나올 수 있다.

### 단위 변환

모든 CPU 값을 `0.591796875`로, 모든 메모리 값을 `0.33349609375`로 나눈다. 결과의
`1.0`은 **대표 머신 한 대분**을 뜻한다. 그래서 `servers.csv`의 `C_cpu = C_mem = 1.0`
은 실제 대표 서버 한 대다.

이건 순수한 단위 환산이고, 목표 사용률을 맞추는 인위적 스케일링이 아니다.

### q_cpu, q_mem (설정 크기)

VM이 **예약한 크기**다. 실제 사용량이 아니다.

- `q_cpu` = max(도착 시점의 요청 CPU, 그 episode의 관측 CPU p100)
- `q_mem` = max(도착 시점의 요청 메모리, 관측 메모리 최댓값)

요청량과 실측 최댓값 중 큰 값을 쓰는 이유는, 요청보다 많이 쓴 VM을 요청량만큼만
잡으면 배치가 실제로 불가능해지기 때문이다.

단위 변환 후 `q`가 1을 넘는 VM(= 대표 머신 한 대에 안 들어감)은 값을 자르지 않고
**VM 전체를 제거**한다.

### 클래스 proxy 라벨

`on_demand` / `spot` / `batch_candidate`는 **이 저장소가 붙인 대리 라벨**이고
Google trace에서 관측된 실제 구매 유형이 아니다. 규칙은:

1. collection의 `scheduler == SCHEDULER_BATCH` → `batch_candidate` (최우선)
2. 아니고 `priority >= 120` → `on_demand`
3. 아니면 → `spot`

각 VM 행에 어느 규칙이 적용됐는지 `class_rule`로 남긴다.

### 시간 슬롯 두 종류

**헷갈리기 쉬운 지점이다.**

| 이름 | 길이 | 개수 | 쓰는 곳 |
|---|---|---:|---|
| `t5`, `t5_day` | 5분 | 288 | 원본 관측 데이터 |
| 모델 슬롯 `t` | 30분 | 48 | 최적화 모델 |

전처리가 5분 관측 6개를 묶어 30분 슬롯 하나로 집계한다. 집계는 단순 평균이 아니라
**관측 duration 가중평균**이다 — 위의 `overlap_us`가 온전한 5분이 아닌 조각들이
있기 때문이다.

### scenario (시나리오)

같은 VM 집합에 대한 **여러 개의 수요 실현값**이다. 확률적 최적화 모델의 입력이다.

- **scenario 0** — 관측값 그대로. 손대지 않는다.
- **scenario 1 이상** — CPU/메모리에 평균보존 lognormal 승수를 곱해 만든 합성값.
  곱한 뒤 각 자원을 `[0, q]`로 자른다.

canonical 데이터셋은 5개, 최근 실험 config는 10개를 쓴다.

### batch family

`batch_candidate` VM들은 개별 배치 대상이 아니라 **묶어서 시간당 작업량으로
추상화**한다. 최대 10개 family로 묶고, 각 family는 자원 크기 `q`, 사용률 `rho`,
총 작업량 `W`를 갖는다.

---

## 4. 값이 거쳐가는 순서

같은 숫자가 단계마다 다른 의미를 가지므로 한 번 정리한다.

```text
Google 정규화 단위          예: cpu_usage = 0.0278
   ↓  (대표 머신 용량으로 나눔)
대표 서버 대수 단위          예: 0.0278 / 0.591796875 = 0.0470  ← 모델이 쓰는 값
   ↓  (5분 6개를 duration 가중평균)
30분 슬롯 값
   ↓  (lognormal 승수, [0,q] 절단)
scenario별 값
```

"1.0"이라는 값이 1단계에서는 trace 최대 머신 한 대, 2단계 이후로는 대표 머신 한
대를 뜻한다. **같은 1.0이 아니다.**
