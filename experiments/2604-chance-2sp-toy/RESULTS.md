# 실험 결과

## 현재 기준
이 문서는 `2604-chance-2sp-toy` 폴더의 최신 수치 결과를 한 곳에 모아 둔 문서입니다.

현재 최신 벤치마크 설정은 다음과 같습니다.
- 전체 VM 수: `24`
- 서버 용량: `8`
- 후보 VM 필터: `vCPU <= 8`, `avg_cpu_mean >= 20`
- 시나리오 수: `10`
- 병렬 실행: 최대 `4`개 작업, 각 작업 `threads = 8`
- time limit: `7200초`
- migration penalty 비교: `lambda = 0.1` 과 `lambda = 0.0`

핵심 출력 파일:
- 전체 요약 표: `results/analysis/chance_2sp_toy_24vm_cap8_avg20_lambda_queue/all_jobs_summary.csv`
- migration 비교 표: `results/analysis/chance_2sp_toy_24vm_cap8_avg20_lambda_queue/lambda_migration_comparison.csv`
- lambda 비교 그림: `results/analysis/chance_2sp_toy_24vm_cap8_avg20_lambda_queue/lambda_migration_comparison.png`

대표 그림:
- 조합 비교 (`lambda = 0.0`): `results/analysis/chance_2sp_toy_24vm_cap8_avg20_lambda_queue/combination_lam000.png`
- 조합 비교 (`lambda = 0.1`): `results/analysis/chance_2sp_toy_24vm_cap8_avg20_lambda_queue/combination_lam010.png`
- 비율 비교 (`lambda = 0.0`): `results/analysis/chance_2sp_toy_24vm_cap8_avg20_lambda_queue/ratio_lam000.png`
- 비율 비교 (`lambda = 0.1`): `results/analysis/chance_2sp_toy_24vm_cap8_avg20_lambda_queue/ratio_lam010.png`
- migration 화살표 예시: `results/runs/chance_2sp_toy_24vm_combination_od_only_od24_sp0_bj0_sc10_cap8_avg20_lam000/server_workload_gantt_scen_01.png`
- suspended / clean 시나리오 예시: `results/runs/chance_2sp_toy_24vm_combination_od_sp_od12_sp12_bj0_sc10_cap8_avg20_lam010/server_workload_gantt_suspended.png`
- suspended / clean 시나리오 예시: `results/runs/chance_2sp_toy_24vm_combination_od_sp_od12_sp12_bj0_sc10_cap8_avg20_lam010/server_workload_gantt_clean.png`

## 읽는 법
- `status`
  `OPTIMAL`은 최적해가 증명된 경우입니다. `TIME_LIMIT`은 2시간 안에 찾은 incumbent feasible solution입니다.
- `used servers`
  하루 동안 한 번이라도 켜진 서버 수입니다.
- `actual migrations`
  실제 placement가 바뀐 migration event 수입니다. 현재는 이 값을 migration 해석의 기준으로 봅니다.
- `migration_count`
  모델 내부 migration 변수 합입니다. 일부 케이스에서는 실제 이벤트 수와 차이가 있으므로, 비교 해석에는 `actual migrations`를 우선합니다.
- `peak util`
  시나리오별 realized load를 서버 용량으로 나눈 값의 최대치입니다.
- `peak overbooking ratio`
  예약 기준 용량 대비 실제 수요가 얼마나 공격적으로 쌓였는지 보여 주는 지표입니다.
- `spot susp prob`
  해당 케이스의 spot workload 중 가장 큰 suspension probability입니다.
- `completion ratio`
  해당 케이스의 spot workload 중 가장 작은 시나리오별 completion ratio입니다.

## 해석 주의
이번 24-VM benchmark는 대부분 `TIME_LIMIT`에 걸렸습니다. 따라서 아래 비교는 대부분 "증명된 최적해끼리의 비교"가 아니라 "2시간 안에 찾은 가장 좋은 해끼리의 비교"입니다.

`OD only / lambda = 0.0`만 `OPTIMAL`입니다. 나머지는 incumbent 비교로 읽는 것이 맞습니다.

## 조합 비교

| 케이스 | lambda | 상태 | 서버 수 | actual migrations | peak util | peak overbooking ratio | spot susp prob | completion ratio | runtime (s) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| OD only | 0.1 | TIME_LIMIT | 6 | 0 | 99.9% | 0.999 | 0.0 | 1.000 | 7200.2 |
| OD only | 0.0 | OPTIMAL | 5 | 410 | 99.9% | 0.999 | 0.0 | 1.000 | 2.2 |
| OD + SP | 0.1 | TIME_LIMIT | 5 | 17 | 100.0% | 1.496 | 0.2 | 0.971 | 7200.2 |
| OD + SP | 0.0 | TIME_LIMIT | 5 | 207 | 99.9% | 1.398 | 0.2 | 0.967 | 7200.5 |
| OD + BJ | 0.1 | TIME_LIMIT | 5 | 0 | 99.9% | 0.999 | 0.0 | 1.000 | 7200.2 |
| OD + BJ | 0.0 | TIME_LIMIT | 5 | 40 | 100.0% | 1.000 | 0.0 | 1.000 | 7201.0 |
| OD + SP + BJ | 0.1 | TIME_LIMIT | 4 | 0 | 100.0% | 1.148 | 0.2 | 0.967 | 7200.2 |
| OD + SP + BJ | 0.0 | TIME_LIMIT | 4 | 110 | 100.0% | 1.140 | 0.2 | 0.979 | 7200.2 |

## 비율 비교

| 케이스 | workload mix (OD/SP/BJ) | lambda | 상태 | 서버 수 | actual migrations | peak util | peak overbooking ratio | spot susp prob | completion ratio | runtime (s) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| balanced | 8 / 8 / 8 | 0.1 | TIME_LIMIT | 8 | 28 | 99.8% | 1.195 | 0.2 | 0.967 | 7200.1 |
| balanced | 8 / 8 / 8 | 0.0 | TIME_LIMIT | 5 | 140 | 100.0% | 1.211 | 0.2 | 0.967 | 7200.2 |
| service-heavy | 14 / 5 / 5 | 0.1 | TIME_LIMIT | 5 | 0 | 100.0% | 1.140 | 0.2 | 0.967 | 7200.2 |
| service-heavy | 14 / 5 / 5 | 0.0 | TIME_LIMIT | 5 | 181 | 100.0% | 1.000 | 0.0 | 1.000 | 7200.5 |
| spot-heavy | 5 / 14 / 5 | 0.1 | TIME_LIMIT | 4 | 0 | 99.9% | 1.372 | 0.2 | 0.967 | 7200.2 |
| spot-heavy | 5 / 14 / 5 | 0.0 | TIME_LIMIT | 4 | 79 | 100.0% | 1.224 | 0.2 | 0.967 | 7200.2 |
| batch-heavy | 5 / 5 / 14 | 0.1 | TIME_LIMIT | 6 | 0 | 100.0% | 1.000 | 0.2 | 0.967 | 7200.2 |
| batch-heavy | 5 / 5 / 14 | 0.0 | TIME_LIMIT | 5 | 0 | 100.0% | 1.000 | 0.2 | 0.967 | 7200.2 |

## 핵심 관찰

### 1. `lambda = 0.1`은 migration을 매우 강하게 억제합니다
- `OD only`는 `410 -> 0`
- `OD + SP`는 `207 -> 17`
- `OD + SP + BJ`는 `110 -> 0`
- `balanced`는 `140 -> 28`
- `service-heavy`는 `181 -> 0`
- `spot-heavy`는 `79 -> 0`

즉, 현재 목적함수에서는 `lambda = 0.1`만으로도 "서버를 더 쓰더라도 이동은 줄이는 방향"이 매우 강하게 작동합니다.

### 2. migration을 막으면 서버 수가 늘어나는 케이스가 분명히 있습니다
- `OD only`: 서버 `5 -> 6`
- `balanced`: 서버 `5 -> 8`
- `batch-heavy`: 서버 `5 -> 6`

반면 `OD + SP`, `OD + BJ`, `OD + SP + BJ`, `service-heavy`, `spot-heavy`는 서버 수가 그대로여도 migration만 크게 줄었습니다.

### 3. spot이 포함된 케이스는 여전히 suspension 상한 근처까지 갑니다
spot이 들어간 대부분 케이스에서 `spot susp prob = 0.2`가 관찰됩니다. 즉, 현재 workload 강도에서는 모델이 spot 중단 허용치를 적극적으로 활용하고 있습니다.

다만 `service-heavy / lambda = 0.0`은 `spot susp prob = 0.0`, `completion ratio = 1.0`이라서, 같은 비율 스윕 안에서도 spot 압박이 충분히 낮은 경우가 있다는 점을 보여 줍니다.

### 4. overbooking ratio가 큰 케이스는 spot 비중이 있는 쪽에 몰립니다
- `OD + SP / lambda = 0.1`: `1.496`
- `spot-heavy / lambda = 0.1`: `1.372`
- `balanced / lambda = 0.0`: `1.211`

즉, spot을 활용하는 조합일수록 예약 기준으로는 더 공격적인 packing이 나타납니다.

### 5. migration 변수 합과 실제 migration event 수는 다를 수 있습니다
예를 들어:
- `OD + BJ / lambda = 0.0`: `migration_count = 259`, `actual migrations = 40`
- `batch-heavy / lambda = 0.0`: `migration_count = 115`, `actual migrations = 0`

따라서 실험 해석에서는 `migration_count`보다 `actual migrations`를 우선해서 보는 것이 맞습니다.

## 시각화 메모
현재 간트차트는 `2602` 실험의 스타일을 따라 실제 migration event를 화살표로 직접 그립니다. 따라서 화살표가 보이지 않는 케이스는 "시각화 누락"이 아니라 실제로 `actual migrations = 0`인 경우입니다.

spot이 있는 케이스는 추가로 두 장을 같이 봐야 합니다.
- `server_workload_gantt_suspended.png`
  spot suspension이 실제로 발생한 시나리오 한 건
- `server_workload_gantt_clean.png`
  suspension 없이 지나간 시나리오 한 건

## 다음에 보면 좋은 것
- `OD only / lambda = 0.0`는 migration을 허용했을 때 서버 1대를 줄일 수 있다는 점을 가장 분명하게 보여 줍니다.
- `balanced / lambda = 0.1`는 migration penalty가 너무 크면 서버 수가 크게 늘 수 있다는 점을 보여 줍니다.
- `OD + SP`와 `spot-heavy`는 spot suspension과 overbooking이 동시에 나타나는 대표 케이스입니다.
