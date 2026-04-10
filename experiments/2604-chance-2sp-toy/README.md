# Chance-Constrained 2SP Toy 실험

## 개요
이 폴더는 Azure VM trace에서 작은 실험용 인스턴스를 만들고, chance-constrained two-stage stochastic placement 모델을 Gurobi로 푸는 실험 코드를 모아 둔 곳입니다.

현재 README는 최신 benchmark의 설정, 핵심 결과, 해석까지 한 번에 볼 수 있도록 정리합니다. 세부 전처리 설명은 [DATA_PREPROCESSING.md](./DATA_PREPROCESSING.md), 모델-구현 대응 검토는 [MODEL_REVIEW.md](./MODEL_REVIEW.md), 더 자세한 결과 메모는 [RESULTS.md](./RESULTS.md)에 있습니다.

## 현재 기준 benchmark 설정
- 전체 VM 수: `24`
- 서버 용량: `8`
- 후보 VM 필터: `vCPU <= 8`, `avg_cpu_mean >= 20`
- 시나리오 수: `10`
- spot completion ratio 제약:
  $\sum_{t \in T_j}\sum_{s \in S} a_{jst}(\xi) \ge \rho |T_j|$
- migration penalty 비교: $\lambda = 0.1$ 과 $\lambda = 0.0$
- 실행 방식: queue 기반, 최대 `4`개 작업 병렬, 각 작업 `threads = 8`
- time limit: `7200초`

## 핵심 스크립트
- `rebuild_sample_data.py`
- `build_dataset.py`
- `run_model.py`
- `visualize.py`
- `run_24vm_lambda_queue.py`

## 결과 파일 위치
- 전체 요약 표: `results/analysis/chance_2sp_toy_24vm_cap8_avg20_lambda_queue/all_jobs_summary.csv`
- lambda 비교 표: `results/analysis/chance_2sp_toy_24vm_cap8_avg20_lambda_queue/lambda_migration_comparison.csv`
- 조합 비교 그림: `results/analysis/chance_2sp_toy_24vm_cap8_avg20_lambda_queue/combination_lam000.png`, `combination_lam010.png`
- 비율 비교 그림: `results/analysis/chance_2sp_toy_24vm_cap8_avg20_lambda_queue/ratio_lam000.png`, `ratio_lam010.png`

대표 간트차트:
- migration이 많은 예시: `results/runs/chance_2sp_toy_24vm_combination_od_only_od24_sp0_bj0_sc10_cap8_avg20_lam000/server_workload_gantt_scen_01.png`
- spot 중단이 있는 시나리오: `results/runs/chance_2sp_toy_24vm_combination_od_sp_od12_sp12_bj0_sc10_cap8_avg20_lam010/server_workload_gantt_suspended.png`
- spot 중단이 없는 시나리오: `results/runs/chance_2sp_toy_24vm_combination_od_sp_od12_sp12_bj0_sc10_cap8_avg20_lam010/server_workload_gantt_clean.png`

## 지표 읽는 법
- `status`: `OPTIMAL`은 최적해가 증명된 경우이고, `TIME_LIMIT`은 2시간 안에 찾은 incumbent feasible solution입니다.
- `used servers`: 하루 동안 한 번이라도 켜진 서버 수입니다.
- `actual migrations`: 실제 placement가 바뀐 migration event 수입니다.
- `peak util`: 시나리오별 realized load를 서버 용량으로 나눈 값의 최대치입니다.
- `peak overbooking`: 예약 기준보다 실제 수요를 얼마나 공격적으로 겹쳐 담았는지 보여 주는 값입니다.
- `OD viol prob`: on-demand SLA 위반 확률의 최대값입니다.
- `SP susp prob`: spot VM suspension probability의 최대값입니다.
- `SP completion`: spot VM completion ratio의 최소값입니다.

## 조합 benchmark 결과

| 케이스 | lambda | 상태 | 서버 수 | actual migrations | peak util | peak overbooking | OD viol prob | SP susp prob | SP completion | runtime (s) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OD only | 0.1 | TIME_LIMIT | 6 | 0 | 99.9% | 0.999 | 0.0 | 0.0 | 1.000 | 7200.2 |
| OD only | 0.0 | OPTIMAL | 5 | 410 | 99.9% | 0.999 | 0.0 | 0.0 | 1.000 | 2.2 |
| OD + SP | 0.1 | TIME_LIMIT | 5 | 17 | 100.0% | 1.496 | 0.1 | 0.2 | 0.971 | 7200.2 |
| OD + SP | 0.0 | TIME_LIMIT | 5 | 207 | 99.9% | 1.398 | 0.1 | 0.2 | 0.967 | 7200.5 |
| OD + BJ | 0.1 | TIME_LIMIT | 5 | 0 | 99.9% | 0.999 | 0.0 | 0.0 | 1.000 | 7200.2 |
| OD + BJ | 0.0 | TIME_LIMIT | 5 | 40 | 100.0% | 1.000 | 0.0 | 0.0 | 1.000 | 7201.0 |
| OD + SP + BJ | 0.1 | TIME_LIMIT | 4 | 0 | 100.0% | 1.148 | 0.1 | 0.2 | 0.967 | 7200.2 |
| OD + SP + BJ | 0.0 | TIME_LIMIT | 4 | 110 | 100.0% | 1.140 | 0.0 | 0.2 | 0.979 | 7200.2 |

## 비율 benchmark 결과

| 케이스 | mix (OD/SP/BJ) | lambda | 상태 | 서버 수 | actual migrations | peak util | peak overbooking | OD viol prob | SP susp prob | SP completion | runtime (s) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| balanced | 8 / 8 / 8 | 0.1 | TIME_LIMIT | 8 | 28 | 99.8% | 1.195 | 0.0 | 0.2 | 0.967 | 7200.1 |
| balanced | 8 / 8 / 8 | 0.0 | TIME_LIMIT | 5 | 140 | 100.0% | 1.211 | 0.1 | 0.2 | 0.967 | 7200.2 |
| service-heavy | 14 / 5 / 5 | 0.1 | TIME_LIMIT | 5 | 0 | 100.0% | 1.140 | 0.0 | 0.2 | 0.967 | 7200.2 |
| service-heavy | 14 / 5 / 5 | 0.0 | TIME_LIMIT | 5 | 181 | 100.0% | 1.000 | 0.0 | 0.0 | 1.000 | 7200.5 |
| spot-heavy | 5 / 14 / 5 | 0.1 | TIME_LIMIT | 4 | 0 | 99.9% | 1.372 | 0.0 | 0.2 | 0.967 | 7200.2 |
| spot-heavy | 5 / 14 / 5 | 0.0 | TIME_LIMIT | 4 | 79 | 100.0% | 1.224 | 0.1 | 0.2 | 0.967 | 7200.2 |
| batch-heavy | 5 / 5 / 14 | 0.1 | TIME_LIMIT | 6 | 0 | 100.0% | 1.000 | 0.0 | 0.2 | 0.967 | 7200.2 |
| batch-heavy | 5 / 5 / 14 | 0.0 | TIME_LIMIT | 5 | 0 | 100.0% | 1.000 | 0.0 | 0.2 | 0.967 | 7200.2 |

## 한눈에 보이는 결과

### 1. migration penalty를 넣으면 migration이 크게 줄어듭니다
- `OD only`: `410 -> 0`
- `OD + SP`: `207 -> 17`
- `OD + SP + BJ`: `110 -> 0`
- `balanced`: `140 -> 28`
- `service-heavy`: `181 -> 0`
- `spot-heavy`: `79 -> 0`

즉, $\lambda = 0.1$은 현재 실험 세팅에서 "서버를 더 쓰더라도 이동은 줄이는 방향"으로 해를 강하게 밀어 줍니다.

### 2. migration을 막으면 서버 수가 늘어나는 경우가 있습니다
- `OD only`: 서버 `5 -> 6`
- `balanced`: 서버 `5 -> 8`
- `batch-heavy`: 서버 `5 -> 6`

반대로 `OD + SP`, `OD + BJ`, `OD + SP + BJ`, `service-heavy`, `spot-heavy`는 서버 수는 유지하면서 migration만 줄이는 방향으로 바뀌었습니다.

### 3. spot이 들어간 케이스는 suspension 상한을 적극적으로 씁니다
`OD + SP`, `OD + SP + BJ`, `balanced`, `spot-heavy`, `batch-heavy`, `service-heavy (lambda=0.1)`에서 `SP susp prob = 0.2`가 나타납니다. 즉, 현재 인스턴스에서는 spot 리소스를 쓰는 대신 허용된 중단 한계를 거의 다 활용하고 있습니다.

### 4. overbooking은 spot 비중이 있는 케이스에서 더 커집니다
- `OD + SP / lambda = 0.1`: `1.496`
- `spot-heavy / lambda = 0.1`: `1.372`
- `balanced / lambda = 0.0`: `1.211`

즉, spot을 포함한 조합이 예약 기준으로는 더 공격적인 packing을 유도합니다.

### 5. 현재 benchmark는 대부분 time limit 결과입니다
이번 24-VM benchmark에서 `OD only / lambda = 0.0`만 `OPTIMAL`입니다. 나머지는 모두 `TIME_LIMIT`이므로, 위 표는 대부분 "2시간 안에 찾은 가장 좋은 incumbent"끼리의 비교로 읽어야 합니다.

## 재현 명령
저장소 루트에서 실행합니다.

```powershell
.\.venv\Scripts\python.exe .\experiments\2604-chance-2sp-toy\run_24vm_lambda_queue.py --time-limit 7200 --max-workers 4 --threads 8
```

## 관련 문서
- 전처리 설명: [DATA_PREPROCESSING.md](./DATA_PREPROCESSING.md)
- 모델 검토 메모: [MODEL_REVIEW.md](./MODEL_REVIEW.md)
- 상세 결과 메모: [RESULTS.md](./RESULTS.md)
- 예전 문서 이름 유지용 안내: [RATIO_SWEEP.md](./RATIO_SWEEP.md), [COMBINATION_SWEEP.md](./COMBINATION_SWEEP.md)
