# Micro top-10 unscaled v1 분석

## 생성 기준

- Source:
  `data/processed/notion_toy_google2019_v5_mode_capacity_bounded`
- Output:
  `data/processed/notion_toy_google2019_micro_top10_unscaled_v1`
- Class별 10개 선택:
  `coverage-weighted mean CPU usage + coverage-weighted mean memory usage`
  내림차순, 동점은 `vm_id` 오름차순
- OD/SP eligibility:
  active한 모든 30분 slot에 scenario-0 5분 관측이 최소 한 개 존재
- 복사 정책:
  canonical `q`, request, lifecycle/provenance, scenario-0 usage를 값 변경 없이 복사
- 제외한 변환:
  target-q scaling, usage rescaling, OD memory transform, minimum-server guarantee
- Model workload scenarios:
  scenario 0은 관측값, scenario 1–9는 CPU sigma `0.24`, memory sigma `0.12`

Builder의 validation은 canonical source의 생성 전후 SHA-256 동일성과 선택된 request/usage
row의 exact equality를 검사한다.

과거 `micro_stress_v1`과 class별 선택 ID도 각각 10/10개로 완전히 같다. 따라서 두
fixture의 차이는 후보 교체가 아니라, 새 fixture에서 과거의 q/usage/memory-pressure
변환을 제거하고 canonical 값을 복원한 것이다.

## Configured q 진단

`q`를 실제 capacity demand로 본 결과가 아니라 configured-size envelope로 본 진단이다.
현재 service capacity constraint는 `q` 예약량이 아니라 scenario usage를 사용한다.

| 대상 | CPU q 합 | MEM q 합 | 모두 동시라고 가정한 packing | lifecycle 반영 최대 packing |
|---|---:|---:|---:|---:|
| OD 10 | 3.2129 | 3.0399 | exact 4대 | exact 4대 |
| SP 10 | 4.1811 | 0.3454 | exact 5대 | exact 2대 |
| BJ source 10 | 2.2828 | 2.4408 | exact 3대 | 해당 없음 |
| OD+SP | 7.3940 | 3.3853 | 하한 8대, FFD 9대 | exact 5대 |

OD+SP의 동시 가정은 lifecycle을 무시하므로 모델 필요 서버 수로 해석하지 않는다. 실제로
동시에 active한 service VM은 최대 12개다.

## Service actual-usage 진단

모든 Spot을 admission한다고 가정하고, 30분 coverage-weighted usage를 VM별 2차원
CPU/MEM exact bin packing으로 계산했다.

| 입력 | 최대 CPU 합 | 최대 MEM 합 | 모든 slot의 최대 exact packing |
|---|---:|---:|---:|
| Observed scenario 0, OD | 1.5013 | 2.2158 | 3대 |
| Observed scenario 0, SP | 0.8312 | 0.0656 | 1대 |
| Observed scenario 0, OD+SP | 2.2176 | 2.2286 | 3대 |
| Scenario 0–9, OD+SP 전체 최대 | 2.2176 | 2.2741 | 3대 |

10개 workload scenario 각각에서도 모든 service VM의 slot별 exact packing 최대값은 3대였다.

변환을 제거한 뒤에도 선택된 OD 자체는 메모리 우세다. Scenario-0 OD의 48-slot 평균은
CPU 1.2241대분, MEM 2.0371대분이다. 이는 OD memory transform 때문이 아니라
canonical trace와 top-average-usage 선택의 결과다. SP는 반대로 CPU 우세라서 OD+SP의
전체 peak는 CPU 2.2176, MEM 2.2286으로 거의 균형이다.

## Batch workload

선택된 batch candidate 10개는 loader에서 3개 family로 바뀐다.

- Workload: 87.4469 slot-unit
- CPU resource volume: 9.1919 server-slot
- MEM resource volume: 10.5973 server-slot
- 48 slot 균등분산 등가량:
  CPU 0.1915대, MEM 0.2208대

Family별 `(q_cpu, q_mem)`, `(rho_cpu, rho_mem)`, `W`:

| Family | q | rho | W |
|---|---|---|---:|
| batch000 | (0.14150, 0.44436) | (0.02528, 0.11140) | 14.0181 |
| batch001 | (0.18647, 0.44436) | (0.08217, 0.05581) | 21.8696 |
| batch002 | (0.42657, 0.22255) | (0.13655, 0.15158) | 51.5591 |

BJ의 0.1915/0.2208은 하루 평균 resource-volume 등가량이며 특정 slot의 load가 아니다.
실제 배치는 family activation, startup/base overhead, service placement와 함께 결정된다.

## 전체 모델 최소 서버 확인

단순 합산 외에 full server-min formulation으로 feasibility를 확인했다.

- 10 workload scenarios
- 모든 Spot 10개 admission 고정
- 모든 Spot preemption `h=0` 고정
- OD CPU excess `=0` 고정
- Batch completion 및 family first-stage schedule 포함
- OD migration은 모델 정의대로 허용

결과:

- 서버 2대: infeasible
- 서버 3대: feasible, Gurobi가 1.85초에 feasibility incumbent를 확인

따라서 이 조건에서 최소 서버 수는 정확히 **3대**다. 기본 config의 `num_servers: 6`은
6대를 항상 켠다는 의미가 아니라 켤 수 있는 candidate server 수다. 서버 on/off 결정은
최적화가 하므로 3대 필요량에 대해 3대의 선택 여유를 둔다.
