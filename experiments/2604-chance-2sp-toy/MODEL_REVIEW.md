# 모델 검토 메모

## 목적
이 문서는 현재 `2604-chance-2sp-toy` 구현이 Notion 모델과 어떻게 대응되는지, 그리고 코드에서 의도적으로 단순화한 부분이 무엇인지 정리합니다.

## 현재 구현의 핵심 구조
- `x_{ist}`
  on-demand workload의 시간별 서버 배치
- `m`
  on-demand workload의 실제 서버 이동을 나타내는 migration 변수
- `y_{js}`
  spot workload의 1단계 서버 배치
- `z_{kst}`
  batch job의 배치
- `a_{jst}(\xi)`
  시나리오별 spot workload 활성 여부
- `\gamma_{st}(\xi)`
  특정 서버-시간-시나리오에서 spot suspension이 일어났는지 나타내는 flag
- `\phi_{st}(\xi)`
  특정 서버-시간-시나리오에서 on-demand SLA violation 지점인지 나타내는 flag
- `\eta_s(\xi)`
  특정 서버가 해당 시나리오에서 하루 중 한 번이라도 on-demand violation을 겪었는지 나타내는 flag
- `\delta_j(\xi)`
  특정 spot workload가 해당 시나리오에서 한 번이라도 중단되었는지 나타내는 flag

## 최신 구현에서 바뀐 점

### 1. spot 완료율 제약
예전에는 기대값 기준 completion ratio를 사용했습니다.

현재는 각 시나리오마다 아래 제약을 직접 둡니다.

$$
\sum_{t \in T_j}\sum_{s \in S} a_{jst}(\xi) \ge \rho |T_j|
\qquad \forall j \in J,\ \forall \xi \in \Xi
$$

의미:
- 어떤 spot workload도 특정 시나리오에서 너무 오래 꺼져 있으면 안 됩니다.
- 기대값으로 평균 내는 방식보다 더 보수적입니다.

### 2. spot activity 하한 제약 제거
예전에는 아래와 같은 하한 tightening을 넣은 적이 있었습니다.

$$
a_{jst}(\xi) \ge y_{js} - \gamma_{st}(\xi)
$$

현재 구현에서는 이 제약을 제거했습니다.

의미:
- 의도적으로 모델을 느슨하게 두고, `a`를 상한 제약과 완료율 제약만으로 통제합니다.

### 3. 목적함수 단순화
현재 목적함수는 두 항만 남겨 둡니다.

$$
\min \sum_{s \in S} u^{used}_s
\;+\;
\lambda \cdot \frac{1}{|I|} \sum_{i \in I} \sum_{(t', t) \in \mathcal{T}_i} m_{i,t',t}
$$

여기서:
- 첫 번째 항은 사용 서버 수
- 두 번째 항은 평균 migration penalty

이전처럼 `\gamma`, `\phi`, `\eta`, `\delta`에 대한 작은 tie-break penalty는 더 이상 넣지 않습니다.

### 4. migration 정의
현재 migration은 단순히 `t-1` 고정 차분이 아닙니다.

on-demand workload가 실제로 관측된 연속 전이 집합

$$
\mathcal{T}_i = \{(t', t)\}
$$

를 사용해, 이전 관측 시점과 현재 관측 시점 사이의 서버 변경을 migration으로 계산합니다.

의미:
- VM이 중간에 유입되거나 일부 시간대만 존재해도 모델이 깨지지 않습니다.

### 5. 시나리오 샘플링 고정
현재 수요 샘플링은 `(vm_id, time, scenario)` 기준 deterministic triangular draw입니다.

의미:
- 같은 원본 VM은 어떤 benchmark에서 어떤 역할로 쓰이더라도 동일한 scenario 수요를 가집니다.
- `lambda = 0.1`과 `lambda = 0.0` 비교에서도 수요 샘플이 바뀌지 않습니다.

## 여전히 남아 있는 단순화

### 1. on-demand SLA는 서버 단위
현재 on-demand violation은 개별 VM 단위가 아니라 서버 단위로 계산합니다.

즉, 특정 서버-시간-시나리오의 overload 지점을 `\phi_{st}(\xi)`로 잡고 있습니다.

이건 다음을 뜻합니다.
- 어느 on-demand VM이 실제로 침해를 받았는지까지는 추적하지 않습니다.
- 구현은 단순하지만, 더 세밀한 SLA 해석은 아닙니다.

### 2. spot suspension은 서버 단위 shock와 연결
`a_{jst}(\xi)`는 개별 workload 변수이지만, 실제 suspension trigger는 `\gamma_{st}(\xi)`와 묶여 있습니다.

즉, 같은 서버 위의 spot workload들이 함께 영향을 받는 구조입니다.

### 3. batch는 1-slot job 집합으로 근사
현재 batch는 원래 VM workload를 시간 슬롯별 job으로 쪼갠 것입니다.

장점:
- 단순하고 해석이 쉽습니다.

한계:
- 슬롯 간 선후관계나 job 연속성은 표현하지 않습니다.

## 현재 코드와 Notion 모델의 관계
전체적으로 보면 현재 구현은 Notion 모델의 toy stand-in으로 충분히 사용할 수 있습니다.

특히 다음은 잘 맞습니다.
- two-stage 구조
- on-demand / spot / batch 분리
- chance constraint
- spot suspension과 on-demand 보호의 상호작용
- migration penalty

반면, 아래는 해석할 때 주의해야 합니다.
- on-demand SLA가 서버 단위라는 점
- batch가 slot-level split job이라는 점
- `TIME_LIMIT` 결과는 최적해가 아니라 incumbent solution이라는 점

## 최신 결과는 어디에 정리하는가
수치 결과와 benchmark 비교는 `RESULTS.md` 한 파일에만 적습니다.

이 문서는 모델 구조와 구현 선택만 설명합니다.
