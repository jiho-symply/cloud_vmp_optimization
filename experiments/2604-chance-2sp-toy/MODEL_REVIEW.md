# 모델 검토 메모

## 목적

이 문서는 Notion 페이지 [VM type modeling](https://www.notion.so/32fa14fa72038089b589f47f81ab5c04)을 기준으로, 현재 `2604-chance-2sp-toy` 구현이 실제로 어떤 수학적 구조를 갖고 있는지 정리한 메모입니다.

기준 파일:

- 현재 구현: [run_model.py](C:\Users\jiho\code\cloud_vmp_optimization\experiments\2604-chance-2sp-toy\run_model.py)
- 검토 시점: `2026-04-13`

이번 정리에서는 최근 수정사항도 반영했습니다.

- `spot_present` 보조 변수는 제거했습니다.
- `gamma <= beta`, `phi <= gamma + 1 - beta` 같은 연결 제약은 제거했습니다.
- `L >= C u + \tau - M(1-\phi)` 형태의 strengthening constraint 는 제거했습니다.
- 따라서 문서에서도 더 이상 `\beta_{st}` 를 쓰지 않습니다.

## 표기 정리

현재 문서에서는 아래 표기를 사용합니다.

- $x_{ist}$: on-demand VM $i$ 의 시점 $t$ 에서 서버 $s$ 로의 배치
- $m_{it}$: on-demand VM $i$ 가 직전 활성 시점 대비 서버를 바꾸었는지 나타내는 migration 변수
- $y_{js}$: spot VM $j$ 의 서버 배치
- $z_{kst}$: batch job $k$ 의 서버-시점 배치
- $u_{st}$: 시점 $t$ 에서 서버 $s$ 가 켜져 있는지 나타내는 변수
- $u_s$: planning horizon 동안 한 번이라도 사용된 서버인지 나타내는 변수
- $a_{jst}(\xi)$: 시나리오 $\xi$ 에서 spot VM $j$ 가 시점 $t$ 에 서버 $s$ 에서 실제로 살아 있는지 나타내는 변수
- $\gamma_{st}(\xi)$: 시나리오 $\xi$ 에서 서버 $s$, 시점 $t$ 의 spot suspension 상태
- $\phi_{st}(\xi)$: 시나리오 $\xi$ 에서 서버 $s$, 시점 $t$ 에 overload 허용을 켰는지 나타내는 변수
- $\eta_s(\xi)$: 시나리오 $\xi$ 에서 서버 $s$ 가 on-demand SLA violation 을 한 번이라도 겪었는지 나타내는 변수
- $\delta_j(\xi)$: 시나리오 $\xi$ 에서 spot VM $j$ 가 한 번이라도 중단되었는지 나타내는 변수
- $L_{st}(\xi)$: 시나리오 $\xi$ 에서 서버 $s$, 시점 $t$ 의 총 CPU load
- $\widetilde{L}_{st}(\xi)$: 에너지 계산용 capped load
- $E_s$: 서버 $s$ 의 총 에너지 사용량

코드 이름과의 대응은 다음과 같습니다.

- $u_s$ 는 코드의 `u_used[s]` 입니다.
- $\widetilde{L}_{st}(\xi)$ 는 코드의 `power_load[s, t, xi]` 입니다.
- $E_s$ 는 코드의 `server_energy[s]` 입니다.
- 문서에서는 $m_{it}$ 로 쓰지만, 코드는 구현 편의상 `(i, t^-, t)` 형태로 저장합니다.

즉, `u_used` 는 새로운 개념이 아니라 문서의 $u_s$ 를 코드에서 구현한 이름입니다.

## Notion 모델의 큰 틀

Notion 모델의 핵심은 다음과 같습니다.

1. 1단계에서 on-demand, spot, batch 의 기본 배치를 정합니다.
2. 2단계에서 시나리오별로 실제 load 와 suspension 을 반영합니다.
3. on-demand 는 chance constraint 로 보호하고, spot 은 허용된 수준까지 중단을 허용합니다.

현재 코드도 이 큰 틀은 그대로 유지합니다.

## 현재 코드의 1단계 구조

### 서버 사용 변수

시점별 서버 on/off 는 $u_{st}$ 이고, horizon 전체에서 서버를 썼는지는 $u_s$ 로 둡니다.

$$
u_s \ge u_{st}
\qquad
\forall s \in S,\ \forall t \in T
$$

$$
u_s \le \sum_{t \in T} u_{st}
\qquad
\forall s \in S
$$

즉, 서버가 어떤 시점에라도 켜져 있으면 $u_s = 1$ 이어야 합니다.

### on-demand 배치와 migration

on-demand VM 은 활성 시점마다 정확히 하나의 서버에 배치됩니다.

$$
\sum_{s \in S} x_{ist} = 1
\qquad
\forall i \in I,\ \forall t \in T_i
$$

그리고 서버가 켜져 있어야 해당 VM 을 놓을 수 있습니다.

$$
x_{ist} \le u_{st}
\qquad
\forall i \in I,\ \forall s \in S,\ \forall t \in T_i
$$

migration 은 수학적으로는 $m_{it}$ 로 읽는 것이 자연스럽습니다. 다만 여기서 $t$ 는 VM $i$ 의 첫 활성 시점을 제외한 "직전 활성 시점과 비교 가능한 시점"입니다.

직전 활성 시점을 $t^-(i,t)$ 라고 쓰면, 현재 구현은 아래 의미를 갖습니다.

$$
m_{it} \ge x_{ist} - x_{is,t^-(i,t)}
\qquad
\forall i \in I,\ \forall s \in S,\ \forall t \in T_i \setminus \{\text{첫 활성 시점}\}
$$

$$
m_{it} \ge x_{is,t^-(i,t)} - x_{ist}
\qquad
\forall i \in I,\ \forall s \in S,\ \forall t \in T_i \setminus \{\text{첫 활성 시점}\}
$$

즉, 직전 활성 시점과 현재 시점의 서버 배치가 다르면 $m_{it}=1$ 이 됩니다.

### spot 과 batch 의 1단계 배치

spot VM 은 서버 하나를 미리 고릅니다.

$$
\sum_{s \in S} y_{js} = 1
\qquad
\forall j \in J
$$

$$
y_{js} \le u_{st}
\qquad
\forall j \in J,\ \forall s \in S,\ \forall t \in T_j
$$

batch job 은 서버-시점 쌍 하나에만 배치됩니다.

$$
\sum_{s \in S}\sum_{t \in T} z_{kst} = 1
\qquad
\forall k \in K
$$

$$
z_{kst} \le u_{st}
\qquad
\forall k \in K,\ \forall s \in S,\ \forall t \in T
$$

## 현재 코드의 2단계 구조

### spot 실제 실행 상태

spot VM 은 자신이 1단계에서 배치된 서버에서만 살아 있을 수 있습니다.

$$
a_{jst}(\xi) \le y_{js}
\qquad
\forall j \in J,\ \forall s \in S,\ \forall t \in T_j,\ \forall \xi \in \Xi
$$

또한 해당 서버-시점-시나리오에서 suspension 이 걸리면 실행될 수 없습니다.

$$
a_{jst}(\xi) \le 1 - \gamma_{st}(\xi)
\qquad
\forall j \in J,\ \forall s \in S,\ \forall t \in T_j,\ \forall \xi \in \Xi
$$

여기서 중요한 점은 현재 구현에는 `spot_present` 같은 보조 변수가 없다는 것입니다. 따라서 어떤 서버-시점에 실제로 spot 이 하나도 배치되지 않았더라도, 그 지점의 $\gamma_{st}(\xi)$ 자체는 별도 제약으로 고정하지 않습니다.

현재 모델에서는 이것이 문제를 만들지 않습니다. 이유는 $\gamma_{st}(\xi)$ 가 오직 $a_{jst}(\xi)$ 를 막는 방향으로만 등장하기 때문입니다. 즉, 그 서버-시점에 spot 이 없으면 $\gamma_{st}(\xi)$ 값은 의미가 거의 없고, 실제 load 나 SLA 계산에도 직접 들어가지 않습니다.

### 서버 load

시나리오별 서버 load 는 on-demand, 실제로 살아 있는 spot, batch 를 합쳐서 계산합니다.

$$
L_{st}(\xi)
=
\sum_{i \in I} d^O_{it}(\xi) x_{ist}
+
\sum_{j \in J} d^S_{jt}(\xi) a_{jst}(\xi)
+
\sum_{k \in K} d^B_k(\xi) z_{kst}
$$

여기서

- $d^O_{it}(\xi)$ 는 on-demand VM 의 시나리오별 CPU demand
- $d^S_{jt}(\xi)$ 는 spot VM 의 시나리오별 CPU demand
- $d^B_k(\xi)$ 는 batch job 의 시나리오별 CPU demand

입니다.

### on-demand SLA violation

현재 구현에서 overload 허용 제약은 다음 한 줄입니다.

$$
L_{st}(\xi) \le C_s u_{st} + M \phi_{st}(\xi)
\qquad
\forall s \in S,\ \forall t \in T,\ \forall \xi \in \Xi
$$

이 제약이 의미하는 것은 명확합니다.

$$
L_{st}(\xi) > C_s u_{st}
\;\Rightarrow\;
\phi_{st}(\xi)=1
$$

즉, overload 가 발생하면 $\phi_{st}(\xi)$ 는 반드시 켜져야 합니다.

반대로

$$
\phi_{st}(\xi)=1
\;\Rightarrow\;
L_{st}(\xi) > C_s u_{st}
$$

는 현재 코드에서 강제하지 않습니다. 예전에 들어 있던 strengthening constraint 를 제거했기 때문입니다.

따라서 현재 구현에서 $\phi_{st}(\xi)$ 는 "overload 가 나면 반드시 1 이어야 하는 변수"이지, 양방향으로 정확히 일치하는 완전한 overload indicator 로 해석하면 안 됩니다.

서버 단위 violation flag 는 아래처럼 연결됩니다.

$$
\eta_s(\xi) \ge \phi_{st}(\xi)
\qquad
\forall s \in S,\ \forall t \in T,\ \forall \xi \in \Xi
$$

$$
\eta_s(\xi) \le \sum_{t \in T} \phi_{st}(\xi)
\qquad
\forall s \in S,\ \forall \xi \in \Xi
$$

그리고 on-demand chance constraint 는 서버별로 다음과 같이 걸립니다.

$$
\sum_{\xi \in \Xi} p_\xi \eta_s(\xi) \le \epsilon^{OD}
\qquad
\forall s \in S
$$

### spot SLA

spot VM 의 suspension flag 는 "어떤 활성 시점에서라도 실행되지 않으면 1" 이 되도록 구성됩니다.

$$
\delta_j(\xi) \ge 1 - \sum_{s \in S} a_{jst}(\xi)
\qquad
\forall j \in J,\ \forall t \in T_j,\ \forall \xi \in \Xi
$$

$$
\delta_j(\xi) \le \sum_{t \in T_j}
\left(
1 - \sum_{s \in S} a_{jst}(\xi)
\right)
\qquad
\forall j \in J,\ \forall \xi \in \Xi
$$

확률 제약은 다음과 같습니다.

$$
\sum_{\xi \in \Xi} p_\xi \delta_j(\xi) \le \epsilon^{SP}
\qquad
\forall j \in J
$$

또한 현재 spot SLA 는 각 시나리오마다 최소 서비스 비율을 직접 보장합니다.

$$
\sum_{t \in T_j}\sum_{s \in S} a_{jst}(\xi)
\ge
\rho |T_j|
\qquad
\forall j \in J,\ \forall \xi \in \Xi
$$

즉, spot completion 은 확률가중 평균으로만 관리하는 것이 아니라, 각 시나리오 내부에서도 최소한 $\rho$ 비율 이상 살아 있어야 합니다.

## 에너지 계산용 capped load

현재 에너지 목적에서는 실제 CPU load 가 capacity 를 넘더라도, 전력 계산에는 최대 capacity 까지만 반영합니다.

이를 위해

$$
\widetilde{L}_{st}(\xi)=\min\{L_{st}(\xi),\ C_s u_{st}\}
$$

를 선형화해서 사용합니다.

구현은 아래 네 식입니다.

$$
\widetilde{L}_{st}(\xi) \le L_{st}(\xi)
$$

$$
\widetilde{L}_{st}(\xi) \le C_s u_{st}
$$

$$
\widetilde{L}_{st}(\xi) \ge L_{st}(\xi) - M \phi_{st}(\xi)
$$

$$
\widetilde{L}_{st}(\xi) \ge C_s u_{st} - M\bigl(1-\phi_{st}(\xi)\bigr)
$$

직관은 다음과 같습니다.

- $\phi_{st}(\xi)=0$ 이면 $\widetilde{L}_{st}(\xi)=L_{st}(\xi)$ 쪽이 선택됩니다.
- $\phi_{st}(\xi)=1$ 이면 $\widetilde{L}_{st}(\xi)=C_s u_{st}$ 쪽이 선택됩니다.

즉, 현재 구현에서 $\phi_{st}(\xi)$ 는 overload flag 역할과 함께, 에너지 계산에서 어느 branch 를 쓰는지도 고르는 selector 역할을 합니다.

## 목적함수

현재 코드는 두 가지 목적을 모두 지원합니다.

### 1. 서버 수 최소화 목적

chance-constrained toy benchmark 에서 기본 목적은 다음과 같습니다.

$$
\min
\sum_{s \in S} u_s
+
\lambda \frac{1}{|I|}
\sum_{i \in I}\sum_{t \in T_i \setminus \{\text{첫 활성 시점}\}} m_{it}
$$

즉, 사용 서버 수와 평균 migration penalty 만 최소화합니다.

### 2. 에너지 최소화 목적

에너지 목적에서는 idle 전력, capped CPU 전력, migration 전력을 더합니다.

$$
\min
\sum_{s \in S}\sum_{t \in T} P^{idle} u_{st}
+
\sum_{s \in S}\sum_{t \in T}\sum_{\xi \in \Xi}
p_\xi \frac{P^{cpu}}{C_s}\widetilde{L}_{st}(\xi)
+
P^{mig}\sum_{i \in I}\sum_{t \in T_i \setminus \{\text{첫 활성 시점}\}} m_{it}
$$

또한 서버별 총 에너지 변수 $E_s$ 를 두고

$$
E_s
=
P^{idle}\sum_{t \in T}u_{st}
+
\frac{P^{cpu}}{C_s}
\sum_{t \in T}\sum_{\xi \in \Xi} p_\xi \widetilde{L}_{st}(\xi)
$$

로 정의합니다.

## symmetry-breaking

현재 구현의 기본 symmetry-breaking 은 다음 한 가지입니다.

$$
u_s \ge u_{s+1}
\qquad
\forall s=1,\dots,|S|-1
$$

즉, 뒤 인덱스 서버를 쓰려면 앞 인덱스 서버도 먼저 쓰도록 강제합니다.

에너지 목적을 사용할 때는 여기에 추가로

$$
E_s \ge E_{s+1}
\qquad
\forall s=1,\dots,|S|-1
$$

도 둡니다.

즉, 에너지 목적에서는 "앞 인덱스 서버가 더 많이 켜지거나 더 높은 에너지를 쓰는 방향"으로 정렬을 유도합니다.

문서에서 `u_s >= u_{s+1}` 를 두 번 다른 의미로 쓰지 않도록, 여기서는 오직 한 번만 명시합니다. 코드의 `u_used[s]` 는 문서의 $u_s$ 와 같은 변수입니다.

## Notion 모델과의 일치점

현재 구현은 아래 부분에서 Notion 모델과 큰 틀을 공유합니다.

1. two-stage 구조를 유지합니다.
2. on-demand, spot, batch 를 분리해 다룹니다.
3. on-demand 는 서버 단위 chance constraint 로 보호합니다.
4. spot 은 suspension 가능 workload 로 두고, 별도 SLA 제약을 둡니다.
5. migration 은 1단계 배치 변화로 모델링합니다.

## Notion 모델과의 차이점

현재 구현은 toy benchmark 목적상 아래와 같은 차이가 있습니다.

### 1. `spot_present` 보조 변수를 쓰지 않습니다

이전 버전에서는 서버-시점에 spot 이 존재하는지 나타내는 보조 변수를 둘 수 있었지만, 현재 구현은 이를 제거했습니다.

그 결과:

- spot 이 없는 서버-시점에서 $\gamma_{st}(\xi)$ 는 의미상 자유로울 수 있습니다.
- 하지만 $\gamma_{st}(\xi)$ 가 spot 실행 변수 $a_{jst}(\xi)$ 를 막는 데만 쓰이므로, spot 이 없는 곳에서의 값은 해의 물리적 의미에 큰 영향을 주지 않습니다.

### 2. $\phi_{st}(\xi)$ 를 양방향으로 조이지 않습니다

현재 구현은

$$
L_{st}(\xi) \le C_s u_{st} + M\phi_{st}(\xi)
$$

만으로 overload 시 $\phi=1$ 을 강제합니다.

하지만 예전의 strengthening constraint 는 제거했기 때문에,

- overload 이면 $\phi=1$ 이어야 한다
- $\phi=1$ 이라고 해서 반드시 strict overload 라고 말할 수는 없다

는 비대칭 해석을 가져야 합니다.

즉, 현재 $\phi$ 는 "완전히 타이트한 overload indicator"라기보다, overload 허용과 capped-load branch 선택에 사용되는 binary 변수입니다.

### 3. batch 는 toy 실험용 단순화가 들어가 있습니다

현재 batch 는 원래 VM workload 를 time slot 단위로 쪼개어 만든 batch job 들을 사용합니다. 즉, production batch scheduler 라기보다 split 가능한 잔여 workload 를 단순화해서 흉내 낸 구조입니다.

### 4. 목적함수가 실험에 따라 달라집니다

Notion 기본 설명은 서버 수 최소화 중심이지만, 현재 코드는 에너지 최소화 benchmark 도 함께 지원합니다.

## 정리

현재 `2604-chance-2sp-toy` 구현은 Notion 의 기본 two-stage chance-constrained VMP 구조를 유지하면서, toy benchmark 실험에 맞게 다음 방향으로 단순화되어 있습니다.

- `u_s` 는 코드의 `u_used[s]` 로 직접 구현됩니다.
- migration 은 수학적으로는 $m_{it}$ 로 읽는 것이 맞습니다.
- `spot_present` 보조 변수는 현재 쓰지 않습니다.
- $\phi_{st}(\xi)$ 는 overload 발생 시 반드시 켜져야 하지만, 그 역방향까지 강제하지는 않습니다.
- 에너지 계산은 $\widetilde{L}_{st}(\xi)=\min\{L_{st}(\xi), C_su_{st}\}$ 를 선형화해서 사용합니다.
- symmetry-breaking 은 기본적으로 $u_s \ge u_{s+1}$ 이고, 에너지 목적일 때만 $E_s \ge E_{s+1}$ 를 추가합니다.

따라서 현재 모델은 Notion 원형과 완전히 동일한 "정교한 원모델"이라기보다, 핵심 구조는 유지하되 실험 가능성과 해석 편의성을 우선한 toy MILP 구현으로 보는 것이 정확합니다.
