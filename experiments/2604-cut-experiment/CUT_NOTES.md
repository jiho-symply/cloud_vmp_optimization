# Cut 아이디어 메모

## 목적

이 문서는 `2604-cut-experiment`에서 어떤 cut 후보를 실험하는지, 왜 넣는지, 어떤 문헌 축을 참고했는지 정리한 메모입니다.

## 현재 구현한 profile

### `baseline`

- 추가 cut 없음
- Gurobi 기본 cut 설정만 사용

### `activation`

추가 식:

- `u_{st} <=` 해당 서버-시점에 배치된 workload 변수 합
- `u_s <=` horizon 동안 해당 서버에 배치된 workload 합

의도:

- 실제로 아무 workload도 없는 server-on 해를 줄이기 위함
- 에너지 목적에서 지배되는 idle server activation을 제거

### `spot_server_link`

추가 식:

$$
\sum_{t \in T_j} a_{jst}(\xi) \ge \rho |T_j| y_{js}
$$

$$
\sum_{t \in T_j} a_{jst}(\xi) \ge |T_j| \left(y_{js} - \delta_j(\xi)\right)
$$

의도:

- spot VM이 선택된 서버와 scenario별 completion / suspension을 더 직접적으로 연결

### `spot_time_link`

추가 식:

$$
a_{jst}(\xi) \ge y_{js} - \delta_j(\xi)
$$

$$
\delta_j(\xi) \ge y_{js} + \gamma_{st}(\xi) - 1
$$

의도:

- `delta_j(\xi)=0`이면 선택된 서버에서 모든 active slot이 살아 있어야 함을 시간 단위로 더 직접적으로 표현
- `gamma`가 실제로 켜진 서버에 배치된 spot VM의 suspension flag `delta`와 더 직접적으로 연결되도록 강화

### `pairwise_cover`

시나리오별 서버-시점에서 두 workload의 수요 합이 capacity를 넘으면

$$
v_1 + v_2 \le 1 + \phi_{st}(\xi)
$$

를 추가합니다.

여기서 `v_1`, `v_2`는 다음 중 하나입니다.

- `x_{ist}`
- `a_{jst}(\xi)`
- `z_{kst}`

의도:

- 두 개만 같이 들어가도 overload가 확실한 조합을 relaxation이 더 빨리 인식하게 만듦

### `triple_cover`

세 workload의 합은 capacity를 넘지만, 모든 pair는 capacity 이내인 minimal triple cover에 대해

$$
v_1 + v_2 + v_3 \le 2 + \phi_{st}(\xi)
$$

를 추가합니다.

의도:

- pairwise cover로는 못 자르는 3원 조합을 추가로 줄이기 위함

구현 메모:

- 각 `(t,\xi)`에서 큰 수요 후보 일부만 사용합니다.
- exhaustive separation이 아니라 강한 후보 위주의 제한된 cover generation입니다.

### `uptime_symmetry`

기본 대칭 제거식

$$
u_s \ge u_{s+1}
$$

외에

$$
\sum_t u_{st} \ge \sum_t u_{s+1,t}
$$

를 추가합니다.

의도:

- 서버 순열 대칭을 더 줄이기 위함

### `solver_cover_focus`

Gurobi built-in cut 중 cover 계열만 강하게 켭니다.

- `Cuts=1`
- `CoverCuts=2`
- `FlowCoverCuts=2`
- `GUBCoverCuts=2`

### `solver_clique_focus`

Gurobi built-in cut 중 clique 계열을 강하게 켭니다.

- `Cuts=1`
- `CliqueCuts=2`
- `ZeroHalfCuts=2`

### `solver_implied_focus`

Gurobi built-in cut 중 implied 계열을 강하게 켭니다.

- `Cuts=1`
- `ImpliedCuts=2`
- `ProjImpliedCuts=2`
- `DualImpliedCuts=2`

### `solver_lift_focus`

Gurobi built-in cut 중 lifting / CG 계열을 강하게 켭니다.

- `Cuts=1`
- `MIRCuts=2`
- `StrongCGCuts=2`
- `RelaxLiftCuts=2`
- `LiftProjectCuts=2`

### `builtin_aggressive`

Gurobi built-in cut family 전반을 공격적으로 켭니다.

### `combined_light`

아래를 함께 켭니다.

- `activation`
- `spot_server_link`
- `spot_time_link`
- `pairwise_cover`
- `uptime_symmetry`

### `combined_full`

아래를 함께 켭니다.

- `activation`
- `spot_server_link`
- `spot_time_link`
- `pairwise_cover`
- `triple_cover`
- `uptime_symmetry`
- `builtin_aggressive`

## 문헌과 관련 아이디어

### 1. GAP / knapsack / cover inequality

- generalized assignment와 knapsack 계열에서는 cover inequality, lifted cover, configuration inequality가 가장 기본적인 강한 절단입니다.
- 현재 모델도 서버-시간 단위로 보면 assignment + capacity 구조이므로, `phi`를 매개로 overload 허용형 cover cut으로 옮길 수 있습니다.

참고:

- Díaz, J. A., Fernández, E. “A branch-and-cut algorithm for the generalized assignment problem with strong valid inequalities.” EJOR. DOI: `10.1016/S0377-2217(97)00054-4`
- “A new extended formulation of the generalized assignment problem and some associated valid inequalities.” Discrete Optimization. `https://www.sciencedirect.com/science/article/pii/S0166218X19303889`

### 2. cloud VM mapping / server consolidation

- cloud VM placement 계열에서는 assignment 구조와 server activation 구조를 강하게 묶는 inequality가 중요합니다.
- 특히 incompatibility, clique, activation linking, symmetry-breaking이 자주 유효합니다.

참고:

- “Strong valid inequalities for virtual machine mapping problems.” EJOR. DOI: `10.1016/j.ejor.2018.12.037`

### 3. chance-constrained packing

- chance-constrained binary packing에서는 probabilistic cover inequality가 자연스러운 후보입니다.
- 이번 구현에서는 scenario-mass separation까지는 가지 않고, 먼저 scenario별 deterministic cover와 spot suspension linking을 우선 넣었습니다.

참고:

- “Chance-constrained binary packing problems.” Optimization Online. `https://optimization-online.org/wp-content/uploads/2012/10/3639.pdf`
- “Lifting of probabilistic cover inequalities.” OR Letters. DOI: `10.1016/j.orl.2017.08.006`

### 4. solver-side cut sweep

- 사용자 cut과 별개로, built-in cut family를 어떤 축으로 켤 때 더 잘 먹히는지 보는 것도 유용합니다.

참고:

- Gurobi parameter reference: `https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html`

## 다음 단계 후보

- probabilistic cover를 `eta_s(\xi)` 또는 `delta_j(\xi)`와 직접 연결하는 scenario-mass cut
- lifted cover를 현재 pairwise / triple cover 위에 순차 lifting으로 보강
- root node separation 기반의 동적 cut generation
- manual cut과 solver cut profile을 조합한 adaptive profile selection

## 2026-04 추가 라운드 메모

이번에는 조합 프로파일을 빼고, 문헌 아이디어를 단일 프로파일 단위로 다시 쪼개서 비교했습니다.

추가한 단일 후보는 다음과 같습니다.

- `spot_completion_server_only`
- `spot_delta_server_only`
- `spot_time_lower_only`
- `delta_gamma_link_only`
- `spot_gamma_aggregate`
- `minimal_cover_general`
- `minimal_cover_fixed`
- `aggregate_fixed_load`
- `strategy_mipfocus_feasible`
- `strategy_mipfocus_bound`
- `strategy_method_dual`
- `strategy_method_barrier`
- `strategy_barrier_nocrossover`
- `strategy_heuristics_high`

관찰은 다음과 같습니다.

- `delta_gamma_link_only`는 spot 관련 수동 컷 중에서 가장 일관되게 primal 개선을 만들었습니다.
- `uptime_symmetry`는 여전히 dual bound 개선이 강했습니다.
- `minimal_cover_general`, `minimal_cover_fixed`는 classical cover 계열이라 넣어볼 가치는 있었지만, 현재 12VM 인스턴스에서는 개선 폭이 작았습니다.
- `aggregate_fixed_load`는 현재 인스턴스에서는 거의 도움이 되지 않았습니다.
- solver 전략 중에서는 `strategy_barrier_nocrossover`가 압도적으로 좋았습니다. root barrier를 매우 빠르게 끝낸 뒤 실제 branch-and-bound tree를 타기 시작했다는 점이 핵심입니다.
- 반대로 `strategy_method_dual`은 root bound를 거의 전혀 못 올렸고, 현 문제에는 맞지 않았습니다.

따라서 현재 단계에서 우선순위는 다음과 같습니다.

- root algorithm 튜닝을 먼저 본다.
- 수동 컷은 `uptime_symmetry`, `delta_gamma_link_only`처럼 실제로 incumbent 또는 bound 한쪽을 확실히 움직이는 것만 남긴다.
- probabilistic / mixing set 계열처럼 `delta`, `eta`를 직접 겨냥하는 시나리오 집계형 컷을 다음 구현 후보로 본다.

## 2026-04 조합 screening 메모

이번에는 가장 강했던 root 전략인 `strategy_barrier_nocrossover`에 수동 컷을 붙여서 다시 screening했습니다.

추가한 새 컷은 다음과 같습니다.

- `state_link`
- `eta_aggregate_load`
- `spot_bridge_lower`
- `spot_bridge_aggregate`
- `eta_cover_general`

그리고 다음 조합을 비교했습니다.

- `combo_barrier_uptime`
- `combo_barrier_delta_gamma`
- `combo_barrier_state_link`
- `combo_barrier_eta_aggregate`
- `combo_barrier_spot_bridge_lower`
- `combo_barrier_eta_cover_general`

해석은 다음과 같습니다.

- `combo_barrier_state_link`가 gap 기준으로 가장 강했습니다. `phi <= u`, `gamma <= u`, `eta <= \sum_t u_{st}` 같은 activation-state linking이 barrier 기반 root 전략과 결합되면서 bound를 크게 올렸습니다.
- `combo_barrier_uptime`는 incumbent와 bound의 균형이 가장 좋았습니다. gap도 많이 줄었고 incumbent 품질도 좋았습니다.
- `combo_barrier_delta_gamma`는 incumbent 개선이 가장 강한 축에 속했습니다.
- `combo_barrier_eta_cover_general`도 의미 있는 개선을 보였지만 `combo_barrier_uptime`보다는 한 단계 아래였습니다.
- `spot_bridge_*` 계열은 기대보다 약했습니다. 현재 인스턴스에서는 `delta_gamma_link`나 `state_link`가 더 직접적으로 작동했습니다.

현재 기준으로 후속 우선순위는 다음과 같습니다.

- gap 축소를 최우선으로 보면 `combo_barrier_state_link`
- incumbent와 gap의 균형을 보면 `combo_barrier_uptime`
- incumbent를 더 낮추는 방향을 보면 `combo_barrier_delta_gamma`
