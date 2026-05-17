# Branch-and-Cut 개선 조사 메모

## 목적

이 문서는 `12VM / OD+SP+BJ / cap=8 / avg20 / energy objective` 벤치마크에서 branch-and-cut 계열 실험이 유의미한 성과를 내지 못한 이유를 정리하고, 관련 문헌에서 가져올 수 있는 개선 아이디어를 최대한 넓게 수집한 메모이다.

## 현재 구현에서 드러난 문제

현재 실험 결과를 보면 branch-and-cut 계열은 설정 실험이나 column-generation warm start보다 일관되게 약했다.

- `bc_server_branch`: 목적값은 나쁘지 않지만 gap 개선은 미미했다.
- `bc_general_cover_usercuts`, `bc_eta_usercuts`, `bc_spot_usercuts`: `callback_cut_counts`가 비어 있어, 실질적으로 분리된 user cut이 거의 없었다.
- `state_link + phi_scenario_mass` 같은 정적 강화는 의미가 있었지만, 동적 separation은 거의 힘을 쓰지 못했다.

현재 구현이 약한 이유는 다음 네 가지로 요약된다.

1. 컷이 `phi`, `gamma` 같은 보조 변수 수준에 머무르고, 실제 chance-budget 역할을 하는 `eta`, `delta` 쪽으로 충분히 투영되지 않았다.
2. 대부분의 컷이 단일 서버-시점 row에서 만든 `minimal cover` 수준이라 다중 row 상호작용을 거의 못 잡는다.
3. 모든 fractional node에서 separation을 시도하는 구조인데, 컷 선택, 병렬성, 효율성, cut persistence 같은 운영 전략이 없다.
4. big-M 완화가 남아 있어, 컷이 추가되더라도 relaxation에 주는 충격이 약하다.

## 현재 모델에 대응되는 핵심 구조

문헌과 연결할 때 현재 모델의 구조는 다음처럼 보는 것이 가장 자연스럽다.

- `u[s,t]`, `u_used[s]`: fixed-charge facility opening / server activation
- `x[i,s,t]`, `y[j,s]`, `z[k,s,t]`: assignment / packing / scheduling
- `m[i,t]`: time-expanded path / migration arc
- `a[j,s,t,xi]`: 시나리오별 spot service activity
- `phi[s,t,xi]`, `gamma[s,t,xi]`: 시나리오별 overload / suspension 보조 변수
- `eta[s,xi]`, `delta[j,xi]`: chance-budget를 직접 소비하는 violation indicator

따라서 문헌에서 강하게 가져와야 할 대상은 `eta`, `delta`, 그리고 `u-x-y-z`를 함께 묶는 강한 linking / cover / mixing 구조다.

## 문헌에서 바로 연결되는 아이디어

### 1. Quantile cut

참고:

- Xie and Ahmed, [On quantile cuts and their closure for chance constrained optimization problems](https://optimization-online.org/wp-content/uploads/2016/03/5367.pdf)

핵심:

- quantile cut은 chance-constrained 문제의 원래 `x` 공간에서 추가하는 컷이다.
- mixing inequality를 확장 변수 공간에서 쓰지 않고, 원 공간으로 투영한 형태로 볼 수 있다.
- 저자들은 quantile closure가 유한한 형태로 기술될 수 있고, linear CCP에서는 polyhedral하다고 보인다.

우리 모델로의 변형:

- 현재는 `eta[s,xi]`, `delta[j,xi]`를 두고 extended formulation으로 풀고 있다.
- 여기에 대응하는 quantile cut을 `u`, `x`, `y`, `z` 중심의 원공간으로 투영하면, `phi`나 `gamma`를 직접 자르지 않고도 chance budget을 더 강하게 표현할 수 있다.
- 특히 `server s`에 대해 여러 시나리오의 `load(s,t,xi)` 요약값을 만든 뒤, 각 시점 또는 시점 묶음에 대해 quantile threshold를 생성하는 방식이 유망하다.

평가:

- 장점: `eta`, `delta` projection과 잘 맞는다.
- 단점: separation이 일반적으로 어렵다.
- 우선순위: 높음

### 2. Mixing set / joint mixing set / blending inequality

참고:

- Küçükyavuz, [On mixing sets arising in chance-constrained programming](https://doi.org/10.1007/s10107-010-0385-3)
- Zhang, Küçükyavuz, Goel, [A Branch-and-Cut Method for Dynamic Decision Making Under Joint Chance Constraints](https://doi.org/10.1287/mnsc.2013.1822)
- A Polyhedral Study on Chance Constrained Program with Random Right-Hand Side, [Optimization Online PDF](https://optimization-online.org/wp-content/uploads/2016/05/5425.pdf)
- Joint chance-constrained programs and the intersection of mixing sets, [Optimization Online PDF](https://optimization-online.org/wp-content/uploads/2019/10/7404.pdf)

핵심:

- finite-scenario chance constraint reformulation은 mixing set과 knapsack 구조를 반복적으로 포함한다.
- 단일 mixing set 컷보다 여러 mixing set의 상호작용을 잡는 blending inequality가 더 강할 수 있다.
- 실제 계산 실험에서는 root에서 blending inequality를 넣고, 그 뒤에는 주기적으로만 강화하는 방식이 잘 작동했다.

우리 모델로의 변형:

- 현재 각 `(s,t)`마다 `phi[s,t,xi] <= u[s,t]`와 `sum p_xi phi[s,t,xi] <= epsilon_od` 같은 단일 row 성질만 사용하고 있다.
- 이를 `server s` 전체 horizon 또는 `server s`의 여러 시점 묶음으로 확장해, 여러 `(s,t)` 행이 공유하는 `u[s,t]`, `u_used[s]`, `eta[s,xi]`를 같이 보는 joint mixing cut을 만들 수 있다.
- 특히 `eta[s,xi] <= sum_t phi[s,t,xi]` 구조를 이용해, `phi`들을 직접 자르지 말고 `eta`를 향해 투영한 blending cut을 root node에서 분리하는 방향이 좋다.

평가:

- 장점: 지금 모델의 chance 구조에 가장 직접적으로 맞는다.
- 단점: 수식 도출과 separation 구현이 다소 무겁다.
- 우선순위: 매우 높음

### 3. Probabilistic cover / lifted probabilistic cover

참고:

- Song, Luedtke, Küçükyavuz, [Chance-Constrained Binary Packing Problems](https://optimization-online.org/2012/10/3639/)
- Han, Song, Küçükyavuz, [Lifting of probabilistic cover inequalities](https://doi.org/10.1016/j.orl.2017.08.006)

핵심:

- deterministic knapsack의 cover inequality를 chance-constrained packing으로 확장한 것이 probabilistic cover다.
- 단순 probabilistic cover뿐 아니라 approximate lifting까지 하면 계산 성능이 더 좋아진다는 결과가 있다.
- Song 등은 분리된 probabilistic cover / pack inequality를 여러 relaxation 해에 대해 유지했다가, 일정 횟수 동안 위반이 없을 때만 버리는 전략도 사용했다.

우리 모델로의 변형:

- 각 `(s,t)`의 server capacity row에서 `OD + BJ + active SP` 조합을 하나의 packing row로 본다.
- `eta[s,xi]` 또는 `delta[j,xi]`를 포함하는 probabilistic cover를 만들고, 단순 minimal cover 다음에 heuristic lifting을 수행한다.
- 현재 callback은 한 번의 fractional 해만 보고 즉시 cut을 버리는 구조에 가까운데, 문헌처럼 여러 relaxation 해를 통해 살아남는 cut pool을 유지하는 편이 낫다.

평가:

- 장점: 현재 cover 기반 구현을 가장 자연스럽게 고칠 수 있다.
- 단점: `phi` slack가 크면 단순 cover는 약해진다. lift 또는 projection이 꼭 필요하다.
- 우선순위: 매우 높음

### 4. Branch-and-cut decomposition

참고:

- Luedtke, [A branch-and-cut decomposition algorithm for solving chance-constrained mathematical programs with finite support](https://jrluedtke.github.io/papers/luedtke-decomp-cc-mpa-vol146-2014.pdf)
- Zeng, An, Kuznia, [Chance Constrained Mixed Integer Program: Bilinear and Linear Formulations, and Benders Decomposition](https://arxiv.org/abs/1403.7875)
- Codato and Fischetti, [Combinatorial Benders’ cuts for mixed-integer linear programming](https://doi.org/10.1287/opre.1060.0286)

핵심:

- chance-constrained MILP는 시나리오별 서브문제를 풀고, 그 결과를 strong valid inequality로 master에 돌려주는 분해 방식이 강하다.
- 단순 Benders보다 integer programming 기술과 결합된 branch-and-cut decomposition이 더 잘 작동할 수 있다.

우리 모델로의 변형:

- master: `u`, `u_used`, `x`, `y`, `z`, `m`
- subproblem: 각 시나리오별 `a`, `phi`, `gamma`, `eta`, `delta`
- scenario 서브문제가 infeasible 또는 높은 violation budget을 요구하면, 그 근거를 이용해 `eta/delta` 또는 `u-x-y-z`에 대한 combinatorial cut을 추가한다.

평가:

- 장점: 현재 decomposition 실험 흐름과 자연스럽게 이어진다.
- 단점: 구현 난이도가 높고, 서브문제 certificate 설계가 필요하다.
- 우선순위: 높음

### 5. Flow cover / weak flow cover

참고:

- Lejeune and Ruszczyński 계열의 fixed-charge flow 문헌 대신, 여기서는 직접적으로
- [Weak flow cover inequalities for the capacitated facility location problem](https://www.sciencedirect.com/science/article/abs/pii/S0377221720306445)
- Atamtürk, Küçükyavuz, Tezel, [Path Cover and Path Pack Inequalities for the Capacitated Fixed-Charge Network Flow Problem](https://optimization-online.org/2016/11/5722/)
- Atamtürk, Gómez, Küçükyavuz, [Three-Partition Flow Cover Inequalities for Constant Capacity Fixed-Charge Network Flow Problems](https://atamturk.ieor.berkeley.edu/pubs/_published/tpf-inequalities.pdf)

핵심:

- facility opening + flow assignment가 동시에 있는 문제에서 flow cover, path cover, path pack이 매우 강하다.
- weak flow cover는 min-cut 기반 separation이 가능하고, path cover / path pack은 single-node flow cover보다 강한 일반화다.

우리 모델로의 변형:

- 각 시점 `t`에서 `u[s,t]`는 facility opening, `x/y/z`는 capacity를 소비하는 flow로 볼 수 있다.
- migration까지 고려하면 on-demand VM의 서버 이동 경로는 time-expanded path 구조를 만든다.
- 따라서 단순 cover 대신 `time window` 전체를 하나의 path 또는 layered network로 보고 `path cover`류 컷을 만드는 것이 더 자연스럽다.

평가:

- 장점: `u-x-z` coupling 강화에 매우 적합하다.
- 단점: spot recourse가 scenario-dependent라 직접 이식은 약간의 변형이 필요하다.
- 우선순위: 높음

### 6. Lifted cover / extended cover for binary-integer knapsack

참고:

- Kaparis and Letchford, [Cover Inequalities for Binary-Integer Knapsack Constraints](https://optimization-online.org/wp-content/uploads/2004/02/830.pdf)
- lifting 관련 후속 문헌과 deterministic knapsack cover 문헌

핵심:

- binary와 integer가 섞인 knapsack row에서는 단순 minimal cover보다 lifted / extended cover가 강하다.
- 커버를 줄이고, coefficient를 올리고, extension을 붙이는 과정이 relaxation을 더 많이 자른다.

우리 모델로의 변형:

- 현재 `load <= C u + M phi`는 본질적으로 binary-integer knapsack 완화다.
- `u[s,t]`, `x[i,s,t]`, `z[k,s,t]`, `a[j,s,t,xi]`가 함께 들어간 row에서 extended cover를 만들 수 있다.
- 특히 `u[s,t] = 1` 근처에서 분수 `x/z/a`가 여러 개 동시에 들어가는 해를 잘라내는 데 적합하다.

평가:

- 장점: 현재 minimal cover보다 분명히 한 단계 강하다.
- 단점: 구현은 가능하지만, `phi`를 함께 둔 상태에서는 얼마나 강하게 작동할지 검증이 필요하다.
- 우선순위: 중간 이상

### 7. Facility location / assignment polyhedral cuts

참고:

- [Capacitated Facility Location: Valid Inequalities and Facets](https://pubsonline.informs.org/doi/10.1287/moor.20.3.562)
- generalized assignment / capacitated facility location 류 polyhedral 문헌

핵심:

- 시설 개방과 고객 배정이 함께 있는 문제는 submodular inequality, capacity cut, implied demand cover, assignment lifting이 강하다.

우리 모델로의 변형:

- 현재 `server s at time t`를 facility로 보고, active VM demand를 고객으로 보면 CFL/GAP 구조가 된다.
- `x`만이 아니라 `y`, `z`까지 포함해 demand cover / implied demand cover를 만들 수 있다.

평가:

- 장점: 문제 구조상 매우 익숙한 family라 구현이 쉽다.
- 단점: 시나리오 recourse를 직접 타격하지는 못한다.
- 우선순위: 중간

### 8. Path-based / set-partitioning에서 가져오는 컷

참고:

- [Subset-Row Inequalities and Unreachability in Path-based Formulations for Routing and Scheduling Problems](https://logistik.bwl.uni-mainz.de/files/2023/07/LM-2023-04.pdf)
- branch-price-and-cut 문헌 일반

핵심:

- path master에서는 subset-row inequality가 root bound 강화에 자주 쓰인다.
- Ryan-Foster branching, path incompatibility, route/path subset-row가 대표적이다.

우리 모델로의 변형:

- on-demand VM 하나의 전체 horizon 배치를 하나의 path column으로 보면, 둘 이상의 “위험한 시점/서버 조합”을 포함하는 path를 제한하는 SRI류 컷을 만들 수 있다.
- compact model에 직접 옮기려면, 특정 서버-시간 집합을 동시에 많이 방문하는 migration path를 막는 path incompatibility cut 형태로 약화해 넣을 수 있다.

평가:

- 장점: migration 구조를 직접 겨냥할 수 있다.
- 단점: compact model로 옮길 때는 수식 번역이 필요하다.
- 우선순위: 중간

## solver 운영 관점에서의 개선 포인트

참고:

- Gurobi callback 문서: [Callbacks](https://docs.gurobi.com/projects/optimizer/en/current/features/callbacks.html)
- Gurobi parameter 목록: [Optimizer Manual PDF](https://docs.gurobi.com/_/downloads/optimizer/en/current/pdf/)
- SCIP cut selector 문서: [SCIP Cut Selectors](https://www.scipopt.org/scip/doc/html/group__CUTSELECTORS.php)
- chance-constrained polyhedral study: [Optimization Online PDF](https://optimization-online.org/wp-content/uploads/2016/05/5425.pdf)
- probabilistic cover 운영 아이디어: [Chance-Constrained Binary Packing Problems](https://optimization-online.org/wp-content/uploads/2012/10/3639.pdf)

중요한 운영 교훈은 다음과 같다.

1. 모든 node에서 separator를 돌리는 것이 항상 좋은 것은 아니다.
   - mixing / blending 문헌에서는 root에서 강한 컷을 넣고, 이후에는 `매 100 nodes`처럼 주기적으로 분리하는 편이 더 좋았다고 보고한다.
2. cut selection이 필요하다.
   - SCIP는 efficacy, parallelism, integer support, density를 섞어서 cut을 고른다.
   - 현재 구현은 “후보가 있으면 바로 추가”에 가까워서, 실제로는 solver가 선호할 만한 컷을 선별하지 못한다.
3. cut persistence가 필요하다.
   - probabilistic cover 문헌은 당장 위반되지 않는 cut도 여러 relaxation 해에 걸쳐 유지했다가 버린다.
   - 지금처럼 한 번 검사하고 끝내면 좋은 cut을 놓칠 수 있다.
4. callback cut이 전혀 안 들어간다면 separation 대상이 약하거나 중복일 가능성이 높다.
   - 즉 “더 자주 넣는 것”보다 “다른 변수 공간으로 투영한 stronger cut”이 필요하다.
5. built-in cut과 custom cut은 경쟁한다.
   - Gurobi에는 `CoverCuts`, `FlowCoverCuts`, `MasterKnapsackCuts`, `MixingCuts`, `MIRCuts`, `StrongCGCuts`, `ZeroHalfCuts` 등이 있다.
   - custom cut이 builtin보다 약하면, solver는 이미 충분히 비슷한 강화를 했을 가능성이 있다.

## 지금 문제에서 가장 유망한 후보군

### 1순위

- `eta/delta`를 직접 겨냥하는 quantile cut
- joint mixing / blending inequality
- probabilistic cover + lifted probabilistic cover

이 세 가지는 지금 문제의 chance-budget 핵심을 직접 건드린다.

### 2순위

- root-only 또는 periodic separation으로 branch-and-cut 운영 방식을 재설계
- flow cover / path cover를 server-time path 구조에 맞게 변형
- extended/lifted cover로 기존 minimal cover 강화

이들은 현재 구현을 크게 뒤엎지 않고도 개선 가능하다.

### 3순위

- branch-and-cut decomposition
- path-based incompatibility / subset-row 계열
- facility-location derived demand cover

효과는 있을 수 있지만 구현량이 크다.

## 바로 시도할 만한 실험 설계

### 실험 A: quantile / mixing root cut

- 목적: `phi` 대신 `eta` 중심의 original-space strengthening이 실제로 root bound를 올리는지 확인
- 구현:
  - root relaxation에서만 quantile cut 생성
  - server별, time-window별, scenario-group별 cut 생성
  - 이후에는 separation 중단

### 실험 B: probabilistic cover with persistence

- 목적: 현재 minimal cover callback이 안 먹는 문제를 cut pool 유지로 해결 가능한지 확인
- 구현:
  - 위반이 약한 cover도 최대 `K`회 relaxation 동안 보관
  - 다시 위반되면 활성화
  - 계속 안 위반되면 제거

### 실험 C: lifted cover on deterministic server-time rows

- 목적: `u-x-z` coupling 강화가 gap을 낮추는지 확인
- 구현:
  - `(s,t)` row에서 deterministic part `OD + BJ` 중심 lifted cover 추가
  - `SP`는 aggregate 또는 conditional 형태로만 포함

### 실험 D: blending root + periodic star cut

- 목적: 모든 node 분리 대신 literature에 가까운 schedule을 적용
- 구현:
  - root에서 blending
  - 이후 50~100 nodes마다만 star / mixing separation

### 실험 E: path-cover style cut for migration

- 목적: on-demand migration path의 분수해를 줄이는지 확인
- 구현:
  - 특정 time window와 server subset에 대해 path incompatibility / path-cover 유사 컷 추가

## 결론

지금 branch-and-cut을 살리려면 “cover를 더 많이 넣는 것”이 아니라 아래 두 가지 방향 전환이 필요하다.

1. `phi/gamma` 위주의 국소 컷에서 벗어나 `eta/delta` 중심의 chance-budget 컷으로 올라갈 것
2. 모든 node separation에서 벗어나 `root 강화 + 주기적 분리 + cut selection/persistence` 운영으로 바꿀 것

문헌 기준으로 가장 유망한 조합은 다음과 같다.

- `quantile / mixing / blending` 중 하나를 root에서 사용
- `probabilistic cover + lifting`을 보조 family로 사용
- `flow/path cover`는 migration 또는 server-time 집계 구조에 제한적으로 사용
- separation은 root-only 또는 periodic로 제한

다음 단계로는 위 후보군을 6~8개로 압축해 `screening benchmark`를 설계하는 것이 가장 합리적이다.
