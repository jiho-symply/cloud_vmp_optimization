# 실험 아이디어 정리

## 유망 세팅 6개를 고른 이유

- `state_link`는 기존 cut 실험에서 가장 강한 gap 축소를 보였습니다.
- `phi_scenario_mass`는 시나리오 수가 10개이고 `epsilon_od=0.1`인 현재 인스턴스에서 사실상 overload 허용 시나리오 수를 직접 제한하는 역할을 합니다.
- `barrier + no crossover`는 이전 screening에서 단독으로도 의미 있는 개선을 보였습니다.
- `uptime_symmetry`는 상황에 따라 도움이 있었지만 항상 필요한지는 불분명해서, 넣은 버전과 뺀 버전을 같이 봅니다.
- `representative_peak` hint는 이전 decomposition 결과에서 incumbent 생성 쪽에서 가장 안정적이었습니다.
- `kwon_threshold_mean`은 rule 기반 초기해 중 가장 해석 가능성이 높고, 실제로 완전히 나쁘지 않았습니다.

## Branch-and-Cut 아이디어

### 1. Server-first branching

- `u_used`, `u`, `y`를 먼저 가지치기해서 서버 on/off 구조를 빠르게 확정합니다.

### 2. Budget-first branching

- `eta`, `delta`, `phi`, `gamma`를 먼저 branch해서 어떤 서버/시나리오에서 violation을 허용하는지를 먼저 굳힙니다.

### 3. General cover user cuts

- 정적으로 다 넣지 않고 fractional node에서 위반된 general minimal cover만 분리합니다.

### 4. Eta fixed-cover user cuts

- `OD + BJ`에 대한 cover를 `eta`와 직접 연결해 overload 허용 시나리오 구조를 더 빨리 강제합니다.

### 5. Spot bridge user cuts

- `a >= y - gamma - delta`, `delta >= y + gamma - 1`를 fractional 위반 시에만 잘라서 `spot activity`와 `server suspension`을 더 직접적으로 묶습니다.

### 6. Budget-first + general cover

- 가장 구조적인 branching과 가장 직접적인 cover 분리를 함께 써서 root 이후 tree 품질을 보려는 조합입니다.

## Column-Generation 계열 아이디어

### 1. Mean restricted master

- `representative_mean`, `greedy_rule_mean`, constant server path를 path pool로 사용합니다.

### 2. Peak restricted master

- `representative_peak`를 anchor로 잡아 더 보수적인 path pool을 씁니다.

### 3. Mixed restricted master

- `mean`, `peak`, `rolling`, `threshold`에서 나온 path를 한 번에 pool에 넣습니다.

### 4. Priced restricted master

- mixed pool에서 시작한 뒤, congestion penalty 기반 heuristic pricing으로 새 path를 추가합니다.

## 해석 원칙

- branch-and-cut 계열은 “bound를 얼마나 빨리 올리는지”를 더 중요하게 봅니다.
- column-generation 계열은 “좋은 first-stage seed를 얼마나 빨리 만드는지”를 더 중요하게 봅니다.
- 유망 세팅 6개는 실제 incumbent와 gap 둘 다를 보고 종합 판단합니다.

## Subagent 검토 반영 메모

- branch-and-cut 쪽에서는 컷 종류를 과도하게 늘리는 것보다 `state_link`, `phi_scenario_mass`, `delta-gamma`처럼 first-stage 구조를 직접 묶는 얇은 컷이 더 유효하다는 피드백을 받았습니다.
- branching은 recourse 변수보다 서버 활성화나 violation budget 쪽을 먼저 가르는 편이 더 자연스럽다는 검토가 나왔습니다.
- column-generation 계열 4개는 full branch-price가 아니라 restricted master warm start 실험으로 해석하는 것이 맞고, 핵심 평가지표도 bound보다는 초기 feasible solution 생성 속도와 incumbent 개선 속도라는 의견을 반영했습니다.
- 실험 표를 읽을 때는 `control baseline`, `best baseline`, `branch-and-cut`, `column-generation start`를 분리해서 보는 것이 좋습니다. 한 줄 랭킹만 보면 해석이 흐려질 수 있습니다.
