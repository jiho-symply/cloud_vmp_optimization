# 2604 Cut Experiment

## 목적

이 폴더는 chance-constrained toy 모델에서 cut profile이 실제 풀이 성능에 얼마나 영향을 주는지 분리해서 실험하기 위한 공간입니다.

이번 재실행 기준은 다음과 같습니다.

- 고정 instance: `12VM`, `OD + SP + BJ = 4 + 4 + 4`
- server capacity: `8`
- sampling filter: `avg_cpu_mean >= 20`, `vCPU <= 8`
- scenarios: `10`
- objective: 에너지 최소화
- migration penalty: `0.0`
- `NoRelHeurTime = 0`
- `TimeLimit = 3600`
- job당 `Threads = 8`
- pool 실행, `max-workers`는 기본 `4`

핵심 원칙은 하나의 고정 인스턴스를 먼저 만든 뒤, 같은 인스턴스 위에서 cut profile만 바꿔 비교하는 것입니다. 즉, 이번 비교는 데이터 샘플링 차이가 아니라 cut의 영향만 보도록 구성했습니다.

## 주요 파일

- 모델: [run_model.py](C:\Users\jiho\code\cloud_vmp_optimization\experiments\2604-cut-experiment\run_model.py)
- cut profile 정의: [cut_profiles.py](C:\Users\jiho\code\cloud_vmp_optimization\experiments\2604-cut-experiment\cut_profiles.py)
- 실행기: [run_12vm_cut_pool.py](C:\Users\jiho\code\cloud_vmp_optimization\experiments\2604-cut-experiment\run_12vm_cut_pool.py)
- 데이터 생성: [build_dataset.py](C:\Users\jiho\code\cloud_vmp_optimization\experiments\2604-cut-experiment\build_dataset.py)
- cut 설명: [CUT_NOTES.md](C:\Users\jiho\code\cloud_vmp_optimization\experiments\2604-cut-experiment\CUT_NOTES.md)

## 현재 profile

- `baseline`
- `activation`
- `spot_server_link`
- `spot_time_link`
- `pairwise_cover`
- `triple_cover`
- `uptime_symmetry`
- `solver_cover_focus`
- `solver_clique_focus`
- `solver_implied_focus`
- `solver_lift_focus`
- `builtin_aggressive`
- `combined_light`
- `combined_full`

## 실행

```powershell
.\.venv\Scripts\python.exe .\experiments\2604-cut-experiment\run_12vm_cut_pool.py --max-workers 4 --threads 8 --time-limit 3600
```

## 출력

- instance: `data/processed/2604-cut-experiment/...`
- run logs: `experiments/2604-cut-experiment/results/runs/...`
- summary: `experiments/2604-cut-experiment/results/analysis/cut_12vm_od_sp_bj_cap8_avg20_energy_pool/cut_experiment_summary.csv`

## 추가 실험

- 단일 컷 / 단일 solver 전략만 비교하는 후속 실험은 [run_12vm_cut_lit_search.py](C:\Users\jiho\code\cloud_vmp_optimization\experiments\2604-cut-experiment\run_12vm_cut_lit_search.py)로 실행했습니다.
- 결과 요약은 [SEARCH_REPORT.md](C:\Users\jiho\code\cloud_vmp_optimization\experiments\2604-cut-experiment\results\analysis\cut_12vm_od_sp_bj_cap8_avg20_lit_search\SEARCH_REPORT.md)에 있습니다.
- screening 결과 표는 [screen_summary.csv](C:\Users\jiho\code\cloud_vmp_optimization\experiments\2604-cut-experiment\results\analysis\cut_12vm_od_sp_bj_cap8_avg20_lit_search\screen_summary.csv)입니다.
- 최종 재평가 결과 표는 [final_summary.csv](C:\Users\jiho\code\cloud_vmp_optimization\experiments\2604-cut-experiment\results\analysis\cut_12vm_od_sp_bj_cap8_avg20_lit_search\final_summary.csv)입니다.
- cut과 root 전략을 함께 screening한 조합 실험은 [run_12vm_cut_combo_search.py](C:\Users\jiho\code\cloud_vmp_optimization\experiments\2604-cut-experiment\run_12vm_cut_combo_search.py)로 실행했습니다.
- 조합 실험 요약은 [SEARCH_REPORT.md](C:\Users\jiho\code\cloud_vmp_optimization\experiments\2604-cut-experiment\results\analysis\cut_12vm_od_sp_bj_cap8_avg20_combo_search\SEARCH_REPORT.md)에 있습니다.
- 조합 screening 표는 [screen_summary.csv](C:\Users\jiho\code\cloud_vmp_optimization\experiments\2604-cut-experiment\results\analysis\cut_12vm_od_sp_bj_cap8_avg20_combo_search\screen_summary.csv)입니다.
- 조합 final 표는 [final_summary.csv](C:\Users\jiho\code\cloud_vmp_optimization\experiments\2604-cut-experiment\results\analysis\cut_12vm_od_sp_bj_cap8_avg20_combo_search\final_summary.csv)입니다.

## 최신 관찰

- 이번 라운드에서는 조합 프로파일을 빼고 단일 프로파일만 비교했습니다.
- 가장 강했던 후보는 `strategy_barrier_nocrossover`였습니다.
- 컷 쪽에서는 `uptime_symmetry`와 `delta_gamma_link_only`가 의미 있는 개선을 보였습니다.
- `minimal_cover_general`, `minimal_cover_fixed`는 생성 수는 적당했지만 현재 인스턴스에서는 개선 폭이 작았습니다.
- `Method=2`만 주는 것보다 `Method=2`, `NodeMethod=2`, `Crossover=0` 조합이 훨씬 강했습니다. 현재 병목이 root relaxation 이후 crossover / root 처리 흐름에 있다는 신호로 해석할 수 있습니다.
- 조합 실험에서는 `combo_barrier_state_link`, `combo_barrier_uptime`, `combo_barrier_delta_gamma`가 특히 강했습니다.
- 같은 `1200초` 기준으로 보면 `strategy_barrier_nocrossover` 단독은 gap `61.47%`였고, `combo_barrier_uptime`은 `54.87%`, `combo_barrier_state_link`는 `37.02%`까지 내려갔습니다.
- 다만 `combo_barrier_state_link`는 bound 강화는 매우 강하지만 incumbent는 상대적으로 덜 좋아서, 목표가 gap 축소인지 incumbent 개선인지에 따라 해석을 나눠야 합니다.

## 해석 주의

- `activation`은 순수한 polyhedral cut이라기보다, 현재 목적함수 아래에서 지배되는 idle-server-on 해를 줄이는 dominance 성격이 있습니다.
- `pairwise_cover`, `triple_cover`는 overload를 무조건 금지하는 것이 아니라, 그런 packing이면 `phi`가 필요하다는 점을 relaxation에 더 강하게 반영합니다.
- `solver_*_focus`와 `builtin_aggressive`는 모델 수식을 바꾸는 것이 아니라 Gurobi built-in cut family를 다르게 켜는 비교 실험입니다.
