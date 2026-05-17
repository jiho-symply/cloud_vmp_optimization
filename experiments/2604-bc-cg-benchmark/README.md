# 2604 Branch-and-Cut / Column-Generation Benchmark

## 목적

이 폴더는 기존 `2604-cut-experiment`, `2604-decomposition-experiment` 결과를 바탕으로,

- 잘 나올 가능성이 높은 세팅 6개
- branch-and-cut 계열 실험 6개
- column-generation 계열 실험 4개

를 같은 `12VM / OD+SP+BJ / cap=8 / avg_cpu_mean>=20 / scenario=10 / energy objective` 인스턴스 위에서 비교하기 위한 공간입니다.

## 실험 구성

### 1. 유망 세팅 6개

- `set_state_link_plain`
- `set_state_phi_plain`
- `set_combo_state_phi_barrier`
- `set_combo_state_uptime_phi_barrier`
- `set_rep_peak_hint_state_phi`
- `set_kwon_threshold_hint_state_phi`

### 2. Branch-and-Cut 6개

- `bc_server_branch`
- `bc_budget_branch`
- `bc_general_cover_usercuts`
- `bc_eta_usercuts`
- `bc_spot_usercuts`
- `bc_budget_general_cover_combo`

### 3. Column-Generation 계열 4개

- `cg_mean_rmp_start`
- `cg_peak_rmp_start`
- `cg_mixed_rmp_start`
- `cg_priced_rmp_start`

## 구현 메모

- branch-and-cut 계열은 기존 compact model 위에
  - branch priority
  - user cut callback
  를 추가하는 방식입니다.
- column-generation 계열은 완전한 branch-and-price가 아니라,
  - on-demand VM trajectory path pool
  - restricted master
  - heuristic pricing
  으로 시작해를 만든 뒤, 그 결과를 full MIP에 `start/hint`로 주입하는 방식입니다.

## 실행

직접 실행:

```powershell
.\.venv\Scripts\python.exe .\experiments\2604-bc-cg-benchmark\run_12vm_bc_cg_benchmark.py --time-limit 10800 --seed-time-limit 1200 --max-workers 4 --threads 8 --strategy-threads 4
```

백그라운드 실행:

```powershell
.\experiments\2604-bc-cg-benchmark\launch_12vm_bc_cg_benchmark.ps1
```

## 출력

- 인스턴스: `data/processed/2604-bc-cg-benchmark/...`
- 실행 로그: `experiments/2604-bc-cg-benchmark/results/runs/...`
- 진행 요약: `experiments/2604-bc-cg-benchmark/results/analysis/chance_2sp_toy_12vm_bc_cg_t10800/live_summary.csv`
- 최종 요약: `experiments/2604-bc-cg-benchmark/results/analysis/chance_2sp_toy_12vm_bc_cg_t10800/final_summary.csv`
- 실험 카탈로그: `experiments/2604-bc-cg-benchmark/results/analysis/chance_2sp_toy_12vm_bc_cg_t10800/experiment_catalog.csv`
