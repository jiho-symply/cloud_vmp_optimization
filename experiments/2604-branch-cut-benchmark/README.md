# 2604 Branch-and-Cut Benchmark

이 폴더는 `12VM / OD+SP+BJ / cap=8 / avg20 / energy objective` 인스턴스에 대해 branch-and-cut 계열 후보 8개를 비교하는 벤치마크를 담는다.

## 벤치마크 구성

- `bc00_control_state_phi`
  - 동적 user cut 없이 `state_link + phi_scenario_mass`만 사용하는 기준선
- `bc01_static_state_phi_eta_general`
  - 기준선 위에 `eta general cover`를 정적으로 추가
- `bc02_static_state_phi_eta_fixed`
  - 기준선 위에 `eta fixed cover`를 정적으로 추가
- `bc03_static_state_phi_delta_mass`
  - 기준선 위에 `delta mass linking`을 정적으로 추가
- `bc04_static_state_phi_aggregate`
  - 기준선 위에 `aggregate fixed-load` 컷을 정적으로 추가
- `bc05_branch_server_first`
  - 컷은 기준선과 같게 두고 `server-first branching priority`만 적용
- `bc06_root_eta_lifted`
  - `eta_lifted` user cut을 root에서만 분리
- `bc07_periodic_hybrid_server`
  - `server-first branching`과 `eta_lifted / eta_general / eta_mass / eta_mass_window`를 함께 쓰는 혼합형

## 실행 정책

- 기본 시간 제한: `10800`초
- 병렬도: `4`개 프로세스
- 각 프로세스 thread 수: `8`
- 동적 callback 프로파일은 `PreCrush=1`을 사용한다.
- root separation은 불필요한 반복을 줄이기 위해 pass 수를 제한한다.

## 실행

```powershell
.\experiments\2604-branch-cut-benchmark\launch_12vm_branch_cut_benchmark.ps1
```

직접 실행할 때는 아래 명령을 사용하면 된다.

```powershell
.\.venv\Scripts\python.exe .\experiments\2604-branch-cut-benchmark\run_12vm_branch_cut_benchmark.py --time-limit 10800 --max-workers 4 --threads 8
```

## 출력 위치

- 분석 요약: `experiments/2604-branch-cut-benchmark/results/analysis/chance_2sp_toy_12vm_branch_cut_round2_t10800`
- 개별 실행 로그: `experiments/2604-branch-cut-benchmark/results/runs/chance_2sp_toy_12vm_branch_cut_round2_t10800`
