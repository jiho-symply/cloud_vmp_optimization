# Codex 작업 프롬프트

아래 작업을 clone한 repository root에서 수행하라. 목표는 현재 Notion 모델을 개선하는 것이 아니라, 현재 수식을 실제 데이터에 안전하게 연결하고 다양한 실험으로 모델의 역학과 튜닝 방향을 관찰하는 것이다.

## 절대 원칙

1. 먼저 repository의 `AGENTS.md`, `docs/data_pipeline_handoff_nrel_nyiso_gct.md`, `docs/MODEL_DATA_MAPPING.md`, `src/notion_energy_vmp/README.md`, 그리고 이번에 추가된 `src/notion_energy_vmp/`, `scripts/*experiment*`, `experiments/sweep_plan.yaml`, `configs/toy_200_real.yaml`을 전부 읽어라.
2. `git status --short`로 기존 사용자 변경을 확인하고 보존하라. raw/processed 원자료를 수정·이동·삭제하지 마라.
3. Notion 페이지 `https://app.notion.com/p/395a14fa720380f5bbd8cda62f50fc93`의 최신 내용을 읽을 수 있으면 읽고, `src/notion_energy_vmp/model.py`가 표시된 변수·목적함수·제약을 그대로 옮겼는지 대조하라.
4. 이번 작업에서는 모델을 강화하거나 고치지 마라. 특히 다음을 기본 모델에 추가하지 마라: ESS 충·방전 배타 binary, exact batch-startup upper bound, observed spot preemption 강제, server initial-state constraint, 추가 migration 제한, 추가 grid bound. 필요한 경우 문제점으로 기록만 하라.
5. 실제 파일의 schema가 코드 가정과 다르면 데이터 로더·검증·canonicalization만 최소 수정하라. 수식 변경이 필요해 보이면 수정하지 말고 근거와 영향만 보고하라.
6. 누락값을 임의로 0/평균/forward-fill하지 마라. 현재 정책은 모든 선택 scenario의 active interval에 hourly observation이 완비된 service VM만 표본 후보로 쓰는 것이다. 후보가 부족하면 즉시 중단하고 class별 후보 수와 누락 패턴을 보고하라.

## 1. 설치 위치와 코드 통합 확인

- package: `src/notion_energy_vmp/`
- entrypoint: `scripts/run_notion_energy_experiment.py`
- background runner: `scripts/run_background.sh`
- suite generator/runner/aggregator:
  - `scripts/generate_experiment_suite.py`
  - `scripts/run_experiment_suite.py`
  - `scripts/aggregate_experiment_results.py`
- base config: `configs/toy_200_real.yaml`
- suite plan: `experiments/sweep_plan.yaml`

import가 repository root에서 확실히 되도록 확인하되, 기존 package 구조를 불필요하게 재편하지 마라. 필요한 dependency만 현재 `.venv`에 설치하라. 시스템 Python을 사용하지 마라.

## 2. 실제 데이터 연결 검증

다음 파일을 실제로 읽고 문서와 schema가 일치하는지 확인하라.

- `data/processed/notion_toy_google2019_v1_pool/vm_requests.csv`
- `vm_usage_hourly_scenarios.csv`
- `spot_preemption_scenarios.csv`
- `batch_families.csv`
- `servers.csv`
- `scenario_probabilities.csv`
- `data/processed/nyiso/electricity_price_nyiso_2019_30min.parquet`
- `data/processed/nrel/renewable_power_2019_30min_assumed_100mw.parquet`

다음을 수치로 출력하고 `data_audit.json`에 남겨라.

- 원본/eligible/선택 class 수와 선택 VM ID;
- scenario별·VM별 active-hour coverage와 duplicate/missing count;
- batch family별 원본 `W_k`, 적용 scale, 최종 `W_k`;
- NYISO zone/date별 24-hour 완전성, DA/RT min/mean/max, negative/extreme count;
- NREL site/date별 24-hour 완전성, 원본 energy min/mean/max, 전역 scale과 최종 energy;
- USD/MWh→USD/kWh 및 MWh→kWh 변환 확인;
- scenario probability 합;
- CPU/MEM peak와 단순 server lower bound;
- ESS/spot price/excess penalty/migration coefficient 최종값;
- 모든 source/canonical file SHA-256.

DA는 선택한 같은 10개 historical date의 시간대별 평균으로 1단계 값 하나를 만들고, RT와 renewable은 날짜 하나를 scenario 하나와 순서대로 연결한다. NYISO/NREL timestamp는 `America/New_York` local date로 맞춘다. 음수·극단 가격은 제거하거나 winsorize하지 마라. `P_sell`과 renewable global capacity scaling은 관측값이 아니라 모델 가정임을 명시하라.

## 3. 테스트와 prepare-only 실행

먼저 기존 전체 테스트를 실행하고, 이번 테스트도 실행하라.

```bash
.venv/bin/python -m pytest -q
```

그 다음 40-VM/2-scenario prepare-only와 200-VM/10-scenario prepare-only를 각각 새 run directory에서 실행하라. 기존 결과 폴더를 자동 삭제하지 마라.

```bash
.venv/bin/python -u scripts/run_notion_energy_experiment.py \
  --config configs/toy_200_real.yaml \
  --target-total-vms 40 \
  --num-scenarios 2 \
  --num-servers 6 \
  --prepare-only \
  --run-dir runs/notion_energy_prepare_v40_xi2

.venv/bin/python -u scripts/run_notion_energy_experiment.py \
  --config configs/toy_200_real.yaml \
  --prepare-only \
  --run-dir runs/notion_energy_prepare_v200_xi10
```

각 `checksums.sha256`을 `sha256sum -c`로 검증하고 `data_audit.json`을 읽어 이상이 없을 때만 solve로 넘어가라.

## 4. Gurobi 모델 충실성 및 실행 전 점검

- `gurobipy` import, 버전, 라이선스 상태를 확인하라.
- 라이선스가 허용하는 동시 Gurobi process 수를 확인하라. 12개 동시 실행이 허용되지 않으면 허용 범위까지 `max-workers`를 줄여라. 라이선스 우회는 금지한다.
- 96 logical CPU 전체를 한 모델에 주지 마라. 기본은 12 concurrent runs × 8 Gurobi threads다.
- 각 run은 `SoftMemLimit=32 GiB`, `NodefileStart=4 GiB`, run별 nodefile directory를 사용한다. 12개 기준 최악의 soft limit 합이 384 GiB이므로 OS와 pipeline을 위한 여유가 남는다.
- CPU affinity는 현재 process에 허용된 CPU 집합 안에서 겹치지 않게 배정한다.
- BLAS/MKL/OpenBLAS thread는 run당 1로 제한해 oversubscription을 막는다.

모델 대조 중 발견한 차이는 먼저 `MODEL_FIDELITY_REVIEW.md`에 `Notion 식 / 현재 코드 / 영향 / 수정 여부(기본은 미수정)` 형식으로 기록하라. 데이터 연결에 필수인 오탈자 수준 외에는 모델을 수정하지 마라.

## 5. 작은 solve로 end-to-end 확인

40-VM/2-scenario 모델을 최대 300초 foreground로 먼저 실행하라. Gurobi `solver.log`가 실행 중 갱신되고, `summary.json`, 상세 CSV, `model_validation_diagnostics.json`, incumbent `.sol`이 생성되는지 확인하라.

다음 진단을 반드시 읽어라.

- false batch startup;
- simultaneous ESS charge/discharge;
- observed spot interruption 이후 active;
- observed interruption 이전 endogenous preemption;
- initial server state;
- objective component reconstruction error;
- CVaR expression과 empirical CVaR;
- status/objective/bound/gap/runtime/node count.

진단값이 양수여도 현재 단계에서는 모델을 고치지 마라. 현재 모델의 관찰 결과로 취급한다.

## 6. 58-run 실험 suite 생성

suite는 다음을 포함한다.

- smoke 1;
- VM sample seed 반복 3;
- VM/scenario scale 4;
- `kappa_SLA`, spot discount, `kappa_mig` OFAT 10;
- `alpha`, `epsilon` OFAT 6;
- renewable ratio, ESS capacity OFAT 6;
- winter/spring/summer/autumn 실제 trace 4;
- 7개 인자의 Latin-hypercube interaction 24.

```bash
.venv/bin/python scripts/generate_experiment_suite.py \
  --plan experiments/sweep_plan.yaml \
  --output-dir experiments/generated/notion_model_validation_v1
```

생성 후 manifest run 수가 58인지, run ID가 unique한지, 모든 config가 parse되는지, 각 config의 scenario date 수와 scenario 수가 같은지 검사하라.

## 7. 병렬 suite를 background로 시작

small solve가 정상이고 license가 허용할 때만 실행한다. 아래 controller 자체를 background로 띄운다.

```bash
nohup .venv/bin/python -u scripts/run_experiment_suite.py \
  --manifest experiments/generated/notion_model_validation_v1/manifest.csv \
  --max-workers 12 \
  --threads-per-run 8 \
  > experiments/generated/notion_model_validation_v1/suite_controller.log 2>&1 &
echo $! > experiments/generated/notion_model_validation_v1/suite_controller.pid
```

license가 허용하는 동시 process 수가 12 미만이면 `--max-workers`만 줄이고 `--threads-per-run 8`은 유지하라. 시작 직후 다음을 확인하라.

- controller PID가 살아 있음;
- `suite_status.jsonl`에 RUNNING 상태가 기록됨;
- 실제 child process 수가 worker 상한 이하임;
- 각 child의 CPU affinity가 겹치지 않음;
- 각 run의 `console.log`와 `solver.log`가 갱신됨;
- 즉시 license error/OOM/path/schema failure가 반복되지 않음.

전체 suite가 끝날 때까지 현재 Codex 세션에서 기다릴 필요는 없다. 다만 최소 smoke/첫 batch가 실제 optimization에 진입했음을 확인한 뒤 종료하라.

## 8. 결과 집계 방법 준비

완료 후 실행할 명령을 검증하라.

```bash
.venv/bin/python scripts/aggregate_experiment_results.py \
  --manifest experiments/generated/notion_model_validation_v1/manifest.csv
```

집계 시 OPTIMAL과 TIME_LIMIT를 분리하고, TIME_LIMIT는 incumbent objective만 비교하지 말고 bound/gap을 같이 사용한다. VM 수가 다른 run의 raw objective를 직접 비교하지 않는다. 먼저 seed 반복으로 sample variability를 보고, OFAT의 방향·threshold를 본 뒤, seasonal dependence와 LHS interaction을 해석한다. 10개 동일확률 scenario에서 `alpha=0.95` CVaR는 사실상 worst sampled scenario에 지배됨을 명시한다.

## 최종 보고 형식

작업이 끝나면 다음만 간결하게 보고하라.

1. 수정/추가한 파일;
2. 실제 schema 차이와 데이터 로더 수정 내용;
3. 40/2 및 200/10 data-audit 핵심 수치;
4. small solve status/objective/bound/gap와 네 가지 formulation diagnostic;
5. suite run 수, worker/thread/memory 설정, controller PID;
6. live monitoring 명령;
7. 아직 해결하지 않은 모델 수식상의 경고. 모델을 임의로 수정하지 않았음을 확인하라.
