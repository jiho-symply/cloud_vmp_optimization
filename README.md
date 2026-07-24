# cloud_vmp_optimization

이 저장소는 Virtual Machine Placement(VMP) 문제를 실험하기 위한 연구용 repository입니다. Azure trace 전처리 코드, 여러 toy/prototype 최적화 모델, 그리고 실험별 결과 묶음을 함께 관리합니다.

실험 코드는 `experiments/` 아래에 있고, 가공된 입력 데이터는 `data/processed/` 아래에 정리되어 있습니다. 각 실험은 자신만의 `results/` 폴더를 내부에 두어 결과 그림과 요약 파일을 함께 보관합니다.

가장 최근의 Google/NYISO/NREL 연계 Notion energy-aware VMP 코드, 실험 설정, 실행 문서는 `experiments/2607-notion-energy-vmp/` 아래에 정리되어 있습니다. 대용량 실행 결과는 Git에서 제외하며, 기존 루트 실행 경로는 호환 심볼릭 링크로 유지됩니다.

## 데이터 다운로드와 재생성

`data/processed/`와 각 실험의 `results/`는 원칙적으로 Git에 올리지 않습니다. 다만 server-min migration 실험 재현에 필요한 작은 `notion_toy_google2019_micro_stress_v1` fixture는 예외로 추적합니다. 새 환경에서는 아래 순서로 원본 Azure trace를 내려받고, 필요한 processed 데이터를 다시 만들면 됩니다.

### 1. 환경 준비

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Azure Public Dataset V2 다운로드

빠른 확인용 metadata와 기본 trace table만 받을 때:

```powershell
.\.venv\Scripts\python.exe .\scripts\download_azure_dataset.py --profile analysis
```

전처리까지 재현하려면 `vm_cpu_readings` shard도 필요합니다. 전체는 크므로 먼저 일부 shard로 다운로드 동작만 확인할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe .\scripts\download_azure_dataset.py --profile prepare-dataset --cpu-files 1-10
```

실험 데이터를 안정적으로 재현하려면 모든 shard를 받습니다. `rebuild_sample_data.py`는 필요한 시간 구간을 읽으면 멈추지만, shard가 너무 적으면 후보 VM이 부족할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe .\scripts\download_azure_dataset.py --profile prepare-dataset --cpu-files all
```

### 3. 기본 processed 데이터 생성

2604/2605 계열 실험은 먼저 공통 샘플 데이터가 필요합니다.

```powershell
.\.venv\Scripts\python.exe .\experiments\2604-chance-2sp-toy\rebuild_sample_data.py --profile 2601-initial-toy-model
```

그 다음 chance-constrained 2SP toy instance를 생성합니다.

```powershell
.\.venv\Scripts\python.exe .\experiments\2604-chance-2sp-toy\build_dataset.py --instance-name chance_2sp_toy_24vm_combination_od_sp_bj_od8_sp8_bj8_sc10_cap8_avg20_lam010 --on-demand-count 8 --spot-count 8 --batch-count 8 --max-vcpu 8 --min-avg-cpu 20 --server-capacity 8 --scenario-seed 42 --lambda-migration 0.1
```

2605 Notion VM type 모델용 JSON은 위 2604 instance를 만든 뒤 생성합니다.

```powershell
.\.venv\Scripts\python.exe .\experiments\2605-vm-type-modeling-1\prepare_data.py
```

생성된 파일들은 `data/processed/` 아래에 저장되며, 위 micro-stress fixture를 제외하면 Git에는 추적되지 않습니다.

## Google ClusterData 2019 VM-like toy pipeline

This is a VM-like toy dataset derived from Google ClusterData 2019 Borg instance traces, not a real public-cloud VM trace.

새 toy pipeline은 Google ClusterData 2019 BigQuery public dataset에서 cell `a`의 하루치
Borg instance usage와 lifecycle/machine history를 제한적으로 추출한다. BigQuery 후보 추출은
기존 `(collection_id, instance_index)` deterministic hash sampling을 유지하지만, 전처리 이후
model-facing VM의 단위는 lifecycle execution을 구분하는
`(collection_id, instance_index, episode_index)`이다. Python 실행은 이 repository의
`.venv`를 사용한다.

```bash
.venv/bin/python -m pip install -r requirements.txt

.venv/bin/python scripts/01_extract_google2019.py \
  --project_id YOUR_GCP_PROJECT \
  --cell a \
  --day_index 0 \
  --max_instances 5000 \
  --seed 42 \
  --output_dir data/raw/google2019_cell_a_day0_cpu_distribution

.venv/bin/python scripts/02_build_toy_instance.py \
  --raw_dir data/raw/google2019_cell_a_day0_cpu_distribution \
  --output_dir data/processed/notion_toy_google2019_v5_mode_capacity_bounded \
  --num_servers 6 \
  --seed 42

.venv/bin/python scripts/03_validate_outputs.py \
  --output_dir data/processed/notion_toy_google2019_v5_mode_capacity_bounded \
  --write_summary
```

`01_extract_google2019.py`는 `instance_usage`, `instance_events`, `collection_events`,
`machine_events`를 대상으로 먼저 BigQuery dry-run을 수행해 estimated bytes processed를 출력한
뒤 실제 query를 실행한다. Trace 시작 offset 600초를 포함한 day window와 겹치는 모든
positive-duration usage interval을 보존하며 300초 미만 partial interval도 버리지 않는다.
Usage에는 `machine_id`, `maximum_usage`, 11개 percentile의
`cpu_usage_distribution`을 함께 저장한다. Top-level instance 조건, 최소 usage row 12개,
seed를 포함한 FARM_FINGERPRINT 정렬과 `max_instances` 제한은 기존 방식 그대로다.

생성 산출물은 `data/processed/notion_toy_google2019_v5_mode_capacity_bounded/` 아래의
`metadata.json`, `preprocessing_diagnostics.json`, `scaling_diagnostics.json`,
`model_params.json`, `servers.csv`, `vm_requests.csv`, `vm_usage_5min_scenarios.csv`,
`vm_usage_hourly_scenarios.csv`, `spot_preemption_scenarios.csv`, `batch_families.csv`,
`batch_workload.csv`, `energy_scenarios.csv`, `scenario_probabilities.csv`,
`toy_instance_summary.md`이다. Notion optimization model의 기본 workload 입력은 hourly file인
`vm_usage_hourly_scenarios.csv`이며, 5-minute scenario file은 trace audit과 재집계용으로
유지한다.

주요 가정은 다음과 같습니다.

- Episode는 `SCHEDULE`로 시작하고, `SCHEDULE`이 누락된 경우 `UPDATE_RUNNING`을 방어적
  시작 event로 사용한다. `EVICT`, `FAIL`, `FINISH`, `KILL`, `LOST`가 episode를 종료한다.
  Terminal event가 없는 실행은 planning-horizon 끝에서 `HORIZON_CENSORED`로 표시한다.
  Running transition을 복원할 수 없지만 horizon에 usage가 있는 key는 full-horizon의
  usage-inferred `HORIZON_CENSORED` episode로 명시적으로 기록한다.
- Exact duplicate usage row는 episode 분할 전에 한 번 제거한다. 서로 다른 row의 interval이
  겹치면 elementary-interval union 기준으로 중복 시간을 한 번만 세고, 동시 관측값은
  resource별로 평균한다.
- 각 source usage row의 CPU maximum은 추출된 11개 percentile 중
  `cpu_usage_distribution[10]`, 즉 p100으로 정한다. BigQuery
  `maximum_usage.cpus`는 audit provenance로만 남고 `q_cpu` 계산에는 쓰지 않는다.
- Episode 내 positive-duration usage fragment 중 CPU distribution이 한 번이라도 비어 있거나,
  assigned `machine_id`의 history가 없거나, positive CPU capacity가 fragment 전체를 덮지
  못하면 해당 episode 전체를 sampling 전에 drop한다. Fully covered p100 capacity 초과값은
  진단 대상으로 남기며 cap, replacement 또는 추가 scaling하지 않는다.
- CPU/MEM은 같은 실제 joint machine shape에서 고른 representative capacity로 나눠
  representative-server fraction으로 단위 변환한다. 변환 후 `servers.csv`의
  `C_cpu=C_mem=1.0`은 실제 대표 server 한 대를 뜻한다. Peak target utilization을 맞추는
  자동 calibration은 적용하지 않는다. `q_cpu`는 arrival request와 episode p100 maximum의
  큰 값이고, `q_mem`은 arrival request와 source memory maximum의 큰 값이다. 변환된
  `q_cpu` 또는 `q_mem`이 1을 넘는 VM은 값을 자르거나 대체 표본을 뽑지 않고 전체 VM을
  제거한다.
- `on_demand`, `spot`, `batch_candidate`는 proxy label이며 Google trace에서 관측된 public-cloud purchase class가 아닙니다. Priority proxy는 Google 2019 tier를 사용합니다: `priority >= 120`은 `on_demand`, `priority < 120`은 `spot`, 단 `collection_events.scheduler == SCHEDULER_BATCH`이면 `batch_candidate`가 우선합니다. 각 VM row에는 `class_rule`을 남깁니다.
- Workload scenario는 5개이고 `scenario_probabilities.csv`는 기본적으로 uniform probability를 사용합니다. Scenario 0은 observed usage를 그대로 사용하고, scenario 1..4는 CPU/MEM에 lognormal multiplicative perturbation을 적용한 뒤 모든 VM class에서 각 resource를 `[0,q]`로 제한합니다. Hourly model input은 CPU와 memory 모두 관측 duration 가중평균으로 집계합니다.
- Spot preemption은 `instance_events.type == EVICT`를 사용합니다. Google ClusterData v3에서 numeric EVICT enum은 4이며, 문자열 `EVICT`도 지원합니다. 실제 EVICT가 없으면 low priority VM에 더 높은 synthetic hazard를 적용합니다. Preempted 이후에는 horizon 끝까지 inactive입니다.
- Batch workload는 개별 VM placement 대상이 아니라 `batch_candidate` VM들을 최대 10개 family로 묶은 hourly workload volume으로 추상화합니다.
- Energy scenario는 synthetic 입력입니다. Day-ahead/real-time/sell price, solar-like renewable generation, ESS capacity/charge/discharge limit, charge/discharge efficiency를 24시간, 5개 scenario로 생성합니다. Risk/ESS scalar parameters는 `model_params.json`에 `alpha`, `epsilon`, `soc_init`, `ess_charge_efficiency`, `ess_discharge_efficiency`로 저장합니다.
