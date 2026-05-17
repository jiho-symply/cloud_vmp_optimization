# cloud_vmp_optimization

이 저장소는 Virtual Machine Placement(VMP) 문제를 실험하기 위한 연구용 repository입니다. Azure trace 전처리 코드, 여러 toy/prototype 최적화 모델, 그리고 실험별 결과 묶음을 함께 관리합니다.

실험 코드는 `experiments/` 아래에 있고, 가공된 입력 데이터는 `data/processed/` 아래에 정리되어 있습니다. 각 실험은 자신만의 `results/` 폴더를 내부에 두어 결과 그림과 요약 파일을 함께 보관합니다.

가장 최근에 정리된 chance-constrained 2SP toy experiment 문서는 `experiments/2604-chance-2sp-toy/` 아래에 있습니다.

## 데이터 다운로드와 재생성

`data/processed/`와 각 실험의 `results/`는 Git에 올리지 않습니다. 새 환경에서는 아래 순서로 원본 Azure trace를 내려받고, 필요한 processed 데이터를 다시 만들면 됩니다.

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

생성된 파일들은 `data/processed/` 아래에 저장되며, Git에는 추적되지 않습니다.
