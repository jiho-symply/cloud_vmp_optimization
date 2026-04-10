# 데이터 전처리

## 목적
이 문서는 최신 `24 VM / cap8 / avg_cpu_mean >= 20` 벤치마크에서 데이터를 어떻게 만들었는지 설명합니다.

전처리는 두 단계입니다.
1. Azure 원시 trace에서 `sample_vm_data.csv`를 만든다.
2. 그 샘플 풀에서 toy instance를 만든다.

## 1단계: 샘플 VM 풀 재구성
실행 스크립트는 `rebuild_sample_data.py`입니다.

입력:
- `data/raw/trace_data/vmtable/vmtable.csv`
- `data/raw/trace_data/vm_cpu_readings/vm_cpu_readings-file-*-of-195.csv.gz`

출력:
- `data/processed/2601-initial-toy-model/sample_vm_data.csv`

핵심 처리:
- 실험 프로필 `2601-initial-toy-model`을 사용합니다.
- `vCPU` bucket을 실제 core 수로 바꿉니다.
- CPU reading을 시간 단위로 묶어 `min_cpu`, `avg_cpu`, `max_cpu`를 계산합니다.
- 이를 다시 core usage로 바꿉니다.

수식:

$$
\text{avg\_core\_usage} = \frac{\text{avg\_cpu} \times \text{vCPU}}{100}
$$

같은 방식으로 `min_core_usage`, `max_core_usage`도 계산합니다.

## 2단계: toy instance 생성
실행 스크립트는 `build_dataset.py`입니다.

최신 벤치마크의 기본 조건:
- 총 VM 수: `24`
- 서버 용량: `8`
- 후보 필터: `vCPU <= 8`, `avg_cpu_mean >= 20`
- 시나리오 수: `10`

### 후보 VM 필터
`sample_vm_data.csv`에서 VM별 요약을 만든 뒤 다음 조건을 만족하는 VM만 남깁니다.
- `vm_category`가 `Interactive` 또는 `Delay-insensitive`
- `vCPU <= 8`
- VM별 평균 CPU 사용률 `avg_cpu_mean >= 20`
- 시간 구간은 `hour < 24`

여기서 `avg_cpu_mean`은 원본 `avg_cpu`의 VM별 평균입니다. 즉, 최신 벤치마크는 평균적으로 꽤 바쁜 VM만 대상으로 샘플링합니다.

### VM 샘플링 규칙
샘플링은 VM lifetime 길이와 무관하게 균등 확률입니다.

즉, horizon 동안 계속 살아 있는 VM과 중간에 유입되는 VM을 따로 가중하지 않습니다.

기본 workload 구성:
- `balanced`: `8 / 8 / 8`
- `service-heavy`: `14 / 5 / 5`
- `spot-heavy`: `5 / 14 / 5`
- `batch-heavy`: `5 / 5 / 14`
- `OD only`: `24 / 0 / 0`
- `OD + SP`: `12 / 12 / 0`
- `OD + BJ`: `12 / 0 / 12`
- `OD + SP + BJ`: `8 / 8 / 8`

### 시나리오 수요 생성
수요 샘플링은 triangular distribution을 쓰지만, 최신 코드에서는 난수를 그냥 순서대로 뽑지 않습니다.

대신 `(vm_id, time, scenario)`를 키로 삼아 deterministic draw를 만듭니다.

의미:
- 같은 원본 VM이면 `on-demand`, `spot`, `batch` 어떤 역할로 쓰이더라도 같은 시간, 같은 시나리오에서 동일한 CPU 수요를 가집니다.
- `lambda = 0.1`과 `lambda = 0.0` 비교에서도 scenario draw는 동일하게 고정됩니다.

삼각분포의 mode는 다음처럼 복원합니다.

$$
\text{mode} = 3 \times \text{avg} - \text{min} - \text{max}
$$

그 다음 mode를 $[\text{min}, \text{max}]$ 구간 안으로 자릅니다.

### Batch 모델링
batch는 더 이상 VM 하나를 하루짜리 job 하나로 보지 않습니다.

대신 원본 batch VM trace를 시간 슬롯 단위로 쪼개서, 각 `(workload, time)`을 길이 1의 batch job 하나로 만듭니다.

즉, 해석은 다음과 같습니다.
- batch workload는 splitting이 허용된다.
- 각 시간 슬롯의 일을 독립적인 1-slot batch job으로 본다.

출력 파일:
- `batch_jobs.csv`
- `batch_job_demands.csv`

### 저장되는 주요 파일
instance 폴더에는 다음이 들어갑니다.
- `instance.json`
- `workload_metadata.csv`
- `selected_vm_traces.csv`
- `scenario_time_series.csv`
- `batch_jobs.csv`
- `batch_job_demands.csv`
- `scenario_detail.csv`
- `scenario_summary.csv`
- `scenario_probabilities.csv`
- `mix_summary.csv`

## 재현 명령

```powershell
.\.venv\Scripts\python.exe .\experiments\2604-chance-2sp-toy\build_dataset.py --on-demand-count 8 --spot-count 8 --batch-count 8 --max-vcpu 8 --min-avg-cpu 20 --server-capacity 8 --scenario-seed 42
```

실제 최신 벤치마크는 단일 케이스가 아니라 `run_24vm_lambda_queue.py`가 여러 케이스를 큐에 넣어 생성합니다.
