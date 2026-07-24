# NREL, NYISO, Google ClusterData 2019 데이터 수집/전처리 인수인계 문서

작성 기준: 2026-07-10. 경로는 clone한 repository root를 기준으로 표기한다.

이 문서는 다른 ChatGPT 또는 다른 연구자가 현재 workspace에서 만든 세 종류의 데이터셋을 그대로 이해하고 재현할 수 있도록 정리한 것이다.

- NREL/NLR renewable resource traces: 2019년 5분 단위 solar/wind resource를 받아 capacity factor와 renewable power trace로 가공.
- NYISO public MIS LBMP prices: 2019년 NYISO zonal day-ahead/real-time price ZIP을 받아 UTC 5분/30분 가격 trace로 가공.
- Google ClusterData 2019 toy VM-like workload: BigQuery public dataset에서 Borg instance usage/event를 제한 추출하고 stochastic VM placement / energy procurement toy instance로 가공.

중요한 해석 주의:

- NREL raw는 실제 발전소 출력이 아니라 기상/resource trace이다. 처리 후 capacity factor와 power는 모델 가정으로 계산한 값이다.
- NYISO raw는 wholesale LBMP 가격이다. 음수 가격과 extreme price는 제거하지 않았다.
- Google ClusterData 2019 데이터는 public-cloud VM trace가 아니다. `collection_id, instance_index` 쌍을 VM-like workload로 해석한 toy dataset이다. `on_demand`, `spot`, `batch_candidate`는 관측된 구매 클래스가 아니라 priority/scheduler 기반 proxy label이다.

## 0. Workspace 지도와 데이터 위치

Workspace root:

```text
.
```

이 문서에서 상대경로로 쓰는 모든 path와 실행 명령은 repository root 기준이다.

### 이번에 만든 세 데이터셋의 실제 위치

| 데이터셋 | raw 위치 | processed 위치 | 설명 |
|---|---|---|---|
| NREL/NLR renewable | `data/raw/nrel/` | `data/processed/nrel/` | 2019 solar/wind resource CSV와 capacity factor/power trace |
| NYISO price | `data/raw/nyiso/` | `data/processed/nyiso/` | 2019 NYISO DA/RT LBMP ZIP과 5분/30분 price trace |
| Google ClusterData 2019 toy | `data/raw/google2019_cell_a_day0_pool/` | `data/processed/notion_toy_google2019_v1_pool/` | BigQuery에서 추출한 Borg instance usage/event와 toy VM-like stochastic instance |

재현 가능한 상대경로:

```text
data/raw/nrel
data/processed/nrel

data/raw/nyiso
data/processed/nyiso

data/raw/google2019_cell_a_day0_pool
data/processed/notion_toy_google2019_v1_pool
```

### 바로 써야 하는 핵심 processed files

모델 또는 후속 분석에서 우선적으로 볼 파일은 다음이다.

| 용도 | 파일 |
|---|---|
| NREL 30분 capacity factor | `data/processed/nrel/renewable_cf_2019_30min.parquet` |
| NREL 30분 renewable power/energy | `data/processed/nrel/renewable_power_2019_30min_assumed_100mw.parquet` |
| NREL site별 capacity 가정 | `data/processed/nrel/site_capacity_assumptions_2019_assumed_100mw.csv` |
| NYISO 5분 price | `data/processed/nyiso/electricity_price_nyiso_2019_5min.parquet` |
| NYISO 30분 price | `data/processed/nyiso/electricity_price_nyiso_2019_30min.parquet` |
| Google VM request metadata | `data/processed/notion_toy_google2019_v1_pool/vm_requests.csv` |
| Google 5분 workload scenario | `data/processed/notion_toy_google2019_v1_pool/vm_usage_5min_scenarios.csv` |
| Google hourly workload scenario, model default | `data/processed/notion_toy_google2019_v1_pool/vm_usage_hourly_scenarios.csv` |
| Google spot preemption path | `data/processed/notion_toy_google2019_v1_pool/spot_preemption_scenarios.csv` |
| Google batch abstraction | `data/processed/notion_toy_google2019_v1_pool/batch_families.csv`, `data/processed/notion_toy_google2019_v1_pool/batch_workload.csv` |
| Google synthetic energy scenario | `data/processed/notion_toy_google2019_v1_pool/energy_scenarios.csv` |
| Google scenario probability | `data/processed/notion_toy_google2019_v1_pool/scenario_probabilities.csv` |
| Google scalar model params | `data/processed/notion_toy_google2019_v1_pool/model_params.json` |
| Google validation summary | `data/processed/notion_toy_google2019_v1_pool/toy_instance_summary.md` |

### Workspace top-level 폴더/파일 역할

현재 workspace top-level 구조와 역할은 다음과 같다.

| path | 종류 | 대략 크기 | 내용 |
|---|---|---:|---|
| `AGENTS.md` | file | 4 KB | 이 workspace의 agent 지침. Python은 `.venv`를 사용하라는 규칙 포함. |
| `README.md` | file | 8 KB | repository 기본 설명, Azure 재현 절차, Google toy pipeline 요약. |
| `README_nrel_renewables.md` | file | 4 KB | NREL/NLR renewable pipeline 전용 README. |
| `README_nyiso_prices.md` | file | 4 KB | NYISO price pipeline 전용 README. |
| `requirements.txt` | file | 4 KB | Python dependency 목록. |
| `pyproject.toml` | file | 4 KB | pytest 설정. `pythonpath = ["src"]`, `testpaths = ["tests"]`. |
| `configs/` | dir | 12 KB | 데이터 pipeline config YAML. 현재 NREL/NYISO config가 있음. |
| `data/` | dir | 약 20 GB | raw/processed/metadata 데이터 저장소. 가장 큰 폴더. |
| `docs/` | dir | 44 KB | 인수인계 문서 등 새 documentation. 이 문서가 여기 있음. |
| `experiments/` | dir | 21 MB | VMP/2SP/chance/CVaR/no-risk 실험 코드와 README. |
| `notebooks/` | dir | 744 KB | 참고 notebook. 현재 Azure trace analysis reference notebook 포함. |
| `references/` | dir | 15 MB | 연구 배경 문서와 논문 PDF. |
| `scripts/` | dir | 76 KB | 실행 entrypoint. 다운로드/전처리/검증 CLI scripts. |
| `src/` | dir | 344 KB | 재사용 가능한 Python source package. NREL, NYISO, Google pipeline 구현. |
| `tests/` | dir | 112 KB | pytest test suite. 데이터 parser/변환/검증 테스트. |
| `.venv/` | dir | 약 949 MB | Python virtual environment. 실행 시 사용. |
| `.git/` | dir | 약 100 MB | Git repository metadata. |
| `.pytest_cache/`, `__pycache__/` | dir | small | pytest/Python cache. 분석에는 필요 없음. |
| `benchmark_migration_v2.py`, `chance_vs_cvar_gurobi.py` | file | small | 독립 실험/benchmark script. |
| `migration_benchmark*.csv` | file | small | migration benchmark 결과 CSV. |

### `data/` 하위 폴더 설명

`data/`는 raw와 processed가 같이 있는 가장 중요한 폴더다.

| path | 내용 |
|---|---|
| `data/metadata/` | Azure Public Dataset V2 metadata, schema, bucket definition 등. |
| `data/raw/` | 외부에서 내려받은 원본 또는 거의 원본에 가까운 데이터. |
| `data/raw/azure2019_data/` | 기존 Azure dataset 관련 raw data. |
| `data/raw/trace_data/` | Azure trace_data 하위 자료. `deployments`, `subscriptions`, `vm_cpu_readings`, `vmtable` 등이 있음. |
| `data/raw/nrel/solar/2019/` | NREL/NLR solar raw CSV 10개와 metadata JSON 10개. |
| `data/raw/nrel/wind/2019/` | NREL/NLR wind raw CSV 10개와 metadata JSON 10개. |
| `data/raw/nyiso/da_lbmp_zonal/2019/` | NYISO day-ahead monthly ZIP 12개. |
| `data/raw/nyiso/rt_lbmp_zonal_5min/2019/` | NYISO real-time monthly ZIP 12개. |
| `data/raw/google2019_cell_a_day0_pool/` | Google BigQuery 추출 parquet 3개와 metadata JSON. |
| `data/processed/` | 모델/분석에 바로 쓸 수 있게 가공된 데이터. |
| `data/processed/nrel/` | NREL CF/power processed outputs. |
| `data/processed/nyiso/` | NYISO price processed outputs. |
| `data/processed/notion_toy_google2019_v1_pool/` | Google toy VM-like stochastic instance processed outputs. |
| `data/processed/2601-initial-toy-model/` | 초기 toy model용 sample VM data. |
| `data/processed/2604-chance-2sp-toy/` | 2604 chance-constrained 2SP toy experiment inputs/results-style processed datasets. |
| `data/processed/2605-CVaR-surrogate/` | 2605 CVaR surrogate experiment processed inputs. |
| `data/processed/2605-no-risk-control/` | no-risk-control experiment processed inputs. |
| `data/processed/2605-vm-type-modeling-1/` | VM type modeling experiment processed inputs. |

### `configs/` 하위 파일

| file | 내용 |
|---|---|
| `configs/nrel_renewables_2019_v0.yaml` | NREL/NLR API endpoint, site list, solar/wind model assumptions, scaling assumptions, output paths. |
| `configs/nyiso_prices_2019_v0.yaml` | NYISO URL templates, zone list, expected row counts, timezone/grid settings, output paths. |

### `scripts/` 하위 파일

| file | 내용 |
|---|---|
| `scripts/download_azure_dataset.py` | 기존 Azure public dataset 다운로드 script. |
| `scripts/fetch_and_build_nrel_2019.py` | NREL/NLR raw fetch와 processed build entrypoint. |
| `scripts/fetch_and_build_nyiso_prices_2019.py` | NYISO raw ZIP fetch와 processed build entrypoint. |
| `scripts/01_extract_google2019.py` | Google ClusterData 2019 BigQuery bounded extract entrypoint. |
| `scripts/02_build_toy_instance.py` | Google raw parquet에서 toy stochastic instance 생성. |
| `scripts/03_validate_outputs.py` | Google toy output validation 및 summary 작성. |

### `src/` 하위 package

| path | 내용 |
|---|---|
| `src/renewable_trace/` | NREL/NLR CSV parsing, API download, solar/wind CF 변환, 30분 집계, power scaling 구현. |
| `src/electricity_price/` | NYISO ZIP download, CSV parsing, timezone/DST 처리, DA/RT alignment, 30분 집계 구현. |
| `src/google2019_toy/` | BigQuery query builder, Google toy VM request/scenario/batch/energy/server output 생성, validation 구현. |

### `experiments/` 하위 폴더

`experiments/`는 데이터 pipeline 자체보다는 VMP/2SP 모델 실험 코드가 들어 있는 영역이다.

| path | 내용 |
|---|---|
| `experiments/2601-initial-toy-model/` | 초기 toy model과 visualization. |
| `experiments/2602-baseline-toy-model/` | baseline toy model과 Gantt visualization. |
| `experiments/2602-refined-toy-model/` | refined toy model과 visualization. |
| `experiments/2604-chance-2sp-toy/` | chance-constrained two-stage stochastic toy model, dataset builder, sweep/queue scripts. |
| `experiments/2604-cut-experiment/` | cut profile/model 실험, cut pool/search/queue scripts. |
| `experiments/2604-bc-cg-benchmark/` | branch-cut/column-generation benchmark 관련 scripts. |
| `experiments/2604-branch-cut-benchmark/` | branch-and-cut benchmark scripts와 recovery script. |
| `experiments/2604-decomposition-experiment/` | decomposition benchmark/search scripts. |
| `experiments/2605-CVaR-surrogate/` | CVaR surrogate model, data prep, config. |
| `experiments/2605-no-risk-control/` | no-risk control model, type-mix benchmark, visualization. |
| `experiments/2605-vm-type-modeling-1/` | Notion VM type modeling model/data prep/visualization/ablation. |
| `experiments/config_2605.py` | 2605 계열 공통 설정 helper. |

### `references/`, `notebooks/`, `tests/`, `docs/`

| path | 내용 |
|---|---|
| `references/` | temporal bin packing, VMP, stochastic demand/risk, energy model 관련 정리 문서와 논문 PDF. |
| `references/papers/` | 관련 논문 PDF 파일. |
| `notebooks/reference/` | Azure 2019 Public Dataset V2 trace analysis notebook. |
| `tests/` | NREL timestamp/wind CF/fetch error, NYISO price, Google toy pipeline tests. |
| `docs/data_pipeline_handoff_nrel_nyiso_gct.md` | 이 인수인계 문서. |

## 공통 실행 환경

항상 repository의 `.venv`를 사용한다.

```bash
# repository root에서 실행
.venv/bin/python -m pip install -r requirements.txt
```

주요 Python dependency:

- `numpy`
- `pandas`
- `pyarrow`
- `requests`
- `pvlib`
- `python-dotenv`
- `PyYAML`
- `google-cloud-bigquery`
- `pytest`

관련 코드:

- NREL:
  - `scripts/fetch_and_build_nrel_2019.py`
  - `configs/nrel_renewables_2019_v0.yaml`
  - `src/renewable_trace/`
- NYISO:
  - `scripts/fetch_and_build_nyiso_prices_2019.py`
  - `configs/nyiso_prices_2019_v0.yaml`
  - `src/electricity_price/`
- Google ClusterData 2019:
  - `scripts/01_extract_google2019.py`
  - `scripts/02_build_toy_instance.py`
  - `scripts/03_validate_outputs.py`
  - `src/google2019_toy/`

## 1. NREL/NLR renewable resource trace

### 목적

10개 미국 data-center proxy site에 대해 2019년 5분 단위 solar/wind resource trace를 다운로드하고, 이를 다음 모델 입력으로 변환했다.

- 5-minute solar/wind capacity factor
- 30-minute solar/wind capacity factor
- 30-minute renewable power and energy trace
- site별 capacity scaling assumption

### 실행 명령

환경 변수 필요:

```bash
export NREL_API_KEY="YOUR_KEY"
export NREL_API_EMAIL="YOUR_EMAIL"
```

`NLR_API_KEY`도 `NREL_API_KEY`의 fallback alias로 허용된다. API key는 metadata에 `<redacted>`로 저장된다.

실행:

```bash
.venv/bin/python scripts/fetch_and_build_nrel_2019.py --fetch --process
```

기타 모드:

```bash
.venv/bin/python scripts/fetch_and_build_nrel_2019.py --dry-run
.venv/bin/python scripts/fetch_and_build_nrel_2019.py --fetch-only
.venv/bin/python scripts/fetch_and_build_nrel_2019.py --process-only
.venv/bin/python scripts/fetch_and_build_nrel_2019.py --fetch-only --force
```

### 설정 파일

설정 파일: `configs/nrel_renewables_2019_v0.yaml`

핵심 설정:

| 항목 | 값 |
|---|---:|
| year | 2019 |
| raw interval | 5 minutes |
| placement interval | 30 minutes |
| timezone | UTC |
| leap_day | false |
| expected raw rows per site | 105,120 |
| expected 5min CF total rows | 1,051,200 |
| expected 30min CF total rows | 175,200 |
| assumed mean demand per site | 100 MW |
| target average renewable-to-demand ratio | 2.0 |
| solar energy mix | 20% |
| wind energy mix | 80% |

### 대상 site

NREL/NLR 다운로드 대상 site는 10개이다.

| site_id | state | lat | lon |
|---|---|---:|---:|
| CA | California | 36.7783 | -119.4179 |
| WA | Washington | 47.7511 | -120.7401 |
| OR | Oregon | 43.8041 | -120.5542 |
| IL | Illinois | 40.6331 | -89.3985 |
| GA | Georgia | 32.1656 | -82.9001 |
| VA | Virginia | 37.4316 | -78.6569 |
| TX | Texas | 31.9686 | -99.9018 |
| FL | Florida | 27.6648 | -81.5158 |
| NC | North Carolina | 35.7596 | -79.0193 |
| SC | South Carolina | 33.8361 | -81.1637 |

### Raw 다운로드

다운로드 방식:

- direct CSV API request 사용.
- `MULTIPOINT` archive download가 아니라 site별 `POINT(lon lat)` 요청.
- solar와 wind를 각각 site별/연도별로 받음.
- 이미 유효한 raw file이 있으면 skip.
- `--force`를 주면 다시 다운로드.
- API 호출 사이 `min_seconds_between_calls = 1.25`.
- `max_retries = 5`, `timeout_seconds = 120`.

Raw solar endpoint:

```text
https://developer.nlr.gov/api/nsrdb/v2/solar/nsrdb-GOES-conus-v4-0-0-download.csv
```

Raw wind endpoint:

```text
https://developer.nlr.gov/api/wind-toolkit/v2/wind/wtk-conus-5min-v1-0-0-download.csv
```

주의: 현재 config와 raw metadata에 기록된 endpoint 문자열은 위와 같다.

Solar request parameter:

- `wkt = POINT(lon lat)`
- `names = 2019`
- `interval = 5`
- `utc = true`
- `leap_day = false`
- `attributes = ghi,dni,dhi,air_temperature,wind_speed,solar_zenith_angle,surface_albedo,surface_pressure`
- `email = NREL_API_EMAIL`
- `api_key = NREL_API_KEY`

Wind request parameter:

- `wkt = POINT(lon lat)`
- `names = 2019`
- `interval = 5`
- `utc = true`
- `leap_day = false`
- `attributes = windspeed_80m,winddirection_80m`
- `email = NREL_API_EMAIL`
- `api_key = NREL_API_KEY`

Raw output:

- Solar CSV 10개:
  - `data/raw/nrel/solar/2019/{site_id}.csv`
  - `data/raw/nrel/solar/2019/{site_id}.metadata.json`
- Wind CSV 10개:
  - `data/raw/nrel/wind/2019/{site_id}.csv`
  - `data/raw/nrel/wind/2019/{site_id}.metadata.json`

현재 workspace의 raw file count:

| raw group | count |
|---|---:|
| solar CSV | 10 |
| solar metadata JSON | 10 |
| wind CSV | 10 |
| wind metadata JSON | 10 |

### Raw CSV parsing

NREL/NLR raw CSV는 보통 앞쪽에 metadata row가 있고, 그 뒤에 실제 timeseries header가 있다. parser는 다음을 요구한다.

- `Year,Month,Day,Hour,Minute,...` header 존재.
- UTC timestamp로 변환.
- 2019 no-leap year이므로 site당 105,120개 5분 timestamp.
- timestamp 중복 없음.
- timestamp strictly sorted.
- 시작 timestamp: `2019-01-01 00:00:00+00:00`.
- 마지막 timestamp: `2019-12-31 23:55:00+00:00`.

### Solar capacity factor 처리

Solar raw resource를 PVWatts 기반의 AC capacity factor로 변환했다.

필수 입력 컬럼:

- `ghi`
- `dni`
- `dhi`
- `air_temperature`
- `wind_speed`

있으면 사용하는 추가 컬럼:

- `surface_albedo`

모델 가정:

| 항목 | 값 |
|---|---:|
| AC rated power | 1,000,000 W |
| DC/AC ratio | 1.2 |
| inverter nominal efficiency | 0.96 |
| system losses | 0.14 |
| gamma_pdc | -0.004 |
| surface azimuth | 180 degrees |
| tilt | site latitude |
| default albedo | 0.2 |

처리 흐름:

1. `pvlib.solarposition.get_solarposition()`으로 태양 위치 계산.
2. `pvlib.irradiance.get_total_irradiance()`로 plane-of-array irradiance 계산.
3. `pvlib.temperature.sapm_cell()`로 cell temperature 계산.
4. `pvlib.pvsystem.pvwatts_dc()`로 DC power 계산.
5. system loss 14% 적용.
6. `pvlib.inverter.pvwatts()`로 AC power 계산.
7. `solar_cf = pac / ac_rated_power_w`.
8. `solar_cf`는 `[0, 1]`로 clip.
9. 거의 0인 값은 0으로 정리.

### Wind capacity factor 처리

Wind raw resource의 `windspeed_80m`을 단순 turbine curve로 변환했다.

모델 가정:

| 항목 | 값 |
|---|---:|
| rated power | 1.5 MW |
| hub height | 80 m |
| cut-in speed | 3.5 m/s |
| rated speed | 12.0 m/s |
| cut-out speed | 25.0 m/s |
| wind loss factor | 0.90 |

처리 흐름:

- `v < cut_in`: `wind_cf = 0`
- `cut_in <= v < rated`: cubic interpolation
- `rated <= v <= cut_out`: `wind_cf = 0.90`
- `v > cut_out`: `wind_cf = 0`
- 마지막에 `[0, 1]`로 clip.

### 5분 CF 생성

각 site에 대해 solar CF와 wind CF를 `timestamp_utc, site_id, state, lat, lon` 기준으로 inner join했다.

Output columns:

- `timestamp_utc`
- `site_id`
- `state`
- `lat`
- `lon`
- `solar_cf`
- `wind_cf`

현재 output:

| file | rows | columns | timestamp range |
|---|---:|---:|---|
| `data/processed/nrel/renewable_cf_2019_5min.parquet` | 1,051,200 | 7 | `2019-01-01 00:00:00+00:00` to `2019-12-31 23:55:00+00:00` |
| `data/processed/nrel/renewable_cf_2019_5min.csv.gz` | 1,051,200 | 7 | same |

Site count: 10. Timestamp count per site: 105,120.

### 30분 CF 생성

5분 CF를 30분 bucket으로 floor한 뒤, 6개 5분 row를 평균냈다.

Validation:

- 각 30분 group은 정확히 6개의 5분 source row를 가져야 한다.
- row 수는 총 175,200개.
- site별 row 수는 17,520개.

현재 output:

| file | rows | columns | timestamp range |
|---|---:|---:|---|
| `data/processed/nrel/renewable_cf_2019_30min.parquet` | 175,200 | 7 | `2019-01-01 00:00:00+00:00` to `2019-12-31 23:30:00+00:00` |
| `data/processed/nrel/renewable_cf_2019_30min.csv.gz` | 175,200 | 7 | same |

### Renewable power scaling

30분 CF를 power trace로 바꿀 때 site별 capacity를 다음 방식으로 정했다.

공통 assumption:

- 각 site의 평균 demand를 100 MW로 가정.
- 평균 renewable generation target은 demand의 2배, 즉 200 MW.
- solar 평균 target은 200 MW의 20%, 즉 40 MW.
- wind 평균 target은 200 MW의 80%, 즉 160 MW.

Site별 capacity:

```text
solar_capacity_mw = target_avg_solar_mw / mean(solar_cf)
wind_capacity_mw  = target_avg_wind_mw  / mean(wind_cf)
```

30분 power:

```text
solar_power_mw      = solar_capacity_mw * solar_cf
wind_power_mw       = wind_capacity_mw * wind_cf
renewable_power_mw  = solar_power_mw + wind_power_mw
renewable_energy_mwh = renewable_power_mw * 0.5
```

현재 output:

| file | rows | columns |
|---|---:|---:|
| `data/processed/nrel/renewable_power_2019_30min_assumed_100mw.parquet` | 175,200 | 12 |
| `data/processed/nrel/renewable_power_2019_30min_assumed_100mw.csv.gz` | 175,200 | 12 |
| `data/processed/nrel/site_capacity_assumptions_2019_assumed_100mw.csv` | 10 | 12 |
| `data/processed/nrel/renewable_cf_2019_summary_by_site.csv` | 10 | 14 |

Power output columns:

- `timestamp_utc`
- `site_id`
- `state`
- `lat`
- `lon`
- `assumed_mean_demand_mw`
- `solar_capacity_mw`
- `wind_capacity_mw`
- `solar_power_mw`
- `wind_power_mw`
- `renewable_power_mw`
- `renewable_energy_mwh`

### NREL site summary 현재 값

`data/processed/nrel/renewable_cf_2019_summary_by_site.csv` 기준:

| site | solar_cf_mean | wind_cf_mean | missing solar | missing wind |
|---|---:|---:|---:|---:|
| CA | 0.218617 | 0.051774 | 0 | 0 |
| FL | 0.205008 | 0.178419 | 0 | 0 |
| GA | 0.197143 | 0.209584 | 0 | 0 |
| IL | 0.174101 | 0.358027 | 0 | 0 |
| NC | 0.186453 | 0.190265 | 0 | 0 |
| OR | 0.210139 | 0.200509 | 0 | 0 |
| SC | 0.194878 | 0.218628 | 0 | 0 |
| TX | 0.217181 | 0.276251 | 0 | 0 |
| VA | 0.185872 | 0.199116 | 0 | 0 |
| WA | 0.180267 | 0.198212 | 0 | 0 |

### NREL capacity assumptions 현재 값

`data/processed/nrel/site_capacity_assumptions_2019_assumed_100mw.csv` 기준:

| site | solar_capacity_mw | wind_capacity_mw |
|---|---:|---:|
| CA | 182.968006 | 3090.349919 |
| FL | 195.114128 | 896.766043 |
| GA | 202.898041 | 763.418397 |
| IL | 229.751157 | 446.893956 |
| NC | 214.531083 | 840.930629 |
| OR | 190.350456 | 797.970445 |
| SC | 205.256233 | 731.835502 |
| TX | 184.178538 | 579.183894 |
| VA | 215.202290 | 803.550619 |
| WA | 221.892539 | 807.215633 |

### NREL validation

NREL pipeline validation:

- raw solar/wind CSV가 존재해야 함.
- site당 raw row count 105,120.
- timestamp unique and sorted.
- merged solar/wind rows site당 105,120.
- total 5min CF rows 1,051,200.
- total 30min CF rows 175,200.
- 30min group당 source row exactly 6.
- `solar_cf`, `wind_cf`는 `[0, 1]`.
- scaled power/energy 음수 없음.

## 2. NYISO electricity price trace

### 목적

PJM Data Miner dependency 없이 NYISO MIS public CSV/ZIP만 사용해서 2019년 NYISO 11개 load zone의 day-ahead와 real-time LBMP를 통합했다.

최종 용도:

- 5-minute electricity price trace
- 30-minute VM-placement/procurement input
- Kwon-style day-ahead procurement and real-time recourse experiment input

### 실행 명령

```bash
.venv/bin/python scripts/fetch_and_build_nyiso_prices_2019.py --fetch --process
```

기타 모드:

```bash
.venv/bin/python scripts/fetch_and_build_nyiso_prices_2019.py --dry-run
.venv/bin/python scripts/fetch_and_build_nyiso_prices_2019.py --fetch-only
.venv/bin/python scripts/fetch_and_build_nyiso_prices_2019.py --process-only
.venv/bin/python scripts/fetch_and_build_nyiso_prices_2019.py --fetch-only --force
```

### 설정 파일

설정 파일: `configs/nyiso_prices_2019_v0.yaml`

핵심 설정:

| 항목 | 값 |
|---|---:|
| market | NYISO |
| year | 2019 |
| raw timezone | America/New_York |
| output timezone | UTC |
| RT raw interval | 5 minutes |
| DA native interval | 60 minutes |
| final interval | 5 minutes |
| placement interval | 30 minutes |
| expected 5min rows total | 1,156,320 |
| expected 30min rows total | 192,720 |
| expected 5min rows per site | 105,120 |
| expected 30min rows per site | 17,520 |

### Raw 다운로드

NYISO raw data는 public ZIP archive이다. API key, login, token이 없다.

Day-ahead zonal LBMP URL pattern:

```text
https://mis.nyiso.com/public/csv/damlbmp/{yyyymm}01damlbmp_zone_csv.zip
```

Real-time zonal LBMP URL pattern:

```text
https://mis.nyiso.com/public/csv/realtime/{yyyymm}01realtime_zone_csv.zip
```

Downloader 설정:

- `max_retries = 5`
- `timeout_seconds = 120`
- `min_seconds_between_calls = 0.25`
- valid ZIP이 있으면 skip.
- `--force`면 다시 다운로드.

Raw output:

- Day-ahead ZIP 12개:
  - `data/raw/nyiso/da_lbmp_zonal/2019/{YYYYMM}01damlbmp_zone_csv.zip`
- Real-time ZIP 12개:
  - `data/raw/nyiso/rt_lbmp_zonal_5min/2019/{YYYYMM}01realtime_zone_csv.zip`

현재 workspace raw count:

| raw group | count |
|---|---:|
| DA monthly ZIP | 12 |
| RT monthly ZIP | 12 |

### NYISO 대상 zone

| site_id | NYISO zone | proxy city |
|---|---|---|
| WEST | WEST | Buffalo |
| GENESE | GENESE | Rochester |
| CENTRL | CENTRL | Syracuse |
| NORTH | NORTH | Watertown |
| MHKVL | MHK VL | Utica |
| CAPITL | CAPITL | Albany |
| HUDVL | HUD VL | Poughkeepsie |
| MILLWD | MILLWD | White Plains |
| DUNWOD | DUNWOD | Yonkers |
| NYC | N.Y.C. | New York City |
| LONGIL | LONGIL | Hicksville |

### Raw CSV parsing

필수 raw columns:

- `Time Stamp`
- `Name`
- `PTID`
- `LBMP ($/MWHr)`
- `Marginal Cost Losses ($/MWHr)`
- `Marginal Cost Congestion ($/MWHr)`

처리 방식:

1. ZIP 내부 CSV들을 모두 읽는다.
2. `Name`이 대상 NYISO zone인 row만 필터링한다.
3. raw timestamp는 `America/New_York` local time으로 해석한다.
4. DST ambiguity 처리:
   - 기본은 `ambiguous="infer"`, `nonexistent="shift_forward"`.
   - infer 실패 시 `ambiguous=False` fallback.
5. UTC로 변환한다.
6. RT dataset은 raw timestamp를 interval end로 보고 5분을 뺀다.
7. RT timestamp는 5분 단위로 round한다.
8. DA는 hourly timestamp 그대로 사용한다.
9. local year가 2019인 row만 유지한다.
10. duplicated timestamp/zone은 value columns 평균으로 collapse한다.

### 5분 grid 정규화

NYISO raw는 DST와 archive irregularity 때문에 그대로 두면 5분 grid가 빈틈이 생길 수 있다. 그래서 control grid를 만든다.

RT grid:

- local `2019-01-01 00:00:00 America/New_York`를 UTC로 변환한 시각부터 시작.
- UTC range는 현재 output 기준 `2019-01-01 05:00:00+00:00` to `2020-01-01 04:55:00+00:00`.
- site별 105,120개 5분 slot.

DA grid:

- 동일 local year 2019 기준 hourly grid.
- site별 8,760개 hourly slot.

정규화 방식:

- 각 site별로 timestamp index를 grid에 reindex.
- `ptid`, `_local_year`는 forward-fill/backward-fill.
- price/loss/congestion numeric column은 time interpolation 후 forward-fill/backward-fill.
- DA는 RT 5분 grid에 `merge_asof(direction="backward", tolerance=59min)`으로 붙인다.

### 보존/삭제 정책

- 음수 LBMP는 보존한다.
- absolute price가 1000 USD/MWh를 넘는 extreme row는 validation/report 대상이지만 제거하지 않는다.
- DA/RT missing value는 최종 output에 없어야 한다.
- duplicated `(timestamp_utc, site_id)`는 없어야 한다.

### 5분 price output

Output files:

- `data/processed/nyiso/electricity_price_nyiso_2019_5min.parquet`
- `data/processed/nyiso/electricity_price_nyiso_2019_5min.csv.gz`

현재 output:

| rows | columns | zones | timestamp count | timestamp range |
|---:|---:|---:|---:|---|
| 1,156,320 | 11 | 11 | 105,120 | `2019-01-01 05:00:00+00:00` to `2020-01-01 04:55:00+00:00` |

Columns:

- `timestamp_utc`
- `market`
- `site_id`
- `nyiso_zone`
- `ptid`
- `da_lbmp_usd_per_mwh`
- `rt_lbmp_usd_per_mwh`
- `da_loss_raw_usd_per_mwh`
- `rt_loss_raw_usd_per_mwh`
- `da_congestion_raw_usd_per_mwh`
- `rt_congestion_raw_usd_per_mwh`

### 30분 price output

5분 price를 30분 bucket으로 floor한 뒤, 6개 5분 row의 평균을 냈다.

Validation:

- 각 30분 group은 정확히 6개 5분 source row를 가져야 한다.
- site별 17,520 row.
- total 192,720 row.

Output files:

- `data/processed/nyiso/electricity_price_nyiso_2019_30min.parquet`
- `data/processed/nyiso/electricity_price_nyiso_2019_30min.csv.gz`

현재 output:

| rows | columns | zones | timestamp count | timestamp range |
|---:|---:|---:|---:|---|
| 192,720 | 11 | 11 | 17,520 | `2019-01-01 05:00:00+00:00` to `2020-01-01 04:30:00+00:00` |

### NYISO summary 현재 값

Summary file:

- `data/processed/nyiso/electricity_price_nyiso_2019_summary_by_zone.csv`

현재 summary row 수: 11.

주요 validation count:

| item | count |
|---|---:|
| missing DA values | 0 |
| missing RT values | 0 |
| duplicated timestamp/site pairs | 0 |
| negative DA rows total | 84 |
| negative RT rows total | 13,982 |

Zone별 negative count:

| site | negative DA | negative RT |
|---|---:|---:|
| CAPITL | 0 | 187 |
| CENTRL | 0 | 1,021 |
| DUNWOD | 0 | 159 |
| GENESE | 0 | 2,747 |
| HUDVL | 0 | 452 |
| LONGIL | 0 | 192 |
| MHKVL | 0 | 1,289 |
| MILLWD | 0 | 206 |
| NORTH | 84 | 6,866 |
| NYC | 0 | 211 |
| WEST | 0 | 652 |

Extreme price note:

- `abs(price) > 1000 USD/MWh` row가 존재한다.
- 대표적으로 NORTH RT min은 `-7033.77`, LONGIL RT max는 `4210.8`.
- 이 값들은 NYISO raw LBMP의 extreme event로 보고 보존했다.

### NYISO validation

NYISO pipeline validation:

- DA ZIP 12개 존재.
- RT ZIP 12개 존재.
- output row count exact match.
- site별 row count exact match.
- duplicated `(timestamp_utc, site_id)` 없음.
- DA/RT LBMP missing 없음.
- 30분 aggregation group마다 source row exactly 6.

## 3. Google ClusterData 2019 toy VM-like workload dataset

### 목적

Google ClusterData 2019 BigQuery public dataset에서 cell `a`, day index 0의 Borg instance usage/event를 제한적으로 추출했다. 이후 `(collection_id, instance_index)`를 VM-like workload로 취급해서 two-stage stochastic VM placement / energy procurement model용 toy instance를 만들었다.

정확한 disclaimer:

```text
This is a VM-like toy dataset derived from Google ClusterData 2019 Borg instance traces, not a real public-cloud VM trace.
```

### BigQuery raw extraction 실행 명령

실제 실행에 사용한 output directory는 pool suffix가 붙은 경로이다.

```bash
.venv/bin/python scripts/01_extract_google2019.py \
  --project_id YOUR_GCP_PROJECT \
  --cell a \
  --day_index 0 \
  --max_instances 10000 \
  --seed 42 \
  --output_dir data/raw/google2019_cell_a_day0_pool
```

주의:

- `collection_events`는 현재 script에서 기본 포함이다.
- BigQuery 실행 전 dry-run을 수행하고 estimated bytes processed를 출력한다.
- 전체 trace를 받지 않고, selected candidate에 관련된 제한된 row만 parquet로 저장한다.

### BigQuery source tables

사용 table:

```text
`google.com:google-cluster-data`.clusterdata_2019_a.instance_usage
`google.com:google-cluster-data`.clusterdata_2019_a.instance_events
`google.com:google-cluster-data`.clusterdata_2019_a.collection_events
```

현재 raw metadata:

| 항목 | 값 |
|---|---:|
| cell | a |
| day_index | 0 |
| day_start_us | 0 |
| day_end_us | 86,400,000,000 |
| max_instances | 10,000 |
| seed | 42 |

Dry-run estimated bytes processed:

| query | estimated bytes |
|---|---:|
| usage_5min | 424,228,037,408 |
| instance_events | 399,189,063,424 |
| collection_events | 304,185,243,104 |

Raw extracted row counts:

| raw parquet | rows |
|---|---:|
| `data/raw/google2019_cell_a_day0_pool/usage_5min.parquet` | 1,799,148 |
| `data/raw/google2019_cell_a_day0_pool/instance_events.parquet` | 275,673 |
| `data/raw/google2019_cell_a_day0_pool/collection_events.parquet` | 5,435 |

Raw metadata file:

- `data/raw/google2019_cell_a_day0_pool/metadata.json`

### BigQuery extraction filter

`instance_usage` extraction:

- `start_time >= @day_start_us`
- `start_time < @day_end_us`
- `end_time - start_time >= 300000000`, i.e. at least 300 seconds.
- `(alloc_collection_id IS NULL OR alloc_collection_id = 0)`
- `t5 = DIV(start_time, 300000000)`
- `t5_day = DIV(start_time - @day_start_us, 300000000)`
- `t5_day BETWEEN 0 AND 287`

Candidate selection:

- candidate key: `(collection_id, instance_index)`
- candidate must have at least 12 observed 5-minute usage rows.
- deterministic sampling uses `FARM_FINGERPRINT(CONCAT(collection_id, ':', instance_index, ':', seed))`.
- sample order: `sample_hash, collection_id, instance_index`.
- limit: 10,000 candidates.

`instance_events` extraction:

- joins selected candidates by `(collection_id, instance_index)`.
- keeps event rows with `e.time < @day_end_us`.
- extracts:
  - `collection_id`
  - `instance_index`
  - `time`
  - `type AS event_type`
  - `priority`
  - `resource_request.cpus AS resource_request_cpu`
  - `resource_request.memory AS resource_request_mem`

`collection_events` extraction:

- joins selected candidate collections.
- keeps `e.time < @day_end_us`.
- maps scheduler enum:
  - `1 -> SCHEDULER_BATCH`
  - `0 -> SCHEDULER_DEFAULT`
  - otherwise `SCHEDULER_{value}`
- extracts:
  - `collection_id`
  - `time`
  - `event_type`
  - `scheduler`
  - `collection_type`
  - `scheduling_class`
  - `priority`

### Toy instance build 실행 명령

현재 생성된 processed pool은 10 scenarios, 200 servers로 만들었다.

```bash
.venv/bin/python scripts/02_build_toy_instance.py \
  --raw_dir data/raw/google2019_cell_a_day0_pool \
  --output_dir data/processed/notion_toy_google2019_v1_pool \
  --max_instances 10000 \
  --seed 42 \
  --num_servers 200 \
  --num_scenarios 10
```

Validation:

```bash
.venv/bin/python scripts/03_validate_outputs.py \
  --output_dir data/processed/notion_toy_google2019_v1_pool \
  --write_summary
```

### VM-like request 생성

입력:

- `usage_5min.parquet`
- `instance_events.parquet`
- `collection_events.parquet`
- raw `metadata.json`의 `day_start_us`

Usage normalization:

- CPU/memory usage numeric 변환.
- 음수 usage는 0으로 clip.
- duration이 5분 미만인 row 제거.
- raw에 `t5_day`가 있으면 그대로 사용.
- raw에 없으면 `day_start_us` 기준으로 `floor((start_time - day_start_us) / 300000000)` 계산.
- 최종 `t5_day`는 0..287 범위만 유지.
- `t_hour = t5_day // 12`.

VM candidate:

- `(collection_id, instance_index)`별 usage row 수가 12개 이상이어야 함.
- deterministic hash order로 최대 10,000개.
- 현재 output VM count: 10,000.

Per-VM statistics:

- `arrival_t5 = min(t5_day)`
- `departure_t5 = max(t5_day) + 1`
- `lifetime_t5 = departure_t5 - arrival_t5`
- `p95_cpu_usage`
- `p95_mem_usage`
- `avg_cpu_usage`
- `avg_mem_usage`
- `max_cpu_usage`
- `max_mem_usage`

Resource request merge:

- `instance_events`에서 `priority`, `resource_request_cpu`, `resource_request_mem`을 `(collection_id, instance_index)`별 max로 요약해 merge.
- missing resource request는 0으로 처리.

Nominal resource:

```text
q_cpu = max(resource_request_cpu, 1.2 * p95_cpu_usage)
q_mem = max(resource_request_mem, 1.2 * p95_mem_usage)
```

`q_cpu`, `q_mem`은 최소 epsilon `1e-6` 이상으로 clip.

### Class proxy mapping

이 class는 cloud purchase class가 아니라 toy model용 proxy label이다.

Google 2019 priority tier interpretation:

- `0-99`: free / best-effort
- `100-115`: BE/BEB
- `116-119`: mid
- `>=120`: production

Toy class rule:

- `batch_candidate`: `collection_events.scheduler == SCHEDULER_BATCH`이면 priority와 무관하게 우선 배정.
- `on_demand`: batch가 아니고 `priority >= 120`.
- `spot`: batch가 아니고 `priority < 120` 또는 priority missing.

각 `vm_requests.csv` row에는 `class_rule`이 들어 있다.

현재 class count:

| class | count |
|---|---:|
| batch_candidate | 5,735 |
| on_demand | 3,476 |
| spot | 789 |

추가 model columns:

- `spot_revenue = 0.05 * q_cpu` for spot, otherwise 0.
- `migration_energy_coeff = 0.01 * q_cpu + 0.005 * q_mem` for on-demand, otherwise 0.

### Resource scale calibration

Server capacity는 `C_cpu = C_mem = 1.0`인 toy normalized capacity이다. 따라서 raw Google normalized usage/request와 toy server capacity가 맞도록 explicit scale calibration step이 있다.

Calibration:

- hourly observed usage를 만든다.
  - CPU: VM별 hourly mean
  - MEM: VM별 hourly max
- total server capacity:
  - CPU: `num_servers * 1.0`
  - MEM: `num_servers * 1.0`
- target peak utilization: 0.75.
- scale:
  - `cpu_scale = min(1.0, 0.75 * total_cpu_capacity / peak_cpu)`
  - `mem_scale = min(1.0, 0.75 * total_mem_capacity / peak_mem)`

현재 processed metadata:

```json
{
  "cpu_scale": 1.0,
  "mem_scale": 1.0,
  "target_peak_utilization": 0.75
}
```

즉 현재 200 servers pool에서는 추가 downscale이 필요 없었다.

Feasibility diagnostic 현재 값:

| metric | value |
|---|---:|
| peak_hourly_cpu_demand / total_server_cpu_capacity | 0.233831 |
| peak_hourly_mem_demand / total_server_mem_capacity | 0.210667 |

### Workload scenario generation

현재 scenario count: 10.

Uniform probabilities:

- `scenario_probabilities.csv`
- scenario 0..9 각각 probability 0.1.

5분 workload scenario:

- output: `vm_usage_5min_scenarios.csv`
- legacy duplicate alias: `vm_usage_scenarios.csv`
- columns:
  - `scenario_id`
  - `vm_id`
  - `t5_day`
  - `t_hour`
  - `cpu_usage`
  - `mem_usage`

Scenario rule:

- Scenario 0: observed usage 그대로.
- Scenario 1..N:
  - CPU noise: lognormal, `sigma = 0.22`, mean adjusted by `-0.5 * sigma^2`.
  - Memory noise: lognormal, `sigma = 0.08`, mean adjusted by `-0.5 * sigma^2`.
  - non-on-demand CPU는 `q_cpu`로 cap.
  - memory는 all VM classes에서 `q_mem`으로 cap.
  - 음수는 0으로 clip.

Hourly workload scenario:

- output: `vm_usage_hourly_scenarios.csv`
- Notion optimization model의 default workload input.
- aggregation:
  - CPU: hourly mean of 5-minute `cpu_usage`.
  - MEM: hourly max of 5-minute `mem_usage`.
- uniqueness:
  - one row per `(scenario_id, vm_id, t_hour)`.

현재 output:

| file | rows | columns | scenarios | VM count | time coverage |
|---|---:|---:|---:|---:|---|
| `vm_usage_5min_scenarios.csv` | 17,991,480 | 6 | 10 | 10,000 | `t5_day` 1..287, `t_hour` 0..23 |
| `vm_usage_hourly_scenarios.csv` | 1,624,930 | 5 | 10 | 10,000 | `t_hour` 0..23 |
| `vm_usage_scenarios.csv` | 17,991,480 | 6 | 10 | 10,000 | legacy duplicate of 5-minute file |

주의: absolute bucket range는 0..287이지만, 현재 selected pool의 실제 observed rows는 `t5_day = 1..287`에 존재한다. 이는 filter 후 selected workload에서 bucket 0 row가 없었다는 뜻이며, validator는 0..287 범위를 허용한다.

### Spot preemption scenario

Output:

- `spot_preemption_scenarios.csv`

Columns:

- `scenario_id`
- `vm_id`
- `t5_day`
- `t_hour`
- `active`
- `preempted`

Rule:

- spot VM만 대상.
- 실제 `instance_events.type`이 EVICT이면 해당 event time을 사용.
- Google trace v3에서 numeric `EVICT = 4`.
- string `"EVICT"`도 eviction으로 처리.
- once preempted, remaining horizon inactive.
- 실제 EVICT가 없으면 synthetic hazard 사용:
  - `hazard = min(0.08, 0.003 + max(0, 100 - priority) / 100 * 0.025)`
  - scenario 0에서는 synthetic preemption을 만들지 않고 observed baseline 유지.

현재 output:

| rows | spot VM count | scenarios | t5_day range |
|---:|---:|---:|---|
| 1,356,660 | 789 | 10 | 1..287 |

### Batch family abstraction

Batch candidate VM은 개별 placement 대상이 아니라 batch workload family로 추상화했다.

Output files:

- `batch_families.csv`
- `batch_workload.csv`

Family grouping:

- `class == batch_candidate`인 VM 대상.
- `q_cpu`, `q_mem`을 rounded bin으로 묶는다.
- family 수는 최대 10.
- 현재 family 수: 10.

`batch_families.csv` columns:

- `family_id`
- `q_cpu_B`
- `q_mem_B`
- `W_k`
- `rho_cpu_B`
- `rho_mem_B`
- `base_cpu`
- `base_mem`
- `startup_cpu`
- `startup_mem`

`batch_workload.csv` rule:

- observed batch usage를 `family_id, t_hour`로 집계.
- `workload_volume`: 해당 hour/family의 unique VM count.
- `cpu_workload`: sum of CPU usage.
- `mem_workload`: sum of MEM usage.
- scenario 1..N에는 lognormal perturbation:
  - workload/cpu scale sigma 0.12.
  - memory scale sigma 0.06.

현재 output:

| file | rows | columns |
|---|---:|---:|
| `batch_families.csv` | 10 | 10 |
| `batch_workload.csv` | 2,400 | 6 |

### Server file

Output:

- `servers.csv`

현재 server count: 200.

Columns:

- `server_id`
- `C_cpu`
- `C_mem`
- `E_idle`
- `E_cpu`
- `min_on_time`
- `min_off_time`

Values:

- `C_cpu = 1.0`
- `C_mem = 1.0`
- `E_idle = 0.35`
- `E_cpu = 0.65`
- `min_on_time = 1`
- `min_off_time = 1`

### Synthetic energy scenario

Output:

- `energy_scenarios.csv`

현재 output:

| rows | scenarios | hours |
|---:|---:|---:|
| 240 | 10 | 24 |

Columns:

- `scenario_id`
- `t_hour`
- `day_ahead_price`
- `real_time_price`
- `sell_price`
- `renewable_generation`
- `ess_capacity`
- `ess_charge_max`
- `ess_discharge_max`
- `ess_charge_efficiency`
- `ess_discharge_efficiency`

Generation rule:

- 24-hour synthetic profile.
- Day-ahead base price:
  - sinusoidal daily component
  - evening peak Gaussian component
  - minimum 20.
- Scenario 0:
  - no DA/RT/renewable noise.
- Scenario 1..N:
  - DA scalar noise: normal mean 0, sd 1.5.
  - RT hourly noise: normal mean 0, sd 4.0.
  - renewable noise: lognormal sigma 0.18.
- RT price lower bound: -20.
- Renewable:
  - solar-like bell curve centered around hour 13.
  - zero before 6 and after 19.
  - scale 0.75.
- Sell price:
  - `min(day_ahead * 0.85, real_time * 0.95)`.
- ESS:
  - `ess_capacity = 1.0`
  - `ess_charge_max = 0.25`
  - `ess_discharge_max = 0.25`
  - `ess_charge_efficiency = 0.92`
  - `ess_discharge_efficiency = 0.92`

### Model params

Output:

- `model_params.json`

현재 값:

```json
{
  "alpha": 0.95,
  "epsilon": 0.05,
  "soc_init": 0.5,
  "ess_charge_efficiency": 0.92,
  "ess_discharge_efficiency": 0.92
}
```

### Google processed output inventory

Processed directory:

```text
data/processed/notion_toy_google2019_v1_pool/
```

Files:

| file | current size | rows |
|---|---:|---:|
| `metadata.json` | 630 bytes | JSON |
| `model_params.json` | 126 bytes | JSON |
| `servers.csv` | 5,460 bytes | 200 |
| `vm_requests.csv` | 4,033,952 bytes | 10,000 |
| `vm_usage_5min_scenarios.csv` | 1,066,344,428 bytes | 17,991,480 |
| `vm_usage_hourly_scenarios.csv` | 90,187,658 bytes | 1,624,930 |
| `vm_usage_scenarios.csv` | 1,066,344,428 bytes | 17,991,480 |
| `spot_preemption_scenarios.csv` | 27,372,549 bytes | 1,356,660 |
| `batch_families.csv` | 1,775 bytes | 10 |
| `batch_workload.csv` | 159,357 bytes | 2,400 |
| `energy_scenarios.csv` | 23,261 bytes | 240 |
| `scenario_probabilities.csv` | 84 bytes | 10 |
| `toy_instance_summary.md` | 691 bytes | Markdown summary |

### Google validation

Validation script:

```bash
.venv/bin/python scripts/03_validate_outputs.py \
  --output_dir data/processed/notion_toy_google2019_v1_pool \
  --write_summary
```

Validation checks:

- required files exist:
  - `metadata.json`
  - `model_params.json`
  - `servers.csv`
  - `vm_requests.csv`
  - `vm_usage_5min_scenarios.csv`
  - `vm_usage_hourly_scenarios.csv`
  - `spot_preemption_scenarios.csv`
  - `batch_families.csv`
  - `batch_workload.csv`
  - `energy_scenarios.csv`
  - `scenario_probabilities.csv`
- no negative CPU/MEM in 5-minute or hourly usage.
- `q_cpu > 0`, `q_mem > 0`.
- 5-minute `t5_day` in 0..287.
- `t_hour` in 0..23.
- `arrival_t5 < departure_t5`.
- all usage VM IDs exist in `vm_requests.csv`.
- hourly uniqueness: one row per `(scenario_id, vm_id, t_hour)`.
- scenario probability scenario IDs match usage scenarios.
- scenario probabilities sum to 1.
- `model_params.json` contains `alpha`, `epsilon`, `soc_init`.
- every spot VM has a preemption/active path per scenario.
- every energy scenario has exactly 24 hourly rows.
- no negative `day_ahead_price`, `sell_price`, or `renewable_generation`.
- writes feasibility summary:
  - `peak_hourly_cpu_demand / total_server_cpu_capacity`
  - `peak_hourly_mem_demand / total_server_mem_capacity`

Current validation summary:

| metric | value |
|---|---:|
| batch_candidate VMs | 5,735 |
| on_demand VMs | 3,476 |
| spot VMs | 789 |
| usage rows 5min | 17,991,480 |
| usage rows hourly | 1,624,930 |
| average lifetime, 5-minute periods | 199.69 |
| p50 q_cpu | 0.011011 |
| p95 q_cpu | 0.061615 |
| p50 q_mem | 0.006622 |
| p95 q_mem | 0.040775 |
| batch families | 10 |
| servers | 200 |
| scenarios | 10 |
| peak hourly CPU demand / capacity | 0.233831 |
| peak hourly MEM demand / capacity | 0.210667 |

## 4. Cross-dataset alignment notes

### Time resolution

| dataset | raw/native | processed model input |
|---|---|---|
| NREL solar/wind | 5 minutes UTC | 5-minute CF, 30-minute CF/power |
| NYISO DA | 1 hour local NY time | forward-filled to 5-minute UTC, averaged to 30-minute |
| NYISO RT | 5 minutes local NY time, interval-end semantics | 5-minute UTC interval-start grid, averaged to 30-minute |
| Google usage | 5-minute buckets from trace microsecond time | 5-minute scenarios and hourly scenarios |
| Google energy | synthetic hourly | hourly |

### Recommended model input files

For a VM placement / energy procurement model, use:

- Renewable:
  - `data/processed/nrel/renewable_power_2019_30min_assumed_100mw.parquet`
  - or if only capacity factor needed: `data/processed/nrel/renewable_cf_2019_30min.parquet`
- Electricity price:
  - `data/processed/nyiso/electricity_price_nyiso_2019_30min.parquet`
  - use 5-minute file only if model is 5-minute.
- Google toy VM workload:
  - default workload input: `data/processed/notion_toy_google2019_v1_pool/vm_usage_hourly_scenarios.csv`
  - VM request metadata: `data/processed/notion_toy_google2019_v1_pool/vm_requests.csv`
  - spot active/preemption: `data/processed/notion_toy_google2019_v1_pool/spot_preemption_scenarios.csv`
  - batch abstraction: `batch_families.csv`, `batch_workload.csv`
  - synthetic energy: `energy_scenarios.csv`
  - scenario probabilities: `scenario_probabilities.csv`
  - scalar model params: `model_params.json`

### Important conceptual caveats

- NREL and NYISO are real public datasets, but the scaled renewable power trace is model-derived.
- Google ClusterData is a Borg cluster trace, not a VM cloud billing trace.
- The Google toy classes are proxy labels:
  - `on_demand`: production-tier priority proxy.
  - `spot`: lower-priority proxy.
  - `batch_candidate`: batch scheduler proxy.
- Google energy scenarios are synthetic and not tied to NYISO/NREL outputs.
- NREL renewable scenarios are deterministic historical resource-derived traces, not stochastic scenarios.
- Raw NYISO price traces preserve negative and extreme values. The Notion experiment loader separately floors model DA/RT purchase-price inputs at 0.01 USD/kWh and keeps the raw values in its audit.

## 5. Reproducibility checklist

To regenerate everything from raw/public sources:

1. Install dependencies:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

2. NREL:

```bash
export NREL_API_KEY="YOUR_KEY"
export NREL_API_EMAIL="YOUR_EMAIL"
.venv/bin/python scripts/fetch_and_build_nrel_2019.py --fetch --process
```

3. NYISO:

```bash
.venv/bin/python scripts/fetch_and_build_nyiso_prices_2019.py --fetch --process
```

4. Google ClusterData raw extract:

```bash
.venv/bin/python scripts/01_extract_google2019.py \
  --project_id YOUR_GCP_PROJECT \
  --cell a \
  --day_index 0 \
  --max_instances 10000 \
  --seed 42 \
  --output_dir data/raw/google2019_cell_a_day0_pool
```

5. Google toy build:

```bash
.venv/bin/python scripts/02_build_toy_instance.py \
  --raw_dir data/raw/google2019_cell_a_day0_pool \
  --output_dir data/processed/notion_toy_google2019_v1_pool \
  --max_instances 10000 \
  --seed 42 \
  --num_servers 200 \
  --num_scenarios 10
```

6. Google validation:

```bash
.venv/bin/python scripts/03_validate_outputs.py \
  --output_dir data/processed/notion_toy_google2019_v1_pool \
  --write_summary
```

7. Full test suite:

```bash
.venv/bin/python -m pytest -q
```

## 6. Current generated dataset quick inventory

### Raw

| group | path | files |
|---|---|---:|
| NREL solar | `data/raw/nrel/solar/2019/` | 10 CSV + 10 metadata JSON |
| NREL wind | `data/raw/nrel/wind/2019/` | 10 CSV + 10 metadata JSON |
| NYISO DA | `data/raw/nyiso/da_lbmp_zonal/2019/` | 12 ZIP |
| NYISO RT | `data/raw/nyiso/rt_lbmp_zonal_5min/2019/` | 12 ZIP |
| Google raw | `data/raw/google2019_cell_a_day0_pool/` | 3 parquet + 1 metadata JSON |

### Processed

| group | path | main files |
|---|---|---|
| NREL | `data/processed/nrel/` | CF 5min/30min parquet+csv.gz, power 30min parquet+csv.gz, summaries |
| NYISO | `data/processed/nyiso/` | price 5min/30min parquet+csv.gz, zone summary |
| Google toy | `data/processed/notion_toy_google2019_v1_pool/` | VM requests, usage scenarios, spot preemption, batch, servers, synthetic energy, params |
