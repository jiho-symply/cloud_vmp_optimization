# NREL/NLR Renewable Trace Pipeline

This pipeline fetches 2019 5-minute solar and wind resource traces for 10 U.S. data-center proxy regions, converts them into model-derived capacity factors, aggregates them to 30-minute VM-placement inputs, and writes scaled renewable-power traces for initial cloud data-center experiments.

The raw files are NREL/NLR resource traces. They are not metered power-plant output. The processed files contain capacity factors and scaled power traces produced from the assumptions in [configs/nrel_renewables_2019_v0.yaml](../configs/nrel_renewables_2019_v0.yaml).

## Environment

Use the repository virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Required environment variables:

```bash
export NREL_API_KEY="YOUR_KEY"
export NREL_API_EMAIL="YOUR_EMAIL"
```

`NLR_API_KEY` is accepted as a fallback alias for `NREL_API_KEY`. Optional metadata variables are `NREL_API_FULL_NAME` and `NREL_API_AFFILIATION`. The script never prints the full API key.

## Run

Preview the point-by-point requests without network calls:

```bash
.venv/bin/python scripts/fetch_and_build_nrel_2019.py --dry-run
```

Fetch raw CSVs and build processed datasets:

```bash
.venv/bin/python scripts/fetch_and_build_nrel_2019.py --fetch --process
```

Other supported modes:

```bash
.venv/bin/python scripts/fetch_and_build_nrel_2019.py --fetch-only
.venv/bin/python scripts/fetch_and_build_nrel_2019.py --process-only
.venv/bin/python scripts/fetch_and_build_nrel_2019.py --fetch-only --force
```

The downloader uses direct `.csv` API requests, one WKT `POINT(lon lat)` at a time and one year at a time. It avoids asynchronous MULTIPOINT archive download.

## Outputs

Raw solar:

- `data/raw/nrel/solar/2019/{site_id}.csv`
- `data/raw/nrel/solar/2019/{site_id}.metadata.json`

Raw wind:

- `data/raw/nrel/wind/2019/{site_id}.csv`
- `data/raw/nrel/wind/2019/{site_id}.metadata.json`

Processed:

- `data/processed/nrel/renewable_cf_2019_5min.parquet`
- `data/processed/nrel/renewable_cf_2019_5min.csv.gz`
- `data/processed/nrel/renewable_cf_2019_30min.parquet`
- `data/processed/nrel/renewable_cf_2019_30min.csv.gz`
- `data/processed/nrel/renewable_power_2019_30min_assumed_100mw.parquet`
- `data/processed/nrel/renewable_power_2019_30min_assumed_100mw.csv.gz`
- `data/processed/nrel/renewable_cf_2019_summary_by_site.csv`
- `data/processed/nrel/site_capacity_assumptions_2019_assumed_100mw.csv`

## Model Assumptions

- Year: `2019`
- Raw interval: `5` minutes
- Placement interval: `30` minutes
- Timezone: UTC
- Leap day: false
- Expected raw rows per site: `105120`
- Solar: 1 MWac fixed-tilt PV, tilt equal to latitude, south-facing azimuth 180 degrees, DC/AC ratio 1.2, 14% losses, PVWatts DC and inverter models.
- Wind: GE 1.5 MW-class simple turbine curve at 80 m, cut-in 3.5 m/s, rated 12.0 m/s, cut-out 25.0 m/s, 0.90 loss factor.
- Scaling: each site assumes 100 MW mean demand, renewable-to-demand ratio 2.0, solar/wind energy mix 20%/80%.

## Validation

The parser requires NREL/NLR-style two-row metadata CSVs followed by a `Year,Month,Day,Hour,Minute,...` header. It validates strict UTC timestamp ordering, uniqueness, expected 2019 no-leap row count, first/last timestamps, and capacity-factor bounds.

Run network-free tests:

```bash
pytest
```
