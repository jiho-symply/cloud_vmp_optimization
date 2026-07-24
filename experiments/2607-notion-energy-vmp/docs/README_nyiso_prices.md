# NYISO Electricity Price Pipeline

This pipeline replaces any PJM Data Miner dependency with NYISO MIS public archived CSV/ZIP data. It does not use API keys, login, or PJM Data Miner.

The pipeline downloads 2019 NYISO zonal LBMP ZIP files, parses all 11 NYISO load zones, aligns day-ahead and real-time prices on a 5-minute UTC control grid, and writes 5-minute and 30-minute datasets for VM-placement experiments.

## Data Sources

Day-ahead zonal LBMP is hourly:

```text
https://mis.nyiso.com/public/csv/damlbmp/{YYYYMM}01damlbmp_zone_csv.zip
```

Real-time zonal LBMP is 5-minute:

```text
https://mis.nyiso.com/public/csv/realtime/{YYYYMM}01realtime_zone_csv.zip
```

The raw NYISO timestamps are in `America/New_York`. Output timestamps are timezone-aware UTC.

## Run

Use the repository virtual environment.

```bash
source .venv/bin/activate
.venv/bin/python scripts/fetch_and_build_nyiso_prices_2019.py --fetch --process
```

Other modes:

```bash
.venv/bin/python scripts/fetch_and_build_nyiso_prices_2019.py --dry-run
.venv/bin/python scripts/fetch_and_build_nyiso_prices_2019.py --fetch-only
.venv/bin/python scripts/fetch_and_build_nyiso_prices_2019.py --process-only
.venv/bin/python scripts/fetch_and_build_nyiso_prices_2019.py --fetch-only --force
```

## Outputs

Raw ZIPs:

- `data/raw/nyiso/da_lbmp_zonal/2019/{YYYYMM}01damlbmp_zone_csv.zip`
- `data/raw/nyiso/rt_lbmp_zonal_5min/2019/{YYYYMM}01realtime_zone_csv.zip`

Processed:

- `data/processed/nyiso/electricity_price_nyiso_2019_5min.parquet`
- `data/processed/nyiso/electricity_price_nyiso_2019_5min.csv.gz`
- `data/processed/nyiso/electricity_price_nyiso_2019_30min.parquet`
- `data/processed/nyiso/electricity_price_nyiso_2019_30min.csv.gz`
- `data/processed/nyiso/electricity_price_nyiso_2019_summary_by_zone.csv`

## Modeling Notes

NYISO DA zonal LBMP is hourly. NYISO RT zonal LBMP is 5-minute. The pipeline treats RT timestamps as interval ends, subtracts 5 minutes to get interval starts, and forward-fills DA hourly prices onto the RT 5-minute control grid.

Prices are wholesale LBMPs in USD/MWh. Negative prices are preserved and no clipping is applied. Values with absolute price above 1000 USD/MWh are reported by validation but not removed.

This dataset supports Kwon-style day-ahead procurement and real-time recourse experiments.
