# Lead Prioritization on Microsoft Fabric (PySpark, Medallion Architecture)

A lead conversion pipeline built for **Microsoft Fabric**: raw outbound sales calls land in a
lakehouse, flow through **bronze, silver, and gold** layers in PySpark with data quality gates, and
end in a model-ready feature table for scoring which leads a seller should call next.

The same code runs **locally** (Spark in local mode, Parquet tables) and **in a Fabric notebook**
(managed Delta tables that Power BI can read in Direct Lake mode), so every layer is testable in CI.

> **Status:** the data foundation (bronze, silver, gold, quality checks, tests) is complete.
> The conversion model, calibration, SHAP explanations, and Power BI report are in progress.

## Data

UCI **Bank Marketing** dataset: 41,188 outbound sales calls by a Portuguese bank, May 2008 to
November 2010, with whether each lead converted (CC BY 4.0, see `data/raw/SOURCE.md`).

## Medallion layers

| Layer | Table | What happens |
|---|---|---|
| Bronze | `bronze_bank_calls` | Raw file landed as-is (all strings) with lineage columns: `source_row_number`, `source_file`, `ingested_at` |
| Silver | `silver_calls` | Typed and renamed columns, sentinel values handled, exact duplicates removed, call dates reconstructed |
| Gold | `gold_lead_features` | Pre-call features plus label, one row per lead, ready for modeling |
| Gold | `gold_campaign_monthly` | Monthly calls, conversions, and economic indicators for reporting and drift analysis |

## Data quality decisions

- **Reconstructing dates.** The file is in date order but only has the month. Bronze captures the
  original row order before anything can reorder it, and silver starts a new year whenever the month
  number goes backwards. A quality check confirms the result matches the documented range
  (May 2008 to November 2010) exactly.
- **Sentinel values.** `pdays = 999` means "never contacted before," not 999 days, so it becomes null
  plus a `was_previously_contacted` flag.
- **Duplicates.** 12 rows that are exact copies on every field are removed, keeping the first.
- **Leakage guard.** `call_duration_sec` is only known after the call ends (a 0-second call is always a
  non-conversion), so it is excluded from the gold feature table and a test enforces it.

Nine quality checks run between layers and stop the pipeline on failure (`src/leadscore/quality.py`).

## What the gold layer already shows

The data covers the 2008 financial crisis, and the campaign changed completely as rates fell:

| Year | Calls | Conversion rate | Avg 3-month Euribor | Share of calls to past leads |
|---|---|---|---|---|
| 2008 | 27,682 | 4.8% | 4.6% | 0.1% |
| 2009 | 11,436 | 19.5% | 1.0% | 13.9% |
| 2010 | 2,058 | 52.1% | 0.8% | 35.2% |

The bank cut call volume sharply and shifted to re-contacting past leads, and conversion rose more
than tenfold. A model trained on 2008 behavior will not transfer to 2010 without retraining, which is
why the modeling stage uses time-based validation instead of random splits.

## Run it

```bash
pip install -e ".[local]"             # needs Java 11+ for PySpark
python -m pytest -q tests             # unit tests for the transforms
python scripts/run_data_pipeline.py   # bronze -> silver -> gold with quality gates
```

**In Microsoft Fabric:** upload `data/raw/bank-additional-full.csv` to the lakehouse at `Files/raw/`,
attach the `leadscore` wheel (`python -m build`) through a Fabric Environment, and run
`notebooks/01_build_lakehouse.ipynb`. Tables are written as Delta with `saveAsTable`.

## Project layout

```
src/leadscore/lakehouse.py   # read/write tables the same way locally and in Fabric
src/leadscore/pipeline.py    # bronze, silver, gold transforms
src/leadscore/quality.py     # data quality checks between layers
notebooks/                   # Fabric notebooks
tests/                       # PySpark unit tests
```

## Roadmap

- LightGBM conversion model with rolling time-based backtests and monthly retraining
- Probability calibration so scores can be used for pipeline forecasting
- SHAP reason codes per lead for sellers
- Ranked call list and lift analysis
- Power BI report on the gold tables (Direct Lake)
