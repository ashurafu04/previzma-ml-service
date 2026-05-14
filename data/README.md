# Previzma Data Pipeline

This directory is for reproducible data preparation. Kaggle is treated as raw
transactional material, not as the Previzma business database.

Official flow:

```text
Kaggle raw CSV
  -> business transformation
  -> Supabase-compatible products / client_segments / sales CSVs
  -> Supabase load
  -> canonical sales_export.csv with real IDs
  -> FastAPI training
```

## Raw Source

Recommended source:

```text
Online Retail II UCI
https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci
```

Place the raw CSV here:

```text
data/raw/online_retail_II.csv
```

Raw Kaggle files are ignored by Git.

## Transform

Run:

```powershell
python -m app.training.transform_kaggle_b2b `
  --input data/raw/online_retail_II.csv `
  --output-dir data/processed `
  --max-output-rows 20000
```

Generated files:

- `data/processed/products.csv`
- `data/processed/client_segments.csv`
- `data/processed/sales.csv`
- `data/processed/previzma_b2b_sales_enriched.csv`
- `data/processed/sales_export.csv`
- `data/processed/dataset_summary.json`
- `data/processed/load_supabase.sql`

Use this if you explicitly want to refresh the local ML export:

```powershell
python -m app.training.transform_kaggle_b2b `
  --input data/raw/online_retail_II.csv `
  --output-dir data/processed `
  --sales-export-output data/sales_export.csv `
  --max-output-rows 20000
```

## Business Mapping

The transformer maps retail transaction patterns into Previzma business concepts:

- `StockCode` / `Description` -> industrial product family
- `Customer ID` / `Country` / purchase volume -> B2B client segment
- `InvoiceDate` -> `saleDate` rebased into 2021-2026
- `Quantity` -> adjusted B2B quantity
- `Quantity * Price` -> demand signal used to produce realistic MAD amounts
- invoices starting with `C`, negative quantity, or negative price -> `CANCELLED`

Target products:

- Industrial Pump X200
- Smart Conveyor S400
- Hydraulic Valve HV90
- Packaging Robot PR300
- Air Compressor AC700
- Industrial Sensor IS50
- Control Panel CP120
- Spare Parts Kit SPK

Target segments:

- Grand comptes Maroc / `GRAND_COMPTE`
- PME industrielles / `PME`
- Distributeurs Afrique du Nord / `DISTRIBUTEUR`

## Supabase Load

The generated `load_supabase.sql` contains psql `\copy` commands and the final
canonical export query.

Supabase SQL Editor does not support `\copy`; use either:

- Supabase Table Editor CSV import, table by table
- `psql` from a trusted local environment

Recommended order:

1. `products.csv`
2. `client_segments.csv`
3. `sales.csv`

Then export the canonical ML CSV using the query in `load_supabase.sql`.

## Train

After Supabase has real IDs and `data/sales_export.csv` is refreshed:

```powershell
python -m app.training.train_model --input data/sales_export.csv --output app/models/forecast_model.joblib
```

Goal:

- ideal `validationMape < 10`
- acceptable V1 demo `validationMape < 15`
- otherwise keep `baseline-statistical-v1` promoted through the model gate

## Limits

This transformation preserves transaction timing and demand patterns, but the
industrial product names, MAD pricing, and B2B segments are synthetic mappings.
It is suitable for demo, integration, and ML pipeline validation, not for real
commercial decision-making without replacement by actual ERP/Sales data.
