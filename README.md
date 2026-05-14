# Previzma ML Service

Stateless FastAPI microservice used by the Previzma Spring Boot backend for sales forecasts and What-If simulations.

This service does not handle authentication, RBAC, tenants, users, CRUD, Supabase, or PostgreSQL. It receives business-ready payloads from Spring Boot and returns either a trained offline forecast model prediction or a statistical baseline fallback computed from the provided sales history.

FastAPI stays stateless: it never calls Spring Boot, never connects to Supabase, and never stores tenant data.

## Endpoints

- `GET /health`
- `POST /predict`
- `POST /backtest`
- `POST /model-comparison`
- `POST /simulate`

OpenAPI documentation is available at:

```text
http://localhost:8000/docs
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
uvicorn app.main:app --reload --port 8000
```

Spring Boot can call this service with:

```text
ML_BASE_URL=http://localhost:8000
ML_PREDICT_PATH=/predict
ML_SIMULATE_PATH=/simulate
```

By default, runtime looks for:

```text
app/models/forecast_model.joblib
app/models/model_metadata.json
```

You can override the artifact paths:

```powershell
$env:PREVIZMA_FORECAST_MODEL_PATH="C:\path\forecast_model.joblib"
$env:PREVIZMA_FORECAST_METADATA_PATH="C:\path\model_metadata.json"
```

Model promotion is gated by validation MAPE. The default threshold is `10.0`,
and can be overridden with:

```powershell
$env:PREVIZMA_MODEL_PROMOTION_MAPE_THRESHOLD="10.0"
```

If `validationMape` is missing, null, or above the threshold, `/predict` uses
`baseline-statistical-v1` even when a trained artifact exists. Internal
promotion statuses are `PROMOTED`, `NOT_PROMOTED`, `MISSING_METADATA`, and
`MISSING_MODEL`.

Offline training also compares target stabilization strategies before saving
the artifact. The candidates are `raw`, `log1p`, `clipped_raw`,
`clipped_log1p`, `baseline_ratio_log1p`, and
`clipped_baseline_ratio_log1p`. Validation metrics are always computed against
the real future revenue, not against clipped values. The selected
`targetStrategy` and all `targetStrategyCandidates` are written to
`model_metadata.json`.

## Test

```powershell
pytest
```

## Contract

All JSON fields use camelCase to match the existing Java Spring Boot client. Spring Boot is responsible for company scoping and for sending the relevant `salesHistory`.

### Forecast request

`POST /predict` receives product and segment context plus a positive forecast `horizon` in days. If a trained model artifact is available, promoted by metadata, supports the requested horizon, and the request has enough history, FastAPI uses it. Otherwise it falls back to `baseline-statistical-v1`. `salesHistory` can be empty, but the fallback will then return an explainable zero baseline with `confidenceScore` equal to `0.0`.

```json
{
  "companyId": 1,
  "productId": 10,
  "productName": "Industrial Pump",
  "productSku": "PUMP-001",
  "clientSegmentId": 20,
  "clientSegmentName": "Grand compte",
  "clientSegmentType": "GRAND_COMPTE",
  "horizon": 30,
  "calculationDate": "2026-05-06",
  "salesHistory": [
    {
      "saleDate": "2026-04-01",
      "quantity": 12,
      "amount": 2400.5,
      "confirmedOrder": true,
      "sourceStatus": "ERP_CONFIRMED"
    }
  ]
}
```

`POST /predict` returns:

```json
{
  "predictedValue": 2400.5,
  "confidenceScore": 0.18,
  "modelVersion": "baseline-statistical-v1",
  "calculationDate": "2026-05-06"
}
```

### Backtest request

`POST /backtest` evaluates the active forecast engine on historical sales only. It simulates past forecast dates: for each cutoff date, the model uses sales before the cutoff, predicts the next `horizon` days, and compares that prediction with the real sales observed in that period. If no trained artifact is available, `/backtest` uses `baseline-statistical-v1`.

FastAPI does not fetch missing history. If Spring Boot sends too little data, the service returns `testedSplits: 0`, null metrics, and `qualityLabel: "UNKNOWN"`.

```json
{
  "horizon": 30,
  "numberOfSplits": 6,
  "salesHistory": [
    {
      "saleDate": "2024-01-10",
      "quantity": 10,
      "amount": 480000,
      "confirmedOrder": true,
      "sourceStatus": "CONFIRMED"
    }
  ]
}
```

`POST /backtest` returns:

```json
{
  "modelVersion": "baseline-statistical-v1",
  "horizon": 30,
  "testedSplits": 6,
  "mae": 125000.0,
  "mape": 8.7,
  "rmse": 158000.0,
  "qualityLabel": "GOOD",
  "backtestWindows": [
    {
      "cutoffDate": "2025-10-01",
      "prediction": 1200000.0,
      "actual": 1100000.0,
      "absoluteError": 100000.0,
      "absolutePercentageError": 9.09
    }
  ]
}
```

Quality labels are based on MAPE:

- `EXCELLENT`: MAPE < 10
- `GOOD`: MAPE < 20
- `FAIR`: MAPE < 35
- `POOR`: MAPE >= 35
- `UNKNOWN`: not enough data or MAPE cannot be computed

### Model comparison request

`POST /model-comparison` evaluates available forecast engines on the same
backtest windows. This is useful before changing production behavior: it shows
whether the trained model is actually better than the transparent baseline for
the submitted `salesHistory`.

The endpoint compares:

- `baseline-statistical-v1`
- `lightgbm-window-v1` when `forecast_model.joblib` is available and supports the requested horizon

```json
{
  "horizon": 30,
  "numberOfSplits": 6,
  "salesHistory": [
    {
      "saleDate": "2024-01-10",
      "quantity": 10,
      "amount": 480000,
      "confirmedOrder": true,
      "sourceStatus": "CONFIRMED"
    }
  ]
}
```

`POST /model-comparison` returns:

```json
{
  "horizon": 30,
  "numberOfSplits": 6,
  "selectedModelVersion": "baseline-statistical-v1",
  "selectionMetric": "MAPE",
  "selectionReason": "baseline-statistical-v1 has lower MAPE than lightgbm-window-v1.",
  "candidates": [
    {
      "modelVersion": "baseline-statistical-v1",
      "testedSplits": 6,
      "mae": 125000.0,
      "mape": 8.7,
      "rmse": 158000.0,
      "qualityLabel": "GOOD"
    },
    {
      "modelVersion": "lightgbm-window-v1",
      "testedSplits": 6,
      "mae": 200000.0,
      "mape": 16.4,
      "rmse": 240000.0,
      "qualityLabel": "GOOD"
    }
  ]
}
```

How to read it:

- `candidates` contains one metric block per evaluated model.
- `testedSplits` should match across candidates because they use the same cutoff windows.
- `selectedModelVersion` is the currently recommended model for that history and horizon.
- `selectionMetric` is usually `MAPE`; it becomes `RMSE` when MAPE cannot be computed, `FALLBACK` when no trained model is available, and `UNKNOWN` when no comparable metric exists.
- `/predict` does not automatically switch strategy based on this endpoint yet. The comparison endpoint is intentionally advisory until product validation.

### Simulation request

`POST /simulate` accepts the same product and segment context. If `baselineValue` is present, the scenario is applied directly to it. Otherwise, the service computes a baseline from `salesHistory`.

Supported `scenarioType` values:

- `PRICE_CHANGE`
- `DEMAND_CHANGE`
- `SUPPLY_DELAY`
- `DISCOUNT_CAMPAIGN`

```json
{
  "companyId": 1,
  "userId": 7,
  "productId": 10,
  "productName": "Industrial Pump",
  "productSku": "PUMP-001",
  "clientSegmentId": 20,
  "clientSegmentName": "Grand compte",
  "clientSegmentType": "GRAND_COMPTE",
  "scenarioType": "PRICE_CHANGE",
  "inputChangePercent": 10,
  "baselineValue": 1000,
  "salesHistory": []
}
```

`POST /simulate` returns:

```json
{
  "baselineValue": 1000.0,
  "resultValue": 1100.0,
  "impactValue": 100.0,
  "impactPercent": 10.0,
  "modelVersion": "baseline-simulation-v1"
}
```

## Baseline Model

The forecast baseline:

- sorts `salesHistory` by `saleDate`
- aggregates revenue from `amount` by calendar month
- computes average monthly revenue
- compares older and recent periods to apply a damped trend
- projects the baseline over the requested horizon in days
- computes `confidenceScore` from history volume, month coverage, and regularity

The simulation baseline:

- uses `baselineValue` when Spring Boot provides it
- otherwise computes monthly baseline revenue from `salesHistory`
- applies the requested scenario with `inputChangePercent`
- returns `baselineValue`, `resultValue`, `impactValue`, and `impactPercent`
- clamps negative results to `0.0`

The backtest engine:

- sorts historical sales by `saleDate`
- selects several cutoff windows from the observed history
- uses sales before each cutoff as training history
- compares the forecast with actual revenue observed during the horizon
- returns MAE, MAPE, RMSE, a quality label, and per-window diagnostics

The model comparison engine:

- reuses the same historical windows for every candidate
- evaluates baseline and trained model candidates independently
- selects the lower MAPE when possible
- falls back to RMSE when MAPE is unavailable
- keeps `/predict` behavior unchanged

## Offline Training

Training starts from a CSV export, not from a database connection. FastAPI never connects to Supabase.

For demo data, use the official data pipeline documented in [data/README.md](data/README.md):

```text
Kaggle Online Retail II
  -> app.training.transform_kaggle_b2b
  -> data/processed/products.csv
  -> data/processed/client_segments.csv
  -> data/processed/sales.csv
  -> Supabase load
  -> canonical sales_export.csv
  -> train_model
```

Kaggle is raw material only. It must not become the business database.

Expected CSV columns:

```text
saleDate,productId,productName,productSku,clientSegmentId,clientSegmentName,clientSegmentType,quantity,amount,confirmedOrder,sourceStatus
```

Example Supabase export query:

```sql
select
  s.sale_date as "saleDate",
  p.id as "productId",
  p.name as "productName",
  p.sku as "productSku",
  cs.id as "clientSegmentId",
  cs.name as "clientSegmentName",
  cs.type as "clientSegmentType",
  s.quantity as "quantity",
  s.amount as "amount",
  s.confirmed_order as "confirmedOrder",
  s.source_status as "sourceStatus"
from sales s
join products p on p.id = s.product_id
join client_segments cs on cs.id = s.client_segment_id
where s.sale_date is not null;
```

Train the model:

```powershell
python -m app.training.train_model --input data/sales_export.csv --output app/models/forecast_model.joblib
```

Transform the Kaggle raw export into Previzma-compatible CSVs:

```powershell
python -m app.training.transform_kaggle_b2b --input data/raw/online_retail_II.csv --output-dir data/processed --max-output-rows 30000
```

The script:

- ignores `CANCELLED` rows
- creates supervised temporal windows for horizons `30`, `60`, and `90`
- trains LightGBM when available
- falls back to XGBoost, then scikit-learn HistGradientBoostingRegressor if needed
- writes `forecast_model.joblib` and `model_metadata.json`
- records validation MAE, MAPE, RMSE, training rows, algorithm, and model version
- compares target strategies `raw`, `log1p`, `clipped_raw`, `clipped_log1p`, `baseline_ratio_log1p`, and `clipped_baseline_ratio_log1p`
- saves the selected `targetStrategy` plus all candidate metrics

Current trained model version with the recommended dependency path:

```text
lightgbm-window-v1
```

## Forecast Engines

`baseline-statistical-v1` is transparent and deterministic. It is always available and is used when the trained artifact is missing or when a request has too little history.

`lightgbm-window-v1` is trained offline from exported CSV data. It uses shared feature engineering for training and runtime prediction, including revenue windows, quantity windows, trend factor, recency, horizon, month, product SKU hash, and segment type hash.

Promotion rule:

- `validationMape <= PREVIZMA_MODEL_PROMOTION_MAPE_THRESHOLD`: model can serve `/predict`
- missing/null `validationMape`: fallback baseline
- `validationMape` above threshold: fallback baseline
- `/model-comparison` still evaluates non-promoted models for diagnosis

## Model Limits

The LightGBM pipeline is a first production-shaped training path, but still V1. It uses historical sales windows only. It does not yet include external signals, inventory constraints, lead times, pricing tables, macro indicators, customer-level behavior, or automated model promotion.

Next step: enrich features, add stronger temporal validation, compare LightGBM vs XGBoost on real exports, and decide how Spring/Angular should expose model quality once this contract stabilizes.
