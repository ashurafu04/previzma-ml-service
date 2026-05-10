# Previzma ML Service

Stateless FastAPI microservice used by the Previzma Spring Boot backend for sales forecasts and What-If simulations.

This service does not handle authentication, RBAC, tenants, users, CRUD, Supabase, PostgreSQL, or model training. It receives business-ready payloads from Spring Boot and returns a real statistical baseline computed from the provided sales history.

FastAPI stays stateless: it never calls Spring Boot, never connects to Supabase, and never stores tenant data.

## Endpoints

- `GET /health`
- `POST /predict`
- `POST /backtest`
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

## Test

```powershell
pytest
```

## Contract

All JSON fields use camelCase to match the existing Java Spring Boot client. Spring Boot is responsible for company scoping and for sending the relevant `salesHistory`.

### Forecast request

`POST /predict` receives product and segment context plus a positive forecast `horizon` in days. `salesHistory` can be empty, but the model will then return an explainable zero baseline with `confidenceScore` equal to `0.0`.

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

`POST /backtest` evaluates the `baseline-statistical-v1` forecast on historical sales only. It simulates past forecast dates: for each cutoff date, the model trains on sales before the cutoff, predicts the next `horizon` days, and compares that prediction with the real sales observed in that period.

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

## Model Limits

This is a statistical baseline, not an XGBoost or LightGBM model. It is deterministic, transparent, and useful for integration and early product validation, but it does not learn seasonal effects, product interactions, macro signals, stock constraints, or customer-specific behavior.

Next step: replace or complement this baseline with XGBoost/LightGBM, feature engineering, backtesting, and model evaluation metrics while keeping the existing `/predict` and `/simulate` contracts stable.
