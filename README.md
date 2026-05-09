# Previzma ML Service

Stateless FastAPI microservice used by the Previzma Spring Boot backend for sales forecasts and What-If simulations.

This service does not handle authentication, RBAC, tenants, users, CRUD, Supabase, PostgreSQL, or real model training. It only exposes deterministic mock ML endpoints for the backend integration.

## Endpoints

- `GET /health`
- `POST /predict`
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

All JSON fields use camelCase to match the existing Java Spring Boot client.

`POST /predict` returns:

```json
{
  "predictedValue": 1234.56,
  "confidenceScore": 0.75,
  "modelVersion": "mock-forecast-v1",
  "calculationDate": "2026-05-06"
}
```

`POST /simulate` returns:

```json
{
  "resultValue": 2450.75,
  "modelVersion": "mock-simulation-v1"
}
```
