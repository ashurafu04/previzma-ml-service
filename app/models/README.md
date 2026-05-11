# Previzma Forecast Model Artifacts

This directory is the default location for offline training outputs:

- `forecast_model.joblib`
- `model_metadata.json`

The FastAPI service loads these files lazily at runtime when they exist. If no
artifact is present, `/predict` and `/backtest` automatically fall back to
`baseline-statistical-v1`.

Generated model artifacts are intentionally ignored by Git.
