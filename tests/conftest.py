import pytest

from app.services.model_registry import MODEL_PATH_ENV, clear_forecast_model_cache


@pytest.fixture(autouse=True)
def isolate_forecast_model_artifact(monkeypatch, tmp_path):
    monkeypatch.setenv(MODEL_PATH_ENV, str(tmp_path / "missing_forecast_model.joblib"))
    clear_forecast_model_cache()

    yield

    clear_forecast_model_cache()
