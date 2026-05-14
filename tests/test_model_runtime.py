import joblib
from math import log1p

from fastapi.testclient import TestClient

from app.main import app
from app.services.model_registry import (
    MODEL_PATH_ENV,
    MISSING_METADATA,
    MISSING_MODEL,
    NOT_PROMOTED,
    PROMOTED,
    PROMOTION_MAPE_THRESHOLD_ENV,
    clear_forecast_model_cache,
    get_forecast_model,
    promotion_status,
)
from app.training.features import FEATURE_NAMES

client = TestClient(app)


class FixedForecastModel:
    def __init__(self, value: float) -> None:
        self.value = value
        self.received_columns: list[str] | None = None
        self.received_type: str | None = None

    def predict(self, rows):
        self.received_columns = list(getattr(rows, "columns", []))
        self.received_type = type(rows).__name__
        return [self.value for _row in rows]


def forecast_payload() -> dict:
    return {
        "companyId": 1,
        "productId": 10,
        "productName": "Industrial Pump",
        "productSku": "PUMP-001",
        "clientSegmentId": 20,
        "clientSegmentName": "Grand compte",
        "clientSegmentType": "GRAND_COMPTE",
        "horizon": 30,
        "calculationDate": "2025-05-01",
        "salesHistory": [
            sale_payload("2025-01-01", 1000.0),
            sale_payload("2025-02-01", 1200.0),
            sale_payload("2025-03-01", 1400.0),
        ],
    }


def sale_payload(sale_date: str, amount: float) -> dict:
    return {
        "saleDate": sale_date,
        "quantity": 1,
        "amount": amount,
        "confirmedOrder": True,
        "sourceStatus": "CONFIRMED",
    }


def write_fixed_model(
    path,
    value: float = 4321.0,
    horizons_supported: list[int] | None = None,
    validation_mape: float | None = 8.0,
    include_validation_mape: bool = True,
    target_strategy: str = "raw",
) -> None:
    metadata = {
        "modelVersion": "lightgbm-window-v1",
        "horizonsSupported": horizons_supported or [30, 60, 90],
        "featureNames": FEATURE_NAMES,
        "targetStrategy": target_strategy,
        "targetClipLower": None,
        "targetClipUpper": None,
    }
    if include_validation_mape:
        metadata["validationMape"] = validation_mape

    joblib.dump(
        {
            "model": FixedForecastModel(value),
            "featureNames": FEATURE_NAMES,
            "metadata": metadata,
        },
        path,
    )


def test_predict_falls_back_to_baseline_when_model_is_absent(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(MODEL_PATH_ENV, str(tmp_path / "missing.joblib"))
    clear_forecast_model_cache()

    response = client.post("/predict", json=forecast_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["modelVersion"] == "baseline-statistical-v1"
    assert body["predictedValue"] != 4321.0
    assert promotion_status() == MISSING_MODEL

    clear_forecast_model_cache()


def test_predict_uses_trained_model_when_artifact_is_present(
    tmp_path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "forecast_model.joblib"
    write_fixed_model(model_path)
    monkeypatch.setenv(MODEL_PATH_ENV, str(model_path))
    clear_forecast_model_cache()

    response = client.post("/predict", json=forecast_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["predictedValue"] == 4321.0
    assert body["modelVersion"] == "lightgbm-window-v1"
    assert body["confidenceScore"] > 0
    loaded_model = get_forecast_model()
    assert loaded_model is not None
    assert loaded_model.model.received_type == "DataFrame"
    assert loaded_model.model.received_columns == FEATURE_NAMES

    clear_forecast_model_cache()


def test_predict_inverse_transforms_log_target_strategy(
    tmp_path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "forecast_model.joblib"
    write_fixed_model(model_path, value=log1p(4321.0), target_strategy="log1p")
    monkeypatch.setenv(MODEL_PATH_ENV, str(model_path))
    clear_forecast_model_cache()

    response = client.post("/predict", json=forecast_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["predictedValue"] == 4321.0
    assert body["modelVersion"] == "lightgbm-window-v1"

    clear_forecast_model_cache()


def test_predict_falls_back_when_model_metadata_has_no_validation_mape(
    tmp_path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "forecast_model.joblib"
    write_fixed_model(model_path, include_validation_mape=False)
    monkeypatch.setenv(MODEL_PATH_ENV, str(model_path))
    clear_forecast_model_cache()

    response = client.post("/predict", json=forecast_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["modelVersion"] == "baseline-statistical-v1"
    assert promotion_status() == MISSING_METADATA

    clear_forecast_model_cache()


def test_predict_falls_back_when_model_validation_mape_is_above_threshold(
    tmp_path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "forecast_model.joblib"
    write_fixed_model(model_path, validation_mape=21.49)
    monkeypatch.setenv(MODEL_PATH_ENV, str(model_path))
    clear_forecast_model_cache()

    response = client.post("/predict", json=forecast_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["modelVersion"] == "baseline-statistical-v1"
    assert body["predictedValue"] != 4321.0
    assert promotion_status() == NOT_PROMOTED

    clear_forecast_model_cache()


def test_predict_uses_model_when_validation_mape_is_under_configured_threshold(
    tmp_path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "forecast_model.joblib"
    write_fixed_model(model_path, validation_mape=12.0)
    monkeypatch.setenv(MODEL_PATH_ENV, str(model_path))
    monkeypatch.setenv(PROMOTION_MAPE_THRESHOLD_ENV, "15.0")
    clear_forecast_model_cache()

    response = client.post("/predict", json=forecast_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["predictedValue"] == 4321.0
    assert body["modelVersion"] == "lightgbm-window-v1"
    assert promotion_status() == PROMOTED

    clear_forecast_model_cache()


def test_predict_falls_back_when_horizon_is_not_supported(
    tmp_path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "forecast_model.joblib"
    write_fixed_model(model_path, horizons_supported=[30])
    monkeypatch.setenv(MODEL_PATH_ENV, str(model_path))
    clear_forecast_model_cache()
    payload = forecast_payload()
    payload["horizon"] = 120

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["modelVersion"] == "baseline-statistical-v1"
    assert body["predictedValue"] != 4321.0

    clear_forecast_model_cache()


def test_backtest_remains_functional_with_trained_model(
    tmp_path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "forecast_model.joblib"
    write_fixed_model(model_path, value=1000.0)
    monkeypatch.setenv(MODEL_PATH_ENV, str(model_path))
    clear_forecast_model_cache()

    response = client.post(
        "/backtest",
        json={
            "horizon": 30,
            "numberOfSplits": 2,
            "salesHistory": [
                sale_payload("2025-01-01", 1000.0),
                sale_payload("2025-02-01", 1000.0),
                sale_payload("2025-03-04", 1000.0),
                sale_payload("2025-04-04", 1000.0),
                sale_payload("2025-05-05", 1000.0),
                sale_payload("2025-06-05", 1000.0),
                sale_payload("2025-07-06", 1000.0),
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["modelVersion"] == "lightgbm-window-v1"
    assert body["testedSplits"] == 2
    assert body["backtestWindows"]

    clear_forecast_model_cache()
