from datetime import date, timedelta

import joblib
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import ModelCandidateResponse, SalesHistoryItem
from app.services.model_comparison_service import (
    run_model_comparison,
    select_model_candidate,
)
from app.services.model_registry import MODEL_PATH_ENV, clear_forecast_model_cache
from app.training.features import FEATURE_NAMES

client = TestClient(app)


class FixedForecastModel:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, rows):
        return [self.value for _row in rows]


def sale(sale_date: date, amount: float) -> SalesHistoryItem:
    return SalesHistoryItem(
        sale_date=sale_date,
        quantity=1,
        amount=amount,
        confirmed_order=True,
        source_status="CONFIRMED",
    )


def sale_payload(sale_date: date, amount: float) -> dict:
    return {
        "saleDate": sale_date.isoformat(),
        "quantity": 1,
        "amount": amount,
        "confirmedOrder": True,
        "sourceStatus": "CONFIRMED",
    }


def spaced_history(amounts: list[float]) -> list[SalesHistoryItem]:
    start_date = date(2025, 1, 1)
    return [
        sale(start_date + timedelta(days=index * 31), amount)
        for index, amount in enumerate(amounts)
    ]


def spaced_payload(amounts: list[float]) -> list[dict]:
    start_date = date(2025, 1, 1)
    return [
        sale_payload(start_date + timedelta(days=index * 31), amount)
        for index, amount in enumerate(amounts)
    ]


def write_fixed_model(path, value: float) -> None:
    joblib.dump(
        {
            "model": FixedForecastModel(value),
            "featureNames": FEATURE_NAMES,
            "metadata": {
                "modelVersion": "lightgbm-window-v1",
                "validationMape": 12.0,
                "horizonsSupported": [30, 60, 90],
                "featureNames": FEATURE_NAMES,
            },
        },
        path,
    )


def test_model_comparison_uses_baseline_when_no_model_is_available() -> None:
    response = run_model_comparison(
        request_payload(
            spaced_history([1000.0, 1200.0, 1400.0, 1600.0]),
            number_of_splits=2,
        )
    )

    assert response.selected_model_version == "baseline-statistical-v1"
    assert response.selection_metric == "FALLBACK"
    assert len(response.candidates) == 1
    assert response.candidates[0].model_version == "baseline-statistical-v1"


def test_model_comparison_includes_lightgbm_when_model_is_available(
    tmp_path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "forecast_model.joblib"
    write_fixed_model(model_path, value=1000.0)
    monkeypatch.setenv(MODEL_PATH_ENV, str(model_path))
    clear_forecast_model_cache()

    response = run_model_comparison(
        request_payload(
            spaced_history([500.0, 500.0, 500.0, 1000.0, 1000.0, 1000.0]),
            number_of_splits=2,
        )
    )

    assert {candidate.model_version for candidate in response.candidates} == {
        "baseline-statistical-v1",
        "lightgbm-window-v1",
    }
    assert response.selected_model_version == "lightgbm-window-v1"
    assert response.selection_metric == "MAPE"
    assert "lower MAPE" in response.selection_reason

    clear_forecast_model_cache()


def test_model_comparison_candidates_use_same_number_of_splits(
    tmp_path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "forecast_model.joblib"
    write_fixed_model(model_path, value=1000.0)
    monkeypatch.setenv(MODEL_PATH_ENV, str(model_path))
    clear_forecast_model_cache()

    response = run_model_comparison(
        request_payload(
            spaced_history([500.0, 500.0, 500.0, 1000.0, 1000.0, 1000.0, 1000.0]),
            number_of_splits=3,
        )
    )

    tested_splits = {candidate.tested_splits for candidate in response.candidates}
    assert tested_splits == {3}

    clear_forecast_model_cache()


def test_model_selection_uses_rmse_when_mape_is_missing() -> None:
    selection = select_model_candidate(
        candidates=[
            ModelCandidateResponse(
                model_version="baseline-statistical-v1",
                tested_splits=2,
                mae=0.0,
                mape=None,
                rmse=0.0,
                quality_label="UNKNOWN",
            ),
            ModelCandidateResponse(
                model_version="lightgbm-window-v1",
                tested_splits=2,
                mae=5.0,
                mape=None,
                rmse=5.0,
                quality_label="UNKNOWN",
            ),
        ],
        trained_model_available=True,
    )

    assert selection.selected_model_version == "baseline-statistical-v1"
    assert selection.selection_metric == "RMSE"
    assert "lower RMSE" in selection.selection_reason


def test_model_comparison_endpoint_returns_camel_case_contract() -> None:
    response = client.post(
        "/model-comparison",
        json={
            "horizon": 30,
            "numberOfSplits": 2,
            "salesHistory": spaced_payload([1000.0, 1200.0, 1400.0, 1600.0]),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["horizon"] == 30
    assert body["numberOfSplits"] == 2
    assert body["selectedModelVersion"] == "baseline-statistical-v1"
    assert body["selectionMetric"] == "FALLBACK"
    assert "selectionReason" in body
    assert body["candidates"][0]["modelVersion"] == "baseline-statistical-v1"
    assert "testedSplits" in body["candidates"][0]
    assert "qualityLabel" in body["candidates"][0]


def test_model_comparison_endpoint_rejects_invalid_horizon() -> None:
    response = client.post(
        "/model-comparison",
        json={
            "horizon": 0,
            "numberOfSplits": 2,
            "salesHistory": spaced_payload([1000.0, 1200.0]),
        },
    )

    assert response.status_code == 422
    body = response.json()
    assert body["message"] == "Request validation failed"
    assert body["path"] == "/model-comparison"
    assert any(
        error["loc"] == ["body", "horizon"] and error["type"] == "greater_than"
        for error in body["validationErrors"]
    )


def request_payload(
    sales_history: list[SalesHistoryItem],
    number_of_splits: int = 6,
):
    from app.schemas import ModelComparisonRequest

    return ModelComparisonRequest(
        horizon=30,
        sales_history=sales_history,
        number_of_splits=number_of_splits,
    )
