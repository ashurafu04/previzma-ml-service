from datetime import date, timedelta

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def sales_history(amounts: list[float]) -> list[dict]:
    start_date = date(2025, 1, 1)
    return [
        {
            "saleDate": (start_date + timedelta(days=index * 31)).isoformat(),
            "quantity": 1,
            "amount": amount,
            "confirmedOrder": True,
            "sourceStatus": "CONFIRMED",
        }
        for index, amount in enumerate(amounts)
    ]


def backtest_payload() -> dict:
    return {
        "horizon": 30,
        "numberOfSplits": 3,
        "salesHistory": sales_history([1000.0] * 8),
    }


def test_backtest_returns_model_quality_metrics_with_camel_case_contract() -> None:
    response = client.post("/backtest", json=backtest_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["modelVersion"] == "baseline-statistical-v1"
    assert body["horizon"] == 30
    assert body["testedSplits"] == 3
    assert body["mae"] == 0.0
    assert body["mape"] == 0.0
    assert body["rmse"] == 0.0
    assert body["qualityLabel"] == "EXCELLENT"
    assert body["backtestWindows"][0] == {
        "cutoffDate": "2025-02-01",
        "prediction": 1000.0,
        "actual": 1000.0,
        "absoluteError": 0.0,
        "absolutePercentageError": 0.0,
    }


def test_backtest_returns_unknown_quality_when_history_is_empty() -> None:
    payload = backtest_payload()
    payload["salesHistory"] = []

    response = client.post("/backtest", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["testedSplits"] == 0
    assert body["mae"] is None
    assert body["mape"] is None
    assert body["rmse"] is None
    assert body["qualityLabel"] == "UNKNOWN"
    assert body["backtestWindows"] == []


def test_backtest_rejects_invalid_horizon() -> None:
    payload = backtest_payload()
    payload["horizon"] = 0

    response = client.post("/backtest", json=payload)

    assert response.status_code == 422
    body = response.json()
    assert body["message"] == "Request validation failed"
    assert body["path"] == "/backtest"
    assert any(
        error["loc"] == ["body", "horizon"] and error["type"] == "greater_than"
        for error in body["validationErrors"]
    )
