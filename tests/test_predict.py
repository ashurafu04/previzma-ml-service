import logging

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


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
        "calculationDate": "2026-05-06",
        "salesHistory": [
            {
                "saleDate": "2026-04-01",
                "quantity": 12,
                "amount": 2400.50,
                "confirmedOrder": True,
                "sourceStatus": "ERP_CONFIRMED",
            }
        ],
    }


def test_predict_returns_forecast_with_camel_case_contract() -> None:
    response = client.post("/predict", json=forecast_payload())

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "predictedValue": 72015.0,
        "confidenceScore": 0.75,
        "modelVersion": "mock-forecast-v1",
        "calculationDate": "2026-05-06",
    }


def test_predict_returns_zero_without_history() -> None:
    payload = forecast_payload()
    payload["salesHistory"] = []

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json()["predictedValue"] == 0.0


def test_predict_rejects_invalid_payload_with_diagnostics(caplog) -> None:
    payload = forecast_payload()
    payload.pop("productId")

    with caplog.at_level(logging.WARNING, logger="uvicorn.error"):
        response = client.post("/predict", json=payload)

    assert response.status_code == 422
    body = response.json()
    assert body["message"] == "Request validation failed"
    assert body["path"] == "/predict"
    assert body["requestMetadata"]["method"] == "POST"
    assert body["requestMetadata"]["path"] == "/predict"
    assert body["requestMetadata"]["contentType"] == "application/json"
    assert body["requestMetadata"]["bodyWasEmpty"] is False
    assert body["receivedBody"] == payload
    assert any(
        error["loc"] == ["body", "productId"] and error["type"] == "missing"
        for error in body["validationErrors"]
    )
    assert "/predict" in caplog.text
    assert "productId" in caplog.text
    assert "Industrial Pump" in caplog.text


def test_predict_rejects_empty_body_with_diagnostics() -> None:
    response = client.post("/predict")

    assert response.status_code == 422
    body = response.json()
    assert body["message"] == "Request validation failed"
    assert body["path"] == "/predict"
    assert body["requestMetadata"]["method"] == "POST"
    assert body["requestMetadata"]["bodyWasEmpty"] is True
    assert body["receivedBody"] is None
    assert any(
        error["loc"] == ["body"] and error["type"] == "missing"
        for error in body["validationErrors"]
    )
