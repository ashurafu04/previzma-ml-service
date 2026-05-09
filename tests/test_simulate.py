from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def simulation_payload() -> dict:
    return {
        "companyId": 1,
        "userId": 7,
        "productId": 10,
        "productName": "Industrial Pump",
        "productSku": "PUMP-001",
        "clientSegmentId": 20,
        "clientSegmentName": "Grand compte",
        "clientSegmentType": "GRAND_COMPTE",
        "scenarioType": "PRICE_CHANGE",
        "inputChangePercent": 10.0,
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


def test_simulate_returns_result_with_camel_case_contract() -> None:
    response = client.post("/simulate", json=simulation_payload())

    assert response.status_code == 200
    assert response.json() == {
        "resultValue": 2640.55,
        "modelVersion": "mock-simulation-v1",
    }


def test_simulate_returns_zero_without_history() -> None:
    payload = simulation_payload()
    payload["salesHistory"] = []

    response = client.post("/simulate", json=payload)

    assert response.status_code == 200
    assert response.json()["resultValue"] == 0.0


def test_simulate_rejects_invalid_payload() -> None:
    payload = simulation_payload()
    payload["salesHistory"][0]["amount"] = -1

    response = client.post("/simulate", json=payload)

    assert response.status_code == 422
    body = response.json()
    assert body["message"] == "Request validation failed"
    assert body["path"] == "/simulate"
    assert body["requestMetadata"]["method"] == "POST"
    assert body["requestMetadata"]["path"] == "/simulate"
    assert body["requestMetadata"]["contentType"] == "application/json"
    assert body["requestMetadata"]["bodyWasEmpty"] is False
    assert body["receivedBody"] == payload
    assert any(
        error["loc"] == ["body", "salesHistory", 0, "amount"]
        for error in body["validationErrors"]
    )
