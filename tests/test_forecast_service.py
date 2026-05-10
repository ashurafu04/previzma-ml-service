from datetime import date

from app.schemas import ForecastRequest, SalesHistoryItem
from app.services.forecast_service import predict_forecast


def forecast_request(
    sales_history: list[SalesHistoryItem],
    horizon: int = 30,
) -> ForecastRequest:
    return ForecastRequest(
        company_id=1,
        product_id=10,
        product_name="Industrial Pump",
        product_sku="PUMP-001",
        client_segment_id=20,
        client_segment_name="Grand compte",
        client_segment_type="GRAND_COMPTE",
        horizon=horizon,
        calculation_date=date(2026, 5, 6),
        sales_history=sales_history,
    )


def sale(sale_date: date, amount: float) -> SalesHistoryItem:
    return SalesHistoryItem(
        sale_date=sale_date,
        quantity=1,
        amount=amount,
        confirmed_order=True,
        source_status="ERP_CONFIRMED",
    )


def test_forecast_with_empty_history_returns_explainable_zero_baseline() -> None:
    response = predict_forecast(forecast_request([]))

    assert response.predicted_value == 0.0
    assert response.confidence_score == 0.0
    assert response.model_version == "baseline-statistical-v1"


def test_forecast_with_one_sale_uses_observed_monthly_revenue() -> None:
    response = predict_forecast(forecast_request([sale(date(2026, 4, 1), 1200.0)]))

    assert response.predicted_value == 1200.0
    assert response.confidence_score == 0.18


def test_forecast_with_growing_history_applies_positive_trend() -> None:
    response = predict_forecast(
        forecast_request(
            [
                sale(date(2026, 1, 1), 1000.0),
                sale(date(2026, 2, 1), 2000.0),
                sale(date(2026, 3, 1), 3000.0),
            ]
        )
    )

    assert response.predicted_value == 2500.0
    assert response.predicted_value > 2000.0


def test_confidence_score_increases_with_more_regular_history() -> None:
    short_history = [sale(date(2026, 1, 1), 1000.0)]
    longer_history = [
        sale(date(2025, 10, 1), 1000.0),
        sale(date(2025, 11, 1), 1100.0),
        sale(date(2025, 12, 1), 1200.0),
        sale(date(2026, 1, 1), 1300.0),
        sale(date(2026, 2, 1), 1400.0),
        sale(date(2026, 3, 1), 1500.0),
    ]

    short_response = predict_forecast(forecast_request(short_history))
    longer_response = predict_forecast(forecast_request(longer_history))

    assert longer_response.confidence_score > short_response.confidence_score
