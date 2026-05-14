from datetime import date

from app.schemas import SalesHistoryItem
from app.training.features import (
    FEATURE_NAMES,
    FeatureContext,
    build_forecast_features,
    feature_vector,
)


def sale(sale_date: date, amount: float, quantity: int = 1) -> SalesHistoryItem:
    return SalesHistoryItem(
        sale_date=sale_date,
        quantity=quantity,
        amount=amount,
        confirmed_order=True,
        source_status="CONFIRMED",
    )


def test_build_forecast_features_produces_stable_numeric_contract() -> None:
    features = build_forecast_features(
        sales_history=[
            sale(date(2025, 1, 1), 1000.0, 10),
            sale(date(2025, 2, 1), 1500.0, 12),
            sale(date(2025, 3, 2), 2000.0, 14),
        ],
        horizon=30,
        cutoff_date=date(2025, 4, 1),
        context=FeatureContext(
            product_sku="PUMP-001",
            client_segment_type="GRAND_COMPTE",
        ),
    )

    assert list(features) == FEATURE_NAMES
    assert features["horizon"] == 30.0
    assert features["month"] == 4.0
    assert features["quarter"] == 2.0
    assert features["sales_count_total"] == 3.0
    assert features["confirmed_sales_count"] == 3.0
    assert features["revenue_total"] == 4500.0
    assert features["revenue_last_30d"] == 2000.0
    assert features["quantity_total"] == 36.0
    assert features["quantity_last_30d"] == 14.0
    assert features["avg_order_amount"] == 1500.0
    assert features["baseline_prediction"] > 0
    assert features["days_since_last_sale"] == 30.0
    assert len(feature_vector(features)) == len(FEATURE_NAMES)
