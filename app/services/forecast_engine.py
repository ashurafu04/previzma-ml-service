from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from app.schemas import SalesHistoryItem
from app.services.baseline import calculate_revenue_baseline, project_monthly_baseline
from app.services.model_registry import get_forecast_model
from app.training.features import FeatureContext, build_forecast_features

BASELINE_MODEL_VERSION = "baseline-statistical-v1"
MIN_MODEL_HISTORY_POINTS = 3


@dataclass(frozen=True)
class ForecastEngineResult:
    predicted_value: float
    confidence_score: float
    model_version: str


def forecast_revenue(
    sales_history: list[SalesHistoryItem],
    horizon: int,
    cutoff_date: date,
    product_sku: str = "UNKNOWN",
    client_segment_type: str = "UNKNOWN",
    allow_model: bool = True,
) -> ForecastEngineResult:
    history = sorted(sales_history, key=lambda item: item.sale_date)
    baseline = calculate_revenue_baseline(history)
    model = get_forecast_model() if allow_model else None

    if (
        model is not None
        and len(history) >= MIN_MODEL_HISTORY_POINTS
        and model.supports_horizon(horizon)
    ):
        features = build_forecast_features(
            sales_history=history,
            horizon=horizon,
            cutoff_date=cutoff_date,
            context=FeatureContext(
                product_sku=product_sku,
                client_segment_type=client_segment_type,
            ),
        )
        feature_row = pd.DataFrame(
            [[features[name] for name in model.feature_names]],
            columns=model.feature_names,
        )
        prediction = max(float(model.model.predict(feature_row)[0]), 0.0)
        return ForecastEngineResult(
            predicted_value=round(prediction, 2),
            confidence_score=_model_confidence_score(
                history_confidence=baseline.confidence_score,
                validation_mape=model.validation_mape,
            ),
            model_version=model.model_version,
        )

    predicted_value = project_monthly_baseline(
        monthly_value=baseline.monthly_value,
        horizon_days=horizon,
    )

    return ForecastEngineResult(
        predicted_value=round(predicted_value, 2),
        confidence_score=baseline.confidence_score,
        model_version=BASELINE_MODEL_VERSION,
    )


def _model_confidence_score(
    history_confidence: float,
    validation_mape: float | None,
) -> float:
    if validation_mape is None:
        return history_confidence

    validation_quality = max(0.0, min(1.0, 1 - validation_mape / 100))
    confidence = history_confidence * 0.6 + validation_quality * 0.4
    return round(min(confidence, 0.95), 2)
