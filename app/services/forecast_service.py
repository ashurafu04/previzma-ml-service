from app.schemas import ForecastRequest, ForecastResponse
from app.services.baseline import calculate_revenue_baseline, project_monthly_baseline

MODEL_VERSION = "baseline-statistical-v1"


def predict_forecast(request: ForecastRequest) -> ForecastResponse:
    baseline = calculate_revenue_baseline(request.sales_history)
    predicted_value = project_monthly_baseline(
        monthly_value=baseline.monthly_value,
        horizon_days=request.horizon,
    )

    return ForecastResponse(
        predicted_value=round(predicted_value, 2),
        confidence_score=baseline.confidence_score,
        model_version=MODEL_VERSION,
        calculation_date=request.calculation_date,
    )
