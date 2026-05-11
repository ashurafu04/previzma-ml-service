from app.schemas import ForecastRequest, ForecastResponse
from app.services.forecast_engine import BASELINE_MODEL_VERSION, forecast_revenue

MODEL_VERSION = BASELINE_MODEL_VERSION


def predict_forecast(request: ForecastRequest) -> ForecastResponse:
    prediction = forecast_revenue(
        sales_history=request.sales_history,
        horizon=request.horizon,
        cutoff_date=request.calculation_date,
        product_sku=request.product_sku,
        client_segment_type=request.client_segment_type,
    )

    return ForecastResponse(
        predicted_value=prediction.predicted_value,
        confidence_score=prediction.confidence_score,
        model_version=prediction.model_version,
        calculation_date=request.calculation_date,
    )
