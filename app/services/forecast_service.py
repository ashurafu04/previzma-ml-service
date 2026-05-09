from app.schemas import ForecastRequest, ForecastResponse

MODEL_VERSION = "mock-forecast-v1"
CONFIDENCE_SCORE = 0.75


def predict_forecast(request: ForecastRequest) -> ForecastResponse:
    if not request.sales_history:
        predicted_value = 0.0
    else:
        average_amount = sum(item.amount for item in request.sales_history) / len(
            request.sales_history
        )
        predicted_value = average_amount * request.horizon

    return ForecastResponse(
        predicted_value=round(predicted_value, 2),
        confidence_score=CONFIDENCE_SCORE,
        model_version=MODEL_VERSION,
        calculation_date=request.calculation_date,
    )
