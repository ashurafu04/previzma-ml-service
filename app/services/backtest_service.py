from datetime import date, timedelta
from math import sqrt
from statistics import mean

from app.schemas import (
    BacktestRequest,
    BacktestResponse,
    BacktestWindowResponse,
    SalesHistoryItem,
)
from app.services.forecast_engine import (
    BASELINE_MODEL_VERSION,
    MIN_MODEL_HISTORY_POINTS,
    forecast_revenue,
)
from app.services.model_registry import get_forecast_model


def run_backtest(request: BacktestRequest) -> BacktestResponse:
    history = _usable_sales_history(request.sales_history)
    trained_model = get_forecast_model()
    trained_model_available = (
        trained_model is not None and trained_model.supports_horizon(request.horizon)
    )
    minimum_history_points = (
        MIN_MODEL_HISTORY_POINTS if trained_model_available else 1
    )
    cutoff_dates = _select_cutoff_dates(
        history=history,
        horizon_days=request.horizon,
        number_of_splits=request.number_of_splits,
        minimum_history_points=minimum_history_points,
    )

    if not cutoff_dates:
        return _unknown_response(request.horizon)

    window_results = [
        _build_window(history=history, cutoff_date=cutoff_date, horizon=request.horizon)
        for cutoff_date in cutoff_dates
    ]
    windows = [window for window, _model_version in window_results]
    model_versions = {model_version for _window, model_version in window_results}
    absolute_errors = [window.absolute_error for window in windows]
    squared_errors = [error**2 for error in absolute_errors]
    percentage_errors = [
        window.absolute_percentage_error
        for window in windows
        if window.absolute_percentage_error is not None
    ]

    mape = round(mean(percentage_errors), 2) if percentage_errors else None

    return BacktestResponse(
        model_version=_response_model_version(model_versions),
        horizon=request.horizon,
        tested_splits=len(windows),
        mae=round(mean(absolute_errors), 2),
        mape=mape,
        rmse=round(sqrt(mean(squared_errors)), 2),
        quality_label=_quality_label(mape),
        backtest_windows=windows,
    )


def _usable_sales_history(
    sales_history: list[SalesHistoryItem],
) -> list[SalesHistoryItem]:
    return sorted(
        [
            item
            for item in sales_history
            if item.confirmed_order or item.amount > 0
        ],
        key=lambda item: item.sale_date,
    )


def _select_cutoff_dates(
    history: list[SalesHistoryItem],
    horizon_days: int,
    number_of_splits: int,
    minimum_history_points: int = 1,
) -> list[date]:
    if len(history) < 2:
        return []

    unique_dates = sorted({item.sale_date for item in history})
    latest_observed_date = unique_dates[-1]
    latest_full_window_cutoff = latest_observed_date - timedelta(days=horizon_days - 1)
    candidates = [
        cutoff_date
        for cutoff_date in unique_dates
        if cutoff_date <= latest_full_window_cutoff
        and sum(1 for item in history if item.sale_date < cutoff_date)
        >= minimum_history_points
    ]

    if not candidates:
        return []

    return _evenly_spaced(candidates, number_of_splits)


def _evenly_spaced(values: list[date], limit: int) -> list[date]:
    if len(values) <= limit:
        return values

    if limit == 1:
        return [values[-1]]

    indexes = [
        round(index * (len(values) - 1) / (limit - 1))
        for index in range(limit)
    ]
    return [values[index] for index in sorted(set(indexes))]


def _build_window(
    history: list[SalesHistoryItem],
    cutoff_date: date,
    horizon: int,
) -> tuple[BacktestWindowResponse, str]:
    forecast_history = [item for item in history if item.sale_date < cutoff_date]
    actual_history = [
        item
        for item in history
        if cutoff_date <= item.sale_date < cutoff_date + timedelta(days=horizon)
    ]

    forecast = forecast_revenue(
        sales_history=forecast_history,
        horizon=horizon,
        cutoff_date=cutoff_date,
    )
    prediction = forecast.predicted_value
    actual = round(sum(item.amount for item in actual_history), 2)
    absolute_error = round(abs(prediction - actual), 2)
    absolute_percentage_error = (
        round(absolute_error / actual * 100, 2) if actual > 0 else None
    )

    return (
        BacktestWindowResponse(
            cutoff_date=cutoff_date,
            prediction=prediction,
            actual=actual,
            absolute_error=absolute_error,
            absolute_percentage_error=absolute_percentage_error,
        ),
        forecast.model_version,
    )


def _unknown_response(horizon: int) -> BacktestResponse:
    return BacktestResponse(
        model_version=BASELINE_MODEL_VERSION,
        horizon=horizon,
        tested_splits=0,
        mae=None,
        mape=None,
        rmse=None,
        quality_label="UNKNOWN",
        backtest_windows=[],
    )


def _quality_label(mape: float | None) -> str:
    if mape is None:
        return "UNKNOWN"

    if mape < 10:
        return "EXCELLENT"

    if mape < 20:
        return "GOOD"

    if mape < 35:
        return "FAIR"

    return "POOR"


def _response_model_version(model_versions: set[str]) -> str:
    if len(model_versions) == 1:
        return next(iter(model_versions))

    return "mixed-" + "+".join(sorted(model_versions))
