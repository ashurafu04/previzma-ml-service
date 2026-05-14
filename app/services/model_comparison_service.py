from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from math import sqrt
from statistics import mean

from app.schemas import (
    ModelCandidateResponse,
    ModelComparisonRequest,
    ModelComparisonResponse,
    SalesHistoryItem,
)
from app.services.backtest_service import (
    _quality_label,
    _select_cutoff_dates,
    _usable_sales_history,
)
from app.services.forecast_engine import (
    BASELINE_MODEL_VERSION,
    MIN_MODEL_HISTORY_POINTS,
    forecast_revenue,
)
from app.services.model_registry import get_forecast_model


@dataclass(frozen=True)
class SelectionResult:
    selected_model_version: str
    selection_metric: str
    selection_reason: str


def run_model_comparison(
    request: ModelComparisonRequest,
) -> ModelComparisonResponse:
    return select_best_model(
        sales_history=request.sales_history,
        horizon=request.horizon,
        number_of_splits=request.number_of_splits,
    )


def select_best_model(
    sales_history: list[SalesHistoryItem],
    horizon: int,
    number_of_splits: int = 6,
) -> ModelComparisonResponse:
    history = _usable_sales_history(sales_history)
    trained_model = get_forecast_model()
    trained_model_available = (
        trained_model is not None and trained_model.supports_horizon(horizon)
    )
    minimum_history_points = (
        MIN_MODEL_HISTORY_POINTS if trained_model_available else 1
    )
    cutoff_dates = _select_cutoff_dates(
        history=history,
        horizon_days=horizon,
        number_of_splits=number_of_splits,
        minimum_history_points=minimum_history_points,
    )

    candidates = [
        _evaluate_candidate(
            model_version=BASELINE_MODEL_VERSION,
            history=history,
            cutoff_dates=cutoff_dates,
            horizon=horizon,
            allow_model=False,
        )
    ]

    if trained_model_available:
        candidates.append(
            _evaluate_candidate(
                model_version=trained_model.model_version,
                history=history,
                cutoff_dates=cutoff_dates,
                horizon=horizon,
                allow_model=True,
            )
        )

    selection = select_model_candidate(
        candidates=candidates,
        trained_model_available=trained_model_available,
    )

    return ModelComparisonResponse(
        horizon=horizon,
        number_of_splits=number_of_splits,
        selected_model_version=selection.selected_model_version,
        selection_metric=selection.selection_metric,
        selection_reason=selection.selection_reason,
        candidates=candidates,
    )


def select_model_candidate(
    candidates: list[ModelCandidateResponse],
    trained_model_available: bool,
) -> SelectionResult:
    if not candidates:
        return SelectionResult(
            selected_model_version=BASELINE_MODEL_VERSION,
            selection_metric="UNKNOWN",
            selection_reason="No model candidates could be evaluated.",
        )

    if not trained_model_available or len(candidates) == 1:
        fallback = candidates[0]
        return SelectionResult(
            selected_model_version=fallback.model_version,
            selection_metric="FALLBACK",
            selection_reason=(
                "No trained forecast model is available for this horizon; "
                f"{fallback.model_version} selected as fallback."
            ),
        )

    comparable_mape_candidates = [
        candidate
        for candidate in candidates
        if candidate.mape is not None and candidate.quality_label != "UNKNOWN"
    ]
    if len(comparable_mape_candidates) == len(candidates):
        selected = min(comparable_mape_candidates, key=lambda candidate: candidate.mape)
        comparison = _comparison_reason(selected, candidates, "MAPE")
        return SelectionResult(
            selected_model_version=selected.model_version,
            selection_metric="MAPE",
            selection_reason=comparison,
        )

    comparable_rmse_candidates = [
        candidate for candidate in candidates if candidate.rmse is not None
    ]
    if comparable_rmse_candidates:
        selected = min(comparable_rmse_candidates, key=lambda candidate: candidate.rmse)
        comparison = _comparison_reason(selected, candidates, "RMSE")
        return SelectionResult(
            selected_model_version=selected.model_version,
            selection_metric="RMSE",
            selection_reason=comparison,
        )

    return SelectionResult(
        selected_model_version=BASELINE_MODEL_VERSION,
        selection_metric="UNKNOWN",
        selection_reason=(
            "No comparable MAPE or RMSE could be computed; "
            f"{BASELINE_MODEL_VERSION} selected conservatively."
        ),
    )


def _evaluate_candidate(
    model_version: str,
    history: list[SalesHistoryItem],
    cutoff_dates: list[date],
    horizon: int,
    allow_model: bool,
) -> ModelCandidateResponse:
    if not cutoff_dates:
        return _unknown_candidate(model_version)

    absolute_errors: list[float] = []
    squared_errors: list[float] = []
    percentage_errors: list[float] = []

    for cutoff_date in cutoff_dates:
        forecast_history = [item for item in history if item.sale_date < cutoff_date]
        actual = _actual_revenue(
            history=history,
            cutoff_date=cutoff_date,
            horizon=horizon,
        )
        forecast = forecast_revenue(
            sales_history=forecast_history,
            horizon=horizon,
            cutoff_date=cutoff_date,
            allow_model=allow_model,
            require_promoted_model=False,
        )
        absolute_error = abs(forecast.predicted_value - actual)
        absolute_errors.append(absolute_error)
        squared_errors.append(absolute_error**2)

        if actual > 0:
            percentage_errors.append(absolute_error / actual * 100)

    mape = round(mean(percentage_errors), 2) if percentage_errors else None

    return ModelCandidateResponse(
        model_version=model_version,
        tested_splits=len(cutoff_dates),
        mae=round(mean(absolute_errors), 2),
        mape=mape,
        rmse=round(sqrt(mean(squared_errors)), 2),
        quality_label=_quality_label(mape),
    )


def _actual_revenue(
    history: list[SalesHistoryItem],
    cutoff_date: date,
    horizon: int,
) -> float:
    return round(
        sum(
            item.amount
            for item in history
            if cutoff_date <= item.sale_date < cutoff_date + timedelta(days=horizon)
        ),
        2,
    )


def _unknown_candidate(model_version: str) -> ModelCandidateResponse:
    return ModelCandidateResponse(
        model_version=model_version,
        tested_splits=0,
        mae=None,
        mape=None,
        rmse=None,
        quality_label="UNKNOWN",
    )


def _comparison_reason(
    selected: ModelCandidateResponse,
    candidates: list[ModelCandidateResponse],
    metric: str,
) -> str:
    alternatives = [
        candidate.model_version
        for candidate in candidates
        if candidate.model_version != selected.model_version
    ]
    if not alternatives:
        return f"{selected.model_version} selected by {metric}."

    return (
        f"{selected.model_version} has lower {metric} than "
        f"{', '.join(alternatives)}."
    )
