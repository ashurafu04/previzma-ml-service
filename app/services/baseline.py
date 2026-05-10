from collections import defaultdict
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Iterable

from app.schemas import SalesHistoryItem

DAYS_PER_MONTH = 30.0
MAX_TREND_CHANGE = 0.5
TREND_DAMPING = 0.5


@dataclass(frozen=True)
class RevenueBaseline:
    monthly_value: float
    confidence_score: float
    history_points: int
    observed_months: int
    trend_multiplier: float


def calculate_revenue_baseline(
    sales_history: Iterable[SalesHistoryItem],
) -> RevenueBaseline:
    history = sorted(sales_history, key=lambda item: item.sale_date)

    if not history:
        return RevenueBaseline(
            monthly_value=0.0,
            confidence_score=0.0,
            history_points=0,
            observed_months=0,
            trend_multiplier=1.0,
        )

    monthly_revenue = _aggregate_monthly_revenue(history)
    first_month = min(monthly_revenue)
    last_month = max(monthly_revenue)
    monthly_values = [
        monthly_revenue.get(month_index, 0.0)
        for month_index in range(first_month, last_month + 1)
    ]

    average_monthly_revenue = sum(monthly_values) / len(monthly_values)
    trend_multiplier = _trend_multiplier(monthly_values)
    confidence_score = _confidence_score(history, monthly_revenue.keys())

    return RevenueBaseline(
        monthly_value=max(average_monthly_revenue * trend_multiplier, 0.0),
        confidence_score=confidence_score,
        history_points=len(history),
        observed_months=len(monthly_values),
        trend_multiplier=trend_multiplier,
    )


def project_monthly_baseline(monthly_value: float, horizon_days: int) -> float:
    horizon_months = horizon_days / DAYS_PER_MONTH
    return max(monthly_value * horizon_months, 0.0)


def _aggregate_monthly_revenue(
    history: Iterable[SalesHistoryItem],
) -> dict[int, float]:
    monthly_revenue: dict[int, float] = defaultdict(float)

    for item in history:
        monthly_revenue[_month_index(item)] += item.amount

    return dict(monthly_revenue)


def _month_index(item: SalesHistoryItem) -> int:
    return item.sale_date.year * 12 + item.sale_date.month


def _trend_multiplier(monthly_values: list[float]) -> float:
    if len(monthly_values) < 2:
        return 1.0

    midpoint = max(1, len(monthly_values) // 2)
    older_values = monthly_values[:midpoint]
    recent_values = monthly_values[midpoint:]

    if not recent_values:
        return 1.0

    older_average = mean(older_values)
    recent_average = mean(recent_values)

    if older_average == 0:
        return 1.15 if recent_average > 0 else 1.0

    relative_change = (recent_average - older_average) / older_average
    capped_change = max(-MAX_TREND_CHANGE, min(MAX_TREND_CHANGE, relative_change))
    return max(0.0, 1 + capped_change * TREND_DAMPING)


def _confidence_score(
    history: list[SalesHistoryItem],
    revenue_months: Iterable[int],
) -> float:
    if not history:
        return 0.0

    sorted_months = sorted(set(revenue_months))
    history_volume_score = min(len(history) / 12, 1.0)
    month_coverage_score = min(len(sorted_months) / 12, 1.0)
    regularity_score = _month_regularity_score(sorted_months)

    confidence = (
        0.10
        + history_volume_score * 0.45
        + month_coverage_score * 0.25
        + regularity_score * 0.20
    )

    return round(min(confidence, 0.95), 2)


def _month_regularity_score(sorted_months: list[int]) -> float:
    if len(sorted_months) < 2:
        return 0.10

    gaps = [
        current_month - previous_month
        for previous_month, current_month in zip(sorted_months, sorted_months[1:])
    ]
    average_gap = mean(gaps)

    if average_gap <= 0:
        return 0.0

    if len(gaps) == 1:
        return 1.0

    coefficient_of_variation = pstdev(gaps) / average_gap
    return max(0.0, min(1.0, 1.0 - coefficient_of_variation))
