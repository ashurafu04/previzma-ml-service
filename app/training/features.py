from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date

from app.schemas import SalesHistoryItem
from app.services.baseline import calculate_revenue_baseline, project_monthly_baseline

FEATURE_NAMES = [
    "horizon",
    "month",
    "quarter",
    "sales_count_total",
    "confirmed_sales_count",
    "revenue_total",
    "revenue_last_30d",
    "revenue_last_60d",
    "revenue_last_90d",
    "quantity_total",
    "quantity_last_30d",
    "avg_order_amount",
    "recent_vs_old_revenue_ratio",
    "trend_factor",
    "baseline_prediction",
    "days_since_last_sale",
    "product_sku_hash",
    "client_segment_type_hash",
]


@dataclass(frozen=True)
class FeatureContext:
    product_sku: str = "UNKNOWN"
    client_segment_type: str = "UNKNOWN"


def build_forecast_features(
    sales_history: list[SalesHistoryItem],
    horizon: int,
    cutoff_date: date,
    context: FeatureContext | None = None,
) -> dict[str, float]:
    context = context or FeatureContext()
    history = _usable_history_before_cutoff(sales_history, cutoff_date)

    sales_count_total = len(history)
    confirmed_sales_count = sum(1 for item in history if item.confirmed_order)
    revenue_total = sum(item.amount for item in history)
    revenue_last_30d = _sum_revenue_since(history, cutoff_date, days=30)
    revenue_last_60d = _sum_revenue_since(history, cutoff_date, days=60)
    revenue_last_90d = _sum_revenue_since(history, cutoff_date, days=90)
    quantity_total = sum(item.quantity for item in history)
    quantity_last_30d = _sum_quantity_since(history, cutoff_date, days=30)
    avg_order_amount = revenue_total / sales_count_total if sales_count_total else 0.0
    older_revenue = max(revenue_total - revenue_last_90d, 0.0)
    recent_vs_old_revenue_ratio = (
        revenue_last_90d / older_revenue
        if older_revenue > 0
        else 1.0 if revenue_last_90d > 0 else 0.0
    )
    baseline = calculate_revenue_baseline(history)
    trend_factor = baseline.trend_multiplier
    baseline_prediction = project_monthly_baseline(
        monthly_value=baseline.monthly_value,
        horizon_days=horizon,
    )
    days_since_last_sale = (
        (cutoff_date - history[-1].sale_date).days if history else 9999
    )

    features = {
        "horizon": float(horizon),
        "month": float(cutoff_date.month),
        "quarter": float((cutoff_date.month - 1) // 3 + 1),
        "sales_count_total": float(sales_count_total),
        "confirmed_sales_count": float(confirmed_sales_count),
        "revenue_total": float(revenue_total),
        "revenue_last_30d": float(revenue_last_30d),
        "revenue_last_60d": float(revenue_last_60d),
        "revenue_last_90d": float(revenue_last_90d),
        "quantity_total": float(quantity_total),
        "quantity_last_30d": float(quantity_last_30d),
        "avg_order_amount": float(avg_order_amount),
        "recent_vs_old_revenue_ratio": float(recent_vs_old_revenue_ratio),
        "trend_factor": float(trend_factor),
        "baseline_prediction": float(baseline_prediction),
        "days_since_last_sale": float(max(days_since_last_sale, 0)),
        "product_sku_hash": _stable_hash(context.product_sku),
        "client_segment_type_hash": _stable_hash(context.client_segment_type),
    }

    return {name: features[name] for name in FEATURE_NAMES}


def feature_vector(features: dict[str, float]) -> list[float]:
    return [features[name] for name in FEATURE_NAMES]


def _usable_history_before_cutoff(
    sales_history: list[SalesHistoryItem],
    cutoff_date: date,
) -> list[SalesHistoryItem]:
    return sorted(
        [
            item
            for item in sales_history
            if item.sale_date < cutoff_date
            and item.amount >= 0
            and (item.confirmed_order or item.amount > 0)
            and item.source_status.upper() != "CANCELLED"
        ],
        key=lambda item: item.sale_date,
    )


def _sum_revenue_since(
    history: list[SalesHistoryItem],
    cutoff_date: date,
    days: int,
) -> float:
    return sum(
        item.amount
        for item in history
        if 0 < (cutoff_date - item.sale_date).days <= days
    )


def _sum_quantity_since(
    history: list[SalesHistoryItem],
    cutoff_date: date,
    days: int,
) -> int:
    return sum(
        item.quantity
        for item in history
        if 0 < (cutoff_date - item.sale_date).days <= days
    )


def _stable_hash(value: str) -> float:
    digest = hashlib.sha256(value.strip().upper().encode("utf-8")).hexdigest()
    return float(int(digest[:8], 16) % 10_000) / 10_000
