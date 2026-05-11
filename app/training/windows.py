from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

from app.schemas import SalesHistoryItem
from app.training.features import FeatureContext, build_forecast_features

DEFAULT_HORIZONS = (30, 60, 90)
REQUIRED_CSV_COLUMNS = {
    "saleDate",
    "productId",
    "productName",
    "productSku",
    "clientSegmentId",
    "clientSegmentName",
    "clientSegmentType",
    "quantity",
    "amount",
    "confirmedOrder",
    "sourceStatus",
}


@dataclass(frozen=True)
class TrainingSale:
    sale_date: date
    product_id: int
    product_name: str
    product_sku: str
    client_segment_id: int
    client_segment_name: str
    client_segment_type: str
    quantity: int
    amount: float
    confirmed_order: bool
    source_status: str

    def to_sales_history_item(self) -> SalesHistoryItem:
        return SalesHistoryItem(
            sale_date=self.sale_date,
            quantity=self.quantity,
            amount=self.amount,
            confirmed_order=self.confirmed_order,
            source_status=self.source_status,
        )


@dataclass(frozen=True)
class TrainingExample:
    features: dict[str, float]
    target: float
    cutoff_date: date
    horizon: int
    product_id: int
    client_segment_id: int


def load_sales_csv(input_path: str | Path) -> list[TrainingSale]:
    path = Path(input_path)

    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_columns = REQUIRED_CSV_COLUMNS - set(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"CSV is missing required columns: {missing}")

        return [
            sale
            for row in reader
            if (sale := _row_to_training_sale(row)) is not None
        ]


def generate_training_examples(
    sales: Iterable[TrainingSale],
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> list[TrainingExample]:
    grouped_sales = _group_sales(sales)
    examples: list[TrainingExample] = []

    for group_sales in grouped_sales.values():
        ordered_sales = sorted(group_sales, key=lambda sale: sale.sale_date)
        history_items = [sale.to_sales_history_item() for sale in ordered_sales]
        context = FeatureContext(
            product_sku=ordered_sales[0].product_sku,
            client_segment_type=ordered_sales[0].client_segment_type,
        )

        for horizon in horizons:
            for cutoff_date in _candidate_cutoff_dates(ordered_sales, horizon):
                history_train = [
                    item for item in history_items if item.sale_date < cutoff_date
                ]
                if not history_train:
                    continue

                target = _future_revenue(
                    history_items=history_items,
                    cutoff_date=cutoff_date,
                    horizon=horizon,
                )
                features = build_forecast_features(
                    sales_history=history_train,
                    horizon=horizon,
                    cutoff_date=cutoff_date,
                    context=context,
                )
                examples.append(
                    TrainingExample(
                        features=features,
                        target=target,
                        cutoff_date=cutoff_date,
                        horizon=horizon,
                        product_id=ordered_sales[0].product_id,
                        client_segment_id=ordered_sales[0].client_segment_id,
                    )
                )

    return sorted(
        examples,
        key=lambda example: (
            example.cutoff_date,
            example.product_id,
            example.client_segment_id,
            example.horizon,
        ),
    )


def _row_to_training_sale(row: dict[str, str]) -> TrainingSale | None:
    source_status = row["sourceStatus"].strip()
    if source_status.upper() == "CANCELLED":
        return None

    amount = max(_parse_float(row["amount"]), 0.0)
    quantity = max(int(_parse_float(row["quantity"])), 0)

    return TrainingSale(
        sale_date=date.fromisoformat(row["saleDate"].strip()),
        product_id=int(row["productId"]),
        product_name=row["productName"].strip(),
        product_sku=row["productSku"].strip(),
        client_segment_id=int(row["clientSegmentId"]),
        client_segment_name=row["clientSegmentName"].strip(),
        client_segment_type=row["clientSegmentType"].strip(),
        quantity=quantity,
        amount=amount,
        confirmed_order=_parse_bool(row["confirmedOrder"]),
        source_status=source_status,
    )


def _group_sales(
    sales: Iterable[TrainingSale],
) -> dict[tuple[int, int], list[TrainingSale]]:
    grouped_sales: dict[tuple[int, int], list[TrainingSale]] = {}

    for sale in sales:
        grouped_sales.setdefault((sale.product_id, sale.client_segment_id), []).append(
            sale
        )

    return grouped_sales


def _candidate_cutoff_dates(
    sales: list[TrainingSale],
    horizon: int,
) -> list[date]:
    if len(sales) < 2:
        return []

    unique_dates = sorted({sale.sale_date for sale in sales})
    latest_observed_date = unique_dates[-1]
    latest_full_window_cutoff = latest_observed_date - timedelta(days=horizon - 1)

    return [
        cutoff_date
        for cutoff_date in unique_dates
        if cutoff_date <= latest_full_window_cutoff
        and any(sale.sale_date < cutoff_date for sale in sales)
    ]


def _future_revenue(
    history_items: list[SalesHistoryItem],
    cutoff_date: date,
    horizon: int,
) -> float:
    return sum(
        item.amount
        for item in history_items
        if cutoff_date <= item.sale_date < cutoff_date + timedelta(days=horizon)
    )


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes", "y"}


def _parse_float(value: str) -> float:
    if not value:
        return 0.0

    return float(value)
