import json
from pathlib import Path

import pandas as pd

from app.training.transform_kaggle_b2b import (
    ENRICHED_COLUMNS,
    SALES_EXPORT_COLUMNS,
    transform_kaggle_b2b,
)
from app.training.windows import load_sales_csv

FIXTURE = Path("tests/fixtures/online_retail_ii_sample.csv")


def test_transform_kaggle_b2b_writes_supabase_and_ml_outputs(tmp_path) -> None:
    output_dir = tmp_path / "processed"
    paths = transform_kaggle_b2b(
        input_path=FIXTURE,
        output_dir=output_dir,
        max_output_rows=12,
    )

    assert paths["products"].exists()
    assert paths["client_segments"].exists()
    assert paths["sales"].exists()
    assert paths["enriched_sales"].exists()
    assert paths["sales_export"].exists()
    assert paths["summary"].exists()
    assert paths["load_sql"].exists()

    enriched_df = pd.read_csv(paths["enriched_sales"])
    sales_export_df = pd.read_csv(paths["sales_export"])

    assert list(enriched_df.columns) == ENRICHED_COLUMNS
    assert list(sales_export_df.columns) == SALES_EXPORT_COLUMNS
    assert set(sales_export_df["productName"]).issubset(
        {
            "Industrial Pump X200",
            "Smart Conveyor S400",
            "Hydraulic Valve HV90",
            "Packaging Robot PR300",
            "Air Compressor AC700",
            "Industrial Sensor IS50",
            "Control Panel CP120",
            "Spare Parts Kit SPK",
        }
    )
    assert not sales_export_df["productName"].str.contains("CHRISTMAS|TRINKET|BAG").any()
    assert sales_export_df["amount"].min() >= 0
    assert sales_export_df["saleDate"].min() >= "2021-01-01"
    assert sales_export_df["saleDate"].max() <= "2026-12-31"
    assert "CONFIRMED" in set(sales_export_df["sourceStatus"])
    assert set(sales_export_df["sourceStatus"]).issubset(
        {"CONFIRMED", "PENDING", "CANCELLED"}
    )


def test_transform_kaggle_b2b_sales_export_is_training_compatible(tmp_path) -> None:
    paths = transform_kaggle_b2b(
        input_path=FIXTURE,
        output_dir=tmp_path / "processed",
        max_output_rows=12,
    )

    sales = load_sales_csv(paths["sales_export"])

    assert sales
    assert all(sale.amount >= 0 for sale in sales)
    assert all(sale.product_name for sale in sales)
    assert all(sale.client_segment_type in {"GRAND_COMPTE", "PME", "DISTRIBUTEUR"} for sale in sales)


def test_transform_kaggle_b2b_summary_contains_validation_stats(tmp_path) -> None:
    paths = transform_kaggle_b2b(
        input_path=FIXTURE,
        output_dir=tmp_path / "processed",
        max_output_rows=12,
    )
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))

    assert 1 <= summary["rowCount"] <= 12
    assert summary["productCount"] >= 1
    assert summary["segmentCount"] >= 1
    assert "sourceStatusPercentages" in summary
    assert summary["amountMin"] >= 0
