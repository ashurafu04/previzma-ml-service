from pathlib import Path

from app.training.windows import generate_training_examples, load_sales_csv

FIXTURE = Path("tests/fixtures/sales_export_minimal.csv")


def test_load_sales_csv_parses_exported_sales() -> None:
    sales = load_sales_csv(FIXTURE)

    assert len(sales) == 18
    assert sales[0].product_id == 10
    assert sales[0].product_sku == "PUMP-001"
    assert sales[0].client_segment_type == "GRAND_COMPTE"
    assert sales[0].amount == 1000.0
    assert sales[0].confirmed_order is True


def test_generate_training_examples_builds_supervised_windows() -> None:
    sales = load_sales_csv(FIXTURE)
    examples = generate_training_examples(sales, horizons=[30, 60, 90])

    assert examples
    assert {example.horizon for example in examples} == {30, 60, 90}
    assert {example.product_id for example in examples} == {10, 11}
    assert all(example.target >= 0 for example in examples)
    assert all("revenue_last_90d" in example.features for example in examples)


def test_generate_training_examples_uses_future_revenue_as_target() -> None:
    sales = load_sales_csv(FIXTURE)
    examples = generate_training_examples(sales, horizons=[30])
    first_example = examples[0]

    assert first_example.cutoff_date.isoformat() == "2025-02-01"
    assert first_example.product_id == 10
    assert first_example.client_segment_id == 20
    assert first_example.target == 1200.0
