from datetime import date, timedelta

from app.schemas import BacktestRequest, SalesHistoryItem
from app.services.backtest_service import run_backtest


def sale(sale_date: date, amount: float) -> SalesHistoryItem:
    return SalesHistoryItem(
        sale_date=sale_date,
        quantity=1,
        amount=amount,
        confirmed_order=True,
        source_status="CONFIRMED",
    )


def spaced_history(amounts: list[float]) -> list[SalesHistoryItem]:
    start_date = date(2025, 1, 1)
    return [
        sale(start_date + timedelta(days=index * 31), amount)
        for index, amount in enumerate(amounts)
    ]


def backtest_request(
    sales_history: list[SalesHistoryItem],
    horizon: int = 30,
    number_of_splits: int = 6,
) -> BacktestRequest:
    return BacktestRequest(
        horizon=horizon,
        sales_history=sales_history,
        number_of_splits=number_of_splits,
    )


def test_backtest_with_empty_history_returns_unknown_quality() -> None:
    response = run_backtest(backtest_request([]))

    assert response.tested_splits == 0
    assert response.mae is None
    assert response.mape is None
    assert response.rmse is None
    assert response.quality_label == "UNKNOWN"
    assert response.backtest_windows == []


def test_backtest_with_insufficient_history_returns_unknown_quality() -> None:
    response = run_backtest(backtest_request(spaced_history([1000.0])))

    assert response.tested_splits == 0
    assert response.quality_label == "UNKNOWN"


def test_backtest_with_growing_history_creates_evaluation_windows() -> None:
    response = run_backtest(
        backtest_request(
            spaced_history([1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]),
            number_of_splits=3,
        )
    )

    assert response.model_version == "baseline-statistical-v1"
    assert response.horizon == 30
    assert response.tested_splits == 3
    assert response.mae is not None
    assert response.mape is not None
    assert response.rmse is not None
    assert response.backtest_windows[0].cutoff_date == date(2025, 2, 1)
    assert response.backtest_windows[-1].cutoff_date == date(2025, 5, 5)


def test_backtest_uses_multiple_evenly_spaced_splits() -> None:
    response = run_backtest(
        backtest_request(
            spaced_history([1000.0] * 8),
            number_of_splits=3,
        )
    )

    assert response.tested_splits == 3
    assert [window.cutoff_date for window in response.backtest_windows] == [
        date(2025, 2, 1),
        date(2025, 4, 4),
        date(2025, 7, 6),
    ]
    assert response.mae == 0.0
    assert response.mape == 0.0
    assert response.rmse == 0.0
    assert response.quality_label == "EXCELLENT"


def test_backtest_calculates_mae_mape_rmse_and_quality_label() -> None:
    response = run_backtest(
        backtest_request(
            spaced_history([1000.0, 2000.0, 1000.0, 2000.0]),
            number_of_splits=2,
        )
    )

    assert response.tested_splits == 2
    assert response.mae == 937.5
    assert response.mape == 68.75
    assert response.rmse == 939.58
    assert response.quality_label == "POOR"
    assert response.backtest_windows[0].prediction == 1000.0
    assert response.backtest_windows[0].actual == 2000.0
    assert response.backtest_windows[0].absolute_error == 1000.0
    assert response.backtest_windows[0].absolute_percentage_error == 50.0
