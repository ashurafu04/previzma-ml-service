from math import log1p

import pytest

from app.training.target_strategies import (
    fit_target_strategy,
    inverse_transform_prediction,
    transform_target,
)


def test_log1p_strategy_is_reversible() -> None:
    strategy = fit_target_strategy("log1p", [0.0, 100.0, 1000.0])

    transformed = transform_target(1234.56, strategy)

    assert transformed == pytest.approx(log1p(1234.56))
    assert inverse_transform_prediction(transformed, strategy) == pytest.approx(1234.56)


def test_clipped_strategy_winsorizes_training_targets() -> None:
    strategy = fit_target_strategy(
        "clipped_raw",
        [0.0, 100.0, 200.0, 10_000.0],
        clip_lower_percentile=25.0,
        clip_upper_percentile=75.0,
    )

    assert strategy.clip_lower is not None
    assert strategy.clip_upper is not None
    assert transform_target(0.0, strategy) == pytest.approx(strategy.clip_lower)
    assert transform_target(99_999.0, strategy) == pytest.approx(strategy.clip_upper)


def test_baseline_ratio_log_strategy_is_reversible_with_baseline() -> None:
    strategy = fit_target_strategy("baseline_ratio_log1p", [100.0, 200.0])

    transformed = transform_target(1500.0, strategy, baseline_prediction=500.0)

    assert transformed == pytest.approx(log1p(3.0))
    assert inverse_transform_prediction(
        transformed,
        strategy,
        baseline_prediction=500.0,
    ) == pytest.approx(1500.0)


def test_clipped_baseline_ratio_strategy_clips_ratio_not_amount() -> None:
    strategy = fit_target_strategy(
        "clipped_baseline_ratio_log1p",
        [100.0, 200.0, 10_000.0],
        baseline_predictions=[100.0, 100.0, 100.0],
        clip_lower_percentile=0.0,
        clip_upper_percentile=50.0,
    )

    transformed = transform_target(10_000.0, strategy, baseline_prediction=100.0)

    assert strategy.clip_upper == pytest.approx(2.0)
    assert transformed == pytest.approx(log1p(2.0))


def test_unknown_target_strategy_is_rejected() -> None:
    with pytest.raises(ValueError):
        fit_target_strategy("unknown", [1.0, 2.0])
