from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from math import sqrt
from pathlib import Path
from statistics import mean
from typing import Any

import joblib
import pandas as pd

from app.training.features import FEATURE_NAMES
from app.training.target_strategies import (
    DEFAULT_TARGET_STRATEGIES,
    TargetStrategy,
    fit_target_strategy,
    inverse_transform_prediction,
    transform_targets,
)
from app.training.windows import DEFAULT_HORIZONS, generate_training_examples, load_sales_csv

BASELINE_PREDICTION_FEATURE = "baseline_prediction"


@dataclass(frozen=True)
class TrainingCandidate:
    model: Any
    algorithm: str
    model_version: str
    strategy: TargetStrategy
    validation_mae: float
    validation_mape: float | None
    validation_rmse: float


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the Previzma forecast model.")
    parser.add_argument("--input", required=True, help="Path to exported sales CSV.")
    parser.add_argument(
        "--output",
        default="app/models/forecast_model.joblib",
        help="Path where the trained model artifact is saved.",
    )
    parser.add_argument(
        "--horizons",
        nargs="*",
        type=int,
        default=list(DEFAULT_HORIZONS),
        help="Forecast horizons in days to include in training windows.",
    )
    parser.add_argument(
        "--target-strategies",
        nargs="*",
        choices=list(DEFAULT_TARGET_STRATEGIES),
        default=list(DEFAULT_TARGET_STRATEGIES),
        help=(
            "Target stabilization strategies to compare. The best candidate is "
            "selected by validation MAPE, then RMSE."
        ),
    )
    parser.add_argument(
        "--clip-lower-percentile",
        type=float,
        default=1.0,
        help="Lower winsorization percentile for clipped target strategies.",
    )
    parser.add_argument(
        "--clip-upper-percentile",
        type=float,
        default=99.0,
        help="Upper winsorization percentile for clipped target strategies.",
    )
    args = parser.parse_args(argv)

    sales = load_sales_csv(args.input)
    examples = generate_training_examples(sales, horizons=args.horizons)
    if len(examples) < 2:
        raise SystemExit("Not enough supervised windows to train a model.")

    split_index = max(1, int(len(examples) * 0.8))
    if split_index >= len(examples):
        split_index = len(examples) - 1

    train_examples = examples[:split_index]
    validation_examples = examples[split_index:]

    x_train = _feature_matrix(train_examples)
    y_train = [example.target for example in train_examples]
    x_validation = _feature_matrix(validation_examples)
    y_validation = [example.target for example in validation_examples]

    candidates = [
        _train_candidate(
            strategy_name=strategy_name,
            x_train=x_train,
            y_train=y_train,
            x_validation=x_validation,
            y_validation=y_validation,
            clip_lower_percentile=args.clip_lower_percentile,
            clip_upper_percentile=args.clip_upper_percentile,
        )
        for strategy_name in args.target_strategies
    ]
    best_candidate = _select_best_candidate(candidates)
    candidate_metadata = [_candidate_metadata(candidate) for candidate in candidates]

    metadata = {
        "modelVersion": best_candidate.model_version,
        "trainedAt": datetime.now(UTC).isoformat(),
        "algorithm": best_candidate.algorithm,
        "horizonsSupported": sorted({example.horizon for example in examples}),
        "trainingRows": len(train_examples),
        "validationRows": len(validation_examples),
        "validationMae": best_candidate.validation_mae,
        "validationMape": best_candidate.validation_mape,
        "validationRmse": best_candidate.validation_rmse,
        "featureNames": FEATURE_NAMES,
        "targetStrategy": best_candidate.strategy.name,
        "targetStrategyCandidates": candidate_metadata,
    }
    metadata.update(best_candidate.strategy.to_metadata())

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_candidate.model,
            "featureNames": FEATURE_NAMES,
            "metadata": metadata,
        },
        output_path,
    )

    metadata_path = output_path.with_name("model_metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


def _build_regressor():
    try:
        from lightgbm import LGBMRegressor

        return (
            LGBMRegressor(
                n_estimators=80,
                learning_rate=0.05,
                max_depth=4,
                min_child_samples=1,
                min_data_in_bin=1,
                min_data_in_leaf=1,
                random_state=42,
                verbosity=-1,
            ),
            "LightGBM LGBMRegressor",
            "lightgbm-window-v1",
        )
    except ImportError:
        pass

    try:
        from xgboost import XGBRegressor

        return (
            XGBRegressor(
                n_estimators=80,
                learning_rate=0.05,
                max_depth=4,
                random_state=42,
                objective="reg:squarederror",
            ),
            "XGBoost XGBRegressor",
            "xgboost-window-v1",
        )
    except ImportError:
        pass

    from sklearn.ensemble import HistGradientBoostingRegressor

    return (
        HistGradientBoostingRegressor(max_iter=80, learning_rate=0.05, random_state=42),
        "scikit-learn HistGradientBoostingRegressor",
        "sklearn-hist-gradient-boosting-v1",
    )


def _feature_matrix(examples) -> pd.DataFrame:
    return pd.DataFrame(
        [[example.features[name] for name in FEATURE_NAMES] for example in examples],
        columns=FEATURE_NAMES,
    )


def _train_candidate(
    strategy_name: str,
    x_train: pd.DataFrame,
    y_train: list[float],
    x_validation: pd.DataFrame,
    y_validation: list[float],
    clip_lower_percentile: float,
    clip_upper_percentile: float,
) -> TrainingCandidate:
    train_baseline_predictions = _baseline_predictions(x_train)
    validation_baseline_predictions = _baseline_predictions(x_validation)
    strategy = fit_target_strategy(
        name=strategy_name,
        targets=y_train,
        baseline_predictions=train_baseline_predictions,
        clip_lower_percentile=clip_lower_percentile,
        clip_upper_percentile=clip_upper_percentile,
    )
    transformed_y_train = transform_targets(
        y_train,
        strategy,
        baseline_predictions=train_baseline_predictions,
    )

    model, algorithm, model_version = _build_regressor()
    model.fit(x_train, transformed_y_train)

    transformed_predictions = [float(value) for value in model.predict(x_validation)]
    validation_predictions = [
        inverse_transform_prediction(
            prediction,
            strategy,
            baseline_prediction=validation_baseline_predictions[index],
        )
        for index, prediction in enumerate(transformed_predictions)
    ]

    return TrainingCandidate(
        model=model,
        algorithm=algorithm,
        model_version=model_version,
        strategy=strategy,
        validation_mae=_mae(validation_predictions, y_validation),
        validation_mape=_mape(validation_predictions, y_validation),
        validation_rmse=_rmse(validation_predictions, y_validation),
    )


def _select_best_candidate(candidates: list[TrainingCandidate]) -> TrainingCandidate:
    if not candidates:
        raise ValueError("At least one target strategy is required.")

    return min(candidates, key=_candidate_sort_key)


def _baseline_predictions(features: pd.DataFrame) -> list[float]:
    if BASELINE_PREDICTION_FEATURE not in features:
        return [1.0 for _ in range(len(features))]

    return [max(float(value), 1.0) for value in features[BASELINE_PREDICTION_FEATURE]]


def _candidate_sort_key(candidate: TrainingCandidate) -> tuple[float, float]:
    mape = candidate.validation_mape
    comparable_mape = mape if mape is not None else float("inf")
    return (comparable_mape, candidate.validation_rmse)


def _candidate_metadata(candidate: TrainingCandidate) -> dict[str, float | str | None]:
    metadata = candidate.strategy.to_metadata()
    metadata.update(
        {
            "validationMae": candidate.validation_mae,
            "validationMape": candidate.validation_mape,
            "validationRmse": candidate.validation_rmse,
        }
    )
    return metadata


def _mae(predictions: list[float], actuals: list[float]) -> float:
    return round(mean(abs(prediction - actual) for prediction, actual in zip(predictions, actuals)), 2)


def _mape(predictions: list[float], actuals: list[float]) -> float | None:
    percentage_errors = [
        abs(prediction - actual) / actual * 100
        for prediction, actual in zip(predictions, actuals)
        if actual > 0
    ]
    return round(mean(percentage_errors), 2) if percentage_errors else None


def _rmse(predictions: list[float], actuals: list[float]) -> float:
    return round(
        sqrt(mean((prediction - actual) ** 2 for prediction, actual in zip(predictions, actuals))),
        2,
    )


if __name__ == "__main__":
    raise SystemExit(main())
