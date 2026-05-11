from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from math import sqrt
from pathlib import Path
from statistics import mean

import joblib
import pandas as pd

from app.training.features import FEATURE_NAMES
from app.training.windows import DEFAULT_HORIZONS, generate_training_examples, load_sales_csv


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

    model, algorithm, model_version = _build_regressor()
    model.fit(x_train, y_train)

    validation_predictions = [
        max(float(value), 0.0) for value in model.predict(x_validation)
    ]
    validation_mae = _mae(validation_predictions, y_validation)
    validation_mape = _mape(validation_predictions, y_validation)
    validation_rmse = _rmse(validation_predictions, y_validation)

    metadata = {
        "modelVersion": model_version,
        "trainedAt": datetime.now(UTC).isoformat(),
        "algorithm": algorithm,
        "horizonsSupported": sorted({example.horizon for example in examples}),
        "trainingRows": len(train_examples),
        "validationRows": len(validation_examples),
        "validationMae": validation_mae,
        "validationMape": validation_mape,
        "validationRmse": validation_rmse,
        "featureNames": FEATURE_NAMES,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
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
