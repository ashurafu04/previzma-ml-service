from __future__ import annotations

from dataclasses import dataclass
from math import expm1, log1p
from typing import Any, Iterable

import numpy as np

RAW_TARGET = "raw"
LOG1P_TARGET = "log1p"
CLIPPED_RAW_TARGET = "clipped_raw"
CLIPPED_LOG1P_TARGET = "clipped_log1p"
BASELINE_RATIO_LOG1P_TARGET = "baseline_ratio_log1p"
CLIPPED_BASELINE_RATIO_LOG1P_TARGET = "clipped_baseline_ratio_log1p"

DEFAULT_TARGET_STRATEGIES = (
    RAW_TARGET,
    LOG1P_TARGET,
    CLIPPED_RAW_TARGET,
    CLIPPED_LOG1P_TARGET,
    BASELINE_RATIO_LOG1P_TARGET,
    CLIPPED_BASELINE_RATIO_LOG1P_TARGET,
)


@dataclass(frozen=True)
class TargetStrategy:
    name: str
    clip_lower: float | None = None
    clip_upper: float | None = None

    @property
    def uses_log(self) -> bool:
        return self.name in {
            LOG1P_TARGET,
            CLIPPED_LOG1P_TARGET,
            BASELINE_RATIO_LOG1P_TARGET,
            CLIPPED_BASELINE_RATIO_LOG1P_TARGET,
        }

    @property
    def uses_clipping(self) -> bool:
        return self.name in {
            CLIPPED_RAW_TARGET,
            CLIPPED_LOG1P_TARGET,
            CLIPPED_BASELINE_RATIO_LOG1P_TARGET,
        }

    @property
    def uses_baseline_ratio(self) -> bool:
        return self.name in {
            BASELINE_RATIO_LOG1P_TARGET,
            CLIPPED_BASELINE_RATIO_LOG1P_TARGET,
        }

    def to_metadata(self) -> dict[str, float | str | None]:
        return {
            "targetStrategy": self.name,
            "targetClipLower": self.clip_lower,
            "targetClipUpper": self.clip_upper,
        }


def fit_target_strategy(
    name: str,
    targets: Iterable[float],
    baseline_predictions: Iterable[float] | None = None,
    clip_lower_percentile: float = 1.0,
    clip_upper_percentile: float = 99.0,
) -> TargetStrategy:
    if name not in DEFAULT_TARGET_STRATEGIES:
        valid = ", ".join(DEFAULT_TARGET_STRATEGIES)
        raise ValueError(f"Unsupported target strategy '{name}'. Expected one of: {valid}")

    if name not in {
        CLIPPED_RAW_TARGET,
        CLIPPED_LOG1P_TARGET,
        CLIPPED_BASELINE_RATIO_LOG1P_TARGET,
    }:
        return TargetStrategy(name=name)

    target_values = [max(float(target), 0.0) for target in targets]
    baseline_values = (
        list(baseline_predictions) if baseline_predictions is not None else []
    )
    if name == CLIPPED_BASELINE_RATIO_LOG1P_TARGET:
        values = np.array(
            [
                target / _safe_baseline_denominator(
                    baseline_values[index] if index < len(baseline_values) else None
                )
                for index, target in enumerate(target_values)
            ],
            dtype=float,
        )
    else:
        values = np.array(target_values, dtype=float)
    if values.size == 0:
        return TargetStrategy(name=name, clip_lower=0.0, clip_upper=0.0)

    lower = float(np.percentile(values, clip_lower_percentile))
    upper = float(np.percentile(values, clip_upper_percentile))
    if upper < lower:
        upper = lower

    return TargetStrategy(name=name, clip_lower=lower, clip_upper=upper)


def transform_targets(
    targets: Iterable[float],
    strategy: TargetStrategy,
    baseline_predictions: Iterable[float] | None = None,
) -> list[float]:
    baselines = list(baseline_predictions) if baseline_predictions is not None else []
    return [
        transform_target(
            target,
            strategy,
            baseline_prediction=baselines[index] if index < len(baselines) else None,
        )
        for index, target in enumerate(targets)
    ]


def transform_target(
    target: float,
    strategy: TargetStrategy,
    baseline_prediction: float | None = None,
) -> float:
    value = max(float(target), 0.0)
    if strategy.uses_baseline_ratio:
        value = value / _safe_baseline_denominator(baseline_prediction)

    value = _clip_if_needed(value, strategy)

    if strategy.uses_log:
        return log1p(value)

    return value


def inverse_transform_prediction_from_metadata(
    prediction: float,
    metadata: dict[str, Any],
    baseline_prediction: float | None = None,
) -> float:
    strategy_name = str(metadata.get("targetStrategy") or RAW_TARGET)
    strategy = TargetStrategy(
        name=strategy_name,
        clip_lower=_optional_float(metadata.get("targetClipLower")),
        clip_upper=_optional_float(metadata.get("targetClipUpper")),
    )
    return inverse_transform_prediction(
        prediction,
        strategy,
        baseline_prediction=baseline_prediction,
    )


def inverse_transform_prediction(
    prediction: float,
    strategy: TargetStrategy,
    baseline_prediction: float | None = None,
) -> float:
    value = float(prediction)
    if strategy.uses_log:
        value = expm1(value)

    if strategy.uses_baseline_ratio:
        value = value * _safe_baseline_denominator(baseline_prediction)

    return max(value, 0.0)


def _clip_if_needed(value: float, strategy: TargetStrategy) -> float:
    if not strategy.uses_clipping:
        return value

    if strategy.clip_lower is not None:
        value = max(value, strategy.clip_lower)

    if strategy.clip_upper is not None:
        value = min(value, strategy.clip_upper)

    return value


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_baseline_denominator(value: float | None) -> float:
    if value is None:
        return 1.0

    return max(float(value), 1.0)
