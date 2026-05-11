from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = REPO_ROOT / "app" / "models" / "forecast_model.joblib"
DEFAULT_METADATA_PATH = REPO_ROOT / "app" / "models" / "model_metadata.json"
MODEL_PATH_ENV = "PREVIZMA_FORECAST_MODEL_PATH"
METADATA_PATH_ENV = "PREVIZMA_FORECAST_METADATA_PATH"

_cached_model: LoadedForecastModel | None = None
_cached_model_path: Path | None = None


@dataclass(frozen=True)
class LoadedForecastModel:
    model: Any
    feature_names: list[str]
    metadata: dict[str, Any]

    @property
    def model_version(self) -> str:
        return str(self.metadata.get("modelVersion", "trained-forecast-model"))

    @property
    def validation_mape(self) -> float | None:
        value = self.metadata.get("validationMape")
        return float(value) if value is not None else None

    def supports_horizon(self, horizon: int) -> bool:
        horizons = self.metadata.get("horizonsSupported")
        if not horizons:
            return True

        return int(horizon) in {int(value) for value in horizons}


def get_forecast_model() -> LoadedForecastModel | None:
    global _cached_model
    global _cached_model_path

    model_path = _model_path()
    if not model_path.exists():
        _cached_model = None
        _cached_model_path = model_path
        return None

    if _cached_model is not None and _cached_model_path == model_path:
        return _cached_model

    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError(f"Invalid forecast model artifact: {model_path}")

    metadata = dict(artifact.get("metadata") or {})
    metadata.update(_load_metadata_file(model_path))
    feature_names = list(artifact.get("featureNames") or metadata.get("featureNames") or [])
    if not feature_names:
        raise ValueError(f"Forecast model artifact has no feature names: {model_path}")

    _cached_model = LoadedForecastModel(
        model=artifact["model"],
        feature_names=feature_names,
        metadata=metadata,
    )
    _cached_model_path = model_path
    return _cached_model


def clear_forecast_model_cache() -> None:
    global _cached_model
    global _cached_model_path

    _cached_model = None
    _cached_model_path = None


def _model_path() -> Path:
    return Path(os.environ.get(MODEL_PATH_ENV, DEFAULT_MODEL_PATH))


def _metadata_path(model_path: Path) -> Path:
    configured_path = os.environ.get(METADATA_PATH_ENV)
    if configured_path:
        return Path(configured_path)

    sibling_metadata_path = model_path.with_name("model_metadata.json")
    if sibling_metadata_path.exists():
        return sibling_metadata_path

    return DEFAULT_METADATA_PATH


def _load_metadata_file(model_path: Path) -> dict[str, Any]:
    metadata_path = _metadata_path(model_path)
    if not metadata_path.exists():
        return {}

    return json.loads(metadata_path.read_text(encoding="utf-8"))
