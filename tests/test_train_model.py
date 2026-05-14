import json
from pathlib import Path

from app.services.model_registry import (
    MODEL_PATH_ENV,
    clear_forecast_model_cache,
    get_forecast_model,
)
from app.training.train_model import main

FIXTURE = Path("tests/fixtures/sales_export_minimal.csv")


def test_train_model_writes_artifact_and_metadata(tmp_path, monkeypatch) -> None:
    output_path = tmp_path / "forecast_model.joblib"

    exit_code = main(
        [
            "--input",
            str(FIXTURE),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    metadata_path = tmp_path / "model_metadata.json"
    assert output_path.exists()
    assert metadata_path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["modelVersion"] == "lightgbm-window-v1"
    assert metadata["algorithm"] == "LightGBM LGBMRegressor"
    assert metadata["horizonsSupported"] == [30, 60, 90]
    assert metadata["trainingRows"] > 0
    assert metadata["validationMae"] >= 0
    assert metadata["validationRmse"] >= 0
    assert metadata["targetStrategy"] in {
        "raw",
        "log1p",
        "clipped_raw",
        "clipped_log1p",
        "baseline_ratio_log1p",
        "clipped_baseline_ratio_log1p",
    }
    assert {candidate["targetStrategy"] for candidate in metadata["targetStrategyCandidates"]} == {
        "raw",
        "log1p",
        "clipped_raw",
        "clipped_log1p",
        "baseline_ratio_log1p",
        "clipped_baseline_ratio_log1p",
    }

    monkeypatch.setenv(MODEL_PATH_ENV, str(output_path))
    clear_forecast_model_cache()
    loaded_model = get_forecast_model()

    assert loaded_model is not None
    assert loaded_model.model_version == "lightgbm-window-v1"

    clear_forecast_model_cache()
