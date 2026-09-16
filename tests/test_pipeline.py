from pathlib import Path

import pandas as pd

from fraud_detection import pipeline
from fraud_detection.models.predict import ModelPredictor


def test_run_training_wires_yaml_values(monkeypatch, tmp_path: Path) -> None:
    config = tmp_path / "settings.yaml"
    config.write_text(
        "dataset:\n  file_name: custom.csv\n  target_column: label\n"
        "train:\n  test_size: 0.3\n  random_state: 19\n",
        encoding="utf-8",
    )
    captured = {}

    class FakeLoader:
        def __init__(self, value):
            captured["dataset"] = value

        def load(self, sample_size=None):
            captured["sample_size"] = sample_size
            return pd.DataFrame({"value": [1, 2, 3, 4], "label": [0, 0, 1, 1]})

    class FakePreprocessor:
        def __init__(self, value):
            captured["preprocess"] = value

        def split(self, df):
            return df[["value"]], df[["value"]], df["label"], df["label"]

        def scale(self, train, test):
            return train, test

    class FakeTrainer:
        def __init__(self, value):
            captured["trainer"] = value

        def train_and_evaluate(self, *args):
            return tmp_path / "model.joblib", {}

    monkeypatch.setattr(pipeline, "DataLoader", FakeLoader)
    monkeypatch.setattr(pipeline, "FraudPreprocessor", FakePreprocessor)
    monkeypatch.setattr(pipeline, "ModelTrainer", FakeTrainer)
    monkeypatch.setattr(pipeline, "validate_dataset", lambda df, config: None)

    pipeline.run_training(tmp_path, sample_size=3, config_path=config)

    assert captured["dataset"].file_name == "custom.csv"
    assert captured["preprocess"].target_column == "label"
    assert captured["preprocess"].test_size == 0.3
    assert captured["preprocess"].random_state == 19
    assert captured["trainer"].random_state == 19
    assert captured["sample_size"] == 3


def test_run_training_persists_raw_feature_bundle(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    pd.DataFrame(
        {
            "amount": list(range(1, 17)),
            "age": [20, 21, 22, 23, 24, 25, 26, 27, 50, 51, 52, 53, 54, 55, 56, 57],
            "label": [0] * 8 + [1] * 8,
        }
    ).to_csv(raw / "synthetic.csv", index=False)
    config = tmp_path / "settings.yaml"
    config.write_text(
        "dataset:\n  file_name: synthetic.csv\n  target_column: label\n"
        "train:\n  test_size: 0.25\n  random_state: 7\n"
        "  model_name: synthetic_model\n  model_version: 'test'\n  threshold: 0.5\n",
        encoding="utf-8",
    )

    pipeline.run_training(tmp_path, config_path=config)

    predictor = ModelPredictor(
        tmp_path / "models" / "trained" / "synthetic_model.joblib"
    )
    predictions = predictor.predict(pd.DataFrame({"amount": [2, 14], "age": [21, 54]}))
    assert predictions.shape == (2,)
    assert predictor.metadata["model_version"] == "test"
