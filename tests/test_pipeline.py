from pathlib import Path

import pandas as pd

from fraud_detection import pipeline


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
