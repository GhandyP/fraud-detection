import json
from pathlib import Path

import pandas as pd

from fraud_detection.cli import main

CONFIG = """dataset:
  file_name: transactions.csv
  target_column: label
train:
  test_size: 0.25
  validation_size: 0.2
  random_state: 7
  model_name: cli_model
"""


def _fixture(root: Path) -> None:
    (root / "config").mkdir()
    (root / "data" / "raw").mkdir(parents=True)
    (root / "config" / "config.yaml").write_text(CONFIG, encoding="utf-8")
    pd.DataFrame(
        {
            "amount": list(range(1, 17)),
            "age": list(range(20, 36)),
            "label": [0] * 8 + [1] * 8,
        }
    ).to_csv(root / "data" / "raw" / "transactions.csv", index=False)


def test_missing_or_malformed_config_returns_configuration_error(
    tmp_path: Path, capsys
) -> None:
    assert main(["validate-data", "--config", "missing.yaml"]) == 2
    assert "Configuration error" in capsys.readouterr().err
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("dataset: [", encoding="utf-8")
    assert main(["validate-data", "--config", str(malformed)]) == 2


def test_validate_data_reports_runtime_errors_and_success(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["validate-data"]) == 0
    assert "Dataset is valid" in capsys.readouterr().out
    invalid = tmp_path / "invalid.csv"
    pd.DataFrame({"amount": [1], "label": [0]}).to_csv(invalid, index=False)
    assert main(["validate-data", "--data", str(invalid)]) == 1
    assert "Error:" in capsys.readouterr().err


def test_train_json_and_sample_size(tmp_path: Path, monkeypatch, capsys) -> None:
    _fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["train", "--sample-size", "16", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert Path(result["artifact_path"]).is_file()
    assert "validation_metrics" in result
    assert "test_metrics" in result


def test_train_forwards_sample_size(monkeypatch, tmp_path: Path) -> None:
    _fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    captured: dict[str, object] = {}

    def fake_training(root, sample_size=None, config_path=None):
        captured.update(root=root, sample_size=sample_size, config_path=config_path)
        raise RuntimeError("stop after forwarding")

    monkeypatch.setattr("fraud_detection.cli.run_training", fake_training)
    assert main(["train", "--sample-size", "5"]) == 1
    assert captured["sample_size"] == 5


def _train_fixture(tmp_path: Path, monkeypatch) -> Path:
    _fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["train", "--json"]) == 0
    return tmp_path / "models" / "trained" / "cli_model.joblib"


def test_evaluate_json_missing_class_and_model_override(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    artifact = _train_fixture(tmp_path, monkeypatch)
    capsys.readouterr()
    assert main(["evaluate", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert {
        "pr_auc",
        "roc_auc",
        "recall",
        "precision",
        "f1",
        "confusion_matrix",
    } <= result.keys()
    assert main(["evaluate", "--model", str(artifact), "--json"]) == 0
    capsys.readouterr()
    one_class = tmp_path / "one-class.csv"
    pd.DataFrame({"amount": [1, 2], "age": [20, 21], "label": [0, 0]}).to_csv(
        one_class, index=False
    )
    assert main(["evaluate", "--data", str(one_class)]) == 1
    assert "both binary classes" in capsys.readouterr().err
    artifact.unlink()
    assert main(["evaluate"]) == 1
    capsys.readouterr()


def test_predict_record_batch_and_schema_errors(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _train_fixture(tmp_path, monkeypatch)
    capsys.readouterr()
    record = json.dumps({"amount": 3, "age": 22})
    assert main(["predict", "--record", record, "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["threshold"] >= 0
    assert result["model_version"] == "1.0"
    item = result["predictions"][0]
    assert {
        "prediction",
        "fraud_probability",
        "threshold",
        "model_version",
    } <= item.keys()
    batch = tmp_path / "batch.csv"
    pd.DataFrame({"amount": [3, 4], "age": [22, 23], "label": [0, 1]}).to_csv(
        batch, index=False
    )
    assert main(["predict", "--input", str(batch), "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)["predictions"]) == 2
    wrong = tmp_path / "wrong.csv"
    pd.DataFrame({"amount": [3], "other": [22]}).to_csv(wrong, index=False)
    assert main(["predict", "--input", str(wrong)]) == 1
    capsys.readouterr()


def test_predict_record_input_are_mutually_exclusive() -> None:
    assert main(["predict"]) == 2
    assert main(["predict", "--record", "{}", "--input", "x.csv"]) == 2


def test_full_cli_fixture_workflow_has_no_repository_outputs(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["validate-data"]) == 0
    capsys.readouterr()
    assert main(["train", "--json"]) == 0
    capsys.readouterr()
    assert main(["predict", "--record", '{"amount": 3, "age": 22}', "--json"]) == 0
    capsys.readouterr()
    assert main(["evaluate", "--data", "data/raw/transactions.csv", "--json"]) == 0
    capsys.readouterr()
    assert not list(Path(__file__).parent.parent.glob("models/**/*.joblib"))


def test_usage_errors_return_two() -> None:
    assert main(["unknown"]) == 2
    assert main(["train", "--sample-size", "0"]) == 2
