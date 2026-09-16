from pathlib import Path

import pytest

from fraud_detection.utils.config import ConfigError, load_config

VALID = """
dataset:
  file_name: transactions.csv
  target_column: fraud
train:
  test_size: 0.25
  random_state: 7
"""


def test_load_config_resolves_default_path_and_types(tmp_path: Path) -> None:
    config = tmp_path / "config" / "config.yaml"
    config.parent.mkdir()
    config.write_text(VALID, encoding="utf-8")

    loaded = load_config(tmp_path)

    assert loaded.dataset.file_name == "transactions.csv"
    assert loaded.dataset.target_column == "fraud"
    assert loaded.train.test_size == 0.25
    assert loaded.train.validation_size == 0.2
    assert loaded.train.split_strategy == "random"
    assert loaded.train.random_state == 7


@pytest.mark.parametrize(
    "yaml_text, message",
    [
        ("dataset: [bad]\ntrain: {}", "dataset"),
        (
            "dataset:\n  file_name: ../data.csv\n  target_column: fraud\ntrain: {}",
            "relative",
        ),
        (
            "dataset:\n  file_name: data.csv\n  target_column: '  '\ntrain: {}",
            "target_column",
        ),
        (
            "dataset:\n  file_name: data.csv\n  target_column: fraud\ntrain:\n  test_size: 1\n  random_state: 1",
            "test_size",
        ),
        (
            "dataset:\n  file_name: data.csv\n  target_column: fraud\ntrain:\n  test_size: .2\n  random_state: true",
            "random_state",
        ),
        (
            "dataset:\n  file_name: data.csv\n  target_column: fraud\ntrain:\n  validation_size: 0.8\n  test_size: 0.2",
            "sum to less",
        ),
        (
            "dataset:\n  file_name: data.csv\n  target_column: fraud\ntrain:\n  split_strategy: temporal",
            "split_strategy",
        ),
        (
            "dataset:\n  file_name: data.csv\n  target_column: fraud\ntrain:\n  timestamp_column: created_at",
            "timestamp",
        ),
    ],
)
def test_load_config_rejects_invalid_values(
    tmp_path: Path, yaml_text: str, message: str
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_config(tmp_path, config)


def test_load_config_rejects_malformed_yaml(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("dataset: [", encoding="utf-8")

    with pytest.raises(ConfigError, match="Malformed YAML"):
        load_config(tmp_path, config)
