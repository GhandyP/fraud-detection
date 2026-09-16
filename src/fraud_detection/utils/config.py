from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when the project YAML configuration is missing or invalid."""


@dataclass(frozen=True)
class DatasetSettings:
    file_name: str
    target_column: str


@dataclass(frozen=True)
class TrainSettings:
    test_size: float
    random_state: int


@dataclass(frozen=True)
class AppConfig:
    dataset: DatasetSettings
    train: TrainSettings


@dataclass
class ProjectPaths:
    root: Path

    @property
    def data_raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def models_trained(self) -> Path:
        return self.root / "models" / "trained"


def _required_string(section: dict[str, Any], name: str) -> str:
    value = section.get(
        name, {"file_name": "creditcard.csv", "target_column": "Class"}[name]
    )
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"dataset.{name} must be a non-empty string")
    return value.strip()


def _validate_file_name(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ConfigError("dataset.file_name must be a relative filename without '..'")
    if path.name != value or not path.name:
        raise ConfigError("dataset.file_name must name a file, not a directory")
    return value


def load_config(root: Path, config_path: str | Path | None = None) -> AppConfig:
    """Load and validate YAML configuration, resolving relative paths from root."""
    root = Path(root).resolve()
    path = Path(config_path) if config_path is not None else Path("config/config.yaml")
    if not path.is_absolute():
        path = root / path
    try:
        with path.open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Malformed YAML in {path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Unable to read configuration {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError("Configuration must be a YAML mapping")
    dataset = raw.get("dataset")
    train = raw.get("train")
    if not isinstance(dataset, dict):
        raise ConfigError("Missing or invalid 'dataset' section; expected a mapping")
    if not isinstance(train, dict):
        raise ConfigError("Missing or invalid 'train' section; expected a mapping")

    file_name = _validate_file_name(_required_string(dataset, "file_name"))
    target = _required_string(dataset, "target_column")
    test_size = train.get("test_size", 0.2)
    if (
        isinstance(test_size, bool)
        or not isinstance(test_size, (int, float))
        or not 0 < test_size < 1
    ):
        raise ConfigError("train.test_size must be a number strictly between 0 and 1")
    random_state = train.get("random_state", 42)
    if (
        isinstance(random_state, bool)
        or not isinstance(random_state, int)
        or random_state < 0
    ):
        raise ConfigError("train.random_state must be a non-negative integer")

    return AppConfig(
        dataset=DatasetSettings(file_name=file_name, target_column=target),
        train=TrainSettings(test_size=float(test_size), random_state=random_state),
    )
