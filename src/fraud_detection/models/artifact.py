# ruff: noqa: TRY004
from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

ARTIFACT_SCHEMA_VERSION = "1.0"
PHASE2B_ARTIFACT_SCHEMA_VERSION = "2.0"
_METRIC_KEYS = {"pr_auc", "roc_auc", "recall", "precision", "f1", "confusion_matrix"}


def validate_threshold(value: Any, *, context: str = "threshold") -> float:
    """Validate and normalize a classification threshold."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{context} must be a finite real number between 0 and 1")
    numeric_value = float(value)
    if not isfinite(numeric_value) or not 0 <= numeric_value <= 1:
        raise ValueError(f"{context} must be a finite real number between 0 and 1")
    return numeric_value


@dataclass
class ModelBundle:
    """Trusted, fitted end-to-end model artifact and its inference contract."""

    pipeline: Pipeline
    feature_names: tuple[str, ...]
    threshold: float
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.pipeline, Pipeline) or [
            name for name, _ in self.pipeline.steps
        ] != ["scaler", "classifier"]:
            raise ValueError(
                "Artifact Pipeline must contain the expected scaler and classifier steps"
            )
        if not isinstance(self.pipeline.named_steps["scaler"], StandardScaler):
            raise ValueError("Artifact scaler step must be StandardScaler")
        if not isinstance(self.pipeline.named_steps["classifier"], LogisticRegression):
            raise ValueError("Artifact classifier step must be LogisticRegression")
        try:
            check_is_fitted(self.pipeline)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Artifact pipeline must be a fitted sklearn Pipeline"
            ) from exc
        if (
            not self.feature_names
            or any(not isinstance(name, str) or not name for name in self.feature_names)
            or len(set(self.feature_names)) != len(self.feature_names)
        ):
            raise ValueError("Artifact feature_names must be non-empty unique strings")
        validate_threshold(self.threshold, context="Artifact threshold")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("Artifact metadata must be a mapping")
        _validate_json_like(self.metadata)
        _validate_versioned_metadata(self.metadata, self.threshold, self.feature_names)


def _validate_metrics(value: Any, context: str) -> None:
    if not isinstance(value, Mapping) or set(value) != _METRIC_KEYS:
        raise ValueError(f"{context} must contain the fixed metric contract")
    matrix = value["confusion_matrix"]
    if (
        not isinstance(matrix, list)
        or len(matrix) != 2
        or any(not isinstance(row, list) or len(row) != 2 for row in matrix)
        or any(
            isinstance(item, bool) or not isinstance(item, Integral) or item < 0
            for row in matrix
            for item in row
        )
    ):
        raise ValueError(f"{context}.confusion_matrix must be a 2x2 count matrix")
    for key in _METRIC_KEYS - {"confusion_matrix"}:
        if (
            isinstance(value[key], bool)
            or not isinstance(value[key], Real)
            or not isfinite(float(value[key]))
            or not 0 <= float(value[key]) <= 1
        ):
            raise ValueError(f"{context}.{key} must be a finite number in [0, 1]")


def _validate_versioned_metadata(
    metadata: Mapping[str, Any], threshold: float, feature_names: tuple[str, ...]
) -> None:
    version = metadata.get("artifact_schema_version")
    if version is None or version == ARTIFACT_SCHEMA_VERSION:
        return
    if version != PHASE2B_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Unsupported artifact schema version")
    required = (
        "model_name",
        "model_version",
        "model_identity",
        "feature_names",
        "threshold",
        "training",
        "dataset_sha256",
        "sample_size",
        "created_at_utc",
        "timestamp_utc",
        "python_version",
        "runtime_packages",
        "runtime_versions",
        "split_strategy",
        "split_counts",
        "threshold_selection",
        "validation_metrics",
        "test_metrics",
    )
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(
            f"Phase 2B metadata missing required fields: {', '.join(missing)}"
        )
    if (
        not isinstance(metadata["model_name"], str)
        or not metadata["model_name"].strip()
        or not isinstance(metadata["model_version"], str)
        or not metadata["model_version"].strip()
        or not isinstance(metadata["model_identity"], Mapping)
        or metadata["model_identity"].get("name") != metadata["model_name"]
        or metadata["model_identity"].get("version") != metadata["model_version"]
        or not isinstance(metadata["feature_names"], (list, tuple))
        or tuple(metadata["feature_names"]) != feature_names
        or metadata["threshold"] != threshold
    ):
        raise ValueError("Phase 2B model identity or feature contract is inconsistent")
    training = metadata["training"]
    if (
        not isinstance(training, Mapping)
        or isinstance(training.get("random_state"), bool)
        or not isinstance(training.get("random_state"), Integral)
        or training["random_state"] < 0
    ):
        raise ValueError("Phase 2B training random_state is invalid")
    dataset_hash = metadata["dataset_sha256"]
    if not isinstance(dataset_hash, str) or not re.fullmatch(
        r"[0-9a-fA-F]{64}", dataset_hash
    ):
        raise ValueError("Phase 2B dataset_sha256 is invalid")
    for key in ("created_at_utc", "timestamp_utc"):
        try:
            parsed = datetime.fromisoformat(metadata[key])
            if parsed.tzinfo is None:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Phase 2B {key} is invalid") from exc
    if (
        not isinstance(metadata["python_version"], str)
        or not isinstance(metadata["runtime_packages"], Mapping)
        or any(
            not isinstance(value, str)
            for value in metadata["runtime_packages"].values()
        )
    ):
        raise ValueError("Phase 2B runtime metadata is invalid")
    runtime = metadata["runtime_versions"]
    if (
        not isinstance(runtime, Mapping)
        or not isinstance(runtime.get("python"), str)
        or not isinstance(runtime.get("packages"), Mapping)
        or any(not isinstance(value, str) for value in runtime["packages"].values())
        or runtime["python"] != metadata["python_version"]
        or dict(runtime["packages"]) != dict(metadata["runtime_packages"])
    ):
        raise ValueError("Phase 2B runtime_versions is inconsistent")
    counts = metadata["split_counts"]
    if (
        not isinstance(counts, Mapping)
        or set(counts) != {"train", "validation", "test"}
        or any(
            isinstance(v, bool) or not isinstance(v, Integral) or v <= 0
            for v in counts.values()
        )
    ):
        raise ValueError(
            "Phase 2B split_counts must contain positive train, validation, and test counts"
        )
    if metadata["split_strategy"] != "random":
        raise ValueError("Phase 2B split_strategy must be 'random'")
    sample_size = metadata["sample_size"]
    if (
        isinstance(sample_size, bool)
        or not isinstance(sample_size, Integral)
        or sample_size <= 0
        or sum(counts.values()) != sample_size
    ):
        raise ValueError("Phase 2B split_counts must agree with sample_size")
    selection = metadata["threshold_selection"]
    if (
        not isinstance(selection, Mapping)
        or selection.get("method") != "max_f1"
        or selection.get("selection_split") != "validation"
        or selection.get("tie_break") != "highest_threshold"
    ):
        raise ValueError("Phase 2B threshold_selection must use validation")
    try:
        selected = validate_threshold(
            selection["threshold"], context="Phase 2B selected threshold"
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Phase 2B threshold_selection has an invalid threshold"
        ) from exc
    if selected != threshold:
        raise ValueError("Phase 2B selected threshold disagrees with bundle")
    _validate_metrics(metadata["validation_metrics"], "Phase 2B validation_metrics")
    _validate_metrics(metadata["test_metrics"], "Phase 2B test_metrics")
    for key, split in (("validation_metrics", "validation"), ("test_metrics", "test")):
        if sum(sum(row) for row in metadata[key]["confusion_matrix"]) != counts[split]:
            raise ValueError(
                f"Phase 2B {key} confusion matrix disagrees with split count"
            )


def _validate_json_like(value: Any, path: str = "metadata") -> None:
    if value is None or isinstance(value, (str, int, bool)):
        return
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{path} must contain only finite numbers")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings")
            _validate_json_like(nested, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _validate_json_like(nested, f"{path}[{index}]")
        return
    raise ValueError(f"{path} must contain only JSON-like values")


def save_bundle(bundle: ModelBundle, path: Path) -> Path:
    """Atomically save a trusted-input joblib bundle."""
    bundle.__post_init__()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = stream.name
        joblib.dump(bundle, temporary)
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass
        raise
    return path


def load_bundle(path: Path) -> ModelBundle:
    """Load a trusted joblib artifact; never load artifacts from untrusted sources."""
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found at {path}")
    try:
        value = joblib.load(path)
    except Exception as exc:
        raise ValueError(f"Unable to load model artifact {path}: {exc}") from exc
    if not isinstance(value, ModelBundle):
        raise ValueError(
            "Malformed or legacy model artifact: expected a Phase 2A ModelBundle"
        )
    try:
        value.__post_init__()
    except ValueError as exc:
        raise ValueError(f"Malformed model artifact: {exc}") from exc
    schema_version = value.metadata.get("artifact_schema_version")
    if schema_version is None:
        raise ValueError("Malformed model artifact: missing required schema metadata")
    if schema_version not in (ARTIFACT_SCHEMA_VERSION, PHASE2B_ARTIFACT_SCHEMA_VERSION):
        raise ValueError(
            "Malformed model artifact: unsupported artifact schema version"
        )
    for key in ("feature_names", "threshold"):
        if key not in value.metadata:
            raise ValueError(
                f"Malformed model artifact: missing required metadata '{key}'"
            )
    try:
        metadata_names = tuple(value.metadata["feature_names"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Malformed model artifact: metadata feature_names must be a sequence"
        ) from exc
    if metadata_names != value.feature_names:
        raise ValueError(
            "Malformed model artifact: metadata feature_names disagree with bundle"
        )
    try:
        metadata_threshold = validate_threshold(
            value.metadata["threshold"], context="Artifact metadata threshold"
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Malformed model artifact: {exc}") from exc
    if metadata_threshold != value.threshold:
        raise ValueError(
            "Malformed model artifact: metadata threshold disagrees with bundle"
        )
    return value
