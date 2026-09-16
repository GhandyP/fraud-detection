# ruff: noqa: TRY004
from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

ARTIFACT_SCHEMA_VERSION = "1.0"


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
    if schema_version != ARTIFACT_SCHEMA_VERSION:
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
