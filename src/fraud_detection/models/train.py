# ruff: noqa: TRY004
from __future__ import annotations

import platform
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_detection.models.artifact import (
    ARTIFACT_SCHEMA_VERSION,
    PHASE2B_ARTIFACT_SCHEMA_VERSION,
    ModelBundle,
    save_bundle,
    validate_threshold,
)
from fraud_detection.models.evaluate import (
    Metrics,
    MetricValue,
    evaluate_scores,
    select_threshold,
)


def _package_versions() -> dict[str, str]:
    names = ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "PyYAML")
    result: dict[str, str] = {}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "unknown"
    return dict(sorted(result.items()))


def _valid_model_name(name: str) -> None:
    if not isinstance(name, str):
        raise ValueError("model_name must be a non-empty simple name")
    path = Path(name)
    if not name.strip() or path.name != name or path.suffix:
        raise ValueError("model_name must be a simple name without a path or suffix")
    if ".." in path.parts:
        raise ValueError("model_name must not contain '..'")


def _validate_train_config(config: TrainConfig) -> None:
    _valid_model_name(config.model_name)
    if not isinstance(config.model_version, str) or not config.model_version.strip():
        raise ValueError("model_version must be a non-empty string")
    if (
        isinstance(config.max_iter, bool)
        or not isinstance(config.max_iter, Integral)
        or config.max_iter <= 0
    ):
        raise ValueError("max_iter must be a positive integer")
    if (
        isinstance(config.random_state, bool)
        or not isinstance(config.random_state, Integral)
        or config.random_state < 0
    ):
        raise ValueError("random_state must be a non-negative integer")
    if config.sample_size is not None and (
        isinstance(config.sample_size, bool)
        or not isinstance(config.sample_size, Integral)
        or config.sample_size <= 0
    ):
        raise ValueError("sample_size must be None or a positive integer")


@dataclass
class TrainConfig:
    model_dir: Path
    model_name: str = "logistic_regression"
    model_version: str = "1.0"
    max_iter: int = 1000
    random_state: int = 42
    threshold: float = 0.5
    dataset_sha256: str | None = None
    sample_size: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelTrainer:
    """Train and persist one raw-feature-to-prediction sklearn pipeline."""

    def __init__(self, config: TrainConfig) -> None:
        _validate_train_config(config)
        config.threshold = validate_threshold(config.threshold)
        self.config = config
        self.model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=config.max_iter,
                        random_state=config.random_state,
                        solver="liblinear",
                    ),
                ),
            ]
        )
        self.feature_names: tuple[str, ...] | None = None
        self._fitted = False
        self._threshold_selection: dict[str, Any] | None = None

    @staticmethod
    def _validate_feature_values(features: pd.DataFrame) -> None:
        if features.empty:
            raise ValueError("Training features must be non-empty")
        for column in features:
            values = features[column]
            if pd.api.types.is_bool_dtype(values) or pd.api.types.is_complex_dtype(
                values
            ):
                raise ValueError("Training features must be real numeric")
            if not pd.api.types.is_numeric_dtype(values):
                raise ValueError("Training features must be numeric")
            try:
                numeric = values.to_numpy(dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError("Training features must be real numeric") from exc
            if not np.isfinite(numeric).all():
                raise ValueError("Training features must be finite")

    @staticmethod
    def _validate_target(y_train: np.ndarray | pd.Series, length: int) -> None:
        if isinstance(y_train, pd.DataFrame):
            raise ValueError("Training target must be a one-dimensional binary input")
        values = np.asarray(y_train)
        if values.ndim != 1:
            raise ValueError("Training target must be a one-dimensional binary input")
        if len(values) != length:
            raise ValueError("Training target length must match the number of rows")
        if values.size == 0 or pd.api.types.is_bool_dtype(values):
            raise ValueError("Training target must contain binary 0/1 values")
        try:
            numeric = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Training target must contain binary 0/1 values") from exc
        if not np.isfinite(numeric).all() or set(numeric.tolist()) != {0.0, 1.0}:
            raise ValueError("Training target must contain binary 0/1 values")

    def _prepare_features(
        self,
        features: pd.DataFrame | np.ndarray,
        feature_names: tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        if isinstance(features, pd.DataFrame):
            prepared = features.copy()
            actual = tuple(prepared.columns)
            if any(not isinstance(name, str) or not name.strip() for name in actual):
                raise ValueError("Feature columns must be non-empty strings")
            if len(set(actual)) != len(actual):
                raise ValueError("Feature columns must be unique")
            if feature_names is not None and actual != feature_names:
                raise ValueError("feature_names must exactly match DataFrame columns")
        elif isinstance(features, np.ndarray):
            if features.ndim != 2:
                raise ValueError(
                    "Training features must be a 2-dimensional DataFrame or ndarray"
                )
            names = feature_names or tuple(
                f"feature_{i}" for i in range(features.shape[1])
            )
            prepared = pd.DataFrame(features, columns=names)
        else:
            raise ValueError(
                "Training features must be a 2-dimensional DataFrame or ndarray"
            )
        self._validate_feature_values(prepared)
        return prepared

    def train(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: np.ndarray | pd.Series,
        feature_names: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        names = tuple(feature_names) if feature_names is not None else None
        X_prepared = self._prepare_features(X_train, names)
        if names is None:
            names = tuple(X_prepared.columns)
        if not names or any(
            not isinstance(name, str) or not name.strip() for name in names
        ):
            raise ValueError("feature_names must be non-empty unique strings")
        if len(set(names)) != len(names) or tuple(X_prepared.columns) != names:
            raise ValueError(
                "feature_names must be non-empty unique strings and match features"
            )
        self._validate_target(y_train, len(X_prepared))
        self.feature_names = names
        self.model.fit(X_prepared, y_train)
        self._fitted = True

    def select_threshold(
        self,
        X_validation: pd.DataFrame | np.ndarray,
        y_validation: np.ndarray | pd.Series,
    ) -> float:
        """Select and configure a validation-only maximum-F1 threshold."""
        if not self._fitted or self.feature_names is None:
            raise ValueError("Model must be trained before selecting a threshold")
        X_prepared = self._prepare_features(X_validation, self.feature_names)
        validation_scores = self.model.predict_proba(X_prepared)[:, 1]
        threshold = select_threshold(np.asarray(y_validation), validation_scores)
        self.config.threshold = threshold
        self._threshold_selection = {
            "method": "max_f1",
            "selection_split": "validation",
            "tie_break": "highest_threshold",
            "threshold": threshold,
        }
        return threshold

    def evaluate(
        self, X_test: pd.DataFrame | np.ndarray, y_test: np.ndarray | pd.Series
    ) -> Metrics:
        if not self._fitted or self.feature_names is None:
            raise ValueError("Model must be trained before evaluation")
        X_prepared = self._prepare_features(X_test, self.feature_names)
        y_scores = self.model.predict_proba(X_prepared)[:, 1]
        return evaluate_scores(y_test, y_scores, threshold=self.config.threshold)

    def save(
        self,
        metrics: Mapping[str, MetricValue] | None = None,
        *,
        validation_metrics: Mapping[str, MetricValue] | None = None,
        test_metrics: Mapping[str, MetricValue] | None = None,
        split_strategy: str | None = None,
        timestamp_column: str | None = None,
        split_counts: Mapping[str, int] | None = None,
    ) -> Path:
        if not self._fitted or self.feature_names is None:
            raise ValueError("Model must be trained before saving")
        phase2b_values = (
            validation_metrics,
            test_metrics,
            split_strategy,
            split_counts,
        )
        phase2b_supplied = any(value is not None for value in phase2b_values)
        if metrics is not None and (
            validation_metrics is not None or test_metrics is not None
        ):
            raise ValueError("Legacy metrics cannot be combined with Phase 2B metrics")
        if phase2b_supplied and not all(value is not None for value in phase2b_values):
            raise ValueError(
                "Phase 2B save requires validation/test metrics and split metadata"
            )
        if phase2b_supplied:
            assert validation_metrics is not None and test_metrics is not None
            assert split_counts is not None and split_strategy is not None
            if split_strategy not in {"random", "temporal"}:
                raise ValueError(
                    "Phase 2B split_strategy must be 'random' or 'temporal'"
                )
            if split_strategy == "temporal" and (
                not isinstance(timestamp_column, str) or not timestamp_column.strip()
            ):
                raise ValueError("Temporal Phase 2B save requires timestamp_column")
            if split_strategy == "random" and timestamp_column is not None:
                raise ValueError("Random Phase 2B save rejects timestamp_column")
            if set(split_counts) != {"train", "validation", "test"} or any(
                isinstance(count, bool) or not isinstance(count, int) or count <= 0
                for count in split_counts.values()
            ):
                raise ValueError(
                    "Phase 2B split_counts must contain positive train, validation, and test counts"
                )
            total = sum(split_counts.values())
            if self.config.sample_size is not None and self.config.sample_size != total:
                raise ValueError(
                    "Phase 2B split_counts must agree with configured sample_size"
                )
            self.config.sample_size = total
        metadata: dict[str, Any] = {
            "artifact_schema_version": (
                PHASE2B_ARTIFACT_SCHEMA_VERSION
                if phase2b_supplied
                else ARTIFACT_SCHEMA_VERSION
            ),
            "model_name": self.config.model_name,
            "model_version": self.config.model_version,
            "model_identity": {
                "name": self.config.model_name,
                "version": self.config.model_version,
            },
            "feature_names": list(self.feature_names),
            "threshold": self.config.threshold,
            "training": {
                "max_iter": self.config.max_iter,
                "random_state": self.config.random_state,
            },
            "dataset_sha256": self.config.dataset_sha256,
            "sample_size": self.config.sample_size,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "runtime_packages": _package_versions(),
            "evaluation_metrics": dict(metrics) if metrics is not None else {},
            "config_metadata": dict(self.config.metadata),
        }
        if self._threshold_selection is not None:
            metadata["threshold_selection"] = dict(self._threshold_selection)
        if phase2b_supplied:
            assert validation_metrics is not None and test_metrics is not None
            assert split_counts is not None and split_strategy is not None
            metadata.update(
                {
                    "split_strategy": split_strategy,
                    "timestamp_column": timestamp_column,
                    "split_counts": dict(split_counts),
                    "validation_metrics": dict(validation_metrics),
                    "test_metrics": dict(test_metrics),
                    "runtime_versions": {
                        "python": platform.python_version(),
                        "packages": _package_versions(),
                    },
                }
            )
        bundle = ModelBundle(
            self.model, self.feature_names, self.config.threshold, metadata
        )
        return save_bundle(
            bundle, self.config.model_dir / f"{self.config.model_name}.joblib"
        )

    def train_validate_test(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: np.ndarray | pd.Series,
        X_validation: pd.DataFrame | np.ndarray,
        y_validation: np.ndarray | pd.Series,
        X_test: pd.DataFrame | np.ndarray,
        y_test: np.ndarray | pd.Series,
        feature_names: tuple[str, ...] | list[str] | None = None,
        *,
        split_strategy: str = "random",
        timestamp_column: str | None = None,
    ) -> tuple[Path, Metrics, Metrics]:
        """Train on train only, select on validation only, then evaluate both splits."""
        self.train(X_train, y_train, feature_names=feature_names)
        self.select_threshold(X_validation, y_validation)
        validation_metrics = self.evaluate(X_validation, y_validation)
        test_metrics = self.evaluate(X_test, y_test)
        counts = {
            "train": len(y_train),
            "validation": len(y_validation),
            "test": len(y_test),
        }
        path = self.save(
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            split_strategy=split_strategy,
            timestamp_column=timestamp_column,
            split_counts=counts,
        )
        return path, validation_metrics, test_metrics

    def train_and_evaluate(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: np.ndarray | pd.Series,
        X_test: pd.DataFrame | np.ndarray,
        y_test: np.ndarray | pd.Series,
        feature_names: tuple[str, ...] | list[str] | None = None,
    ) -> tuple[Path, Metrics]:
        self.train(X_train, y_train, feature_names=feature_names)
        metrics = self.evaluate(X_test, y_test)
        return self.save(metrics), metrics
