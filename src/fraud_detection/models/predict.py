# ruff: noqa: TRY004
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from fraud_detection.models.artifact import ModelBundle, load_bundle


class ModelPredictor:
    """Load a trusted Phase 2A bundle and predict from raw features."""

    def __init__(self, model_path: Path) -> None:
        self.bundle: ModelBundle = load_bundle(model_path)
        self.model = self.bundle.pipeline

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.bundle.feature_names

    @property
    def threshold(self) -> float:
        return self.bundle.threshold

    @property
    def metadata(self) -> Mapping[str, Any]:
        return MappingProxyType(deepcopy(dict(self.bundle.metadata)))

    def _prepare(self, features: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(features, pd.DataFrame):
            actual = tuple(features.columns)
            if actual != self.feature_names:
                raise ValueError(
                    f"Feature columns must exactly match {list(self.feature_names)}; "
                    f"got {list(actual)}"
                )
            prepared = features
        elif isinstance(features, np.ndarray):
            if features.ndim != 2:
                raise ValueError(
                    "features must be a 2-dimensional DataFrame or ndarray"
                )
            if features.shape[1] != len(self.feature_names):
                raise ValueError(
                    f"Expected {len(self.feature_names)} features, got {features.shape[1]}"
                )
            prepared = pd.DataFrame(features, columns=self.feature_names)
        else:
            raise ValueError("features must be a 2-dimensional DataFrame or ndarray")
        self._validate_numeric(prepared)
        return prepared

    @staticmethod
    def _validate_numeric(features: pd.DataFrame) -> None:
        for column in features:
            values = features[column]
            if pd.api.types.is_bool_dtype(values) or pd.api.types.is_complex_dtype(
                values
            ):
                raise ValueError("Prediction features must be real numeric")
            if not pd.api.types.is_numeric_dtype(values):
                raise ValueError("Prediction features must be numeric")
            try:
                numeric = values.to_numpy(dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError("Prediction features must be real numeric") from exc
            if not np.isfinite(numeric).all():
                raise ValueError("Prediction features must be finite and non-missing")

    def predict(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        scores = self.model.predict_proba(self._prepare(features))[:, 1]
        return (scores >= self.threshold).astype(int)

    def predict_proba(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self._prepare(features))
