from __future__ import annotations

import math

import numpy as np
import pandas as pd

from fraud_detection.data.preprocessor import PreprocessConfig


class DataValidationError(ValueError):
    """Raised when training data cannot satisfy the pipeline contract."""


def validate_dataset(df: pd.DataFrame, config: PreprocessConfig) -> None:
    """Validate data before preprocessing or model fitting."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise DataValidationError("Dataset must be a non-empty pandas DataFrame")
    if not df.columns.is_unique:
        raise DataValidationError("Dataset columns must be unique")
    target = config.target_column
    if target not in df.columns:
        raise DataValidationError(f"Dataset is missing target column '{target}'")

    features = df.drop(columns=[target])
    if features.empty:
        raise DataValidationError("Dataset must contain at least one feature column")
    non_numeric = [
        name for name in features if not pd.api.types.is_numeric_dtype(features[name])
    ]
    if non_numeric:
        raise DataValidationError(f"Feature columns must be numeric: {non_numeric}")
    if features.isna().any().any() or not np.isfinite(features.to_numpy()).all():
        raise DataValidationError(
            "Feature columns must not contain missing or non-finite values"
        )
    if df[target].isna().any():
        raise DataValidationError("Target column must not contain missing values")
    if (
        pd.api.types.is_numeric_dtype(df[target])
        and not np.isfinite(df[target].to_numpy()).all()
    ):
        raise DataValidationError("Target column must not contain non-finite values")
    classes = df[target].unique()
    if len(classes) != 2:
        raise DataValidationError("Target column must contain exactly two classes")
    counts = df[target].value_counts()
    if (counts < 2).any():
        raise DataValidationError("Each target class must contain at least two samples")

    n_samples = len(df)
    test_count = math.ceil(n_samples * config.test_size)
    train_count = n_samples - test_count
    if test_count < len(classes) or train_count < len(classes):
        raise DataValidationError(
            "Configured stratified split must leave at least one sample of every class "
            f"in train and test (test_size={config.test_size})"
        )
