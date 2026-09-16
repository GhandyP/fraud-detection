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
    if any(not isinstance(name, str) or not name.strip() for name in df.columns):
        raise DataValidationError("Dataset feature names must be non-empty strings")
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
    if features.isna().any().any():
        raise DataValidationError(
            "Feature columns must not contain missing or non-finite values"
        )
    for name in features:
        column = features[name]
        if pd.api.types.is_bool_dtype(column) or pd.api.types.is_complex_dtype(column):
            raise DataValidationError("Feature columns must be real numeric values")
        try:
            values = column.to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise DataValidationError(
                "Feature columns must be real numeric values"
            ) from exc
        if not np.isfinite(values).all():
            raise DataValidationError(
                "Feature columns must not contain missing or non-finite values"
            )
    if not pd.api.types.is_numeric_dtype(df[target]) or pd.api.types.is_bool_dtype(
        df[target]
    ):
        raise DataValidationError("Target column must contain numeric 0/1 values")
    if df[target].isna().any():
        raise DataValidationError("Target column must contain finite 0/1 values")
    try:
        target_values = df[target].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            "Target column must contain finite 0/1 values"
        ) from exc
    if not np.isfinite(target_values).all():
        raise DataValidationError("Target column must contain finite 0/1 values")
    classes = set(df[target].unique())
    if classes != {0, 1}:
        raise DataValidationError(
            "Target column must contain exactly two numeric values: 0 and 1"
        )
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
