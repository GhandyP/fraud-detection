from __future__ import annotations

import math

import numpy as np
import pandas as pd

from fraud_detection.data.preprocessor import PreprocessConfig


class DataValidationError(ValueError):
    """Raised when training data cannot satisfy the pipeline contract."""


def validate_dataset(df: pd.DataFrame, config: PreprocessConfig) -> None:
    """Validate data before preprocessing or model fitting."""
    if not isinstance(config, PreprocessConfig):
        raise DataValidationError("config must be a PreprocessConfig instance")
    if not isinstance(config.target_column, str) or not config.target_column.strip():
        raise DataValidationError("target_column must be a non-empty string")
    if not isinstance(config.split_strategy, str):
        raise DataValidationError("split_strategy must be a string")
    if config.timestamp_column is not None and not isinstance(
        config.timestamp_column, str
    ):
        raise DataValidationError("timestamp_column must be a string or None")
    if config.split_strategy != "random":
        raise DataValidationError(
            "Only the 'random' split strategy is supported; temporal splitting "
            "is reserved for a later phase"
        )
    if config.timestamp_column is not None:
        raise DataValidationError(
            "timestamp_column is not supported until temporal splitting is implemented"
        )
    for name, value in (
        ("validation_size", config.validation_size),
        ("test_size", config.test_size),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DataValidationError(f"{name} must be a finite proportion")
        if not math.isfinite(float(value)):
            raise DataValidationError(f"{name} must be a finite proportion")
        if value < 0 or (name == "test_size" and value == 0):
            raise DataValidationError(
                f"{name} must be positive, or validation_size may be zero for two-way splitting"
            )
    if config.validation_size + config.test_size >= 1:
        raise DataValidationError(
            "validation_size and test_size must sum to less than 1"
        )
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
    remaining_count = n_samples - test_count
    validation_fraction = config.validation_size / (1 - config.test_size)
    if config.validation_size == 0:
        train_count = remaining_count
        if min(train_count, test_count) < len(classes):
            raise DataValidationError(
                "Configured stratified split must leave at least one sample of every "
                "class in train and test partitions "
                f"(test_size={config.test_size})"
            )
        return

    validation_count = math.ceil(remaining_count * validation_fraction)
    train_count = remaining_count - validation_count
    if min(train_count, validation_count, test_count) < len(classes):
        raise DataValidationError(
            "Configured stratified split must leave at least one sample of every class "
            "in train, validation, and test partitions; each target class also needs "
            "at least three samples "
            f"(validation_size={config.validation_size}, test_size={config.test_size})"
        )
    if any(count < 3 for count in counts):
        raise DataValidationError(
            "Each target class must contain at least three samples for a three-way "
            "stratified split"
        )
