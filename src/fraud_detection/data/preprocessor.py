from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class PreprocessConfig:
    target_column: str = "Class"
    test_size: float = 0.2
    random_state: int = 42
    validation_size: float = 0.0
    split_strategy: str = "random"
    timestamp_column: str | None = None


@dataclass(frozen=True)
class DataSplits:
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


class FraudPreprocessor:
    """Clean data and create train/test splits."""

    def __init__(self, config: PreprocessConfig) -> None:
        self.config: PreprocessConfig = config

    def split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        if self.config.target_column not in df.columns:
            raise ValueError(f"Missing target column: {self.config.target_column}")

        X = df.drop(columns=[self.config.target_column])
        y = df[self.config.target_column]

        return train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

    def split_three_way(self, df: pd.DataFrame) -> DataSplits:
        """Create deterministic, stratified, disjoint train/validation/test data."""
        from fraud_detection.data.validation import (
            DataValidationError,
            validate_dataset,
        )

        if self.config.split_strategy != "random":
            raise DataValidationError(
                "Only the 'random' split strategy is supported; temporal splitting "
                "is reserved for a later phase"
            )
        if self.config.timestamp_column is not None:
            raise DataValidationError(
                "timestamp_column is not supported by random splitting; "
                "temporal splitting is reserved for a later phase"
            )
        if (
            isinstance(self.config.validation_size, bool)
            or not isinstance(self.config.validation_size, (int, float))
            or not math.isfinite(float(self.config.validation_size))
            or self.config.validation_size <= 0
        ):
            raise DataValidationError(
                "Three-way splitting requires validation_size to be a finite "
                "positive proportion; configure validation_size > 0"
            )
        validate_dataset(df, self.config)
        if not df.index.is_unique:
            raise DataValidationError(
                "Three-way splitting requires unique row indices to prove disjointness"
            )

        X = df.drop(columns=[self.config.target_column])
        y = df[self.config.target_column]
        test_size = self.config.test_size
        validation_size = self.config.validation_size
        remaining_size = 1.0 - test_size
        try:
            X_remaining, X_test, y_remaining, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=self.config.random_state,
                stratify=y,
            )
            X_train, X_validation, y_train, y_validation = train_test_split(
                X_remaining,
                y_remaining,
                test_size=validation_size / remaining_size,
                random_state=self.config.random_state,
                stratify=y_remaining,
            )
        except ValueError as exc:
            raise DataValidationError(
                "Configured three-way stratified split is not feasible; "
                "increase the dataset size or adjust validation_size/test_size"
            ) from exc
        result = DataSplits(
            X_train=X_train,
            X_validation=X_validation,
            X_test=X_test,
            y_train=y_train,
            y_validation=y_validation,
            y_test=y_test,
        )
        partitions = (result.X_train, result.X_validation, result.X_test)
        if (
            set(partitions[0].index) & set(partitions[1].index)
            or set(partitions[0].index) & set(partitions[2].index)
            or set(partitions[1].index) & set(partitions[2].index)
        ):
            raise DataValidationError("Three-way split partitions must be disjoint")
        for name, labels in (
            ("train", result.y_train),
            ("validation", result.y_validation),
            ("test", result.y_test),
        ):
            if set(labels.unique()) != {0, 1}:
                raise DataValidationError(
                    f"Each split must contain both target classes; {name} does not"
                )
        return result
