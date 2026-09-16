import numpy as np
import pandas as pd
import pytest

from fraud_detection.data.preprocessor import FraudPreprocessor, PreprocessConfig
from fraud_detection.data.validation import DataValidationError, validate_dataset


def temporal_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            "amount": np.arange(12, dtype=float),
            "fraud": [0, 1] * 6,
        }
    )


def test_temporal_split_is_ordered_deterministic_and_excludes_timestamp() -> None:
    frame = temporal_frame()
    config = PreprocessConfig(
        target_column="fraud",
        test_size=0.25,
        validation_size=0.25,
        split_strategy="temporal",
        timestamp_column="timestamp",
    )
    first = FraudPreprocessor(config).split_three_way(frame)
    second = FraudPreprocessor(config).split_three_way(frame)

    assert list(first.X_train["amount"]) == [0, 1, 2, 3, 4, 5]
    assert list(first.X_validation["amount"]) == [6, 7, 8]
    assert list(first.X_test["amount"]) == [9, 10, 11]
    assert "timestamp" not in first.X_train.columns
    assert all(
        left.equals(right)
        for left, right in zip(first.__dict__.values(), second.__dict__.values())
    )


def test_temporal_duplicate_timestamps_keep_original_order() -> None:
    frame = temporal_frame()
    frame["timestamp"] = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    config = PreprocessConfig(
        target_column="fraud",
        test_size=0.25,
        validation_size=0.25,
        split_strategy="temporal",
        timestamp_column="timestamp",
    )
    result = FraudPreprocessor(config).split_three_way(frame)
    assert list(result.X_train["amount"]) == [0, 1, 2, 3, 4, 5]
    assert list(result.X_validation["amount"]) == [6, 7, 8]
    assert list(result.X_test["amount"]) == [9, 10, 11]


@pytest.mark.parametrize(
    "values",
    [[0, 1, np.nan, 3], [0, 1, np.inf, 3], ["a", "b", "c", "d"]],
)
def test_temporal_timestamp_values_must_be_finite_ordering_keys(values: list) -> None:
    frame = pd.DataFrame(
        {"timestamp": values, "amount": range(4), "fraud": [0, 0, 1, 1]}
    )
    config = PreprocessConfig(
        target_column="fraud",
        split_strategy="temporal",
        timestamp_column="timestamp",
        validation_size=0.2,
    )
    with pytest.raises(DataValidationError, match="timestamp"):
        validate_dataset(frame, config)


def test_temporal_requires_timestamp_and_partition_classes() -> None:
    frame = temporal_frame()
    with pytest.raises(DataValidationError, match="timestamp_column"):
        FraudPreprocessor(PreprocessConfig(split_strategy="temporal")).split_three_way(
            frame
        )

    frame["fraud"] = [0] * 6 + [1] * 6
    config = PreprocessConfig(
        target_column="fraud",
        split_strategy="temporal",
        timestamp_column="timestamp",
        validation_size=0.2,
    )
    with pytest.raises(
        DataValidationError, match="temporal split.*both target classes"
    ):
        FraudPreprocessor(config).split_three_way(frame)


def test_random_rejects_timestamp_column() -> None:
    config = PreprocessConfig(timestamp_column="timestamp")
    with pytest.raises(DataValidationError, match="random"):
        validate_dataset(temporal_frame(), config)
