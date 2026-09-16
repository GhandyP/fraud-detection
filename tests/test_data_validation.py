import numpy as np
import pandas as pd
import pytest

from fraud_detection.data.preprocessor import FraudPreprocessor, PreprocessConfig
from fraud_detection.data.validation import DataValidationError, validate_dataset


def config() -> PreprocessConfig:
    return PreprocessConfig(target_column="fraud", test_size=0.25, random_state=1)


def test_validate_dataset_accepts_binary_numeric_data() -> None:
    df = pd.DataFrame(
        {
            "amount": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "fraud": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )

    validate_dataset(df, config())


@pytest.mark.parametrize(
    "df, message",
    [
        (pd.DataFrame(), "non-empty"),
        (pd.DataFrame([[1, 0], [2, 1]], columns=["x", "x"]), "unique"),
        (pd.DataFrame({"x": [1, 2], "other": [0, 1]}), "missing target"),
        (pd.DataFrame({"x": ["a", "b"], "fraud": [0, 1]}), "numeric"),
        (pd.DataFrame({"x": [1, np.nan], "fraud": [0, 1]}), "missing"),
        (pd.DataFrame({"x": [1, np.inf], "fraud": [0, 1]}), "non-finite"),
        (pd.DataFrame({"x": [1, 2, 3, 4], "fraud": [0, 0, 0, 0]}), "exactly two"),
        (pd.DataFrame({"x": [1, 2, 3], "fraud": [0, 0, 1]}), "at least two"),
    ],
)
def test_validate_dataset_rejects_invalid_data(df: pd.DataFrame, message: str) -> None:
    with pytest.raises(DataValidationError, match=message):
        validate_dataset(df, config())


def test_validate_dataset_accepts_legacy_two_way_minimum_counts() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, 4], "fraud": [0, 0, 1, 1]})

    validate_dataset(
        df, PreprocessConfig(target_column="fraud", test_size=0.5, random_state=1)
    )


def test_validate_dataset_rejects_split_without_each_class() -> None:
    df = pd.DataFrame({"x": range(4), "fraud": [0, 0, 1, 1]})
    invalid = PreprocessConfig(target_column="fraud", test_size=0.1, random_state=1)

    with pytest.raises(DataValidationError, match="stratified split"):
        validate_dataset(df, invalid)


def test_three_way_split_is_deterministic_disjoint_and_stratified() -> None:
    df = pd.DataFrame(
        {"amount": range(12), "fraud": [0] * 6 + [1] * 6},
        index=range(100, 112),
    )
    settings = PreprocessConfig(
        target_column="fraud", test_size=0.2, validation_size=0.2, random_state=7
    )

    first = FraudPreprocessor(settings).split_three_way(df)
    second = FraudPreprocessor(settings).split_three_way(df)

    for left, right in zip(first.__dataclass_fields__, second.__dataclass_fields__):
        assert getattr(first, left).equals(getattr(second, right))
    index_sets = [
        set(part.index) for part in (first.X_train, first.X_validation, first.X_test)
    ]
    assert not (
        index_sets[0] & index_sets[1]
        or index_sets[0] & index_sets[2]
        or index_sets[1] & index_sets[2]
    )
    assert all(
        set(labels) == {0, 1}
        for labels in (first.y_train, first.y_validation, first.y_test)
    )


@pytest.mark.parametrize(
    "settings, message",
    [
        (PreprocessConfig(split_strategy="temporal"), "split strategy"),
        (PreprocessConfig(timestamp_column="created_at"), "timestamp_column"),
        (PreprocessConfig(validation_size=0.8, test_size=0.2), "sum to less"),
    ],
)
def test_three_way_split_rejects_unsupported_or_invalid_contract(
    settings: PreprocessConfig, message: str
) -> None:
    df = pd.DataFrame({"x": range(12), "Class": [0] * 6 + [1] * 6})
    with pytest.raises(DataValidationError, match=message):
        FraudPreprocessor(settings).split_three_way(df)


def test_three_way_split_rejects_zero_validation_size() -> None:
    df = pd.DataFrame({"x": range(12), "Class": [0] * 6 + [1] * 6})

    with pytest.raises(DataValidationError, match="validation_size.*positive"):
        FraudPreprocessor(PreprocessConfig()).split_three_way(df)


def test_three_way_split_rejects_insufficient_class_counts() -> None:
    df = pd.DataFrame({"x": range(6), "Class": [0, 0, 0, 1, 1, 1]})
    settings = PreprocessConfig(validation_size=0.2)
    with pytest.raises(DataValidationError, match="three samples"):
        FraudPreprocessor(settings).split_three_way(df)


def test_validate_dataset_accepts_nullable_numeric_features() -> None:
    df = pd.DataFrame(
        {
            "amount": pd.Series([1, 2, 3, 4, 5, 6, 7, 8], dtype="Int64"),
            "fraud": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )

    validate_dataset(df, config())
