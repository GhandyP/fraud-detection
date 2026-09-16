import numpy as np
import pandas as pd
import pytest

from fraud_detection.data.preprocessor import PreprocessConfig
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


def test_validate_dataset_rejects_split_without_each_class(tmp_path) -> None:
    df = pd.DataFrame({"x": range(4), "fraud": [0, 0, 1, 1]})
    invalid = PreprocessConfig(target_column="fraud", test_size=0.1, random_state=1)

    with pytest.raises(DataValidationError, match="stratified split"):
        validate_dataset(df, invalid)


def test_validate_dataset_accepts_nullable_numeric_features() -> None:
    df = pd.DataFrame(
        {
            "amount": pd.Series([1, 2, 3, 4, 5, 6, 7, 8], dtype="Int64"),
            "fraud": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )

    validate_dataset(df, config())
