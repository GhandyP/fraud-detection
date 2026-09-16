from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import pandas as pd


@dataclass
class DatasetConfig:
    raw_dir: Path
    file_name: str = "creditcard.csv"


class DataLoader:
    """Load transaction data from disk."""

    def __init__(self, config: DatasetConfig) -> None:
        self.config: DatasetConfig = config

    def load(self, sample_size: int | None = None) -> pd.DataFrame:
        file_path = self.config.raw_dir / self.config.file_name
        if not file_path.is_file():
            raise FileNotFoundError(f"Dataset must be a regular file: {file_path}")
        if isinstance(sample_size, bool) or (
            sample_size is not None
            and (not isinstance(sample_size, Integral) or sample_size <= 0)
        ):
            raise ValueError("sample_size must be None or a positive integer")

        try:
            df = pd.read_csv(file_path, nrows=sample_size)
        except (pd.errors.EmptyDataError, OSError, pd.errors.ParserError) as exc:
            raise ValueError(f"Unable to read CSV dataset {file_path}: {exc}") from exc
        if df.empty:
            raise ValueError(f"CSV dataset is empty: {file_path}")
        return df
