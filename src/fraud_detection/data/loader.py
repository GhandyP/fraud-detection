from __future__ import annotations

from dataclasses import dataclass
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
        if not file_path.exists():
            raise FileNotFoundError(f"Missing dataset at {file_path}")

        if sample_size:
            df = pd.read_csv(file_path, nrows=sample_size)
        else:
            df = pd.read_csv(file_path)

        return df
