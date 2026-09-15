from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class PreprocessConfig:
    target_column: str = "Class"
    test_size: float = 0.2
    random_state: int = 42


class FraudPreprocessor:
    """Clean data and create train/test splits."""

    def __init__(self, config: PreprocessConfig) -> None:
        self.config: PreprocessConfig = config
        self.scaler: StandardScaler = StandardScaler()

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

    def scale(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
