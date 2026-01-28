from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score


@dataclass
class TrainConfig:
    model_dir: Path
    model_name: str = "logistic_regression"
    max_iter: int = 1000
    random_state: int = 42


class ModelTrainer:
    """Train a baseline model for fraud detection."""

    def __init__(self, config: TrainConfig) -> None:
        self.config: TrainConfig = config
        self.model: LogisticRegression = LogisticRegression(
            class_weight="balanced",
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
        )

    def train(self, X_train: np.ndarray, y_train: np.ndarray | pd.Series) -> None:
        self.model.fit(X_train, y_train)

    def evaluate(
        self, X_test: np.ndarray, y_test: np.ndarray | pd.Series
    ) -> dict[str, float]:
        y_pred = self.model.predict(X_test)
        y_scores = self.model.predict_proba(X_test)[:, 1]
        return {
            "recall": recall_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "pr_auc": average_precision_score(y_test, y_scores),
        }

    def save(self) -> Path:
        self.config.model_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.config.model_dir / f"{self.config.model_name}.joblib"
        joblib.dump(self.model, model_path)
        return model_path

    def train_and_evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray | pd.Series,
        X_test: np.ndarray,
        y_test: np.ndarray | pd.Series,
    ) -> tuple[Path, dict[str, float]]:
        self.train(X_train, y_train)
        metrics = self.evaluate(X_test, y_test)
        model_path = self.save()
        return model_path, metrics
