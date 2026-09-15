from __future__ import annotations

from pathlib import Path

import numpy as np

from fraud_detection.data.loader import DataLoader, DatasetConfig
from fraud_detection.data.preprocessor import FraudPreprocessor, PreprocessConfig
from fraud_detection.models.train import ModelTrainer, TrainConfig
from fraud_detection.utils.config import ProjectPaths
from fraud_detection.utils.logger import get_logger


def run_training(root: Path, sample_size: int | None = None) -> None:
    logger = get_logger(__name__)
    paths = ProjectPaths(root=root)

    dataset_config = DatasetConfig(raw_dir=paths.data_raw)
    loader = DataLoader(dataset_config)
    df = loader.load(sample_size=sample_size)

    preprocessor = FraudPreprocessor(PreprocessConfig())
    X_train, X_test, y_train, y_test = preprocessor.split(df)
    X_train_scaled, X_test_scaled = preprocessor.scale(X_train, X_test)

    trainer = ModelTrainer(TrainConfig(model_dir=paths.models_trained))
    y_train_np: np.ndarray = y_train.to_numpy()
    y_test_np: np.ndarray = y_test.to_numpy()

    model_path, metrics = trainer.train_and_evaluate(
        X_train_scaled, y_train_np, X_test_scaled, y_test_np
    )

    logger.info("Model saved to %s", model_path)
    logger.info("Metrics: %s", metrics)


if __name__ == "__main__":
    run_training(root=Path("."))
