from __future__ import annotations

from pathlib import Path

import numpy as np

from fraud_detection.data.loader import DataLoader, DatasetConfig
from fraud_detection.data.preprocessor import FraudPreprocessor, PreprocessConfig
from fraud_detection.data.validation import validate_dataset
from fraud_detection.models.train import ModelTrainer, TrainConfig
from fraud_detection.utils.config import ProjectPaths, load_config
from fraud_detection.utils.logger import get_logger


def run_training(
    root: Path, sample_size: int | None = None, config_path: str | Path | None = None
) -> None:
    logger = get_logger(__name__)
    paths = ProjectPaths(root=root)
    app_config = load_config(paths.root, config_path=config_path)

    dataset_config = DatasetConfig(
        raw_dir=paths.data_raw, file_name=app_config.dataset.file_name
    )
    loader = DataLoader(dataset_config)
    df = loader.load(sample_size=sample_size)

    preprocessor_config = PreprocessConfig(
        target_column=app_config.dataset.target_column,
        test_size=app_config.train.test_size,
        random_state=app_config.train.random_state,
    )
    validate_dataset(df, preprocessor_config)
    preprocessor = FraudPreprocessor(preprocessor_config)
    X_train, X_test, y_train, y_test = preprocessor.split(df)
    X_train_scaled, X_test_scaled = preprocessor.scale(X_train, X_test)

    trainer = ModelTrainer(
        TrainConfig(
            model_dir=paths.models_trained,
            random_state=app_config.train.random_state,
        )
    )
    y_train_np: np.ndarray = y_train.to_numpy()
    y_test_np: np.ndarray = y_test.to_numpy()

    model_path, metrics = trainer.train_and_evaluate(
        X_train_scaled, y_train_np, X_test_scaled, y_test_np
    )

    logger.info("Model saved to %s", model_path)
    logger.info("Metrics: %s", metrics)


if __name__ == "__main__":
    run_training(root=Path("."))
