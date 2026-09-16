from __future__ import annotations

from pathlib import Path

from fraud_detection.data.loader import DataLoader, DatasetConfig
from fraud_detection.data.preprocessor import FraudPreprocessor, PreprocessConfig
from fraud_detection.data.validation import validate_dataset
from fraud_detection.models.train import ModelTrainer, TrainConfig
from fraud_detection.utils.config import ProjectPaths, load_config
from fraud_detection.utils.hashing import sha256_file
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
        validation_size=app_config.train.validation_size,
        split_strategy=app_config.train.split_strategy,
        random_state=app_config.train.random_state,
    )
    validate_dataset(df, preprocessor_config)
    preprocessor = FraudPreprocessor(preprocessor_config)
    splits = preprocessor.split_three_way(df)
    X_train, X_validation, X_test = (
        splits.X_train,
        splits.X_validation,
        splits.X_test,
    )
    y_train, y_validation, y_test = (
        splits.y_train,
        splits.y_validation,
        splits.y_test,
    )
    feature_names = tuple(str(name) for name in X_train.columns)
    dataset_path = paths.data_raw / app_config.dataset.file_name
    dataset_sha256 = sha256_file(dataset_path) if dataset_path.is_file() else None

    trainer = ModelTrainer(
        TrainConfig(
            model_dir=paths.models_trained,
            model_name=app_config.train.model_name,
            model_version=app_config.train.model_version,
            threshold=app_config.train.threshold,
            random_state=app_config.train.random_state,
            dataset_sha256=dataset_sha256,
            sample_size=sample_size if sample_size is not None else len(df),
        )
    )
    model_path, validation_metrics, test_metrics = trainer.train_validate_test(
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        feature_names,
        split_strategy=app_config.train.split_strategy,
    )

    logger.info("Model saved to %s", model_path)
    logger.info("Validation metrics: %s", validation_metrics)
    logger.info("Test metrics: %s", test_metrics)


if __name__ == "__main__":
    run_training(root=Path("."))
