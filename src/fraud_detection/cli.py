"""Command-line interface for the fraud detection pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from fraud_detection.data.loader import DataLoader, DatasetConfig
from fraud_detection.data.preprocessor import PreprocessConfig
from fraud_detection.data.validation import DataValidationError, validate_dataset
from fraud_detection.models.artifact import load_bundle
from fraud_detection.models.evaluate import evaluate_scores
from fraud_detection.models.predict import ModelPredictor
from fraud_detection.pipeline import run_training
from fraud_detection.utils.config import ConfigError, ProjectPaths, load_config


class _PositiveInt(argparse.Action):
    """Parse a positive integer while keeping argparse errors testable."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        try:
            value = int(values)
        except ValueError as exc:
            raise argparse.ArgumentError(self, "must be a positive integer") from exc
        if value <= 0:
            raise argparse.ArgumentError(self, "must be a positive integer")
        setattr(namespace, self.dest, value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fraud", exit_on_error=False)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate-data", prog="fraud validate-data", exit_on_error=False
    )
    validate.add_argument("--config", type=Path)
    validate.add_argument("--data", type=Path)
    validate.set_defaults(handler=_validate_data)

    train = subparsers.add_parser("train", prog="fraud train", exit_on_error=False)
    train.add_argument("--config", type=Path)
    train.add_argument("--sample-size", action=_PositiveInt)
    train.add_argument("--json", action="store_true")
    train.set_defaults(handler=_train)

    evaluate = subparsers.add_parser(
        "evaluate", prog="fraud evaluate", exit_on_error=False
    )
    evaluate.add_argument("--config", type=Path)
    evaluate.add_argument("--data", type=Path)
    evaluate.add_argument("--model", type=Path)
    evaluate.add_argument("--json", action="store_true")
    evaluate.set_defaults(handler=_evaluate)

    predict = subparsers.add_parser(
        "predict", prog="fraud predict", exit_on_error=False
    )
    predict.add_argument("--config", type=Path)
    predict.add_argument("--model", type=Path)
    prediction_input = predict.add_mutually_exclusive_group(required=True)
    prediction_input.add_argument("--record")
    prediction_input.add_argument("--input", type=Path)
    predict.add_argument("--json", action="store_true")
    predict.set_defaults(handler=_predict)
    return parser


def _preprocess_config(config: Any) -> PreprocessConfig:
    return PreprocessConfig(
        target_column=config.dataset.target_column,
        test_size=config.train.test_size,
        validation_size=config.train.validation_size,
        split_strategy=config.train.split_strategy,
        timestamp_column=config.train.timestamp_column,
        random_state=config.train.random_state,
    )


def _validate_data(args: argparse.Namespace, root: Path) -> int:
    config = load_config(root, args.config)
    data_path = (
        (args.data if args.data.is_absolute() else root / args.data)
        if args.data is not None
        else ProjectPaths(root).data_raw / config.dataset.file_name
    )
    loader = DataLoader(
        DatasetConfig(raw_dir=data_path.parent, file_name=data_path.name)
    )
    validate_dataset(loader.load(), _preprocess_config(config))
    print(f"Dataset is valid: {data_path.resolve()}")
    return 0


def _train(args: argparse.Namespace, root: Path) -> int:
    config = load_config(root, args.config)
    run_training(root, sample_size=args.sample_size, config_path=args.config)
    artifact_path = (
        ProjectPaths(root).models_trained / f"{config.train.model_name}.joblib"
    )
    bundle = load_bundle(artifact_path)
    metadata = bundle.metadata
    validation_metrics = metadata.get("validation_metrics", {})
    test_metrics = metadata.get("test_metrics", {})
    if args.json:
        print(
            json.dumps(
                {
                    "artifact_path": str(artifact_path.resolve()),
                    "validation_metrics": validation_metrics,
                    "test_metrics": test_metrics,
                }
            )
        )
    else:
        print(f"Artifact: {artifact_path.resolve()}")
        print(f"Validation metrics: {validation_metrics}")
        print(f"Test metrics: {test_metrics}")
    return 0


def _path_from_root(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _artifact_path(args: argparse.Namespace, root: Path, config: Any) -> Path:
    return _path_from_root(
        root,
        args.model
        if args.model is not None
        else ProjectPaths(root).models_trained / f"{config.train.model_name}.joblib",
    )


def _load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as exc:
        raise ValueError(f"Unable to read CSV dataset {path}: {exc}") from exc


def _model_version(predictor: ModelPredictor) -> str:
    metadata = predictor.metadata
    version = metadata.get("model_version")
    if isinstance(version, str):
        return version
    identity = metadata.get("model_identity")
    if isinstance(identity, dict) and isinstance(identity.get("version"), str):
        return identity["version"]
    return "unknown"


def _evaluate(args: argparse.Namespace, root: Path) -> int:
    config = load_config(root, args.config)
    predictor = ModelPredictor(_artifact_path(args, root, config))
    data_path = (
        _path_from_root(root, args.data)
        if args.data is not None
        else ProjectPaths(root).data_raw / config.dataset.file_name
    )
    data = _load_csv(data_path)
    target = config.dataset.target_column
    if target not in data.columns:
        raise ValueError(f"Missing target column: {target}")
    features = data.drop(columns=[target])
    scores = predictor.predict_proba(features)[:, 1]
    metrics = evaluate_scores(data[target].to_numpy(), scores, predictor.threshold)
    if args.json:
        print(
            json.dumps(
                {
                    **metrics,
                    "threshold": predictor.threshold,
                    "model_version": _model_version(predictor),
                }
            )
        )
    else:
        print(f"Model version: {_model_version(predictor)}")
        print(f"Threshold: {predictor.threshold}")
        for name, value in metrics.items():
            print(f"{name}: {value}")
    return 0


def _predict(args: argparse.Namespace, root: Path) -> int:
    config = load_config(root, args.config)
    predictor = ModelPredictor(_artifact_path(args, root, config))
    if args.record is not None:
        try:
            record = json.loads(args.record)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON record: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError("Record must be a JSON object")
        data = pd.DataFrame([record])
    else:
        data = _load_csv(_path_from_root(root, args.input))
        target = config.dataset.target_column
        if target in data.columns:
            data = data.drop(columns=[target])
    predictions = predictor.predict(data)
    probabilities = predictor.predict_proba(data)[:, 1]
    version = _model_version(predictor)
    results = [
        {
            "prediction": int(prediction),
            "is_fraud": bool(prediction),
            "fraud_probability": float(probability),
            "threshold": predictor.threshold,
            "model_version": version,
        }
        for prediction, probability in zip(predictions, probabilities, strict=True)
    ]
    if args.json:
        print(
            json.dumps(
                {
                    "predictions": results,
                    "threshold": predictor.threshold,
                    "model_version": version,
                }
            )
        )
    else:
        for result in results:
            print(
                f"Prediction: {result['prediction']} | Fraud probability: "
                f"{result['fraud_probability']:.6f} | Threshold: {result['threshold']} | "
                f"Model version: {result['model_version']}"
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return its stable process status code."""
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit) as exc:
        # argparse uses SystemExit for help and a few parser-level errors.
        if isinstance(exc, SystemExit):
            return exc.code if isinstance(exc.code, int) else 2
        return 2

    try:
        return args.handler(args, Path.cwd().resolve())
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except (DataValidationError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"Pipeline error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
