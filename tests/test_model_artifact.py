from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud_detection.models.artifact import ModelBundle, load_bundle
from fraud_detection.models.predict import ModelPredictor
from fraud_detection.models.train import ModelTrainer, TrainConfig
from fraud_detection.utils.hashing import sha256_file


def _train(
    tmp_path: Path, threshold: float = 0.5
) -> tuple[ModelTrainer, Path, pd.DataFrame, dict[str, Any]]:
    X = pd.DataFrame(
        {
            "amount": [1.0, 2.0, 3.0, 4.0, 8.0, 9.0, 10.0, 11.0],
            "age": [20, 21, 22, 23, 58, 59, 60, 61],
        }
    )
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    trainer = ModelTrainer(TrainConfig(tmp_path, threshold=threshold, max_iter=100))
    path, metrics = trainer.train_and_evaluate(X.iloc[:4], y[:4], X.iloc[4:], y[4:])
    return trainer, path, X, metrics


def test_pipeline_contains_scaler_and_classifier(tmp_path: Path) -> None:
    trainer, _, _, _ = _train(tmp_path)
    assert isinstance(trainer.model, Pipeline)
    assert isinstance(trainer.model.named_steps["scaler"], StandardScaler)
    assert isinstance(trainer.model.named_steps["classifier"], LogisticRegression)


def test_phase2b_save_rejects_partial_arguments_and_legacy_mix(tmp_path: Path) -> None:
    trainer, _, _, _ = _train(tmp_path)
    with pytest.raises(ValueError, match="Phase 2B save"):
        trainer.save(validation_metrics={})
    with pytest.raises(ValueError, match="Legacy metrics"):
        trainer.save({"f1": 0.5}, validation_metrics={}, test_metrics={})


def test_train_validate_test_fits_only_train_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = ModelTrainer(TrainConfig(tmp_path, max_iter=100, dataset_sha256="0" * 64))
    X_train = pd.DataFrame({"amount": [101.0, 102.0, 103.0, 104.0]})
    y_train = np.array([0, 1, 0, 1])
    X_validation = pd.DataFrame({"amount": [201.0, 202.0, 203.0, 204.0]})
    y_validation = np.array([0, 1, 0, 1])
    X_test = pd.DataFrame({"amount": [301.0, 302.0, 303.0, 304.0]})
    y_test = np.array([0, 1, 0, 1])
    fit_spy = Mock(wraps=trainer.model.fit)
    monkeypatch.setattr(trainer.model, "fit", fit_spy)
    monkeypatch.setattr(trainer, "save", Mock(return_value=tmp_path / "ignored.joblib"))

    trainer.train_validate_test(
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    )

    fit_spy.assert_called_once()
    fitted_features, fitted_labels = fit_spy.call_args.args[:2]
    pd.testing.assert_frame_equal(fitted_features, X_train)
    np.testing.assert_array_equal(fitted_labels, y_train)


def test_phase2b_save_derives_consistent_sample_size(tmp_path: Path) -> None:
    trainer = ModelTrainer(
        TrainConfig(tmp_path, sample_size=None, max_iter=100, dataset_sha256="0" * 64)
    )
    X = pd.DataFrame({"amount": list(range(1, 13))})
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    path, validation, test = trainer.train_validate_test(
        X.iloc[:6], y[:6], X.iloc[6:10], y[6:10], X.iloc[10:], y[10:]
    )
    metadata = load_bundle(path).metadata
    assert metadata["sample_size"] == 12
    assert metadata["validation_metrics"] == validation
    assert metadata["test_metrics"] == test


def _train_phase2b(tmp_path: Path) -> Path:
    trainer = ModelTrainer(TrainConfig(tmp_path, max_iter=100, dataset_sha256="0" * 64))
    X = pd.DataFrame({"amount": list(range(1, 13))})
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    return trainer.train_validate_test(
        X.iloc[:6], y[:6], X.iloc[6:10], y[6:10], X.iloc[10:], y[10:]
    )[0]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda metadata: metadata.pop("validation_metrics"),
            "missing required fields",
        ),
        (
            lambda metadata: metadata["threshold_selection"].update(
                selection_split="test"
            ),
            "must use validation",
        ),
        (
            lambda metadata: metadata["threshold_selection"].update(threshold=0.25),
            "disagrees with bundle",
        ),
        (
            lambda metadata: metadata["threshold_selection"].update(
                tie_break="lowest_threshold"
            ),
            "must use validation",
        ),
        (
            lambda metadata: metadata.update(dataset_sha256=None),
            "dataset_sha256 is invalid",
        ),
        (
            lambda metadata: metadata.update(dataset_sha256="z" * 64),
            "dataset_sha256 is invalid",
        ),
        (
            lambda metadata: metadata["runtime_packages"].update(numpy=1),
            "runtime metadata is invalid",
        ),
        (
            lambda metadata: metadata["runtime_versions"].update(python="other"),
            "runtime_versions is inconsistent",
        ),
        (
            lambda metadata: metadata["runtime_versions"]["packages"].update(
                numpy="different"
            ),
            "runtime_versions is inconsistent",
        ),
        (
            lambda metadata: metadata["validation_metrics"].update(
                confusion_matrix=[[1, 2, 3]]
            ),
            "2x2 count matrix",
        ),
        (
            lambda metadata: metadata.update(sample_size=11),
            "agree with sample_size",
        ),
        (
            lambda metadata: metadata.update(artifact_schema_version="9.9"),
            "Unsupported",
        ),
    ],
)
def test_load_rejects_malformed_phase2b_metadata(
    tmp_path: Path, mutate: Any, message: str
) -> None:
    path = _train_phase2b(tmp_path)
    bundle = joblib.load(path)
    metadata = deepcopy(bundle.metadata)
    mutate(metadata)
    bundle.metadata = metadata
    joblib.dump(bundle, path)
    with pytest.raises(ValueError, match=message):
        load_bundle(path)


def test_phase2b_predictor_uses_inclusive_persisted_threshold(tmp_path: Path) -> None:
    path = _train_phase2b(tmp_path)
    predictor = ModelPredictor(path)
    features = pd.DataFrame({"amount": [1.0, 6.0, 12.0]})
    scores = predictor.predict_proba(features)[:, 1]
    expected = (scores >= predictor.metadata["threshold"]).astype(int)
    np.testing.assert_array_equal(predictor.predict(features), expected)
    assert predictor.threshold == predictor.metadata["threshold"]


def test_save_without_metrics_is_supported(tmp_path: Path) -> None:
    trainer, _, _, _ = _train(tmp_path)
    path = trainer.save()

    assert ModelPredictor(path).metadata["evaluation_metrics"] == {}


def test_bundle_round_trip_and_predictor_uses_raw_pipeline(tmp_path: Path) -> None:
    trainer, path, X, _ = _train(tmp_path)
    predictor = ModelPredictor(path)
    np.testing.assert_allclose(
        predictor.predict_proba(X), trainer.model.predict_proba(X)
    )
    assert predictor.feature_names == ("amount", "age")
    assert predictor.metadata["artifact_schema_version"] == "1.0"
    assert predictor.threshold == 0.5


def test_metrics_threshold_and_metadata_are_persisted(tmp_path: Path) -> None:
    _, path, _, metrics = _train(tmp_path, threshold=0.8)
    predictor = ModelPredictor(path)
    assert predictor.threshold == 0.8
    assert predictor.metadata["feature_names"] == ["amount", "age"]
    assert predictor.metadata["training"]["random_state"] == 42
    assert predictor.metadata["evaluation_metrics"] == metrics


def test_config_metadata_cannot_override_reserved_fields(tmp_path: Path) -> None:
    trainer = ModelTrainer(
        TrainConfig(
            tmp_path,
            metadata={"threshold": 0.1, "evaluation_metrics": "wrong", "owner": "test"},
        )
    )
    X = pd.DataFrame(
        {
            "amount": [1.0, 2.0, 3.0, 4.0, 8.0, 9.0, 10.0, 11.0],
            "age": [20, 21, 22, 23, 58, 59, 60, 61],
        }
    )
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    path, _ = trainer.train_and_evaluate(X.iloc[:4], y[:4], X.iloc[4:], y[4:])
    metadata = ModelPredictor(path).metadata
    assert metadata["threshold"] == 0.5
    assert isinstance(metadata["evaluation_metrics"], dict)
    assert metadata["config_metadata"]["threshold"] == 0.1
    assert metadata["config_metadata"]["owner"] == "test"


def test_metadata_is_read_only_and_nested_copy_is_safe(tmp_path: Path) -> None:
    _, path, _, _ = _train(tmp_path)
    metadata = ModelPredictor(path).metadata
    with pytest.raises(TypeError):
        metadata["new"] = "value"  # type: ignore[index]
    metadata["training"]["random_state"] = 0  # type: ignore[index]
    assert ModelPredictor(path).metadata["training"]["random_state"] == 42


def test_feature_schema_and_malformed_artifact_errors(tmp_path: Path) -> None:
    _, path, X, _ = _train(tmp_path)
    predictor = ModelPredictor(path)
    with pytest.raises(ValueError, match="Feature columns"):
        predictor.predict(X[["age", "amount"]])
    with pytest.raises(ValueError, match="Expected 2 features"):
        predictor.predict(X.to_numpy()[:, :1])
    legacy = tmp_path / "legacy.joblib"
    joblib.dump(LogisticRegression(), legacy)
    with pytest.raises(ValueError, match="legacy"):
        ModelPredictor(legacy)
    with pytest.raises(FileNotFoundError):
        load_bundle(tmp_path / "missing.joblib")


def test_load_rejects_unfitted_and_malformed_bundles(tmp_path: Path) -> None:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(solver="liblinear")),
        ]
    )
    fitted_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(solver="liblinear")),
        ]
    )
    fitted_pipeline.fit([[1.0], [2.0], [8.0], [9.0]], [0, 0, 1, 1])
    valid_metadata = {"artifact_schema_version": "1.0"}
    cases: list[tuple[Any, tuple[str, ...], Any, Any, str]] = [
        (pipeline, ("amount",), 0.5, valid_metadata, "fitted"),
        (fitted_pipeline, ("amount", "amount"), 0.5, valid_metadata, "feature_names"),
        (fitted_pipeline, ("amount",), float("nan"), valid_metadata, "threshold"),
        (fitted_pipeline, ("amount",), True, valid_metadata, "threshold"),
        (fitted_pipeline, ("amount",), 0.5, [], "metadata"),
    ]
    for index, (model, names, threshold, metadata, message) in enumerate(cases):
        path = tmp_path / f"invalid-{index}.joblib"
        joblib.dump(ModelBundle.__new__(ModelBundle), path)
        value = ModelBundle.__new__(ModelBundle)
        value.pipeline = model
        value.feature_names = names
        value.threshold = threshold
        value.metadata = metadata
        joblib.dump(value, path)
        with pytest.raises(ValueError, match=message):
            load_bundle(path)


def test_file_sha256_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"phase-2a")
    first = sha256_file(path)
    assert first == sha256_file(path)
    assert len(first) == 64


@pytest.mark.parametrize(
    "threshold", [True, "0.5", float("nan"), float("inf"), -0.1, 1.1]
)
def test_trainer_rejects_invalid_threshold(tmp_path: Path, threshold: Any) -> None:
    with pytest.raises(ValueError, match="threshold"):
        ModelTrainer(TrainConfig(tmp_path, threshold=threshold))


def test_bundle_rejects_non_finite_metadata() -> None:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(solver="liblinear")),
        ]
    )
    pipeline.fit([[1.0], [2.0], [8.0], [9.0]], [0, 0, 1, 1])
    with pytest.raises(ValueError, match="finite"):
        ModelBundle(pipeline, ("amount",), 0.5, {"score": float("inf")})


def test_bundle_type_rejects_invalid_pipeline() -> None:
    with pytest.raises(ValueError, match="Pipeline"):
        ModelBundle("not-a-pipeline", ("x",), 0.5, {})  # type: ignore[arg-type]


def test_ndarray_round_trip_uses_artifact_feature_names(tmp_path: Path) -> None:
    trainer = ModelTrainer(TrainConfig(tmp_path, max_iter=100))
    X = np.array([[1.0], [2.0], [3.0], [4.0], [8.0], [9.0]])
    y = np.array([0, 0, 0, 1, 0, 1])
    path = trainer.train_and_evaluate(X[:4], y[:4], X[4:], y[4:], ["amount"])[0]

    predictor = ModelPredictor(path)
    assert predictor.predict_proba(X).shape == (6, 2)


@pytest.mark.parametrize("features", [np.array([1.0, 2.0]), np.empty((0, 1))])
def test_trainer_rejects_invalid_feature_shapes(
    tmp_path: Path, features: np.ndarray
) -> None:
    trainer = ModelTrainer(TrainConfig(tmp_path))
    with pytest.raises(ValueError, match="2-dimensional|non-empty"):
        trainer.train(features, np.array([0, 1]))


def test_trainer_rejects_wrong_length_and_nonbinary_targets(tmp_path: Path) -> None:
    trainer = ModelTrainer(TrainConfig(tmp_path))
    X = np.array([[1.0], [2.0], [8.0], [9.0]])
    with pytest.raises(ValueError, match="length"):
        trainer.train(X, np.array([0, 1]))
    with pytest.raises(ValueError, match="binary"):
        trainer.train(X, np.array([0, 1, 1, 2]))


def test_load_rejects_reserved_metadata_mismatch(tmp_path: Path) -> None:
    _, path, _, _ = _train(tmp_path)
    value = joblib.load(path)
    value.metadata["threshold"] = 0.1
    joblib.dump(value, path)
    with pytest.raises(ValueError, match="threshold disagrees"):
        load_bundle(path)
