import warnings

import numpy as np
import pytest

from fraud_detection.models.evaluate import evaluate_predictions, evaluate_scores
from fraud_detection.models.train import ModelTrainer, TrainConfig


def test_evaluate_predictions_returns_fraud_metrics() -> None:
    metrics = evaluate_predictions(
        np.array([0, 0, 1, 1]),
        np.array([0, 0, 0, 1]),
        np.array([0.1, 0.4, 0.35, 0.8]),
    )

    assert metrics["pr_auc"] == pytest.approx(5 / 6)
    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["confusion_matrix"] == [[2, 0], [1, 1]]


def test_evaluate_scores_uses_inclusive_threshold_boundary() -> None:
    metrics = evaluate_scores(
        np.array([0, 1, 0, 1]), np.array([0.2, 0.5, 0.5, 0.9]), threshold=0.5
    )

    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]


def test_zero_positive_predictions_are_deterministic_without_warnings() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        metrics = evaluate_scores(
            np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.3, 0.4]), threshold=0.9
        )

    assert not caught
    assert metrics["recall"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["confusion_matrix"] == [[2, 0], [2, 0]]


@pytest.mark.parametrize(
    ("y_true", "y_pred", "y_scores", "message"),
    [
        ([0, 1], [[0], [1]], [0.1, 0.9], "one-dimensional"),
        ([0, 1], [0], [0.1, 0.9], "equal lengths"),
        ([0, 2], [0, 1], [0.1, 0.9], "binary"),
        ([0, 1], [0, 2], [0.1, 0.9], "binary"),
        ([0, 1], [0, 1], [0.1, np.inf], "finite"),
    ],
)
def test_invalid_metric_inputs_raise_clear_errors(
    y_true: object, y_pred: object, y_scores: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_predictions(
            np.asarray(y_true), np.asarray(y_pred), np.asarray(y_scores)
        )


@pytest.mark.parametrize("threshold", [True, -0.1, 1.1, np.nan, np.inf, "bad"])
def test_invalid_threshold_raises(threshold: object) -> None:
    with pytest.raises(ValueError, match="threshold"):
        evaluate_scores(np.array([0, 1]), np.array([0.1, 0.9]), threshold=threshold)  # type: ignore[arg-type]


def test_ranking_metrics_require_both_target_classes() -> None:
    with pytest.raises(ValueError, match="both binary classes"):
        evaluate_scores(np.array([0, 0]), np.array([0.1, 0.2]))


def test_model_trainer_uses_centralized_metric_contract(tmp_path) -> None:
    trainer = ModelTrainer(TrainConfig(tmp_path, max_iter=100))
    X = np.array([[1.0], [2.0], [8.0], [9.0]])
    y = np.array([0, 0, 1, 1])
    trainer.train(X, y)

    metrics = trainer.evaluate(X, y)
    assert set(metrics) == {
        "pr_auc",
        "roc_auc",
        "recall",
        "precision",
        "f1",
        "confusion_matrix",
    }
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]
