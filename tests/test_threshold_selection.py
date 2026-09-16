from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from fraud_detection.models.evaluate import select_threshold
from fraud_detection.models.train import ModelTrainer, TrainConfig


def test_select_threshold_maximizes_f1() -> None:
    threshold = select_threshold(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])
    )

    assert threshold == pytest.approx(0.35)


def test_select_threshold_breaks_ties_with_highest_threshold() -> None:
    threshold = select_threshold(np.array([0, 1]), np.array([0.2, 0.2]))

    assert threshold == pytest.approx(0.2)


def test_select_threshold_includes_endpoints_and_score_equality() -> None:
    assert select_threshold(np.array([1]), np.array([0.0])) == 0.0
    assert select_threshold(np.array([1]), np.array([1.0])) == 1.0


@pytest.mark.parametrize(
    ("labels", "scores", "message"),
    [
        ([[0], [1]], [0.1, 0.9], "one-dimensional"),
        ([0, 1], [0.1], "equal lengths"),
        ([0, 2], [0.1, 0.9], "binary"),
        ([0, 1], [-0.1, 0.9], "probability"),
        ([0, 1], [0.1, np.inf], "finite"),
        ([0, 1], [True, False], "probability"),
    ],
)
def test_select_threshold_rejects_invalid_inputs(
    labels: object, scores: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        select_threshold(np.asarray(labels), np.asarray(scores))


def test_trainer_select_threshold_updates_config_and_saved_metadata(
    tmp_path: Path,
) -> None:
    X = pd.DataFrame({"value": [1.0, 2.0, 3.0, 8.0, 9.0, 10.0]})
    y = np.array([0, 0, 0, 1, 1, 1])
    trainer = ModelTrainer(TrainConfig(tmp_path, max_iter=100))
    trainer.train(X, y)

    threshold = trainer.select_threshold(X.iloc[:4], y[:4])
    path = trainer.save()

    assert trainer.config.threshold == threshold
    metadata = joblib.load(path).metadata
    assert metadata["threshold"] == threshold
    assert metadata["threshold_selection"] == {
        "method": "max_f1",
        "selection_split": "validation",
        "tie_break": "highest_threshold",
        "threshold": threshold,
    }


def test_trainer_select_threshold_requires_training(tmp_path: Path) -> None:
    trainer = ModelTrainer(TrainConfig(tmp_path))

    with pytest.raises(ValueError, match="trained"):
        trainer.select_threshold(np.array([[1.0]]), np.array([0]))
