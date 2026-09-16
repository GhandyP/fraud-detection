from __future__ import annotations

from typing import TypeAlias

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

MetricValue: TypeAlias = float | list[list[int]]
Metrics: TypeAlias = dict[str, MetricValue]


def _binary_input(values: np.ndarray, name: str) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if values.size == 0:
        raise ValueError(f"{name} must not be empty")
    if np.issubdtype(values.dtype, np.bool_):
        raise ValueError(f"{name} must contain binary 0/1 values")
    try:
        numeric = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain binary 0/1 values") from exc
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0.0, 1.0)).all():
        raise ValueError(f"{name} must contain binary 0/1 values")
    return numeric.astype(int)


def _score_input(y_scores: np.ndarray) -> np.ndarray:
    if y_scores.ndim != 1:
        raise ValueError("y_scores must be one-dimensional")
    if y_scores.size == 0:
        raise ValueError("y_scores must not be empty")
    try:
        scores = np.asarray(y_scores, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("y_scores must contain finite numeric values") from exc
    if not np.isfinite(scores).all():
        raise ValueError("y_scores must contain finite numeric values")
    return scores


def _validate_threshold(threshold: float) -> float:
    if isinstance(threshold, bool):
        raise ValueError(  # noqa: TRY004
            "threshold must be a finite number between 0 and 1"
        )
    try:
        value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("threshold must be a finite number between 0 and 1") from exc
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("threshold must be a finite number between 0 and 1")
    return value


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray
) -> Metrics:
    """Evaluate binary predictions and ranking scores with fraud-oriented metrics."""
    true = _binary_input(np.asarray(y_true), "y_true")
    predicted = _binary_input(np.asarray(y_pred), "y_pred")
    scores = _score_input(np.asarray(y_scores))
    if not len(true) == len(predicted) == len(scores):
        raise ValueError("y_true, y_pred, and y_scores must have equal lengths")
    if np.unique(true).size < 2:
        raise ValueError("y_true must contain both binary classes for ranking metrics")

    matrix = confusion_matrix(true, predicted, labels=[0, 1])
    return {
        "pr_auc": float(average_precision_score(true, scores)),
        "roc_auc": float(roc_auc_score(true, scores)),
        "recall": float(recall_score(true, predicted, zero_division=0)),
        "precision": float(precision_score(true, predicted, zero_division=0)),
        "f1": float(f1_score(true, predicted, zero_division=0)),
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def select_threshold(y_validation: np.ndarray, validation_scores: np.ndarray) -> float:
    """Select the highest threshold that maximizes validation F1.

    Candidates are zero, one, and each unique validation probability. Predictions
    use inclusive ``score >= threshold`` semantics; the highest threshold wins
    when multiple candidates have the same F1 score.
    """
    labels = _binary_input(np.asarray(y_validation), "y_validation")
    raw_scores = np.asarray(validation_scores)
    if np.issubdtype(raw_scores.dtype, np.bool_):
        raise ValueError(
            "validation_scores must be finite probability values in [0, 1]"
        )
    scores = _score_input(raw_scores)
    if (
        not np.isfinite(scores).all()
        or not np.logical_and(scores >= 0, scores <= 1).all()
    ):
        raise ValueError(
            "validation_scores must be finite probability values in [0, 1]"
        )
    if len(labels) != len(scores):
        raise ValueError("y_validation and validation_scores must have equal lengths")

    candidates = np.unique(np.concatenate((np.array([0.0, 1.0]), scores)))
    best_threshold = 0.0
    best_f1 = -1.0
    for threshold in candidates:
        predicted = (scores >= threshold).astype(int)
        current_f1 = float(f1_score(labels, predicted, zero_division=0))
        if current_f1 > best_f1 or (
            current_f1 == best_f1 and threshold > best_threshold
        ):
            best_f1 = current_f1
            best_threshold = float(threshold)
    return best_threshold


def evaluate_scores(
    y_true: np.ndarray, y_scores: np.ndarray, threshold: float = 0.5
) -> Metrics:
    """Threshold scores and evaluate them using the centralized metric contract."""
    value = _validate_threshold(threshold)
    scores = _score_input(np.asarray(y_scores))
    predicted = (scores >= value).astype(int)
    return evaluate_predictions(y_true, predicted, scores)
