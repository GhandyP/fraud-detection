from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray
) -> dict[str, float]:
    return {
        "recall": recall_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "pr_auc": average_precision_score(y_true, y_scores),
    }
