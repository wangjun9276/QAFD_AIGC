"""Binary classification metrics for IQAG evaluation."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def binary_metrics(labels: np.ndarray, fake_probabilities: np.ndarray) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(fake_probabilities, dtype=np.float64)
    if labels.shape != scores.shape:
        raise ValueError("labels and fake_probabilities must have identical shapes.")
    if labels.size == 0:
        raise ValueError("Cannot compute metrics for an empty prediction set.")

    predictions = (scores >= 0.5).astype(np.int64)
    metrics: Dict[str, float] = {
        "acc": float((predictions == labels).mean()),
        "real_acc": float((predictions[labels == 0] == 0).mean()) if np.any(labels == 0) else float("nan"),
        "fake_acc": float((predictions[labels == 1] == 1).mean()) if np.any(labels == 1) else float("nan"),
    }
    if np.unique(labels).size == 2:
        metrics["auc"] = float(roc_auc_score(labels, scores))
        metrics["ap"] = float(average_precision_score(labels, scores))
    else:
        metrics["auc"] = float("nan")
        metrics["ap"] = float("nan")
    return metrics
