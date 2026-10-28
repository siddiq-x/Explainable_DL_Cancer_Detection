"""Binary classification metrics for imbalanced medical data.

Everything here operates on 1-D arrays of ground-truth labels (``0``/``1``) and
predicted probabilities. Accuracy alone is misleading on CBIS-DDSM's ~59/41
split, so the module surfaces the clinically relevant set: sensitivity,
specificity, F1, ROC-AUC and PR-AUC (average precision).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _as_1d(values: Any, dtype: Any = float) -> np.ndarray:
    """Accept a vector or single-column array, not multi-class scores."""
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1 and not (array.ndim == 2 and array.shape[1] == 1):
        raise ValueError("Expected a vector or an (N, 1) array.")
    return array.reshape(-1)


def validate_binary_inputs(y_true: Any, y_prob: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Validate labels before integer conversion and probabilities before scoring."""
    labels = _as_1d(y_true)
    probabilities = _as_1d(y_prob)
    if labels.size == 0 or labels.size != probabilities.size:
        raise ValueError("Labels and probabilities must be nonempty and equal in length.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must contain only 0 (benign) or 1 (malignant).")
    if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Probabilities must be finite values in [0, 1].")
    return labels.astype(int), probabilities


def _safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """ROC-AUC, or ``nan`` when it is undefined (only one class present)."""
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _safe_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """PR-AUC (average precision), or ``nan`` when undefined."""
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def compute_binary_metrics(
    y_true: Any,
    y_prob: Any,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute the full metric set for a binary classifier.

    Args:
        y_true: Ground-truth labels in ``{0, 1}``.
        y_prob: Predicted probability of the positive (malignant) class.
        threshold: Decision threshold used for the hard-label metrics.

    Returns:
        Dict with accuracy, precision, recall/sensitivity, specificity, f1,
        roc_auc, pr_auc, the confusion-matrix counts (tn/fp/fn/tp) and support.
    """
    y_true, y_prob = validate_binary_inputs(y_true, y_prob)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("The decision threshold must be finite and in [0, 1].")
    y_pred = (y_prob >= float(threshold)).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = recall_score(y_true, y_pred, zero_division=0) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    return {
        "threshold": float(threshold),
        "n_samples": int(y_true.size),
        "n_positive": int(y_true.sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(sensitivity),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _safe_roc_auc(y_true, y_prob),
        "pr_auc": _safe_pr_auc(y_true, y_prob),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def find_best_threshold(y_true: Any, y_prob: Any) -> Tuple[float, float]:
    """Find the threshold maximising Youden's J (sensitivity + specificity - 1).

    Fit this on validation data only; keep it fixed for held-out test reporting.

    Returns:
        ``(best_threshold, best_youden_j)``. Falls back to ``(0.5, nan)`` when
        the ROC curve is undefined.
    """
    y_true, y_prob = validate_binary_inputs(y_true, y_prob)
    if np.unique(y_true).size < 2:
        return 0.5, float("nan")

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    # sklearn's first threshold is infinity (older versions use max(score)+1).
    # Exclude that sentinel so even tied/reversed rankings return a usable value.
    valid = np.isfinite(thresholds) & (thresholds >= 0) & (thresholds <= 1)
    fpr, tpr, thresholds = fpr[valid], tpr[valid], thresholds[valid]
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    return float(thresholds[best_idx]), float(j_scores[best_idx])


def roc_points(y_true: Any, y_prob: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(fpr, tpr)`` for plotting the ROC curve."""
    y_true, y_prob = validate_binary_inputs(y_true, y_prob)
    if np.unique(y_true).size < 2:
        return np.array([]), np.array([])
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return fpr, tpr


def pr_points(y_true: Any, y_prob: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(recall, precision)`` for plotting the precision-recall curve."""
    y_true, y_prob = validate_binary_inputs(y_true, y_prob)
    if np.unique(y_true).size < 2:
        return np.array([]), np.array([])
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return recall, precision


def format_metrics(metrics: Dict[str, float]) -> str:
    """Render a metrics dict as a readable multi-line report."""
    lines = [
        f"  threshold   : {metrics['threshold']:.2f}",
        f"  n / positives: {metrics['n_samples']} / {metrics['n_positive']}",
        f"  accuracy    : {metrics['accuracy']:.4f}",
        f"  precision   : {metrics['precision']:.4f}",
        f"  recall (sensitivity): {metrics['sensitivity']:.4f}",
        f"  specificity : {metrics['specificity']:.4f}",
        f"  f1          : {metrics['f1']:.4f}",
        f"  ROC-AUC     : {metrics['roc_auc']:.4f}",
        f"  PR-AUC (AP) : {metrics['pr_auc']:.4f}",
        "  confusion matrix (rows=true, cols=pred):",
        f"    TN={metrics['tn']:<5d} FP={metrics['fp']:<5d}",
        f"    FN={metrics['fn']:<5d} TP={metrics['tp']:<5d}",
    ]
    return "\n".join(lines)
