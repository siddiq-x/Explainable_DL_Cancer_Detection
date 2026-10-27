"""Unit tests for the binary metrics module (known inputs -> known outputs)."""

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_binary_metrics,
    find_best_threshold,
    format_metrics,
    pr_points,
    roc_points,
)


def test_perfect_separation():
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9]

    m = compute_binary_metrics(y_true, y_prob, threshold=0.5)

    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["sensitivity"] == 1.0
    assert m["specificity"] == 1.0
    assert m["f1"] == 1.0
    assert m["roc_auc"] == 1.0
    assert m["pr_auc"] == 1.0
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (2, 0, 0, 2)


def test_known_confusion_matrix_and_metrics():
    y_true = [1, 1, 1, 0, 0, 0]
    y_prob = [0.9, 0.8, 0.3, 0.7, 0.2, 0.1]  # preds: 1,1,0,1,0,0

    m = compute_binary_metrics(y_true, y_prob, threshold=0.5)

    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (2, 1, 1, 2)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["sensitivity"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)


def test_threshold_changes_hard_labels_but_not_auc():
    y_true = [0, 0, 1, 1]
    y_prob = [0.2, 0.6, 0.4, 0.7]

    low = compute_binary_metrics(y_true, y_prob, threshold=0.3)
    high = compute_binary_metrics(y_true, y_prob, threshold=0.8)

    # AUC is threshold-independent.
    assert low["roc_auc"] == high["roc_auc"]
    # A higher threshold predicts fewer positives.
    assert (high["tp"] + high["fp"]) <= (low["tp"] + low["fp"])


def test_all_wrong_gives_zero_auc():
    y_true = [0, 0, 1, 1]
    y_prob = [0.9, 0.8, 0.2, 0.1]

    m = compute_binary_metrics(y_true, y_prob)

    assert m["accuracy"] == 0.0
    assert m["roc_auc"] == 0.0
    assert m["sensitivity"] == 0.0
    assert m["specificity"] == 0.0


def test_single_class_auc_is_nan_but_metrics_still_compute():
    m = compute_binary_metrics([1, 1, 1], [0.5, 0.6, 0.9])

    assert np.isnan(m["roc_auc"])
    assert np.isnan(m["pr_auc"])
    assert m["sensitivity"] == 1.0  # all predicted positive at 0.5
    assert np.isnan(m["specificity"])  # no negatives -> undefined


def test_find_best_threshold_maximises_youden_j():
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9]

    threshold, j = find_best_threshold(y_true, y_prob)

    assert 0.2 <= threshold <= 0.8
    assert j == pytest.approx(1.0)


def test_curve_points_shapes():
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9]

    fpr, tpr = roc_points(y_true, y_prob)
    recall, precision = pr_points(y_true, y_prob)

    assert fpr.shape == tpr.shape
    assert recall.shape == precision.shape


def test_format_metrics_contains_key_fields():
    m = compute_binary_metrics([0, 1], [0.1, 0.9])
    report = format_metrics(m)

    for token in ("ROC-AUC", "PR-AUC", "sensitivity", "specificity", "f1"):
        assert token in report
