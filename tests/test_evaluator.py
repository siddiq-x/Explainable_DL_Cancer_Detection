"""Unit tests for the evaluator: checkpoint round-trip, inference, artifacts."""

import os
import tempfile

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.evaluator import (
    collect_predictions,
    evaluate_loader,
    load_model_from_checkpoint,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    save_evaluation_artifacts,
)
from src.models.classifier import MammographyClassifier


@pytest.fixture
def tiny_loader():
    torch.manual_seed(0)
    images = torch.randn(8, 3, 64, 64)
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.float32)
    return DataLoader(TensorDataset(images, labels), batch_size=4)


def test_checkpoint_round_trip_and_inference(tiny_loader):
    model = MammographyClassifier(backbone="resnet18", pretrained=False)
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "best.pt")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "backbone": "resnet18",
                "num_classes": 2,
                "config": {"model": {"backbone": "resnet18", "num_classes": 2}},
                "epoch": 3,
            },
            path,
        )

        loaded, checkpoint = load_model_from_checkpoint(path)
        assert checkpoint["epoch"] == 3
        assert not loaded.training  # eval mode

        y_true, y_prob, meta = collect_predictions(loaded, tiny_loader, return_meta=True)

    assert y_true.shape == (8,)
    assert y_prob.shape == (8,)
    assert np.all((y_prob >= 0) & (y_prob <= 1))
    assert set(np.unique(y_true)).issubset({0.0, 1.0})
    assert meta["image_path"] == []  # tuple batches carry no paths


def test_load_missing_checkpoint_raises():
    with pytest.raises(FileNotFoundError):
        load_model_from_checkpoint("does/not/exist.pt")


def test_evaluate_loader_returns_metrics(tiny_loader):
    model = MammographyClassifier(backbone="resnet18", pretrained=False)
    metrics, y_true, y_prob = evaluate_loader(model, tiny_loader, threshold=0.5)

    assert y_true.shape == y_prob.shape == (8,)
    for key in ("accuracy", "roc_auc", "pr_auc", "sensitivity", "specificity", "f1"):
        assert key in metrics


def test_save_evaluation_artifacts_writes_all_files(tiny_loader):
    torch.manual_seed(1)
    model = MammographyClassifier(backbone="resnet18", pretrained=False)
    metrics, y_true, y_prob = evaluate_loader(model, tiny_loader, threshold=0.5)

    with tempfile.TemporaryDirectory() as tmp_dir:
        written = save_evaluation_artifacts(metrics, y_true, y_prob, tmp_dir)

        assert set(written) == {
            "metrics", "report", "roc_curve", "pr_curve", "confusion_matrix",
        }
        for path in written.values():
            assert os.path.exists(path) and os.path.getsize(path) > 0
        assert os.path.exists(os.path.join(tmp_dir, "metrics.json"))


def test_individual_plot_helpers(tiny_loader):
    model = MammographyClassifier(backbone="resnet18", pretrained=False)
    metrics, y_true, y_prob = evaluate_loader(model, tiny_loader)

    with tempfile.TemporaryDirectory() as tmp_dir:
        for name, fn, arg in (
            ("roc.png", plot_roc_curve, (y_true, y_prob)),
            ("pr.png", plot_precision_recall_curve, (y_true, y_prob)),
        ):
            out = fn(*arg, os.path.join(tmp_dir, name))
            assert os.path.exists(out)

        cm = plot_confusion_matrix(metrics, os.path.join(tmp_dir, "cm.png"))
        assert os.path.exists(cm)
