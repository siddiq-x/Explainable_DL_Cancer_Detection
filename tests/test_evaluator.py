"""Unit tests for the evaluator: checkpoint round-trip, inference, artifacts."""

import os
import tempfile
import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.evaluator import (
    build_evaluation_config,
    build_evaluation_splits,
    collect_predictions,
    evaluate_loader,
    load_model_from_checkpoint,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    save_evaluation_artifacts,
)
from src.models.classifier import MammographyClassifier
from src.evaluation.metrics import compute_binary_metrics


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

        loaded, checkpoint = load_model_from_checkpoint(path, device="cpu")
        assert checkpoint["epoch"] == 3
        assert not loaded.training  # eval mode

        y_true, y_prob, meta = collect_predictions(loaded, tiny_loader, device="cpu", return_meta=True)
        original_true, original_prob, _ = collect_predictions(model, tiny_loader, device="cpu")
        np.testing.assert_array_equal(y_true, original_true)
        np.testing.assert_allclose(y_prob, original_prob)

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
            "metrics", "report", "roc_curve", "pr_curve", "confusion_matrix", "predictions",
        }
        for path in written.values():
            assert os.path.exists(path) and os.path.getsize(path) > 0
        assert os.path.exists(os.path.join(tmp_dir, "metrics.json"))
        predictions = pd.read_csv(written["predictions"])
        np.testing.assert_allclose(predictions["malignant_probability"], y_prob)


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


def test_saved_checkpoint_controls_split_and_preprocessing():
    saved = {
        "experiment_name": "actual_model", "seed": 17,
        "model": {"backbone": "resnet18"},
        "data": {"image_size": 64, "test_split_ratio": 0.2, "val_split_ratio": 0.1,
                 "dataset_dir": "old_mount", "manifest_path": "old_manifest", "batch_size": 8},
    }
    snapshot = deepcopy(saved)
    current = {
        "experiment_name": "wrong_model", "seed": 999,
        "data": {"image_size": 224, "test_split_ratio": 0.4, "val_split_ratio": 0.3,
                 "dataset_dir": "new_mount", "manifest_path": "new_manifest", "batch_size": 4},
        "evaluation": {"threshold": 0.6},
    }
    config = build_evaluation_config({"config": saved}, current)
    assert config["seed"] == 17
    assert config["experiment_name"] == "actual_model"
    assert config["data"]["image_size"] == 64
    assert config["data"]["test_split_ratio"] == 0.2
    assert config["data"]["val_split_ratio"] == 0.1
    assert config["data"]["dataset_dir"] == "new_mount"
    assert config["data"]["manifest_path"] == "new_manifest"
    assert config["data"]["batch_size"] == 4
    assert config["evaluation"]["threshold"] == 0.6
    assert saved == snapshot


def test_missing_saved_split_config_is_not_silently_guessed():
    with pytest.raises(ValueError, match="split ratios"):
        build_evaluation_config({"config": {}}, {"seed": 42})


def test_saved_cv_patient_assignments_are_used(tmp_path):
    path = tmp_path / "manifest.csv"
    pd.DataFrame({"patient_id": ["A", "A", "B", "C"], "label": ["BENIGN"] * 4}).to_csv(path, index=False)
    config = {
        "seed": 42,
        "data": {"manifest_path": str(path), "image_size": 64},
        "split_patient_ids": {"train": ["C"], "val": ["B"], "test": ["A"]},
    }
    config = build_evaluation_config({"config": config}, {})
    splits = build_evaluation_splits(config)
    assert len(splits["test"]) == 2
    assert splits["train"]["patient_id"].tolist() == ["C"]
    config["split_patient_ids"]["val"] = ["A"]
    with pytest.raises(ValueError, match="overlap"):
        build_evaluation_splits(config)


def test_single_class_artifacts_use_null_and_preserve_sample_order(tmp_path):
    y_true, y_prob = np.array([0, 0]), np.array([0.2, 0.7])
    info = pd.DataFrame({"patient_id": ["second", "first"], "image_path": ["b.jpg", "a.jpg"]}, index=[3, 9])
    written = save_evaluation_artifacts(compute_binary_metrics(y_true, y_prob), y_true, y_prob, str(tmp_path), info)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["roc_auc"] is None and metrics["sensitivity"] is None
    predictions = pd.read_csv(written["predictions"])
    assert predictions["patient_id"].tolist() == ["second", "first"]
    np.testing.assert_allclose(predictions["malignant_probability"], y_prob)
    report = (tmp_path / "classification_report.txt").read_text()
    assert "BENIGN" in report and "MALIGNANT" in report
