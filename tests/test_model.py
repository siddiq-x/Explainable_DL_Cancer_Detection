"""Unit tests for the baseline classifier, losses and training loop.

These tests are self-contained (synthetic tensors, ``pretrained=False``) so they
run fast on CPU without downloading weights or needing the real dataset.
"""

import logging
import os
import tempfile

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.classifier import MammographyClassifier
from src.models.losses import BinaryFocalLoss, build_criterion, compute_pos_weight
from src.models.trainer import Trainer


@pytest.mark.parametrize("backbone", ["resnet18", "resnet50"])
def test_classifier_output_shape_and_probabilities(backbone):
    model = MammographyClassifier(backbone=backbone, num_classes=2, pretrained=False)
    model.eval()

    images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(images)
        probs = model.predict_proba(images)

    # Binary head emits a single logit per sample.
    assert logits.shape == (2, 1)
    assert probs.shape == (2, 1)
    assert torch.all((probs >= 0) & (probs <= 1))


def test_freeze_layers_toggles_backbone_grads():
    model = MammographyClassifier(
        backbone="resnet18", pretrained=False, freeze_layers=True
    )
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert any(p.requires_grad for p in model.head.parameters())

    model.unfreeze_backbone()
    assert any(p.requires_grad for p in model.backbone.parameters())


def test_focal_loss_is_finite_and_non_negative():
    criterion = BinaryFocalLoss(gamma=2.0, alpha=0.25)
    logits = torch.tensor([[2.0], [-2.0], [0.0]])
    targets = torch.tensor([1.0, 0.0, 1.0])

    loss = criterion(logits, targets)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_focal_gamma_zero_matches_bce_without_alpha():
    criterion = BinaryFocalLoss(gamma=0.0, alpha=None)
    logits = torch.tensor([[1.5], [-0.5], [0.25]])
    targets = torch.tensor([1.0, 0.0, 1.0])

    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        logits.reshape(-1), targets.reshape(-1).float()
    )

    assert torch.allclose(criterion(logits, targets), expected, atol=1e-6)


def test_compute_pos_weight():
    labels = ["MALIGNANT", "BENIGN", "BENIGN_WITHOUT_CALLBACK", "BENIGN"]
    # 1 positive, 3 negatives -> 3.0
    assert compute_pos_weight(labels) == pytest.approx(3.0)


def test_build_criterion_selects_loss_type():
    focal = build_criterion(
        {"loss": {"type": "focal", "gamma": 2.0, "alpha": 0.25}}, pos_weight=1.5
    )
    assert isinstance(focal, BinaryFocalLoss)

    bce = build_criterion(
        {"loss": {"type": "bce", "class_weighting": True}}, pos_weight=2.0
    )
    assert isinstance(bce, torch.nn.BCEWithLogitsLoss)

    with pytest.raises(ValueError):
        build_criterion({"loss": {"type": "not_a_loss"}})


def test_tiny_batch_overfit_sanity_check():
    """A tiny model must be able to drive training loss down on a few samples."""
    torch.manual_seed(0)
    images = torch.randn(8, 3, 64, 64)
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.float32)
    loader = DataLoader(TensorDataset(images, labels), batch_size=8, shuffle=False)

    model = MammographyClassifier(backbone="resnet18", pretrained=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    first_loss = last_loss = None
    steps = 60
    for step in range(steps):
        for batch_images, batch_labels in loader:
            model.train()
            logits = model(batch_images).reshape(-1)
            loss = criterion(logits, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step == 0:
                first_loss = loss.item()
            last_loss = loss.item()

    model.eval()
    with torch.no_grad():
        preds = (torch.sigmoid(model(images).reshape(-1)) >= 0.5).float()
    accuracy = (preds == labels).float().mean().item()

    assert last_loss < first_loss, "Training loss did not decrease"
    assert accuracy >= 0.875, f"Failed to overfit the tiny batch (acc={accuracy})"


def test_trainer_runs_and_writes_checkpoints():
    torch.manual_seed(0)
    images = torch.randn(8, 3, 64, 64)
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.float32)
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)

    model = MammographyClassifier(backbone="resnet18", pretrained=False)
    logger = logging.getLogger("test_trainer")
    logger.addHandler(logging.NullHandler())

    with tempfile.TemporaryDirectory() as tmp_dir:
        config = {
            "experiment_name": "test_run",
            "training": {
                "epochs": 2,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "early_stopping_patience": 5,
                "checkpoint_dir": tmp_dir,
                "monitor": "val_loss",
                "grad_clip": 1.0,
                "loss": {"type": "bce", "class_weighting": False},
                "scheduler": {"type": "none"},
            },
        }

        trainer = Trainer(model, loader, loader, config, logger=logger)
        history = trainer.train()

        assert len(history["train_loss"]) == 2
        assert history["val_loss"][0] > 0
        assert os.path.exists(os.path.join(tmp_dir, "test_run", "best.pt"))
        assert os.path.exists(os.path.join(tmp_dir, "test_run", "last.pt"))
