"""Training loop for the baseline benign-vs-malignant classifier.

Handles class-imbalance-aware loss, Adam optimization, LR scheduling, gradient
clipping, early stopping, checkpointing, and metric logging. The loop is
deliberately backbone-agnostic: it consumes a :class:`torch.nn.Module` and two
:class:`torch.utils.data.DataLoader` objects whose batches are the dicts produced
by :class:`src.data.dataset.MammographyDataset`.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.losses import build_criterion
from src.utils.logging_setup import setup_logger


def _resolve_device(device: Optional[str]) -> torch.device:
    """Resolve ``"auto"``/``None`` to CUDA when available, else CPU."""
    if device in (None, "auto", "cuda-if-available"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    training_cfg: Mapping[str, Any],
) -> Tuple[Optional[torch.optim.lr_scheduler._LRScheduler], bool]:
    """Build the LR scheduler from config.

    Returns:
        A ``(scheduler, steps_on_plateau)`` tuple. ``steps_on_plateau`` indicates
        the scheduler must be stepped with a metric (``ReduceLROnPlateau``).
    """
    sched_cfg = dict(training_cfg.get("scheduler") or {})
    sched_type = str(sched_cfg.get("type", "none")).lower()

    if sched_type in ("none", ""):
        return None, False

    if sched_type in ("reduce_on_plateau", "plateau"):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(sched_cfg.get("factor", 0.5)),
            patience=int(sched_cfg.get("patience", 2)),
            min_lr=float(sched_cfg.get("min_lr", 1e-6)),
        )
        return scheduler, True

    if sched_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(sched_cfg.get("t_max", training_cfg.get("epochs", 30))),
            eta_min=float(sched_cfg.get("min_lr", 1e-6)),
        )
        return scheduler, False

    if sched_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(sched_cfg.get("step_size", 10)),
            gamma=float(sched_cfg.get("gamma", 0.1)),
        )
        return scheduler, False

    raise ValueError(
        f"Unsupported scheduler type '{sched_type}'. "
        "Expected 'none', 'reduce_on_plateau', 'cosine' or 'step'."
    )


def _unpack_batch(batch: Any) -> Tuple[Any, Any]:
    """Extract ``(images, labels)`` from a batch.

    Supports the dict batches produced by
    :class:`src.data.dataset.MammographyDataset` as well as plain
    ``(image, label)`` tuples/lists (useful for lightweight synthetic datasets).
    """
    if isinstance(batch, Mapping):
        return batch["image"], batch["label"]
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TypeError(
        "Unsupported batch type: expected a dict with 'image'/'label' keys "
        "or an (image, label) tuple."
    )


class Trainer:
    """Train, validate, checkpoint and early-stop a mammography classifier.

    Args:
        model: The classifier to train.
        train_loader: DataLoader yielding ``{"image", "label", ...}`` batches.
        val_loader: Validation DataLoader (may be ``None`` for train-only runs).
        config: Full parsed config dict (``data``/``model``/``training``/...).
        criterion: Optional loss module. Built from ``config["training"]`` when
            omitted.
        device: ``"auto"``/``None`` (default), ``"cuda"``, ``"cpu"``.
        logger: Optional logger; a default file+console logger is created when
            omitted.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: Mapping[str, Any],
        criterion: Optional[nn.Module] = None,
        device: Optional[str] = None,
        logger: Optional[Any] = None,
    ) -> None:
        self.config = dict(config)
        self.training_cfg = dict(config.get("training") or {})
        self.device = _resolve_device(device or self.training_cfg.get("device", "auto"))
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger or setup_logger()

        self.criterion = criterion or build_criterion(self.training_cfg)
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=float(self.training_cfg.get("learning_rate", 1e-4)),
            weight_decay=float(self.training_cfg.get("weight_decay", 0.0)),
        )
        self.scheduler, self._sched_on_plateau = _build_scheduler(
            self.optimizer, self.training_cfg
        )

        self.grad_clip = self.training_cfg.get("grad_clip")
        self.epochs = int(self.training_cfg.get("epochs", 30))
        self.patience = int(self.training_cfg.get("early_stopping_patience", 5))
        self.monitor = str(self.training_cfg.get("monitor", "val_loss"))
        self._lower_is_better = "loss" in self.monitor

        experiment_name = str(self.config.get("experiment_name", "experiment"))
        checkpoint_root = str(self.training_cfg.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir = os.path.join(checkpoint_root, experiment_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.history: Dict[str, list] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }
        self.start_epoch = 0
        self.best_score = float("inf") if self._lower_is_better else -float("inf")
        self.epochs_no_improve = 0

    def _run_epoch(self, loader: DataLoader, train: bool) -> Tuple[float, float]:
        """Run one pass over ``loader``; return ``(mean_loss, accuracy)``."""
        self.model.train(train)
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in loader:
            images, labels = _unpack_batch(batch)
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).float()

            with torch.set_grad_enabled(train):
                logits = self.model(images)
                # Align target shape with logits (binary head emits [B, 1]);
                # focal loss reshapes internally, BCE requires exact match.
                loss = self.criterion(logits, labels.view_as(logits))

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if self.grad_clip:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), float(self.grad_clip)
                        )
                    self.optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            preds = (torch.sigmoid(logits.reshape(-1)) >= 0.5).float()
            correct += (preds == labels.reshape(-1)).sum().item()
            total += batch_size

        total = max(total, 1)
        return total_loss / total, correct / total

    def _save_checkpoint(self, filename: str, epoch: int, is_best: bool) -> str:
        """Persist model/optimizer state plus the config (reproducibility)."""
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_score": self.best_score,
                "is_best": is_best,
                "backbone": getattr(self.model, "backbone_name", None),
                "num_classes": getattr(self.model, "num_classes", None),
                "config": self.config,
                "history": self.history,
            },
            path,
        )
        return path

    def train(self) -> Dict[str, list]:
        """Run the full training loop and return the metric history."""
        self.logger.info(
            "Training on %s for up to %d epochs (monitor=%s, early_stopping_patience=%d)",
            self.device,
            self.epochs,
            self.monitor,
            self.patience,
        )
        self.logger.info("Checkpoints -> %s", self.checkpoint_dir)

        for epoch in range(self.start_epoch, self.epochs):
            started = time.time()

            train_loss, train_acc = self._run_epoch(self.train_loader, train=True)
            if self.val_loader is not None:
                val_loss, val_acc = self._run_epoch(self.val_loader, train=False)
            else:
                val_loss, val_acc = train_loss, train_acc

            if self.scheduler is not None:
                if self._sched_on_plateau:
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            for key, value in (
                ("train_loss", train_loss),
                ("train_acc", train_acc),
                ("val_loss", val_loss),
                ("val_acc", val_acc),
                ("lr", current_lr),
            ):
                self.history[key].append(value)

            self.logger.info(
                "Epoch %d/%d - train_loss %.4f - train_acc %.4f - "
                "val_loss %.4f - val_acc %.4f - lr %.2e - %.1fs",
                epoch + 1,
                self.epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                current_lr,
                time.time() - started,
            )

            self._save_checkpoint("last.pt", epoch, is_best=False)

            score = val_loss if self._lower_is_better else val_acc
            improved = (
                score < self.best_score if self._lower_is_better else score > self.best_score
            )
            if improved:
                self.best_score = score
                self.epochs_no_improve = 0
                best_path = self._save_checkpoint("best.pt", epoch, is_best=True)
                self.logger.info("New best %s=%.4f -> %s", self.monitor, score, best_path)
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    self.logger.info(
                        "Early stopping triggered at epoch %d (best %s=%.4f)",
                        epoch + 1,
                        self.monitor,
                        self.best_score,
                    )
                    break

        return self.history
