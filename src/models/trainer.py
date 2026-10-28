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
from copy import deepcopy
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import unpack_batch
from src.models.losses import build_criterion, compute_pos_weight
from src.utils.logging_setup import setup_logger
from src.utils.seeding import seed_everything


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
        self.config = deepcopy(dict(config))
        self.training_cfg = dict(config.get("training") or {})
        self.device = _resolve_device(device or self.training_cfg.get("device", "auto"))
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger or setup_logger()

        self.criterion = (criterion if criterion is not None else build_criterion(self.training_cfg)).to(self.device)
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
        if self.monitor not in ("val_loss", "val_acc", "val_auc"):
            raise ValueError("monitor must be val_loss, val_acc or val_auc.")
        if self.monitor == "val_auc" and val_loader is None:
            raise ValueError("val_auc monitoring requires a validation loader.")
        if self.epochs < 1 or self.patience < 1:
            raise ValueError("epochs and early_stopping_patience must be positive.")
        self._lower_is_better = self.monitor == "val_loss"

        experiment_name = str(self.config.get("experiment_name", "experiment"))
        checkpoint_root = str(self.training_cfg.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir = os.path.join(checkpoint_root, experiment_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.history: Dict[str, list] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_auc": [],
            "lr": [],
        }
        self.start_epoch = 0
        self.best_score = float("inf") if self._lower_is_better else -float("inf")
        self.epochs_no_improve = 0

    def _run_epoch(
        self, loader: DataLoader, train: bool
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Run one pass over ``loader``.

        Returns:
            ``(mean_loss, accuracy, y_true, y_prob)`` where the label/probability
            arrays enable epoch-level AUC tracking.
        """
        self.model.train(train)
        total_loss = 0.0
        correct = 0
        total = 0
        all_true: List[float] = []
        all_prob: List[float] = []

        for batch in loader:
            images, labels = unpack_batch(batch)
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

            probs = torch.sigmoid(logits.reshape(-1))
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            preds = (probs >= 0.5).float()
            correct += (preds == labels.reshape(-1)).sum().item()
            total += batch_size
            all_true.extend(labels.reshape(-1).detach().cpu().tolist())
            all_prob.extend(probs.detach().cpu().tolist())

        if total == 0:
            raise ValueError("Cannot train or validate on an empty loader.")
        return (
            total_loss / total,
            correct / total,
            np.asarray(all_true),
            np.asarray(all_prob),
        )

    def _save_checkpoint(self, filename: str, epoch: int, is_best: bool) -> str:
        """Persist model/optimizer state plus the config (reproducibility)."""
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_score": self.best_score,
                "epochs_no_improve": self.epochs_no_improve,
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
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

            train_loss, train_acc, _, _ = self._run_epoch(self.train_loader, train=True)
            if self.val_loader is not None:
                val_loss, val_acc, val_true, val_prob = self._run_epoch(
                    self.val_loader, train=False
                )
                val_auc = self._safe_auc(val_true, val_prob)
            else:
                val_loss, val_acc, val_auc = train_loss, train_acc, float("nan")

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
                ("val_auc", val_auc),
                ("lr", current_lr),
            ):
                self.history[key].append(value)

            self.logger.info(
                "Epoch %d/%d - train_loss %.4f - train_acc %.4f - "
                "val_loss %.4f - val_acc %.4f - val_auc %.4f - lr %.2e - %.1fs",
                epoch + 1,
                self.epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                val_auc,
                current_lr,
                time.time() - started,
            )

            score = self._monitor_value(val_loss, val_acc, val_auc)
            improved = np.isfinite(score) and (
                score < self.best_score
                if self._lower_is_better
                else score > self.best_score
            )
            if improved:
                self.best_score = score
                self.epochs_no_improve = 0
                best_path = self._save_checkpoint("best.pt", epoch, is_best=True)
                self.logger.info("New best %s=%.4f -> %s", self.monitor, score, best_path)
            else:
                self.epochs_no_improve += 1
            # Save after updating best_score/counters so last.pt is consistent.
            self._save_checkpoint("last.pt", epoch, is_best=False)
            if self.epochs_no_improve >= self.patience:
                self.logger.info(
                    "Early stopping triggered at epoch %d (best %s=%.4f)",
                    epoch + 1, self.monitor, self.best_score,
                )
                break

        return self.history

    def _monitor_value(self, val_loss: float, val_acc: float, val_auc: float) -> float:
        """Select the value to monitor for checkpointing/early stopping."""
        return {
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_auc": val_auc,
        }[self.monitor]

    @staticmethod
    def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """ROC-AUC, or ``nan`` when undefined (e.g. single class in the split)."""
        from sklearn.metrics import roc_auc_score

        if np.unique(np.asarray(y_true)).size < 2:
            return float("nan")
        try:
            return float(roc_auc_score(y_true, y_prob))
        except ValueError:
            return float("nan")


def _aggregate(values: List[float]) -> Tuple[float, float]:
    """Return ``(mean, std)`` ignoring NaNs; ``(nan, nan)`` if all are NaN."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(finite.mean()), float(finite.std())


def cross_validate(
    config: Mapping[str, Any],
    model_factory: Callable[[], nn.Module],
    manifest_path: str,
    dataset_dir: str,
    k: int = 5,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    logger: Optional[Any] = None,
    output_dir: Optional[str] = None,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Run patient-wise, stratified K-fold cross-validation.

    Trains ``k`` independent models. Within each outer training fold, reserve
    inner validation patients for early stopping. Restore the selected best
    checkpoint before scoring the untouched outer fold. These cross-validation
    scores are distinct from the original baseline's held-out test report.

    Args:
        config: Full parsed config (``data``/``training``/``evaluation``/...).
        model_factory: Zero-arg callable returning a *fresh* untrained model.
        manifest_path: Manifest CSV path.
        dataset_dir: Root the manifest paths are relative to.
        k: Number of folds.
        seed: Overrides ``config["seed"]``.
        device: ``"auto"``/``None``/``"cuda"``/``"cpu"``.
        logger: Optional logger.
        output_dir: When given, per-fold + summary results are written here as
            ``kfold_results.json``.
        threshold: Decision threshold for hard-label metrics (defaults to
            ``config["evaluation"]["threshold"]`` or 0.5).

    Returns:
        Dict with ``folds`` (per-fold metrics + history) and ``summary``
        (mean/std of roc_auc, pr_auc, f1, accuracy, sensitivity, specificity).
    """
    import json

    # Lazy imports: keep the training import graph free of matplotlib/sklearn-metrics.
    from src.data.augmentation import get_train_transforms
    from src.data.dataset import MammographyDataset
    from src.data.preprocessing import get_base_transforms
    from src.data.splits import create_stratified_group_kfold, split_patient_validation
    from src.evaluation.evaluator import collect_predictions
    from src.evaluation.metrics import compute_binary_metrics

    seed = int(seed if seed is not None else config.get("seed", 42))
    data_cfg = config.get("data") or {}
    image_size = int(data_cfg.get("image_size", 224))
    batch_size = int(data_cfg.get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 0))
    eval_cfg = config.get("evaluation") or {}
    threshold = float(
        threshold if threshold is not None else eval_cfg.get("threshold", 0.5)
    )
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("The decision threshold must be finite and in [0, 1].")

    logger = logger or setup_logger()
    folds_splits = create_stratified_group_kfold(manifest_path, k=k, seed=seed)
    logger.info("[k-fold] running %d folds (patient-wise, stratified)", k)

    folds: List[Dict[str, Any]] = []
    collected: Dict[str, List[float]] = {
        "roc_auc": [],
        "pr_auc": [],
        "f1": [],
        "accuracy": [],
        "sensitivity": [],
        "specificity": [],
    }

    for fold_idx, (development_df, test_df) in enumerate(folds_splits):
        fold_seed = seed + fold_idx
        seed_everything(fold_seed)
        train_df, val_df = split_patient_validation(
            development_df, val_size=float(data_cfg.get("val_split_ratio", 0.15)), seed=fold_seed,
        )
        train_ds = MammographyDataset(
            train_df, root_dir=dataset_dir, transform=get_train_transforms(image_size)
        )
        val_ds = MammographyDataset(
            val_df, root_dir=dataset_dir, transform=get_base_transforms(image_size)
        )
        test_ds = MammographyDataset(
            test_df, root_dir=dataset_dir, transform=get_base_transforms(image_size)
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        logger.info(
            "[k-fold] fold %d/%d - train=%d inner_val=%d outer_test=%d",
            fold_idx + 1,
            k,
            len(train_ds),
            len(val_ds),
            len(test_ds),
        )
        fold_name = f"{config.get('experiment_name', 'model')}_fold{fold_idx}"
        fold_config = deepcopy(dict(config))
        fold_config.update(experiment_name=fold_name, seed=fold_seed)
        fold_config["split_patient_ids"] = {
            "train": sorted(train_df["patient_id"].unique().tolist()),
            "val": sorted(val_df["patient_id"].unique().tolist()),
            "test": sorted(test_df["patient_id"].unique().tolist()),
        }
        criterion = build_criterion(
            fold_config.get("training") or {}, pos_weight=compute_pos_weight(train_df["label"]),
        )
        trainer = Trainer(
            model=model_factory(),
            train_loader=train_loader,
            val_loader=val_loader,
            config=fold_config,
            criterion=criterion,
            device=device,
            logger=logger,
        )
        history = trainer.train()

        best_path = os.path.join(trainer.checkpoint_dir, "best.pt")
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        trainer.model.load_state_dict(best["model_state_dict"])
        y_true, y_prob, _ = collect_predictions(
            trainer.model, test_loader, device=str(trainer.device)
        )
        metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)
        for key in collected:
            collected[key].append(metrics[key])

        folds.append(
            {
                "fold": fold_idx,
                "n_train": len(train_ds),
                "n_val": len(val_ds),
                "n_test": len(test_ds),
                "seed": fold_seed,
                "best_epoch": int(best["epoch"]) + 1,
                "best_checkpoint": best_path,
                "patients": fold_config["split_patient_ids"],
                "metrics": metrics,
                "history": {key: list(values) for key, values in history.items()},
            }
        )
        logger.info(
            "[k-fold] fold %d - roc_auc=%.4f pr_auc=%.4f f1=%.4f sens=%.4f spec=%.4f",
            fold_idx + 1,
            metrics["roc_auc"],
            metrics["pr_auc"],
            metrics["f1"],
            metrics["sensitivity"],
            metrics["specificity"],
        )
        del best, trainer

    summary: Dict[str, float] = {"k": float(k)}
    for key, values in collected.items():
        mean, std = _aggregate(values)
        summary[f"{key}_mean"] = mean
        summary[f"{key}_std"] = std

    logger.info(
        "[k-fold] summary - roc_auc=%.4f±%.4f  pr_auc=%.4f±%.4f  f1=%.4f±%.4f",
        summary["roc_auc_mean"],
        summary["roc_auc_std"],
        summary["pr_auc_mean"],
        summary["pr_auc_std"],
        summary["f1_mean"],
        summary["f1_std"],
    )

    result: Dict[str, Any] = {
        "k": k,
        "threshold": threshold,
        "protocol": "patient-wise outer folds; inner validation selects best checkpoint",
        "folds": folds,
        "summary": summary,
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "kfold_results.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, allow_nan=False)
        logger.info("[k-fold] results written to %s", out_path)

    return result
