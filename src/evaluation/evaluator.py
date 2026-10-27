"""Evaluation runner: checkpoint loading, inference, metrics and plots.

Given a checkpoint produced by :class:`src.models.trainer.Trainer`, this module
reconstructs the model, runs the requested split, and writes a machine-readable
metrics file plus ROC, precision-recall and confusion-matrix plots.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe (CLI / servers / Kaggle)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data.dataset import unpack_batch  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    compute_binary_metrics,
    format_metrics,
    pr_points,
    roc_points,
)
from src.models.classifier import MammographyClassifier  # noqa: E402


def resolve_device(device: Optional[str] = None) -> torch.device:
    """Resolve ``None``/``"auto"`` to CUDA when available, else CPU."""
    if device in (None, "auto", "cuda-if-available"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: Optional[str] = None,
) -> Tuple[MammographyClassifier, Dict[str, Any]]:
    """Rebuild a :class:`MammographyClassifier` from a saved checkpoint.

    The backbone/num_classes are recovered from the checkpoint itself, falling
    back to its stored config, so no external config is required to load a model.

    Returns:
        ``(model_in_eval_mode, checkpoint_dict)``.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    dev = resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    config = checkpoint.get("config") or {}
    model_cfg = config.get("model") or {}

    backbone = checkpoint.get("backbone") or model_cfg.get("backbone", "resnet50")
    num_classes = checkpoint.get("num_classes") or model_cfg.get("num_classes", 2)

    model = MammographyClassifier(
        backbone=backbone, num_classes=int(num_classes), pretrained=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(dev)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def collect_predictions(
    model: MammographyClassifier,
    loader: DataLoader,
    device: Optional[str] = None,
    return_meta: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[str]]]:
    """Run inference over ``loader`` and collect labels, probabilities and paths.

    Args:
        model: Model in eval mode.
        loader: DataLoader yielding dict or tuple batches.
        device: Device to run on (``None``/``"auto"`` -> CUDA if available).
        return_meta: Also collect ``image_path``/``mask_path`` when available
            (used later for Grad-CAM-vs-ROI validation).

    Returns:
        ``(y_true, y_prob, meta)`` where ``meta`` has ``image_path`` and
        ``mask_path`` lists (empty when unavailable).
    """
    dev = resolve_device(device)
    model.to(dev)
    model.eval()

    y_true: List[float] = []
    y_prob: List[float] = []
    meta: Dict[str, List[str]] = {"image_path": [], "mask_path": []}

    for batch in loader:
        images, labels = unpack_batch(batch)
        images = images.to(dev)
        probs = model.predict_proba(images).reshape(-1)
        y_true.extend(labels.reshape(-1).float().cpu().tolist())
        y_prob.extend(probs.cpu().tolist())

        if return_meta and isinstance(batch, Mapping):
            meta["image_path"].extend(list(batch.get("image_path", [])))
            meta["mask_path"].extend(list(batch.get("mask_path", [])))

    return np.asarray(y_true), np.asarray(y_prob), meta


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, path: str) -> str:
    """Save the ROC curve plot and return its path."""
    fpr, tpr = roc_points(y_true, y_prob)
    roc_auc = compute_binary_metrics(y_true, y_prob)["roc_auc"]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="chance")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_precision_recall_curve(
    y_true: np.ndarray, y_prob: np.ndarray, path: str
) -> str:
    """Save the precision-recall curve plot and return its path."""
    recall, precision = pr_points(y_true, y_prob)
    metrics = compute_binary_metrics(y_true, y_prob)
    prevalence = metrics["n_positive"] / max(metrics["n_samples"], 1)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, label=f"PR (AP = {metrics['pr_auc']:.3f})")
    ax.axhline(prevalence, color="k", linestyle="--", linewidth=1,
               label=f"baseline (prevalence = {prevalence:.2f})")
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion_matrix(metrics: Mapping[str, float], path: str) -> str:
    """Save the confusion matrix as an annotated heatmap and return its path."""
    matrix = np.array(
        [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]], dtype=int
    )

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["pred benign", "pred malignant"])
    ax.set_yticks([0, 1], labels=["true benign", "true malignant"])
    ax.set_xlabel(f"threshold = {metrics['threshold']:.2f}")
    ax.set_title("Confusion matrix")

    threshold = matrix.max() / 2 if matrix.max() > 0 else 0.5
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
                fontsize=12,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def evaluate_loader(
    model: MammographyClassifier,
    loader: DataLoader,
    threshold: float = 0.5,
    device: Optional[str] = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Compute metrics for one loader.

    Returns:
        ``(metrics, y_true, y_prob)``.
    """
    y_true, y_prob, _ = collect_predictions(model, loader, device=device)
    metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)
    return metrics, y_true, y_prob


def save_evaluation_artifacts(
    metrics: Dict[str, float],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: str,
) -> Dict[str, str]:
    """Write metrics JSON, a text report, and the three plots into ``output_dir``.

    Returns:
        Mapping of artifact name -> written path.
    """
    os.makedirs(output_dir, exist_ok=True)

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("Classification report\n")
        fh.write("=====================\n")
        fh.write(format_metrics(metrics) + "\n")

    return {
        "metrics": metrics_path,
        "report": report_path,
        "roc_curve": plot_roc_curve(y_true, y_prob, os.path.join(output_dir, "roc_curve.png")),
        "pr_curve": plot_precision_recall_curve(
            y_true, y_prob, os.path.join(output_dir, "pr_curve.png")
        ),
        "confusion_matrix": plot_confusion_matrix(
            metrics, os.path.join(output_dir, "confusion_matrix.png")
        ),
    }
