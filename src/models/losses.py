"""Loss functions and helpers for imbalanced binary classification.

CBIS-DDSM is mildly imbalanced (~59% benign / 41% malignant), so the training
loss can either up-weight the positive class (``pos_weight``) or apply focal
loss, which down-weights easy examples. Both are exposed here and selected via
the ``training.loss`` section of the YAML config.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_pos_weight(labels: Iterable[Any]) -> float:
    """Compute ``num_negative / num_positive`` for ``BCEWithLogitsLoss``.

    ``BENIGN`` and ``BENIGN_WITHOUT_CALLBACK`` are both treated as the negative
    class; ``MALIGNANT`` is the positive class.

    Args:
        labels: Iterable of raw label strings from the manifest.

    Returns:
        The positive-class weight (``1.0`` if there are no positives).
    """
    positives = 0
    negatives = 0
    for label in labels:
        if str(label).strip().upper() == "MALIGNANT":
            positives += 1
        else:
            negatives += 1
    if positives == 0:
        return 1.0
    return negatives / positives


class BinaryFocalLoss(nn.Module):
    """Binary focal loss (Lin et al., 2017) operating on raw logits.

    Down-weights easy examples via ``(1 - p_t) ** gamma`` so training focuses on
    hard, misclassified samples.

    Args:
        gamma: Focusing parameter (``0`` reduces to plain BCE).
        alpha: Optional balance factor for the positive class. ``None`` disables
            per-class balancing.
        pos_weight: Optional positive-class weight forwarded to the underlying
            BCE. Using both ``alpha`` and ``pos_weight`` is allowed but
            double-corrects for class imbalance.
        reduction: ``"mean"`` (default), ``"sum"`` or ``"none"``.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = 0.25,
        pos_weight: Optional[float] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = None if alpha is None else float(alpha)
        self.reduction = reduction
        if pos_weight is None:
            self.register_buffer("pos_weight", None)
        else:
            self.register_buffer(
                "pos_weight",
                torch.as_tensor(pos_weight, dtype=torch.float32).reshape(1),
            )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.reshape(-1)
        targets = targets.reshape(-1).float()

        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight
        )

        # p_t = probability assigned to the true class.
        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
        loss = (1.0 - p_t).clamp(min=0.0) ** self.gamma * bce

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_criterion(
    training_cfg: Mapping[str, Any],
    pos_weight: Optional[float] = None,
) -> nn.Module:
    """Build the training loss from the ``training.loss`` config section.

    Recognised ``type`` values:
        * ``"focal"`` -> :class:`BinaryFocalLoss`
        * ``"bce"``   -> :class:`torch.nn.BCEWithLogitsLoss`

    ``class_weighting`` (default ``True``) controls whether ``pos_weight`` is
    applied. It is ignored when ``pos_weight`` is ``None``.

    Args:
        training_cfg: The ``training`` section of the parsed config.
        pos_weight: Positive-class weight, typically from
            :func:`compute_pos_weight`.

    Returns:
        A loss module that accepts ``(logits, targets)``.
    """
    loss_cfg = dict(training_cfg.get("loss") or {})
    loss_type = str(loss_cfg.get("type", "bce")).lower()
    class_weighting = bool(loss_cfg.get("class_weighting", True))
    effective_pos_weight = pos_weight if class_weighting else None

    if loss_type == "focal":
        return BinaryFocalLoss(
            gamma=float(loss_cfg.get("gamma", 2.0)),
            alpha=loss_cfg.get("alpha", 0.25),
            pos_weight=effective_pos_weight,
        )

    if loss_type == "bce":
        weight_tensor = None
        if effective_pos_weight is not None:
            weight_tensor = torch.as_tensor(
                effective_pos_weight, dtype=torch.float32
            ).reshape(1)
        return nn.BCEWithLogitsLoss(pos_weight=weight_tensor)

    raise ValueError(
        f"Unsupported loss type '{loss_type}'. Expected 'focal' or 'bce'."
    )
