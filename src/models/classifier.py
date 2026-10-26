"""Classifier models for binary benign-vs-malignant mammography classification.

The model wraps a pre-trained torchvision backbone (ResNet / EfficientNet) and
replaces its classification head with a small dropout + linear head suited to
the binary CBIS-DDSM task. For ``num_classes == 2`` the head emits a single
logit so the network pairs naturally with ``BCEWithLogitsLoss`` / focal loss
(the dataset yields scalar float labels in ``{0, 1}``).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models

# Backbones we explicitly support and know how to attach a custom head to.
SUPPORTED_BACKBONES: Tuple[str, ...] = (
    "resnet18",
    "resnet34",
    "resnet50",
    "efficientnet_b0",
)


def _build_backbone(name: str, pretrained: bool) -> Tuple[nn.Module, int]:
    """Instantiate a torchvision backbone with its classifier head removed.

    Args:
        name: One of :data:`SUPPORTED_BACKBONES`.
        pretrained: Whether to load ImageNet weights.

    Returns:
        A ``(module, feature_dim)`` tuple where ``module`` maps an image batch to
        a feature tensor and ``feature_dim`` is the flattened feature size.
    """
    name = name.lower()
    if name not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported backbone '{name}'. Expected one of {SUPPORTED_BACKBONES}."
        )

    weights = "DEFAULT" if pretrained else None

    if name.startswith("resnet"):
        backbone = getattr(models, name)(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, feature_dim

    # efficientnet_b0: classifier is Sequential(Dropout, Linear).
    backbone = models.efficientnet_b0(weights=weights)
    feature_dim = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()
    return backbone, feature_dim


class MammographyClassifier(nn.Module):
    """Pre-trained backbone + custom head for benign vs. malignant classification.

    Args:
        backbone: Backbone architecture name (see :data:`SUPPORTED_BACKBONES`).
        num_classes: Number of classes. ``2`` selects a single-logit binary
            head; other values produce a ``num_classes``-wide logit vector.
        pretrained: Load ImageNet weights for the backbone.
        freeze_layers: If ``True``, freeze the backbone so only the new head is
            trained (useful as a fast linear-probe baseline).
        dropout: Dropout probability applied before the final linear layer.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_layers: bool = False,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes
        # Binary problems use a single logit (paired with BCE/focal loss).
        self.out_features = 1 if num_classes == 2 else num_classes

        self.backbone, feature_dim = _build_backbone(backbone, pretrained)

        if freeze_layers:
            self.freeze_backbone()

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, self.out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape ``(batch, out_features)``."""
        features = self.backbone(x)
        features = torch.flatten(features, start_dim=1)
        return self.head(features)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities (sigmoid for binary, softmax otherwise)."""
        logits = self.forward(x)
        if self.out_features == 1:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=1)

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (only the head will be trained)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters (fine-tuning)."""
        for param in self.backbone.parameters():
            param.requires_grad = True
