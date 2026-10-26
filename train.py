"""CLI entry point: train the baseline benign-vs-malignant classifier.

Examples
--------
Local smoke run on mock data (fast, CPU)::

    python train.py --epochs 2 --batch-size 8 --backbone resnet18 --no-pretrained

Full run on a cloud GPU (Kaggle/Colab), pointing at the mounted dataset::

    export CANCER_DATASET_DIR=/kaggle/input/cbis-ddsm-breast-cancer-image-dataset
    export CANCER_MANIFEST_PATH=/kaggle/working/manifest.csv
    python train.py --config configs/default_config.yaml
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader

from src.data.augmentation import get_train_transforms
from src.data.dataset import MammographyDataset
from src.data.preprocessing import get_base_transforms
from src.data.splits import create_splits
from src.models.classifier import MammographyClassifier
from src.models.losses import build_criterion, compute_pos_weight
from src.models.trainer import Trainer
from src.utils.config import load_config
from src.utils.logging_setup import setup_logger
from src.utils.seeding import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the explainable cancer-detection baseline classifier."
    )
    parser.add_argument(
        "--config", default="configs/default_config.yaml", help="Path to YAML config."
    )
    parser.add_argument("--epochs", type=int, help="Override training.epochs.")
    parser.add_argument("--batch-size", type=int, help="Override data.batch_size.")
    parser.add_argument("--num-workers", type=int, help="Override data.num_workers.")
    parser.add_argument("--backbone", help="Override model.backbone.")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not load ImageNet weights (faster offline/smoke runs).",
    )
    parser.add_argument(
        "--freeze-layers",
        action="store_true",
        help="Freeze the backbone and train only the head.",
    )
    parser.add_argument("--device", help='Force device: "cpu" or "cuda".')
    parser.add_argument("--checkpoint-dir", help="Override training.checkpoint_dir.")
    parser.add_argument("--experiment-name", help="Override experiment_name.")
    return parser.parse_args()


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply CLI overrides onto the parsed config in place."""
    data_cfg = config.setdefault("data", {})
    model_cfg = config.setdefault("model", {})
    training_cfg = config.setdefault("training", {})

    if args.epochs is not None:
        training_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        data_cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        data_cfg["num_workers"] = args.num_workers
    if args.backbone:
        model_cfg["backbone"] = args.backbone
    if args.no_pretrained:
        model_cfg["pretrained"] = False
    if args.freeze_layers:
        model_cfg["freeze_layers"] = True
    if args.device:
        training_cfg["device"] = args.device
    if args.checkpoint_dir:
        training_cfg["checkpoint_dir"] = args.checkpoint_dir
    if args.experiment_name:
        config["experiment_name"] = args.experiment_name
    return config


def build_dataloaders(
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, DataLoader, Any]:
    """Create leakage-free, patient-wise stratified dataloaders.

    Returns:
        ``(train_loader, val_loader, test_loader, train_df)``.
    """
    data_cfg = config["data"]
    seed = int(config.get("seed", 42))
    use_cuda = torch.cuda.is_available()

    manifest_path = data_cfg["manifest_path"]
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Manifest not found at '{manifest_path}'. "
            "Run `python scripts/setup_dataset.py` (mock) or set "
            "CANCER_MANIFEST_PATH to a Kaggle-built manifest."
        )

    train_df, val_df, test_df = create_splits(
        manifest_path,
        test_size=float(data_cfg.get("test_split_ratio", 0.15)),
        val_size=float(data_cfg.get("val_split_ratio", 0.15)),
        seed=seed,
    )

    image_size = int(data_cfg.get("image_size", 224))
    dataset_dir = data_cfg["dataset_dir"]

    train_ds = MammographyDataset(
        train_df, root_dir=dataset_dir, transform=get_train_transforms(image_size)
    )
    val_ds = MammographyDataset(
        val_df, root_dir=dataset_dir, transform=get_base_transforms(image_size)
    )
    test_ds = MammographyDataset(
        test_df, root_dir=dataset_dir, transform=get_base_transforms(image_size)
    )

    common = dict(
        batch_size=int(data_cfg.get("batch_size", 32)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=use_cuda,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_loader, val_loader, test_loader, train_df


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)

    seed = int(config.get("seed", 42))
    seed_everything(seed)

    experiment_name = str(config.get("experiment_name", "experiment"))
    logger = setup_logger(
        name="train", log_file=os.path.join("logs", f"{experiment_name}.log")
    )
    logger.info("Experiment '%s' | seed=%d", experiment_name, seed)
    logger.info("Config: %s", config)

    train_loader, val_loader, test_loader, train_df = build_dataloaders(config)
    logger.info(
        "Splits - train=%d | val=%d | test=%d (patient-wise, stratified)",
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
    )

    model_cfg = config["model"]
    model = MammographyClassifier(
        backbone=str(model_cfg.get("backbone", "resnet50")),
        num_classes=int(model_cfg.get("num_classes", 2)),
        pretrained=bool(model_cfg.get("pretrained", True)),
        freeze_layers=bool(model_cfg.get("freeze_layers", False)),
    )

    pos_weight = compute_pos_weight(train_df["label"].tolist())
    criterion = build_criterion(config["training"], pos_weight=pos_weight)
    logger.info(
        "Loss: %s | pos_weight=%.4f",
        type(criterion).__name__,
        pos_weight,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        criterion=criterion,
        device=args.device,
        logger=logger,
    )
    history = trainer.train()

    logger.info(
        "Training complete. best %s=%.4f. Final train_loss=%.4f val_loss=%.4f",
        trainer.monitor,
        trainer.best_score,
        history["train_loss"][-1],
        history["val_loss"][-1],
    )
    logger.info("Best checkpoint: %s", os.path.join(trainer.checkpoint_dir, "best.pt"))


if __name__ == "__main__":
    main()
