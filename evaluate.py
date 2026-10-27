"""CLI entry point: evaluate a trained checkpoint, or run K-fold cross-validation.

Examples
--------
Evaluate a checkpoint on the held-out test split::

    python evaluate.py --checkpoint checkpoints/baseline_resnet50/best.pt

Evaluate on all splits and write reports to a custom directory::

    python evaluate.py --checkpoint checkpoints/baseline_resnet50/best.pt \
        --split all --output-dir reports

Patient-wise stratified 5-fold cross-validation (trains models from scratch)::

    python evaluate.py --k-folds 5 --output-dir reports

On a cloud GPU, point the config at the mounted dataset first::

    export CANCER_DATASET_DIR=/kaggle/input/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from src.data.dataset import MammographyDataset
from src.data.preprocessing import get_base_transforms
from src.data.splits import create_splits
from src.evaluation.evaluator import (
    evaluate_loader,
    load_model_from_checkpoint,
    resolve_device,
    save_evaluation_artifacts,
)
from src.evaluation.metrics import format_metrics
from src.models.classifier import MammographyClassifier
from src.models.trainer import cross_validate
from src.utils.config import load_config
from src.utils.logging_setup import setup_logger
from src.utils.seeding import seed_everything

SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint or run K-fold cross-validation."
    )
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument(
        "--checkpoint",
        help="Checkpoint to evaluate (required unless --k-folds is used).",
    )
    parser.add_argument(
        "--split", default="test", choices=(*SPLITS, "all"),
        help="Which split(s) to evaluate (default: test).",
    )
    parser.add_argument("--threshold", type=float, help="Decision threshold.")
    parser.add_argument("--output-dir", help="Where reports/plots are written.")
    parser.add_argument("--batch-size", type=int, help="Override data.batch_size.")
    parser.add_argument("--num-workers", type=int, help="Override data.num_workers.")
    parser.add_argument("--device", help='Force device: "cpu" or "cuda".')
    parser.add_argument(
        "--k-folds", type=int, default=0,
        help="Run patient-wise stratified K-fold CV (trains from scratch).",
    )
    return parser.parse_args()


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    data_cfg = config.setdefault("data", {})
    eval_cfg = config.setdefault("evaluation", {})
    if args.batch_size is not None:
        data_cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        data_cfg["num_workers"] = args.num_workers
    if args.threshold is not None:
        eval_cfg["threshold"] = args.threshold
    if args.output_dir:
        eval_cfg["output_dir"] = args.output_dir
    return config


def _build_loader(config: Dict[str, Any], dataframe, shuffle: bool = False) -> DataLoader:
    data_cfg = config["data"]
    dataset = MammographyDataset(
        dataframe,
        root_dir=data_cfg["dataset_dir"],
        transform=get_base_transforms(int(data_cfg.get("image_size", 224))),
    )
    return DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 32)),
        shuffle=shuffle,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )


def _model_factory(config: Dict[str, Any]):
    model_cfg = config.get("model") or {}

    def factory() -> MammographyClassifier:
        return MammographyClassifier(
            backbone=str(model_cfg.get("backbone", "resnet50")),
            num_classes=int(model_cfg.get("num_classes", 2)),
            pretrained=bool(model_cfg.get("pretrained", True)),
            freeze_layers=bool(model_cfg.get("freeze_layers", False)),
        )

    return factory


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    eval_cfg = config.get("evaluation") or {}

    seed_everything(int(config.get("seed", 42)))
    experiment_name = str(config.get("experiment_name", "experiment"))
    output_root = str(eval_cfg.get("output_dir", "reports"))
    logger = setup_logger(
        name="evaluate", log_file=os.path.join("logs", f"evaluate_{experiment_name}.log")
    )

    data_cfg = config["data"]
    manifest_path = data_cfg["manifest_path"]
    dataset_dir = data_cfg["dataset_dir"]

    # --- K-fold cross-validation mode (trains models from scratch) ---
    if args.k_folds and args.k_folds > 1:
        logger.info("Running %d-fold cross-validation", args.k_folds)
        result = cross_validate(
            config=config,
            model_factory=_model_factory(config),
            manifest_path=manifest_path,
            dataset_dir=dataset_dir,
            k=int(args.k_folds),
            seed=int(config.get("seed", 42)),
            device=args.device,
            logger=logger,
            output_dir=os.path.join(output_root, f"{experiment_name}_kfold"),
        )
        summary = result["summary"]
        logger.info(
            "K-fold summary: ROC-AUC %.4f±%.4f | PR-AUC %.4f±%.4f | F1 %.4f±%.4f",
            summary["roc_auc_mean"], summary["roc_auc_std"],
            summary["pr_auc_mean"], summary["pr_auc_std"],
            summary["f1_mean"], summary["f1_std"],
        )
        return

    # --- Single-checkpoint evaluation mode ---
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required unless --k-folds is used.")

    device = resolve_device(args.device)
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device=args.device)
    logger.info(
        "Loaded checkpoint %s (backbone=%s, epoch=%s, best_score=%s)",
        args.checkpoint, checkpoint.get("backbone"),
        checkpoint.get("epoch"), checkpoint.get("best_score"),
    )
    logger.info("Evaluating on device %s", device)

    train_df, val_df, test_df = create_splits(
        manifest_path,
        test_size=float(data_cfg.get("test_split_ratio", 0.15)),
        val_size=float(data_cfg.get("val_split_ratio", 0.15)),
        seed=int(config.get("seed", 42)),
    )
    split_frames = {"train": train_df, "val": val_df, "test": test_df}
    to_run = SPLITS if args.split == "all" else (args.split,)

    threshold = float(eval_cfg.get("threshold", 0.5))
    for split in to_run:
        loader = _build_loader(config, split_frames[split])
        metrics, y_true, y_prob = evaluate_loader(
            model, loader, threshold=threshold, device=str(device)
        )
        logger.info("=== %s split (%d samples) ===\n%s", split, len(loader.dataset), format_metrics(metrics))

        artifact_dir = os.path.join(output_root, experiment_name, split)
        written = save_evaluation_artifacts(metrics, y_true, y_prob, artifact_dir)
        logger.info("Artifacts for '%s': %s", split, written)


if __name__ == "__main__":
    main()
