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
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from src.data.dataset import MammographyDataset
from src.data.preprocessing import get_base_transforms
from src.evaluation.evaluator import (
    build_evaluation_config,
    build_evaluation_splits,
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
PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint or run K-fold cross-validation."
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/default_config.yaml"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
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
    parser.add_argument("--dataset-dir", help="Image root; overrides CANCER_DATASET_DIR.")
    parser.add_argument("--manifest-path", help="Manifest CSV; overrides CANCER_MANIFEST_PATH.")
    parser.add_argument("--device", help='Force device: "cpu" or "cuda".')
    mode.add_argument(
        "--k-folds", type=int, default=0,
        help="Run patient-wise stratified K-fold CV (trains from scratch).",
    )
    args = parser.parse_args(argv)
    if args.checkpoint is None and args.k_folds < 2:
        parser.error("--k-folds must be at least 2.")
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        parser.error("--threshold must be in [0, 1].")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    if args.num_workers is not None and args.num_workers < 0:
        parser.error("--num-workers cannot be negative.")
    return args


def apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    data_cfg = config.setdefault("data", {})
    eval_cfg = config.setdefault("evaluation", {})
    if args.batch_size is not None:
        data_cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        data_cfg["num_workers"] = args.num_workers
    if args.dataset_dir:
        data_cfg["dataset_dir"] = args.dataset_dir
    if args.manifest_path:
        data_cfg["manifest_path"] = args.manifest_path
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


def main(argv=None) -> None:
    args = parse_args(argv)
    runtime_config = apply_overrides(load_config(args.config), args)
    model = checkpoint = None
    if args.k_folds:
        config = runtime_config
    else:
        model, checkpoint = load_model_from_checkpoint(args.checkpoint, device=args.device)
        config = build_evaluation_config(checkpoint, runtime_config)

    # Project-relative defaults also work when invoked from /kaggle/working.
    for key in ("dataset_dir", "manifest_path"):
        path = Path(config["data"][key]).expanduser()
        config["data"][key] = str(path if path.is_absolute() else PROJECT_ROOT / path)
    data_cfg = config["data"]
    manifest_path = data_cfg["manifest_path"]
    dataset_dir = data_cfg["dataset_dir"]
    if not Path(manifest_path).is_file():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}. Use --manifest-path to select the "
            "original training manifest (the baseline uses data/manifest.csv)."
        )
    if not Path(dataset_dir).is_dir():
        raise FileNotFoundError(f"Image root not found: {dataset_dir}. Set --dataset-dir to the mounted csv/jpeg dataset root.")

    eval_cfg = config.get("evaluation") or {}
    seed_everything(int(config["seed"]))
    experiment_name = str(config.get("experiment_name", "experiment"))
    output_root = str(eval_cfg.get("output_dir", "reports"))
    logger = setup_logger(
        name="evaluate", log_file=os.path.join("logs", f"evaluate_{experiment_name}.log")
    )

    # --- K-fold cross-validation mode (trains models from scratch) ---
    if args.k_folds:
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
    device = resolve_device(args.device)
    logger.info(
        "Loaded checkpoint %s (backbone=%s, epoch=%s, best_score=%s)",
        args.checkpoint, checkpoint.get("backbone"),
        checkpoint.get("epoch"), checkpoint.get("best_score"),
    )
    logger.info("Evaluating on device %s", device)
    logger.info(
        "Using checkpoint split settings: seed=%s, test=%s, val=%s, image_size=%s",
        config["seed"], data_cfg.get("test_split_ratio"), data_cfg.get("val_split_ratio"), data_cfg["image_size"],
    )

    split_frames = build_evaluation_splits(config)
    to_run = SPLITS if args.split == "all" else (args.split,)

    threshold = float(eval_cfg.get("threshold", 0.5))
    manifest_digest = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    for split in to_run:
        frame = split_frames[split]
        loader = _build_loader(config, frame)
        metrics, y_true, y_prob = evaluate_loader(
            model, loader, threshold=threshold, device=str(device)
        )
        logger.info("=== %s split (%d samples) ===\n%s", split, len(loader.dataset), format_metrics(metrics))

        artifact_dir = os.path.join(output_root, experiment_name, split)
        sample_info = frame.assign(manifest_row=frame.index)
        written = save_evaluation_artifacts(metrics, y_true, y_prob, artifact_dir, sample_info=sample_info)
        run_info = {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_epoch": int(checkpoint["epoch"]) + 1 if "epoch" in checkpoint else None,
            "split": split,
            "split_method": "saved patient assignments" if "split_patient_ids" in config else "reconstructed from original manifest, saved seed and split ratios",
            "manifest_sha256": manifest_digest,
            "n_patients": int(frame["patient_id"].nunique()),
            "metric_unit": "manifest row (abnormality), not patient",
            "positive_class": "MALIGNANT",
            "pr_auc_definition": "average_precision_score",
            "device": str(device),
            "config": config,
        }
        info_path = Path(artifact_dir) / "run_info.json"
        info_path.write_text(json.dumps(run_info, indent=2, allow_nan=False), encoding="utf-8")
        written["run_info"] = str(info_path)
        logger.info("Artifacts for '%s': %s", split, written)


if __name__ == "__main__":
    main()
