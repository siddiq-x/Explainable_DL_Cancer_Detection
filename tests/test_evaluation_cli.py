"""End-to-end legacy-checkpoint evaluation from an unrelated working directory."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch
import yaml

from evaluate import parse_args
from src.data.dataset import MammographyDataset
from src.data.splits import create_splits
from src.models.classifier import MammographyClassifier


def test_cli_uses_legacy_checkpoint_split_and_explicit_paths(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (64, 64), (80, 80, 80)).save(image_root / "image.png")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([
        {"patient_id": f"P_{i:03d}", "image_path": "image.png", "roi_mask_path": "",
         "label": "MALIGNANT" if i % 2 else "BENIGN"}
        for i in range(40)
    ]).to_csv(manifest, index=False)
    saved_config = {
        "seed": 17, "experiment_name": "legacy_baseline",
        "data": {"image_size": 64, "test_split_ratio": 0.2, "val_split_ratio": 0.2},
        "model": {"backbone": "resnet18", "num_classes": 2},
    }
    model = MammographyClassifier(backbone="resnet18", pretrained=False)
    checkpoint = tmp_path / "best.pt"
    torch.save({"config": saved_config, "epoch": 7, "backbone": "resnet18",
                "num_classes": 2, "model_state_dict": model.state_dict()}, checkpoint)
    runtime = tmp_path / "runtime.yaml"
    runtime.write_text(yaml.safe_dump({
        "seed": 999, "experiment_name": "wrong_name",
        "data": {"image_size": 224, "test_split_ratio": 0.4, "val_split_ratio": 0.1},
    }))
    env = dict(os.environ, CANCER_DATASET_DIR="missing_mount", CANCER_MANIFEST_PATH="missing_manifest")
    script = Path(__file__).resolve().parents[1] / "evaluate.py"
    result = subprocess.run([
        sys.executable, str(script), "--config", str(runtime), "--checkpoint", str(checkpoint),
        "--dataset-dir", str(image_root), "--manifest-path", str(manifest),
        "--output-dir", str(tmp_path / "reports"), "--device", "cpu", "--num-workers", "0",
        "--batch-size", "4",
    ], cwd=tmp_path, env=env, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    report_dir = tmp_path / "reports/legacy_baseline/test"
    predictions = pd.read_csv(report_dir / "predictions.csv")
    _, _, expected_test = create_splits(str(manifest), test_size=0.2, val_size=0.2, seed=17)
    assert predictions["patient_id"].tolist() == expected_test["patient_id"].tolist()
    metrics = json.loads((report_dir / "metrics.json").read_text())
    assert metrics["n_samples"] == 8
    assert sum(metrics[key] for key in ("tn", "fp", "fn", "tp")) == 8
    info = json.loads((report_dir / "run_info.json").read_text())
    assert info["checkpoint_epoch"] == 8
    assert info["config"]["seed"] == 17
    assert info["config"]["data"]["image_size"] == 64
    assert len(info["manifest_sha256"]) == 64
    for filename in ("roc_curve.png", "pr_curve.png", "confusion_matrix.png", "classification_report.txt"):
        assert (report_dir / filename).stat().st_size > 0


@pytest.mark.parametrize("args", [
    [], ["--k-folds", "0"], ["--k-folds", "1"], ["--k-folds", "-2"],
    ["--checkpoint", "best.pt", "--k-folds", "5"],
    ["--checkpoint", "best.pt", "--threshold", "nan"],
])
def test_cli_rejects_invalid_modes_and_thresholds(args):
    with pytest.raises(SystemExit):
        parse_args(args)


@pytest.mark.parametrize("label", ["UNKNOWN", np.nan, ""])
def test_dataset_rejects_labels_instead_of_silently_scoring_them_as_benign(label):
    with pytest.raises(ValueError, match="labels"):
        MammographyDataset(pd.DataFrame({"label": [label]}), root_dir="unused")
