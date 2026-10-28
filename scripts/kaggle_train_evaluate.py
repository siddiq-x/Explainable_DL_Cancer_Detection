"""Standalone Kaggle batch entry point: recover the baseline and evaluate it.

Attach awsaf49/cbis-ddsm-breast-cancer-image-dataset, enable a T4 GPU and
Internet, and run this file as a Kaggle script. Outputs under /kaggle/working
are persisted by a completed batch version (Save & Run All / kernels push).
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
from importlib.util import find_spec
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


REPOSITORY = "https://github.com/siddiq-x/Explainable_DL_Cancer_Detection.git"
CODE_COMMIT = "d0385179a7054b7f1d984bc2bde699689cde1bd5"
EXPERIMENT = "baseline_resnet50_recovery_20261028"
WORK = Path("/kaggle/working")
REPO = Path("/kaggle/temp/Explainable_DL_Cancer_Detection")


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(command, cwd=WORK) -> None:
    print("Running:", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, check=True)


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    summary = {
        "status": "starting",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "experiment_name": EXPERIMENT,
        "source_commit": CODE_COMMIT,
        "run_kind": "fresh training to replace an unavailable original checkpoint",
    }
    write_json(WORK / "run_summary.json", summary)
    try:
        os.environ["PYTHONUNBUFFERED"] = "1"
        packages = {
            "numpy": "numpy", "pandas": "pandas", "PIL": "Pillow",
            "sklearn": "scikit-learn", "yaml": "PyYAML", "matplotlib": "matplotlib",
        }
        missing = [package for module, package in packages.items() if find_spec(module) is None]
        if missing:
            run([sys.executable, "-m", "pip", "install", *missing])

        import pandas as pd
        import torch
        import torchvision
        import yaml

        if not torch.cuda.is_available():
            raise RuntimeError("This training run requires a Kaggle GPU accelerator.")
        summary["environment"] = {
            "python": sys.version,
            "torch": torch.__version__, "torchvision": torchvision.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            **{package: version(package) for package in packages.values()},
        }
        print(json.dumps(summary["environment"], indent=2), flush=True)

        roots = sorted({
            path.parent.parent for path in Path("/kaggle/input").rglob("dicom_info.csv")
            if path.parent.name == "csv" and (path.parent.parent / "jpeg").is_dir()
        })
        if len(roots) != 1:
            raise RuntimeError(f"Expected one attached CBIS-DDSM JPEG dataset; found {roots}")
        dataset = roots[0]
        REPO.parent.mkdir(parents=True, exist_ok=True)
        if not (REPO / ".git").is_dir():
            run(["git", "clone", REPOSITORY, REPO])
        run(["git", "checkout", "--detach", CODE_COMMIT], cwd=REPO)
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        if actual != CODE_COMMIT:
            raise RuntimeError("Source checkout does not match the pinned commit.")
        run(["git", "archive", "--format=zip", f"--output={WORK / 'source_code.zip'}", "HEAD"], cwd=REPO)
        shutil.copy2(Path(__file__), WORK / "kaggle_run_source.py")
        manifest = WORK / "manifest.csv"
        shutil.copy2(REPO / "data/manifest.csv", manifest)
        frame = pd.read_csv(manifest)
        missing_images = [path for path in frame["image_path"].unique() if not (dataset / path).is_file()]
        if missing_images:
            raise FileNotFoundError(f"Missing {len(missing_images)} images; first: {missing_images[:3]}")
        if len(frame) != 3568 or frame["patient_id"].nunique() != 1566:
            raise RuntimeError("The manifest does not match the original baseline dataset.")

        sys.path.insert(0, str(REPO))
        from src.data.splits import create_splits

        config = yaml.safe_load((REPO / "configs/default_config.yaml").read_text())
        config["experiment_name"] = EXPERIMENT
        config["data"].update(dataset_dir=str(dataset), manifest_path=str(manifest), num_workers=4)
        config["training"].update(checkpoint_dir=str(WORK / "checkpoints"), device="cuda")
        config["evaluation"].update(output_dir=str(WORK / "reports"), threshold=0.5)
        frames = dict(zip(("train", "val", "test"), create_splits(
            str(manifest), test_size=config["data"]["test_split_ratio"],
            val_size=config["data"]["val_split_ratio"], seed=config["seed"],
        )))
        counts = {split: len(part) for split, part in frames.items()}
        if counts != {"train": 2536, "val": 506, "test": 526}:
            raise RuntimeError(f"Unexpected baseline split counts: {counts}")
        config["split_patient_ids"] = {
            split: sorted(part["patient_id"].unique().tolist()) for split, part in frames.items()
        }
        config["source_commit"] = CODE_COMMIT
        config["manifest_sha256"] = sha256(manifest)
        config_path = WORK / "run_config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        os.environ["CANCER_DATASET_DIR"] = str(dataset)
        os.environ["CANCER_MANIFEST_PATH"] = str(manifest)
        summary.update(status="training", split_counts=counts, manifest_sha256=sha256(manifest))
        write_json(WORK / "run_summary.json", summary)

        run([sys.executable, "-u", REPO / "train.py", "--config", config_path, "--device", "cuda"])
        checkpoint_dir = WORK / "checkpoints" / EXPERIMENT
        best_path = checkpoint_dir / "best.pt"
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        last = torch.load(checkpoint_dir / "last.pt", map_location="cpu", weights_only=False)
        history = last["history"]
        best_index = int(best["epoch"])
        summary.update(
            status="evaluating", best_epoch=best_index + 1,
            epochs_trained=len(history["train_loss"]),
            best_val_loss=history["val_loss"][best_index],
            best_val_acc=history["val_acc"][best_index],
            best_val_auc=history["val_auc"][best_index],
            best_checkpoint_sha256=sha256(best_path),
        )
        del best, last
        write_json(WORK / "run_summary.json", summary)
        reports = WORK / "reports" / EXPERIMENT
        reports.mkdir(parents=True, exist_ok=True)
        write_json(reports / "training_history.json", history)
        pd.DataFrame(history).rename_axis("epoch_index").to_csv(reports / "training_history.csv")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        epochs = range(1, len(history["train_loss"]) + 1)
        for key, label in (("train_loss", "Train"), ("val_loss", "Validation")):
            ax.plot(epochs, history[key], label=label)
        ax.axvline(summary["best_epoch"], linestyle="--", color="gray", label="Best checkpoint")
        ax.set(xlabel="Epoch", ylabel="Focal loss", title=EXPERIMENT)
        ax.legend()
        fig.savefig(reports / "loss_curves.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        run([
            sys.executable, "-u", REPO / "evaluate.py", "--config", config_path,
            "--checkpoint", best_path, "--dataset-dir", dataset, "--manifest-path", manifest,
            "--split", "all", "--threshold", "0.5", "--device", "cuda",
            "--num-workers", "4", "--output-dir", WORK / "reports",
        ])
        summary["test_metrics"] = json.loads((reports / "test/metrics.json").read_text())
        summary["status"] = "complete"
        summary["completed_at"] = datetime.now(timezone.utc).isoformat()
        summary["elapsed_seconds"] = round(time.monotonic() - started, 1)
        write_json(WORK / "run_summary.json", summary)
        shutil.copy2(WORK / "run_summary.json", reports / "run_summary.json")
        shutil.make_archive(str(WORK / "phase4_results"), "zip", root_dir=WORK / "reports")
        print("COMPLETE:", json.dumps(summary, indent=2), flush=True)
    except Exception as error:
        summary.update(status="failed", error=str(error), elapsed_seconds=round(time.monotonic() - started, 1))
        write_json(WORK / "run_summary.json", summary)
        (WORK / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
