# Explainable Deep Learning Framework for Early Cancer Detection Using Medical Imaging

> **Batch 12** — Department of CSE (Data Science)

## Overview
This project builds an explainable deep learning pipeline for early breast cancer detection using mammography images (binary classification: Benign vs. Malignant). It emphasizes interpretability by integrating Grad-CAM to visualize model attention, aligning model focus with expert-annotated regions of interest (ROIs).

*Disclaimer: This is a research/educational prototype and is **not for clinical use**.*

## Dataset
- **Primary Dataset**: [CBIS-DDSM (Curated Breast Imaging Subset of DDSM)](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM)
- **Fallback Dataset**: [PatchCamelyon (PCam)](https://github.com/basveeling/pcam)
- *Note: Training is optimized for execution on cloud GPUs (e.g., Kaggle, Google Colab) to avoid local storage/compute constraints.*

## Project Structure
- `api/` - (Stretch Goal) FastAPI application for model deployment
- `configs/` - YAML configuration files for hyperparameter management
- `data/` - Raw datasets and generated manifests (git-ignored)
- `docs/` - Documentation, architecture decisions, and test plans
- `notebooks/` - Exploratory Data Analysis and visualization scripts
- `scripts/` - Utility scripts (e.g., data downloading/mocking)
- `src/` - Core source code (data loading, models, evaluation, explainability)
- `tests/` - Unit and integration tests

## Setup Instructions
1. **Clone the repository**
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Data Preparation**

   The full CBIS-DDSM archive is ~165 GB of raw DICOM, so we **never download it locally**. Instead, develop on mock data locally and run real training on a cloud GPU where the JPEG-converted dataset is mounted for free.

   **Local development (no download):**
   ```bash
   python scripts/setup_dataset.py   # generates mock data + manifest for pipeline/CI testing
   ```

   **Real data on Kaggle/Colab (no local storage):**
   Attach the Kaggle dataset `awsaf49/cbis-ddsm-breast-cancer-image-dataset`, then build the manifest from the mounted files:
   ```bash
   python scripts/build_manifest_kaggle.py \
       --dataset-dir /kaggle/input/cbis-ddsm-breast-cancer-image-dataset \
       --output /kaggle/working/manifest.csv
   ```
   Point the pipeline at the mounted data via environment variables (no code/config edits needed):
   ```bash
   export CANCER_DATASET_DIR=/kaggle/input/cbis-ddsm-breast-cancer-image-dataset
   export CANCER_MANIFEST_PATH=/kaggle/working/manifest.csv
   ```
   Commit the generated `manifest.csv` (it is small) so the data path is reproducible without re-mounting the full dataset. If compute/access is constrained, use the **PCam fallback** documented in `docs/dataset.md`.

## Training
Train the baseline benign-vs-malignant classifier with `train.py`. Hyperparameters
live in `configs/default_config.yaml` and can be overridden on the CLI.

```bash
# Local smoke test on mock data (CPU, no weight download)
python train.py --epochs 2 --batch-size 8 --backbone resnet18 --no-pretrained --num-workers 0

# Full run on a cloud GPU (Kaggle/Colab), pointing at the mounted dataset
export CANCER_DATASET_DIR=/kaggle/input/cbis-ddsm-breast-cancer-image-dataset
export CANCER_MANIFEST_PATH=/kaggle/working/manifest.csv
python train.py --config configs/default_config.yaml
```

- **Backbones:** `resnet18` · `resnet34` · `resnet50` · `efficientnet_b0`
  (`model.backbone`; pass `--freeze-layers` to train only the head).
- **Class imbalance** is handled via a **focal loss** (default) or class-weighted
  BCE — see the `training.loss` block in the config.
- **Outputs:** checkpoints in `checkpoints/<experiment_name>/{best,last}.pt`
  (each stores the model/optimizer state, config and metric history), logs in
  `logs/<experiment_name>.log`.

## Evaluation
Evaluate a trained checkpoint on the held-out test split (or `--split all` for
train/val/test):

```bash
python evaluate.py --checkpoint checkpoints/baseline_resnet50/best.pt --split all
```

This writes `reports/<experiment>/<split>/` containing `metrics.json`,
`classification_report.txt`, `roc_curve.png`, `pr_curve.png` and
`confusion_matrix.png`, plus `predictions.csv` and `run_info.json` for
reproducibility. Reported metrics: accuracy, precision,
sensitivity/recall, specificity, F1, **ROC-AUC** and **PR-AUC** (plus the
confusion-matrix counts). PR-AUC uses average precision. Saved checkpoint
settings determine the seed, image size and split ratios; paths can be
overridden with `--dataset-dir` and `--manifest-path` (these take precedence
over environment variables).

**Evaluate a saved checkpoint on Kaggle:** import
[`notebooks/03_evaluation_kaggle.ipynb`](notebooks/03_evaluation_kaggle.ipynb),
attach the CBIS-DDSM dataset and completed recovery run's `best.pt`, and run the cells in
order. It discovers mounted paths and packages reports as `phase4_results.zip`.
See [`docs/evaluation.md`](docs/evaluation.md) for the complete handoff and
completion evidence.

**Phase 4 completed:** the Kaggle recovery baseline achieved test **ROC-AUC
0.7841**, **PR-AUC (AP) 0.7208** and **accuracy 67.68%** on 526 abnormality rows
from 235 held-out patients. At threshold 0.5, sensitivity was **30.84%** and
specificity **92.95%**: 148 of 214 malignant rows were missed. Checkpoints and
plots are saved on Kaggle and downloaded locally. See
[`docs/results.md`](docs/results.md) for the full results and artifact paths.

Patient-wise stratified **K-fold cross-validation** (mean ± std stability):

```bash
python evaluate.py --k-folds 5
```

Each fold trains a fresh model, uses separate inner validation patients for
early stopping, and scores the restored best weights on the outer held-out
fold. Per-fold patient assignments and seeds are recorded in the results.

> For context, [Shen et al. (2019)](https://doi.org/10.1038/s41598-019-48995-4)
> reported CBIS-DDSM per-image ROC-AUC 0.88 (single model) and 0.91 (ensemble).
> Their subset, reporting unit and training protocol differ; see the comparison
> in `docs/results.md`. Scores are **not pass/fail gates** for evaluation tooling.

## Reproducibility
We ensure reproducibility via global seeding utilities in `src/utils/seeding.py` and strict hyperparameter tracking. Dataset locations are overridable via the `CANCER_DATASET_DIR` and `CANCER_MANIFEST_PATH` environment variables, so the identical codebase runs unchanged on a laptop (mock data) and on a cloud GPU (full dataset).
