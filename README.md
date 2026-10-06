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

## Reproducibility
We ensure reproducibility via global seeding utilities in `src/utils/seeding.py` and strict hyperparameter tracking. Dataset locations are overridable via the `CANCER_DATASET_DIR` and `CANCER_MANIFEST_PATH` environment variables, so the identical codebase runs unchanged on a laptop (mock data) and on a cloud GPU (full dataset).
