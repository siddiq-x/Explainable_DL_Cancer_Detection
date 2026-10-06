# Implementation Plan
### Explainable Deep Learning Framework for Early Cancer Detection Using Medical Imaging

> **Batch 12** — Department of CSE (Data Science)
>
> **Scope (v2 — focused):** Single modality — **breast mammography**, dataset **CBIS-DDSM**, task **benign vs. malignant** (binary). Pipeline is dataset-agnostic; **PCam** is the lightweight fallback. Training targets **Colab/Kaggle GPU**.
>
> **MVP = Chunks 0–6.** Chunks 7–10 are **stretch goals** — build only if the MVP lands early. Chunk 11 (docs) is always done.
>
> **Disclaimer:** Research/educational prototype — **not for clinical use.**
>
> Each chunk below is self-contained. Complete and verify one before moving to the next.

---

## Chunk 0 — Dataset Selection & Acquisition  *(NEW — do this first)*
**Objective:** Secure a real, labelled dataset. The whole project hinges on this; a handful of sample images cannot train or validate a medical model.

> **Approach — no local 165 GB download.** The full CBIS-DDSM archive is raw DICOM (~165 GB). We do **not** download it locally. Instead: develop locally on mock data (`scripts/setup_dataset.py`), and build/consume the real dataset on a **cloud GPU (Kaggle/Colab)** where the ~6 GB JPEG mirror mounts read-only for free. Dataset paths are supplied via the `CANCER_DATASET_DIR` / `CANCER_MANIFEST_PATH` env vars, so the same code runs in both environments unchanged.

| # | Task | File / Location |
|---|------|-----------------|
| 1 | Attach **CBIS-DDSM** JPEG mirror on Kaggle (`awsaf49/cbis-ddsm-breast-cancer-image-dataset`) — mounted, not downloaded | Kaggle `/kaggle/input/...` |
| 2 | Build a **patient-wise manifest** CSV in the notebook: `patient_id, image_path, roi_mask_path, label`; merge `BENIGN_WITHOUT_CALLBACK` → benign. Handles DICOM→jpeg path mapping via `dicom_info.csv` | `scripts/build_manifest_kaggle.py` → `data/manifest.csv` |
| 3 | **Class-distribution analysis** (benign vs malignant counts, mass vs calcification) — printed by the builder and documented in the notebook | `notebooks/01_data_exploration.ipynb` |
| 4 | Verify ROI masks load and align with their images (builder's `--verify` reports missing files; needed later for Grad-CAM IoU) | script / notebook |
| 5 | Document **PCam fallback** path in case of access/compute issues | `docs/dataset.md` |

**Deliverables:** Mounted dataset, patient-wise manifest (committed), documented class distribution.

> **✅ Done (real-data run on Kaggle):** Manifest built & committed at `data/manifest.csv`.
> **1566 patients · 3568 abnormalities.** Class distribution (merged binary): **BENIGN 2111 / MALIGNANT 1457** (~59/41 — informs PR-AUC + class weighting in Chunks 3–4). Verification: **0/3568 images missing**, **1/3568 masks missing** (a known CBIS-DDSM source defect — that lesion is still usable for classification; it is simply excluded from the Grad-CAM-vs-ROI IoU check in Chunk 6).

---

## Chunk 1 — Project Setup & Environment
**Objective:** Establish a reproducible development environment and project scaffold.

| # | Task | File / Location |
|---|------|-----------------|
| 1 | `requirements.txt` with pinned versions — PyTorch, torchvision, NumPy, Pandas, OpenCV, Pillow, scikit-image, pydicom, scikit-learn, matplotlib, seaborn, pytorch-grad-cam, pyyaml, tqdm, pytest. *(SHAP, FastAPI, Uvicorn, Streamlit are stretch-only — Chunks 4/9.)* | `requirements.txt` |
| 2 | Default hyperparameters — LR, batch size, image size (224), epochs, backbone, **seed** | `configs/default_config.yaml` |
| 3 | Package init files across `src/` sub-packages | `src/*/__init__.py` |
| 4 | `.gitignore` — `data/`, `__pycache__`, checkpoints, `.env` | `.gitignore` |
| 5 | Global **seeding + logging** utility (reproducibility) | `src/utils/seeding.py`, `src/utils/logging_setup.py` |
| 6 | README with title, team, setup; validate clean install (note Colab/Kaggle option) | `README.md` |

**Deliverables:** Working environment, all packages installed, reproducible config skeleton.

---

## Chunk 2 — Data Loading & Preprocessing Pipeline
**Objective:** Build a pipeline to ingest mammograms and output model-ready tensors, split without leakage.

| # | Task | File / Location |
|---|------|-----------------|
| 1 | Custom PyTorch `Dataset` — reads image + label (and ROI mask path) from the manifest | `src/data/dataset.py` |
| 2 | Preprocessing — resize 224×224, intensity normalization, grayscale→3ch, DICOM/JPEG decode | `src/data/preprocessing.py` |
| 3 | Augmentations — flip, rotation, zoom, mild contrast (label-preserving) | `src/data/augmentation.py` |
| 4 | **Patient-wise, stratified** train/val/test split (no `patient_id` in two splits) | `src/data/splits.py` |
| 5 | YAML config loader | `src/utils/config.py` |
| 6 | Unit tests — tensor shape, values ∈ [0,1], label integrity, **no patient leakage across splits** | `tests/test_data_pipeline.py` |

**Deliverables:** `DataLoader` yielding `(image_tensor, label)` batches, leakage-free splits, tests passing.

---

## Chunk 3 — Baseline Classification Model
**Objective:** Train a functional benign-vs-malignant classifier that handles class imbalance.

| # | Task | File / Location |
|---|------|-----------------|
| 1 | Model class — pre-trained ResNet-50 / EfficientNet-B0 + custom head; layer-freezing support | `src/models/classifier.py` |
| 2 | Training loop — **class weighting / Focal Loss**, Adam, LR scheduler, early stopping, checkpointing; log config + seed | `src/models/trainer.py` |
| 3 | CLI — load config, build loaders, train, log metrics | `train.py` |
| 4 | Unit tests — output shape, **tiny-batch overfitting** sanity check | `tests/test_model.py` |

**Deliverables:** Trained baseline checkpoint, training/validation loss curves.

---

## Chunk 4 — Evaluation & Metrics Module
**Objective:** Rigorously evaluate using clinical-grade metrics suited to imbalanced data.

| # | Task | File / Location |
|---|------|-----------------|
| 1 | Metrics — Accuracy, Precision, Recall (Sensitivity), Specificity, F1, **ROC-AUC + PR-AUC**, confusion matrix | `src/evaluation/metrics.py` |
| 2 | Evaluation runner — load checkpoint, test-set inference, classification report + plots (ROC, PR, confusion matrix) | `src/evaluation/evaluator.py` |
| 3 | CLI entry point | `evaluate.py` |
| 4 | K-Fold cross-validation support | `src/models/trainer.py` |
| 5 | Unit tests with known inputs/outputs | `tests/test_metrics.py` |

> **Metrics are targets, not pass/fail gates.** Report achieved scores **against published CBIS-DDSM baselines** (typical ROC-AUC ≈ 0.80–0.87). A score below a round number is not a project failure — context vs. baseline is what matters.

**Deliverables:** Classification report, ROC + PR curves, confusion matrix.

---

## Chunk 5 — Explainability Engine  *(moved up — must exist before V&V)*
**Objective:** Generate interpretable visual explanations for every prediction.

| # | Task | File / Location |
|---|------|-----------------|
| 1 | Grad-CAM — hook last conv layer, generate class-activation heatmap, overlay on mammogram | `src/explainability/gradcam.py` |
| 2 | Grad-CAM++ (and/or Score-CAM) variant for sharper localization | `src/explainability/gradcam.py` |
| 3 | CLI — image + checkpoint → prediction + heatmap PNG | `explain.py` |
| 4 | Tests — heatmap shape matches input, non-zero output | `tests/test_explainability.py` |
| 5 | *(Optional)* SHAP — only if tabular patient risk factors are added later | `src/explainability/shap_explainer.py` |

**Deliverables:** For any input → prediction label + confidence + Grad-CAM overlay (PNG).

---

## Chunk 6 — Verification & Validation
**Objective:** Verify implementation correctness and validate against clinical-style requirements — **quantitatively, using the dataset's own expert annotations** (no dependency on live radiologists).

### 6.1 Verification — *"Are we building the system right?"*

| # | Task | File / Location |
|---|------|-----------------|
| 1 | **Requirement-to-Module Traceability Matrix** | `docs/traceability_matrix.md` |
| 2 | **Unit Testing** (one file per module): preprocessing, model, explainability, metrics | `tests/test_*.py` |
| 3 | **Integration Testing** — end-to-end: raw image → preprocess → model → Grad-CAM → output | `tests/test_integration.py` |
| 4 | **Data-Quality Checks** — hash-based duplicate detection, corrupted-file detection, missing-label validation | `src/data/quality_checks.py` |

### 6.2 Validation — *"Are we building the right system?"*

| # | Task | File / Location |
|---|------|-----------------|
| 1 | **Model Validation** — patient-wise split, K-Fold CV (k=5); report Accuracy/Sensitivity/Specificity/F1/ROC-AUC/PR-AUC **vs. published baselines** | `src/evaluation/validation.py` |
| 2 | **Explanation Validation (key differentiator)** — compute **IoU / overlap of Grad-CAM heatmaps vs. ground-truth ROI masks**; report mean IoU | `src/evaluation/explanation_iou.py` |
| 3 | *(Optional)* Qualitative clinician review if available — **not** a required gate | `docs/clinical_review.md` |

### 6.3 Validation Metrics & Test Plan

| Test ID | Type | Module | Input | Expected Output | Target (not hard gate) |
|---------|------|--------|-------|-----------------|------------------------|
| V-01 | Unit | Preprocessing | Raw DICOM/JPEG | Normalized tensor (224×224) | Shape match, values ∈ [0,1] |
| V-02 | Unit | Augmentation | Image + label | Augmented image | Label integrity preserved |
| V-03 | Unit | Split | Manifest | Train/val/test | **No patient_id in two splits** |
| V-04 | Unit | Classification | Image tensor | Label + probability | Probability ∈ [0,1] |
| V-05 | Unit | Grad-CAM | Image + prediction | Heatmap overlay | Non-empty, matching size |
| V-06 | Integration | Full Pipeline | Raw image | Prediction + heatmap | No errors, valid output |
| V-07 | Data Quality | Duplicate/Corruption/Missing-label | Dataset + manifest | Issue lists | All issues flagged |
| V-08 | Model | Classification | Test set | ROC-AUC, PR-AUC, F1 | Report **vs. published baseline** |
| V-09 | Model | Sensitivity | Malignant cases | Sensitivity | Prioritize high sensitivity (clinical) |
| V-10 | Model | K-Fold Stability | Full dataset (k=5) | Mean ± std AUC | Low variance (std small) |
| V-11 | Explainability | Grad-CAM vs ROI mask | Images + masks | Mean IoU / overlap | Heatmap concentrates on lesion |
| V-12 | *(Optional)* Clinical | Expert review | Heatmap demo | Qualitative feedback | Non-blocking |

**Validation Flow:**
```
Requirements → Verification → Model Validation → Explanation-vs-ROI Validation → (optional) Expert Review → Validated System
```

**Deliverables:** All tests passing, traceability matrix, data-quality report, validation report (metrics vs. baseline + mean Grad-CAM IoU).

---
---

# Stretch Goals (build only if MVP is complete and stable)

## Chunk 7 — Lesion Localization *(Stretch)*
**Objective:** Localize suspicious regions beyond binary classification.

| # | Task | File / Location |
|---|------|-----------------|
| 1 | Localization head — U-Net segmentation decoder (CBIS-DDSM masks available) or bbox regression | `src/models/localization.py` |
| 2 | Multi-task training (classification + localization) | `src/models/trainer.py` |
| 3 | Localization metrics — IoU, Dice | `src/evaluation/localization_metrics.py` |
| 4 | Visualization — predicted masks/boxes overlaid | Notebooks / scripts |

**Deliverables:** Class label + localized region, with IoU/Dice scores.

---

## Chunk 8 — Risk Prediction & Calibration *(Stretch)*
**Objective:** Produce calibrated risk scores with clinical risk levels.

| # | Task | File / Location |
|---|------|-----------------|
| 1 | Risk scoring — map probabilities to Low/Medium/High with configurable thresholds | `src/models/risk_predictor.py` |
| 2 | **Calibration** — reliability diagram, temperature / Platt scaling | `src/evaluation/risk_calibration.py` |
| 3 | Per-patient risk report (JSON) — score, level, heatmap path | Output artifacts |

**Deliverables:** Risk JSON per patient, calibration curve.

---

## Chunk 9 — API & Clinical Dashboard *(Stretch)*
**Objective:** Expose the model through a web interface.

| # | Task | File / Location |
|---|------|-----------------|
| 1 | FastAPI — `POST /predict` (upload → prediction + heatmap), `GET /health` | `api/main.py` |
| 2 | Pydantic request/response schemas | `api/schemas.py` |
| 3 | Inference module — model singleton, preprocessing, prediction | `api/inference.py` |
| 4 | Streamlit frontend — upload + side-by-side prediction/risk/Grad-CAM | `frontend/` |
| 5 | API endpoint tests | `tests/test_api.py` |

**Deliverables:** Running web app — upload an image, receive diagnosis + explanation.

---

## Chunk 10 — Continuous Monitoring & Feedback Loop *(Stretch)*
**Objective:** Track performance over time and enable iterative improvement.

| # | Task | File / Location |
|---|------|-----------------|
| 1 | Structured prediction logging — timestamp, image ID, prediction, confidence | `src/utils/logging_setup.py` |
| 2 | Feedback endpoint — `POST /feedback`; SQLite/CSV storage | `api/main.py` |
| 3 | Retraining script — aggregate feedback, fine-tune, compare new vs old on held-out set | `src/models/retrainer.py` |
| 4 | Performance-drift tracking | Scripts / notebooks |

**Deliverables:** Feedback storage, retraining pipeline, comparison report.

---

## Chunk 11 — Documentation & Final Packaging
**Objective:** Prepare for submission and deployment.

| # | Task | File / Location |
|---|------|-----------------|
| 1 | Complete README — overview, architecture diagram, setup, training, evaluation, sample results, **disclaimer** | `README.md` |
| 2 | Methodology documentation | `docs/methodology.md` |
| 3 | Results — tables, plots (incl. Grad-CAM IoU), analysis | `docs/results.md` |
| 4 | Dataset & ethics note (de-identification, not-for-clinical-use) | `docs/dataset.md` |
| 5 | Code quality — docstrings, type hints, dead-code removal | All source files |
| 6 | Dependency freeze + convenience scripts | `requirements.txt`, `scripts/` |

**Deliverables:** Polished, documented, submission-ready project.

---

## Progress Tracker

| Chunk | Title | Scope | Status |
|:-----:|-------|:-----:|:------:|
| 0 | Dataset Selection & Acquisition | MVP | ✅ |
| 1 | Project Setup & Environment | MVP | ✅ |
| 2 | Data Loading & Preprocessing | MVP | ✅ |
| 3 | Baseline Classification Model | MVP | ⬜ |
| 4 | Evaluation & Metrics | MVP | ⬜ |
| 5 | Explainability Engine | MVP | ⬜ |
| 6 | Verification & Validation | MVP | ⬜ |
| 7 | Lesion Localization | Stretch | ⬜ |
| 8 | Risk Prediction & Calibration | Stretch | ⬜ |
| 9 | API & Clinical Dashboard | Stretch | ⬜ |
| 10 | Continuous Monitoring | Stretch | ⬜ |
| 11 | Documentation & Packaging | Always | ⬜ |

*Legend: ✅ done · 🟡 in progress (tooling ready; awaiting real-data run on Kaggle) · ⬜ not started.*
