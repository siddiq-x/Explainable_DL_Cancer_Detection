# Baseline results — Phase 4

## Completed Kaggle run

- **Run:** [dazai0/cancer-baseline-phase4-recovery, version 1](https://www.kaggle.com/code/dazai0/cancer-baseline-phase4-recovery)
  (private; sign in as the owner to view outputs).
- **Experiment:** `baseline_resnet50_recovery_20261028`.
- **Executed:** 2026-10-28, 17:41:41–18:50:53 UTC; **69.2 minutes** on Tesla T4.
- **Training/evaluation code:** `d0385179a7054b7f1d984bc2bde699689cde1bd5`.
- **Runner:** `scripts/kaggle_train_evaluate.py`, tracked in commit `f337ec6`.
- **Model:** ImageNet-pretrained ResNet-50, single malignant logit, 224×224 RGB
  input, ImageNet normalization, batch size 32.
- **Training:** seed 42, focal loss (gamma 2, alpha 0.25), Adam (learning rate
  0.0001, weight decay 0.00001), ReduceLROnPlateau, gradient clipping 1.0.
- **Checkpoint selection:** minimum validation loss; best **epoch 12**;
  early stopping at **epoch 17** (patience 5, maximum 30 epochs).
- **Selected validation loss:** 0.057751489; validation accuracy 0.669960474;
  validation ROC-AUC 0.798225254.

This is a newly trained recovery model. The original October 26 run selected
epoch 8, but its checkpoint was unavailable through Kaggle's saved outputs.
The metrics below belong to the recovery model.

## Dataset and reporting protocol

CBIS-DDSM manifest: **3,568 abnormality rows from 1,566 patients**. The original
seed-42 patient-wise splits were retained. Exact patient assignments, manifest
and resolved configuration are saved with the run. Training, validation and
test patient sets are mutually disjoint.

Each scored observation is a **manifest abnormality row**. Several abnormalities
may reference the same mammogram, so these are neither unique-image scores nor
patient-level scores. Benign is class 0 and malignant is class 1. The model
receives the whole mammogram resized to 224×224. Evaluation uses deterministic
preprocessing for all three splits, including the training split.

All hard-label metrics below use the preselected **0.5 threshold**. ROC-AUC and
PR-AUC use continuous sigmoid scores; **PR-AUC means average precision (AP)**.

| Split | Rows | Patients | Malignant rows | Accuracy | Precision | Sensitivity | Specificity | F1 | ROC-AUC | PR-AUC (AP) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 2536 | 1096 | 1016 | 0.7839 | 0.9535 | 0.4843 | 0.9842 | 0.6423 | 0.9249 | 0.8996 |
| Validation | 506 | 235 | 227 | 0.6700 | 0.8571 | 0.3172 | 0.9570 | 0.4630 | 0.7982 | 0.7712 |
| **Test** | **526** | **235** | **214** | **0.6768** | **0.7500** | **0.3084** | **0.9295** | **0.4371** | **0.7841** | **0.7208** |

### Test confusion matrix

| Actual / Predicted | Benign | Malignant |
|---|---:|---:|
| Benign | 290 (TN) | 22 (FP) |
| Malignant | 148 (FN) | 66 (TP) |

### Interpretation

- ROC-AUC **0.7841** shows useful ranking discrimination. AP **0.7208** is above
  the test malignant prevalence of **0.4068**.
- At threshold 0.5, the model detects **66 of 214 malignant rows (30.8%)** and
  misses **148 (69.2%)**, while correctly classifying **290 of 312 benign rows
  (92.9%)**. The achieved sensitivity is well below the project's high-sensitivity
  objective; accuracy alone would obscure this limitation.
- Training ROC-AUC (0.9249) is higher than validation/test ROC-AUC (0.7982/0.7841),
  consistent with a generalization gap. The saved loss curve shows training loss
  continuing to fall after the selected validation-loss minimum.
- These are fixed-threshold baseline results. Future operating-point selection
  should use validation data and a prespecified sensitivity/specificity objective.
  This run does not establish probability calibration or cross-validation stability.

## Published CBIS-DDSM reference

Shen et al., *Deep Learning to Improve Breast Cancer Detection on Screening
Mammography*, Scientific Reports 9, 12495 (2019),
[doi:10.1038/s41598-019-48995-4](https://doi.org/10.1038/s41598-019-48995-4),
reported **per-image ROC-AUC 0.88** for their best single model and **0.91** for
a four-model ensemble on CBIS-DDSM (abstract and Results).

Their experiment used **2,478 images from 1,249 women**, with 1,903/199/376
training/validation/test images, patient-wise splitting, 1152×896 whole-image
inputs, and lesion-annotated patch pretraining before whole-image training.
Our **0.7841** is numerically lower, but the data subset, scoring unit, split,
resolution and training procedure differ. This is literature context rather
than a matched benchmark comparison. A protocol-matched study remains part of
the later validation work.

## Saved artifacts and verification

Downloaded output root (relative to the project):

```text
reports/kaggle_phase4_recovery/
  checkpoints/baseline_resnet50_recovery_20261028/{best,last}.pt
  reports/baseline_resnet50_recovery_20261028/
    train/, val/, test/
    training_history.json
    training_history.csv
    loss_curves.png
    run_summary.json
  phase4_results.zip
  run_summary.json
  run_config.yaml
  manifest.csv
  source_code.zip
  kaggle_run_source.py
  logs/
```

Each split directory contains `metrics.json`, `classification_report.txt`,
`predictions.csv`, `run_info.json`, `roc_curve.png`, `pr_curve.png` and
`confusion_matrix.png`. `best.pt` contains history through epoch 12; `last.pt`
contains all 17 epochs. Checkpoints and generated reports are external artifacts;
the tables and provenance in this document are tracked in Git.

**Verified locally on 2026-10-29:**

- Recomputed every metric from the three prediction CSVs (agreement within 1e-12).
- Matched row indices, image paths, patient IDs and labels to the original manifest.
- Checked all split assignments, patient isolation and complete patient coverage.
- Loaded both checkpoints and checked saved epochs, histories, config and best score.
- Matched the downloaded best-checkpoint SHA-256 to the runtime digest.
- Validated both ZIP archives and inspected the test ROC/PR/confusion plots and
  full loss curve.

SHA-256 values:

```text
manifest.csv
d75fe858bd861b503451d05fd3661057315fe11f6d154b7ef700aeb730741abd

best.pt
2466c50204b5b00f1b7b0baf7e73a3a3f810b1a30769e13e4757804ad8660442
```

**Phase 4 is complete.** Real five-fold training/stability evaluation (V-10) and
explanation validation remain part of Phase 6. Phase 5 (Grad-CAM) is next.
