# Phase 4 — Evaluation handoff

## Current position

- Phase 3 training was reported complete on Kaggle on 2026-10-26: ResNet-50,
  best epoch **8**, validation loss **0.0607**, validation accuracy **0.6976**;
  early stopping at epoch **13**.
- Phase 4 evaluation code and local synthetic-data checks are available.
- The local project contains neither the real checkpoint nor real test reports.
  Test ROC-AUC, PR-AUC, sensitivity and specificity remain **unmeasured here**.

## Resume on Kaggle

Use `notebooks/03_evaluation_kaggle.ipynb` after syncing this project's changes
to GitHub. Upload/import that notebook in Kaggle and run its cells in order.

Attach these inputs:

1. `awsaf49/cbis-ddsm-breast-cancer-image-dataset`.
2. The saved Phase 3 `best.pt` from your training notebook output or an uploaded
   Kaggle Dataset. If available, also attach the sibling `last.pt` for full loss
   history. The notebook lists candidates when several checkpoints are present.

The notebook handles both the old and newer `/kaggle/input/datasets/...` mount
layouts, uses explicit paths, checks that the checkpoint exists, evaluates only
the held-out test split, and packages results into `phase4_results.zip`.
Inference can use CPU or GPU; it does not require downloading ImageNet weights.

For an existing Kaggle checkout and a checkpoint still under `/kaggle/working`,
the equivalent cell is:

```python
from pathlib import Path
import subprocess, sys

repo = Path('/kaggle/working/Explainable_DL_Cancer_Detection')
subprocess.run([
    sys.executable, str(repo / 'evaluate.py'),
    '--checkpoint', '/kaggle/working/checkpoints/baseline_resnet50/best.pt',
    '--dataset-dir', '/kaggle/input/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset',
    '--manifest-path', str(repo / 'data/manifest.csv'),
    '--split', 'test', '--device', 'auto', '--num-workers', '2',
    '--output-dir', '/kaggle/working/phase4_results',
], cwd=repo, check=True)
```

After a session reset, reattach saved inputs and rerun the notebook from the top.
Files under `/kaggle/working`, including a cloned repo, may need to be restored.
The explicit `--manifest-path` supersedes stale `CANCER_MANIFEST_PATH` variables.

## Reproducibility and outputs

The checkpoint supplies the experiment name, seed, image size and split ratios.
The current config/CLI supplies dataset locations, batch size, worker count and
evaluation options. This prevents a changed YAML seed from turning training
patients into apparent test patients. Use the **original training manifest**:
legacy checkpoints store split settings, not a snapshot/hash of that manifest.
The baseline's original test split has **526 rows** (train 2536, validation 506).

Each `reports/<experiment>/<split>/` contains:

| File | Contents |
|---|---|
| `metrics.json` | Binary metrics, threshold, counts; undefined values are `null` |
| `classification_report.txt` | Aggregate metrics and per-class precision/recall/F1/support |
| `roc_curve.png`, `pr_curve.png` | Threshold-independent discrimination curves |
| `confusion_matrix.png` | Counts at the configured threshold (default 0.5) |
| `predictions.csv` | Patient/image identifiers, manifest row, label, malignant probability, predicted label |
| `run_info.json` | Checkpoint epoch (1-based), resolved config, manifest SHA-256, split and patient count |

Scores are computed per **manifest abnormality row**, not per patient; grouping
by patient is the leakage-prevention rule for splitting. Multiple abnormalities
can reference the same mammogram. `pr_auc` is **average precision**, not
trapezoidal integration of the PR curve. ROC-AUC and PR-AUC are reported as
undefined on single-class inputs. If tuning a decision threshold, select it
on validation data and keep it fixed for the test run.

`best.pt` contains history only up to the best epoch. Use `last.pt` from the same
run for all epochs; the notebook can generate the appropriate loss plot.

## Cross-validation

```bash
python evaluate.py --k-folds 5 --config configs/default_config.yaml
```

This trains five new models. Each outer fold reserves its own **inner validation
patients** for checkpoint selection/early stopping, then evaluates the restored
best weights on the untouched outer fold. Class weighting is calculated from
the training subset alone; each model has a recorded seed. CV stratifies one
row per patient, treating any malignant finding as a malignant patient.

`kfold_results.json` includes each fold's patient assignments, best epoch,
history, outer-fold metrics and mean/population-standard-deviation summary.
Checkpoints retain their exact split patient IDs for subsequent evaluation.
These CV results use the whole manifest under a different protocol; they are
reported separately from the original baseline's held-out test results.

## Completion evidence

Bring back the test report, prediction CSV, run metadata and plots, and record
the achieved values with the saved Kaggle output location. Only then mark
Phase 4 complete. Compare published results using matching units (patch/image/
patient), split protocols and subsets; the plan's rough AUC range is context,
not a measured result for this model.
