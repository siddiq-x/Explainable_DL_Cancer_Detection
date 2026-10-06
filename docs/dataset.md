# Dataset Documentation

## Primary Dataset: CBIS-DDSM
The primary dataset for this project is the **Curated Breast Imaging Subset of DDSM (CBIS-DDSM)**. To ease data loading and avoid handling raw DICOM files, we recommend using a JPEG/PNG converted version from Kaggle (e.g., `awsaf49/cbis-ddsm-breast-cancer-image-dataset`).

### Target Variable
The task is binary classification: **Benign vs. Malignant**. Cases labeled as `BENIGN_WITHOUT_CALLBACK` are merged into the `Benign` category.

### Data Organization
The dataset should be placed in `data/raw/`. A manifest file (`data/manifest.csv`) links each patient ID to their corresponding image path, ROI mask path, and label.

## Fallback Dataset: PatchCamelyon (PCam)
If you face issues with computing power, storage space, or downloading CBIS-DDSM, **PatchCamelyon (PCam)** is the official lightweight fallback dataset.

### Why PCam?
- **Small size**: Images are 96x96 pixels.
- **Easy to load**: Fits easily in memory or typical Kaggle/Colab environments.
- **Similar task**: Binary classification of metastatic tissue in histopathologic scans of lymph node sections.

### PCam Usage Instructions
1. PCam is available via Hugging Face Datasets or Kaggle (`keras/patchcamelyon`).
2. Alternatively, you can use `torchvision.datasets.PCAM` directly in PyTorch.
3. If using PCam, skip ROI-based Grad-CAM validation (as PCam does not provide pixel-level ROI masks) and use standard Grad-CAM visual verification instead.
