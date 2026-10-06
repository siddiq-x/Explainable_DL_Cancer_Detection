"""
Build a patient-wise manifest from the real CBIS-DDSM dataset.

This script is designed to run **in a Kaggle/Colab notebook** where the full
JPEG-converted dataset is mounted read-only — so you never download 165 GB (or
even the ~6 GB mirror) to your laptop.

Target dataset (Kaggle):
    awsaf49/cbis-ddsm-breast-cancer-image-dataset

That dataset ships:
    <root>/csv/calc_case_description_train_set.csv
    <root>/csv/calc_case_description_test_set.csv
    <root>/csv/mass_case_description_train_set.csv
    <root>/csv/mass_case_description_test_set.csv
    <root>/csv/dicom_info.csv          # maps original DICOM paths -> jpeg paths
    <root>/jpeg/<SeriesUID>/<n>.jpg    # the actual images & ROI masks

The well-known pain point: the description CSVs reference *original DICOM* paths
(e.g. "Mass-Training_P_00001_LEFT_CC/.../1-1.dcm"), NOT the jpeg paths. We use
dicom_info.csv to translate those into the real jpeg file locations.

Output: a manifest CSV with columns
    patient_id, image_path, roi_mask_path, original_label, label, abnormality_type
where image_path / roi_mask_path are RELATIVE to --dataset-dir, matching how
`src/data/dataset.py` joins `root_dir` + path.

Usage (inside a Kaggle notebook):
    python scripts/build_manifest_kaggle.py \
        --dataset-dir /kaggle/input/cbis-ddsm-breast-cancer-image-dataset \
        --output /kaggle/working/manifest.csv

Then point the pipeline at it:
    export CANCER_DATASET_DIR=/kaggle/input/cbis-ddsm-breast-cancer-image-dataset
    export CANCER_MANIFEST_PATH=/kaggle/working/manifest.csv

Commit the resulting manifest.csv to the repo (it is small) so the data path is
reproducible without re-attaching the full dataset.
"""

import argparse
import os
import sys
import pandas as pd


# Description CSVs we merge. Each row is one abnormality (a patient may have >1).
DESCRIPTION_CSVS = [
    ("calc", "calc_case_description_train_set.csv"),
    ("calc", "calc_case_description_test_set.csv"),
    ("mass", "mass_case_description_train_set.csv"),
    ("mass", "mass_case_description_test_set.csv"),
]

# Column names in the description CSVs (CBIS-DDSM standard, note the spaces).
COL_PATIENT = "patient_id"
COL_PATHOLOGY = "pathology"
COL_IMAGE = "image file path"
COL_ROI_MASK = "ROI mask file path"
COL_ABNORMALITY = "abnormality type"


def _segments(path: str) -> list:
    """Split a path into non-empty segments, normalizing separators."""
    if not isinstance(path, str):
        return []
    p = path.strip().replace("\\", "/").strip("/")
    return [seg for seg in p.split("/") if seg]


def _to_rel_jpeg(jpeg_path: str) -> str:
    """Normalize a jpeg path to be relative to dataset_dir, using '/' separators.

    Strips any leading folders up to and including 'jpeg/' so the result is
    'jpeg/<uid>/<file>.jpg' regardless of mirror-specific prefixes.
    """
    rel = jpeg_path.replace("\\", "/").strip("/")
    if "jpeg/" in rel:
        rel = "jpeg/" + rel.split("jpeg/", 1)[1]
    elif not rel.startswith("jpeg/"):
        rel = "jpeg/" + rel
    return rel


def _build_dicom_lookup(dataset_dir: str) -> dict:
    """Map DICOM-path segment (SeriesInstanceUID) -> relative jpeg path.

    Uses dicom_info.csv when present. Returns {} if unavailable, in which case
    we fall back to using the description-CSV paths directly.
    """
    dicom_info_path = os.path.join(dataset_dir, "csv", "dicom_info.csv")
    if not os.path.exists(dicom_info_path):
        print(f"[warn] dicom_info.csv not found at {dicom_info_path}; "
              "will use raw CSV paths (may need manual adjustment).")
        return {}

    info = pd.read_csv(dicom_info_path)
    # dicom_info.csv typically has 'image_path' (jpeg location, e.g.
    # 'CBIS-DDSM/jpeg/<uid>/1-1.jpg') and a 'SeriesInstanceUID' column.
    jpeg_col = None
    for candidate in ("image_path", "image file path", "file_path"):
        if candidate in info.columns:
            jpeg_col = candidate
            break
    if jpeg_col is None:
        print(f"[warn] no jpeg path column in dicom_info.csv "
              f"(have: {list(info.columns)}); falling back to raw paths.")
        return {}

    lookup = {}
    for _, row in info.iterrows():
        jpeg_path = row[jpeg_col]
        if not isinstance(jpeg_path, str):
            continue
        rel = _to_rel_jpeg(jpeg_path)
        # Index by the series UID column (most reliable join key)...
        if "SeriesInstanceUID" in info.columns and isinstance(row["SeriesInstanceUID"], str):
            lookup[row["SeriesInstanceUID"].strip()] = rel
        # ...and by every path segment, so a description-CSV path that contains
        # the UID folder anywhere will match regardless of surrounding folders.
        for seg in _segments(jpeg_path):
            lookup.setdefault(seg, rel)
    print(f"[info] built dicom_info lookup with {len(lookup)} entries.")
    return lookup


def _resolve(path: str, lookup: dict, dataset_dir: str) -> str:
    """Resolve a description-CSV path to a relative jpeg path under dataset_dir."""
    if not isinstance(path, str) or not path.strip():
        return ""
    # Try each segment of the DICOM path against the lookup; the SeriesInstanceUID
    # folder is the stable component shared with dicom_info.csv.
    for seg in _segments(path):
        if seg in lookup:
            return lookup[seg]
    # Fallback: use the path as-is (normalized), hoping it already points at jpeg.
    return _to_rel_jpeg(path)


def build_manifest(dataset_dir: str, output_path: str, verify: bool = True) -> pd.DataFrame:
    csv_dir = os.path.join(dataset_dir, "csv")
    if not os.path.isdir(csv_dir):
        print(f"[error] expected CSV folder not found: {csv_dir}", file=sys.stderr)
        print("        Is --dataset-dir pointing at the mounted Kaggle dataset root?",
              file=sys.stderr)
        sys.exit(1)

    lookup = _build_dicom_lookup(dataset_dir)

    rows = []
    for abnormality, csv_name in DESCRIPTION_CSVS:
        csv_path = os.path.join(csv_dir, csv_name)
        if not os.path.exists(csv_path):
            print(f"[warn] missing description CSV (skipping): {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        for _, r in df.iterrows():
            original_label = str(r.get(COL_PATHOLOGY, "")).strip().upper()
            # Merge BENIGN_WITHOUT_CALLBACK -> BENIGN (binary task).
            label = "MALIGNANT" if original_label == "MALIGNANT" else "BENIGN"

            image_rel = _resolve(r.get(COL_IMAGE, ""), lookup, dataset_dir)
            mask_rel = _resolve(r.get(COL_ROI_MASK, ""), lookup, dataset_dir)

            rows.append({
                "patient_id": str(r.get(COL_PATIENT, "")).strip(),
                "image_path": image_rel,
                "roi_mask_path": mask_rel,
                "original_label": original_label,
                "label": label,
                "abnormality_type": r.get(COL_ABNORMALITY, abnormality),
            })

    manifest = pd.DataFrame(rows)
    manifest = manifest[manifest["patient_id"] != ""].reset_index(drop=True)

    if manifest.empty:
        print("[error] no rows built — check dataset structure and CSV columns.",
              file=sys.stderr)
        sys.exit(1)

    # Report class distribution up front (medical data is imbalanced).
    print("\n[info] class distribution (merged binary label):")
    print(manifest["label"].value_counts().to_string())
    print(f"\n[info] unique patients: {manifest['patient_id'].nunique()}")
    print(f"[info] total abnormalities (rows): {len(manifest)}")

    if verify:
        missing_img = _count_missing(manifest["image_path"], dataset_dir)
        missing_mask = _count_missing(manifest["roi_mask_path"], dataset_dir)
        print(f"\n[verify] images not found on disk: {missing_img}/{len(manifest)}")
        print(f"[verify] masks not found on disk:  {missing_mask}/{len(manifest)}")
        if missing_img > 0:
            print("[verify] NOTE: nonzero missing images usually means the "
                  "dicom_info.csv path mapping needs adjustment for this mirror.")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    manifest.to_csv(output_path, index=False)
    print(f"\n[done] manifest written to {output_path}")
    return manifest


def _count_missing(rel_paths: pd.Series, dataset_dir: str) -> int:
    missing = 0
    for rel in rel_paths:
        if not rel or not os.path.exists(os.path.join(dataset_dir, rel)):
            missing += 1
    return missing


def main():
    parser = argparse.ArgumentParser(
        description="Build a patient-wise CBIS-DDSM manifest from a mounted "
                    "Kaggle/Colab dataset (no local 165 GB download needed)."
    )
    parser.add_argument(
        "--dataset-dir", required=True,
        help="Root of the mounted CBIS-DDSM dataset "
             "(e.g. /kaggle/input/cbis-ddsm-breast-cancer-image-dataset).",
    )
    parser.add_argument(
        "--output", default="data/manifest.csv",
        help="Where to write the manifest CSV (default: data/manifest.csv).",
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Skip checking that resolved image/mask paths exist on disk.",
    )
    args = parser.parse_args()

    build_manifest(args.dataset_dir, args.output, verify=not args.no_verify)


if __name__ == "__main__":
    main()
