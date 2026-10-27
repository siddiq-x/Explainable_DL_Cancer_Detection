import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from typing import List, Tuple

def create_splits(manifest_path: str, test_size: float = 0.15, val_size: float = 0.15, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Creates patient-wise, stratified train/val/test splits to ensure no patient 
    leakage across splits (a patient's images must all be in the same split).
    
    Args:
        manifest_path (str): Path to the manifest CSV file.
        test_size (float): Proportion of data for the test set.
        val_size (float): Proportion of training data for the validation set.
        seed (int): Random seed for reproducibility.
        
    Returns:
        Tuple containing train, val, and test dataframes.
    """
    df = pd.read_csv(manifest_path)
    
    # We must split by patient ID, not just by individual images.
    # Group by patient_id and determine the patient's primary label for stratification
    patient_labels = df.groupby('patient_id')['label'].first().reset_index()
    
    # Split patients into (Train+Val) and Test
    train_val_patients, test_patients = train_test_split(
        patient_labels, 
        test_size=test_size, 
        stratify=patient_labels['label'],
        random_state=seed
    )
    
    # Split (Train+Val) into Train and Val
    # We adjust val_size to be a proportion of the remaining data
    adjusted_val_size = val_size / (1.0 - test_size)
    
    train_patients, val_patients = train_test_split(
        train_val_patients,
        test_size=adjusted_val_size,
        stratify=train_val_patients['label'],
        random_state=seed
    )
    
    # Map back to full dataset (all images for these patients)
    train_df = df[df['patient_id'].isin(train_patients['patient_id'])].copy()
    val_df = df[df['patient_id'].isin(val_patients['patient_id'])].copy()
    test_df = df[df['patient_id'].isin(test_patients['patient_id'])].copy()
    
    return train_df, val_df, test_df


def create_stratified_group_kfold(
    manifest_path: str,
    k: int = 5,
    seed: int = 42,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Create patient-wise, stratified K-fold splits for cross-validation.

    Patients (not images) are the grouping unit, so no patient appears in both
    the train and validation part of a fold — the same leakage guarantee as
    :func:`create_splits`, but repeated ``k`` times with a different held-out set
    each time.

    Args:
        manifest_path: Path to the manifest CSV.
        k: Number of folds.
        seed: Random seed for reproducibility.

    Returns:
        A list of ``(train_df, val_df)`` tuples, one per fold.
    """
    df = pd.read_csv(manifest_path)
    # Normalize to the binary task for stratification (BENIGN_WITHOUT_CALLBACK -> BENIGN).
    df = df.copy()
    df["label"] = df["label"].astype(str).str.upper().str.strip()
    df["label"] = df["label"].where(df["label"] == "MALIGNANT", "BENIGN")

    patients = df.groupby("patient_id")["label"].first().reset_index()
    if patients["patient_id"].nunique() < k:
        raise ValueError(
            f"Not enough patients ({patients['patient_id'].nunique()}) for k={k} folds."
        )

    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    folds: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    for train_idx, val_idx in splitter.split(
        patients, y=patients["label"], groups=patients["patient_id"]
    ):
        train_patients = set(patients.iloc[train_idx]["patient_id"])
        val_patients = set(patients.iloc[val_idx]["patient_id"])
        folds.append(
            (
                df[df["patient_id"].isin(train_patients)].copy(),
                df[df["patient_id"].isin(val_patients)].copy(),
            )
        )
    return folds
