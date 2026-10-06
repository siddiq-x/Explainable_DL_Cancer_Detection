import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

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
