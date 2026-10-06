import os
import pytest
import pandas as pd
import torch
from torchvision import transforms

from src.data.dataset import MammographyDataset
from src.data.preprocessing import get_base_transforms
from src.data.splits import create_splits

# Use the mock manifest for testing
MANIFEST_PATH = 'data/manifest.csv'
RAW_DATA_DIR = 'data/raw'

@pytest.fixture
def manifest_exists():
    return os.path.exists(MANIFEST_PATH)

def test_splits_no_patient_leakage(manifest_exists):
    if not manifest_exists:
        pytest.skip("Manifest not found. Run scripts/setup_dataset.py first.")
        
    train_df, val_df, test_df = create_splits(MANIFEST_PATH, test_size=0.15, val_size=0.15, seed=42)
    
    train_patients = set(train_df['patient_id'].unique())
    val_patients = set(val_df['patient_id'].unique())
    test_patients = set(test_df['patient_id'].unique())
    
    # Assert intersections are empty (no leakage)
    assert len(train_patients.intersection(val_patients)) == 0, "Patient leakage between train and val"
    assert len(train_patients.intersection(test_patients)) == 0, "Patient leakage between train and test"
    assert len(val_patients.intersection(test_patients)) == 0, "Patient leakage between val and test"

def test_dataset_output_shapes_and_types(manifest_exists):
    if not manifest_exists:
        pytest.skip("Manifest not found. Run scripts/setup_dataset.py first.")
        
    train_df, _, _ = create_splits(MANIFEST_PATH)
    transform = get_base_transforms(image_size=224)
    
    dataset = MammographyDataset(train_df, root_dir=RAW_DATA_DIR, transform=transform)
    
    assert len(dataset) > 0
    
    sample = dataset[0]
    
    assert 'image' in sample
    assert 'label' in sample
    
    # Check tensor properties
    assert isinstance(sample['image'], torch.Tensor)
    assert isinstance(sample['label'], torch.Tensor)
    
    # Shape should be [3, 224, 224] for RGB converted images
    assert sample['image'].shape == (3, 224, 224)
    
    # Label should be a scalar float (for BCEWithLogitsLoss)
    assert sample['label'].dim() == 0
    assert sample['label'].dtype == torch.float32
