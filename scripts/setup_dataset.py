import os
import sys
import pandas as pd
import numpy as np
from PIL import Image

# Allow importing sibling scripts (build_manifest_kaggle) regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_DIR = 'data/raw'
MANIFEST_PATH = 'data/manifest.csv'


def check_real_data():
    """Check whether a real CBIS-DDSM dataset is present under RAW_DATA_DIR.

    The real dataset is identified by the presence of a ``csv/`` folder holding
    the standard CBIS-DDSM description files (this is the awsaf49 Kaggle layout).
    We deliberately do NOT key off loose .jpg files, because the mock generator
    also writes .jpg files into ``images/`` — that heuristic can't tell mock from
    real and misclassifies both.
    """
    csv_dir = os.path.join(RAW_DATA_DIR, 'csv')
    if not os.path.isdir(csv_dir):
        return False
    description_csvs = [
        'calc_case_description_train_set.csv',
        'calc_case_description_test_set.csv',
        'mass_case_description_train_set.csv',
        'mass_case_description_test_set.csv',
    ]
    return any(os.path.exists(os.path.join(csv_dir, name)) for name in description_csvs)

def generate_mock_data(num_samples=100):
    """Generate dummy data if real data is not available, for pipeline testing."""
    print("Real dataset not found in data/raw/.")
    print("Generating mock dataset for testing purposes...")
    
    os.makedirs(os.path.join(RAW_DATA_DIR, 'images'), exist_ok=True)
    os.makedirs(os.path.join(RAW_DATA_DIR, 'masks'), exist_ok=True)
    
    manifest_data = []
    
    for i in range(num_samples):
        patient_id = f"P_{str(i).zfill(4)}"
        
        # 40% benign, 20% benign_without_callback, 40% malignant
        rand_val = np.random.rand()
        if rand_val < 0.4:
            original_label = 'BENIGN'
            label = 'BENIGN'
        elif rand_val < 0.6:
            original_label = 'BENIGN_WITHOUT_CALLBACK'
            label = 'BENIGN'
        else:
            original_label = 'MALIGNANT'
            label = 'MALIGNANT'
            
        is_mass = np.random.rand() > 0.5
        abnormality_type = 'mass' if is_mass else 'calcification'
        
        image_name = f"{patient_id}_image.jpg"
        mask_name = f"{patient_id}_mask.jpg"
        
        img_path = os.path.join(RAW_DATA_DIR, 'images', image_name)
        mask_path = os.path.join(RAW_DATA_DIR, 'masks', mask_name)
        
        # Generate dummy 224x224 images
        img = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        Image.fromarray(img).save(img_path)
        
        # Generate dummy mask
        mask = np.zeros((224, 224), dtype=np.uint8)
        mask[50:150, 50:150] = 255
        Image.fromarray(mask).save(mask_path)
        
        manifest_data.append({
            'patient_id': patient_id,
            'image_path': f"images/{image_name}",
            'roi_mask_path': f"masks/{mask_name}",
            'original_label': original_label,
            'label': label,
            'abnormality_type': abnormality_type
        })
        
    df = pd.DataFrame(manifest_data)
    df.to_csv(MANIFEST_PATH, index=False)
    print(f"Mock manifest created at {MANIFEST_PATH} with {num_samples} samples.")
    print("NOTE: Please download the real CBIS-DDSM dataset from Kaggle for actual training.")
    print("      Example Kaggle dataset: awsaf49/cbis-ddsm-breast-cancer-image-dataset")

def process_real_data():
    """Process a downloaded/mounted real CBIS-DDSM dataset into a manifest.

    This delegates to ``build_manifest_kaggle.build_manifest``, which merges the
    four calc/mass train/test description CSVs, maps the DICOM-style paths to the
    actual jpeg files via ``dicom_info.csv``, merges
    ``BENIGN_WITHOUT_CALLBACK`` -> benign, and writes ``data/manifest.csv``.
    """
    print("Real dataset detected (found data/raw/csv/). Building manifest...")
    try:
        from build_manifest_kaggle import build_manifest
    except ImportError as exc:
        print(f"[error] could not import build_manifest_kaggle: {exc}", file=sys.stderr)
        print("        Ensure scripts/build_manifest_kaggle.py is present.", file=sys.stderr)
        sys.exit(1)

    build_manifest(dataset_dir=RAW_DATA_DIR, output_path=MANIFEST_PATH, verify=True)
    
if __name__ == "__main__":
    if check_real_data():
        process_real_data()
    else:
        generate_mock_data()
