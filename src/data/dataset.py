import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd

class MammographyDataset(Dataset):
    """
    Custom PyTorch Dataset for loading CBIS-DDSM mammograms and labels.
    """
    def __init__(self, dataframe: pd.DataFrame, root_dir: str, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): Dataframe containing 'image_path' and 'label'.
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        
        # Map labels to binary integers
        self.label_mapping = {'BENIGN': 0, 'MALIGNANT': 1}
        # In case the user hasn't mapped 'BENIGN_WITHOUT_CALLBACK' yet
        self.label_mapping['BENIGN_WITHOUT_CALLBACK'] = 0

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_rel_path = self.dataframe.loc[idx, 'image_path']
        img_name = os.path.join(self.root_dir, img_rel_path)
        
        # Load image and convert to RGB (standard for pre-trained backbones)
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            raise IOError(f"Error loading image {img_name}: {e}")

        # Get label and convert to float32 tensor
        str_label = self.dataframe.loc[idx, 'label'].strip().upper()
        label = self.label_mapping.get(str_label, 0) # default to 0 if unknown, though shouldn't happen
        label_tensor = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        # Include image path and mask path for explainability overlay later
        mask_path = str(self.dataframe.loc[idx, 'roi_mask_path'])
        
        return {
            'image': image,
            'label': label_tensor,
            'image_path': img_rel_path,
            'mask_path': mask_path
        }
