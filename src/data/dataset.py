"""PyTorch Dataset for CBIS-DDSM mammograms, plus the shared batch contract."""

from __future__ import annotations

import os
from typing import Any, Mapping, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd


def unpack_batch(batch: Any) -> Tuple[Any, Any]:
    """Extract ``(images, labels)`` from a batch.

    Supports the dict batches produced by :class:`MammographyDataset` as well as
    plain ``(image, label)`` tuples/lists (useful for lightweight synthetic
    datasets and tests). This is the single shared definition of the batch
    contract used by the trainer and the evaluator.
    """
    if isinstance(batch, Mapping):
        return batch["image"], batch["label"]
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TypeError(
        "Unsupported batch type: expected a dict with 'image'/'label' keys "
        "or an (image, label) tuple."
    )


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
        normalized_labels = self.dataframe['label'].astype('string').str.strip().str.upper()
        invalid = ~normalized_labels.isin(self.label_mapping)
        if invalid.any():
            raise ValueError(f"Unknown or missing labels in manifest: {self.dataframe.loc[invalid, 'label'].unique().tolist()}")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_rel_path = self.dataframe.loc[idx, 'image_path']
        img_name = os.path.join(self.root_dir, img_rel_path)
        
        # Load image and convert to RGB (standard for pre-trained backbones)
        try:
            with Image.open(img_name) as source:
                image = source.convert('RGB')
        except Exception as e:
            raise IOError(f"Error loading image {img_name}: {e}")

        # Get label and convert to float32 tensor
        str_label = self.dataframe.loc[idx, 'label'].strip().upper()
        label = self.label_mapping[str_label]
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
