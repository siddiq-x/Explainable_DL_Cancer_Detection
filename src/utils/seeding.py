import os
import random
import numpy as np
import torch

def seed_everything(seed: int = 42) -> None:
    """
    Seeds all random number generators to ensure reproducibility.
    
    Args:
        seed (int): The random seed to use.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Global seed set to {seed}")
