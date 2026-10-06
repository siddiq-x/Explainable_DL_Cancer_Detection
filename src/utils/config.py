import yaml
import os
from typing import Dict, Any

# Environment variables that override values in the YAML config at load time.
# This lets the *same* code run locally (small/mock data) and on a cloud GPU
# (Kaggle/Colab, where the full dataset is mounted) without editing files.
#   e.g. on Kaggle:
#     export CANCER_DATASET_DIR=/kaggle/input/cbis-ddsm-breast-cancer-image-dataset
#     export CANCER_MANIFEST_PATH=/kaggle/working/manifest.csv
ENV_OVERRIDES = {
    "CANCER_DATASET_DIR": ("data", "dataset_dir"),
    "CANCER_MANIFEST_PATH": ("data", "manifest_path"),
}


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Override selected config values from environment variables, if set."""
    for env_var, (section, key) in ENV_OVERRIDES.items():
        value = os.environ.get(env_var)
        if value:
            config.setdefault(section, {})[key] = value
    return config


def load_config(config_path: str = "configs/default_config.yaml") -> Dict[str, Any]:
    """
    Loads hyperparameters from a YAML configuration file.

    Values can be overridden at runtime via environment variables (see
    ``ENV_OVERRIDES``). This is the recommended way to point the pipeline at a
    cloud-mounted dataset (e.g. Kaggle/Colab) without modifying the repo.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary (with env overrides applied).
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    config = _apply_env_overrides(config)

    return config
