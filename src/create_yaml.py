"""
YAML Configuration Generator.

This script generates the 'data.yaml' file required for YOLO training.
It dynamically resolves paths to the organized dataset to ensure
compatibility across different environments.
"""

import yaml
import config
from pathlib import Path

def create_data_yaml():
    """
    Creates 'data.yaml' pointing to the organized dataset (Train/Val/Test).
    """
    # The root path of the organized dataset
    dataset_root = config.FINAL_DATASET_DIR.resolve()
    
    # Structure required by Ultralytics YOLO
    # Using relative paths from the 'path' defined below
    yaml_content = {
        'path': str(dataset_root),  # Absolute path to dataset root
        'train': 'train/images',    # Relative path
        'val': 'val/images',        # Relative path
        'test': 'test/images',      # Relative path
        'nc': len(config.CLASS_NAMES),
        'names': config.CLASS_NAMES
    }

    yaml_file = config.DATA_DIR / 'data.yaml'
    
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    
    print(f"✅ Configuration file created at: {yaml_file}")
    print(f"   dataset_root: {dataset_root}")
    print(f"   Classes: {config.CLASS_NAMES}")

if __name__ == '__main__':
    create_data_yaml()