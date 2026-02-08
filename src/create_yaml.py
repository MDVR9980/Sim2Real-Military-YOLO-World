"""
YAML Configuration Generator.

This script generates the 'data.yaml' file required for YOLO training.
It resolves absolute paths dynamically to prevent 'File Not Found' errors
when moving the project between different machines (e.g., Laptop vs Lab PC).
"""

import yaml
import config
from pathlib import Path

def create_data_yaml():
    """
    Creates data.yaml pointing to the processed dataset with absolute paths.
    """
    data_path = config.PROCESSED_DATA_DIR.resolve()
    
    # Structure required by Ultralytics YOLO
    yaml_content = {
        'path': str(data_path),  # Absolute path to dataset root
        'train': 'train/images', # Relative path to train images
        'val': 'val/images',     # Relative path to val images
        'nc': len(config.CLASS_NAMES),
        'names': config.CLASS_NAMES
    }

    yaml_file = config.DATA_DIR / 'data.yaml'
    
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    
    print(f"✅ Configuration file created at: {yaml_file}")
    print(f"   Dataset Root: {data_path}")

if __name__ == '__main__':
    create_data_yaml()