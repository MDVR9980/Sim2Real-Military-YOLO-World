"""
Dataset Preparation Script.

This script converts raw Pascal VOC XML annotations into YOLO formatted TXT files.
It implements the 'Sim-to-Real' logic by mapping generic class labels (e.g., 'w0')
in the simulation dataset to specific weapon codes (e.g., 'w190') required for the project.
"""

import xml.etree.ElementTree as ET
import os
import glob
import shutil
from pathlib import Path
from tqdm import tqdm
import config  # Import project configuration

def convert_box(size: tuple, box: tuple) -> tuple:
    """
    Converts bounding box coordinates from Pascal VOC (xmin, xmax, ymin, ymax)
    to YOLO normalized format (x_center, y_center, width, height).

    Args:
        size (tuple): Image size (width, height).
        box (tuple): Bounding box coordinates (xmin, xmax, ymin, ymax).

    Returns:
        tuple: Normalized coordinates (x, y, w, h).
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

def convert_annotation(xml_file: str, output_path: Path, mapping_dict: dict = None):
    """
    Parses an XML file and writes the corresponding YOLO label file.

    Args:
        xml_file (str): Path to the source XML file.
        output_path (Path): Directory to save the .txt file.
        mapping_dict (dict, optional): Dictionary to map source class names 
                                       to target class names (e.g., {'w0': 'w190'}).
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        txt_filename = output_path / Path(xml_file).with_suffix('.txt').name
        
        # Get valid class names from config
        final_classes = list(config.CLASS_NAMES.values())

        with open(txt_filename, 'w') as out_file:
            for obj in root.iter('object'):
                cls_name = obj.find('name').text
                
                # Apply domain adaptation mapping if provided (Crucial for Sim data)
                if mapping_dict and cls_name in mapping_dict:
                    cls_name = mapping_dict[cls_name]
                
                # Filter out classes not in our final list
                if cls_name not in final_classes:
                    continue

                cls_id = final_classes.index(cls_name)
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                     float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                bb = convert_box((w, h), b)
                out_file.write(f"{str(cls_id)} " + " ".join([str(a) for a in bb]) + '\n')
    
    except Exception as e:
        print(f"Warning: Failed to convert {xml_file}. Error: {e}")

def process_folder(source_subpath: str, split_type: str, mapping: dict):
    """
    Processes a specific dataset folder (Sim or Real), converts annotations,
    and moves images to the destination directory.

    Args:
        source_subpath (str): Relative path in the raw data folder.
        split_type (str): 'train' or 'val'.
        mapping (dict): Class name mapping logic.
    """
    source_dir = config.RAW_DATA_DIR / source_subpath
    dest_root = config.PROCESSED_DATA_DIR
    
    images_dir = dest_root / split_type / 'images'
    labels_dir = dest_root / split_type / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Find XML files (case-insensitive search recommended in future, here using .xml)
    xml_files = glob.glob(str(source_dir / '*.xml'))
    print(f"Processing {len(xml_files)} files from {source_subpath} -> {split_type}")

    for xml_file in tqdm(xml_files, desc=f"Converting {source_subpath}"):
        convert_annotation(xml_file, labels_dir, mapping)
        
        # Copy corresponding image
        base_name = os.path.splitext(xml_file)[0]
        image_found = False
        for ext in ['.jpg', '.png', '.jpeg', '.JPG']:
            img_candidate = base_name + ext
            if os.path.exists(img_candidate):
                shutil.copy(img_candidate, images_dir)
                image_found = True
                break
        
        if not image_found:
            # Optional: Log missing images
            pass

if __name__ == '__main__':
    # Clean up previous data to ensure a fresh build
    if config.PROCESSED_DATA_DIR.exists():
        print("Cleaning old processed data...")
        shutil.rmtree(config.PROCESSED_DATA_DIR)
    
    print("=== Starting Data Preparation ===")
    
    # 1. Process Main Dataset (CMMG Banshee)
    # Strategy: Sim data maps 'w0' to 'w190' for Training. Real data remains as is for Validation.
    process_folder('Main_Dataset/Sim', 'train', mapping={'w0': 'w190'})
    process_folder('Main_Dataset/Real', 'val', mapping={})
    
    # 2. Process Bonus Dataset (Lobaev DXL-5)
    # Strategy: Sim data maps 'w0' to 'w146' for Training.
    process_folder('Bonus_Dataset/Sim', 'train', mapping={'w0': 'w146'})
    process_folder('Bonus_Dataset/Real', 'val', mapping={})

    print("✅ Dataset Preparation Completed Successfully!")