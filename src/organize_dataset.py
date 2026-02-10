"""
Dataset Organization Script.

This script is responsible for structuring the raw dataset provided by the professor
into a format compatible with YOLO training (Sim-to-Real strategy).

Strategy:
    1. 'train' folder (Simulated) -> Moves to Training Set.
    2. 'aid_train' folder (Aux Real) -> Splits 80% to Training, 20% to Validation.
    3. 'test' folder (Real) -> Moves to Test Set.

It ensures that images and their corresponding labels are moved correctly.
"""

import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm
import config  # Imports paths from config.py

def copy_files(file_list: list, source_dir: Path, split_name: str, desc: str):
    """
    Copies a list of image files and their corresponding labels to the destination split.

    Args:
        file_list (list): List of image filenames to copy.
        source_dir (Path): The directory where these files currently reside.
        split_name (str): The target split ('train', 'val', or 'test').
        desc (str): Description for the progress bar.
    """
    # Define destination paths
    img_dest = config.FINAL_DATASET_DIR / split_name / "images"
    lbl_dest = config.FINAL_DATASET_DIR / split_name / "labels"
    
    # Create directories if they don't exist
    img_dest.mkdir(parents=True, exist_ok=True)
    lbl_dest.mkdir(parents=True, exist_ok=True)
    
    # Iterate and copy
    for filename in tqdm(file_list, desc=desc):
        # 1. Copy Image
        src_img = source_dir / filename
        dst_img = img_dest / filename
        shutil.copy2(src_img, dst_img)
        
        # 2. Copy Label (if exists)
        # Assumes label has the same name as image but with .txt extension
        label_name = os.path.splitext(filename)[0] + ".txt"
        
        # Check for label in 'labels' subfolder (if source structure has images/labels split)
        # OR check in the same folder (flat structure)
        possible_label_paths = [
            source_dir / label_name,                 # Same folder
            source_dir.parent / "labels" / label_name # Standard YOLO structure
        ]
        
        label_copied = False
        for lbl_path in possible_label_paths:
            if lbl_path.exists():
                shutil.copy2(lbl_path, lbl_dest / label_name)
                label_copied = True
                break
        
        # Optional: Warn if label is missing (except for test set if it's unlabelled)
        # if not label_copied and split_name != 'test':
        #     print(f"Warning: No label found for {filename}")

def main():
    """
    Main execution function for organizing the dataset.
    """
    print("🚀 Starting Dataset Organization for Sim-to-Real Project...")
    
    # 1. Clean up destination directory to avoid duplicates
    if config.FINAL_DATASET_DIR.exists():
        print(f"🧹 Cleaning existing data at {config.FINAL_DATASET_DIR}...")
        shutil.rmtree(config.FINAL_DATASET_DIR)
    
    # 2. Check Source Directories
    # Based on your 'ls' output: 'train', 'aid_train', 'test' are inside student_dataset
    sim_source = config.SOURCE_DATASET_DIR / "train"
    aux_source = config.SOURCE_DATASET_DIR / "aid_train"
    test_source = config.SOURCE_DATASET_DIR / "test"
    
    if not sim_source.exists():
        raise FileNotFoundError(f"❌ Sim data not found at: {sim_source}")

    # 3. Process Simulation Data (Sim -> Train)
    # We take all images from the 'train' folder (Simulated data)
    # Note: Assuming images are either in root of 'train' or inside 'train/images'
    # Let's handle the case where they might be in 'train/images' or just 'train/'
    if (sim_source / "images").exists():
        sim_source_imgs = sim_source / "images"
    else:
        sim_source_imgs = sim_source

    sim_files = [f for f in os.listdir(sim_source_imgs) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"🔹 Found {len(sim_files)} Simulated images.")
    copy_files(sim_files, sim_source_imgs, "train", "Copying Sim Data")

    # 4. Process Auxiliary Real Data (Aid_Train -> Split Train/Val)
    # This acts as Domain Adaptation data.
    if (aux_source / "images").exists():
        aux_source_imgs = aux_source / "images"
    else:
        aux_source_imgs = aux_source

    if aux_source_imgs.exists():
        aux_files = [f for f in os.listdir(aux_source_imgs) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        random.shuffle(aux_files) # Shuffle for random split
        
        # Split 80/20
        split_idx = int(len(aux_files) * 0.8)
        train_aux = aux_files[:split_idx]
        val_aux = aux_files[split_idx:]
        
        print(f"🔹 Found {len(aux_files)} Auxiliary Real images.")
        print(f"   Splitting: {len(train_aux)} to Train (Domain Adapt), {len(val_aux)} to Val.")
        
        copy_files(train_aux, aux_source_imgs, "train", "Copying Aux -> Train")
        copy_files(val_aux, aux_source_imgs, "val", "Copying Aux -> Val")
    else:
        print("⚠️ Warning: 'aid_train' folder not found. Skipping Domain Adaptation step.")

    # 5. Process Test Data (Test -> Test)
    if (test_source / "images").exists():
        test_source_imgs = test_source / "images"
    else:
        test_source_imgs = test_source

    if test_source_imgs.exists():
        test_files = [f for f in os.listdir(test_source_imgs) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        print(f"🔹 Found {len(test_files)} Test images.")
        copy_files(test_files, test_source_imgs, "test", "Copying Test Data")
    
    print("\n✅ Dataset Organization Complete!")
    print(f"📂 Data is ready at: {config.FINAL_DATASET_DIR}")

if __name__ == "__main__":
    main()