"""
Visualization and Reporting Utility.

This script aggregates the visual outputs generated during training (loss curves,
confusion matrices) and organizes them into the 'reports/figures' directory.
It also plots a custom loss curve for better analysis.
"""

import shutil
import os
from pathlib import Path
import config
import matplotlib.pyplot as plt
import pandas as pd

def organize_plots():
    """Copies essential training plots to the reports folder."""
    source_dir = config.REPORTS_DIR / "runs" / config.RUN_NAME
    dest_dir = config.REPORTS_DIR / "figures"
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    files_to_copy = [
        "results.png", 
        "confusion_matrix.png", 
        "F1_curve.png", 
        "PR_curve.png",
        "labels.jpg"
    ]
    
    print("🖼️  Organizing training plots...")
    copied_count = 0
    for file_name in files_to_copy:
        src = source_dir / file_name
        dst = dest_dir / file_name
        
        if src.exists():
            shutil.copy(src, dst)
            print(f"   [+] Copied: {file_name}")
            copied_count += 1
        else:
            print(f"   [!] Missing: {file_name} (Training might not be finished)")
            
    print(f"✅ Successfully organized {copied_count} figures in {dest_dir}")

def plot_custom_loss():
    """Generates a clean training loss curve from the results CSV."""
    csv_path = config.REPORTS_DIR / "runs" / config.RUN_NAME / "results.csv"
    if not csv_path.exists():
        return

    # Read YOLO training logs
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip() # Clean column names
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train/box_loss'], label='Box Loss', linewidth=2)
    plt.plot(df['epoch'], df['train/cls_loss'], label='Class Loss', linewidth=2)
    
    plt.title("Training Loss Convergence", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    save_path = config.REPORTS_DIR / "figures" / "custom_loss_curve.png"
    plt.savefig(save_path)
    print(f"📈 Custom loss curve saved to: {save_path}")

if __name__ == '__main__':
    organize_plots()
    plot_custom_loss()