"""
Comparison Demo Script.

This script visually compares the performance of the 'Zero-Shot' baseline model
against the 'Fine-Tuned' Sim-to-Real model on a sample real-world image.
It generates a side-by-side plot for qualitative evaluation.
"""

from ultralytics import YOLOWorld
import cv2
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add 'src' to python path to import config correctly
sys.path.append(str(Path(__file__).parent / 'src'))
import config

def run_comparison():
    """Loads models and runs inference on a test image."""
    
    # Paths definition
    trained_model_path = config.REPORTS_DIR / "runs" / config.RUN_NAME / "weights" / "best.pt"
    # Select a representative real image for testing
    test_image_path = config.RAW_DATA_DIR / "Main_Dataset/Real/101-102.png" 

    # Validation checks
    if not trained_model_path.exists():
        print(f"❌ Error: Trained model not found at: {trained_model_path}")
        return
    
    if not test_image_path.exists():
        print(f"❌ Error: Test image not found at: {test_image_path}")
        print("Please check the file name in 'compare_results.py'.")
        return

    print("⏳ Loading Models for Comparison...")
    
    # 1. Zero-Shot Model (Baseline)
    # This model has NO specific training on our data, only text prompts.
    model_base = YOLOWorld(config.MODEL_NAME) 
    model_base.set_classes(config.TEXT_PROMPTS) 

    # 2. Fine-Tuned Model (Proposed Method)
    # This model has adapted to the domain.
    model_tuned = YOLOWorld(trained_model_path)

    print("📸 Running Inference on Real Image...")
    # Zero-shot often needs lower confidence threshold
    res_base = model_base.predict(str(test_image_path), conf=0.1)[0]
    # Tuned model should have higher confidence
    res_tuned = model_tuned.predict(str(test_image_path), conf=0.25)[0] 

    # Visualization
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot Baseline
    ax[0].imshow(cv2.cvtColor(res_base.plot(), cv2.COLOR_BGR2RGB))
    ax[0].set_title("Baseline: Zero-Shot (Before Training)", fontsize=18)
    ax[0].axis('off')

    # Plot Result
    ax[1].imshow(cv2.cvtColor(res_tuned.plot(), cv2.COLOR_BGR2RGB))
    ax[1].set_title("Proposed: Sim-to-Real (After Training)", fontsize=18, color='green')
    ax[1].axis('off')

    # Save and Show
    output_path = config.REPORTS_DIR / "figures" / "final_comparison.png"
    plt.savefig(output_path)
    print(f"✅ Comparison result saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    run_comparison()