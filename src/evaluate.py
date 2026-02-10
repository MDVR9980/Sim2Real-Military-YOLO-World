"""
Evaluation Script for Sim2Real Project.

This script loads the best fine-tuned model and evaluates its performance
on the unseen 'Real-World Test Set'. It reports standard metrics like mAP@50.

Author: Student Name
Date: Feb 2026
"""

from ultralytics import YOLOWorld
import config
import pandas as pd
from pathlib import Path

def run_evaluation():
    """
    Runs the validation process on the 'test' split defined in data.yaml.
    Saves the metrics to a CSV file in the reports directory.
    """
    # 1. Locate the best trained weights
    # Assuming the run name is defined in config.RUN_NAME
    weights_path = config.REPORTS_DIR / "runs" / config.RUN_NAME / "weights" / "best.pt"
    
    if not weights_path.exists():
        print(f"❌ Error: Model weights not found at {weights_path}")
        print("   Please run training first.")
        return

    print(f"🚀 Loading model from: {weights_path}")
    model = YOLOWorld(weights_path)

    # 2. Run Validation on TEST set
    # Note: We use split='test' because 'val' was used during training.
    # The 'test' set is the pure real-world data provided by the professor.
    print("📊 Starting evaluation on Real-World Test Data...")
    metrics = model.val(
        data=str(config.DATA_DIR / 'data.yaml'),
        split='test',  # CRITICAL: Evaluate on Test set
        project=str(config.REPORTS_DIR / "evaluation"),
        name="final_test_results",
        device=config.DEVICE
    )

    # 3. Extract and Display Key Metrics
    map50 = metrics.box.map50
    map50_95 = metrics.box.map
    precision = metrics.box.mp
    recall = metrics.box.mr

    print("\n" + "="*40)
    print("       FINAL SIM-TO-REAL RESULTS       ")
    print("="*40)
    print(f"🎯 mAP @ 50% IoU:      {map50:.4f}")
    print(f"🎯 mAP @ 50-95% IoU:   {map50_95:.4f}")
    print(f"🎯 Precision:          {precision:.4f}")
    print(f"🎯 Recall:             {recall:.4f}")
    print("="*40)

    # 4. Save Metrics to CSV for the Final Report
    results_dict = {
        "Metric": ["mAP@50", "mAP@50-95", "Precision", "Recall"],
        "Score": [map50, map50_95, precision, recall]
    }
    df = pd.DataFrame(results_dict)
    csv_output = config.REPORTS_DIR / "final_test_metrics.csv"
    df.to_csv(csv_output, index=False)
    print(f"✅ Metrics saved to: {csv_output}")

if __name__ == "__main__":
    run_evaluation()