"""
Model Evaluation Module.

This script loads the fine-tuned model and performs comprehensive evaluation
on the validation set (Real-world images). It calculates standard metrics
like mAP@50 and mAP@50-95 and saves them for the final report.
"""

from ultralytics import YOLOWorld
import config
import pandas as pd
from pathlib import Path

def evaluate():
    """
    Runs validation and saves metrics to a CSV file.
    """
    # Define path to the best trained model
    model_path = config.REPORTS_DIR / "runs" / config.RUN_NAME / "weights" / "best.pt"
    
    if not model_path.exists():
        print(f"❌ Error: Trained model not found at {model_path}")
        print("Tip: Run the training script (step 2) first.")
        return

    print(f"📊 Loading model for evaluation: {model_path}")
    model = YOLOWorld(model_path)
    
    # Run validation on the dataset defined in data.yaml
    metrics = model.val(
        data=str(config.DATA_DIR / 'data.yaml'),
        split='val',
        project=str(config.REPORTS_DIR / "evaluation"),
        name="val_results"
    )
    
    # Display Key Metrics
    print("\n" + "-"*30)
    print(" FINAL VALIDATION RESULTS")
    print("-"*30)
    map50 = metrics.box.map50
    map50_95 = metrics.box.map
    print(f"🎯 mAP@50    (IoU=0.50):      {map50:.4f}")
    print(f"🎯 mAP@50-95 (IoU=0.50:0.95): {map50_95:.4f}")
    
    # Save Metrics to CSV for Documentation
    results_data = {
        "Metric": ["mAP@50", "mAP@50-95", "Precision", "Recall"],
        "Value": [map50, map50_95, metrics.box.mp, metrics.box.mr]
    }
    df = pd.DataFrame(results_data)
    csv_path = config.REPORTS_DIR / "final_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"📄 Detailed metrics saved to: {csv_path}")

if __name__ == '__main__':
    evaluate()