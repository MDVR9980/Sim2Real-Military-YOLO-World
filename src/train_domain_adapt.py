"""
Domain Adaptation Training Script.

This module orchestrates the training of the YOLO-World model.
It utilizes a Sim-to-Real strategy where the model is fine-tuned on 
synthetic data (Sim) and validated against real-world data (Real)
to bridge the domain gap.
"""

from ultralytics import YOLOWorld
import config
import os
import torch
import gc

def train_model() -> str:
    """
    Executes the training loop using hyperparameters defined in config.py.
    
    Returns:
        str: Path to the best saved model weights.
    """
    # 1. Resource Management: Clear GPU Cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    print(f"🚀 Starting Sim-to-Real Training: {config.PROJECT_NAME}")
    print(f"⚙️  Settings: Epochs={config.EPOCHS}, Batch={config.BATCH_SIZE}, Device={config.DEVICE}")

    # 2. Load Model
    # Checks if a pretrained model exists locally, otherwise downloads it.
    model_path = config.MODELS_DIR / "pre_trained" / config.MODEL_NAME
    if not model_path.exists():
        print("Model not found locally, downloading default weights...")
        model_path = config.MODEL_NAME 
    
    model = YOLOWorld(model_path)

    # 3. Start Training (Fine-tuning)
    results = model.train(
        data=str(config.DATA_DIR / 'data.yaml'),
        epochs=config.EPOCHS,
        imgsz=config.IMAGE_SIZE,
        batch=config.BATCH_SIZE,
        device=config.DEVICE,
        workers=config.WORKERS,
        project=str(config.REPORTS_DIR / "runs"),  # Output directory
        name=config.RUN_NAME,
        save=True,
        plots=True,
        exist_ok=True,     # Overwrite existing run if name is same
        close_mosaic=10,   # Disable mosaic augmentation in last 10 epochs
        warmup_epochs=3
    )

    print("✅ Training sequence completed.")
    
    # Path to the best performing model based on Validation mAP
    best_weight = config.REPORTS_DIR / "runs" / config.RUN_NAME / "weights" / "best.pt"
    print(f"💾 Best model weights saved at: {best_weight}")
    
    return str(best_weight)

if __name__ == '__main__':
    train_model()