"""
Configuration Module for Sim2Real-Military-YOLO-World.

This module defines the project structure, hyperparameters for training,
and class mappings used for the Sim-to-Real domain adaptation task.
It ensures that all scripts utilize consistent paths and settings.
"""

from pathlib import Path

# --- Project Directory Structure ---
# Automatically resolve the project root relative to this file
BASE_DIR = Path(__file__).resolve().parent.parent

# Source Data Path (Where your professor's data sits currently)
SOURCE_DATASET_DIR = BASE_DIR / "student_dataset"

# Destination Data Path (Where YOLO will read data from)
DATA_DIR = BASE_DIR / "data"
FINAL_DATASET_DIR = DATA_DIR / "dataset"  # Organized data goes here

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# --- Training Hyperparameters (Optimized for RTX 5080) ---
EPOCHS = 100          # Sufficient epochs for convergence
BATCH_SIZE = 64       # High batch size utilizing 16GB VRAM
IMAGE_SIZE = 640      # Standard YOLO input resolution
WORKERS = 8           # Number of CPU threads for data loading
DEVICE = 0            # GPU ID

# --- Model & Experiment Settings ---
MODEL_NAME = "yolov8l-worldv2.pt"  # Large model for better accuracy
PROJECT_NAME = "Sim2Real-Military-Final"
RUN_NAME = "sim2real_run_rtx5080"

# --- Class Mappings ---
# Based on the professor's PDF requirements.
# Ensure this matches the 'classes.txt' file in student_dataset.
CLASS_NAMES = {
    0: "unarmed person",
    1: "armed person",
    2: "weapon"
}