"""
Configuration Module for Sim2Real-Military-YOLO-World.

This module defines the project structure, hyperparameters for training,
and class mappings used for the Sim-to-Real domain adaptation task.
It ensures that all scripts utilize consistent paths and settings.
"""

import os
from pathlib import Path

# --- Project Directory Structure ---
# Automatically resolve the project root relative to this file
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "final_dataset_voc"
PROCESSED_DATA_DIR = DATA_DIR / "processed_yolo"

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# --- Training Hyperparameters ---
# Adjust these based on available hardware resources (e.g., University Lab GPU)
EPOCHS = 50          # Recommended: 50-100 for convergence
BATCH_SIZE = 16      # Decrease to 8 or 4 if CUDA Out of Memory occurs
IMAGE_SIZE = 640     # Standard YOLO input resolution
WORKERS = 8          # Number of CPU threads for data loading
DEVICE = 0           # GPU ID (set to 'cpu' if no GPU is available)

# --- Model & Experiment Settings ---
MODEL_NAME = "yolov8s-worldv2.pt"  # Base pre-trained model
PROJECT_NAME = "Sim2Real-Military"
RUN_NAME = "sim2real_finetune"     # Name of the experiment folder

# --- Domain Adaptation Class Mappings ---
# Maps integer class IDs to specific military object names.
# This mapping must be consistent across training and inference.
CLASS_NAMES = {
    0: "h0",    # Unarmed Person
    1: "h1",    # Armed Person
    2: "w190",  # CMMG Banshee (SMG) - Main Target
    3: "w146",  # Lobaev DXL-5 (Sniper) - Bonus Target
    4: "w0"     # Generic Weapon (Other guns in real dataset)
}

# --- Text Prompts for Zero-Shot Inference ---
# Descriptive texts used by YOLO-World to generate embeddings.
TEXT_PROMPTS = [
    "unarmed person",
    "armed person",
    "CMMG Banshee submachine gun",
    "Lobaev DXL-5 sniper rifle",
    "weapon gun"
]