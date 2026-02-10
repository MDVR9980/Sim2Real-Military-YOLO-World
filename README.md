# Sim-to-Real Military Object Detection using YOLO-World 🛡️

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![YOLO-World](https://img.shields.io/badge/YOLO--World-v8-green)
![Task](https://img.shields.io/badge/Task-Domain%20Adaptation-orange)

## 📌 Abstract
This project implements a **Sim-to-Real Domain Adaptation** pipeline for detecting armed personnel and weapons in military environments. Due to the scarcity and danger of collecting real-world military datasets, we utilize synthetic data (from simulation environments) and a small set of auxiliary real-world data to fine-tune the **YOLO-World** open-vocabulary model.

The goal is to bridge the "Reality Gap" and achieve high detection performance on real-world test data that the model has never seen before.

## 🚀 Key Features
- **Sim-to-Real Strategy:** Training on Synthetic data + Auxiliary Real data to learn domain-invariant features.
- **YOLO-World Architecture:** Utilizing the power of vision-language models for robust detection.
- **Automated Pipeline:** Scripts for dataset organization, training, evaluation, and visualization.
- **High-Performance Config:** Optimized for RTX 5080 with large batch processing.

## 📂 Project Structure

```text
Sim2Real-Military-YOLO-World/
│
├── data/                  # Processed data ready for YOLO (GitIgnored)
├── student_dataset/       # Raw input data provided by professor (GitIgnored)
├── models/                # Model weights (Pre-trained & Fine-tuned)
├── reports/               # Training logs, figures, and metrics
│   └── figures/           # Final plots for the report
│
├── src/                   # Source Code
│   ├── config.py          # Central configuration & Hyperparameters
│   ├── organize_dataset.py # Script to restructure raw data
│   ├── create_yaml.py     # Script to generate data.yaml
│   ├── train_domain_adapt.py # Main training loop (Sim2Real)
│   ├── evaluate.py        # Evaluation on Real Test set
│   └── visualize.py       # Plot generation tool
│
├── main.py                # Main entry point (CLI Menu)
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/username/Sim2Real-Military-YOLO-World.git
   cd Sim2Real-Military-YOLO-World
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Model Weights:**
   Download `yolov8l-worldv2.pt` and place it in the `models/` directory.

## ⚙️ Usage

The project is managed via a central CLI tool. Simply run:

```bash
python main.py
```

Follow the on-screen menu:
1.  **Organize Dataset:** Formats the `student_dataset` into YOLO structure (`data/`).
2.  **Generate YAML:** Creates the configuration file.
3.  **Start Training:** Begins the Fine-tuning process on RTX 5080.
4.  **Evaluate:** Tests the model on the unseen Real-World dataset.
5.  **Generate Reports:** Creates visualization plots in `reports/figures`.

## 🧠 Methodology (Sim-to-Real)

We employ a data-level domain adaptation strategy:

| Data Split | Source | Usage | Purpose |
|------------|--------|-------|---------|
| **Train** | Simulated Data | Training | Learn basic object features (Shape, Pose). |
| **Train** | Aux Real Data (80%) | Training | Bridge the texture/lighting gap (Reality Gap). |
| **Val** | Aux Real Data (20%) | Validation | Tune hyperparameters and prevent overfitting. |
| **Test** | Real Test Data | Testing | Final evaluation on strictly unseen data. |

## 📊 Classes
The model is trained to detect the following classes based on the provided specifications:
- `0: unarmed person`
- `1: armed person`
- `2: weapon`

## 📈 Results
*Metrics will be populated after the final run.*
- **mAP@50:** TBD
- **Precision:** TBD
- **Recall:** TBD

## 🤝 Credits
- **Ultralytics YOLO:** For the SOTA object detection framework.
- **YOLO-World:** For the open-vocabulary capabilities.