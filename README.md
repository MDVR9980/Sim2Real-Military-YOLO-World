# Sim2Real-Military-YOLO-World

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![YOLO-World](https://img.shields.io/badge/YOLO--World-v8-green)

**Sim-to-Real Domain Adaptation** for detecting specific military armaments using **YOLO-World**. 
This project demonstrates how to train an open-vocabulary object detection model on **synthetic data** (sourced from video games) to detect rare and dangerous objects in **real-world** scenarios where data is scarce.

## 📌 Project Overview

Training deep learning models for military surveillance faces a major challenge: **Data Scarcity**. Collecting labeled images of specific weapons (e.g., *CMMG Banshee*, *Lobaev DXL-5*) in combat scenarios is difficult and dangerous.

This project solves this by using a **Sim-to-Real** approach:
1.  **Source Domain (Sim):** Synthetic images captured from realistic video games.
2.  **Target Domain (Real):** A small validation set of real-world images.
3.  **Model:** YOLO-World (Real-time Open-Vocabulary Object Detection).
4.  **Technique:** Prompt-based fine-tuning to align visual features from the simulation with textual descriptions of the real weapons.

## 📂 Project Structure

```text
Sim2Real-Military-YOLO-World/
├── data/                  # Dataset storage (Ignored in Git)
├── models/                # Model weights (Ignored in Git)
├── notebooks/             # Jupyter Notebooks for analysis & demo
├── reports/               # Training logs, figures, and CSV metrics
├── src/                   # Source code
│   ├── config.py          # Central configuration
│   ├── dataset_prep.py    # XML to YOLO converter
│   ├── train_domain_adapt.py # Main training script
│   ├── evaluate.py        # Evaluation metrics
│   └── ...
├── main.py                # CLI Entry point
├── requirements.txt       # Dependencies
└── README.md              # Project documentation
```

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Sim2Real-Military-YOLO-World.git
    cd Sim2Real-Military-YOLO-World
    ```

2.  **Create a virtual environment (Optional but recommended):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

The project is controlled via a central CLI script. Run the following command to see the menu:

```bash
python main.py
```

### Step-by-Step Workflow:

1.  **Data Preparation:** 
    Select **Option 1** in the menu. This converts your raw Pascal VOC (XML) dataset into YOLO format and generates the `data.yaml` configuration file. It handles the mapping of generic game labels (e.g., `w0`) to specific weapon codes (e.g., `w190`).

2.  **Training (Sim-to-Real):**
    Select **Option 2**. This initiates the fine-tuning process. The model learns from the synthetic data while minimizing the domain gap using YOLO-World's text-image alignment capabilities.

3.  **Evaluation:**
    Select **Option 3**. The model is tested on the **Real-World** validation set. Metrics like mAP@50 and mAP@50-95 are saved to `reports/final_metrics.csv`.

4.  **Visualization:**
    Select **Option 5** to generate a "Before vs. After" comparison image, showing how the model improved from Zero-Shot baseline to the Fine-Tuned state.

## 📊 Methodology

### Class Mapping Strategy
To bridge the gap between the game environment and reality, we map generic simulation labels to specific real-world identifiers during the data preparation phase:

| Class ID | Code | Description | Domain Source |
| :--- | :--- | :--- | :--- |
| 0 | `h0` | Unarmed Person | Sim & Real |
| 1 | `h1` | Armed Person | Sim & Real |
| 2 | `w190` | **CMMG Banshee** | Sim (mapped from w0) |
| 3 | `w146` | **Lobaev DXL-5** | Sim (mapped from w0) |
| 4 | `w0` | Generic Weapon | Real (Noise class) |

### Zero-Shot vs. Fine-Tuning
*   **Zero-Shot:** The model uses only text prompts (e.g., "sniper rifle") to detect objects without seeing any images.
*   **Fine-Tuning:** The model updates its weights using the synthetic images, learning the specific visual features of the *Lobaev* and *CMMG* weapons that text alone cannot describe.

## 📈 Results

*Evaluation metrics on Real-World Data:*

| Metric | Zero-Shot Baseline | Sim-to-Real (Ours) |
| :--- | :---: | :---: |
| mAP@50 | 0.XX | **0.YY** |
| Precision | 0.XX | **0.YY** |

*(Detailed loss curves and confusion matrices are available in the `reports/figures` directory)*

## 🤝 Credits
*   **YOLO-World:** Ultralytics & Tencent AI Lab
*   **Dataset:** Custom collected synthetic data (Arma 3 / CS2) and real-world samples.