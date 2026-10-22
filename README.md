# DRAS: Continuous Driver Risk Assessment System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-00C0A3.svg)](https://developers.google.com/mediapipe)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00ffff.svg)](https://github.com/ultralytics/ultralytics)

Official repository for **Continuous Driver Risk Assessment via Real-Time Affective and Behavioral Tracking (DRAS)**.

> **Project Report:** Vellore Institute of Technology (VIT), Chennai  
> **Authors:** Bharanee B (22BCE1371), Naren Karttik (22BCE1088), Sundar J (22BCE1463)  
> **Project Guide:** Dr. Sivakami R.  

---

## 📌 Overview

Most Advanced Driver Assistance Systems (ADAS) rely solely on single-modality detectors using rigid geometric eye closure thresholds. This produces false positives for drivers with naturally narrow eye apertures and completely ignores cognitive-emotional impairments and manual distractions.

**DRAS** is an edge-deployable multimodal system running on a standard forward-facing camera stream (~30 FPS). It fuses four perception streams through an empirically grounded **70:30 Visual-Behavioral to Affective late-fusion engine**:

1. **Physical Fatigue**: Adaptive Eye Aspect Ratio (A-EAR) with continuous Exponentially Weighted Moving Average (EWMA) baseline calibration, Mouth Aspect Ratio (MAR) yawn tracking, 30-second rolling PERCLOS, and microsleep event detection (5–24 frames).
2. **Affective State**: Fine-tuned PyTorch ResNet-34 classifier on FER2013 (7 emotion classes) with 10-frame majority voting and confidence gating, mapping affective states to an empirical Impairment Score ($E \in [0.0, 1.0]$).
3. **Spatial Deviation**: Monocular Head Pose Estimation (solvePnP yaw, pitch, roll) with exponential smoothing ($\alpha = 0.4$) and MediaPipe Pose slouch/lean monitoring.
4. **Visual-Manual Distraction**: Ultralytics YOLOv8n mobile phone detection ($conf \ge 0.60$), triggering an immediate safety hard floor ($R_{total} \ge 85\%$).

```
             ┌───────────────────────────────────────────────┐
             │            Webcam Input (30 FPS)              │
             └──────────────────────┬────────────────────────┘
                                    │
       ┌────────────────┬───────────┴───────────┬────────────────┐
       ▼                ▼                       ▼                ▼
┌──────────────┐ ┌──────────────┐      ┌────────────────┐ ┌──────────────┐
│   Fatigue    │ │   Affect     │      │ Spatial / Pose │ │ Distraction  │
│  MediaPipe   │ │  ResNet-34   │      │ MediaPipe Pose │ │   YOLOv8n    │
│  A-EAR/MAR   │ │ 7 Emotions   │      │ solvePnP Angles│ │ Phone Detect │
└──────┬───────┘ └──────┬───────┘      └───────┬────────┘ └──────┬───────┘
       │                │                      │                 │
       │     Fatigue F  │ Emotion E   Posture P│      Distract D │
       │     (0.50)     │             (0.25)   │      (0.25)     │
       └───────────┬────┼──────────────────────┴─────────┬───────┘
                   │    │                                │
                   ▼    │                                ▼
       ┌────────────────┴──────┐                 ┌───────────────┐
       │   Visual-Behavioral   │                 │ Phone Floor   │
       │   Composite Score (V) │                 │  (R >= 85%)   │
       └───────────┬───────────┘                 └───────┬───────┘
                   │                                     │
                   ▼                                     ▼
       ┌─────────────────────────────────────────────────────────┐
       │     Late-Fusion Engine: R_total = 0.70 V + 0.30 E       │
       └────────────────────────────┬────────────────────────────┘
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │                     Alert Taxonomy                      │
       │  Tier 1: Safe (0-30%)        Tier 2: Moderate (31-55%)  │
       │  Tier 3: High Risk (56-79%)  Tier 4: Critical (80-100%) │
       └─────────────────────────────────────────────────────────┘
```

---

## 📊 Benchmark Results

| Module | Benchmark Dataset | Architecture / Method | Metric Score | Comparison Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **Drowsiness** | NTHU-DDD (36 subjects, 65,708 frames) | Random Forest (20 temporal features, 30-frame window) | **93.17% Acc**<br>**93.79% F1**<br>**98.09% AUC** | Outperforms Lamba et al. (2025) baseline (90.48% Acc, 83.87% Recall) |
| **Affective** | FER2013 (7 emotion classes) | ResNet-34 (Focal Loss $\gamma=2.0$, Cosine Annealing) | **70.24% Test Acc** | Outperforms MobileNetV2 (58%) & CLCM (63%) lightweight baselines |
| **Distraction** | COCO Class 67 (Phone) | YOLOv8n ($conf \ge 0.60$) | **Real-time (<15ms)** | 85% safety hard floor triggered |

---

## 📁 Repository Structure

```
Driver-Risk-Assesment-System/
├── app.py                      # Main real-time DRAS application
├── requirements.txt            # Dependency manifest
├── README.md                   # Project documentation
├── .gitignore                  # Environment, dataset & cache filters
├── 22BCE1371_Report_Signed.pdf # Signed academic project report
│
├── proj/                       # Core Perception & Risk Engine (Appendix A1)
│   ├── __init__.py
│   ├── analyzers.py            # Drowsiness, Pose, Emotion, & Object detectors
│   ├── logic.py                # solvePnP, late-fusion formula, & alert taxonomy
│   ├── utils.py                # Geometric utilities & background audio alerts
│   └── face_landmarker.task    # MediaPipe Face Mesh model asset (3.7 MB)
│
├── drowsiness/                 # NTHU-DDD Drowsiness Pipeline (Appendix A1.6 & A2)
│   ├── feature_extraction.py   # 20-feature 30-frame temporal extraction
│   ├── train_evaluate.py       # RF, MLP, KNN, SVM benchmark training
│   ├── classifier_comparison.png # Figure 5.1 in report
│   ├── confusion_matrix_final.png# Figure 5.2 in report
│   └── evaluation_results.txt  # Benchmark metrics output
│
├── emotion_fer/                # FER2013 Affective Pipeline (Section 4.4 & 5.3)
│   ├── emotion_resnet.ipynb    # ResNet-34 Focal Loss fine-tuning notebook
│   ├── evaluate_fer.py         # FER evaluation script
│   └── FER2013_Confusion_Matrix.png # Figure 5.3 in report
│
├── models/                     # Production Model Weights (< 100 MB each)
│   ├── best_fer_resnet34.pth   # Trained ResNet-34 state dict (85.3 MB)
│   ├── drowsy_classifier.pkl   # Compressed RF classifier (47.4 MB)
│   └── yolov8n.pt              # YOLOv8n phone detector (6.5 MB)
│
└── scripts/                    # Utility Scripts
    ├── download_fer2013.py     # FER2013 dataset downloader
    └── download_nthuddd.py     # NTHU-DDD dataset downloader
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Totto1403/Driver-Risk-Assesment-System.git
cd Driver-Risk-Assesment-System
```

### 2. Set Up a Virtual Environment
```bash
# Using Python 3.10+
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux / macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running DRAS

To launch the real-time driver risk monitoring interface:
```bash
python app.py
```

### In-App Controls
* **`R`**: Reset session state, clear temporal buffers, and trigger a fresh 3-second camera and Adaptive EAR baseline calibration.
* **`Q`**: Safely terminate the application.

### Command-Line Arguments
```bash
python app.py --webcam 0             # Target camera index (default: 0)
python app.py --no-fer               # Run without FER (higher FPS on constrained CPUs)
python app.py --no-classifier        # Run using geometric rules only
```

---

## 📜 Citation & Academic Context

If using DRAS in academic research, please refer to the project documentation:
```bibtex
@article{DRAS2026,
  title={Continuous Driver Risk Assessment via Real-Time Affective and Behavioral Tracking},
  author={Bharanee B and Naren Karttik and Sundar J and Sivakami R.},
  school={School of Computer Science and Engineering, Vellore Institute of Technology, Chennai},
  year={2026}
}
```
