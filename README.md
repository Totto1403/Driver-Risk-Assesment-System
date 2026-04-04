# DRAS: Driver Risk Assessment System

DRAS is a multimodal, edge-computable pipeline that monitors physical fatigue, affective state, postural deviation, and visual-manual distractions. Instead of a binary "Drowsy/Alert" output, it fuses these streams into a continuous, interpretable Risk Score (0–100%) using an empirically validated 70:30 Visual-to-Affective weighting.

## Key Features
* **Adaptive EAR (A-EAR):** Eliminates false positives for drivers with naturally narrow eyes by calibrating a personal baseline during startup.
* **Affective Risk Integration:** Utilizes a custom-trained ResNet-34 model (67.96% test accuracy on FER2013) to factor cognitive-emotional impairment (e.g., Anger, Fear) into the driving risk profile.
* **Multi-Scale Fatigue Tracking:** Concurrently tracks Microsleeps (<1s closures), PERCLOS (30-second eye closure percentage), and sustained drowsiness.
* **Visual-Manual Distraction:** YOLOv8n integration for real-time mobile phone detection, applying an immediate safety-critical risk floor if a device is detected.
* **CPU-Viable Edge Deployment:** The geometric pipeline executes fully on CPU, allowing the system to run on standard hardware with under 100ms latency.

---

## 🛠️ Installation Setup

**1. Create a Virtual Environment**
It is highly recommended to run this project inside a virtual environment to prevent dependency conflicts.
`python -m venv venv`

**2. Activate the Virtual Environment**
* **Windows:**
  `venv\Scripts\activate`
* **macOS / Linux:**
  `source venv/bin/activate`

**3. Install Dependencies**
Ensure your virtual environment is active, then install the required packages (PyTorch, MediaPipe, Ultralytics YOLOv8, OpenCV, etc.).
`pip install -r requirements.txt`

---

## 🚀 Running the Application

To start the real-time driver assessment pipeline using your default webcam, simply run:
`python app.py`

### In-App Controls
* Press **`R`** to safely **Reset** the system. This clears all fatigue history, temporal buffers, and initiates a fresh 3-second camera calibration for your Adaptive EAR baseline.
* Press **`Q`** to **Quit** the application safely.

### Command Line Arguments
You can customize the application startup by passing the following flags:
* `--webcam <int>` : Change the target camera index (default is `0`).
* `--no-fer` : Disables the facial expression recognition module for faster, geometry-only testing.
* `--no-classifier` : Bypasses the Random Forest classifier and relies purely on geometric EAR thresholds.

---