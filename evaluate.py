"""
evaluate.py  —  DRAS Drowsiness Module Evaluation
===================================================
FIXES vs original:
  - EAR threshold NOW matches DrowsinessDetector default (0.30) for fair comparison.
  - Added F1-score, Specificity, and a full threshold sensitivity sweep plot.
  - Results table printed to console and saved as PNG.
"""

import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

from proj.analyzers import DrowsinessDetector

# ==========================================
# CONFIGURATION
# ==========================================
# IMPORTANT: Keep this aligned with DrowsinessDetector's default threshold.
# Changing only one side produces misleading evaluation results.
EVAL_EAR_THRESHOLD = 0.28 #Tune

DATASET_DIR = "./dataset"
DROWSY_DIR  = os.path.join(DATASET_DIR, "drowsy")
ALERT_DIR   = os.path.join(DATASET_DIR, "notdrowsy")

# Range for sensitivity sweep
SWEEP_THRESHOLDS = np.arange(0.15, 0.45, 0.025)

# ==========================================
# DETECTOR  (adaptive threshold OFF for
# reproducible threshold sweep)
# ==========================================
detector = DrowsinessDetector(
    ear_threshold=EVAL_EAR_THRESHOLD,
    use_adaptive_threshold=False   # keep fixed for evaluation
)

# ==========================================
# PROCESSING LOOP
# ==========================================
def process_folder(folder_path: str, ground_truth_label: int,
                   threshold: float) -> tuple[list, list]:
    extensions  = ["*.jpg", "*.jpeg", "*.png"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, "**", ext), recursive=True))

    y_true_out, y_pred_out = [], []
    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        results = detector.process(frame)
        if results["face_detected"]:
            prediction = 1 if results["ear"] < threshold else 0
            y_true_out.append(ground_truth_label)
            y_pred_out.append(prediction)
    return y_true_out, y_pred_out


# ==========================================
# SINGLE-THRESHOLD FULL EVALUATION
# ==========================================
def run_full_evaluation(threshold: float = EVAL_EAR_THRESHOLD):
    print(f"\n{'='*55}")
    print(f"  DRAS Evaluation  |  EAR Threshold = {threshold:.3f}")
    print(f"{'='*55}")

    y_true, y_pred = [], []
    y_t, y_p = process_folder(ALERT_DIR,  ground_truth_label=0, threshold=threshold)
    y_true += y_t; y_pred += y_p
    y_t, y_p = process_folder(DROWSY_DIR, ground_truth_label=1, threshold=threshold)
    y_true += y_t; y_pred += y_p

    if not y_true:
        print("No images processed. Check DATASET_DIR path.")
        return

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"\n  Total frames evaluated : {len(y_true)}")
    print(f"  True Positives (TP)    : {tp}")
    print(f"  True Negatives (TN)    : {tn}")
    print(f"  False Positives (FP)   : {fp}")
    print(f"  False Negatives (FN)   : {fn}")
    print(f"\n  Accuracy               : {acc*100:.2f}%")
    print(f"  Precision              : {prec*100:.2f}%")
    print(f"  Recall (Sensitivity)   : {rec*100:.2f}%")
    print(f"  Specificity            : {spec*100:.2f}%")
    print(f"  F1-Score               : {f1*100:.2f}%")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
        xticklabels=["Alert", "Drowsy"],
        yticklabels=["Alert", "Drowsy"],
    )
    axes[0].set_title(
        f"Confusion Matrix  (EAR thr = {threshold:.3f})\n"
        f"Acc={acc*100:.2f}%  Prec={prec*100:.2f}%  "
        f"Rec={rec*100:.2f}%  F1={f1*100:.2f}%",
        fontsize=11,
    )
    axes[0].set_ylabel("Actual Label")
    axes[0].set_xlabel("Predicted Label")

    # ── Per-class bar chart ───────────────────────────────────────────────────
    metrics_labels  = ["Accuracy", "Precision", "Recall", "Specificity", "F1"]
    metrics_values  = [acc, prec, rec, spec, f1]
    bar_colors      = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    axes[1].bar(metrics_labels, [v * 100 for v in metrics_values], color=bar_colors)
    axes[1].set_ylim(0, 110)
    axes[1].set_ylabel("Score (%)")
    axes[1].set_title("Performance Metrics Summary")
    for i, v in enumerate(metrics_values):
        axes[1].text(i, v * 100 + 1, f"{v*100:.1f}%", ha="center", fontsize=9)

    plt.tight_layout()
    out_file = "final_confusion_matrix.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"\n  Saved → {out_file}")
    plt.show()


# ==========================================
# EAR THRESHOLD SENSITIVITY SWEEP
# ==========================================
def run_sensitivity_sweep():
    """Sweeps EAR threshold and plots Precision/Recall/F1 curves."""
    print("\n[Sensitivity Sweep] collecting EAR values for all images...")

    # Collect raw EAR readings once (threshold-independent)
    ear_data = []   # list of (true_label, ear_value)
    for label, folder in [(0, ALERT_DIR), (1, DROWSY_DIR)]:
        extensions  = ["*.jpg", "*.jpeg", "*.png"]
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
        for img_path in image_paths:
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            results = detector.process(frame)
            if results["face_detected"]:
                ear_data.append((label, results["ear"]))

    if not ear_data:
        print("No data collected for sweep.")
        return

    y_true_all = [d[0] for d in ear_data]
    ear_all    = [d[1] for d in ear_data]

    precisions, recalls, f1s, accs = [], [], [], []
    for thr in SWEEP_THRESHOLDS:
        y_pred = [1 if e < thr else 0 for e in ear_all]
        precisions.append(precision_score(y_true_all, y_pred, zero_division=0))
        recalls.append(recall_score(y_true_all, y_pred, zero_division=0))
        f1s.append(f1_score(y_true_all, y_pred, zero_division=0))
        accs.append(accuracy_score(y_true_all, y_pred))

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(SWEEP_THRESHOLDS, [p * 100 for p in precisions], "b-o",  label="Precision",       ms=4)
    plt.plot(SWEEP_THRESHOLDS, [r * 100 for r in recalls],    "r-s",  label="Recall",          ms=4)
    plt.plot(SWEEP_THRESHOLDS, [f * 100 for f in f1s],        "g-^",  label="F1-Score",        ms=4)
    plt.plot(SWEEP_THRESHOLDS, [a * 100 for a in accs],       "k--v", label="Accuracy",        ms=4)
    plt.axvline(EVAL_EAR_THRESHOLD, color="purple", linestyle=":", linewidth=2,
                label=f"Current thr = {EVAL_EAR_THRESHOLD:.2f}")
    plt.xlabel("EAR Threshold")
    plt.ylabel("Score (%)")
    plt.title("EAR Threshold Sensitivity Analysis — NTHU-DDD")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(SWEEP_THRESHOLDS, [f"{t:.2f}" for t in SWEEP_THRESHOLDS], rotation=45)
    plt.tight_layout()
    sweep_file = "sensitivity_sweep.png"
    plt.savefig(sweep_file, dpi=300, bbox_inches="tight")
    print(f"\n  Sensitivity sweep saved → {sweep_file}")
    plt.show()


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    run_full_evaluation(threshold=EVAL_EAR_THRESHOLD)
    run_sensitivity_sweep()