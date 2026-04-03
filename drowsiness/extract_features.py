"""
DRAS Feature Extraction — Enhanced Temporal Features
======================================================

Extracts a richer feature set from NTHU-DDD frames including:
  - Base geometry: EAR (bilateral + per-eye), MAR, head pose (yaw/pitch/roll)
  - Adaptive EAR normalisation (A-EAR)
  - PERCLOS at three window sizes (30, 60, 90 frames)
  - Microsleep count (rolling 60 seconds)
  - Yawn count (rolling 60 seconds)
  - Blink dynamics: current closure length
  - Blink rate (blinks per 30-second window)
  - EAR velocity (frame-to-frame rate of change — detects rapid closure onset)
  - Head pose variability (rolling std of yaw over 30 frames — detects nodding)

The larger feature set allows the classifier to differentiate fatigue stages
more precisely, particularly PERCLOS at multiple timescales and EAR velocity.

DATASET FOLDER STRUCTURE EXPECTED:
  Training_Evaluation_Dataset/
    001/   <-- Drowsy images
    002/   <-- Alert images

USAGE:
  python extract_features.py --root . --out output

NEXT STEP:
  python train_proper.py --train output/features_train_advanced.csv
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from collections import deque
import re

mp_fm = mp.solutions.face_mesh

L_EYE = [362, 385, 387, 263, 373, 380]
R_EYE = [33,  160, 158, 133, 153, 144]
MOUTH = [61, 291, 17, 0]
LM_IDX = [1, 152, 263, 33, 287, 57]

MODEL_PTS = np.array([
    [0.0,    0.0,    0.0],
    [0.0,   -330.0, -65.0],
    [-225.0,  170.0, -135.0],
    [225.0,   170.0, -135.0],
    [-150.0, -150.0, -125.0],
    [150.0,  -150.0, -125.0],
], dtype=np.float64)


def ear(lm, idx, W, H):
    p = [(lm[i].x * W, lm[i].y * H) for i in idx]
    A = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
    B = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
    C = np.linalg.norm(np.array(p[0]) - np.array(p[3]))
    return (A + B) / (2.0 * C + 1e-7)


def mar(lm, W, H):
    pts = [(lm[i].x * W, lm[i].y * H) for i in MOUTH]
    vert  = np.linalg.norm(np.array(pts[3]) - np.array(pts[2]))
    horiz = np.linalg.norm(np.array(pts[0]) - np.array(pts[1]))
    return vert / (horiz + 1e-7)


def head_pose(lm, W, H):
    img_pts = np.array([(lm[i].x * W, lm[i].y * H) for i in LM_IDX], dtype=np.float64)
    cam = np.array([[W, 0, W/2], [0, W, H/2], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1))
    ok, rvec, _ = cv2.solvePnP(MODEL_PTS, img_pts, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
    yaw   = np.degrees(np.arctan2(rmat[1, 0],  rmat[0, 0]))
    roll  = np.degrees(np.arctan2(rmat[2, 1],  rmat[2, 2]))
    return yaw, pitch, roll


def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'([0-9]+)', str(s))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",  required=True, help="Path to dataset root folder")
    parser.add_argument("--out",   default="output", help="Output directory")
    args = parser.parse_args()

    root    = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "features_train_advanced.csv"

    print("=" * 55)
    print("  DRAS Feature Extraction — Enhanced Temporal")
    print("=" * 55)

    dataset_dir = root / "Training_Evaluation_Dataset"
    folder_map  = {"001": 1, "002": 0}   # 001 = Drowsy, 002 = Alert

    all_images = []
    for folder, label in folder_map.items():
        d = dataset_dir / folder
        if d.exists():
            found = [(p, label)
                     for p in d.rglob("*.*")
                     if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
            all_images.extend(found)
            print(f"  {folder} ({('Drowsy' if label else 'Alert'):<6}): {len(found):,} images")
        else:
            print(f"  [WARNING] Folder not found: {d}")

    if not all_images:
        print("[ERROR] No images found. Check dataset path.")
        return

    print(f"\nTotal: {len(all_images):,} images")
    print("Sorting to preserve temporal order within each scenario...")
    all_images.sort(key=lambda x: natural_sort_key(x[0].name))

    face_mesh = mp_fm.FaceMesh(
        max_num_faces=1, refine_landmarks=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.4
    )

    EAR_THRESH = 0.30
    MAR_THRESH = 0.55

    # ── Per-scenario state ────────────────────────────────────────────────────
    current_scenario  = ""
    scenario_ears     = []
    baseline_ear      = 0.30

    # Rolling windows
    ear_buf_30  = deque(maxlen=30)    # PERCLOS over 1 second
    ear_buf_60  = deque(maxlen=60)    # PERCLOS over 2 seconds
    ear_buf_90  = deque(maxlen=90)    # PERCLOS over 3 seconds
    ear_velocity_buf = deque(maxlen=5)  # For EAR rate-of-change
    yaw_buf_30  = deque(maxlen=30)    # Head pose variability
    blink_buf_30 = deque(maxlen=900)  # Blink events (30 s @ 30 FPS)
    microsleep_buf = deque(maxlen=1800)
    yawn_buf    = deque(maxlen=1800)

    closure_frames = 0
    yawn_frames    = 0
    prev_ear       = baseline_ear

    all_rows = []

    print(f"\nExtracting features from {len(all_images):,} frames...")

    for i, (img_path, label) in enumerate(all_images):
        # Scenario key from first two filename parts
        parts    = img_path.stem.split("_")
        scenario = "_".join(parts[:2]) if len(parts) >= 2 else img_path.parent.name

        # Reset all state when scenario changes
        if scenario != current_scenario:
            current_scenario  = scenario
            scenario_ears     = []
            baseline_ear      = 0.30
            ear_buf_30.clear()
            ear_buf_60.clear()
            ear_buf_90.clear()
            ear_velocity_buf.clear()
            yaw_buf_30.clear()
            blink_buf_30.clear()
            microsleep_buf.clear()
            yawn_buf.clear()
            closure_frames = 0
            yawn_frames    = 0
            prev_ear       = 0.30

        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        H, W = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res  = face_mesh.process(rgb)

        if not res.multi_face_landmarks:
            # No face: pad buffers with baseline values
            for buf in (ear_buf_30, ear_buf_60, ear_buf_90):
                buf.append(baseline_ear)
            microsleep_buf.append(0)
            yawn_buf.append(0)
            blink_buf_30.append(0)
            closure_frames = 0
            prev_ear = baseline_ear
            continue

        lm = res.multi_face_landmarks[0].landmark

        # ── Base geometry ─────────────────────────────────────────────────────
        el  = ear(lm, L_EYE, W, H)
        er  = ear(lm, R_EYE, W, H)
        ea  = (el + er) / 2.0
        ma  = mar(lm, W, H)
        yaw, pitch, roll = head_pose(lm, W, H)

        # ── Adaptive baseline (A-EAR) ─────────────────────────────────────────
        if len(scenario_ears) < 45:
            scenario_ears.append(ea)
            if len(scenario_ears) > 10:
                baseline_ear = np.percentile(scenario_ears, 75)
        ear_norm = ea / (baseline_ear + 1e-6)

        # ── EAR velocity ──────────────────────────────────────────────────────
        # Rate of change of EAR — rapid drops indicate sudden eye closure onset
        ear_delta = ea - prev_ear
        ear_velocity_buf.append(ear_delta)
        ear_velocity = float(np.mean(ear_velocity_buf))  # negative = closing
        prev_ear = ea

        # ── PERCLOS at three timescales ───────────────────────────────────────
        for buf in (ear_buf_30, ear_buf_60, ear_buf_90):
            buf.append(ea)
        perclos_30 = sum(1 for e in ear_buf_30 if e < EAR_THRESH) / (len(ear_buf_30) or 1)
        perclos_60 = sum(1 for e in ear_buf_60 if e < EAR_THRESH) / (len(ear_buf_60) or 1)
        perclos_90 = sum(1 for e in ear_buf_90 if e < EAR_THRESH) / (len(ear_buf_90) or 1)

        # ── Head pose variability ─────────────────────────────────────────────
        # Rolling std of yaw over last 30 frames — detects head nodding
        yaw_buf_30.append(yaw)
        yaw_std = float(np.std(yaw_buf_30)) if len(yaw_buf_30) > 5 else 0.0

        # ── Microsleep tracking ───────────────────────────────────────────────
        is_microsleep = 0
        if ea < EAR_THRESH:
            closure_frames += 1
        else:
            if 3 <= closure_frames <= 24:
                is_microsleep = 1
            closure_frames = 0
        microsleep_buf.append(is_microsleep)
        microsleeps_60s = sum(microsleep_buf)

        # ── Blink tracking ────────────────────────────────────────────────────
        # A blink = very short closure (1-2 frames). Excessive blinking = fatigue.
        is_blink = 0
        if closure_frames == 2:   # exactly 2-frame closure = normal blink complete
            is_blink = 1
        blink_buf_30.append(is_blink)
        blinks_30s = sum(blink_buf_30)

        # ── Yawn tracking ─────────────────────────────────────────────────────
        is_yawn = 0
        if ma > MAR_THRESH:
            yawn_frames += 1
        else:
            if yawn_frames > 20:
                is_yawn = 1
            yawn_frames = 0
        yawn_buf.append(is_yawn)
        yawns_60s = sum(yawn_buf)

        all_rows.append({
            # Core EAR features
            "ear_avg":        ea,
            "ear_norm":       ear_norm,
            "ear_l":          el,
            "ear_r":          er,
            "ear_velocity":   ear_velocity,
            # Mouth
            "mar":            ma,
            # Head pose
            "yaw":            yaw,
            "pitch":          pitch,
            "roll":           roll,
            "yaw_std":        yaw_std,
            # Multi-scale PERCLOS
            "perclos":        perclos_90,   # primary (keep name for compatibility)
            "perclos_30":     perclos_30,
            "perclos_60":     perclos_60,
            "perclos_90":     perclos_90,
            # Event counts
            "closure_len":    closure_frames,
            "microsleeps":    microsleeps_60s,
            "blinks_30s":     blinks_30s,
            "yawns":          yawns_60s,
            # Metadata
            "label":          label,
            "scenario":       scenario,
        })

        if i % 1000 == 0:
            print(f"  {i:>7,}/{len(all_images):,}  |  "
                  f"MS={microsleeps_60s}  YW={yawns_60s}  "
                  f"P90={perclos_90:.2f}  EAR={ea:.3f}",
                  end="\r")

    face_mesh.close()

    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)

    print(f"\n\nSUCCESS: {len(df):,} frames extracted to {out_csv}")
    print(f"  Alert  frames: {(df['label']==0).sum():,}")
    print(f"  Drowsy frames: {(df['label']==1).sum():,}")
    print(f"  Scenarios:     {df['scenario'].nunique()}")
    print(f"\nNext step: python train_proper.py --train {out_csv}")


if __name__ == "__main__":
    main()
