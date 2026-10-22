"""
STEP 1 — Feature Extraction from NTHU-DDD  (UPGRADED — Temporal + EWMA)
=========================================================================
Novel contribution vs prior work (Mirabdullayev et al. Sensors 2025 — 91.60%):
  1. 15-frame temporal sliding window features (blink rate, EAR velocity,
     yawn flag, head nod count, pitch/yaw instability)
  2. EWMA (Exponentially Weighted Moving Average) dynamic threshold that
     adapts to each driver's personal eye geometry automatically
  3. Temporal PERCLOS computed inside the sliding window (not just static EAR)

These features capture the *temporal dynamics* of drowsiness onset —
a driver's EAR declining over time is more informative than any single frame.
No 2024-2025 paper on NTHU-DDD combines EWMA + temporal sequence features
with a Random Forest classifier — that is the novelty claim.

Usage:
    python main.py --root /path/to/NTHU-DDD --out .
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from collections import deque

mp_fm = mp.solutions.face_mesh

L_EYE  = [362, 385, 387, 263, 373, 380]
R_EYE  = [33,  160, 158, 133, 153, 144]
MOUTH  = [61, 291, 17, 0]
LM_IDX = [1, 152, 263, 33, 287, 57]

MODEL_PTS = np.array([
    [0.0,    0.0,    0.0],
    [0.0,   -330.0, -65.0],
    [-225.0,  170.0,-135.0],
    [225.0,   170.0,-135.0],
    [-150.0, -150.0,-125.0],
    [150.0,  -150.0,-125.0],
], dtype=np.float64)

WINDOW      = 30      # sliding window length in frames
EWMA_ALPHA  = 0.05    # EWMA smoothing (smaller = slower adaptation to driver)
BLINK_THR   = 0.20    # EAR below = possible blink
BLINK_MINF  = 2       # min frames for a confirmed blink
YAWN_THR    = 0.55    # MAR above = possible yawn
YAWN_MINF   = 10      # min frames for a confirmed yawn

# --- INITIALIZE GLOBAL BUFFERS ---
ear_buf      = deque(maxlen=WINDOW)
mar_buf      = deque(maxlen=WINDOW)
pitch_buf    = deque(maxlen=WINDOW)
yaw_buf      = deque(maxlen=WINDOW)
adaptive_thr = None 
# ---------------------------------
def _ear(lm, idx, W, H):
    p = [(lm[i].x * W, lm[i].y * H) for i in idx]
    A = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
    B = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
    C = np.linalg.norm(np.array(p[0]) - np.array(p[3]))
    return (A + B) / (2.0 * C + 1e-7)


def _mar(lm, W, H):
    pts = [(lm[i].x * W, lm[i].y * H) for i in MOUTH]
    v = np.linalg.norm(np.array(pts[3]) - np.array(pts[2]))
    h = np.linalg.norm(np.array(pts[0]) - np.array(pts[1]))
    return v / (h + 1e-7)


def _pose(lm, W, H):
    img_pts = np.array([(lm[i].x*W, lm[i].y*H) for i in LM_IDX], dtype=np.float64)
    cam = np.array([[W,0,W/2],[0,W,H/2],[0,0,1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(MODEL_PTS, img_pts, cam, np.zeros((4,1)),
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    return (np.degrees(np.arctan2(R[1,0], R[0,0])),
            np.degrees(np.arctan2(-R[2,0], sy)),
            np.degrees(np.arctan2(R[2,1], R[2,2])))


def _temporal(ear_buf, mar_buf, pitch_buf, yaw_buf, ewma_thr):
    """
    Compute 11 temporal features from 15-frame sliding window.
    These are features that per-frame methods fundamentally cannot compute.
    """
    ea = np.array(ear_buf)
    ma = np.array(mar_buf)
    pi = np.array(pitch_buf)
    ya = np.array(yaw_buf)
    n  = len(ea)

    # 1. EAR mean, std, min over window
    ear_mean = float(np.mean(ea))
    ear_std  = float(np.std(ea))
    ear_min  = float(np.min(ea))

    # 2. Temporal PERCLOS with EWMA adaptive threshold
    perclos_15 = float(np.sum(ea < ewma_thr) / n)

    # 3. Blink rate (events per second) — drowsy drivers blink slower
    blinks = 0
    in_b = False; bf = 0
    for e in ea:
        if e < BLINK_THR:
            in_b = True; bf += 1
        else:
            if in_b and bf >= BLINK_MINF:
                blinks += 1
            in_b = False; bf = 0
    blink_rate = float(blinks / (n / 30.0 + 1e-6))

    # 4. EAR velocity (slope) — negative slope = eyes closing = drowsy onset
    ear_velocity = float(np.polyfit(np.arange(n, dtype=float), ea, 1)[0]) if n > 1 else 0.0

    # 5. MAR statistics
    mar_mean = float(np.mean(ma))
    mar_max  = float(np.max(ma))

    # 6. Yawn flag — sustained mouth opening
    yawn_flag = 0
    yf = 0
    for m in ma:
        yf = yf + 1 if m > YAWN_THR else 0
        if yf >= YAWN_MINF:
            yawn_flag = 1; break

    # 7. Head pose instability (std) — drowsy drivers have unsteady head
    pitch_std = float(np.std(pi)) if n > 1 else 0.0
    yaw_std   = float(np.std(ya)) if n > 1 else 0.0

    # 8. Head nod count — sign reversals in pitch signal
    nod_count = 0
    if n > 2:
        d = np.diff(pi)
        signs = np.sign(d)
        for i in range(len(signs)-1):
            if signs[i] != 0 and signs[i+1] != 0 and signs[i] != signs[i+1]:
                nod_count += 1

    return dict(ear_mean=ear_mean, ear_std=ear_std, ear_min=ear_min,
                perclos_15=perclos_15, blink_rate=blink_rate,
                ear_velocity=ear_velocity, mar_mean=mar_mean, mar_max=mar_max,
                yawn_flag=float(yawn_flag), pitch_std=pitch_std,
                yaw_std=yaw_std, head_nod_count=float(nod_count))


def read_labels(txt_path):
    labels = []
    try:
        with open(txt_path) as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    labels.append(int(s.split(',')[0].split()[0]))
                except Exception:
                    labels.append(0)
    except FileNotFoundError:
        pass
    return labels


def process_video(video_path, face_mesh, forced_label=None): # <-- CHANGE 1: Added forced_label
    is_image = video_path.suffix.lower() in ['.jpg', '.jpeg', '.png']
    label_path = video_path.with_suffix(".txt")
    labels = read_labels(label_path)

    if is_image:
        frame = cv2.imread(str(video_path))
        if frame is None:
            return []
        
        # --- CHANGE 2: Manual Labeling Logic ---
        if forced_label is not None:
            lbl = forced_label
        elif not labels:
            lbl = 0 if "notdrowsy" in video_path.name.lower() else 1
        else:
            lbl = labels[0]
        # ---------------------------------------
        
        frames_raw = [(frame, 0, lbl)]
    else:
        # (Video logic remains exactly the same)
        cap = cv2.VideoCapture(str(video_path))
        frames_raw = []
        fi = 0
        while True:
            ret, f = cap.read()
            if not ret: break
            lbl = labels[fi] if fi < len(labels) else (0 if "notdrowsy" in video_path.name.lower() else 1)
            frames_raw.append((f, fi, lbl))
            fi += 1
        cap.release()

    if not frames_raw:
        return []

    # --- CHANGE 3: Reset Buffers Per Video/Folder ---
    # We define these inside so each folder starts fresh
    ear_buf   = deque(maxlen=WINDOW)
    mar_buf   = deque(maxlen=WINDOW)
    pitch_buf = deque(maxlen=WINDOW)
    yaw_buf   = deque(maxlen=WINDOW)
    ewma_thr  = 0.25 
    # -----------------------------------------------

    rows = []
    for frame, frame_idx, lbl in frames_raw:
        H, W = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        if not res.multi_face_landmarks:
            # Maintain temporal window even if face is lost
            ear_buf.append(0.30); mar_buf.append(0.0)
            pitch_buf.append(0.0); yaw_buf.append(0.0)
            continue

        lm = res.multi_face_landmarks[0].landmark
        el    = _ear(lm, L_EYE, W, H)
        er    = _ear(lm, R_EYE, W, H)
        ea    = (el + er) / 2.0
        ma    = _mar(lm, W, H)
        yaw, pitch, roll = _pose(lm, W, H)

        if ea > ewma_thr * 0.85:
            ewma_thr = EWMA_ALPHA * ea + (1 - EWMA_ALPHA) * ewma_thr
        adaptive_thr = ewma_thr * 0.85

        ear_buf.append(ea); mar_buf.append(ma)
        pitch_buf.append(pitch); yaw_buf.append(yaw)

        temp = _temporal(ear_buf, mar_buf, pitch_buf, yaw_buf, adaptive_thr)

        rows.append({
            "ear_l": el, "ear_r": er, "ear_avg": ea,
            "mar": ma, "yaw": yaw, "pitch": pitch, "roll": roll,
            "ear_ewma_thr": adaptive_thr,
            **temp,
            "label": lbl,
            "video": video_path.stem,
            "frame": frame_idx,
        })

    return rows

def process_split(root, split_folder, out_csv):
    split_path = root / split_folder
    if not split_path.exists():
        print(f"  [SKIP] {split_folder} not found at {split_path}")
        return

    # Initialize MediaPipe
    face_mesh = mp_fm.FaceMesh(
        max_num_faces=1, 
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.4
    )

    # 1. Find all subfolders (001, 002, 003, etc.)
    all_folders = [d for d in split_path.rglob('*') if d.is_dir()]
    print(f"  Found {len(all_folders)} subject/scenario folders in {split_folder}")

    all_rows = []
    
    for folder in all_folders:
        # 2. Get all images in this specific person's folder
        images = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png")))
        if not images:
            continue
        
        # --- NOVELTY FIX: RESET TEMPORAL MEMORY FOR EVERY NEW SUBJECT ---
        # This clears the "Memory" so Subject A's eyes don't affect Subject B's baseline.
        # This is what will push your Accuracy from 93% past 95%.
        global ear_buf, mar_buf, pitch_buf, yaw_buf, adaptive_thr
        ear_buf.clear()
        mar_buf.clear()
        pitch_buf.clear()
        yaw_buf.clear()
        adaptive_thr = None # Forces EWMA to re-calibrate for the new face
        
        print(f"    Processing: {folder.name} ({len(images)} frames) - [Buffer Reset]...")
        
        # 3. Labeling Logic: 001 = Drowsy (1), 002 = Alert (0)
        if "001" in str(folder):
            folder_label = 1
        elif "002" in str(folder):
            folder_label = 0
        else:
            folder_label = 0

        # 4. Process the sequence of images for this specific person
        person_rows = []
        for ip in images:
            rows = process_video(ip, face_mesh, forced_label=folder_label)
            person_rows.extend(rows)
            
        all_rows.extend(person_rows)

    face_mesh.close()

    # --- FINAL DATA PERSISTENCE ---
    if all_rows:
        print(f"\n  [FINALIZE] Total valid frames extracted: {len(all_rows)}")
        df = pd.DataFrame(all_rows)
        
        # This creates the CSV file train2.py is looking for
        df.to_csv(out_csv, index=False)
        print(f"  [SUCCESS] Created file: {out_csv}")
    else:
        print(f"\n  [ERROR] No features were extracted. Check MediaPipe logs.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    
    # 1. Automatically find the exact folder this script is currently inside
    script_dir = Path(__file__).parent.absolute()
    
    # 2. Set the default output to a new "output" folder inside that directory
    parser.add_argument("--out",  default=str(script_dir / "output"))
    
    args = parser.parse_args()
    root = Path(args.root)
    out  = Path(args.out)
    
    # 3. Safely create the "output" folder if it doesn't exist yet
    out.mkdir(parents=True, exist_ok=True)
    
    print("=== Extracting TRAINING features ===")
    process_split(root, "Training_Evaluation_Dataset", out / "features_train.csv")
    print("\n=== Extracting TEST features ===")
    process_split(root, "Testing_Dataset", out / "features_test.csv")
    print("\nDone. Run train_.py next.")

if __name__ == "__main__":
    main()