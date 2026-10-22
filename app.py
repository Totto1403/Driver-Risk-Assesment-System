"""
DRAS — Integrated Driver Risk Assessment System
================================================
Main Application Execution Script (app.py)
"""

import cv2
import time
import sys
import platform
import argparse
import threading
import numpy as np
from collections import deque

import proj.utils as utils
from proj.analyzers import DrowsinessDetector, FerAnalyzer, ObjectDetector, PoseAnalyzer
from proj.logic import DriverRiskLogic

# ── Paths ─────────────────────────────────────────────────────────────────────
FER_MODEL_PATH       = "models/best_fer_resnet34.pth"
CLASSIFIER_PATH      = "models/drowsy_classifier.pkl"
WEBCAM_ID            = 0
ALERT_DURATION_SEC   = 3

# ── Head pose landmark indices ────────────────────────────────────────────────
_HPE_IDXS = [1, 152, 33, 263, 61, 291]
_MODEL_PTS_3D = np.array([
    [0.0,    0.0,    0.0],
    [0.0,    330.0, -65.0],
    [-225.0, -170.0, -135.0],
    [225.0,  -170.0, -135.0],
    [-150.0,  150.0, -125.0],
    [150.0,   150.0, -125.0],
], dtype=np.float64)

# =============================================================================
# CROSS-PLATFORM ALERT SOUND
# =============================================================================
def _play_beep():
    os_name = platform.system()
    try:
        if os_name == "Windows":
            import winsound
            winsound.Beep(1000, 500)
        elif os_name == "Darwin":
            import subprocess
            subprocess.run(["afplay", "/System/Library/Sounds/Sosumi.aiff"],
                           capture_output=True, timeout=2)
        else:
            try:
                import pygame
                pygame.mixer.init()
                pygame.mixer.music.load("/usr/share/sounds/alsa/Front_Left.wav")
                pygame.mixer.music.play()
            except Exception:
                try:
                    import subprocess
                    subprocess.run(["beep", "-f", "1000", "-l", "500"],
                                   capture_output=True, timeout=2)
                except Exception:
                    print("\a", end="", flush=True)
    except Exception:
        print("\a", end="", flush=True)

def play_alert():
    threading.Thread(target=_play_beep, daemon=True).start()

# =============================================================================
# HEAD POSE ESTIMATOR
# =============================================================================
def compute_head_pose(landmarks, frame_w, frame_h):
    img_pts = np.array([
        (landmarks[i].x * frame_w, landmarks[i].y * frame_h)
        for i in _HPE_IDXS
    ], dtype=np.float64)

    focal  = frame_w
    center = (frame_w / 2, frame_h / 2)
    cam    = np.array([[focal, 0, center[0]],
                       [0, focal, center[1]],
                       [0,     0,         1]], dtype=np.float64)
    dist   = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, _ = cv2.solvePnP(
        _MODEL_PTS_3D, img_pts, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return 0.0, 0.0, 0.0

    R, _ = cv2.Rodrigues(rvec)
    sy   = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy >= 1e-6:
        pitch = np.arctan2(R[2, 1],  R[2, 2])
        yaw   = np.arctan2(-R[2, 0], sy)
        roll  = np.arctan2(R[1, 0],  R[0, 0])
    else:
        pitch = np.arctan2(-R[1, 2], R[1, 1])
        yaw   = np.arctan2(-R[2, 0], sy)
        roll  = 0.0
    return float(np.degrees(yaw)), float(np.degrees(pitch)), float(np.degrees(roll))

# =============================================================================
# DROWSINESS CLASSIFIER WRAPPER (TEMPORAL UPGRADE)
# =============================================================================
class DrowsinessClassifier:
    DROWSY_PROBABILITY_THRESHOLD = 0.55
    WINDOW = 30
    BLINK_THR = 0.20
    BLINK_MINF = 2
    YAWN_THR = 0.55
    YAWN_MINF = 10

    def __init__(self, pkl_path: str):
        import joblib
        data              = joblib.load(pkl_path)
        self._clf         = data["clf"]
        self._scaler      = data["scaler"]
        self._features    = data["features"]
        self._model_name  = data.get("model_name", "Classifier")
        
        # Temporal Memory Buffers
        self.ear_buf   = deque(maxlen=self.WINDOW)
        self.mar_buf   = deque(maxlen=self.WINDOW)
        self.pitch_buf = deque(maxlen=self.WINDOW)
        self.yaw_buf   = deque(maxlen=self.WINDOW)
        
        print(f"[DRAS] Loaded temporal drowsiness classifier: {self._model_name}")

    def reset(self):
        self.ear_buf.clear()
        self.mar_buf.clear()
        self.pitch_buf.clear()
        self.yaw_buf.clear()

    def predict(self, drowsiness_result: dict, raw_yaw: float, raw_pitch: float, raw_roll: float) -> tuple[bool, float]:
        # 1. Extract instantaneous features
        ear_avg  = drowsiness_result.get("ear", 0.0)
        ear_l    = drowsiness_result.get("ear_l", ear_avg) # Fallback to avg if not explicitly returned
        ear_r    = drowsiness_result.get("ear_r", ear_avg)
        mar      = drowsiness_result.get("mar", 0.0)
        ewma_thr = drowsiness_result.get("adaptive_threshold", 0.25)
        
        # 2. Update sliding windows
        self.ear_buf.append(ear_avg)
        self.mar_buf.append(mar)
        self.pitch_buf.append(raw_pitch)
        self.yaw_buf.append(raw_yaw)
        
        ea = np.array(self.ear_buf)
        ma = np.array(self.mar_buf)
        pi = np.array(self.pitch_buf)
        ya = np.array(self.yaw_buf)
        n  = len(ea)

        # 3. Calculate Temporal Features (Real-Time)
        ear_mean = float(np.mean(ea)) if n > 0 else 0.0
        ear_std  = float(np.std(ea)) if n > 0 else 0.0
        ear_min  = float(np.min(ea)) if n > 0 else 0.0
        perclos_15 = float(np.sum(ea < ewma_thr) / n) if n > 0 else 0.0

        blinks = 0
        in_b = False; bf = 0
        for e in ea:
            if e < self.BLINK_THR:
                in_b = True; bf += 1
            else:
                if in_b and bf >= self.BLINK_MINF:
                    blinks += 1
                in_b = False; bf = 0
        blink_rate = float(blinks / (n / 30.0 + 1e-6))

        ear_velocity = float(np.polyfit(np.arange(n, dtype=float), ea, 1)[0]) if n > 1 else 0.0
        mar_mean = float(np.mean(ma)) if n > 0 else 0.0
        mar_max  = float(np.max(ma)) if n > 0 else 0.0

        yawn_flag = 0
        yf = 0
        for m in ma:
            yf = yf + 1 if m > self.YAWN_THR else 0
            if yf >= self.YAWN_MINF:
                yawn_flag = 1; break

        pitch_std = float(np.std(pi)) if n > 1 else 0.0
        yaw_std   = float(np.std(ya)) if n > 1 else 0.0

        nod_count = 0
        if n > 2:
            d = np.diff(pi)
            signs = np.sign(d)
            for i in range(len(signs)-1):
                if signs[i] != 0 and signs[i+1] != 0 and signs[i] != signs[i+1]:
                    nod_count += 1

        # 4. Assemble the exact 20-feature row required by the Random Forest
        feat_dict = {
            "ear_l": ear_l, "ear_r": ear_r, "ear_avg": ear_avg,
            "mar": mar, "yaw": raw_yaw, "pitch": raw_pitch, "roll": raw_roll,
            "ear_ewma_thr": ewma_thr,
            "ear_mean": ear_mean, "ear_std": ear_std, "ear_min": ear_min,
            "perclos_15": perclos_15, "blink_rate": blink_rate,
            "ear_velocity": ear_velocity, "mar_mean": mar_mean, "mar_max": mar_max,
            "yawn_flag": float(yawn_flag), "pitch_std": pitch_std,
            "yaw_std": yaw_std, "head_nod_count": float(nod_count)
        }

        row = np.array([[feat_dict.get(f, 0.0) for f in self._features]], dtype=np.float64)
        row_s = self._scaler.transform(row)
        prob  = float(self._clf.predict_proba(row_s)[0, 1])
        
        return (prob >= self.DROWSY_PROBABILITY_THRESHOLD), prob

# =============================================================================
# OVERLAY RENDERER
# =============================================================================
def draw_overlay(frame, results, is_alert_active, alert_msg, alert_reason, fps, fer_lat, mp_lat, clf_prob=None):
    if results is None:
        return

    h, w = frame.shape[:2]
    panel_width = 320
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_width, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    WHITE, GREEN, YELLOW, RED = (255, 255, 255), (0, 220, 0), (0, 200, 255), (0, 0, 255)
    CYAN, ORANGE = (255, 255, 0), (0, 165, 255)
    FONT = cv2.FONT_HERSHEY_DUPLEX

    score = results["risk_score"]
    risk_color = GREEN if score < 30 else YELLOW if score < 60 else RED

    y = 30
    cv2.putText(frame, "DRIVER STATUS", (15, y), FONT, 0.7, CYAN, 2)
    y += 40
    cv2.putText(frame, f"Risk Score: {score}%", (15, y), FONT, 0.7, risk_color, 2)
    y += 35
    cv2.putText(frame, f"Status: {results['status']}", (15, y), FONT, 0.6, WHITE, 2)
    y += 30
    if results.get("reason"):
        cv2.putText(frame, f"{results['reason']}", (15, y), FONT, 0.55, ORANGE, 2)
        y += 35

    cv2.line(frame, (10, y), (panel_width - 10, y), (80, 80, 80), 1)
    y += 30

    emotion = results.get("emotion", "Neutral")
    em_color = RED if emotion in ["Angry", "Fear", "Sadness"] else CYAN
    cv2.putText(frame, f"Emotion: {emotion}", (15, y), FONT, 0.6, em_color, 2)
    y += 30

    cv2.putText(frame, "Fatigue Metrics", (15, y), FONT, 0.65, CYAN, 2)
    y += 30
    cv2.putText(frame, f"Yawns: {results.get('yawns', 0)}", (15, y), FONT, 0.55, WHITE, 1)
    y += 25
    cv2.putText(frame, f"Microsleeps: {results.get('microsleeps', 0)}", (15, y), FONT, 0.55, WHITE, 1)
    y += 25
    perclos = results.get("perclos", 0.0)
    cv2.putText(frame, f"PERCLOS: {perclos*100:.1f}%", (15, y), FONT, 0.55, WHITE, 1)
    y += 30

    posture = results.get("posture", "Normal")
    pos_color = RED if posture in ["Slouching", "Leaning"] else GREEN
    cv2.putText(frame, f"Posture: {posture}", (15, y), FONT, 0.6, pos_color, 2)
    y += 40

    cv2.putText(frame, "Head Angles", (15, y), FONT, 0.65, CYAN, 2)
    y += 30
    cv2.putText(frame, f"Yaw: {results.get('yaw',0):.1f}", (15, y), FONT, 0.55, WHITE, 1)
    y += 25
    cv2.putText(frame, f"Pitch: {results.get('pitch',0):.1f}", (15, y), FONT, 0.55, WHITE, 1)
    y += 25
    cv2.putText(frame, f"Roll: {results.get('roll',0):.1f}", (15, y), FONT, 0.55, WHITE, 1)
    y += 30

    cv2.line(frame, (10, y), (panel_width - 10, y), (80, 80, 80), 1)
    y += 25
    cv2.putText(frame, f"FPS: {fps:.1f}", (15, y), FONT, 0.55, GREEN, 1)
    y += 25
    cv2.putText(frame, f"FER: {fer_lat:.0f} ms", (15, y), FONT, 0.55, GREEN, 1)
    y += 25
    cv2.putText(frame, f"MP: {mp_lat:.0f} ms", (15, y), FONT, 0.55, GREEN, 1)

    if is_alert_active:
        cx, cy = w // 2, h // 2
        cv2.rectangle(frame, (cx - 300, cy - 60), (cx + 300, cy + 60), (0, 0, 255), -1)
        cv2.putText(frame, alert_msg, (cx - 280, cy - 10), FONT, 0.9, WHITE, 2)
        cv2.putText(frame, alert_reason, (cx - 280, cy + 30), FONT, 0.7, CYAN, 2)

# =============================================================================
# MAIN LOOP
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="DRAS — Integrated Driver Risk Assessment")
    parser.add_argument("--webcam",         type=int,   default=WEBCAM_ID)
    parser.add_argument("--fer-model",      type=str,   default=FER_MODEL_PATH)
    parser.add_argument("--classifier",     type=str,   default=CLASSIFIER_PATH)
    parser.add_argument("--no-fer",         action="store_true", help="Disable FER")
    parser.add_argument("--no-classifier",  action="store_true", help="Use pure geometric EAR")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.webcam)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam {args.webcam}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    drowsiness_detector = DrowsinessDetector(mar_threshold=0.40, use_adaptive_threshold=True)
    pose_analyzer   = PoseAnalyzer()
    object_detector = ObjectDetector()
    driver_logic = DriverRiskLogic()

    fer_analyzer  = None
    if not args.no_fer:
        try:
            fer_analyzer = FerAnalyzer(model_path=args.fer_model)
        except Exception as e:
            print(f"[WARNING] FER model not loaded: {e}")

    drowsiness_clf = None
    if not args.no_classifier:
        try:
            drowsiness_clf = DrowsinessClassifier(args.classifier)
        except Exception as e:
            print(f"[WARNING] Classifier not loaded: {e}")

    alert_msg, alert_reason, alert_end_time = "", "", 0.0
    frame_count, fps_acc, fer_lat_acc, mp_lat_acc = 0, 0.0, 0.0, 0.0
    avg_fps, avg_fer_lat, avg_mp_lat = 0.0, 0.0, 0.0
    clf_prob = None

    print("\n[DRAS] System ready. Press Q to quit. Press R to reset.")

    while cap.isOpened():
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok: continue
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # ── 1. Drowsiness + geometry ──────────────────────────────────────────
        t_mp = time.perf_counter()
        drowsiness_result = drowsiness_detector.process(frame)
        
        # Polished Boot-Up sequence for calibration
        if drowsiness_result.get("calibrating", False):
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.putText(frame, "DRAS SYSTEM BOOTING...", (w//2 - 200, h//2 - 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, "Calibrating Adaptive EAR. Please look forward.", (w//2 - 270, h//2 + 20), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
            cv2.imshow("DRAS — Driver Risk Assessment System", frame)
            
            if cv2.waitKey(5) & 0xFF == ord('q'): break
            continue
            
        mp_lat_acc += (time.perf_counter() - t_mp) * 1000

        # ── 2. Pose landmarks ─────────────────────────────────────────────────
        pose_result   = pose_analyzer.process_frame(frame)
        pose_analyzer.draw_landmarks(frame, pose_result)
        pose_landmarks = pose_result.pose_landmarks.landmark if pose_result.pose_landmarks else None

        # ── 3. Phone detection ────────────────────────────────────────────────
        phone_detected, phone_bboxes = object_detector.process(frame)

        # ── 4. Classify drowsiness ────────────────────────────────────────────
        if drowsiness_result["face_detected"]:
            lm = drowsiness_result["landmarks"]
            raw_yaw, raw_pitch, raw_roll = compute_head_pose(lm, w, h)
            drowsiness_result["_closure_len_for_clf"] = drowsiness_detector._current_closure_len

            if drowsiness_clf is not None:
                is_drowsy_clf, clf_prob = drowsiness_clf.predict(drowsiness_result, raw_yaw, raw_pitch, raw_roll)
                if is_drowsy_clf and not drowsiness_result["drowsy"]:
                    drowsiness_result["drowsy"] = True
            else:
                clf_prob = None

        # ── 5. FER ────────────────────────────────────────────────────────────
        emotion_label = "Neutral"
        if drowsiness_result["face_detected"] and fer_analyzer is not None:
            lm_fer = drowsiness_result["landmarks"]
            face_crop = utils.get_face_crop(frame, lm_fer)
            if face_crop is not None and face_crop.size > 0:
                t_fer        = time.perf_counter()
                face_tensor  = utils.preprocess_fer(face_crop)
                emotion_label = fer_analyzer.predict(face_tensor)
                fer_lat_acc  += (time.perf_counter() - t_fer) * 1000

        # ── 6. Late-Fusion ────────────────────────────────────────────────────
        results     = None
        is_alert_on = time.time() < alert_end_time

        if drowsiness_result["face_detected"]:
            results = driver_logic.update(drowsiness_result, pose_landmarks, frame.shape, emotion_label, phone_detected)
            
            # Pass all geometric fatigue metrics to the UI dictionary
            results["perclos"]     = drowsiness_result.get("perclos", 0.0)
            results["microsleeps"] = drowsiness_result.get("microsleep_count", 0)
            results["ear"]         = drowsiness_result.get("ear", 0.0)
            results["adaptive_threshold"] = drowsiness_result.get("adaptive_threshold", 0.0)
            results["mar"]         = drowsiness_result.get("mar", 0.0)
            results["yaw"]         = raw_yaw
            results["pitch"]       = raw_pitch
            results["roll"]        = raw_roll

            if results["play_alert"] and alert_msg != "RESET":
                play_alert()
                alert_msg      = results["status"]
                alert_reason   = results["reason"]
                alert_end_time = time.time() + ALERT_DURATION_SEC
                is_alert_on    = True

            draw_overlay(frame, results, is_alert_on, alert_msg, alert_reason, avg_fps, avg_fer_lat, avg_mp_lat, clf_prob=clf_prob)

            for (x1, y1, x2, y2) in phone_bboxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, "PHONE DETECTED", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No driver detected.", (340, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # ── 7. FPS accounting ─────────────────────────────────────────────────
        frame_count += 1
        elapsed      = time.perf_counter() - t0
        fps_acc     += 1.0 / (elapsed + 1e-9)

        if frame_count % 10 == 0:
            avg_fps, avg_fer_lat, avg_mp_lat = fps_acc/10, fer_lat_acc/10, mp_lat_acc/10
            fps_acc = fer_lat_acc = mp_lat_acc = 0.0

        cv2.imshow("DRAS — Driver Risk Assessment System", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"): break

        if key == ord("r"):
            print("\n[DRAS] MANUAL RESET TRIGGERED. Recalibrating...")
            
            # 1. Trigger the built-in reset methods
            drowsiness_detector.reset()
            driver_logic.reset()
            
            # 2. Manually tell the DrowsinessDetector to recalibrate the driver's face
            drowsiness_detector._calibrated = False
            drowsiness_detector._calib_ear_samples.clear()
            
            # 3. Clear the emotion buffer and UI alerts
            if fer_analyzer is not None: 
                fer_analyzer._vote_buffer.clear()
            alert_msg, alert_reason, alert_end_time, clf_prob = "", "", 0.0, None

            # 4. Clear the NEW temporal classifier's rolling memory buffers
            if drowsiness_clf is not None:
                drowsiness_clf.reset()

    cap.release()
    cv2.destroyAllWindows()
    print("[DRAS] Stopped.")

if __name__ == "__main__":
    main()