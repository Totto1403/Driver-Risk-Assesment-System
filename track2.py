"""
DRAS — Integrated Driver Risk Assessment System
================================================
This is the fully integrated main.py. It combines:
  1. DrowsinessDetector (MediaPipe geometry + temporal features)
  2. Trained XGBoost / Random Forest classifier (drowsy_classifier.pkl)
  3. FerAnalyzer (ResNet-34 emotion recognition)
  4. PoseAnalyzer (MediaPipe body posture)
  5. ObjectDetector (YOLOv8n phone detection)
  6. DriverRiskLogic (late-fusion: R = 0.70 * V + 0.30 * E)

Pipeline:
  Frame -> DrowsinessDetector (EAR, MAR, PERCLOS, Microsleeps)
        -> Head Pose Estimator (Yaw, Pitch, Roll)
        -> Feature Vector -> Drowsiness Classifier (RF / XGBoost)
        -> FerAnalyzer (Emotion)
        -> PoseAnalyzer (Posture)
        -> ObjectDetector (Phone)
        -> DriverRiskLogic (fused Risk Score)
        -> Overlay + Alert

Formulas (consistent with paper):
  EAR   = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)
  MAR   = ||Qtop-Qbottom|| / ||Qleft-Qright||
  t_EAR = clip(EAR_baseline - 0.08, 0.18, 0.35)  [adaptive]
  F     = min(100, F_EAR[0-40] + F_PERCLOS[0-30] + F_ms[0-20] + F_yawn[0-10])
  V     = 0.50*F + 0.25*P + 0.25*D
  R     = 0.70*V + 0.30*E   [phone detected -> R >= 85]

Usage:
  python main.py
  python main.py --no-fer          # skip FER model (faster, for testing)
  python main.py --no-classifier   # skip RF/XGB, use pure geometric threshold
  python main.py --webcam 1        # use a different camera index
"""

import cv2
import time
import sys
import platform
import argparse
import threading
import numpy as np

import proj.utils as utils
from proj.analyzers import DrowsinessDetector, FerAnalyzer, ObjectDetector, PoseAnalyzer
from proj.logic import DriverRiskLogic

# ── Paths ─────────────────────────────────────────────────────────────────────
FER_MODEL_PATH       = "models/resnet34-epoch.12-val_loss.0.494-val_acc.0.846-val_f1.0.843.ckpt"
CLASSIFIER_PATH      = "drowsiness/drowsy_classifier.pkl"
WEBCAM_ID            = 0
ALERT_DURATION_SEC   = 3

# ── Head pose landmark indices (consistent with DriverRiskLogic / paper) ──────
# Order: Nose, Chin, Left Eye corner, Right Eye corner, Left Mouth, Right Mouth
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
            # Linux: try pygame, then beep, then terminal bell
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
# HEAD POSE ESTIMATOR (standalone, for classifier feature extraction)
# =============================================================================
def compute_head_pose(landmarks, frame_w, frame_h):
    """
    Returns (yaw_deg, pitch_deg, roll_deg).
    Consistent with DriverRiskLogic._get_head_pose() and the paper.
    """
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
# DROWSINESS CLASSIFIER WRAPPER
# =============================================================================
class DrowsinessClassifier:
    """
    Wraps the trained sklearn / XGBoost pipeline from drowsy_classifier.pkl.
    Adds the classifier's probabilistic output to the drowsiness result dict.

    Feature order must match train_proper.py:
      ear_avg, ear_norm, mar, yaw, pitch, roll,
      perclos, closure_len, yawns [, microsleeps]
    """
    DROWSY_PROBABILITY_THRESHOLD = 0.55   # >= this prob -> drowsy

    def __init__(self, pkl_path: str):
        import joblib
        data              = joblib.load(pkl_path)
        self._clf         = data["clf"]
        self._scaler      = data["scaler"]
        self._features    = data["features"]
        self._model_name  = data.get("model_name", "Classifier")
        print(f"[DRAS] Loaded drowsiness classifier: {self._model_name}")
        print(f"[DRAS] Feature set ({len(self._features)}): {self._features}")

    def predict(self, drowsiness_result: dict,
                yaw: float, pitch: float, roll: float) -> tuple[bool, float]:
        """
        Returns (is_drowsy: bool, drowsy_probability: float).
        Builds the feature vector from the DrowsinessDetector result dict
        and the head-pose angles computed before calling this method.
        """
        ear_avg   = drowsiness_result.get("ear", 0.0)
        threshold = drowsiness_result.get("adaptive_threshold", 0.30)
        ear_norm  = ear_avg / (threshold + 1e-6)
        mar       = drowsiness_result.get("mar", 0.0)
        perclos   = drowsiness_result.get("perclos", 0.0)
        cl_len    = drowsiness_result.get("_closure_len_for_clf", 0)
        yawns     = drowsiness_result.get("yawn_count", 0)
        microsleeps = drowsiness_result.get("microsleep_count", 0)

        # Build dict and pull only the features the model was trained on
        feat_dict = {
            "ear_avg":     ear_avg,
            "ear_norm":    ear_norm,
            "mar":         mar,
            "yaw":         yaw,
            "pitch":       pitch,
            "roll":        roll,
            "perclos":     perclos,
            "closure_len": cl_len,
            "yawns":       yawns,
            "microsleeps": microsleeps,
        }
        row = np.array([[feat_dict.get(f, 0.0) for f in self._features]], dtype=np.float64)

        row_s = self._scaler.transform(row)
        prob  = float(self._clf.predict_proba(row_s)[0, 1])
        return (prob >= self.DROWSY_PROBABILITY_THRESHOLD), prob


# =============================================================================
# OVERLAY RENDERER
# =============================================================================
def draw_overlay(frame, results, is_alert_active,
                 alert_msg, alert_reason,
                 fps, fer_lat, mp_lat,
                 clf_prob=None):

    if results is None:
        return

    h, w = frame.shape[:2]

    # ================= PANEL =================
    panel_width = 320
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_width, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # ================= COLORS =================
    WHITE = (255, 255, 255)
    GREEN = (0, 220, 0)
    YELLOW = (0, 200, 255)
    RED = (0, 0, 255)
    CYAN = (255, 255, 0)
    ORANGE = (0, 165, 255)
    GRAY = (180, 180, 180)

    # ================= FONT =================
    FONT = cv2.FONT_HERSHEY_DUPLEX

    # ================= RISK =================
    score = results["risk_score"]

    if score < 30:
        risk_color = GREEN
    elif score < 60:
        risk_color = YELLOW
    else:
        risk_color = RED

    y = 30

    cv2.putText(frame, "DRIVER STATUS", (15, y),
                FONT, 0.7, CYAN, 2)

    y += 40

    cv2.putText(frame, f"Risk Score: {score}%",
                (15, y), FONT, 0.7, risk_color, 2)

    y += 35

    cv2.putText(frame, f"Status: {results['status']}",
                (15, y), FONT, 0.6, WHITE, 2)

    y += 30

    if results.get("reason"):
        cv2.putText(frame, f"{results['reason']}",
                    (15, y), FONT, 0.55, ORANGE, 2)
        y += 35

    # ================= DIVIDER =================
    cv2.line(frame, (10, y), (panel_width - 10, y), (80, 80, 80), 1)
    y += 30

    # ================= EMOTION =================
    emotion = results.get("emotion", "Neutral")
    em_color = RED if emotion in ["Angry", "Fear"] else CYAN

    cv2.putText(frame, f"Emotion: {emotion}",
                (15, y), FONT, 0.6, em_color, 2)

    y += 30

    # ================= FATIGUE =================
    cv2.putText(frame, "Fatigue Metrics",
                (15, y), FONT, 0.65, CYAN, 2)

    y += 30

    cv2.putText(frame, f"Yawns: {results.get('yawns', 0)}",
                (15, y), FONT, 0.55, WHITE, 1)

    y += 25

    cv2.putText(frame, f"Microsleeps: {results.get('microsleeps', 0)}",
                (15, y), FONT, 0.55, WHITE, 1)

    y += 25

    perclos = results.get("perclos", 0.0)
    cv2.putText(frame, f"PERCLOS: {perclos*100:.1f}%",
                (15, y), FONT, 0.55, WHITE, 1)

    y += 30

    # ================= POSTURE =================
    posture = results.get("posture", "Normal")
    pos_color = RED if posture in ["Slouching", "Leaning"] else GREEN

    cv2.putText(frame, f"Posture: {posture}",
                (15, y), FONT, 0.6, pos_color, 2)

    y += 40

    # ================= HEAD POSE =================
    cv2.putText(frame, "Head Angles",
                (15, y), FONT, 0.65, CYAN, 2)

    y += 30

    cv2.putText(frame, f"Yaw: {results.get('yaw',0):.1f}",
                (15, y), FONT, 0.55, WHITE, 1)

    y += 25

    cv2.putText(frame, f"Pitch: {results.get('pitch',0):.1f}",
                (15, y), FONT, 0.55, WHITE, 1)

    y += 25

    cv2.putText(frame, f"Roll: {results.get('roll',0):.1f}",
                (15, y), FONT, 0.55, WHITE, 1)

    y += 30

    # ================= PERFORMANCE =================
    cv2.line(frame, (10, y), (panel_width - 10, y), (80, 80, 80), 1)
    y += 25

    cv2.putText(frame, f"FPS: {fps:.1f}",
                (15, y), FONT, 0.55, GREEN, 1)

    y += 25

    cv2.putText(frame, f"FER: {fer_lat:.0f} ms",
                (15, y), FONT, 0.55, GREEN, 1)

    y += 25

    cv2.putText(frame, f"MP: {mp_lat:.0f} ms",
                (15, y), FONT, 0.55, GREEN, 1)

    # ================= ALERT =================
    if is_alert_active:
        cx, cy = w // 2, h // 2

        cv2.rectangle(frame,
                      (cx - 300, cy - 60),
                      (cx + 300, cy + 60),
                      (0, 0, 255), -1)

        cv2.putText(frame, alert_msg,
                    (cx - 280, cy - 10),
                    FONT, 0.9, WHITE, 2)

        cv2.putText(frame, alert_reason,
                    (cx - 280, cy + 30),
                    FONT, 0.7, CYAN, 2)

# =============================================================================
# MAIN LOOP
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="DRAS — Integrated Driver Risk Assessment")
    parser.add_argument("--webcam",         type=int,   default=WEBCAM_ID)
    parser.add_argument("--fer-model",      type=str,   default=FER_MODEL_PATH)
    parser.add_argument("--classifier",     type=str,   default=CLASSIFIER_PATH)
    parser.add_argument("--no-fer",         action="store_true",
                        help="Disable FER (run without emotion module)")
    parser.add_argument("--no-classifier",  action="store_true",
                        help="Use pure geometric EAR threshold instead of trained classifier")
    args = parser.parse_args()

    # ── Camera ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.webcam)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam {args.webcam}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(f"[DRAS] Camera opened (index {args.webcam}).")

    # ── Module initialisation ─────────────────────────────────────────────────
    drowsiness_detector = DrowsinessDetector(
        mar_threshold=0.55,
        use_adaptive_threshold=True
    )
    pose_analyzer   = PoseAnalyzer()
    object_detector = ObjectDetector()

    # FER model (optional — skip with --no-fer for faster startup / testing)
    fer_analyzer  = None
    if not args.no_fer:
        try:
            fer_analyzer = FerAnalyzer(model_path=args.fer_model)
        except Exception as e:
            print(f"[WARNING] FER model not loaded: {e}")
            print("[WARNING] Running without emotion module.")

    # Drowsiness classifier (optional — skip with --no-classifier)
    drowsiness_clf = None
    if not args.no_classifier:
        try:
            drowsiness_clf = DrowsinessClassifier(args.classifier)
        except Exception as e:
            print(f"[WARNING] Drowsiness classifier not loaded: {e}")
            print("[WARNING] Falling back to geometric EAR threshold.")

    # Fusion engine
    driver_logic = DriverRiskLogic()

    # ── State ─────────────────────────────────────────────────────────────────
    alert_msg      = ""
    alert_reason   = ""
    alert_end_time = 0.0

    frame_count    = 0
    fps_acc        = 0.0
    fer_lat_acc    = 0.0
    mp_lat_acc     = 0.0
    avg_fps        = 0.0
    avg_fer_lat    = 0.0
    avg_mp_lat     = 0.0

    clf_prob       = None   # last classifier probability

    print("\n[DRAS] System ready. Press Q to quit.")

    while cap.isOpened():
        t0 = time.perf_counter()

        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # ── 1. Drowsiness + geometry ──────────────────────────────────────────
        t_mp = time.perf_counter()
        drowsiness_result = drowsiness_detector.process(frame)
        if drowsiness_result.get("calibrating", False):
            cv2.putText(frame, "CALIBRATING... LOOK FORWARD",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 2)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
            continue
        mp_lat_acc += (time.perf_counter() - t_mp) * 1000

        # ── 2. Pose landmarks ─────────────────────────────────────────────────
        pose_result   = pose_analyzer.process_frame(frame)
        pose_analyzer.draw_landmarks(frame, pose_result)
        pose_landmarks = (pose_result.pose_landmarks.landmark
                          if pose_result.pose_landmarks else None)

        # ── 3. Phone detection ────────────────────────────────────────────────
        phone_detected, phone_bboxes = object_detector.process(frame)

        # ── 4. Classify drowsiness (if classifier available) ──────────────────
        if drowsiness_result["face_detected"] and not drowsiness_result.get("calibrating", False):
            lm = drowsiness_result["landmarks"]

            # Compute raw head pose (needed as classifier features)
            raw_yaw, raw_pitch, raw_roll = compute_head_pose(lm, w, h)

            # Expose closure_len to the classifier wrapper
            # (_current_closure_len is internal; we proxy it via the drowsy flag logic)
            drowsiness_result["_closure_len_for_clf"] = drowsiness_detector._current_closure_len

            if drowsiness_clf is not None:
                is_drowsy_clf, clf_prob = drowsiness_clf.predict(
                    drowsiness_result, raw_yaw, raw_pitch, raw_roll
                )
                # Override the geometric drowsy flag with the classifier's decision.
                # The geometric flag (15 consecutive frames) stays as a hard safety net;
                # the classifier augments it with temporal pattern knowledge.
                if is_drowsy_clf and not drowsiness_result["drowsy"]:
                    drowsiness_result["drowsy"] = True   # Classifier caught it first
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
            results = driver_logic.update(
                drowsiness_result,
                pose_landmarks,
                frame.shape,
                emotion_label,
                phone_detected
            )
            # Propagate PERCLOS + microsleeps through to results dict
            # (DriverRiskLogic already reads from drowsiness_result, so these are
            #  already factored in; we pass them for the overlay only)
            results["perclos"]    = drowsiness_result.get("perclos", 0.0)
            results["microsleeps"] = drowsiness_result.get("microsleep_count", 0)

            if results["play_alert"] and alert_msg != "RESET":
                play_alert()
                alert_msg      = results["status"]
                alert_reason   = results["reason"]
                alert_end_time = time.time() + ALERT_DURATION_SEC
                is_alert_on    = True

            draw_overlay(frame, results, is_alert_on,
                         alert_msg, alert_reason,
                         avg_fps, avg_fer_lat, avg_mp_lat,
                         clf_prob=clf_prob)

            # Draw phone bounding boxes
            for (x1, y1, x2, y2) in phone_bboxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, "PHONE DETECTED",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)

        else:
            cv2.putText(frame, "No driver detected.",
                        (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # ── 7. FPS accounting ─────────────────────────────────────────────────
        frame_count += 1
        elapsed      = time.perf_counter() - t0
        fps_acc     += 1.0 / (elapsed + 1e-9)

        if frame_count % 10 == 0:
            avg_fps      = fps_acc     / 10
            avg_fer_lat  = fer_lat_acc / 10
            avg_mp_lat   = mp_lat_acc  / 10
            fps_acc = fer_lat_acc = mp_lat_acc = 0.0

        cv2.imshow("DRAS — Driver Risk Assessment System", frame)
        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key == ord("q"):
            break

        # RESET BUTTON
        if key == ord("r"):
            print("[DRAS] MANUAL RESET TRIGGERED")

            drowsiness_detector.reset()
            driver_logic.reset()

            # Clear alert state
            alert_msg = ""
            alert_reason = ""
            alert_end_time = 0.0
            clf_prob = None

    cap.release()
    cv2.destroyAllWindows()
    print("[DRAS] Stopped.")


if __name__ == "__main__":
    main()
