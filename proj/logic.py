import numpy as np
import cv2

# The order: Nose, Chin, Left Eye, Right Eye, Left Mouth, Right Mouth
HPE_IDXS = [1, 152, 33, 263, 61, 291]

MODEL_POINTS_3D = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 330.0, -65.0],
    [-225.0, -170.0, -135.0],
    [225.0, -170.0, -135.0],
    [-150.0, 150.0, -125.0],
    [150.0, 150.0, -125.0]
], dtype=np.float64)

NOSE           = 0
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12

# ── Affective risk mapping ─────────────────────────────────────────────────────
# Maps each FER category to an impairment score 0.0–1.0.
# Based on empirical driving performance literature:
#   Angry/Fear -> highest impairment; Neutral/Happy -> none.
EMOTION_RISK_MAP = {
    "Angry":    0.90,
    "Fear":     0.80,
    "Disgust":  0.65,
    "Surprise": 0.55,
    "Sad":      0.45,
    "Happiness":0.00,
    "Neutral":  0.00,
    # capitalised variants produced by FerAnalyzer
    "Anger":    0.90,
    "Happy":    0.00,
}


class DriverRiskLogic:
    """
    ENHANCED:
      - New MICROSLEEP alert tier (between Moderate and High)
      - PERCLOS threshold integration — high PERCLOS alone triggers Moderate risk
      - Emotion risk now uses the fine-grained EMOTION_RISK_MAP instead of
        a binary Angry/Fear check, enabling smoother affective contribution
      - Consistent with the updated DrowsinessDetector (adaptive threshold,
        microsleep_count, perclos in the result dict)
    """

    # ── Head Pose Thresholds ──────────────────────────────────────────────────
    HPE_YAW_THRESHOLD   = 25.0
    HPE_PITCH_THRESHOLD = 20.0
    HPE_ROLL_THRESHOLD  = 20.0
    HPE_SMOOTHING       = 0.6

    # ── Posture Thresholds ────────────────────────────────────────────────────
    SLOUCH_THRESHOLD = 0.025
    LEAN_THRESHOLD   = 0.10

    # ── Frame duration limits ─────────────────────────────────────────────────
    # (@ ~30 FPS assumption; values in frames)
    POSTURE_MOD_TIME       = 90    # 3 s
    POSTURE_HIGH_TIME      = 300   # 10 s
    DISTRACTION_MOD_TIME   = 30    # 1 s
    DISTRACTION_HIGH_TIME  = 120   # 4 s
    YAWN_MOD_COUNT         = 1
    YAWN_HIGH_COUNT        = 3

    # ── PERCLOS threshold ─────────────────────────────────────────────────────
    PERCLOS_MODERATE = 0.15        # 15 % → Moderate risk
    PERCLOS_HIGH     = 0.30        # 30 % → High risk

    # ── Microsleep threshold ──────────────────────────────────────────────────
    MICROSLEEP_MOD_COUNT   = 3     # 2 microsleeps in 60 s → Moderate
    MICROSLEEP_HIGH_COUNT  = 6     # 4 microsleeps in 60 s → High

    # ── Late-fusion weights ───────────────────────────────────────────────────
    W_VISUAL    = 0.70
    W_AFFECTIVE = 0.30

    # ── Within visual sub-score weights ──────────────────────────────────────
    W_FATIGUE     = 0.50
    W_POSTURE     = 0.25
    W_DISTRACTION = 0.25

    def __init__(self):
        # Calibration state
        self.calibration_frames          = 0
        self.REQUIRED_CALIBRATION_FRAMES = 30
        self.baseline_yaw   = 0.0
        self.baseline_pitch = 0.0
        self.baseline_roll  = 0.0

        # Smoothed head pose
        self.smooth_yaw   = 0.0
        self.smooth_pitch = 0.0
        self.smooth_roll  = 0.0

        # Frame counters
        self.distracted_frames   = 0
        self.bad_posture_frames  = 0
        self.head_tilt_frames    = 0

    def reset(self):
        self.calibration_frames = 0
        self.baseline_yaw = 0.0
        self.baseline_pitch = 0.0
        self.baseline_roll = 0.0

        self.smooth_yaw = 0.0
        self.smooth_pitch = 0.0
        self.smooth_roll = 0.0

        self.distracted_frames = 0
        self.bad_posture_frames = 0
        self.head_tilt_frames = 0

        print("[DRAS] DriverRiskLogic reset")
    # ── Head Pose Estimation ──────────────────────────────────────────────────
    def _get_head_pose(self, face_landmarks, frame_shape):
        h, w = frame_shape[:2]
        image_points_2d = np.array([
            (int(face_landmarks[idx].x * w), int(face_landmarks[idx].y * h))
            for idx in HPE_IDXS
        ], dtype=np.float64)

        focal_length  = w
        center        = (w / 2, h / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype=np.float64
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vector, _ = cv2.solvePnP(
            MODEL_POINTS_3D, image_points_2d,
            camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return 0.0, 0.0, 0.0

        R, _ = cv2.Rodrigues(rotation_vector)
        sy   = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        if sy >= 1e-6:
            pitch = np.arctan2(R[2, 1],  R[2, 2])
            yaw   = np.arctan2(-R[2, 0], sy)
            roll  = np.arctan2(R[1, 0],  R[0, 0])
        else:
            pitch = np.arctan2(-R[1, 2], R[1, 1])
            yaw   = np.arctan2(-R[2, 0], sy)
            roll  = 0.0

        return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)

    # ── Posture estimation ────────────────────────────────────────────────────
    def _calculate_posture(self, pose_landmarks) -> str:
        if not pose_landmarks:
            return "No Pose"
        try:
            ls = pose_landmarks[LEFT_SHOULDER]
            rs = pose_landmarks[RIGHT_SHOULDER]
            if ls.visibility < 0.5 or rs.visibility < 0.5:
                return "Partial Pose"

            nose_y        = pose_landmarks[NOSE].y
            shoulder_avg_y = (ls.y + rs.y) / 2
            slouch_diff   = nose_y - shoulder_avg_y

            nose_x        = pose_landmarks[NOSE].x
            shoulder_avg_x = (ls.x + rs.x) / 2
            lean_diff     = abs(nose_x - shoulder_avg_x)

            if slouch_diff > self.SLOUCH_THRESHOLD:
                return "Slouching"
            elif lean_diff > self.LEAN_THRESHOLD:
                return "Leaning"
            return "Good"
        except Exception:
            return "Error"

    # ── Main update ───────────────────────────────────────────────────────────
    def update(self, drowsiness_result: dict, pose_landmarks,
               frame_shape, emotion_label: str, phone_detected: bool = False) -> dict:

        # ── No face detected ──────────────────────────────────────────────────
        if not drowsiness_result["face_detected"]:
            self.calibration_frames = 0
            self.baseline_yaw = self.baseline_pitch = self.baseline_roll = 0.0
            return {
                "status": "No Driver Detected!", "play_alert": False,
                "risk_score": 0, "yaw": 0, "pitch": 0, "roll": 0,
                "emotion": "None", "posture": "N/A",
                "yawns": 0, "reason": "",
                "perclos": 0.0, "microsleeps": 0,
            }

        # ── DrowsinessDetector is still calibrating its adaptive EAR ─────────
        if drowsiness_result.get("calibrating", False):
            return {
                "status": "CALIBRATING CAMERA...",
                "play_alert": False, "risk_score": 0,
                "yaw": 0, "pitch": 0, "roll": 0,
                "emotion": "None", "posture": "Calibrating",
                "yawns": 0, "reason": "Hold still and look at the road",
                "perclos": 0.0, "microsleeps": 0,
            }

        landmarks      = drowsiness_result["landmarks"]
        yawn_count     = drowsiness_result.get("yawn_count", 0)
        perclos        = drowsiness_result.get("perclos", 0.0)
        microsleep_cnt = drowsiness_result.get("microsleep_count", 0)

        # ── Head pose ─────────────────────────────────────────────────────────
        raw_yaw, raw_pitch, raw_roll = self._get_head_pose(landmarks, frame_shape)

        # Camera baseline calibration (logic-level, for head-pose zero-ing)
        if self.calibration_frames < self.REQUIRED_CALIBRATION_FRAMES:
            self.baseline_yaw   += raw_yaw
            self.baseline_pitch += raw_pitch
            self.baseline_roll  += raw_roll
            self.calibration_frames += 1
            if self.calibration_frames == self.REQUIRED_CALIBRATION_FRAMES:
                self.baseline_yaw   /= self.REQUIRED_CALIBRATION_FRAMES
                self.baseline_pitch /= self.REQUIRED_CALIBRATION_FRAMES
                self.baseline_roll  /= self.REQUIRED_CALIBRATION_FRAMES
            return {
                "status": "CALIBRATING CAMERA...",
                "play_alert": False, "risk_score": 0,
                "yaw": raw_yaw, "pitch": raw_pitch, "roll": raw_roll,
                "emotion": "None", "posture": "Calibrating",
                "yawns": 0, "reason": "Hold still and look at the road",
                "perclos": 0.0, "microsleeps": 0,
            }

        cal_yaw   = raw_yaw   - self.baseline_yaw
        cal_pitch = raw_pitch - self.baseline_pitch
        cal_roll  = raw_roll  - self.baseline_roll

        self.smooth_yaw   = (cal_yaw   * (1 - self.HPE_SMOOTHING)) + (self.smooth_yaw   * self.HPE_SMOOTHING)
        self.smooth_pitch = (cal_pitch * (1 - self.HPE_SMOOTHING)) + (self.smooth_pitch * self.HPE_SMOOTHING)
        self.smooth_roll  = (cal_roll  * (1 - self.HPE_SMOOTHING)) + (self.smooth_roll  * self.HPE_SMOOTHING)

        # ── Frame counters ────────────────────────────────────────────────────
        is_distracted = (abs(self.smooth_yaw)   > self.HPE_YAW_THRESHOLD or
                         abs(self.smooth_pitch) > self.HPE_PITCH_THRESHOLD)
        self.distracted_frames = self.distracted_frames + 1 if is_distracted \
                                 else max(0, self.distracted_frames - 2)

        is_head_tilted = abs(self.smooth_roll) > self.HPE_ROLL_THRESHOLD
        self.head_tilt_frames = self.head_tilt_frames + 1 if is_head_tilted \
                                else max(0, self.head_tilt_frames - 2)

        posture_status = self._calculate_posture(pose_landmarks)
        bad_posture    = posture_status in ("Slouching", "Leaning")
        self.bad_posture_frames = self.bad_posture_frames + 1 if bad_posture \
                                  else max(0, self.bad_posture_frames - 2)

        # ── Visual sub-score (0–100) ──────────────────────────────────────────
        fatigue_raw = drowsiness_result["fatigue_score"]   # 0–100 from DrowsinessDetector

        distraction_score = 0
        if self.distracted_frames > self.DISTRACTION_HIGH_TIME:
            distraction_score = 100
        elif self.distracted_frames > self.DISTRACTION_MOD_TIME:
            distraction_score = 50

        posture_score = 0
        if self.bad_posture_frames > self.POSTURE_HIGH_TIME or \
           self.head_tilt_frames   > self.POSTURE_HIGH_TIME:
            posture_score = 100
        elif self.bad_posture_frames > self.POSTURE_MOD_TIME or \
             self.head_tilt_frames   > self.POSTURE_MOD_TIME:
            posture_score = 50

        phone_score = 100 if phone_detected else 0

        visual_risk = int(
            fatigue_raw     * self.W_FATIGUE     +
            posture_score   * self.W_POSTURE     +
            distraction_score * self.W_DISTRACTION
        )
        # Phone is a hard override on visual risk
        if phone_detected:
            visual_risk = max(visual_risk, 85)

        visual_risk = min(100, visual_risk)

        # ── Affective sub-score (0–100) ───────────────────────────────────────
        affective_risk = int(
            EMOTION_RISK_MAP.get(emotion_label, 0.0) * 100
        )

        # ── Total risk ────────────────────────────────────────────────────────
        total_risk = (visual_risk * self.W_VISUAL) + (affective_risk * self.W_AFFECTIVE)
        total_risk = min(100.0, total_risk)

        # ── Priority-tiered alert assignment ─────────────────────────────────
        play_alert = False
        status     = "Safe / Attentive"
        reason     = ""

        # ── TIER 1 — CRITICAL ─────────────────────────────────────────────────
        if drowsiness_result["drowsy"]:
            status     = "CRITICAL: WAKE UP!"
            reason     = "Sustained Eye Closure (Microsleep / Sleeping)"
            total_risk = max(total_risk, 95)
            play_alert = True

        elif phone_detected:
            status     = "CRITICAL: DROP PHONE!"
            reason     = "Cell Phone Usage Detected"
            total_risk = max(total_risk, 85)
            play_alert = True

        # ── TIER 2 — HIGH RISK ────────────────────────────────────────────────
        elif perclos >= self.PERCLOS_HIGH:
            status     = "WARNING: HIGH RISK"
            reason     = f"High PERCLOS: {perclos*100:.0f}% eyes closed (30s)"
            total_risk = max(total_risk, 70)
            play_alert = True

        elif microsleep_cnt >= self.MICROSLEEP_HIGH_COUNT:
            status     = "WARNING: HIGH RISK"
            reason     = f"Frequent Microsleeps ({microsleep_cnt} in 60s)"
            total_risk = max(total_risk, 65)
            play_alert = True

        elif self.bad_posture_frames > self.POSTURE_HIGH_TIME:
            status     = "WARNING: HIGH RISK"
            reason     = f"Prolonged {posture_status} (>10 s)"
            total_risk = max(total_risk, 60)
            play_alert = True

        elif self.head_tilt_frames > self.POSTURE_HIGH_TIME:
            status     = "WARNING: HIGH RISK"
            reason     = "Prolonged Head Tilt (>10 s)"
            total_risk = max(total_risk, 60)
            play_alert = True

        elif yawn_count >= self.YAWN_HIGH_COUNT:
            status     = "WARNING: HIGH RISK"
            reason     = f"Excessive Yawning ({yawn_count}× in 60 s)"
            total_risk = max(total_risk, 60)
            play_alert = True

        elif self.distracted_frames > self.DISTRACTION_HIGH_TIME:
            status     = "WARNING: HIGH RISK"
            reason     = "Prolonged Head Turn (>4 s)"
            total_risk = max(total_risk, 60)
            play_alert = True

        elif total_risk > 50:
            status     = "WARNING: HIGH RISK"
            reason     = f"High Affective Impairment ({emotion_label})"
            play_alert = True

        # ── TIER 3 — MICROSLEEP WARNING (new) ────────────────────────────────
        elif microsleep_cnt >= self.MICROSLEEP_MOD_COUNT:
            status     = "Caution: Microsleeps Detected"
            reason     = f"{microsleep_cnt} Involuntary Eye Closures in 60 s"
            total_risk = max(total_risk, 40)
            play_alert = True

        elif perclos >= self.PERCLOS_MODERATE:
            status     = "Caution: Drowsy Pattern"
            reason     = f"PERCLOS {perclos*100:.0f}% — Eyes Closing Frequently"
            total_risk = max(total_risk, 35)
            play_alert = True

        # ── TIER 4 — MODERATE ─────────────────────────────────────────────────
        elif self.bad_posture_frames > self.POSTURE_MOD_TIME:
            status = "Moderate Risk"
            reason = f"Check Posture ({posture_status})"
            total_risk = max(total_risk, 30)

        elif self.head_tilt_frames > self.POSTURE_MOD_TIME:
            status = "Moderate Risk"
            reason = "Head Tilted (Possible Fatigue)"
            total_risk = max(total_risk, 30)

        elif yawn_count >= self.YAWN_MOD_COUNT:
            status = "Moderate Risk"
            reason = "Early Fatigue Signs (Yawning)"
            total_risk = max(total_risk, 30)

        elif self.distracted_frames > self.DISTRACTION_MOD_TIME:
            status = "Moderate Risk"
            reason = "Looking Away from Road"
            total_risk = max(total_risk, 30)

        return {
            "status":     status,
            "play_alert": play_alert,
            "risk_score": int(total_risk),
            "yaw":        self.smooth_yaw,
            "pitch":      self.smooth_pitch,
            "roll":       self.smooth_roll,
            "emotion":    emotion_label,
            "posture":    posture_status,
            "yawns":      yawn_count,
            "reason":     reason,
            "perclos":    perclos,
            "microsleeps": microsleep_cnt,
        }
