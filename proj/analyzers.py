import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import timm
from collections import deque, Counter
import time
from ultralytics import YOLO

# ── Landmark indices (MediaPipe 478-point Face Mesh) ─────────────────────────
LEFT_EYE    = [362, 385, 387, 263, 373, 380]
RIGHT_EYE   = [33,  160, 158, 133, 153, 144]
MOUTH_OUTER = [61, 291, 13, 14, 17, 0, 267, 37]
MOUTH_TOP    = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT   = 61
MOUTH_RIGHT  = 291

# ── Geometry helpers ─────────────────────────────────────────────────────────
def _euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def _eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    pts = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in eye_indices]
    A = _euclidean(pts[1], pts[5])
    B = _euclidean(pts[2], pts[4])
    C = _euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C + 1e-6)

def _mouth_aspect_ratio(landmarks, img_w, img_h):
    top    = (landmarks[MOUTH_TOP].x    * img_w, landmarks[MOUTH_TOP].y    * img_h)
    bottom = (landmarks[MOUTH_BOTTOM].x * img_w, landmarks[MOUTH_BOTTOM].y * img_h)
    left   = (landmarks[MOUTH_LEFT].x   * img_w, landmarks[MOUTH_LEFT].y   * img_h)
    right  = (landmarks[MOUTH_RIGHT].x  * img_w, landmarks[MOUTH_RIGHT].y  * img_h)
    return _euclidean(top, bottom) / (_euclidean(left, right) + 1e-6)

def _has_tasks_api() -> bool:
    try:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        return True
    except (ImportError, AttributeError):
        return False

def _build_face_landmarker():
    import pathlib, urllib.request
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    model_path = pathlib.Path(__file__).parent / "face_landmarker.task"
    if not model_path.exists():
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        print(f"Downloading face_landmarker.task to {model_path} ...")
        urllib.request.urlretrieve(url, model_path)
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.6,
        min_face_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)

def _build_face_mesh_legacy():
    return mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.6, min_tracking_confidence=0.6,
    )


# ── Drowsiness Module ────────────────────────────────────────────────────────
class DrowsinessDetector:
    """
    ENHANCED: Now includes:
      - Adaptive EAR threshold (Sharma et al., 2025 A-EAR concept)
      - PERCLOS calculation (% eye closure over rolling 30-sec window)
      - Microsleep detection (brief closures < 1 sec, Khalil et al., 2024)
      - Consistent default threshold (0.30) aligned with evaluate.py
    """

    # ── Calibration ──────────────────────────────────────────────────────────
    CALIBRATION_FRAMES     = 90          # 3 sec @ 30 FPS to build personal EAR baseline
    ADAPTIVE_OFFSET        = 0.08        # threshold = baseline_ear - offset
    ADAPTIVE_MIN_THRESHOLD = 0.18        # never go below this (safety floor)
    ADAPTIVE_MAX_THRESHOLD = 0.35        # never go above this

    # ── PERCLOS window ───────────────────────────────────────────────────────
    PERCLOS_WINDOW_SEC     = 30.0        # standard 30-second PERCLOS window
    PERCLOS_ALERT_THRESH   = 0.15        # >15 % closed = drowsy (standard)

    # ── Microsleep detection ─────────────────────────────────────────────────
    MICROSLEEP_MIN_FRAMES  = 5           # closed for ≥ 3 frames = microsleep candidate
    MICROSLEEP_MAX_FRAMES  = 25          # but < 25 frames (< ~0.8 s) = microsleep
                                          # ≥ 25 frames = full drowsy event

    def __init__(self,
                 ear_threshold=0.30,      # FIXED: aligned with evaluate.py
                 ear_consec_frames=15,
                 mar_threshold=0.55,
                 mar_consec_frames=20,
                 use_adaptive_threshold=True):

        self.initial_ear_threshold  = ear_threshold
        self.ear_threshold          = ear_threshold     # may be overwritten by adaptive logic
        self.ear_consec_frames      = ear_consec_frames
        self.mar_threshold          = mar_threshold
        self.mar_consec_frames      = mar_consec_frames
        self.use_adaptive_threshold = use_adaptive_threshold

        # ── Calibration state ────────────────────────────────────────────────
        self._calib_ear_samples: list[float] = []
        self._calibrated        = False
        self.calibrating        = False          # flag so logic.py can show "CALIBRATING"

        # ── Temporal buffers ─────────────────────────────────────────────────
        self._eye_counter   = 0
        self._yawn_counter  = 0
        self._ear_history   = deque(maxlen=5)    # 5-frame moving avg (noise suppression)
        self._yawn_times: list[float] = []
        self._yawn_window_sec = 60.0

        # ── PERCLOS state ────────────────────────────────────────────────────
        self._perclos_log: list[tuple] = []      # list of (timestamp, is_closed)

        # ── Microsleep state ─────────────────────────────────────────────────
        self._current_closure_len = 0            # frames in the current closure streak
        self._microsleep_count    = 0            # microsleeps in last 60 s
        self._microsleep_times: list[float] = []

        # ── Status flags ─────────────────────────────────────────────────────
        self.eyes_closed = False
        self.yawning     = False
        self.drowsy      = False

        # ── MediaPipe backend ─────────────────────────────────────────────────
        if _has_tasks_api():
            self._face_mesh = _build_face_landmarker()
            self._use_new   = True
        else:
            self._face_mesh = _build_face_mesh_legacy()
            self._use_new   = False

    # ── Public API ───────────────────────────────────────────────────────────
    def process(self, frame: np.ndarray) -> dict:
        result = dict(
            face_detected=False, ear=0.0, mar=0.0,
            eyes_closed=False, yawning=False, drowsy=False,
            yawn_count=0, fatigue_score=0, landmarks=None,
            perclos=0.0, microsleep=False, microsleep_count=0,
            adaptive_threshold=self.ear_threshold,
            calibrating=False
        )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        lm = self._get_landmarks(rgb)

        if lm is None:
            self._eye_counter = self._yawn_counter = 0
            self._current_closure_len = 0
            return result

        result["face_detected"] = True
        result["landmarks"]     = lm

        # ── EAR ──────────────────────────────────────────────────────────────
        ear_l = _eye_aspect_ratio(lm, LEFT_EYE,  w, h)
        ear_r = _eye_aspect_ratio(lm, RIGHT_EYE, w, h)
        ear   = (ear_l + ear_r) / 2.0
        self._ear_history.append(ear)
        ear_smooth = float(np.mean(self._ear_history))

        # ── Adaptive threshold calibration ───────────────────────────────
        if self.use_adaptive_threshold and not self._calibrated:
            self._calib_ear_samples.append(ear_smooth)
            self.calibrating = True
            result["calibrating"] = True

            # HARD RESET during calibration
            self._eye_counter = 0
            self._current_closure_len = 0
            self._microsleep_times.clear()
            self._perclos_log.clear()

            if len(self._calib_ear_samples) >= self.CALIBRATION_FRAMES:
                baseline = float(np.percentile(self._calib_ear_samples, 75))
                adaptive = baseline - self.ADAPTIVE_OFFSET

                self.ear_threshold = float(
                    np.clip(adaptive, self.ADAPTIVE_MIN_THRESHOLD, self.ADAPTIVE_MAX_THRESHOLD)
                )

                self._calibrated = True
                self.calibrating = False

                print(f"[DRAS] Adaptive EAR threshold set: {self.ear_threshold:.3f}")

            result["ear"] = round(ear_smooth, 3)
            result["adaptive_threshold"] = self.ear_threshold

            self._draw(frame, lm, w, h, result)
            return result

        # ── Proceed with normal detection ONLY after calibration is complete ──
        result["adaptive_threshold"] = self.ear_threshold
        result["ear"] = round(ear_smooth, 3)

        eyes_closed_now = ear_smooth < self.ear_threshold
        result["eyes_closed"] = eyes_closed_now
        self.eyes_closed = eyes_closed_now

        # ── Consecutive frame counter + microsleep detection ─────────────────
        if eyes_closed_now:
            self._eye_counter += 1
            self._current_closure_len += 1
        else:
            if self._current_closure_len >= self.MICROSLEEP_MIN_FRAMES:
                if self._current_closure_len < self.MICROSLEEP_MAX_FRAMES:
                    self._microsleep_times.append(time.time())
            else:
                # RESET false microsleep buildup
                self._microsleep_times.clear()

            self._eye_counter = 0
            self._current_closure_len = 0       

        # Prune old microsleep events
        now = time.time()
        self._microsleep_times = [t for t in self._microsleep_times if now - t < 10.0]
        result["microsleep_count"] = len(self._microsleep_times)

        # Full drowsy event = sustained closure ≥ ear_consec_frames
        result["drowsy"] = self._eye_counter >= self.ear_consec_frames
        result["microsleep"] = (
            self._current_closure_len >= self.MICROSLEEP_MIN_FRAMES and
            self._current_closure_len < self.MICROSLEEP_MAX_FRAMES
        )
        self.drowsy = result["drowsy"]

        # ── PERCLOS ───────────────────────────────────────────────────────────
        self._perclos_log.append((now, eyes_closed_now))
        cutoff = now - self.PERCLOS_WINDOW_SEC
        self._perclos_log = [(t, c) for t, c in self._perclos_log if t >= cutoff]
        if self._perclos_log:
            perclos = sum(1 for _, c in self._perclos_log if c) / len(self._perclos_log)
        else:
            perclos = 0.0
        result["perclos"] = round(perclos, 3)

        # ── MAR / Yawn ────────────────────────────────────────────────────────
        mar = _mouth_aspect_ratio(lm, w, h)
        result["mar"] = round(mar, 3)
        if mar > self.mar_threshold:
            self._yawn_counter += 1
        else:
            if self._yawn_counter >= self.mar_consec_frames:
                self._yawn_times.append(time.time())
            self._yawn_counter = 0

        result["yawning"] = self._yawn_counter >= self.mar_consec_frames
        self.yawning = result["yawning"]

        self._yawn_times = [t for t in self._yawn_times if now - t < self._yawn_window_sec]
        result["yawn_count"] = len(self._yawn_times)

        # ── Fatigue score ─────────────────────────────────────────────────────
        # Base: EAR proximity to threshold (0-40)
        ear_score  = max(0, min(40, int((1 - ear_smooth / self.ear_threshold) * 40)))
        # PERCLOS contribution (0-30)
        perclos_score = min(30, int(perclos * 200))
        # Microsleep contribution (0-20)
        ms_score   = min(20, result["microsleep_count"] * 7)
        # Yawn contribution (0-10)
        yawn_score = min(10, result["yawn_count"] * 3)
        result["fatigue_score"] = min(100, ear_score + perclos_score + ms_score + yawn_score)

        self._draw(frame, lm, w, h, result)
        return result

    def reset(self):
        # Reset fatigue state
        self._eye_counter = 0
        self._current_closure_len = 0
        self._microsleep_times.clear()
        self._perclos_log.clear()
        self._yawn_times.clear()
        self._yawn_counter = 0

        # Reset flags
        self.eyes_closed = False
        self.yawning = False
        self.drowsy = False

        print("[DRAS] DrowsinessDetector reset")        

    # ── Private helpers ───────────────────────────────────────────────────────
    def _get_landmarks(self, rgb: np.ndarray):
        if self._use_new:
            mp_img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            detection = self._face_mesh.detect(mp_img)
            return detection.face_landmarks[0] if detection.face_landmarks else None
        else:
            out = self._face_mesh.process(rgb)
            return out.multi_face_landmarks[0].landmark if out.multi_face_landmarks else None

    def _draw(self, frame, lm, w, h, result):
        eye_col  = (0, 0, 255) if result["drowsy"]   else \
                   (0, 165, 255) if result["eyes_closed"] else (0, 255, 0)
        yawn_col = (0, 0, 255) if result["yawning"]  else (0, 255, 0)

        for eye_idx in (LEFT_EYE, RIGHT_EYE):
            pts = np.array(
                [(int(lm[i].x * w), int(lm[i].y * h)) for i in eye_idx], dtype=np.int32
            )
            cv2.polylines(frame, [pts], True, eye_col, 1)

        mouth_pts = np.array(
            [(int(lm[i].x * w), int(lm[i].y * h)) for i in MOUTH_OUTER], dtype=np.int32
        )
        cv2.polylines(frame, [mouth_pts], True, yawn_col, 1)

        metrics = [
            (f"EAR: {result['ear']:.3f} (thr:{result['adaptive_threshold']:.3f})", eye_col),
            (f"MAR: {result['mar']:.2f}",                                          yawn_col),
            (f"PERCLOS: {result['perclos']*100:.1f}%",
             (0, 0, 255) if result["perclos"] > self.PERCLOS_ALERT_THRESH else (200, 200, 200)),
            (f"Microsleeps(60s): {result['microsleep_count']}",
             (0, 0, 255) if result["microsleep_count"] >= 5 else (200, 200, 200)),
            (f"Yawns(60s): {result['yawn_count']}",                               (200, 200, 200)),
            (f"Fatigue: {result['fatigue_score']}%",
             (0, 0, 255) if result["fatigue_score"] > 60 else (0, 200, 255)),
        ]
        for i, (text, colour) in enumerate(metrics):
            cv2.putText(frame, text, (10, 20 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, colour, 1, cv2.LINE_AA)


# ── Pose / Posture Module ──────────────────────────────────────────────────
class PoseAnalyzer:
    def __init__(self):
        self.mp_pose    = mp.solutions.pose
        self.pose       = self.mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def process_frame(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        return self.pose.process(image_rgb)

    def draw_landmarks(self, frame, pose_results):
        if pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245, 66, 230),  thickness=2, circle_radius=2),
            )


# ── FER Module ─────────────────────────────────────────────────────────────
class FERModel(nn.Module):
    def __init__(self, model_arch: str, pretrained: bool, num_classes: int = 7):
        super().__init__()
        self.model = timm.create_model(
            model_arch, pretrained=pretrained, num_classes=num_classes, in_chans=3
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class FerAnalyzer:
    """
    ENHANCED:
      - Temporal majority-vote buffer (last N predictions) eliminates single-frame
        emotion jitter that caused false HIGH-RISK alerts.
      - Confidence gating: if the model is unsure (top prob < MIN_CONFIDENCE),
        the result is treated as 'Neutral' to avoid noisy risk amplification.
    """
    FER_CATEGORIES = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

    MIN_CONFIDENCE    = 0.40   # below this probability → report "Neutral"
    VOTE_BUFFER_LEN   = 10     # majority vote over last 10 frames (~0.33 s @ 30 FPS)

    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading FER model on device: {self.device}")
        self.model = self._load_model(model_path)
        if not self.model:
            raise RuntimeError("Failed to load FER model.")
        print("FER model (ResNet-34) loaded successfully.")

        # Temporal vote buffer
        self._vote_buffer: deque[str] = deque(maxlen=self.VOTE_BUFFER_LEN)

    # ── Model loading ─────────────────────────────────────────────────────────
    def _load_model(self, model_path: str):
        try:
            model      = FERModel(model_arch='resnet34', pretrained=False, num_classes=7)
            cp         = torch.load(model_path, map_location=self.device)
            state_dict = cp["state_dict"]
            state_dict = {k.replace("model.model.", ""): v for k, v in state_dict.items()}
            model.model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading FER model from {model_path}: {e}")
            return None

    # ── Prediction ────────────────────────────────────────────────────────────
    def predict(self, preprocessed_face_tensor) -> str:
        """Returns majority-voted, confidence-gated emotion label (Capitalized)."""
        if self.model is None:
            return "Neutral"

        tensor = preprocessed_face_tensor.to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)

        probs      = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        top_idx    = int(np.argmax(probs))
        top_prob   = float(probs[top_idx])
        top_label  = self.FER_CATEGORIES[top_idx]

        # Confidence gate — uncertain predictions become Neutral
        if top_prob < self.MIN_CONFIDENCE:
            top_label = 'neutral'

        self._vote_buffer.append(top_label)

        # Majority vote across the buffer
        voted = Counter(self._vote_buffer).most_common(1)[0][0]
        return voted.capitalize()

    # ── Utility ───────────────────────────────────────────────────────────────
    def get_emotion_probabilities(self, preprocessed_face_tensor) -> dict:
        """Returns the full probability dict (useful for UI visualisation)."""
        if self.model is None:
            return {c: 0.0 for c in self.FER_CATEGORIES}
        tensor = preprocessed_face_tensor.to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        return dict(zip(self.FER_CATEGORIES, [float(p) for p in probs]))


# ── Object Detection Module (YOLOv8) ──────────────────────────────────────
class ObjectDetector:
    """
    ENHANCED: Confidence threshold raised to 0.6 (Smith & Jones, 2024 quantized
    YOLOv8 recommendation) to reduce false positives from cups / wallets.
    """
    PHONE_CLASS_ID = 67   # COCO class ID for 'cell phone'

    def __init__(self, model_path: str = 'yolov8n.pt', conf_threshold: float = 0.60):
        print("Loading YOLOv8 model for phone detection...")
        self.model          = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def process(self, frame: np.ndarray):
        """Returns (phone_detected: bool, bboxes: list of (x1,y1,x2,y2))."""
        results = self.model.predict(
            frame, classes=[self.PHONE_CLASS_ID],
            conf=self.conf_threshold, verbose=False
        )
        phone_detected = False
        bboxes         = []
        for r in results:
            for box in r.boxes:
                phone_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bboxes.append((x1, y1, x2, y2))
        return phone_detected, bboxes
