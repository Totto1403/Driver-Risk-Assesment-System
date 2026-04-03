"""
Driver Risk Monitor — Fixed Live Tracker
-----------------------------------------
Fixes:
  1. Closed eyes NOW trigger drowsy alert — hybrid classifier + direct EAR rule
  2. Emotions displayed live with risk feedback
  3. PERCLOS computed correctly in real-time (rolling 90-frame window ~3s at 30fps)
  4. Two-tier alert: MODERATE / CRITICAL with specific reasons shown

Usage:
    python track.py

Controls: Q=quit  R=recalibrate  S=screenshot
"""

import os, ssl
if not os.environ.get('PYTHONHTTPSVERIFY', ''):
    if getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import threading
import sys
from collections import deque
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
#  LOAD DROWSINESS CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
MODEL_PATH = "drowsy_classifier.pkl"
if not Path(MODEL_PATH).exists():
    print(f"ERROR: {MODEL_PATH} not found.")
    sys.exit(1)

bundle     = joblib.load(MODEL_PATH)
clf        = bundle["clf"]
scaler     = bundle["scaler"]
FEATURES   = bundle["features"]
MODEL_NAME = bundle["model_name"]
print(f"Drowsiness model loaded: {MODEL_NAME}")

# ══════════════════════════════════════════════════════════════════════════════
#  EMOTION — OpenCV DNN (no new packages, uses cv2 already installed)
#  Downloads two small files (~60MB total) on first run.
#  Both are standard public models — no login needed.
# ══════════════════════════════════════════════════════════════════════════════
import urllib.request, os

def _download(url, dest):
    if not os.path.exists(dest):
        print(f"Downloading {os.path.basename(dest)}...")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"  Done: {dest}")
        except Exception as e:
            print(f"  Download failed: {e}")

# Mini emotion CNN trained on FER+ — pure OpenCV DNN, no tensorflow, no protobuf
_PROTO = "emotion_deploy.prototxt"
_MODEL = "emotion_net.caffemodel"

_download(
    "https://raw.githubusercontent.com/oarriaga/face_classification/master/trained_models/fer2013_mini_XCEPTION.102-0.66.hdf5",
    "emotion_xception.hdf5"   # placeholder — we use the cv2 approach below
)

# Use the well-known EmotiW caffe model via OpenCV DNN
_PROTO_URL  = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
_LABELS     = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]

EMOTION_READY    = False
_emotion_net     = None
_WEIGHTS_PATH    = "emotion_mini.onnx"

# Download mini ONNX emotion model — plain urllib, no pip install needed
_ONNX_URL = ("https://github.com/onnx/models/raw/main/validated/"
             "vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx")

_download(_ONNX_URL, _WEIGHTS_PATH)

if os.path.exists(_WEIGHTS_PATH):
    try:
        _emotion_net  = cv2.dnn.readNetFromONNX(_WEIGHTS_PATH)
        EMOTION_READY = True
        print("Emotion model (OpenCV DNN / ONNX) ready — no pip install needed.")
    except Exception as e:
        print(f"Emotion DNN load failed: {e}")
else:
    print("[WARNING] Emotion model download failed — emotion will show Neutral.")
    print("Check internet connection and re-run track.py.")

# ══════════════════════════════════════════════════════════════════════════════
#  YOLO
# ══════════════════════════════════════════════════════════════════════════════
USE_YOLO = False
try:
    from ultralytics import YOLO
    yolo     = YOLO("yolov8n.pt")
    USE_YOLO = True
    print("YOLOv8 loaded.")
except Exception as e:
    print(f"YOLOv8 not available ({e}) — phone detection disabled.")

PHONE_CLASS = 67

# ══════════════════════════════════════════════════════════════════════════════
#  MEDIAPIPE
# ══════════════════════════════════════════════════════════════════════════════
mp_fm   = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
face_mesh = mp_fm.FaceMesh(max_num_faces=1, refine_landmarks=False,
                            min_detection_confidence=0.5, min_tracking_confidence=0.4)
pose_est  = mp_pose.Pose(model_complexity=0,
                          min_detection_confidence=0.5, min_tracking_confidence=0.4)

L_EYE  = [362, 385, 387, 263, 373, 380]
R_EYE  = [33,  160, 158, 133, 153, 144]
MOUTH  = [61, 291, 17, 0]
LM_IDX = [1, 152, 263, 33, 287, 57]
MODEL_3D = np.array([
    [0.0,    0.0,    0.0],
    [0.0,   -330.0, -65.0],
    [-225.0,  170.0,-135.0],
    [225.0,   170.0,-135.0],
    [-150.0, -150.0,-125.0],
    [150.0,  -150.0,-125.0],
], dtype=np.float64)

def _ear(lm, idx, W, H):
    p = [(lm[i].x*W, lm[i].y*H) for i in idx]
    A = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
    B = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
    C = np.linalg.norm(np.array(p[0]) - np.array(p[3]))
    return (A+B) / (2*C + 1e-7)

def _mar(lm, W, H):
    pts = [(lm[i].x*W, lm[i].y*H) for i in MOUTH]
    v = np.linalg.norm(np.array(pts[3]) - np.array(pts[2]))
    h = np.linalg.norm(np.array(pts[0]) - np.array(pts[1]))
    return v / (h + 1e-7)

def _head_pose(lm, W, H):
    img_pts = np.array([(lm[i].x*W, lm[i].y*H) for i in LM_IDX], dtype=np.float64)
    cam  = np.array([[W,0,W/2],[0,W,H/2],[0,0,1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(MODEL_3D, img_pts, cam, np.zeros((4,1)),
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return 0.0, 0.0, 0.0
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    return (np.degrees(np.arctan2(R[1,0], R[0,0])),
            np.degrees(np.arctan2(-R[2,0], sy)),
            np.degrees(np.arctan2(R[2,1], R[2,2])))

def _face_crop(lm, frame, W, H, pad=0.15):
    xs = [l.x*W for l in lm]; ys = [l.y*H for l in lm]
    x1 = max(0, int(min(xs)-pad*W)); y1 = max(0, int(min(ys)-pad*H))
    x2 = min(W, int(max(xs)+pad*W)); y2 = min(H, int(max(ys)+pad*H))
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None

# ══════════════════════════════════════════════════════════════════════════════
#  EMOTION INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
# FER+ label order for emotion-ferplus-8.onnx
_FERPLUS_LABELS = ["Neutral","Happiness","Surprise","Sadness",
                   "Anger","Disgust","Fear","Contempt"]

def infer_emotion(face_bgr):
    """Returns (label, confidence) using OpenCV DNN — zero new pip installs."""
    if face_bgr is None or face_bgr.size == 0:
        return "Neutral", 0.0

    if EMOTION_READY and _emotion_net is not None:
        try:
            # emotion-ferplus model expects: 1x1x64x64 float32 grayscale
            gray  = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            blob  = cv2.dnn.blobFromImage(gray, scalefactor=1.0,
                                          size=(64, 64), swapRB=False)
            _emotion_net.setInput(blob)
            scores = _emotion_net.forward()[0]          # shape: (8,)
            # Softmax
            e_x   = np.exp(scores - scores.max())
            probs = e_x / e_x.sum()
            idx   = int(np.argmax(probs))
            label = _FERPLUS_LABELS[idx]
            conf  = float(probs[idx])
            # Map Contempt → Disgust for EMOTION_CONFIG compatibility
            if label == "Contempt":
                label = "Disgust"
            return label, conf
        except Exception:
            pass

    return "Neutral", 0.0

# ══════════════════════════════════════════════════════════════════════════════
#  EMOTION CONFIG
# ══════════════════════════════════════════════════════════════════════════════
# Risk weight (0–100) and feedback message for each emotion
# Keys match FER+ / emotion-ferplus-8.onnx output labels
EMOTION_CONFIG = {
    "Anger":    {"risk": 90, "color": (0,0,220),    "msg": "ROAD RAGE RISK — Breathe, stay calm"},
    "Fear":     {"risk": 85, "color": (0,50,200),   "msg": "PANIC DETECTED — Pull over safely"},
    "Disgust":  {"risk": 55, "color": (30,100,200), "msg": "Negative state — Stay focused"},
    "Sadness":  {"risk": 25, "color": (200,130,0),  "msg": "Low mood — Take a break soon"},
    "Surprise": {"risk": 10, "color": (180,180,0),  "msg": "Surprise — Refocus on road"},
    "Happiness":{"risk": 0,  "color": (50,180,50),  "msg": ""},
    "Neutral":  {"risk": 0,  "color": (100,180,100),"msg": ""},
}

# ══════════════════════════════════════════════════════════════════════════════
#  STATE
# ══════════════════════════════════════════════════════════════════════════════
class S:
    # EAR / drowsiness
    ear_buf      = deque(maxlen=90)   # 90 frames ~3s rolling PERCLOS window
    perclos      = 0.0
    ear_avg      = 0.30
    ear_l = ear_r = 0.30
    mar          = 0.0
    yaw = pitch = roll = 0.0

    # Classifier output
    clf_drowsy_p = 0.0

    # Direct EAR rule (PARALLEL to classifier — catches closed eyes)
    eyes_closed_frames = 0  # consecutive frames below EAR threshold
    eyes_closed_alert  = False

    # Yawn
    yawn_frames  = 0
    yawn_active  = False
    yawn_count   = 0

    # Emotion (updated every N frames for performance)
    emotion      = "Neutral"
    emotion_conf = 0.0
    emotion_risk = 0.0   # 0–100

    # Phone
    phone        = False
    phone_conf   = 0.0
    phone_start  = None
    phone_dur    = 0.0
    phone_score  = 0.0

    # Posture
    posture_score = 0.0

    # Risk output
    visual_risk   = 0.0
    total_risk    = 0.0
    level         = "SAFE"
    reasons       = []   # list of active alert reasons

    # Calibration
    calibrating   = True
    calib_n       = 0
    CALIB         = 45
    calib_ears    = []
    ear_thr       = 0.22   # personalised per driver after calibration
    mar_thr       = 0.60   # yawn threshold
    CONSEC_CLOSED = 8      # consecutive frames below thr = microsleep (~0.3s at 30fps)

    # Misc
    last_alert    = 0.0
    fps           = 30.0
    face_visible  = False


def _compute_risk():
    S.reasons = []

    # ── Drowsiness score ─────────────────────────────────────────────────────
    # HYBRID: classifier probability OR direct EAR rule, whichever is higher
    clf_sub  = S.clf_drowsy_p * 100
    ear_sub  = min(100, S.perclos * 100 * 1.8)   # PERCLOS weighted heavily
    if S.eyes_closed_alert:
        ear_sub = max(ear_sub, 85)               # direct microsleep → high score
        S.reasons.append("EYES CLOSED")
    drowsy_sub = max(clf_sub, ear_sub)

    if drowsy_sub > 50:
        S.reasons.append(f"DROWSY ({drowsy_sub:.0f}%)")

    # ── Yawn score ───────────────────────────────────────────────────────────
    yawn_sub = min(60, S.yawn_count * 15)
    if S.yawn_active:
        yawn_sub = max(yawn_sub, 40)
        S.reasons.append("YAWNING")

    # ── Head pose score ───────────────────────────────────────────────────────
    head_sub = 0.0
    if abs(S.yaw) > 25:
        head_sub += min(50, (abs(S.yaw) - 25) * 2)
        S.reasons.append(f"HEAD TURNED ({S.yaw:.0f}°)")
    if S.pitch < -15:
        head_sub += min(60, (abs(S.pitch) - 15) * 3)
        S.reasons.append(f"HEAD NODDING ({S.pitch:.0f}°)")
    head_sub = min(70, head_sub)

    # ── Phone score ───────────────────────────────────────────────────────────
    if S.phone:
        S.reasons.append(f"PHONE ({S.phone_dur:.0f}s)")

    # ── Emotion risk ──────────────────────────────────────────────────────────
    cfg = EMOTION_CONFIG.get(S.emotion, EMOTION_CONFIG["Neutral"])
    S.emotion_risk = cfg["risk"] * S.emotion_conf
    if S.emotion_risk > 30:
        S.reasons.append(S.emotion)

    # ── Visual risk = weighted blend ─────────────────────────────────────────
    S.visual_risk = min(100,
        drowsy_sub * 0.40 +
        yawn_sub   * 0.10 +
        head_sub   * 0.15 +
        S.phone_score * 0.25 +
        S.posture_score * 0.10
    )

    # ── Total = 70% visual + 30% affective ───────────────────────────────────
    S.total_risk = min(100, S.visual_risk * 0.70 + S.emotion_risk * 0.30)

    # ── Level ─────────────────────────────────────────────────────────────────
    if S.total_risk >= 75 or S.eyes_closed_alert:
        S.level = "CRITICAL"
    elif S.total_risk >= 45:
        S.level = "MODERATE"
    else:
        S.level = "SAFE"


# ══════════════════════════════════════════════════════════════════════════════
#  AUDIO
# ══════════════════════════════════════════════════════════════════════════════
HAS_AUDIO = False
try:
    import pygame
    pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
    HAS_AUDIO = True
except Exception:
    pass

def _beep(critical=False):
    if not HAS_AUDIO: return
    def _b():
        freq = 1400 if critical else 900
        for _ in range(3 if critical else 1):
            t   = np.linspace(0, 0.15, int(22050*0.15), False)
            buf = (np.sin(2*np.pi*freq*t) * 32767).astype(np.int16)
            pygame.sndarray.make_sound(buf).play()
            time.sleep(0.22)
    threading.Thread(target=_b, daemon=True).start()

# ══════════════════════════════════════════════════════════════════════════════
#  HUD
# ══════════════════════════════════════════════════════════════════════════════
FONT  = cv2.FONT_HERSHEY_SIMPLEX
LCOL  = {"SAFE": (40,200,40), "MODERATE": (0,160,255), "CRITICAL": (0,0,220)}
PANEL_W = 280

def _draw_hud(frame):
    h, w = frame.shape[:2]
    rc   = LCOL[S.level]
    cfg  = EMOTION_CONFIG.get(S.emotion, EMOTION_CONFIG["Neutral"])

    # Semi-transparent left panel
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (PANEL_W, h), (12,12,12), -1)
    cv2.addWeighted(ov, 0.60, frame, 0.40, 0, frame)

    # ── Title ─────────────────────────────────────────────────────────────────
    cv2.putText(frame, "DRIVER RISK MONITOR", (6,22), FONT, 0.48, (170,210,255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS {S.fps:.0f}", (w-80,18), FONT, 0.40, (110,110,110), 1)

    # ── Risk bar ──────────────────────────────────────────────────────────────
    by=30; bh=28
    cv2.rectangle(frame,(6,by),(PANEL_W-8,by+bh),(50,50,50),-1)
    bw = int((PANEL_W-14)*S.total_risk/100)
    cv2.rectangle(frame,(6,by),(6+bw,by+bh),rc,-1)
    cv2.rectangle(frame,(6,by),(PANEL_W-8,by+bh),(160,160,160),1)
    cv2.putText(frame,f"RISK {S.total_risk:.0f}%  [{S.level}]",
                (9,by+19), FONT,0.47,(255,255,255),1,cv2.LINE_AA)

    # ── Calibration ───────────────────────────────────────────────────────────
    if S.calibrating:
        prog = S.calib_n/S.CALIB
        cv2.rectangle(frame,(0,by+bh+2),(int(PANEL_W*prog),by+bh+12),(0,255,190),-1)
        cv2.putText(frame,f"Calibrating... {S.calib_n}/{S.CALIB}",
                    (6,by+bh+28),FONT,0.42,(0,255,190),1)
        return

    y = 74
    def row(lbl, val, col=(190,215,255), bold=False):
        nonlocal y
        cv2.putText(frame,lbl,(8,y),FONT,0.38,(130,130,130),1,cv2.LINE_AA)
        cv2.putText(frame,val,(145,y),FONT,0.40 if not bold else 0.42,col,
                    2 if bold else 1,cv2.LINE_AA)
        y += 19

    def divider():
        nonlocal y
        y += 3
        cv2.line(frame,(6,y),(PANEL_W-8,y),(65,65,65),1)
        y += 12

    W_  = (0,80,220)  # warning colour
    G_  = (60,200,60) # good colour

    # ── Drowsiness block ──────────────────────────────────────────────────────
    ear_col = W_ if S.ear_avg < S.ear_thr else (190,215,255)
    row("EAR",       f"{S.ear_avg:.3f}  [thr:{S.ear_thr:.2f}]", ear_col)
    row("PERCLOS",   f"{S.perclos*100:.1f}%",
        W_ if S.perclos > 0.15 else (190,215,255))
    row("MAR",       f"{S.mar:.3f}",
        W_ if S.mar > S.mar_thr else (190,215,255))
    row("Yawn count",f"{S.yawn_count}",
        W_ if S.yawn_count >= 2 else (190,215,255))
    row("Head Yaw",  f"{S.yaw:.1f}°",
        W_ if abs(S.yaw) > 25 else (190,215,255))
    row("Head Pitch",f"{S.pitch:.1f}°",
        W_ if S.pitch < -15 else (190,215,255))
    row("Drowsy P",  f"{S.clf_drowsy_p*100:.0f}%",
        W_ if S.clf_drowsy_p > 0.5 else G_)

    divider()

    # ── Emotion block ──────────────────────────────────────────────────────────
    em_col  = cfg["color"]
    em_risk = S.emotion_risk
    row("EMOTION",   f"{S.emotion}",        em_col, bold=True)
    row("  conf",    f"{S.emotion_conf*100:.0f}%", em_col)
    row("  risk",    f"{em_risk:.0f}/100",
        W_ if em_risk > 40 else (190,215,255))

    # Emotion feedback message
    if cfg["msg"]:
        msg_y = y + 2
        cv2.rectangle(frame,(4,msg_y-13),(PANEL_W-4,msg_y+4),
                      tuple(int(c*0.3) for c in cfg["color"]),-1)
        cv2.putText(frame, cfg["msg"][:30], (6,msg_y),
                    FONT,0.33,cfg["color"],1,cv2.LINE_AA)
        y += 20

    divider()

    # ── Phone block ───────────────────────────────────────────────────────────
    row("Phone",     "DETECTED" if S.phone else "clear",
        W_ if S.phone else G_)
    if S.phone:
        row("  duration",f"{S.phone_dur:.1f}s", W_)

    divider()

    # ── Risk summary ──────────────────────────────────────────────────────────
    row("Visual risk",f"{S.visual_risk:.0f}/100", rc)
    row("Emot. risk", f"{S.emotion_risk:.0f}/100",em_col)
    row("TOTAL RISK", f"{S.total_risk:.0f}/100",  rc, bold=True)

    # ── Alert banners (right side) ────────────────────────────────────────────
    banners = []

    if S.eyes_closed_alert:
        banners.append(("⚠  EYES CLOSED — MICROSLEEP", (0,0,200)))
    if S.yawn_active:
        banners.append(("YAWNING DETECTED — Stay alert", (0,100,220)))
    if S.phone and S.phone_dur > 2:
        banners.append((f"PHONE {S.phone_dur:.0f}s — DANGEROUS", (0,0,180)))
    if S.emotion in ("Anger","Fear") and S.emotion_conf > 0.45:
        banners.append((f"{S.emotion.upper()} — {cfg['msg']}", cfg["color"]))
    elif S.emotion in ("Disgust","Sadness") and S.emotion_conf > 0.55:
        banners.append((cfg["msg"], cfg["color"]))
    if abs(S.yaw) > 30:
        banners.append((f"HEAD TURNED {S.yaw:.0f}° — Eyes on road", (0,140,255)))
    if S.pitch < -20:
        banners.append((f"HEAD NODDING — DROWSY", (0,60,200)))

    # Main level banner always shown
    if S.level == "CRITICAL":
        main_b = ("!! CRITICAL RISK — PULL OVER SAFELY !!", (0,0,200))
    elif S.level == "MODERATE":
        main_b = ("MODERATE RISK — Stay Alert", (0,120,255))
    else:
        main_b = None

    bx = PANEL_W + 10
    ay = h - (len(banners) + (1 if main_b else 0)) * 34 - 10

    if main_b:
        bg = frame.copy()
        cv2.rectangle(bg,(bx,ay),(w-8,ay+28),main_b[1],-1)
        cv2.addWeighted(bg,0.75,frame,0.25,0,frame)
        cv2.putText(frame,main_b[0],(bx+8,ay+19),FONT,0.50,(255,255,255),1,cv2.LINE_AA)
        ay += 34

    for txt, col in banners:
        bg = frame.copy()
        cv2.rectangle(bg,(bx,ay),(w-8,ay+28),col,-1)
        cv2.addWeighted(bg,0.68,frame,0.32,0,frame)
        cv2.putText(frame,txt[:55],(bx+8,ay+19),FONT,0.46,(255,255,255),1,cv2.LINE_AA)
        ay += 34

    # ── Active reasons strip below risk bar ───────────────────────────────────
    if S.reasons:
        rtxt = "  |  ".join(S.reasons[:4])
        cv2.putText(frame, rtxt, (PANEL_W+10, 22), FONT, 0.40, (255,220,100), 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera."); sys.exit(1)

print("\n=== Live Tracking Started ===")
print("Q=quit  R=recalibrate  S=screenshot")
print(f"EAR threshold will be personalised during 45-frame calibration.\n")

fc = 0
t0 = time.time()
EMOTION_INTERVAL = 5   # run emotion inference every N frames (performance)

while True:
    ret, frame = cap.read()
    if not ret: break
    fc += 1
    if fc % 15 == 0:
        S.fps = 15 / (time.time()-t0+1e-9)
        t0 = time.time()

    H, W = frame.shape[:2]
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ════════════════════════════════════════════════════════════════════════
    #  FACE MESH
    # ════════════════════════════════════════════════════════════════════════
    res = face_mesh.process(rgb)
    face_crop = None
    S.face_visible = bool(res.multi_face_landmarks)

    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark

        el = _ear(lm, L_EYE, W, H)
        er = _ear(lm, R_EYE, W, H)
        ea = (el + er) / 2
        S.ear_l = el; S.ear_r = er; S.ear_avg = ea
        S.mar   = _mar(lm, W, H)
        S.yaw, S.pitch, S.roll = _head_pose(lm, W, H)
        face_crop = _face_crop(lm, frame, W, H)

        # ── Calibration ──────────────────────────────────────────────────────
        if S.calibrating:
            S.calib_ears.append(ea)
            S.calib_n += 1
            if S.calib_n >= S.CALIB:
                # Personalise threshold: 80th percentile of open-eye EAR * 0.78
                S.ear_thr  = float(np.percentile(S.calib_ears, 80)) * 0.78
                S.ear_thr  = max(0.16, min(0.25, S.ear_thr))  # clamp to sane range
                S.calibrating = False
                print(f"[Calibrated] EAR threshold = {S.ear_thr:.3f}")
            _draw_hud(frame)
            cv2.imshow("Driver Risk Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        # ── Rolling EAR buffer for PERCLOS ───────────────────────────────────
        S.ear_buf.append(ea)
        if len(S.ear_buf) >= 10:
            S.perclos = sum(1 for e in S.ear_buf if e < S.ear_thr) / len(S.ear_buf)

        # ── Direct EAR rule — consecutive closed frames ───────────────────────
        if ea < S.ear_thr:
            S.eyes_closed_frames += 1
        else:
            S.eyes_closed_frames = 0
        S.eyes_closed_alert = S.eyes_closed_frames >= S.CONSEC_CLOSED

        # ── Yawn detection ────────────────────────────────────────────────────
        if S.mar > S.mar_thr:
            S.yawn_frames += 1
            S.yawn_active = True
            if S.yawn_frames == 30:   # ~1s sustained yawn
                S.yawn_count += 1
        else:
            S.yawn_frames = 0
            S.yawn_active = False

        # ── Classifier inference ──────────────────────────────────────────────
        feat   = np.array([[el, er, ea, S.mar, S.yaw, S.pitch, S.roll, S.perclos]])
        feat_s = scaler.transform(feat)
        try:
            S.clf_drowsy_p = clf.predict_proba(feat_s)[0][1]
        except Exception:
            S.clf_drowsy_p = float(clf.predict(feat_s)[0])

        # ── Emotion inference (every N frames) ───────────────────────────────
        if fc % EMOTION_INTERVAL == 0:
            S.emotion, S.emotion_conf = infer_emotion(face_crop)

        # ── Draw eye landmarks ────────────────────────────────────────────────
        eye_col = (0,60,220) if ea < S.ear_thr else (0,220,100)
        for idx in L_EYE + R_EYE:
            cx,cy = int(lm[idx].x*W), int(lm[idx].y*H)
            cv2.circle(frame,(cx,cy),2,eye_col,-1)

        # Draw EAR value near eye
        lex = int(lm[263].x*W); ley = int(lm[263].y*H)
        cv2.putText(frame,f"EAR:{ea:.2f}",(lex-60,ley-12),
                    FONT,0.38,eye_col,1,cv2.LINE_AA)

        # Draw emotion label on face
        if face_crop is not None:
            xs = [l.x*W for l in lm]; ys = [l.y*H for l in lm]
            fx,fy = int(min(xs)), max(0,int(min(ys))-22)
            em_cfg = EMOTION_CONFIG.get(S.emotion, EMOTION_CONFIG["Neutral"])
            cv2.putText(frame,
                        f"{S.emotion} {S.emotion_conf*100:.0f}%",
                        (fx,fy), FONT, 0.50, em_cfg["color"], 2, cv2.LINE_AA)

    else:
        # No face
        S.ear_buf.append(0.30)
        S.eyes_closed_frames = 0
        S.eyes_closed_alert  = False
        S.clf_drowsy_p = 0.0

    # ════════════════════════════════════════════════════════════════════════
    #  YOLO PHONE
    # ════════════════════════════════════════════════════════════════════════
    if USE_YOLO and fc % 2 == 0:
        yres  = yolo(frame, verbose=False, classes=[PHONE_CLASS], conf=0.35)
        found = False
        for r in yres:
            if r.boxes:
                for box in r.boxes:
                    if int(box.cls[0]) == PHONE_CLASS:
                        found = True; S.phone_conf = float(box.conf[0])
                        x1,y1,x2,y2 = map(int,box.xyxy[0])
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,220),2)
                        cv2.putText(frame,f"PHONE {S.phone_conf*100:.0f}%",
                                    (x1,y1-8),FONT,0.52,(0,0,220),2)
        S.phone = found
        if S.phone:
            if S.phone_start is None: S.phone_start = time.time()
            S.phone_dur   = time.time()-S.phone_start
            S.phone_score = min(100, 60+S.phone_dur*5)
        else:
            S.phone_start=None; S.phone_dur=0
            S.phone_score = max(0,S.phone_score-3)

    # ════════════════════════════════════════════════════════════════════════
    #  POSTURE (every 3 frames)
    # ════════════════════════════════════════════════════════════════════════
    if fc % 3 == 0:
        pres = pose_est.process(rgb)
        if pres.pose_landmarks:
            lp     = pres.pose_landmarks.landmark
            lean   = abs(lp[11].y - lp[12].y)
            slouch = max(0,(lp[11].y+lp[12].y)/2 - lp[0].y - 0.25)
            S.posture_score = min(100, slouch*280 + lean*170)
        else:
            S.posture_score = max(0, S.posture_score-2)

    # ════════════════════════════════════════════════════════════════════════
    #  RISK + ALERT
    # ════════════════════════════════════════════════════════════════════════
    if not S.calibrating:
        _compute_risk()

    now = time.time()
    if S.level=="CRITICAL" and now-S.last_alert>5:
        _beep(critical=True);  S.last_alert=now
    elif S.level=="MODERATE" and now-S.last_alert>15:
        _beep(critical=False); S.last_alert=now

    # ════════════════════════════════════════════════════════════════════════
    #  DRAW + SHOW
    # ════════════════════════════════════════════════════════════════════════
    _draw_hud(frame)
    cv2.imshow("Driver Risk Monitor", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('r'):
        S.calibrating=True; S.calib_n=0; S.calib_ears=[]
        S.eyes_closed_frames=0; S.yawn_count=0
        print("[Recalibrating...]")
    elif key == ord('s'):
        fn = f"screenshot_{int(time.time())}.png"
        cv2.imwrite(fn,frame); print(f"Saved: {fn}")

cap.release()
cv2.destroyAllWindows()
face_mesh.close(); pose_est.close()
print("Session ended.")