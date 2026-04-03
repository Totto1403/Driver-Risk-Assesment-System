import winsound
import cv2
import torch
import numpy as np
from torchvision import transforms
import threading

def _play_beep():
    """Internal function to handle the actual beep generation."""
    winsound.Beep(1000, 500) # 1000Hz frequency, 500ms duration

def play_alert():
    """
    Plays an alert sound in a separate background thread.
    This prevents the main video feed from freezing while the alarm sounds.
    """
    threading.Thread(target=_play_beep, daemon=True).start()

def get_face_crop(frame, landmarks, margin=0.3):
    """Crops the face from the frame using MediaPipe landmarks for the Emotion Model."""
    h, w, _ = frame.shape
    x_min, y_min = w, h
    x_max, y_max = 0, 0

    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        if x < x_min: x_min = x
        if x > x_max: x_max = x
        if y < y_min: y_min = y
        if y > y_max: y_max = y

    x_margin = int((x_max - x_min) * margin)
    y_margin = int((y_max - y_min) * margin)
    
    x_min = max(0, x_min - x_margin)
    x_max = min(w, x_max + x_margin)
    y_min = max(0, y_min - y_margin)
    y_max = min(h, y_max + y_margin)

    if x_max > x_min and y_max > y_min:
        return frame[y_min:y_max, x_min:x_max]
    return None

FER_IMG_SIZE = (224, 224) 

def preprocess_fer(face_crop_bgr):  
    """Preprocesses the BGR face crop specifically for the custom ResNet34 model."""
    if face_crop_bgr is None:
        return None

    # Convert to Grayscale as expected by the trained model
    face_crop_gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(FER_IMG_SIZE),
        transforms.ToTensor(),
        # Repeat the grayscale channel 3 times to satisfy ResNet34's 3-channel requirement
        transforms.Lambda(lambda t: t.repeat(3, 1, 1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return preprocess(face_crop_gray).unsqueeze(0)