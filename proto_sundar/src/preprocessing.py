import cv2
import numpy as np

def preprocess_face(frame, face_coords):
    x, y, w, h = face_coords

    # 🔹 Add padding (helps capture full expression)
    padding = 20
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = w + 2 * padding
    h = h + 2 * padding

    # 🔹 Crop face
    face = frame[y:y+h, x:x+w]

    # ⚠️ Safety check
    if face.size == 0:
        return None

    # 🔥 Convert BGR → RGB (IMPORTANT for MobileNet)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # 🔹 Resize to match training input
    face = cv2.resize(face, (224, 224))

    # 🔹 Normalize
    face = face / 255.0

    # 🔹 Reshape for model
    face = np.reshape(face, (1, 224, 224, 3))

    return face