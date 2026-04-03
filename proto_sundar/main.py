import cv2
from collections import deque

from src.face_detection import detect_faces
from src.preprocessing import preprocess_face
from src.emotion_model import predict_emotion

cap = cv2.VideoCapture(0)

# 🔥 Smoothing window
emotion_window = deque(maxlen=15)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detect_faces(frame)

    for (x, y, w, h) in faces:
        processed_face = preprocess_face(frame, (x, y, w, h))

        if processed_face is None:
            continue

        emotion, confidence = predict_emotion(processed_face)

        # 🔥 Confidence filtering (keep but don't merge classes)
        if confidence < 0.5:
            emotion = "Neutral"

        # 🔥 Smoothing
        emotion_window.append(emotion)
        final_emotion = max(set(emotion_window), key=emotion_window.count)

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display emotion + confidence
        text = f"{final_emotion} ({confidence:.2f})"
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()