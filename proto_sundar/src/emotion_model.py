import tensorflow as tf
import numpy as np
import os

MODEL_PATH = os.path.join("models", "emotion_model_final")

# 🔥 Load SavedModel as inference layer
model = tf.keras.layers.TFSMLayer(
    MODEL_PATH,
    call_endpoint="serving_default"
)

emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

def predict_emotion(face):
    outputs = model(face)

    preds = list(outputs.values())[0].numpy()

    emotion_idx = np.argmax(preds)
    emotion = emotion_labels[emotion_idx]
    confidence = np.max(preds)

    return emotion, confidence