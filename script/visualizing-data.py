import os
import json
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# -------------------------------------------------
# Absolute paths (NON-NEGOTIABLE)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "Models", "trained_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "..", "Models", "labels.json")

# -------------------------------------------------
# Config
# -------------------------------------------------
MAX_SEQ_LEN = 100
CONFIDENCE_THRESHOLD = 0.5


# --- Load model ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
model = load_model(MODEL_PATH)

# --- Load labels ---
with open(LABELS_PATH, "r") as f:
    label_map = json.load(f)
labels = [label for label, _ in sorted(label_map.items(), key=lambda x: x[1])]

# --- MediaPipe ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- Normalize function ---
def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks)
    landmarks -= np.mean(landmarks, axis=0)
    landmarks /= np.max(np.linalg.norm(landmarks, axis=1)) + 1e-6
    return landmarks.flatten()

def prepare_sequence(sequence, max_len=MAX_SEQ_LEN):
    padded = sequence.copy()
    while len(padded) < max_len:
        padded.insert(0, padded[0])
    padded = np.array(padded[-max_len:])
    return padded.reshape(1, max_len, -1).astype(np.float32)

# --- Setup Matplotlib plot ---
plt.ion()
fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.bar(labels, [0]*len(labels))
ax.set_ylim(0, 1)
ax.set_ylabel("Confidence")
ax.set_title("Live Gesture Prediction")
fig.canvas.manager.set_window_title("📊 GhostPen - Prediction Visualizer")

def update_plot(prediction):
    for i, bar in enumerate(bars):
        bar.set_height(prediction[i])
    fig.canvas.draw()
    fig.canvas.flush_events()

# --- Live Prediction ---
def main():
    cap = cv2.VideoCapture(0)
    sequence = []

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            label = None
            conf = 0.0
            prediction = np.zeros(len(labels))

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                coords = normalize_landmarks(coords)
                sequence.append(coords)

                if len(sequence) > MAX_SEQ_LEN:
                    sequence.pop(0)

                if len(sequence) >= 10:
                    input_tensor = prepare_sequence(sequence)
                    prediction = model.predict(input_tensor, verbose=0)[0]
                    conf = np.max(prediction)
                    label = labels[np.argmax(prediction)]

                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            update_plot(prediction)

            # Show top label
            if conf >= CONFIDENCE_THRESHOLD:
                cv2.putText(frame, f"{label} ({conf:.2f})", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("GhostPen - Live Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.close()

if __name__ == "__main__":
    main()
