import os
import json
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque, Counter

# =====================================================
# PATH SETUP (ROBUST, SCRIPT-RELATIVE)
# =====================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "Models", "trained_model.tflite")
LABELS_PATH = os.path.join(BASE_DIR, "Models", "labels.json")

# =====================================================
# CONFIG
# =====================================================
MAX_SEQ_LEN = 100
MIN_SEQ_LEN = 15
CONFIDENCE_THRESHOLD = 0.6
SMOOTHING_WINDOW = 10
N_FEATURES = 63

# =====================================================
# LOAD TFLITE MODEL
# =====================================================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"TFLite model not found at {MODEL_PATH}")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# =====================================================
# LOAD LABELS
# =====================================================
if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(f"Label map not found at {LABELS_PATH}")

with open(LABELS_PATH, "r") as f:
    label_map = json.load(f)

labels = [k for k, _ in sorted(label_map.items(), key=lambda x: x[1])]

# =====================================================
# MEDIAPIPE
# =====================================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# =====================================================
# UTILS
# =====================================================
def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks)
    landmarks -= np.mean(landmarks, axis=0)
    landmarks /= np.max(np.linalg.norm(landmarks, axis=1)) + 1e-6
    return landmarks.flatten()

def prepare_sequence(seq):
    if not seq:
        return np.zeros((1, MAX_SEQ_LEN, N_FEATURES), dtype=np.float32)

    padded = list(seq)
    while len(padded) < MAX_SEQ_LEN:
        padded.insert(0, padded[0])

    padded = np.array(padded[-MAX_SEQ_LEN:])
    return padded.reshape(1, MAX_SEQ_LEN, N_FEATURES).astype(np.float32)

# =====================================================
# MAIN
# =====================================================
def main():
    print("[INFO] Starting GhostPen Live Recognition")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera not accessible")
        input("Press Enter to exit...")
        return

    print("[INFO] Camera opened")

    sequence = deque(maxlen=MAX_SEQ_LEN)
    smooth_preds = deque(maxlen=SMOOTHING_WINDOW)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Frame read failed")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            label = None
            conf = 0.0

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                coords = normalize_landmarks(coords)
                sequence.append(coords)

                if len(sequence) >= MIN_SEQ_LEN:
                    inp = prepare_sequence(sequence)
                    interpreter.set_tensor(input_details[0]["index"], inp)
                    interpreter.invoke()
                    pred = interpreter.get_tensor(output_details[0]["index"])[0]

                    smooth_preds.append(np.argmax(pred))
                    label = Counter(smooth_preds).most_common(1)[0][0]
                    conf = pred[label]
                    label = labels[label]

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

            if label and conf >= CONFIDENCE_THRESHOLD:
                cv2.putText(
                    frame,
                    f"{label} ({conf:.2f})",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            cv2.imshow("GhostPen - Live Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] GhostPen stopped")

# =====================================================
if __name__ == "__main__":
    main()
