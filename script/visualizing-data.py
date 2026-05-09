import os
import sys
import json
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import matplotlib.pyplot as plt
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODEL_TFLITE, LABELS_PATH,
    MAX_SEQ_LEN, MIN_SEQ_LEN, N_FEATURES,
    CONFIDENCE_THRESHOLD,
)

# ── Load TFLite model ─────────────────────────────────────────────────────────
if not os.path.exists(MODEL_TFLITE):
    raise FileNotFoundError(
        f"TFLite model not found: {MODEL_TFLITE}\n"
        "Run training-data.py first to generate it."
    )

interpreter = tf.lite.Interpreter(model_path=MODEL_TFLITE)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ── Load labels ───────────────────────────────────────────────────────────────
if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(f"Labels not found: {LABELS_PATH}")

with open(LABELS_PATH, "r") as f:
    label_map = json.load(f)

labels       = [k for k, _ in sorted(label_map.items(), key=lambda x: x[1])]
NEUTRAL_LABEL = "neutral"

# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# ── Utilities ─────────────────────────────────────────────────────────────────
def normalize_landmarks(landmarks) -> np.ndarray:
    lm = np.array(landmarks, dtype=np.float32)
    lm -= lm.mean(axis=0)
    lm /= (np.max(np.linalg.norm(lm, axis=1)) + 1e-6)
    return lm.flatten()


def prepare_sequence(seq: deque) -> np.ndarray:
    """Pre-pad to MAX_SEQ_LEN — matches training and inference exactly."""
    frames = list(seq)[-MAX_SEQ_LEN:]
    pad_len = MAX_SEQ_LEN - len(frames)
    if pad_len > 0:
        frames = [frames[0]] * pad_len + frames
    return np.array(frames, dtype=np.float32).reshape(1, MAX_SEQ_LEN, N_FEATURES)


# ── Matplotlib setup ──────────────────────────────────────────────────────────
plt.ion()
fig, ax = plt.subplots(figsize=(max(8, len(labels)), 4))

bar_colors = ["#4a9eff"] * len(labels)
bars       = ax.bar(labels, [0.0] * len(labels), color=bar_colors)

ax.set_ylim(0, 1)
ax.set_ylabel("Confidence")
ax.set_title("GhostPen — Live Prediction Confidence")
ax.axhline(y=CONFIDENCE_THRESHOLD, color="red", linestyle="--",
           linewidth=1, label=f"threshold ({CONFIDENCE_THRESHOLD})")
ax.legend(fontsize=8)
plt.tight_layout()

try:
    fig.canvas.manager.set_window_title("GhostPen — Prediction Visualizer")
except Exception:
    pass  # not all backends support this


def update_plot(prediction: np.ndarray, top_label: str | None):
    """Update bar heights and highlight the top prediction."""
    top_idx = int(np.argmax(prediction))
    for i, bar in enumerate(bars):
        bar.set_height(float(prediction[i]))
        # Highlight top bar in green if above threshold, red otherwise
        if i == top_idx:
            above = prediction[i] >= CONFIDENCE_THRESHOLD
            bar.set_color("#00c853" if above else "#ff5252")
        else:
            bar.set_color("#4a9eff")
    fig.canvas.draw()
    fig.canvas.flush_events()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return

    sequence = deque(maxlen=MAX_SEQ_LEN)
    prediction = np.zeros(len(labels), dtype=np.float32)

    print("[INFO] Visualizer running. Press 'q' in the camera window to quit.")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame  = cv2.flip(frame, 1)
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            label = None
            conf  = 0.0

            if result.multi_hand_landmarks:
                hand_lm = result.multi_hand_landmarks[0]
                coords  = [(lm.x, lm.y, lm.z) for lm in hand_lm.landmark]
                coords  = normalize_landmarks(coords)
                sequence.append(coords)

                if len(sequence) >= MIN_SEQ_LEN:
                    inp = prepare_sequence(sequence)
                    interpreter.set_tensor(input_details[0]["index"], inp)
                    interpreter.invoke()
                    prediction = interpreter.get_tensor(output_details[0]["index"])[0]

                    conf  = float(np.max(prediction))
                    label = labels[int(np.argmax(prediction))]

                mp_drawing.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS
                )
            else:
                # FIX: reset on hand lost — prevents stale context
                sequence.clear()
                prediction = np.zeros(len(labels), dtype=np.float32)

            # Update chart
            update_plot(prediction, label)

            # ── Camera feed overlay ───────────────────────────────────────────
            if label and label != NEUTRAL_LABEL and conf >= CONFIDENCE_THRESHOLD:
                cv2.putText(
                    frame, f"{label}  {conf:.2f}",
                    (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2
                )
            elif label == NEUTRAL_LABEL:
                cv2.putText(
                    frame, "neutral",
                    (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1
                )

            # Show top-2 predictions as text for debugging
            if len(prediction) > 0 and prediction.sum() > 0:
                top2_idx = np.argsort(prediction)[-2:][::-1]
                debug_text = "  ".join(
                    f"{labels[i]}:{prediction[i]:.2f}" for i in top2_idx
                )
                cv2.putText(
                    frame, debug_text,
                    (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
                )

            cv2.imshow("GhostPen — Live Feed", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.close()
    print("[INFO] Visualizer stopped.")


if __name__ == "__main__":
    main()
