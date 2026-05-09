import os
import sys
import json
import time
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MODEL_TFLITE, LABELS_PATH,
    MAX_SEQ_LEN, MIN_SEQ_LEN, N_FEATURES,
    CONFIDENCE_THRESHOLD, SMOOTHING_WINDOW,
)

# ── Load TFLite interpreter ───────────────────────────────────────────────────
if not os.path.exists(MODEL_TFLITE):
    raise FileNotFoundError(
        f"TFLite model not found: {MODEL_TFLITE}\n"
        "Run training-data.py first, or copy trained_model.tflite to Models/"
    )

interpreter = tf.lite.Interpreter(model_path=MODEL_TFLITE)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ── Load labels ───────────────────────────────────────────────────────────────
if not os.path.exists(LABELS_PATH):
    raise FileNotFoundError(f"Labels not found: {LABELS_PATH}")

with open(LABELS_PATH) as f:
    label_map = json.load(f)

labels = [k for k, _ in sorted(label_map.items(), key=lambda x: x[1])]
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
    """
    Pre-pad sequence to MAX_SEQ_LEN by repeating the first real frame.
    This MATCHES the padding used in training (config.PADDING_MODE = 'pre').
    """
    frames = list(seq)
    if len(frames) == 0:
        return np.zeros((1, MAX_SEQ_LEN, N_FEATURES), dtype=np.float32)

    # Trim to last MAX_SEQ_LEN frames if longer
    frames = frames[-MAX_SEQ_LEN:]

    # Pre-pad with first frame
    pad_len = MAX_SEQ_LEN - len(frames)
    if pad_len > 0:
        padding = [frames[0]] * pad_len
        frames  = padding + frames

    return np.array(frames, dtype=np.float32).reshape(1, MAX_SEQ_LEN, N_FEATURES)


# ── Word builder state ────────────────────────────────────────────────────────
class WordBuilder:
    """
    Accumulates confirmed letters into a word.
    A letter is 'confirmed' when the smoothed prediction holds stable
    for CONFIRM_FRAMES consecutive frames on the same non-neutral label.
    """
    CONFIRM_FRAMES = 20   # ~0.7s at 30fps — hold gesture this long to confirm

    def __init__(self):
        self.word          = ""
        self.last_label    = None
        self.hold_count    = 0
        self.last_added    = ""   # prevents adding same letter twice in a row
        self.last_add_time = 0.0

    def update(self, label: str | None) -> str | None:
        """Returns the confirmed letter if one was just added, else None."""
        if label is None or label == NEUTRAL_LABEL:
            self.hold_count = 0
            self.last_label = None
            return None

        if label == self.last_label:
            self.hold_count += 1
        else:
            self.last_label = label
            self.hold_count = 1

        # Confirm: held for CONFIRM_FRAMES and not same as last added
        # Also enforce 1.5s gap between repeated same letters
        now = time.time()
        if (
            self.hold_count == self.CONFIRM_FRAMES
            and (label != self.last_added or now - self.last_add_time > 1.5)
        ):
            self.word         += label
            self.last_added    = label
            self.last_add_time = now
            return label

        return None

    def backspace(self):
        self.word = self.word[:-1]

    def space(self):
        self.word += " "

    def clear(self):
        self.word = ""


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("[INFO] Starting GhostPen Live Recognition")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera not accessible.")
        input("Press Enter to exit...")
        return

    sequence     = deque(maxlen=MAX_SEQ_LEN)
    smooth_preds = deque(maxlen=SMOOTHING_WINDOW)
    word_builder = WordBuilder()

    consecutive_failures = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures > 10:
                    print("[ERROR] Camera feed lost after 10 consecutive failures.")
                    break
                continue
            consecutive_failures = 0

            frame  = cv2.flip(frame, 1)
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            current_label = None
            frame_conf    = 0.0
            vote_conf     = 0.0

            if result.multi_hand_landmarks:
                hand_lm = result.multi_hand_landmarks[0]
                coords  = [(lm.x, lm.y, lm.z) for lm in hand_lm.landmark]
                coords  = normalize_landmarks(coords)
                sequence.append(coords)

                if len(sequence) >= MIN_SEQ_LEN:
                    inp = prepare_sequence(sequence)
                    interpreter.set_tensor(input_details[0]["index"], inp)
                    interpreter.invoke()
                    pred = interpreter.get_tensor(output_details[0]["index"])[0]

                    smooth_preds.append(int(np.argmax(pred)))

                    # FIX 1: capture index and confidence BEFORE converting to string
                    top_idx   = Counter(smooth_preds).most_common(1)[0][0]
                    top_votes = Counter(smooth_preds).most_common(1)[0][1]

                    frame_conf = float(pred[top_idx])          # raw frame prob
                    vote_conf  = top_votes / len(smooth_preds) # smoothed vote ratio

                    # FIX 2: only convert to string label AFTER using index
                    current_label = labels[top_idx]

                mp_drawing.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS
                )
            else:
                # No hand detected — reset sequence to avoid stale context
                sequence.clear()
                smooth_preds.clear()

            # ── Word builder update ──────────────────────────────────────────
            display_label = (
                current_label
                if current_label and frame_conf >= CONFIDENCE_THRESHOLD
                else None
            )
            confirmed = word_builder.update(display_label)

            # ── HUD ──────────────────────────────────────────────────────────
            h, w = frame.shape[:2]

            # Current gesture label
            if display_label and display_label != NEUTRAL_LABEL:
                label_text = f"{display_label}  {vote_conf:.0%}"
                color      = (0, 255, 0) if confirmed is None else (0, 200, 255)
                cv2.putText(frame, label_text, (10, 46),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

            # Hold progress bar
            if word_builder.hold_count > 0 and display_label not in (None, NEUTRAL_LABEL):
                progress = word_builder.hold_count / WordBuilder.CONFIRM_FRAMES
                bar_w    = int((w - 20) * progress)
                cv2.rectangle(frame, (10, h - 20), (w - 10, h - 8),
                              (60, 60, 60), -1)
                cv2.rectangle(frame, (10, h - 20), (10 + bar_w, h - 8),
                              (0, 200, 255), -1)

            # Word display strip at bottom
            word_display = word_builder.word if word_builder.word else "_"
            cv2.rectangle(frame, (0, h - 55), (w, h - 25), (30, 30, 30), -1)
            cv2.putText(frame, word_display, (10, h - 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            cv2.imshow("GhostPen — Live Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('\b') or key == 8:  # backspace
                word_builder.backspace()
            elif key == ord(' '):
                word_builder.space()
            elif key == ord('c'):
                word_builder.clear()

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[INFO] GhostPen stopped.")
    if word_builder.word.strip():
        print(f"[INFO] Final word buffer: '{word_builder.word}'")


if __name__ == "__main__":
    main()
