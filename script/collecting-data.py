import os
import sys
import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime

# ── Path fix: always find config regardless of working directory ─────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR, COLLECTION_MAX_FRAMES, N_FEATURES
)

# ── Gesture label from env (set by GUI) ─────────────────────────────────────
LABEL = os.environ.get("GESTURE_NAME", "").strip()
if not LABEL:
    print("ERROR: Gesture label not provided.")
    print("Set env var GESTURE_NAME or launch from the GhostPen GUI.")
    sys.exit(1)

# ── MediaPipe ────────────────────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def normalize_landmarks(landmarks):
    """
    Centre on palm centroid, scale by max distance from centroid.
    Returns flat array of shape (63,).
    """
    landmarks = np.array(landmarks, dtype=np.float32)
    landmarks -= landmarks.mean(axis=0)
    max_dist = np.max(np.linalg.norm(landmarks, axis=1))
    landmarks /= (max_dist + 1e-6)
    return landmarks.flatten()


def count_existing_samples(label: str) -> int:
    folder = os.path.join(DATA_DIR, label)
    if not os.path.isdir(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.endswith(".npy")])


def save_sequence(sequence: list, label: str) -> str:
    if not sequence:
        return ""
    os.makedirs(os.path.join(DATA_DIR, label), exist_ok=True)
    filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".npy"
    path = os.path.join(DATA_DIR, label, filename)
    np.save(path, np.array(sequence, dtype=np.float32))
    return path


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        sys.exit(1)

    sequence   = []
    collecting = False
    countdown  = 0   # frames remaining in pre-collection countdown

    saved_this_session = 0
    existing = count_existing_samples(LABEL)

    print(f"\nLabel: {LABEL} | Existing samples: {existing}")
    print("Controls:")
    print("  'c' — start/stop collecting")
    print("  'r' — discard current sequence (reset)")
    print("  'q' — quit\n")

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                hand_lm = result.multi_hand_landmarks[0]
                coords  = [(lm.x, lm.y, lm.z) for lm in hand_lm.landmark]
                coords  = normalize_landmarks(coords)

                if collecting and countdown == 0:
                    sequence.append(coords)
                    if len(sequence) >= COLLECTION_MAX_FRAMES:
                        path = save_sequence(sequence, LABEL)
                        saved_this_session += 1
                        total = existing + saved_this_session
                        print(f"  Saved [{total}]: {os.path.basename(path)}")
                        sequence   = []
                        collecting = False

                mp_drawing.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS
                )

            # ── Countdown logic ──────────────────────────────────────────────
            if collecting and countdown > 0:
                countdown -= 1

            # ── HUD ─────────────────────────────────────────────────────────
            total_samples = existing + saved_this_session

            cv2.putText(frame, f"Label: {LABEL}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.putText(frame,
                        f"Samples: {total_samples}  |  Session: {saved_this_session}",
                        (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (200, 200, 200), 1)

            if collecting and countdown > 0:
                msg   = f"GET READY... {countdown // 10 + 1}"
                color = (0, 200, 255)
            elif collecting:
                msg   = f"RECORDING  {len(sequence)}/{COLLECTION_MAX_FRAMES}"
                color = (0, 255, 0)
            else:
                msg   = "PAUSED — press 'c' to record"
                color = (0, 0, 220)

            cv2.putText(frame, msg, (10, 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            cv2.imshow("GhostPen — Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                if not collecting:
                    # Start: 30-frame countdown (~1 sec at 30fps) to let hand settle
                    collecting = True
                    countdown  = 30
                    sequence   = []
                else:
                    # Manual stop mid-sequence
                    if sequence:
                        path = save_sequence(sequence, LABEL)
                        saved_this_session += 1
                        print(f"  Saved (manual stop): {os.path.basename(path)}")
                    sequence   = []
                    collecting = False

            elif key == ord('r'):
                # Discard current sequence without saving
                sequence   = []
                collecting = False
                print("  Sequence discarded.")

            elif key == ord('q'):
                if collecting and sequence:
                    path = save_sequence(sequence, LABEL)
                    saved_this_session += 1
                    print(f"  Saved on quit: {os.path.basename(path)}")
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. {saved_this_session} new samples saved for '{LABEL}'.")
    print(f"Total for '{LABEL}': {existing + saved_this_session}")


if __name__ == "__main__":
    main()
