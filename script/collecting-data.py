import os
import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime

# --- Config ---
DATA_DIR = "GHOSTPEN/DATA/collected"
MAX_FRAMES = 100

LABEL = os.environ.get("GESTURE_NAME")
if LABEL is None or not LABEL.strip():
    print("ERROR: Gesture label not provided.")
    print("Launch this script from the GhostPen GUI.")
    exit(1)

LABEL = LABEL.strip()

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- Normalize hand landmarks ---
def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks)
    landmarks -= np.mean(landmarks, axis=0)
    landmarks /= np.max(np.linalg.norm(landmarks, axis=1)) + 1e-6
    return landmarks.flatten()

# --- Save Sequence to File ---
def save_sequence(sequence, label):
    if not sequence:
        print("Empty sequence, nothing saved.")
        return

    os.makedirs(os.path.join(DATA_DIR, label), exist_ok=True)
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".npy"
    path = os.path.join(DATA_DIR, label, filename)
    np.save(path, np.array(sequence))
    print(f"Saved: {path}")

# --- Main Collection ---
def main():
    cap = cv2.VideoCapture(0)
    sequence = []
    collecting = False

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
        print("\nPress 'c' to start/stop collecting")
        print("Press 'q' to quit")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                coords = normalize_landmarks(coords)

                if collecting:
                    sequence.append(coords)
                    if len(sequence) >= MAX_FRAMES:
                        save_sequence(sequence, LABEL)
                        sequence = []
                        collecting = False

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

            cv2.putText(
                frame, f"Label: {LABEL}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

            cv2.putText(
                frame, f"Collecting: {collecting} | Frames: {len(sequence)}",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0) if collecting else (0, 0, 255), 2
            )

            cv2.imshow("GhostPen - Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                collecting = not collecting
                if not collecting and sequence:
                    save_sequence(sequence, LABEL)
                    sequence = []
            elif key == ord('q'):
                if collecting and sequence:
                    save_sequence(sequence, LABEL)
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
