import os
import numpy as np
import json
from tensorflow.keras.models import load_model

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "DATA", "collected")

MODEL_PATH = os.path.join(BASE_DIR, "..", "Models", "trained_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "..", "Models", "labels.json")

# --- Config ---
MAX_SEQ_LEN = 100

# --- Output container ---
diagnostic = {
    "gesture_preview": [],
    "model_prediction_samples": [],
    "errors": []
}

# --- Load label map ---
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r") as f:
        label_map = json.load(f)
    labels = [label for label, _ in sorted(label_map.items(), key=lambda x: x[1])]
else:
    diagnostic["errors"].append("❌ labels.json not found.")
    labels = []

# --- Load model ---
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        diagnostic["errors"].append(f"❌ Failed to load model: {e}")
        model = None
else:
    diagnostic["errors"].append("❌ trained_model.h5 not found.")
    model = None

# --- Scan and test ---
if os.path.exists(DATA_DIR) and os.listdir(DATA_DIR):
    for label in sorted(os.listdir(DATA_DIR)):
        folder = os.path.join(DATA_DIR, label)
        if not os.path.isdir(folder):
            continue

        files = [f for f in os.listdir(folder) if f.endswith(".npy")]
        diagnostic["gesture_preview"].append((label, len(files)))

        # Pick one file per label for prediction
        if model and files:
            for file in files[:1]:
                try:
                    path = os.path.join(folder, file)
                    data = np.load(path)

                    # Pad to MAX_SEQ_LEN
                    sequence = data.copy().tolist()
                    while len(sequence) < MAX_SEQ_LEN:
                        sequence.insert(0, sequence[0])
                    sequence = np.array(sequence[-MAX_SEQ_LEN:])

                    input_tensor = sequence.reshape(1, MAX_SEQ_LEN, -1).astype(np.float32)

                    pred = model.predict(input_tensor, verbose=0)[0]
                    predicted_label = labels[np.argmax(pred)]
                    confidence = float(np.max(pred))

                    diagnostic["model_prediction_samples"].append({
                        "true_label": label,
                        "predicted": predicted_label,
                        "confidence": round(confidence, 3),
                        "file": file
                    })

                except Exception as e:
                    diagnostic["errors"].append(f" Error with {label}/{file}: {str(e)}")
else:
    diagnostic["errors"].append("No gesture folders or .npy files found.")

# --- Print summary ---
print("\n Sample Summary:")
for label, count in diagnostic["gesture_preview"]:
    print(f" - {label}: {count} samples")

if diagnostic["model_prediction_samples"]:
    print("\n🔍 Predictions on Sample Files:")
    for entry in diagnostic["model_prediction_samples"]:
        print(f" - {entry['file']} | Truth: {entry['true_label']} → Predicted: {entry['predicted']} ({entry['confidence']})")

if diagnostic["errors"]:
    print("\n❗ Errors:")
    for e in diagnostic["errors"]:
        print(" -", e)
