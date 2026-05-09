import os
import sys
import json
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR, MODEL_TFLITE, LABELS_PATH,
    MAX_SEQ_LEN, N_FEATURES,
)

# ── Output container ──────────────────────────────────────────────────────────
diagnostic = {
    "gesture_preview": [],
    "model_prediction_samples": [],
    "errors": [],
    "warnings": [],
}

# ── Load label map ────────────────────────────────────────────────────────────
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r") as f:
        label_map = json.load(f)
    labels = [k for k, _ in sorted(label_map.items(), key=lambda x: x[1])]
    print(f"Labels loaded: {labels}")
else:
    diagnostic["errors"].append(f"labels.json not found at {LABELS_PATH}")
    labels = []

# ── Load TFLite model ─────────────────────────────────────────────────────────
interpreter = None
if os.path.exists(MODEL_TFLITE):
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_TFLITE)
        interpreter.allocate_tensors()
        input_details  = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Verify input shape matches config
        expected_shape = [1, MAX_SEQ_LEN, N_FEATURES]
        actual_shape   = list(input_details[0]["shape"])
        if actual_shape != expected_shape:
            diagnostic["errors"].append(
                f"Model input shape mismatch: expected {expected_shape}, got {actual_shape}. "
                f"Check MAX_SEQ_LEN and N_FEATURES in config.py."
            )
        else:
            print(f"Model input shape: {actual_shape} — OK")

    except Exception as e:
        diagnostic["errors"].append(f"Failed to load TFLite model: {e}")
        interpreter = None
else:
    diagnostic["errors"].append(
        f"trained_model.tflite not found at {MODEL_TFLITE}\n"
        "  Run training-data.py first to generate it."
    )

# ── Padding helper (matches training + inference exactly) ─────────────────────
def pad_sequence(seq: np.ndarray) -> np.ndarray:
    """Pre-pad with first frame to MAX_SEQ_LEN. Matches config.PADDING_MODE='pre'."""
    if len(seq) >= MAX_SEQ_LEN:
        return seq[-MAX_SEQ_LEN:]
    pad_width  = MAX_SEQ_LEN - len(seq)
    first_frame = seq[0:1]
    padding    = np.repeat(first_frame, pad_width, axis=0)
    return np.vstack([padding, seq])

# ── Scan data and run predictions ─────────────────────────────────────────────
if os.path.exists(DATA_DIR) and os.listdir(DATA_DIR):
    for label in sorted(os.listdir(DATA_DIR)):
        folder = os.path.join(DATA_DIR, label)
        if not os.path.isdir(folder):
            continue

        files = [f for f in os.listdir(folder) if f.endswith(".npy")]
        diagnostic["gesture_preview"].append((label, len(files)))

        # Warn if below recommended sample count
        if len(files) < 500:
            diagnostic["warnings"].append(
                f"{label}: {len(files)} samples (recommend 500+, need {500 - len(files)} more)"
            )

        # Run one prediction per label
        if interpreter and files:
            for fname in files[:1]:
                try:
                    path = os.path.join(folder, fname)
                    data = np.load(path).reshape(-1, N_FEATURES)

                    padded = pad_sequence(data)
                    inp    = padded.reshape(1, MAX_SEQ_LEN, N_FEATURES).astype(np.float32)

                    interpreter.set_tensor(input_details[0]["index"], inp)
                    interpreter.invoke()
                    pred = interpreter.get_tensor(output_details[0]["index"])[0]

                    predicted_label = labels[np.argmax(pred)]
                    confidence      = float(np.max(pred))

                    diagnostic["model_prediction_samples"].append({
                        "true_label":  label,
                        "predicted":   predicted_label,
                        "confidence":  round(confidence, 3),
                        "correct":     label == predicted_label,
                        "file":        fname,
                    })

                except Exception as e:
                    diagnostic["errors"].append(f"Error with {label}/{fname}: {e}")
else:
    diagnostic["errors"].append(
        f"No gesture folders found in {DATA_DIR}\n"
        "  Run collecting-data.py to record gesture samples first."
    )

# ── Print summary ─────────────────────────────────────────────────────────────
print("\n── Sample counts ────────────────────────────────────────")
for label, count in diagnostic["gesture_preview"]:
    bar    = "✓" if count >= 500 else f"✗ need {500 - count} more"
    print(f"  {label:>8}: {count:>5} samples  {bar}")

if diagnostic["model_prediction_samples"]:
    correct = sum(1 for e in diagnostic["model_prediction_samples"] if e["correct"])
    total   = len(diagnostic["model_prediction_samples"])
    print(f"\n── Predictions on one sample per class ({correct}/{total} correct) ──")
    for entry in diagnostic["model_prediction_samples"]:
        tick = "✓" if entry["correct"] else "✗"
        print(
            f"  {tick} {entry['file'][:30]:<32} "
            f"true={entry['true_label']:>8}  "
            f"pred={entry['predicted']:>8}  "
            f"conf={entry['confidence']:.3f}"
        )

if diagnostic["warnings"]:
    print("\n── Warnings ─────────────────────────────────────────────")
    for w in diagnostic["warnings"]:
        print(f"  ⚠  {w}")

if diagnostic["errors"]:
    print("\n── Errors ───────────────────────────────────────────────")
    for e in diagnostic["errors"]:
        print(f"  ✗  {e}")

if not diagnostic["errors"] and not diagnostic["warnings"]:
    print("\n✓ All checks passed.")
