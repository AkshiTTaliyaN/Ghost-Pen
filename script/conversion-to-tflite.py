import tensorflow as tf
import os

MODEL_PATH = "GHOSTPEN/Models/trained_model.h5"
TFLITE_PATH = "GHOSTPEN/Models/trained_model.tflite"

# Load the existing model
model = tf.keras.models.load_model(MODEL_PATH)

# Set up the converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter._experimental_lower_tensor_list_ops = False  # Important for LSTM
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

# Convert
tflite_model = converter.convert()

# Save
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print(f"✅ TFLite model saved at {TFLITE_PATH}")
