import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Directory layout ─────────────────────────────────────────────────────────
DATA_DIR       = os.path.join(ROOT_DIR, "DATA", "collected")
MODEL_DIR      = os.path.join(ROOT_DIR, "Models")
SCRIPT_DIR     = os.path.join(ROOT_DIR, "script")

# ── Model files ──────────────────────────────────────────────────────────────
MODEL_H5       = os.path.join(MODEL_DIR, "trained_model.h5")
MODEL_TFLITE   = os.path.join(MODEL_DIR, "trained_model.tflite")
LABELS_PATH    = os.path.join(MODEL_DIR, "labels.json")
PREDICTIONS_PATH = os.path.join(MODEL_DIR, "sample_predictions.npy")

# ── Model hyperparameters ────────────────────────────────────────────────────
MAX_SEQ_LEN    = 100        # frames per gesture sequence
MIN_SEQ_LEN    = 30         
N_LANDMARKS    = 21         # MediaPipe hand landmarks
N_COORDS       = 3          # x, y, z per landmark
N_FEATURES     = N_LANDMARKS * N_COORDS   # = 63

# ── Training hyperparameters ─────────────────────────────────────────────────
SEED           = 42
EPOCHS         = 50         
BATCH_SIZE     = 32
TEST_SPLIT     = 0.2
VAL_SPLIT      = 0.1
PATIENCE       = 8          
LSTM_UNITS     = 128
DENSE_UNITS    = 64
DROPOUT_RATE   = 0.4

# ── Data collection ──────────────────────────────────────────────────────────
COLLECTION_MAX_FRAMES = 100  # frames captured per gesture recording

# ── Inference ────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.65  
SMOOTHING_WINDOW     = 10    

PADDING_MODE = 'pre'
