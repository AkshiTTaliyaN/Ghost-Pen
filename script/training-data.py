import os
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# --- Reproducibility: Set all seeds ---
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU for full determinism
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "..", "DATA", "collected")
MODEL_DIR = os.path.join(BASE_DIR, "..", "Models")

MODEL_PATH = os.path.join(MODEL_DIR, "trained_model.h5")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

MAX_SEQ_LEN = 100
N_FEATURES = 21 * 3  # 63 (x, y, z) per 21 landmarks

# --- Load Data ---
def load_dataset():
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        raise FileNotFoundError(f"❌ No gesture folders found in {DATA_DIR}.")

    X, y = [], []
    label_map = {label: idx for idx, label in enumerate(sorted(os.listdir(DATA_DIR)))}

    for label, idx in label_map.items():
        folder = os.path.join(DATA_DIR, label)
        for file in os.listdir(folder):
            try:
                sequence = np.load(os.path.join(folder, file))
                sequence = sequence.reshape(-1, N_FEATURES)
                X.append(sequence)
                y.append(idx)
            except Exception as e:
                print(f"⚠️ Skipping {file}: {e}")

    if not X:
        raise ValueError("No valid data loaded.")
    
    return X, y, label_map

# --- Build Model ---
def build_model(num_classes):
    model = Sequential([
        Masking(mask_value=0.0, input_shape=(MAX_SEQ_LEN, N_FEATURES)),
        LSTM(128, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# --- Train ---
def main():
    print("Loading dataset...")
    X, y, label_map = load_dataset()

    X = pad_sequences(X, maxlen=MAX_SEQ_LEN, padding='post', dtype='float32')
    y = to_categorical(y)

    print(f"Loaded {len(X)} sequences from {len(label_map)} gesture classes")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    print("Building and training model...")
    model = build_model(num_classes=y.shape[1])
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)

    with open(LABELS_PATH, "w") as f:
        json.dump(label_map, f, sort_keys=True)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Label map saved to {LABELS_PATH}")

    # Optional: Save predictions for analysis
    predictions = model.predict(X_test)
    np.save(os.path.join(MODEL_DIR, "sample_predictions.npy"), predictions)



if __name__ == "__main__":
    main()
