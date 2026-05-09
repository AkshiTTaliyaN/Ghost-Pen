import os
import sys
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR, MODEL_DIR, MODEL_H5, MODEL_TFLITE, LABELS_PATH, PREDICTIONS_PATH,
    MAX_SEQ_LEN, N_FEATURES, SEED, EPOCHS, BATCH_SIZE, TEST_SPLIT, VAL_SPLIT,
    PATIENCE, LSTM_UNITS, DENSE_UNITS, DROPOUT_RATE, PADDING_MODE,
)

# ── Reproducibility ──────────────────────────────────────────────────────────
os.environ['PYTHONHASHSEED']    = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ── Augmentation ─────────────────────────────────────────────────────────────
def augment_sequence(seq: np.ndarray) -> list[np.ndarray]:
    """
    Returns a list of augmented variants of the input sequence.
    seq shape: (T, 63)
    """
    augmented = []

    # 1. Gaussian noise
    noisy = seq + np.random.normal(0, 0.005, seq.shape).astype(np.float32)
    augmented.append(noisy)

    # 2. Scale (simulate different hand sizes / camera distances)
    scale = np.random.uniform(0.88, 1.12)
    augmented.append((seq * scale).astype(np.float32))

    # 3. Mirror (simulate left hand): flip x-coordinate of each landmark
    #    Landmarks are flattened as [x0,y0,z0, x1,y1,z1, ...]
    mirrored = seq.copy()
    mirrored[:, 0::3] *= -1   # negate every x value
    augmented.append(mirrored.astype(np.float32))

    # 4. Time warp: sub-sample to 80% length then pad back
    T = seq.shape[0]
    indices = np.sort(np.random.choice(T, int(T * 0.8), replace=False))
    warped  = seq[indices]
    augmented.append(warped)   # pad_sequences handles variable length

    return augmented


# ── Data loading ─────────────────────────────────────────────────────────────
def load_dataset(augment: bool = True):
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        raise FileNotFoundError(f"No gesture folders found in {DATA_DIR}")

    X, y = [], []
    label_map = {
        label: idx
        for idx, label in enumerate(sorted(os.listdir(DATA_DIR)))
        if os.path.isdir(os.path.join(DATA_DIR, label))
    }

    for label, idx in label_map.items():
        folder = os.path.join(DATA_DIR, label)
        files  = [f for f in os.listdir(folder) if f.endswith(".npy")]
        print(f"  {label}: {len(files)} samples", end="")

        for fname in files:
            try:
                seq = np.load(os.path.join(folder, fname))
                seq = seq.reshape(-1, N_FEATURES)
                X.append(seq)
                y.append(idx)

                if augment:
                    for aug_seq in augment_sequence(seq):
                        X.append(aug_seq.reshape(-1, N_FEATURES))
                        y.append(idx)
            except Exception as e:
                print(f"\n  WARNING: skipping {fname}: {e}")

        count_after = len([yi for yi in y if yi == idx])
        print(f"  →  {count_after} after augmentation")

    if not X:
        raise ValueError("No valid sequences loaded.")

    return X, y, label_map


# ── Padding ───────────────────────────────────────────────────────────────────
def pad(sequences):
    """
    Pre-pad all sequences to MAX_SEQ_LEN.
    Pre-padding (repeating the first real frame) matches inference behaviour
    and is more semantically meaningful than zero-padding.
    """
    padded = []
    for seq in sequences:
        seq = np.array(seq)
        if len(seq) >= MAX_SEQ_LEN:
            padded.append(seq[-MAX_SEQ_LEN:])
        else:
            pad_width = MAX_SEQ_LEN - len(seq)
            first_frame = seq[0:1]
            padding = np.repeat(first_frame, pad_width, axis=0)
            padded.append(np.vstack([padding, seq]))
    return np.array(padded, dtype=np.float32)


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model(num_classes: int) -> tf.keras.Model:
    model = Sequential([
        # Masking won't help with first-frame pre-padding (non-zero),
        # but kept for forward compatibility if zero-padding ever used.
        Masking(mask_value=0.0, input_shape=(MAX_SEQ_LEN, N_FEATURES)),
        LSTM(LSTM_UNITS, return_sequences=True),
        Dropout(DROPOUT_RATE),
        LSTM(64, return_sequences=False),     # second LSTM layer added
        BatchNormalization(),
        Dense(DENSE_UNITS, activation='relu'),
        Dropout(DROPOUT_RATE * 0.5),
        Dense(num_classes, activation='softmax'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading dataset...")
    X_raw, y_raw, label_map = load_dataset(augment=True)

    print(f"\nTotal sequences after augmentation: {len(X_raw)}")
    print(f"Classes: {list(label_map.keys())}")

    X = pad(X_raw)
    y = to_categorical(y_raw, num_classes=len(label_map))

    # Three-way split: train / val / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SPLIT, random_state=SEED, stratify=y_train
    )

    print(f"\nSplit — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=MODEL_H5,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("\nBuilding and training model...")
    model = build_model(num_classes=len(label_map))
    model.summary()

    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\n── Test set evaluation ──────────────────────────────────")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test accuracy : {acc:.4f}")
    print(f"  Test loss     : {loss:.4f}")

    y_pred  = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true  = np.argmax(y_test, axis=1)
    labels  = [k for k, _ in sorted(label_map.items(), key=lambda x: x[1])]

    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, target_names=labels))

    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true, y_pred))

    # ── Save label map ────────────────────────────────────────────────────────
    with open(LABELS_PATH, "w") as f:
        json.dump(label_map, f, sort_keys=True, indent=2)
    print(f"\nLabel map saved: {LABELS_PATH}")

    # ── Convert to TFLite ─────────────────────────────────────────────────────
    print("Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()

    with open(MODEL_TFLITE, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved: {MODEL_TFLITE}")

    # ── Save sample predictions for check.py ─────────────────────────────────
    preds = model.predict(X_test, verbose=0)
    np.save(PREDICTIONS_PATH, preds)
    print(f"Sample predictions saved: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
