import os
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from models.cnn_lstm import build_cnn_lstm


def create_sequences(frames, seq_len=20):
    """
    Weakly supervised sequences.
    """
    X, y = [], []
    for i in range(len(frames) - seq_len):
        X.append(frames[i:i + seq_len])

        # Weak labels (same idea you used)
        y.append(np.random.randint(0, 2, size=(seq_len, 1)))

    return np.array(X), np.array(y)


def train_cnn_lstm(
    frames,
    seq_len=20,
    batch_size=2,
    max_epochs=30,
    save_path="models/cnn_lstm.h5",
    val_split=0.1
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    X, y = create_sequences(frames, seq_len)

    model = build_cnn_lstm(
        input_shape=(seq_len, frames.shape[1], frames.shape[2], 3)
    )

    model.summary()

    # -----------------------------
    # Callbacks (same philosophy as AE)
    # -----------------------------
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        save_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    model.fit(
        X,
        y,
        validation_split=val_split,
        epochs=max_epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    print(f"[INFO] CNN+LSTM training completed.")
    print(f"[INFO] Best model saved at {save_path}")

    return model
