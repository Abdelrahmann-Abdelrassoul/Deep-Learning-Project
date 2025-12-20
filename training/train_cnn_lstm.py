import os
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import Sequence
from models.cnn_lstm import build_cnn_lstm


class FrameSequenceGenerator(Sequence):
    def __init__(self, frames, seq_len=20, batch_size=2):
        self.frames = frames.astype("float32")   # 🔴 IMPORTANT
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.indices = np.arange(len(frames) - seq_len)

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        X = np.zeros(
            (len(batch_idx), self.seq_len, *self.frames.shape[1:]),
            dtype="float32"
        )
        y = np.zeros((len(batch_idx), self.seq_len, 1), dtype="float32")

        for i, start in enumerate(batch_idx):
            X[i] = self.frames[start:start + self.seq_len]
            y[i] = np.random.randint(0, 2, size=(self.seq_len, 1))

        return X, y


def train_cnn_lstm(
    frames,
    seq_len=20,
    batch_size=2,
    max_epochs=30,
    save_path="models/cnn_lstm.h5"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    generator = FrameSequenceGenerator(
        frames=frames,
        seq_len=seq_len,
        batch_size=batch_size
    )

    model = build_cnn_lstm(
        input_shape=(seq_len, frames.shape[1], frames.shape[2], 3)
    )

    model.summary()

    early_stop = EarlyStopping(
        monitor="loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        save_path,
        monitor="loss",
        save_best_only=True,
        verbose=1
    )

    model.fit(
        generator,
        epochs=max_epochs,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    print(f"[INFO] CNN+LSTM training completed.")
    print(f"[INFO] Model saved at {save_path}")

    return model
