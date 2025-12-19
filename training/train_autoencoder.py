import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from models.autoencoder import build_autoencoder


def train_autoencoder(
    frames,
    img_size=(224, 224),
    batch_size=16,
    max_epochs=50,              # ← AUTO-TUNED
    save_path="models/autoencoder.h5",
    val_split=0.1
):
    """
    Trains Autoencoder with:
    - Early stopping
    - Best model checkpoint
    - Automatic epoch selection
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model = build_autoencoder(img_size + (3,))
    model.summary()

    # -----------------------------
    # Callbacks (THIS WAS MISSING)
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

    history = model.fit(
        frames,
        frames,
        validation_split=val_split,
        epochs=max_epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    print("[INFO] Autoencoder training completed.")
    print(f"[INFO] Best model saved at: {save_path}")

    return model, history
