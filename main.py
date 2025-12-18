"""
Main script to run the full key frame detection training pipeline.

Steps:
1. Preprocess videos -> extract frames (SumMe dataset)
2. Load frames for training
3. Train Autoencoder
4. Train CNN + LSTM
"""

import os
from preprocessing.extract_frames import extract_summe_dataset
from preprocessing.dataset_loader import load_frames_dataset
from training.train_autoencoder import train_autoencoder
from training.train_cnn_lstm import train_cnn_lstm

# -----------------------------
# Config
# -----------------------------
SUMME_VIDEO_DIR = "data/SumMe/videos"
SUMME_FRAMES_DIR = "data/SumMe/frames"
FPS = 2

IMG_SIZE = (224, 224)
BATCH_SIZE_AE = 16
EPOCHS_AE = 10

SEQ_LEN = 20
BATCH_SIZE_LSTM = 2
EPOCHS_LSTM = 5


# -----------------------------
# Step 2: Load frames dataset
# -----------------------------
print("[STEP 2] Loading frames dataset for training...")
frames_dataset = load_frames_dataset(SUMME_FRAMES_DIR, img_size=IMG_SIZE)
print(f"[INFO] Total frames loaded: {frames_dataset.shape[0]}")

# -----------------------------
# Step 3: Train Autoencoder
# -----------------------------
print("[STEP 3] Training Autoencoder...")
autoencoder_model = train_autoencoder(
    frames_dataset,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE_AE,
    epochs=EPOCHS_AE,
    save_path="models/autoencoder.h5"
)

# -----------------------------
# Step 4: Train CNN + LSTM
# -----------------------------
print("[STEP 4] Training CNN + LSTM...")
cnn_lstm_model = train_cnn_lstm(
    frames_dataset,
    seq_len=SEQ_LEN,
    batch_size=BATCH_SIZE_LSTM,
    epochs=EPOCHS_LSTM,
    save_path="models/cnn_lstm.h5"
)

print("[DONE] Training pipeline completed successfully!")
