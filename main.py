"""
Main script to run the full key frame detection training pipeline.
1. Extract frames from SumMe videos
2. Load frames dataset
3. Train Autoencoder
4. Train CNN + LSTM
"""

from preprocessing.extract_frames import extract_summe_dataset
from preprocessing.dataset_loader import load_frames_dataset
from training.train_autoencoder import train_autoencoder
from training.train_cnn_lstm import train_cnn_lstm_model
from inference.detect_keyframes_ae import detect_keyframes_autoencoder
# -----------------------------
# Configuration Variables
# -----------------------------
SUMME_VIDEO_DIR = "data/SumMe/videos"
SUMME_FRAMES_DIR = "data/SumMe/frames"
FPS = 2

IMG_SIZE = (224, 224)
BATCH_SIZE_AE = 5
EPOCHS_AE = 3

SEQ_LEN = 20
BATCH_SIZE_LSTM = 2
EPOCHS_LSTM = 2


TEST_FRAMES_DIR = "data/test/frames"

# -----------------------------
# Step 1: Extract frames
# -----------------------------
# print("[STEP 1] Extracting frames from SumMe videos...")
# extract_summe_dataset(SUMME_VIDEO_DIR, SUMME_FRAMES_DIR, fps=FPS)

# -----------------------------
# Step 2: Load frames dataset
# -----------------------------
# print("[STEP 2] Loading frames dataset for training...")
# frames_dataset = load_frames_dataset(SUMME_FRAMES_DIR, img_size=IMG_SIZE)
# print(f"[INFO] Total frames loaded: {frames_dataset.shape[0]}")

# -----------------------------
# Step 3: Train Autoencoder
# -----------------------------
# print("[STEP 3] Training Autoencoder...")
# train_autoencoder(
#     frames_dataset,
#     img_size=IMG_SIZE,
#     batch_size=BATCH_SIZE_AE,
#     epochs=EPOCHS_AE,
#     save_path="models/autoencoder.h5"
# )

print("[STEP 5] Detecting keyframes using Autoencoder...")

keyframes_ae, errors_ae = detect_keyframes_autoencoder(
    frames_dir=TEST_FRAMES_DIR,
    model_path="models/autoencoder.h5",
    img_size=IMG_SIZE,
    threshold_mode="mean_std",   # or "percentile"
    percentile=90,              # only used if threshold_mode="percentile"
    visualize=True,
    save_keyframes=True,
    output_dir="results/autoencoder_keyframes"
)

print("[DONE] Autoencoder keyframe detection completed.")

# -----------------------------
# Step 4: Train CNN + LSTM
# -----------------------------
# print("[STEP 4] Training CNN + LSTM...")
# train_cnn_lstm_model(
#     frames_dataset,
#     seq_len=SEQ_LEN,
#     batch_size=BATCH_SIZE_LSTM,
#     epochs=EPOCHS_LSTM,
#     save_path="models/cnn_lstm.h5"
# )

# print("[DONE] Training pipeline completed successfully!")
