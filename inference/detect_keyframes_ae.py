import numpy as np
from tensorflow.keras.models import load_model
from preprocessing.dataset_loader import load_frames

frames = load_frames("data/test_video/frames")
model = load_model("models/autoencoder.h5")

reconstructed = model.predict(frames)
errors = np.mean((frames - reconstructed) ** 2, axis=(1,2,3))

threshold = np.mean(errors) + np.std(errors)
keyframes = np.where(errors > threshold)[0]

print("Detected Key Frames (Autoencoder):", keyframes)
