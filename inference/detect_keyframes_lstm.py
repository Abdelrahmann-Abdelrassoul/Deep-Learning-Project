import numpy as np
from tensorflow.keras.models import load_model

model = load_model("models/cnn_lstm.h5")
frames = np.load("data/test_video/X_test.npy")

scores = model.predict(frames)[0,:,0]
threshold = 0.5
keyframes = np.where(scores > threshold)[0]

print("Detected Key Frames (CNN-LSTM):", keyframes)
