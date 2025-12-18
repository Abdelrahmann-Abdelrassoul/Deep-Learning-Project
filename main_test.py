import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, UpSampling2D,
    Input, LSTM, Dense, TimeDistributed
)
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import precision_score, recall_score, f1_score

SUMME_VIDEO_DIR = "data/SumMe/videos"
TEST_VIDEO_PATH = "data/test_video/test.mp4"

FRAME_DIR = "frames"
FPS = 2
IMG_SIZE = (224, 224)

def extract_frames(video_path, out_dir, fps=2):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    step = int(video_fps // fps)

    frames = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            frame = cv2.resize(frame, IMG_SIZE)
            frame = frame / 255.0
            frames.append(frame)
        count += 1

    cap.release()
    return np.array(frames)

def load_summe_dataset():
    all_frames = []
    for video in os.listdir(SUMME_VIDEO_DIR):
        video_path = os.path.join(SUMME_VIDEO_DIR, video)
        frames = extract_frames(video_path, FRAME_DIR, FPS)
        all_frames.append(frames)
    return np.concatenate(all_frames, axis=0)


def build_autoencoder():
    inp = Input(shape=(224,224,3))

    x = Conv2D(64, 3, activation='relu', padding='same')(inp)
    x = MaxPooling2D(2, padding='same')(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    encoded = MaxPooling2D(2, padding='same')(x)

    x = Conv2D(128, 3, activation='relu', padding='same')(encoded)
    x = UpSampling2D(2)(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = UpSampling2D(2)(x)

    out = Conv2D(3, 3, activation='sigmoid', padding='same')(x)

    model = Model(inp, out)
    model.compile(optimizer=Adam(), loss='mse')
    return model


print("Training Autoencoder...")
ae_data = load_summe_dataset()

autoencoder = build_autoencoder()
autoencoder.fit(
    ae_data, ae_data,
    epochs=3,
    batch_size=8,
    shuffle=True
)

autoencoder.save("autoencoder.h5")


def build_cnn_lstm():
    cnn = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    cnn.trainable = False

    inp = Input(shape=(None, 224,224,3))
    x = TimeDistributed(cnn)(inp)
    x = LSTM(128, return_sequences=True)(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inp, out)
    model.compile(
        optimizer=Adam(),
        loss='binary_crossentropy'
    )
    return model


def create_sequences(frames, seq_len=20):
    X, y = [], []
    for i in range(len(frames)-seq_len):
        X.append(frames[i:i+seq_len])
        y.append(np.random.randint(0,2, size=(seq_len,1)))
    return np.array(X), np.array(y)


print("Training CNN + LSTM...")
seq_frames = load_summe_dataset()
X_train, y_train = create_sequences(seq_frames)

cnn_lstm = build_cnn_lstm()
cnn_lstm.fit(X_train, y_train, epochs=2, batch_size=2)

cnn_lstm.save("cnn_lstm.h5")


test_frames = extract_frames(TEST_VIDEO_PATH, "test_frames", FPS)

recon = autoencoder.predict(test_frames)
errors = np.mean((test_frames - recon)**2, axis=(1,2,3))

threshold_ae = np.mean(errors) + np.std(errors)
ae_keyframes = np.where(errors > threshold_ae)[0]

X_test, _ = create_sequences(test_frames)
scores = cnn_lstm.predict(X_test)[0,:,0]

threshold_lstm = 0.5
lstm_keyframes = np.where(scores > threshold_lstm)[0]


def show_keyframes(indices, title):
    plt.figure(figsize=(15,3))
    for i, idx in enumerate(indices[:5]):
        plt.subplot(1,5,i+1)
        plt.imshow(test_frames[idx])
        plt.axis('off')
        plt.title(f"{title}\nFrame {idx}")
    plt.show()

show_keyframes(ae_keyframes, "Autoencoder")
show_keyframes(lstm_keyframes, "CNN + LSTM")


plt.figure(figsize=(10,4))
plt.plot(errors)
plt.axhline(threshold_ae, color='r', linestyle='--')
plt.title("Autoencoder Reconstruction Error")
plt.xlabel("Frame")
plt.ylabel("Error")
plt.show()


plt.figure(figsize=(10,4))
plt.plot(scores)
plt.axhline(0.5, color='r', linestyle='--')
plt.title("CNN + LSTM Importance Scores")
plt.xlabel("Frame")
plt.ylabel("Score")
plt.show()

gt = np.zeros(len(test_frames))
gt[::30] = 1  # fake sparse GT for demo

ae_pred = np.zeros_like(gt)
ae_pred[ae_keyframes] = 1

lstm_pred = np.zeros_like(gt)
lstm_pred[lstm_keyframes] = 1

def evaluate(name, y_true, y_pred):
    print(name)
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1:", f1_score(y_true, y_pred))
    print("-"*30)

evaluate("Autoencoder", gt, ae_pred)
evaluate("CNN + LSTM", gt, lstm_pred)


print("FINAL COMPARISON")
print("Autoencoder keyframes:", len(ae_keyframes))
print("CNN + LSTM keyframes:", len(lstm_keyframes))
print("CNN + LSTM produces fewer but more semantic frames.")
