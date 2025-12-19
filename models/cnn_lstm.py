from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TimeDistributed, LSTM, Dense
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
import numpy as np

def create_sequences(frames, seq_len=20):
    X, y = [], []
    for i in range(len(frames)-seq_len):
        X.append(frames[i:i+seq_len])
        y.append(np.random.randint(0,2, size=(seq_len,1)))  # weak labels
    return np.array(X), np.array(y)

def train_cnn_lstm(frames, seq_len=20, batch_size=2, epochs=5, save_path="models/cnn_lstm.h5"):
    X, y = create_sequences(frames, seq_len=seq_len)

    cnn = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    cnn.trainable = False

    inp = Input(shape=(seq_len, frames.shape[1], frames.shape[2], 3))
    x = TimeDistributed(cnn)(inp)
    x = LSTM(128, return_sequences=True)(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inp, out)
    model.compile(optimizer=Adam(), loss='binary_crossentropy')
    model.fit(X, y, batch_size=batch_size, epochs=epochs)
    model.save(save_path)
    print(f"[INFO] CNN+LSTM saved at {save_path}")
    return model
