import numpy as np
from models.cnn_lstm import build_cnn_lstm

# Dummy placeholders for explanation
X_train = np.load("data/SumMe/X_train.npy")
y_train = np.load("data/SumMe/y_train.npy")

model = build_cnn_lstm()
model.fit(X_train, y_train, epochs=15, batch_size=2)

model.save("models/cnn_lstm.h5")
