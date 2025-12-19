from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TimeDistributed, LSTM, Dense
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam


def build_cnn_lstm(input_shape, lstm_units=128):
    """
    input_shape: (seq_len, H, W, 3)
    """
    # CNN backbone
    cnn = ResNet50(
        weights="imagenet",
        include_top=False,
        pooling="avg"
    )
    cnn.trainable = False

    inp = Input(shape=input_shape)
    x = TimeDistributed(cnn)(inp)
    x = LSTM(lstm_units, return_sequences=True)(x)
    out = Dense(1, activation="sigmoid")(x)

    model = Model(inp, out)
    model.compile(
        optimizer=Adam(1e-4),
        loss="binary_crossentropy"
    )

    return model
