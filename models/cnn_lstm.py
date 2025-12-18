from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, TimeDistributed

def build_cnn_lstm():
    """
    CNN + LSTM model for frame importance prediction.
    """
    cnn = ResNet50(weights="imagenet", include_top=False, pooling="avg")

    for layer in cnn.layers:
        layer.trainable = False

    input_seq = Input(shape=(None, 224, 224, 3))
    features = TimeDistributed(cnn)(input_seq)
    x = LSTM(128, return_sequences=True)(features)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(input_seq, outputs)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
