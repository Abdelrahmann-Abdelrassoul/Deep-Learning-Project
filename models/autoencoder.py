from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam


def build_autoencoder(input_shape=(224, 224, 3)):
    inp = Input(shape=input_shape)

    x = Conv2D(64, 3, activation="relu", padding="same")(inp)
    x = MaxPooling2D(2, padding="same")(x)

    x = Conv2D(128, 3, activation="relu", padding="same")(x)
    encoded = MaxPooling2D(2, padding="same")(x)

    x = Conv2D(128, 3, activation="relu", padding="same")(encoded)
    x = UpSampling2D(2)(x)

    x = Conv2D(64, 3, activation="relu", padding="same")(x)
    x = UpSampling2D(2)(x)

    out = Conv2D(3, 3, activation="sigmoid", padding="same")(x)

    model = Model(inp, out)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="mae"
    )

    return model
