from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam

def build_autoencoder(img_size=(224,224,3)):
    inp = Input(shape=img_size)
    x = Conv2D(64, 3, activation='relu', padding='same')(inp)
    x = MaxPooling2D(2, padding='same')(x)
    x = Conv2D(128,3, activation='relu', padding='same')(x)
    encoded = MaxPooling2D(2, padding='same')(x)

    x = Conv2D(128,3, activation='relu', padding='same')(encoded)
    x = UpSampling2D(2)(x)
    x = Conv2D(64,3, activation='relu', padding='same')(x)
    x = UpSampling2D(2)(x)
    out = Conv2D(3,3, activation='sigmoid', padding='same')(x)

    model = Model(inp, out)
    model.compile(optimizer=Adam(), loss='mse')
    return model
