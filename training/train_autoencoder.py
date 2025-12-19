from models.autoencoder import build_autoencoder

def train_autoencoder(frames, img_size=(224,224), batch_size=16, epochs=10, save_path="models/autoencoder.h5"):
    model = build_autoencoder(img_size + (3,))
    model.fit(frames, frames, batch_size=batch_size, epochs=epochs, shuffle=True)
    model.save(save_path)
    print(f"[INFO] Autoencoder saved at {save_path}")
    return model
