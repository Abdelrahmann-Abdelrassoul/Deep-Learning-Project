from preprocessing.dataset_loader import load_frames
from models.autoencoder import build_autoencoder

frames = load_frames("data/SumMe/frames")

model = build_autoencoder()
model.fit(
    frames,
    frames,
    epochs=20,
    batch_size=16,
    shuffle=True
)

model.save("models/autoencoder.h5")
