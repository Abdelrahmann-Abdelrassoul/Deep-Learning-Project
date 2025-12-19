import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from preprocessing.dataset_loader import load_frames_dataset


def detect_keyframes_autoencoder(
    frames_dir,
    model_path,
    img_size=(224,224),
    threshold_mode="mean_std",
    percentile=90,
    visualize=True,
    save_keyframes=True,
    output_dir="results/autoencoder_keyframes"
):
    """
    Detect keyframes using Autoencoder reconstruction error
    """

    print("[INFO] Loading frames for inference...")
    frames = load_frames_dataset(frames_dir, img_size=img_size)

    print("[INFO] Loading autoencoder model (compile=False)...")
    model = load_model(model_path, compile=False)

    # Recompile ONLY for safety (not strictly needed for inference)
    model.compile(optimizer=Adam(), loss="mse")

    print("[INFO] Running reconstruction...")
    reconstructed = model.predict(frames, verbose=0)

    errors = np.mean((frames - reconstructed) ** 2, axis=(1,2,3))

    # -----------------------------
    # Thresholding
    # -----------------------------
    if threshold_mode == "mean_std":
        threshold = np.mean(errors) + np.std(errors)
    else:
        threshold = np.percentile(errors, percentile)

    keyframes = np.where(errors > threshold)[0]

    print(f"[RESULT] Detected {len(keyframes)} keyframes (Autoencoder)")

    # -----------------------------
    # Save keyframe images
    # -----------------------------
    if save_keyframes:
        os.makedirs(output_dir, exist_ok=True)
        print(f"[INFO] Saving keyframes to {output_dir}")

        for idx in keyframes:
            frame = (frames[idx] * 255).astype(np.uint8)
            save_path = os.path.join(output_dir, f"keyframe_{idx:05d}.jpg")
            cv2.imwrite(save_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    # -----------------------------
    # Visualization
    # -----------------------------
    if visualize:
        plt.figure(figsize=(10,4))
        plt.plot(errors, label="Reconstruction Error")
        plt.scatter(keyframes, errors[keyframes], color="red", label="Keyframes")
        plt.xlabel("Frame Index")
        plt.ylabel("MSE")
        plt.title("Autoencoder Reconstruction Error")
        plt.legend()
        plt.tight_layout()
        plt.show()

        n_show = min(5, len(keyframes))
        if n_show > 0:
            plt.figure(figsize=(15,3))
            for i in range(n_show):
                idx = keyframes[i]
                plt.subplot(1, n_show, i+1)
                plt.imshow(frames[idx])
                plt.title(f"Frame {idx}")
                plt.axis("off")
            plt.suptitle("Detected Keyframes (Autoencoder)")
            plt.show()

    return keyframes, errors
