import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from preprocessing.dataset_loader import load_frames_dataset


def detect_keyframes_autoencoder(
    frames_dir,
    model_path,
    img_size=(224,224),
    diff_percentile=90,
    min_scene_len=5,          # prevents micro-scenes
    visualize=True,
    save_keyframes=True,
    output_dir="results/autoencoder_keyframes"
):
    """
    Scene-based keyframe detection using Autoencoder reconstruction error.
    Returns and saves ORIGINAL frames (not reconstructed).
    """

    # -----------------------------
    # Load frames (ORIGINAL)
    # -----------------------------
    print("[INFO] Loading frames...")
    frames = load_frames_dataset(frames_dir, img_size=img_size)

    # -----------------------------
    # Load Autoencoder
    # -----------------------------
    print("[INFO] Loading AE...")
    model = load_model(model_path, compile=False)

    # -----------------------------
    # Reconstruction (for error only)
    # -----------------------------
    print("[INFO] Reconstructing...")
    recon = model.predict(frames, verbose=0)
    errors = np.mean((frames - recon) ** 2, axis=(1, 2, 3))

    # -----------------------------
    # 1. Frame-to-frame error difference
    # -----------------------------
    diffs = np.abs(np.diff(errors))
    threshold = np.percentile(diffs, diff_percentile)
    scene_boundaries = np.where(diffs > threshold)[0] + 1

    # -----------------------------
    # 2. Build scene segments
    # -----------------------------
    scenes = []
    start = 0

    for cut in scene_boundaries:
        if cut - start >= min_scene_len:
            scenes.append((start, cut))
        start = cut

    if len(frames) - start >= min_scene_len:
        scenes.append((start, len(frames)))

    print(f"[INFO] Detected {len(scenes)} scenes")

    # -----------------------------
    # 3. One keyframe per scene
    # -----------------------------
    keyframes = []
    for (s, e) in scenes:
        idx = s + np.argmax(errors[s:e])   # index ONLY
        keyframes.append(idx)

    keyframes = np.array(keyframes)
    print(f"[RESULT] Returned {len(keyframes)} keyframes (1 per scene)")

    # -----------------------------
    # Save ORIGINAL keyframe images
    # -----------------------------
    if save_keyframes:
        os.makedirs(output_dir, exist_ok=True)

        for idx in keyframes:
            # Convert normalized frame back to uint8
            original_frame = (frames[idx] * 255).astype(np.uint8)

            # Save directly, no color conversion needed
            cv2.imwrite(
                os.path.join(output_dir, f"scene_{idx:05d}.jpg"),
                original_frame
            )
    print(f"[INFO] Saved keyframes to {output_dir}")

    # -----------------------------
    # Visualization (SAVE + SHOW)
    # -----------------------------
    if visualize:
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(12, 4))
        plt.plot(errors, label="Reconstruction Error")
        plt.scatter(keyframes, errors[keyframes], c="red", label="Scene Keyframes")
        plt.xlabel("Frame Index")
        plt.ylabel("Reconstruction Error")
        plt.title("Scene-based AE Keyframe Detection")
        plt.legend()
        plt.tight_layout()

        # SAVE visualization
        vis_path = os.path.join(output_dir, "reconstruction_error_plot.png")
        plt.savefig(vis_path, dpi=200)
        print(f"[INFO] Visualization saved to {vis_path}")

        plt.show()

    return keyframes, errors
