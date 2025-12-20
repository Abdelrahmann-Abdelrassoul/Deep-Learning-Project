import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from preprocessing.dataset_loader import load_frames_dataset


def detect_keyframes_cnn_lstm(
    frames_dir,
    model_path,
    img_size=(224,224),
    seq_len=20,
    diff_percentile=90,
    min_scene_len=5,
    visualize=True,
    save_keyframes=True,
    output_dir="results/cnn_lstm_keyframes"
):
    """
    Scene-based keyframe detection using CNN+LSTM importance scores.
    AE-equivalent logic (diff-based).
    Returns and saves ORIGINAL frames.
    """

    # -----------------------------
    # Load frames
    # -----------------------------
    print("[INFO] Loading frames...")
    frames = load_frames_dataset(frames_dir, img_size=img_size)

    # -----------------------------
    # Load CNN+LSTM
    # -----------------------------
    print("[INFO] Loading CNN+LSTM...")
    model = load_model(model_path, compile=False)

    # -----------------------------
    # Build sequences
    # -----------------------------
    print("[INFO] Building sequences...")
    sequences = np.array([
        frames[i:i + seq_len]
        for i in range(len(frames) - seq_len)
    ])

    # -----------------------------
    # Predict importance scores
    # -----------------------------
    print("[INFO] Predicting importance scores...")
    preds = model.predict(sequences, verbose=0)

    # -----------------------------
    # Aggregate to per-frame score
    # -----------------------------
    scores = np.zeros(len(frames))
    counts = np.zeros(len(frames))

    for i in range(len(preds)):
        for t in range(seq_len):
            idx = i + t
            scores[idx] += preds[i, t, 0]
            counts[idx] += 1

    scores /= np.maximum(counts, 1)

    # -----------------------------
    # 1. Frame-to-frame score difference (AE STYLE)
    # -----------------------------
    diffs = np.abs(np.diff(scores))
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
        idx = s + np.argmax(scores[s:e])
        keyframes.append(idx)

    keyframes = np.array(keyframes)
    print(f"[RESULT] Returned {len(keyframes)} keyframes (1 per scene)")

    # -----------------------------
    # Save ORIGINAL keyframes
    # -----------------------------
    if save_keyframes:
        os.makedirs(output_dir, exist_ok=True)
        for idx in keyframes:
            frame = (frames[idx] * 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(output_dir, f"scene_{idx:05d}.jpg"),
                frame
            )

    print(f"[INFO] Saved keyframes to {output_dir}")

    # -----------------------------
    # Visualization
    # -----------------------------
    if visualize:
        plt.figure(figsize=(12, 4))
        plt.plot(scores, label="CNN+LSTM Importance Score")
        plt.scatter(keyframes, scores[keyframes], c="red", label="Scene Keyframes")
        plt.xlabel("Frame Index")
        plt.ylabel("Importance Score")
        plt.title("AE-equivalent CNN+LSTM Keyframe Detection")
        plt.legend()
        plt.tight_layout()

        vis_path = os.path.join(output_dir, "importance_score_plot.png")
        plt.savefig(vis_path, dpi=200)
        print(f"[INFO] Visualization saved to {vis_path}")

        plt.show()

    return keyframes, scores
