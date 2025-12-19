## Project Structure
```
KeyFrameDetection/
│
├── data/
│   ├── SumMe/
│   │   ├── videos/
│   │   └── frames/
│   └── test_video/
│       └── test.mp4
│
├── preprocessing/
│   ├── extract_frames.py
│   └── dataset_loader.py
│
├── models/
│   ├── autoencoder.py
│   └── cnn_lstm.py
│
├── training/
│   ├── train_autoencoder.py
│   └── train_cnn_lstm.py
│
├── inference/
│   ├── detect_keyframes_ae.py
│   └── detect_keyframes_lstm.py
│
├── evaluation/
│   └── metrics.py
│
├── diagrams/
│   └── block_diagrams.txt
│
└── README.md

```
## Journey
“Scene boundaries are detected by large temporal changes in autoencoder reconstruction error, rather than absolute error values. Each scene is represented by the middle frame.”

“Scene boundaries were detected using changes in autoencoder reconstruction error. Frames were grouped into scenes, and a single representative frame with maximum reconstruction error was selected per scene.”

“To avoid over-segmentation caused by transient reconstruction error spikes, short adjacent scenes were merged using a minimum temporal duration constraint.”

“Color differences were initially observed due to unnecessary color space conversion during frame saving. Since frames were loaded and processed in BGR format, removing the redundant RGB→BGR conversion preserved the original colors.”

## Data Preprocessing Pipeline
Step-by-step:

Extract frames using OpenCV

Resize frames → 224 × 224

Normalize pixels to [0,1]

Sample frames (e.g., 2 FPS to reduce redundancy)

Store frames as sequences

Video → Frames → Resize → Normalize → Frame Sequences

## Architecture (Block Diagram Description)
Input Frame
   ↓
CNN Encoder
   ↓
Latent Representation
   ↓
CNN Decoder
   ↓
Reconstructed Frame


# Key Frame Detection Using Deep Learning Techniques

## 1. Introduction

Key frame detection aims to select a small subset of frames that best represent the visual and semantic content of a video. This project focuses on detecting key frames from a **30-second colored video** using **two deep learning techniques** and comparing their outputs.

The motivation behind key frame detection includes:

* Video summarization
* Efficient storage and retrieval
* Fast browsing of video content

---

## 2. Problem Analysis

A 30-second video at 30 FPS contains around **900 frames**, many of which are redundant. The main challenges are:

* **Redundancy**: Consecutive frames are often visually similar
* **Subjective importance**: What defines a “key” frame varies
* **Temporal continuity**: Important events occur over time, not in isolation
* **High dimensionality**: Video data is large and complex

To address these challenges, we explore:

1. An **unsupervised approach** (Autoencoder)
2. A **temporal supervised approach** (CNN + LSTM)

---

## 3. Dataset Description

### Training Dataset: SumMe

* Public benchmark dataset for video summarization
* Contains short videos (1–6 minutes)
* Includes human-annotated key frames and importance scores

### Testing Data

* A 30-second colored video downloaded from the internet
* Used **only for inference**, not training

---

## 4. Model 1: CNN Autoencoder (Unsupervised)

### Architecture Description

The autoencoder learns to reconstruct input frames. Frames that are difficult to reconstruct indicate **scene changes** or **novel content**.

**Block Diagram (Textual)**:

```
Input Frame → CNN Encoder → Latent Space → CNN Decoder → Reconstructed Frame
```

### Key Frame Detection Logic

1. Compute reconstruction error for each frame
2. Plot error over time
3. Frames with high reconstruction error are selected as key frames

### Strengths

* No labels required
* Simple and fast

### Weaknesses

* No temporal modeling
* Limited semantic understanding

---

## 5. Model 2: CNN + LSTM (Temporal Model)

### Architecture Description

This model combines spatial feature extraction with temporal sequence modeling.

**Block Diagram (Textual)**:

```
Frames → CNN → Feature Sequence → LSTM → Importance Score → Key Frame
```

### Key Frame Detection Logic

1. Predict importance score for each frame
2. Apply threshold or select top-K frames

### Strengths

* Captures temporal dependencies
* Better semantic relevance

### Weaknesses

* Requires labeled data
* Higher computational cost

---

## 6. Visualization Code

### 6.1 Visualizing Key Frames

```python
import matplotlib.pyplot as plt
import cv2
import os

frames_dir = "data/test_video/frames"
keyframes = [10, 45, 90, 140]  # example indices

plt.figure(figsize=(15, 4))
for i, idx in enumerate(keyframes):
    img = cv2.imread(os.path.join(frames_dir, f"frame_{idx:04d}.jpg"))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(1, len(keyframes), i+1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Frame {idx}")

plt.show()
```

---

### 6.2 Reconstruction Error Curve (Autoencoder)

```python
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10,4))
plt.plot(errors)
plt.axhline(threshold, color='r', linestyle='--')
plt.xlabel("Frame Index")
plt.ylabel("Reconstruction Error")
plt.title("Autoencoder Reconstruction Error Over Time")
plt.show()
```

---

### 6.3 Importance Scores Over Time (CNN + LSTM)

```python
plt.figure(figsize=(10,4))
plt.plot(scores)
plt.axhline(0.5, color='r', linestyle='--')
plt.xlabel("Frame Index")
plt.ylabel("Importance Score")
plt.title("CNN + LSTM Frame Importance Scores")
plt.show()
```

---

## 7. Evaluation Metrics

We evaluate the models using:

* Precision
* Recall
* F1-score

The CNN + LSTM model consistently achieves higher F1-score due to its temporal awareness.

---

## 8. Comparison Between Models

| Aspect                 | Autoencoder  | CNN + LSTM |
| ---------------------- | ------------ | ---------- |
| Learning Type          | Unsupervised | Supervised |
| Temporal Modeling      | No           | Yes        |
| Semantic Understanding | Low          | High       |
| Output Quality         | Moderate     | High       |

---

## 9. Discussion

The autoencoder is effective at detecting **scene changes**, but often misses semantically important frames. The CNN + LSTM model provides more **human-like summaries**, selecting frames that align better with human annotations.

---

## 10. Conclusion

This project demonstrates that:

* Unsupervised methods are useful baselines
* Temporal deep learning models significantly improve key frame detection
* Combining both approaches provides strong analytical insight

Future work may include attention mechanisms or transformer-based video models.

---

## 11. References

* SumMe Dataset
* Keras & TensorFlow Documentation
* Video Summarization Research Papers
