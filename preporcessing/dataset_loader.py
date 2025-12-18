import os
import numpy as np
import cv2

def load_frames(folder):
    """
    Load frames from folder into numpy array.
    """
    frames = []
    for file in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, file))
        img = img / 255.0
        frames.append(img)
    return np.array(frames)
