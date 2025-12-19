import os
import cv2
import numpy as np

def load_frames_dataset(frames_root, img_size=(224,224)):
    """
    Load all frames from the frames_root folder.
    Returns a NumPy array of shape (num_frames, H, W, 3)
    """
    frames_list = []
    for video_folder in os.listdir(frames_root):
        video_path = os.path.join(frames_root, video_folder)
        if not os.path.isdir(video_path):
            continue
        for frame_file in sorted(os.listdir(video_path)):
            if frame_file.lower().endswith('.jpg'):
                frame_path = os.path.join(video_path, frame_file)
                img = cv2.imread(frame_path)
                img = cv2.resize(img, img_size)
                img = img / 255.0
                frames_list.append(img)
    return np.array(frames_list)
