import cv2
import os

def extract_frames(video_path, output_dir, fps=2):
    """
    Extract frames from a single video at a given FPS.
    Saves frames to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps else 0
    print(f"[INFO] Processing video: {video_path}")
    print(f"       FPS: {video_fps:.2f}, Total Frames: {total_frames}, Duration: {duration:.2f}s")

    frame_interval = max(int(video_fps // fps), 1)

    count = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            frame_resized = cv2.resize(frame, (224, 224))
            frame_file = os.path.join(output_dir, f"frame_{saved:04d}.jpg")
            cv2.imwrite(frame_file, frame_resized)
            print(f"[FRAME] Saved frame {saved} (Original Index: {count})")
            saved += 1

        count += 1

    cap.release()
    print(f"[DONE] Finished extracting {saved} frames from {video_path}\n")
    return saved


def extract_summe_dataset(summe_dir, output_root, fps=2):
    """
    Extract frames from all videos in the SumMe dataset.
    Saves frames in separate folders for each video.
    """
    os.makedirs(output_root, exist_ok=True)
    video_files = [f for f in os.listdir(summe_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]

    for video_file in video_files:
        video_path = os.path.join(summe_dir, video_file)
        video_name = os.path.splitext(video_file)[0]
        output_dir = os.path.join(output_root, video_name)

        if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
            print(f"[SKIP] Frames already exist for video: {video_file}")
            continue

        extract_frames(video_path, output_dir, fps=fps)

    print("[INFO] All videos processed.")

if __name__ == "__main__":
    SUMME_VIDEO_DIR = "../data/SumMe/videos"
    OUTPUT_FRAME_DIR = "../data/SumMe/frames"
    #extract_summe_dataset(SUMME_VIDEO_DIR, OUTPUT_FRAME_DIR,fps=2)

    TEST_VIDEO_DIR = "../data/test/video"
    TEST_OUTPUT_FRAME_DIR = "../data/test/frames"
    extract_summe_dataset(TEST_VIDEO_DIR, TEST_OUTPUT_FRAME_DIR,fps=2)