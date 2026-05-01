"""Capture labeled webcam frames for custom fine-tuning data.

Run:
    python ml/data/capture_webcam_frames.py

Press SPACE to save a frame, Q to quit.
Then annotate saved frames in Label Studio (https://labelstud.io/).
"""
import time
import cv2
from pathlib import Path

OUTPUT_DIR = Path("ml/data/raw/custom_frames")
TARGET_FRAMES = 100


def capture() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    count = 0
    print(f"Capturing up to {TARGET_FRAMES} frames.")
    print("Press SPACE to save a frame, Q to quit.")
    while count < TARGET_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Capture (SPACE=save, Q=quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            path = OUTPUT_DIR / f"frame_{count:04d}_{int(time.time())}.jpg"
            cv2.imwrite(str(path), frame)
            count += 1
            print(f"  Saved {path} ({count}/{TARGET_FRAMES})")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nCaptured {count} frames to {OUTPUT_DIR}")
    print("Next: annotate in Label Studio, export as COCO JSON, then run prepare_dataset.py")


if __name__ == "__main__":
    capture()
