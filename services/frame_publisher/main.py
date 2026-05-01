import os
import time
import dotenv
from frame_publisher.webcam_capture import WebcamCapture
from frame_publisher.mediapipe_extractor import MediaPipeExtractor
from frame_publisher.publisher import FramePublisher

dotenv.load_dotenv()

PROJECT_ID = os.environ["GCP_PROJECT_ID"]
TOPIC_NAME = os.environ["PUBSUB_RAW_FRAMES_TOPIC"]
TARGET_FPS = float(os.getenv("FRAME_FPS", "2"))
CAMERA_ID = os.getenv("CAMERA_ID", "cam0")
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
FRAME_DELAY = 1.0 / TARGET_FPS


def main() -> None:
    cap = WebcamCapture(device_index=CAMERA_INDEX)
    extractor = MediaPipeExtractor()
    publisher = FramePublisher.from_env(PROJECT_ID, TOPIC_NAME)
    print(f"HomeVision frame publisher started at {TARGET_FPS} FPS → topic: {TOPIC_NAME}")
    try:
        while True:
            loop_start = time.time()
            frame = cap.read()
            if frame is None:
                continue
            result = extractor.extract(frame)
            publisher.publish(camera_id=CAMERA_ID, keypoints=result.to_dict(), timestamp=time.time())
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, FRAME_DELAY - elapsed))
    except KeyboardInterrupt:
        print("Shutting down.")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
