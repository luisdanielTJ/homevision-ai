from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Protocol
import numpy as np


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "pose_landmarker_lite.task"


@dataclass
class KeypointResult:
    person_detected: bool
    head_turn_count: int
    object_raised: bool
    nose_x: float
    nose_y: float
    left_wrist_y: float
    right_wrist_y: float
    left_shoulder_y: float
    right_shoulder_y: float

    def to_dict(self) -> dict:
        return asdict(self)


def _empty_result(head_turns: int = 0) -> KeypointResult:
    return KeypointResult(
        person_detected=False,
        head_turn_count=head_turns,
        object_raised=False,
        nose_x=0.0,
        nose_y=0.0,
        left_wrist_y=0.0,
        right_wrist_y=0.0,
        left_shoulder_y=0.0,
        right_shoulder_y=0.0,
    )


class MediaPipeExtractor:
    def __init__(self, model_path: Path | None = None) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
        from mediapipe.tasks.python import BaseOptions

        resolved = model_path or DEFAULT_MODEL_PATH
        if not resolved.exists():
            raise FileNotFoundError(
                f"Pose model not found at {resolved}. Run:\n"
                f"  python services/frame_publisher/download_model.py"
            )

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(resolved)),
            num_poses=1,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._mp_image_cls = mp.Image
        self._mp_format = mp.ImageFormat.SRGB
        self._prev_nose_x: float | None = None
        self._head_turns: int = 0

    def extract(self, frame: np.ndarray) -> KeypointResult:
        rgb = frame[:, :, ::-1].copy()  # BGR → RGB
        mp_image = self._mp_image_cls(image_format=self._mp_format, data=rgb)
        result = self._landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return _empty_result(self._head_turns)

        lm = result.pose_landmarks[0]
        # MediaPipe Tasks landmarks are plain objects with x/y/z/visibility
        NOSE = 0
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_WRIST = 15
        RIGHT_WRIST = 16

        nose_x = lm[NOSE].x
        if self._prev_nose_x is not None and abs(nose_x - self._prev_nose_x) > 0.08:
            self._head_turns += 1
        self._prev_nose_x = nose_x

        left_wrist_y = lm[LEFT_WRIST].y
        right_wrist_y = lm[RIGHT_WRIST].y
        left_shoulder_y = lm[LEFT_SHOULDER].y
        right_shoulder_y = lm[RIGHT_SHOULDER].y
        object_raised = (left_wrist_y < left_shoulder_y) or (right_wrist_y < right_shoulder_y)

        return KeypointResult(
            person_detected=True,
            head_turn_count=self._head_turns,
            object_raised=object_raised,
            nose_x=nose_x,
            nose_y=lm[NOSE].y,
            left_wrist_y=left_wrist_y,
            right_wrist_y=right_wrist_y,
            left_shoulder_y=left_shoulder_y,
            right_shoulder_y=right_shoulder_y,
        )

    def reset(self) -> None:
        self._prev_nose_x = None
        self._head_turns = 0
