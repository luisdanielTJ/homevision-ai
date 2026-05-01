import numpy as np
from unittest.mock import MagicMock, patch
from frame_publisher.mediapipe_extractor import (
    KeypointResult,
    _empty_result,
    DEFAULT_MODEL_PATH,
)


def _make_landmark(x=0.5, y=0.5, z=0.0):
    lm = MagicMock()
    lm.x = x
    lm.y = y
    lm.z = z
    return lm


def _make_extractor_with_mock_landmarker():
    """Build a MediaPipeExtractor whose internal landmarker is fully mocked."""
    from frame_publisher.mediapipe_extractor import MediaPipeExtractor

    with patch("frame_publisher.mediapipe_extractor.MediaPipeExtractor.__init__", return_value=None):
        extractor = MediaPipeExtractor.__new__(MediaPipeExtractor)
        extractor._landmarker = MagicMock()
        extractor._mp_image_cls = MagicMock(return_value=MagicMock())
        extractor._mp_format = MagicMock()
        extractor._prev_nose_x = None
        extractor._head_turns = 0
    return extractor


def test_empty_result_has_correct_defaults():
    result = _empty_result(head_turns=2)
    assert result.person_detected is False
    assert result.head_turn_count == 2
    assert result.object_raised is False


def test_keypoint_result_to_dict_contains_required_keys():
    result = KeypointResult(
        person_detected=True,
        head_turn_count=1,
        object_raised=False,
        nose_x=0.5,
        nose_y=0.3,
        left_wrist_y=0.6,
        right_wrist_y=0.6,
        left_shoulder_y=0.4,
        right_shoulder_y=0.4,
    )
    data = result.to_dict()
    assert "person_detected" in data
    assert "head_turn_count" in data
    assert "object_raised" in data
    assert "nose_x" in data


def test_extract_no_pose_returns_empty():
    extractor = _make_extractor_with_mock_landmarker()
    extractor._landmarker.detect.return_value = MagicMock(pose_landmarks=[])
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    from frame_publisher.mediapipe_extractor import MediaPipeExtractor
    result = MediaPipeExtractor.extract(extractor, frame)

    assert result.person_detected is False


def test_extract_with_pose_returns_person_detected():
    extractor = _make_extractor_with_mock_landmarker()
    # 33 landmarks (MediaPipe pose has 33)
    landmarks = [_make_landmark(x=0.5, y=0.5) for _ in range(33)]
    # wrist above shoulder: wrist.y < shoulder.y (smaller y = higher in image)
    landmarks[15].y = 0.3   # LEFT_WRIST above shoulder
    landmarks[11].y = 0.5   # LEFT_SHOULDER
    landmarks[16].y = 0.6   # RIGHT_WRIST below shoulder
    landmarks[12].y = 0.5   # RIGHT_SHOULDER
    mock_result = MagicMock(pose_landmarks=[landmarks])
    extractor._landmarker.detect.return_value = mock_result
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    from frame_publisher.mediapipe_extractor import MediaPipeExtractor
    result = MediaPipeExtractor.extract(extractor, frame)

    assert result.person_detected is True
    assert result.object_raised is True


def test_head_turn_increments_on_large_nose_delta():
    extractor = _make_extractor_with_mock_landmarker()
    extractor._prev_nose_x = 0.5

    landmarks = [_make_landmark(x=0.5, y=0.5) for _ in range(33)]
    landmarks[0].x = 0.65   # NOSE moved > 0.08 from 0.5
    mock_result = MagicMock(pose_landmarks=[landmarks])
    extractor._landmarker.detect.return_value = mock_result
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    from frame_publisher.mediapipe_extractor import MediaPipeExtractor
    result = MediaPipeExtractor.extract(extractor, frame)

    assert result.head_turn_count == 1


def test_reset_clears_state():
    extractor = _make_extractor_with_mock_landmarker()
    extractor._head_turns = 3
    extractor._prev_nose_x = 0.5

    from frame_publisher.mediapipe_extractor import MediaPipeExtractor
    MediaPipeExtractor.reset(extractor)

    assert extractor._head_turns == 0
    assert extractor._prev_nose_x is None
