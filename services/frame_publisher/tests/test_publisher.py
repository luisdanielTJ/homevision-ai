import json
from unittest.mock import MagicMock
from frame_publisher.publisher import FramePublisher


def test_publish_sends_message_with_required_fields():
    mock_client = MagicMock()
    mock_future = MagicMock()
    mock_client.publish.return_value = mock_future

    publisher = FramePublisher(client=mock_client, topic_path="projects/p/topics/t")
    keypoints = {
        "person_detected": True,
        "head_turn_count": 1,
        "object_raised": False,
        "nose_x": 0.5,
        "nose_y": 0.3,
        "left_wrist_y": 0.6,
        "right_wrist_y": 0.6,
        "left_shoulder_y": 0.4,
        "right_shoulder_y": 0.4,
    }
    publisher.publish(camera_id="cam0", keypoints=keypoints, timestamp=1000.0)

    mock_client.publish.assert_called_once()
    call_args = mock_client.publish.call_args
    payload = json.loads(call_args[0][1].decode())
    assert payload["camera_id"] == "cam0"
    assert payload["timestamp"] == 1000.0
    assert "keypoints" in payload


def test_publish_skips_when_no_person_detected():
    mock_client = MagicMock()
    publisher = FramePublisher(client=mock_client, topic_path="projects/p/topics/t")
    keypoints = {
        "person_detected": False,
        "head_turn_count": 0,
        "object_raised": False,
        "nose_x": 0.0,
        "nose_y": 0.0,
        "left_wrist_y": 0.0,
        "right_wrist_y": 0.0,
        "left_shoulder_y": 0.0,
        "right_shoulder_y": 0.0,
    }
    publisher.publish(camera_id="cam0", keypoints=keypoints, timestamp=1000.0)
    mock_client.publish.assert_not_called()


def test_publish_waits_for_future_result():
    mock_client = MagicMock()
    mock_future = MagicMock()
    mock_client.publish.return_value = mock_future

    publisher = FramePublisher(client=mock_client, topic_path="projects/p/topics/t")
    keypoints = {"person_detected": True, "head_turn_count": 0, "object_raised": False,
                 "nose_x": 0.5, "nose_y": 0.3, "left_wrist_y": 0.6, "right_wrist_y": 0.6,
                 "left_shoulder_y": 0.4, "right_shoulder_y": 0.4}
    publisher.publish(camera_id="cam0", keypoints=keypoints, timestamp=1000.0)

    mock_future.result.assert_called_once_with(timeout=5)
