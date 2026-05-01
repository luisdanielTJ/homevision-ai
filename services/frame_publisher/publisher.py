import json
from google.cloud import pubsub_v1


class FramePublisher:
    def __init__(self, client: pubsub_v1.PublisherClient, topic_path: str) -> None:
        self._client = client
        self._topic_path = topic_path

    def publish(self, camera_id: str, keypoints: dict, timestamp: float) -> None:
        if not keypoints.get("person_detected", False):
            return
        payload = json.dumps({
            "camera_id": camera_id,
            "timestamp": timestamp,
            "keypoints": keypoints,
        }).encode()
        future = self._client.publish(self._topic_path, payload)
        future.result(timeout=5)

    @classmethod
    def from_env(cls, project_id: str, topic_name: str) -> "FramePublisher":
        client = pubsub_v1.PublisherClient()
        topic_path = client.topic_path(project_id, topic_name)
        return cls(client=client, topic_path=topic_path)
