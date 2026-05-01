from dataclasses import dataclass


@dataclass
class _CameraState:
    start_ts: float
    last_ts: float


class DwellTracker:
    def __init__(self) -> None:
        self._state: dict[str, _CameraState] = {}

    def update(self, camera_id: str, timestamp: float) -> None:
        if camera_id not in self._state:
            self._state[camera_id] = _CameraState(start_ts=timestamp, last_ts=timestamp)
        else:
            self._state[camera_id].last_ts = timestamp

    def dwell_time(self, camera_id: str) -> float:
        if camera_id not in self._state:
            return 0.0
        s = self._state[camera_id]
        return s.last_ts - s.start_ts

    def reset(self, camera_id: str) -> None:
        self._state.pop(camera_id, None)
