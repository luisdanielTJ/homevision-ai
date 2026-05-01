import pytest
from behavior_engine.dwell_tracker import DwellTracker


def test_new_camera_starts_at_zero():
    tracker = DwellTracker()
    tracker.update("cam0", timestamp=0.0)
    assert tracker.dwell_time("cam0") == pytest.approx(0.0)


def test_dwell_increases_over_time():
    tracker = DwellTracker()
    tracker.update("cam0", timestamp=0.0)
    tracker.update("cam0", timestamp=10.0)
    assert tracker.dwell_time("cam0") == pytest.approx(10.0)


def test_reset_clears_dwell():
    tracker = DwellTracker()
    tracker.update("cam0", timestamp=0.0)
    tracker.update("cam0", timestamp=20.0)
    tracker.reset("cam0")
    assert tracker.dwell_time("cam0") == pytest.approx(0.0)


def test_unknown_camera_returns_zero():
    tracker = DwellTracker()
    assert tracker.dwell_time("unknown") == pytest.approx(0.0)


def test_multiple_cameras_are_independent():
    tracker = DwellTracker()
    tracker.update("cam0", timestamp=0.0)
    tracker.update("cam0", timestamp=5.0)
    tracker.update("cam1", timestamp=0.0)
    assert tracker.dwell_time("cam0") == pytest.approx(5.0)
    assert tracker.dwell_time("cam1") == pytest.approx(0.0)
