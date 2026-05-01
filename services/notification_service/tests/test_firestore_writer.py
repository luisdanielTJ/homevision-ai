from unittest.mock import MagicMock
from notification_service.firestore_writer import FirestoreWriter


def test_write_calls_firestore_collection_add():
    mock_db = MagicMock()
    mock_collection = MagicMock()
    mock_db.collection.return_value = mock_collection

    writer = FirestoreWriter(db=mock_db)
    alert = {"camera_id": "cam0", "timestamp": 1000.0, "dwell_time": 17.0, "alert_text": "Help!"}
    writer.write(alert)

    mock_db.collection.assert_called_once_with("alerts")
    mock_collection.add.assert_called_once_with(alert)


def test_mock_writer_write_does_not_raise():
    writer = FirestoreWriter(mock=True)
    writer.write({"camera_id": "cam0", "timestamp": 1.0, "dwell_time": 15.0, "alert_text": "test"})


def test_mock_writer_get_recent_returns_empty_list():
    writer = FirestoreWriter(mock=True)
    assert writer.get_recent() == []
