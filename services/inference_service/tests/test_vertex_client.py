from inference_service.vertex_client import VertexInferenceClient, InferenceResult


def _mock_client() -> VertexInferenceClient:
    return VertexInferenceClient(
        project_id="test",
        region="us-central1",
        automl_endpoint_id="mock",
        yolov8_endpoint_id="mock",
        mock=True,
    )


def test_inference_result_to_dict_has_required_keys():
    result = InferenceResult(
        model_name="automl",
        person_detected=True,
        confidence=0.92,
        bbox=[0.1, 0.2, 0.8, 0.9],
        latency_ms=45.0,
    )
    d = result.to_dict()
    assert d["model_name"] == "automl"
    assert d["person_detected"] is True
    assert d["confidence"] == 0.92
    assert "bbox" in d
    assert "latency_ms" in d


def test_mock_automl_returns_detected_when_person_present():
    client = _mock_client()
    result = client.predict_automl({"person_detected": True})
    assert result.person_detected is True
    assert result.model_name == "automl"
    assert result.confidence > 0


def test_mock_automl_returns_not_detected_when_no_person():
    client = _mock_client()
    result = client.predict_automl({"person_detected": False})
    assert result.person_detected is False
    assert result.confidence == 0.0


def test_mock_yolov8_returns_detected_when_person_present():
    client = _mock_client()
    result = client.predict_yolov8({"person_detected": True})
    assert result.person_detected is True
    assert result.model_name == "yolov8"


def test_mock_yolov8_higher_confidence_than_automl():
    client = _mock_client()
    automl = client.predict_automl({"person_detected": True})
    yolov8 = client.predict_yolov8({"person_detected": True})
    assert yolov8.confidence > automl.confidence
