import time
from dataclasses import dataclass
from google.cloud import aiplatform


@dataclass
class InferenceResult:
    model_name: str
    person_detected: bool
    confidence: float
    bbox: list[float]
    latency_ms: float

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "person_detected": self.person_detected,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "latency_ms": self.latency_ms,
        }


class VertexInferenceClient:
    def __init__(
        self,
        project_id: str,
        region: str,
        automl_endpoint_id: str,
        yolov8_endpoint_id: str,
        mock: bool = False,
    ) -> None:
        self._mock = mock
        if not mock:
            aiplatform.init(project=project_id, location=region)
            self._automl_ep = aiplatform.Endpoint(automl_endpoint_id)
            self._yolov8_ep = aiplatform.Endpoint(yolov8_endpoint_id)

    def predict_automl(self, keypoints: dict) -> InferenceResult:
        if self._mock:
            detected = keypoints.get("person_detected", False)
            return InferenceResult(
                model_name="automl",
                person_detected=detected,
                confidence=0.91 if detected else 0.0,
                bbox=[0.1, 0.1, 0.9, 0.9] if detected else [],
                latency_ms=12.0,
            )
        t0 = time.perf_counter()
        resp = self._automl_ep.predict(instances=[keypoints])
        latency = (time.perf_counter() - t0) * 1000
        pred = resp.predictions[0]
        return InferenceResult(
            model_name="automl",
            person_detected=pred.get("person_detected", False),
            confidence=float(pred.get("confidence", 0.0)),
            bbox=pred.get("bbox", []),
            latency_ms=latency,
        )

    def predict_yolov8(self, keypoints: dict) -> InferenceResult:
        if self._mock:
            detected = keypoints.get("person_detected", False)
            return InferenceResult(
                model_name="yolov8",
                person_detected=detected,
                confidence=0.95 if detected else 0.0,
                bbox=[0.12, 0.08, 0.88, 0.95] if detected else [],
                latency_ms=8.0,
            )
        t0 = time.perf_counter()
        resp = self._yolov8_ep.predict(instances=[keypoints])
        latency = (time.perf_counter() - t0) * 1000
        pred = resp.predictions[0]
        return InferenceResult(
            model_name="yolov8",
            person_detected=pred.get("person_detected", False),
            confidence=float(pred.get("confidence", 0.0)),
            bbox=pred.get("bbox", []),
            latency_ms=latency,
        )
