"""Export trained YOLOv8 weights to TorchScript for Vertex AI serving.

Run after training:
    WEIGHTS_PATH=/tmp/model/best.pt python ml/training/yolov8/export_model.py
"""
import os
from pathlib import Path
from ultralytics import YOLO

WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", "/tmp/model/best.pt")
EXPORT_DIR = os.environ.get("EXPORT_DIR", "/tmp/model/exported")


def export() -> None:
    model = YOLO(WEIGHTS_PATH)
    Path(EXPORT_DIR).mkdir(parents=True, exist_ok=True)
    path = model.export(format="torchscript", imgsz=640)
    print(f"Exported model to: {path}")


if __name__ == "__main__":
    export()
