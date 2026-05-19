import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from perception.detector import YOLODetector, Detection, FrameResult


@pytest.fixture
def mock_detector(tmp_path):
    cfg = """
model:
  weights: yolov8n.pt
  device: cpu
  imgsz: 640
  conf_threshold: 0.35
  iou_threshold: 0.45
tracking:
  enabled: false
  tracker: bytetrack.yaml
  persist: true
classes:
  filter: null
  names_override: null
stream:
  source: 0
  max_fps: 30
  resize_output: [640, 480]
"""
    cfg_file = tmp_path / "yolo_config.yaml"
    cfg_file.write_text(cfg)

    with patch("perception.detector.YOLO") as MockYOLO:
        mock_model = MagicMock()
        mock_model.names = {0: "person", 2: "car"}
        MockYOLO.return_value = mock_model
        yield YOLODetector(config_path=str(cfg_file)), mock_model


def test_frame_result_summary():
    det = Detection(
        track_id=1, class_id=0, class_name="person",
        confidence=0.9, bbox_xyxy=[10, 20, 50, 80],
        bbox_xywh=[30, 50, 40, 60], area=2400
    )
    result = FrameResult(frame_id=1, timestamp=1.0, detections=[det])
    summary = result.summary()
    assert summary["total_objects"] == 1
    assert summary["class_counts"]["person"] == 1


def test_detection_to_dict():
    det = Detection(
        track_id=5, class_id=2, class_name="car",
        confidence=0.75, bbox_xyxy=[0, 0, 100, 50],
        bbox_xywh=[50, 25, 100, 50], area=5000
    )
    d = det.to_dict()
    assert d["class_name"] == "car"
    assert d["track_id"] == 5
    assert d["confidence"] == 0.75