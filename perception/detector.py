from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import cv2
import yaml


@dataclass
class Detection:
    track_id: Optional[int]
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]       # [x1, y1, x2, y2]
    bbox_xywh: list[float]       # [cx, cy, w, h]
    area: float

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 3),
            "bbox": self.bbox_xyxy,
            "area": round(self.area, 1),
        }


@dataclass
class FrameResult:
    frame_id: int
    timestamp: float
    detections: list[Detection] = field(default_factory=list)
    raw_frame: Optional[np.ndarray] = None

    def summary(self) -> dict:
        class_counts = {}
        for d in self.detections:
            class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1
        return {
            "frame_id": self.frame_id,
            "timestamp": round(self.timestamp, 3),
            "total_objects": len(self.detections),
            "class_counts": class_counts,
            "detections": [d.to_dict() for d in self.detections],
        }


class YOLODetector:
    def __init__(self, config_path: str = "configs/yolo_config.yaml", config: dict | None = None):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        if config:
            for section, values in config.items():
                if isinstance(values, dict) and isinstance(cfg.get(section), dict):
                    cfg[section].update(values)

        self.model_cfg = cfg["model"]
        self.track_cfg = cfg["tracking"]
        self.class_filter = cfg["classes"].get("filter")

        try:
            self.model = YOLO(self.model_cfg["weights"])
            self.model.to(self.model_cfg["device"])
        except Exception as exc:
            raise RuntimeError(f"Unable to load YOLO weights '{self.model_cfg['weights']}': {exc}") from exc

        self._frame_id = 0
        self._class_names = self.model.names  # {0: 'person', 1: 'bicycle', ...}

    def detect(self, frame: np.ndarray, timestamp: float = 0.0) -> FrameResult:
        self._frame_id += 1

        kwargs = dict(
            source=frame,
            imgsz=self.model_cfg["imgsz"],
            conf=self.model_cfg["conf_threshold"],
            iou=self.model_cfg["iou_threshold"],
            device=self.model_cfg["device"],
            verbose=False,
            classes=self.class_filter,
        )

        try:
            if self.track_cfg["enabled"]:
                results = self.model.track(
                    persist=self.track_cfg["persist"],
                    tracker=self.track_cfg["tracker"],
                    **kwargs,
                )
            else:
                results = self.model(**kwargs)
        except Exception as exc:
            raise RuntimeError(f"YOLO inference failed on frame {self._frame_id}: {exc}") from exc

        detections = self._parse_results(results[0])
        return FrameResult(
            frame_id=self._frame_id,
            timestamp=timestamp,
            detections=detections,
            raw_frame=frame,
        )

    def _parse_results(self, result) -> list[Detection]:
        detections = []
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        xywh = boxes.xywh.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        track_ids = (
            boxes.id.cpu().numpy().astype(int)
            if boxes.id is not None
            else [None] * len(xyxy)
        )

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i].tolist()
            area = (x2 - x1) * (y2 - y1)
            detections.append(
                Detection(
                    track_id=track_ids[i],
                    class_id=int(cls_ids[i]),
                    class_name=self._class_names[int(cls_ids[i])],
                    confidence=float(confs[i]),
                    bbox_xyxy=[x1, y1, x2, y2],
                    bbox_xywh=xywh[i].tolist(),
                    area=area,
                )
            )

        return detections
