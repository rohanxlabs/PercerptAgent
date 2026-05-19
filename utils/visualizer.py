import cv2
import numpy as np
from perception.detector import FrameResult

COLOR_MAP = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "yellow": (0, 255, 255),
    "blue": (255, 100, 0),
    "white": (255, 255, 255),
}

CLASS_COLORS = {}  # auto-assigned per class


def _class_color(class_name: str) -> tuple:
    if class_name not in CLASS_COLORS:
        h = abs(hash(class_name)) % 180
        # HSV → BGR
        color_hsv = np.uint8([[[h, 200, 220]]])
        bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        CLASS_COLORS[class_name] = tuple(int(c) for c in bgr)
    return CLASS_COLORS[class_name]


def draw_detections(frame: np.ndarray, frame_result: FrameResult) -> np.ndarray:
    out = frame.copy()
    for det in frame_result.detections:
        x1, y1, x2, y2 = (int(v) for v in det.bbox_xyxy)
        color = _class_color(det.class_name)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"{det.class_name}"
        if det.track_id is not None:
            label += f" #{det.track_id}"
        label += f" {det.confidence:.2f}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def draw_agent_annotations(frame: np.ndarray, annotations: list[dict]) -> np.ndarray:
    out = frame.copy()
    y_offset = 30
    for ann in annotations:
        color = COLOR_MAP.get(ann.get("color", "white"), (255, 255, 255))
        label = ann.get("label", "")
        cv2.putText(out, f"[AGENT] {label}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        y_offset += 28
    return out


def draw_hud(frame: np.ndarray, scene_snapshot: dict, agent_result: dict = None) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    overlay = out.copy()
    cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, out, 0.5, 0, out)

    counts = scene_snapshot.get("class_counts", {})
    count_str = "  ".join(f"{k}:{v}" for k, v in counts.items())
    cv2.putText(out, f"Objects: {count_str}", (10, h - 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    if agent_result:
        decision = agent_result.get("decision", "")[:80]
        cv2.putText(out, f"Agent: {decision}", (10, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)

    return out