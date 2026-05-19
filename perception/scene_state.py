from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional
import time
from perception.detector import Detection, FrameResult


@dataclass
class TrackedObject:
    track_id: int
    class_name: str
    first_seen: float
    last_seen: float
    detection_count: int = 1
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=30))
    confidence_avg: float = 0.0

    @property
    def duration(self) -> float:
        return self.last_seen - self.first_seen

    def update(self, det: Detection):
        self.last_seen = time.time()
        self.detection_count += 1
        self.bbox_history.append(det.bbox_xyxy)
        self.confidence_avg = (
            self.confidence_avg * (self.detection_count - 1) + det.confidence
        ) / self.detection_count

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "duration_sec": round(self.duration, 2),
            "detection_count": self.detection_count,
            "confidence_avg": round(self.confidence_avg, 3),
            "last_bbox": list(self.bbox_history[-1]) if self.bbox_history else [],
        }


class SceneState:
    """
    Maintains live scene context across frames:
    - Active tracked objects
    - Per-class presence history
    - Event log (entries/exits/changes)
    """

    def __init__(self, stale_timeout: float = 3.0):
        self.stale_timeout = stale_timeout  # seconds before object considered gone
        self._tracked: dict[int, TrackedObject] = {}
        self._class_history: dict[str, list[float]] = defaultdict(list)
        self._events: deque = deque(maxlen=100)
        self._frame_count = 0

    def update(self, frame_result: FrameResult):
        self._frame_count += 1
        now = time.time()
        seen_ids = set()

        for det in frame_result.detections:
            tid = det.track_id if det.track_id is not None else id(det)
            seen_ids.add(tid)

            if tid not in self._tracked:
                obj = TrackedObject(
                    track_id=tid,
                    class_name=det.class_name,
                    first_seen=now,
                    last_seen=now,
                    confidence_avg=det.confidence,
                )
                obj.bbox_history.append(det.bbox_xyxy)
                self._tracked[tid] = obj
                self._log_event("ENTER", det.class_name, tid, frame_result.frame_id)
            else:
                self._tracked[tid].update(det)

            self._class_history[det.class_name].append(now)

        # mark stale objects as exited
        stale = [
            tid
            for tid, obj in self._tracked.items()
            if tid not in seen_ids and (now - obj.last_seen) > self.stale_timeout
        ]
        for tid in stale:
            obj = self._tracked.pop(tid)
            self._log_event("EXIT", obj.class_name, tid, frame_result.frame_id)

    def _log_event(self, event_type: str, class_name: str, track_id: int, frame_id: int):
        self._events.append({
            "event": event_type,
            "class": class_name,
            "track_id": track_id,
            "frame_id": frame_id,
            "timestamp": time.time(),
        })

    def get_snapshot(self) -> dict:
        """Compact scene description for the agent."""
        active = [obj.to_dict() for obj in self._tracked.values()]
        class_counts = defaultdict(int)
        for obj in self._tracked.values():
            class_counts[obj.class_name] += 1

        return {
            "frame_count": self._frame_count,
            "active_objects": active,
            "class_counts": dict(class_counts),
            "recent_events": list(self._events)[-10:],
        }

    def get_object_history(self, class_name: str) -> list[dict]:
        return [
            obj.to_dict()
            for obj in self._tracked.values()
            if obj.class_name == class_name
        ]