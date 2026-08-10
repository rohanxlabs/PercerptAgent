import cv2
import time
import threading
from queue import Queue, Empty
from typing import Optional
import yaml


class StreamHandler:
    """
    Reads frames from camera or video file.
    Runs capture in a background thread; main thread pulls from queue.
    """

    def __init__(self, config_path: str = "configs/yolo_config.yaml", source=None):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)["stream"]

        self.source = cfg["source"] if source is None else source
        self.max_fps = cfg["max_fps"]
        self.resize = tuple(cfg.get("resize_output", []))

        self._cap: Optional[cv2.VideoCapture] = None
        self._queue: Queue = Queue(maxsize=4)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._finished = threading.Event()

    def start(self):
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open source: {self.source}")
        self._stop_event.clear()
        self._finished.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap:
            self._cap.release()

    def _capture_loop(self):
        interval = 1.0 / self.max_fps
        while not self._stop_event.is_set():
            t0 = time.time()
            ret, frame = self._cap.read()
            if not ret:
                break
            if self.resize:
                frame = cv2.resize(frame, self.resize)
            if not self._queue.full():
                self._queue.put((frame, time.time()))
            elapsed = time.time() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)
        self._finished.set()

    def read(self, timeout: float = 1.0) -> Optional[tuple]:
        """Returns (frame, timestamp) or None if no frame available."""
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            if self._finished.is_set():
                return None
            return None

    def __iter__(self):
        while True:
            item = self.read()
            if item is None:
                break
            yield item
