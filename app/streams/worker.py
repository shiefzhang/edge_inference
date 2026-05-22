from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2

from app.config import get_settings
from app.models.registry import ModelRegistry
from app.streams.drawing import draw_result, encode_jpeg
from app.streams.rtsp import RtspPublisher

MAX_RTSP_FPS = 60


@dataclass
class WorkerState:
    running: bool = False
    source: Optional[str] = None
    connection_id: Optional[str] = None
    model_id: Optional[str] = None
    fps: float = 0.0
    frames: int = 0
    last_error: Optional[str] = None


class StreamWorker:
    def __init__(self, stream_id: int, registry: ModelRegistry) -> None:
        self.stream_id = stream_id
        self.registry = registry
        self.state = WorkerState()
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._latest_jpeg: Optional[bytes] = None
        self._publisher: Optional[RtspPublisher] = None

    def start(self, source: str, model_id: str, connection_id: Optional[str], rtsp_enabled: bool = True) -> None:
        self.stop()
        with self._lock:
            self.state = WorkerState(True, source, connection_id, model_id)
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, args=(source, rtsp_enabled), daemon=True, name=f"stream-{self.stream_id}")
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        if self._publisher:
            self._publisher.stop()
        with self._lock:
            self.state.running = False
            self._thread = None
            self._publisher = None

    def switch_model(self, model_id: str) -> None:
        if model_id not in self.registry.modules:
            raise KeyError(model_id)
        with self._lock:
            self.state.model_id = model_id

    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def snapshot(self) -> WorkerState:
        with self._lock:
            return WorkerState(**self.state.__dict__)

    def _run(self, source: str, rtsp_enabled: bool) -> None:
        settings = get_settings()
        capture_source = int(source) if source.isdigit() else source
        cap = cv2.VideoCapture(capture_source)
        if not cap.isOpened():
            self._set_error(f"cannot open source: {source}")
            return
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or settings.frame_width
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or settings.frame_height
        fps = self._normalize_capture_fps(cap.get(cv2.CAP_PROP_FPS), settings.frame_fps)
        if rtsp_enabled:
            self._publisher = RtspPublisher(self.stream_id, width, height, fps)
            self._publisher.start()
        last_tick = time.time()
        last_frames = 0
        try:
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    self._set_error("frame read failed")
                    time.sleep(0.2)
                    continue
                model_id = self.snapshot().model_id
                if model_id:
                    result = self.registry.infer(model_id, frame)
                    frame = result.annotated_frame if result.annotated_frame is not None else draw_result(frame, result)
                jpeg = encode_jpeg(frame)
                with self._lock:
                    self._latest_jpeg = jpeg
                    self.state.frames += 1
                    self.state.last_error = None
                if self._publisher:
                    self._publisher.write(frame)
                now = time.time()
                if now - last_tick >= 1.0:
                    with self._lock:
                        self.state.fps = (self.state.frames - last_frames) / (now - last_tick)
                        last_frames = self.state.frames
                    last_tick = now
        except Exception as exc:
            self._set_error(str(exc))
        finally:
            cap.release()
            if self._publisher:
                self._publisher.stop()
            with self._lock:
                self.state.running = False

    def _set_error(self, message: str) -> None:
        with self._lock:
            self.state.last_error = message
            self.state.running = False

    @staticmethod
    def _normalize_capture_fps(raw_fps: float, fallback_fps: int) -> int:
        try:
            fps = int(round(raw_fps))
        except (TypeError, ValueError, OverflowError):
            fps = 0
        if fps <= 0 or fps > MAX_RTSP_FPS:
            return fallback_fps
        return fps
