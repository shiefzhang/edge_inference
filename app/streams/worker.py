from __future__ import annotations

import threading
import time
import logging
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from app.config import get_settings
from app.models.registry import ModelRegistry
from app.streams.drawing import draw_result, encode_jpeg
from app.streams.rtsp import RtspPublisher

MAX_RTSP_FPS = 60
OPENCV_FFMPEG_CAPTURE_OPTIONS = (
    "rtsp_transport;tcp|"
    "stimeout;3000000|"
    "rw_timeout;3000000|"
    "max_delay;500000|"
    "probesize;32768|"
    "analyzeduration;500000"
)
logger = logging.getLogger(__name__)


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
        self._stop_event: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._latest_jpeg: Optional[bytes] = None
        self._publisher: Optional[RtspPublisher] = None
        self._run_generation = 0
        self._stage = "idle"

    def start(self, source: str, model_id: Optional[str], connection_id: Optional[str], rtsp_enabled: bool = True) -> None:
        self.stop()
        with self._lock:
            self.state = WorkerState(True, source, connection_id, model_id)
            self._run_generation += 1
            self._stage = "starting"
            stop_event = threading.Event()
            self._stop_event = stop_event
            self._thread = threading.Thread(target=self._run, args=(source, rtsp_enabled, stop_event, self._run_generation), daemon=True, name=f"stream-{self.stream_id}")
            self._thread.start()
            logger.info("stream %s start requested generation=%s source=%s model=%s rtsp=%s", self.stream_id, self._run_generation, source, model_id or "none", rtsp_enabled)

    def stop(self) -> None:
        with self._lock:
            stop_event = self._stop_event
            thread = self._thread
            publisher = self._publisher
            generation = self._run_generation
            frames = self.state.frames
            fps = self.state.fps
            last_error = self.state.last_error
            stage = self._stage
            self.state.running = False
            self.state.fps = 0.0
            self.state.frames = 0
            self._publisher = None
            self._stop_event = None
            self._latest_jpeg = None
            if not thread or not thread.is_alive():
                self._thread = None
        logger.info(
            "stream %s stop requested generation=%s frames=%s fps=%.2f stage=%s last_error=%s thread_alive=%s publisher_active=%s",
            self.stream_id,
            generation,
            frames,
            fps,
            stage,
            last_error or "",
            bool(thread and thread.is_alive()),
            bool(publisher),
        )
        if stop_event:
            stop_event.set()
        if publisher:
            publisher.stop()
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=0.2)
            if not thread.is_alive():
                with self._lock:
                    if self._thread is thread:
                        self._thread = None
        with self._lock:
            still_alive = bool(thread and thread.is_alive())
            running = self.state.running
            stage = self._stage
            frames = self.state.frames
            fps = self.state.fps
        logger.info(
            "stream %s stop returned generation=%s thread_alive=%s running=%s frames=%s fps=%.2f stage=%s",
            self.stream_id,
            generation,
            still_alive,
            running,
            frames,
            fps,
            stage,
        )

    def switch_model(self, model_id: str) -> None:
        if model_id not in self.registry.modules:
            raise KeyError(model_id)
        with self._lock:
            self.state.model_id = model_id

    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            if not self.state.running:
                return None
            return self._latest_jpeg

    def is_running(self) -> bool:
        with self._lock:
            return self.state.running

    def snapshot(self) -> WorkerState:
        with self._lock:
            return WorkerState(**self.state.__dict__)

    def _run(self, source: str, rtsp_enabled: bool, stop_event: threading.Event, generation: int) -> None:
        settings = get_settings()
        capture_source = int(source) if source.isdigit() else source
        if _is_network_source(source):
            existing_options = os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", OPENCV_FFMPEG_CAPTURE_OPTIONS)
            logger.info("stream %s capture ffmpeg options source=%s options=%s", self.stream_id, source, existing_options)
        cap = cv2.VideoCapture()
        self._set_stage("capture_opening", generation)
        logger.info("stream %s opening capture generation=%s source=%s", self.stream_id, generation, source)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, settings.capture_open_timeout_ms)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, settings.capture_read_timeout_ms)
        cap.open(capture_source)
        if not cap.isOpened():
            self._set_error(f"cannot open source: {source}", generation)
            return
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or settings.frame_width
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or settings.frame_height
        fps = self._normalize_capture_fps(cap.get(cv2.CAP_PROP_FPS), settings.frame_fps)
        self._set_stage("capture_opened", generation)
        logger.info(
            "stream %s capture opened generation=%s source=%s size=%sx%s capture_fps=%s rtsp=%s",
            self.stream_id,
            generation,
            source,
            width,
            height,
            fps,
            rtsp_enabled,
        )
        if rtsp_enabled and not stop_event.is_set():
            publisher: Optional[RtspPublisher] = RtspPublisher(self.stream_id, width, height, fps)
            with self._lock:
                if generation == self._run_generation and not stop_event.is_set():
                    self._publisher = publisher
                else:
                    publisher = None
            if publisher:
                self._set_stage("rtsp_starting", generation)
                publisher.start()
                logger.info("stream %s rtsp publisher started generation=%s url=%s", self.stream_id, generation, publisher.url)
                if stop_event.is_set():
                    with self._lock:
                        if self._publisher is publisher:
                            self._publisher = None
                    publisher.stop()
        last_tick = time.time()
        last_report = last_tick
        last_frames = 0
        try:
            while not stop_event.is_set():
                self._set_stage("capture_read", generation)
                ok, frame = cap.read()
                if not ok:
                    if stop_event.is_set():
                        break
                    self._set_error("frame read failed", generation)
                    time.sleep(0.2)
                    continue
                model_id = self.snapshot().model_id
                if model_id:
                    try:
                        self._set_stage(f"inference:{model_id}", generation)
                        result = self.registry.infer(model_id, frame, timeout=settings.inference_timeout_seconds, stop_event=stop_event)
                        frame = result.annotated_frame if result.annotated_frame is not None else draw_result(frame, result)
                        with self._lock:
                            self.state.last_error = None
                    except InterruptedError:
                        break
                    except Exception as exc:
                        message = str(exc)
                        logger.exception("stream %s model %s inference failed on frame %s", self.stream_id, model_id, self.state.frames + 1)
                        frame = self._draw_error(frame, message)
                        with self._lock:
                            self.state.last_error = message
                if stop_event.is_set():
                    break
                self._set_stage("encode_jpeg", generation)
                jpeg = encode_jpeg(frame)
                with self._lock:
                    self._latest_jpeg = jpeg
                    self.state.frames += 1
                now = time.time()
                if now - last_tick >= 1.0:
                    with self._lock:
                        self.state.fps = (self.state.frames - last_frames) / (now - last_tick)
                        last_frames = self.state.frames
                        current_frames = self.state.frames
                        current_fps = self.state.fps
                        last_error = self.state.last_error
                    last_tick = now
                    if now - last_report >= 5:
                        logger.info(
                            "stream %s heartbeat generation=%s frames=%s fps=%.2f stage=%s frame_size=%sx%s last_error=%s",
                            self.stream_id,
                            generation,
                            current_frames,
                            current_fps,
                            self._stage,
                            frame.shape[1],
                            frame.shape[0],
                            last_error or "",
                        )
                        last_report = now
                    logger.debug(
                        "stream %s heartbeat generation=%s frames=%s fps=%.2f stage=%s frame_size=%sx%s last_error=%s",
                        self.stream_id,
                        generation,
                        current_frames,
                        current_fps,
                        self._stage,
                        frame.shape[1],
                        frame.shape[0],
                        last_error or "",
                    )
                with self._lock:
                    publisher = self._publisher if generation == self._run_generation else None
                if publisher:
                    self._set_stage("rtsp_write", generation)
                    publisher.write(frame)
                self._set_stage("loop_wait", generation)
        except Exception as exc:
            logger.exception("stream %s stopped by worker error", self.stream_id)
            self._set_error(str(exc), generation)
        finally:
            logger.info("stream %s worker cleanup begin generation=%s stage=%s", self.stream_id, generation, self._stage)
            cap.release()
            with self._lock:
                publisher = self._publisher if generation == self._run_generation else None
                if generation == self._run_generation:
                    self._publisher = None
            if publisher:
                publisher.stop()
            with self._lock:
                if generation == self._run_generation:
                    self.state.running = False
                    self._stage = "stopped"
                    if self._thread is threading.current_thread():
                        self._thread = None
            logger.info("stream %s worker cleanup done generation=%s", self.stream_id, generation)

    def _set_error(self, message: str, generation: Optional[int] = None) -> None:
        with self._lock:
            if generation is not None and generation != self._run_generation:
                return
            self.state.last_error = message
            self.state.running = False
            self._stage = "error"

    def _set_stage(self, stage: str, generation: Optional[int] = None) -> None:
        with self._lock:
            if generation is not None and generation != self._run_generation:
                return
            self._stage = stage

    @staticmethod
    def _normalize_capture_fps(raw_fps: float, fallback_fps: int) -> int:
        try:
            fps = int(round(raw_fps))
        except (TypeError, ValueError, OverflowError):
            fps = 0
        if fps <= 0 or fps > MAX_RTSP_FPS:
            return fallback_fps
        return fps

    @staticmethod
    def _draw_error(frame: np.ndarray, message: str) -> np.ndarray:
        output = frame.copy()
        text = f"Inference error: {message[:120]}"
        cv2.rectangle(output, (0, 0), (output.shape[1], 44), (0, 0, 180), -1)
        cv2.putText(output, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return output


def _is_network_source(source: str) -> bool:
    return source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
