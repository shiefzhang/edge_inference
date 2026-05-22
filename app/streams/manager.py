from __future__ import annotations

from typing import List

from app.config import get_settings
from app.models.registry import ModelRegistry
from app.schemas import StreamStatus
from app.streams.worker import StreamWorker


class StreamManager:
    def __init__(self, registry: ModelRegistry, stream_count: int | None = None) -> None:
        settings = get_settings()
        self.registry = registry
        count = stream_count or settings.stream_count
        self.workers = {stream_id: StreamWorker(stream_id, registry) for stream_id in range(1, count + 1)}

    def add_stream(self) -> int:
        stream_id = max(self.workers.keys(), default=0) + 1
        self.workers[stream_id] = StreamWorker(stream_id, self.registry)
        return stream_id

    def remove_stream(self, stream_id: int) -> None:
        worker = self._worker(stream_id)
        state = worker.snapshot()
        if state.running:
            raise RuntimeError("stream is running")
        worker.stop()
        del self.workers[stream_id]

    def stream_count(self) -> int:
        return len(self.workers)

    def start(self, stream_id: int, source: str, model_id: str, connection_id: str | None = None, rtsp_enabled: bool = True) -> None:
        self._worker(stream_id).start(source, model_id, connection_id, rtsp_enabled)

    def stop(self, stream_id: int) -> None:
        self._worker(stream_id).stop()

    def switch_model(self, stream_id: int, model_id: str) -> None:
        self._worker(stream_id).switch_model(model_id)

    def latest_jpeg(self, stream_id: int):
        return self._worker(stream_id).latest_jpeg()

    def is_running(self, stream_id: int) -> bool:
        return self._worker(stream_id).is_running()

    def statuses(self) -> List[StreamStatus]:
        return [self.status(stream_id) for stream_id in self.workers]

    def status(self, stream_id: int) -> StreamStatus:
        settings = get_settings()
        state = self._worker(stream_id).snapshot()
        return StreamStatus(
            id=stream_id,
            running=state.running,
            source=state.source,
            connection_id=state.connection_id,
            model_id=state.model_id,
            fps=round(state.fps, 2),
            frames=state.frames,
            last_error=state.last_error,
            browser_url=f"/api/video/{stream_id}",
            rtsp_url=f"rtsp://{settings.rtsp_public_host}:{settings.mediamtx_port}/stream/{stream_id}",
        )

    def close(self) -> None:
        for worker in self.workers.values():
            worker.stop()

    def _worker(self, stream_id: int) -> StreamWorker:
        if stream_id not in self.workers:
            raise KeyError(stream_id)
        return self.workers[stream_id]
