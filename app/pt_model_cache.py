from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

_ultralytics_dir = Path(__file__).resolve().parents[1] / "data" / "ultralytics"
_ultralytics_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_ultralytics_dir))

from ultralytics import YOLO

from app.schemas import ModelFileDetailOut, ModelLabelOut

logger = logging.getLogger(__name__)


@dataclass
class CachedPtModel:
    path: Path
    model: Optional[YOLO] = None
    loaded: bool = False
    warmup_done: bool = False
    device: str = ""
    memory_allocated_mb: int = 0
    memory_reserved_mb: int = 0
    labels: dict[int, str] = field(default_factory=dict)
    error: str = ""


class PtModelCache:
    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir
        self._items: dict[Path, CachedPtModel] = {}
        self._lock = threading.RLock()

    def preload_all(self) -> None:
        for path in sorted(self.models_dir.glob("*.pt"), key=lambda item: item.name.lower()):
            try:
                self.get(path, warmup=True)
            except Exception:
                logger.exception("failed to preload pt model %s", path.name)

    def get(self, path: Path | str, warmup: bool = True) -> YOLO:
        item = self._load(path, warmup=warmup)
        if item.model is None:
            raise RuntimeError(item.error or f"failed to load {item.path.name}")
        return item.model

    def detail(self, path: Path | str) -> CachedPtModel:
        return self._load(path, warmup=True)

    def _load(self, path: Path | str, warmup: bool) -> CachedPtModel:
        resolved = self._resolve(path)
        with self._lock:
            item = self._items.get(resolved)
            if item and item.loaded and (item.warmup_done or not warmup):
                return item
            item = item or CachedPtModel(resolved)
            self._items[resolved] = item
            before_allocated, before_reserved = _cuda_memory()
            try:
                if item.model is None:
                    item.model = YOLO(str(resolved))
                    item.loaded = True
                    item.labels = {int(key): str(value) for key, value in item.model.names.items()}
                if warmup and not item.warmup_done:
                    _warmup(item.model)
                    item.warmup_done = True
                item.device = _model_device(item.model)
                after_allocated, after_reserved = _cuda_memory()
                item.memory_allocated_mb = max(0, after_allocated - before_allocated)
                item.memory_reserved_mb = max(0, after_reserved - before_reserved)
                item.error = ""
            except Exception as exc:
                item.error = str(exc)
                logger.exception("failed to load pt model %s", resolved.name)
            return item

    def _resolve(self, path: Path | str) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.models_dir / candidate
        return candidate.resolve()


def to_detail(item: CachedPtModel) -> ModelFileDetailOut:
    stat = item.path.stat()
    return ModelFileDetailOut(
        name=item.path.name,
        size=stat.st_size,
        modified_time=__import__("datetime").datetime.fromtimestamp(stat.st_mtime, __import__("datetime").timezone.utc).isoformat(),
        loaded=item.loaded,
        warmup_done=item.warmup_done,
        device=item.device,
        memory_allocated_mb=item.memory_allocated_mb,
        memory_reserved_mb=item.memory_reserved_mb,
        labels=[ModelLabelOut(id=key, name=value) for key, value in sorted(item.labels.items())],
        error=item.error,
    )


_cache: Optional[PtModelCache] = None


def get_pt_model_cache(models_dir: Path) -> PtModelCache:
    global _cache
    if _cache is None or _cache.models_dir != models_dir:
        _cache = PtModelCache(models_dir)
    return _cache


def _warmup(model: YOLO) -> None:
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(image, verbose=False, imgsz=640)


def _cuda_memory() -> tuple[int, int]:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0, 0
        torch.cuda.synchronize()
        return round(torch.cuda.memory_allocated() / 1024 / 1024), round(torch.cuda.memory_reserved() / 1024 / 1024)
    except Exception:
        return 0, 0


def _model_device(model: YOLO) -> str:
    try:
        return str(next(model.model.parameters()).device)
    except Exception:
        return ""
