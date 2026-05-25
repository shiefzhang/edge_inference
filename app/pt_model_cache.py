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
    task: str = ""
    loaded: bool = False
    warmup_done: bool = False
    input_imgsz: int = 0
    device: str = ""
    memory_allocated_mb: int = 0
    memory_reserved_mb: int = 0
    labels: dict[int, str] = field(default_factory=dict)
    error: str = ""


class PtModelCache:
    def __init__(self, models_dir: Path, extension: str = ".pt") -> None:
        self.models_dir = models_dir
        self.extension = extension if extension.startswith(".") else f".{extension}"
        self._items: dict[Path, CachedPtModel] = {}
        self._lock = threading.RLock()

    def preload_all(self) -> None:
        for path in sorted(self.models_dir.glob(f"*{self.extension}"), key=lambda item: item.name.lower()):
            try:
                self.get(path, warmup=True)
            except Exception:
                logger.exception("failed to preload model %s", path.name)

    def get(self, path: Path | str, warmup: bool = True, task: str | None = None) -> YOLO:
        item = self._load(path, warmup=warmup, task=task)
        if item.model is None or (warmup and not item.warmup_done):
            raise RuntimeError(item.error or f"failed to load {item.path.name}")
        return item.model

    def detail(self, path: Path | str, warmup: bool = False, task: str | None = None) -> CachedPtModel:
        return self._load(path, warmup=warmup, task=task)

    def peek(self, path: Path | str) -> CachedPtModel:
        resolved = self._resolve(path)
        with self._lock:
            return self._items.get(resolved) or CachedPtModel(resolved)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def _load(self, path: Path | str, warmup: bool, task: str | None = None) -> CachedPtModel:
        resolved = self._resolve(path)
        normalized_task = (task or "").strip()
        with self._lock:
            item = self._items.get(resolved)
            task_matches = not normalized_task or not item or item.task == normalized_task
            if item and item.loaded and task_matches and (item.warmup_done or not warmup):
                return item
            item = item or CachedPtModel(resolved)
            self._items[resolved] = item
            before_allocated, before_reserved = _cuda_memory()
            try:
                if item.model is None or not task_matches:
                    item.model = YOLO(str(resolved), task=normalized_task) if normalized_task else YOLO(str(resolved))
                    item.task = normalized_task or str(getattr(item.model, "task", "") or "")
                    item.loaded = True
                    item.warmup_done = False
                    item.input_imgsz = 0
                    item.labels = {int(key): str(value) for key, value in item.model.names.items()}
                if warmup and not item.warmup_done:
                    item.input_imgsz = _warmup(item.model, resolved)
                    item.warmup_done = True
                item.device = _model_device(item.model)
                after_allocated, after_reserved = _cuda_memory()
                item.memory_allocated_mb = max(0, after_allocated - before_allocated)
                item.memory_reserved_mb = max(0, after_reserved - before_reserved)
                item.error = ""
            except Exception as exc:
                item.error = str(exc)
                logger.exception("failed to load model %s", resolved.name)
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


_caches: dict[tuple[Path, str], PtModelCache] = {}


def get_pt_model_cache(models_dir: Path, extension: str = ".pt") -> PtModelCache:
    normalized_extension = extension if extension.startswith(".") else f".{extension}"
    key = (models_dir.resolve(), normalized_extension)
    if key not in _caches:
        _caches[key] = PtModelCache(models_dir, normalized_extension)
    return _caches[key]


def clear_model_caches() -> None:
    for cache in _caches.values():
        cache.clear()
    _caches.clear()


def clear_model_caches_except(models_dir: Path, extension: str) -> None:
    normalized_extension = extension if extension.startswith(".") else f".{extension}"
    keep = (models_dir.resolve(), normalized_extension)
    for key in list(_caches):
        if key == keep:
            continue
        _caches[key].clear()
        del _caches[key]


def yolo_predict_kwargs(model: YOLO) -> dict[str, int]:
    input_imgsz = int(getattr(model, "_edge_input_imgsz", 0) or 0)
    return {"imgsz": input_imgsz} if input_imgsz else {}


def _warmup(model: YOLO, path: Path) -> int:
    candidates = _warmup_imgsz_candidates(path)
    last_error: Exception | None = None
    for imgsz in candidates:
        image = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        try:
            model.predict(image, verbose=False, imgsz=imgsz)
            setattr(model, "_edge_input_imgsz", imgsz)
            if hasattr(model, "overrides") and isinstance(model.overrides, dict):
                model.overrides["imgsz"] = imgsz
            return imgsz
        except Exception as exc:
            last_error = exc
    if last_error:
        raise last_error
    raise RuntimeError(f"no warmup size candidates for {path.name}")


def _warmup_imgsz_candidates(path: Path) -> list[int]:
    if path.suffix.lower() == ".onnx":
        fixed = _onnx_fixed_imgsz(path)
        if fixed:
            return [fixed]
    return [640, 320]


def _onnx_fixed_imgsz(path: Path) -> int:
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        shape = session.get_inputs()[0].shape
        height, width = shape[2], shape[3]
        if isinstance(height, int) and isinstance(width, int) and height == width:
            return height
    except Exception:
        logger.debug("failed to inspect onnx input shape for %s", path.name, exc_info=True)
    return 0


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
