from __future__ import annotations

import importlib
import inspect
import io
import logging
import os
import sys
import threading
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np
from PIL import Image

_ultralytics_dir = Path(__file__).resolve().parents[2] / "data" / "ultralytics"
_ultralytics_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_ultralytics_dir))

from ultralytics import YOLO

from app.config import get_settings
from app.models.base import BaseInferenceModule, InferenceBox, InferenceResult, ModelMetadata, RED

logger = logging.getLogger(__name__)
_stdio_redirect_lock = threading.RLock()


class _LogStream(io.TextIOBase):
    def __init__(self, level: int) -> None:
        self.level = level
        self._buffer = ""

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                logger.log(self.level, line.rstrip())
        return len(text)

    def flush(self) -> None:
        if self._buffer.strip():
            logger.log(self.level, self._buffer.rstrip())
        self._buffer = ""


class FuncLogicModule(BaseInferenceModule):
    def __init__(
        self,
        model_id: str,
        name: str,
        task: str,
        logic_module: str,
        logic_function: Optional[str],
        models_dir: Path,
        config: Dict[str, Any],
    ) -> None:
        self.logic_module = logic_module
        self.logic_function = logic_function
        self.models_dir = models_dir
        self.config = config
        self.conf = float(config.get("conf", config.get("conf_threshold", 0.25)))
        self._function: Optional[Callable[..., Any]] = None
        self._models: Dict[str, YOLO] = {}
        self._lock = threading.Lock()
        self.metadata = ModelMetadata(
            id=model_id,
            name=name,
            task=task,
            path=str(config.get("model_path", "")),
            labels={},
            function_entrypoint=f"{logic_module}:{logic_function or ''}",
            description=None,
        )

    def load(self) -> None:
        with self._lock:
            if self._function is None:
                self._function = self._load_function()
            self._ensure_models()

    def infer(self, frame: np.ndarray) -> InferenceResult:
        self.load()
        assert self._function is not None
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        kwargs = self._build_call_kwargs()
        with self._lock:
            raw = self._call_function(image, kwargs)
        annotated, detections = self._normalize_output(raw, frame)
        result = InferenceResult(annotated_frame=annotated)
        for detection in detections:
            result.labels.extend(self._alarm_labels(detection))
            box = self._box_from_detection(detection)
            if box:
                result.boxes.append(box)
        return result

    def _call_function(self, image: Image.Image, kwargs: Dict[str, Any]) -> Any:
        stdout = _LogStream(logging.INFO)
        stderr = _LogStream(logging.ERROR)
        with _stdio_redirect_lock:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                try:
                    return self._function(image, **kwargs)
                finally:
                    stdout.flush()
                    stderr.flush()

    def _load_function(self) -> Callable[..., Any]:
        settings = get_settings()
        func_dir = str(settings.model_logic_dir)
        if func_dir not in sys.path:
            sys.path.insert(0, func_dir)
        importlib.invalidate_caches()
        module_name = self.logic_module
        if module_name in sys.modules:
            del sys.modules[module_name]
        module = importlib.import_module(module_name)
        function_name = self.logic_function or self._detect_logic_function(module)
        self.logic_function = function_name
        return getattr(module, function_name)

    def _detect_logic_function(self, module) -> str:
        module_stem = self.logic_module.rsplit(".", 1)[-1]
        candidates = [module_stem.removeprefix("model_"), module_stem]
        for name in candidates:
            value = getattr(module, name, None)
            if callable(value):
                return name
        for name, value in inspect.getmembers(module, inspect.isfunction):
            if value.__module__ == module.__name__ and not name.startswith("_"):
                return name
        raise ValueError(f"no callable model logic function found in {self.logic_module}")

    def _ensure_models(self) -> None:
        for arg_name, model_path in self._model_bindings().items():
            if arg_name not in self._models:
                self._models[arg_name] = YOLO(str(self.models_dir / model_path))
        primary = self.config.get("model_path")
        if primary and self.metadata.labels == {}:
            primary_model = next(reversed(self._models.values()), None)
            if primary_model is not None:
                self.metadata.labels = dict(primary_model.names)

    def _model_bindings(self) -> Dict[str, str]:
        configured = dict(self.config.get("model_bindings") or {})
        signature = inspect.signature(self._function) if self._function else None
        model_args = [
            name
            for name, param in (signature.parameters.items() if signature else [])
            if name.endswith("_model") and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
        ]
        primary = self.config.get("model_path")
        if primary and model_args:
            configured.setdefault(model_args[-1], str(primary))
        for arg_name, default_path in self._default_aux_models().items():
            if arg_name in model_args:
                configured.setdefault(arg_name, default_path)
        return {key: value for key, value in configured.items() if value}

    def _default_aux_models(self) -> Dict[str, str]:
        return {
            "human_model": str(self.config.get("human_model_path", "05person_best11m.pt")),
            "default_model": str(self.config.get("default_model_path", "yolo11n.pt")),
        }

    def _build_call_kwargs(self) -> Dict[str, Any]:
        kwargs = {"conf_threshold": self.conf}
        kwargs.update(self.config.get("logic_kwargs") or {})
        kwargs.update(self._models)
        return kwargs

    def _normalize_output(self, raw: Any, fallback_frame: np.ndarray) -> tuple[np.ndarray, list[Any]]:
        detections: list[Any] = []
        annotated = None
        if isinstance(raw, tuple):
            if len(raw) >= 1:
                annotated = raw[0]
            if len(raw) >= 2 and raw[1] is not None:
                detections = list(raw[1])
        elif isinstance(raw, list):
            detections = raw
        elif isinstance(raw, np.ndarray):
            annotated = raw
        if isinstance(annotated, Image.Image):
            annotated = np.array(annotated)
        if isinstance(annotated, np.ndarray):
            if annotated.ndim != 3 or annotated.shape[2] < 3:
                return fallback_frame.copy(), detections
            if annotated.shape[:2] != fallback_frame.shape[:2]:
                annotated = cv2.resize(annotated, (fallback_frame.shape[1], fallback_frame.shape[0]))
            return cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR), detections
        return fallback_frame.copy(), detections

    def _alarm_labels(self, detection: Any) -> list[str]:
        item = self._detection_dict(detection)
        if not item.get("violation"):
            return []
        content = item.get("viol_content") or item.get("label") or "alarm"
        return [str(content)]

    def _box_from_detection(self, detection: Any) -> Optional[InferenceBox]:
        item = self._detection_dict(detection)
        box = item.get("box")
        if not box or len(box) != 4:
            return None
        label = str(item.get("viol_content") or item.get("label") or item.get("type") or "alarm")
        score = float(item.get("conf") or 0.0)
        color = tuple(reversed(item.get("viol_color") or (255, 0, 0))) if item.get("violation") else RED
        return InferenceBox(tuple(int(v) for v in box), label, score, color)

    @staticmethod
    def _detection_dict(detection: Any) -> Dict[str, Any]:
        if hasattr(detection, "model_dump"):
            return detection.model_dump()
        if isinstance(detection, dict):
            return detection
        return getattr(detection, "__dict__", {})
