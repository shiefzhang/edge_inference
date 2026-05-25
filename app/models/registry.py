from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
import importlib
import importlib.util
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from app.config import get_settings
from app.models.base import BaseInferenceModule, InferenceResult
from app.schemas import ModelFunctionOut, ModelInfo


class ModelRegistry:
    def __init__(self, definitions: List[ModelFunctionOut], models_dir: Path | None = None) -> None:
        settings = get_settings()
        self.models_dir = models_dir or settings.models_dir
        self.modules: Dict[str, BaseInferenceModule] = {}
        self.executors: Dict[str, ThreadPoolExecutor] = {}
        self._lock = threading.RLock()
        self.reload(definitions)

    def reload(self, definitions: List[ModelFunctionOut]) -> None:
        with self._lock:
            for executor in self.executors.values():
                executor.shutdown(wait=False, cancel_futures=True)
            self.modules = {}
            for definition in definitions:
                if not definition.enabled:
                    continue
                module = self._build_module(definition)
                module.metadata.function_entrypoint = definition.entrypoint
                module.metadata.description = definition.description
                self.modules[module.metadata.id] = module
            self.executors = {model_id: self._new_executor(model_id) for model_id in self.modules}

    def _build_module(self, definition: ModelFunctionOut) -> BaseInferenceModule:
        module_name, function_name = definition.entrypoint.split(":", 1)
        importlib.invalidate_caches()
        if module_name.startswith("app.user_functions."):
            builder = self._load_user_builder(module_name, function_name)
        else:
            builder = getattr(importlib.import_module(module_name), function_name)
        return builder(definition, self.models_dir, self.modules)

    def _load_user_builder(self, module_name: str, function_name: str):
        settings = get_settings()
        module_stem = module_name.rsplit(".", 1)[-1]
        path = settings.user_functions_dir / f"{module_stem}.py"
        if not path.exists():
            raise FileNotFoundError(path)
        if module_name in sys.modules:
            del sys.modules[module_name]
        spec = importlib.util.spec_from_file_location(module_name, path)
        if not spec or not spec.loader:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return getattr(module, function_name)

    def load_all(self) -> None:
        for module in self.modules.values():
            module.load()

    def preload_all(self) -> None:
        for model_id, module in self.modules.items():
            try:
                module.load()
            except Exception:
                logging.getLogger(__name__).exception("failed to preload model function %s", model_id)

    def list_models(self) -> List[ModelInfo]:
        return [ModelInfo(**module.metadata.__dict__) for module in self.modules.values()]

    def infer(
        self,
        model_id: str,
        frame: np.ndarray,
        timeout: float | None = None,
        stop_event: Optional[threading.Event] = None,
    ) -> InferenceResult:
        with self._lock:
            if model_id not in self.modules:
                raise KeyError(model_id)
            module = self.modules[model_id]
            executor = self.executors[model_id]
        future = executor.submit(module.infer, frame)
        deadline = time.monotonic() + timeout if timeout else None
        while True:
            if stop_event and stop_event.is_set():
                future.cancel()
                raise InterruptedError("model inference interrupted by stream stop")
            wait_seconds = 0.1
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    future.cancel()
                    self._replace_executor(model_id, executor)
                    raise TimeoutError(f"model inference timed out after {timeout}s")
                wait_seconds = min(wait_seconds, remaining)
            try:
                return future.result(timeout=wait_seconds)
            except TimeoutError:
                continue

    def _new_executor(self, model_id: str) -> ThreadPoolExecutor:
        return ThreadPoolExecutor(max_workers=1, thread_name_prefix=model_id)

    def _replace_executor(self, model_id: str, executor: ThreadPoolExecutor) -> None:
        with self._lock:
            if self.executors.get(model_id) is not executor:
                return
            executor.shutdown(wait=False, cancel_futures=True)
            self.executors[model_id] = self._new_executor(model_id)

    def close(self) -> None:
        with self._lock:
            for executor in self.executors.values():
                executor.shutdown(wait=False, cancel_futures=True)
