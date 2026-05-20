from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List

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
        self.reload(definitions)

    def reload(self, definitions: List[ModelFunctionOut]) -> None:
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
        self.executors = {model_id: ThreadPoolExecutor(max_workers=1, thread_name_prefix=model_id) for model_id in self.modules}

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

    def list_models(self) -> List[ModelInfo]:
        return [ModelInfo(**module.metadata.__dict__) for module in self.modules.values()]

    def infer(self, model_id: str, frame: np.ndarray) -> InferenceResult:
        if model_id not in self.modules:
            raise KeyError(model_id)
        future = self.executors[model_id].submit(self.modules[model_id].infer, frame)
        return future.result()

    def close(self) -> None:
        for executor in self.executors.values():
            executor.shutdown(wait=False, cancel_futures=True)
