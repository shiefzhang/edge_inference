from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


MODEL_EXTENSIONS = {".pt", ".onnx"}
MODEL_TYPES = {"pt": ".pt", "onnx": ".onnx"}


def model_extension(model_type: str) -> str:
    return MODEL_TYPES.get(model_type.lower(), ".pt")


def model_dir(settings, model_type: str) -> Path:
    return settings.onnx_models_dir if model_type == "onnx" else settings.pt_models_dir


def with_model_extension(value: Any, extension: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    path = Path(text)
    if path.suffix.lower() in MODEL_EXTENSIONS:
        return str(path.with_suffix(extension))
    return f"{text}{extension}"


def without_model_extension(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    path = Path(text)
    if path.suffix.lower() in MODEL_EXTENSIONS:
        return str(path.with_suffix(""))
    return text


def config_for_runtime(config: dict[str, Any], extension: str) -> dict[str, Any]:
    output = deepcopy(config)
    for key in ("model_path", "human_model_path", "default_model_path"):
        if output.get(key):
            output[key] = with_model_extension(output[key], extension)
    bindings = output.get("model_bindings")
    if isinstance(bindings, dict):
        output["model_bindings"] = {key: with_model_extension(value, extension) for key, value in bindings.items()}
    return output


def config_for_display(config: dict[str, Any]) -> dict[str, Any]:
    output = deepcopy(config)
    for key in ("model_path", "human_model_path", "default_model_path"):
        if output.get(key):
            output[key] = without_model_extension(output[key])
    bindings = output.get("model_bindings")
    if isinstance(bindings, dict):
        output["model_bindings"] = {key: without_model_extension(value) for key, value in bindings.items()}
    return output
