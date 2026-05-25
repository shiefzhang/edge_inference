from __future__ import annotations

from pathlib import Path
from typing import Dict

from app.models.base import BaseInferenceModule
from app.models.func_logic import FuncLogicModule
from app.models.person_crop_classifier import PersonCropClassifierModule
from app.models.yolo_detector import YoloDetectorModule
from app.model_paths import config_for_runtime, model_extension
from app.schemas import ModelFunctionOut


def build_yolo_detector(definition: ModelFunctionOut, models_dir: Path, modules: Dict[str, BaseInferenceModule]) -> BaseInferenceModule:
    config = config_for_runtime(definition.config, model_extension(models_dir.name))
    return YoloDetectorModule(
        model_id=definition.id,
        name=definition.name,
        model_path=models_dir / str(config["model_path"]),
        conf=float(config.get("conf", 0.35)),
        allowed_labels=config.get("allowed_labels"),
    )


def build_person_crop_classifier(definition: ModelFunctionOut, models_dir: Path, modules: Dict[str, BaseInferenceModule]) -> BaseInferenceModule:
    config = config_for_runtime(definition.config, model_extension(models_dir.name))
    person_detector_id = str(config.get("person_detector_id", "person_detector"))
    person_detector = modules.get(person_detector_id)
    if not isinstance(person_detector, YoloDetectorModule):
        raise ValueError(f"person detector '{person_detector_id}' must be defined before {definition.id}")
    return PersonCropClassifierModule(
        model_id=definition.id,
        name=definition.name,
        model_path=models_dir / str(config["model_path"]),
        person_detector=person_detector,
        pass_labels=set(config.get("pass_labels", [])),
        fail_labels=set(config.get("fail_labels", [])),
        neutral_labels=set(config.get("neutral_labels", [])),
        conf=float(config.get("conf", 0.25)),
    )


def build_func_model(definition: ModelFunctionOut, models_dir: Path, modules: Dict[str, BaseInferenceModule]) -> BaseInferenceModule:
    config = config_for_runtime(definition.config, model_extension(models_dir.name))
    logic_module = str(config.get("logic_module") or "")
    if not logic_module:
        raise ValueError("config.logic_module is required for func model logic")
    logic_function = config.get("logic_function")
    return FuncLogicModule(
        model_id=definition.id,
        name=definition.name,
        task=definition.task,
        logic_module=logic_module,
        logic_function=str(logic_function) if logic_function else None,
        models_dir=models_dir,
        config=config,
    )
