from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional, Set

import numpy as np

_ultralytics_dir = Path(__file__).resolve().parents[2] / "data" / "ultralytics"
_ultralytics_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_ultralytics_dir))

from ultralytics import YOLO

from app.models.base import BLUE, GRAY, RED, BaseInferenceModule, InferenceBox, InferenceResult, ModelMetadata
from app.models.yolo_detector import YoloDetectorModule


class PersonCropClassifierModule(BaseInferenceModule):
    def __init__(
        self,
        model_id: str,
        name: str,
        model_path: Path,
        person_detector: YoloDetectorModule,
        pass_labels: Set[str],
        fail_labels: Set[str],
        neutral_labels: Optional[Set[str]] = None,
        conf: float = 0.25,
    ) -> None:
        self.model_path = model_path
        self.person_detector = person_detector
        self.pass_labels = pass_labels
        self.fail_labels = fail_labels
        self.neutral_labels = neutral_labels or set()
        self.conf = conf
        self._model: Optional[YOLO] = None
        self._lock = threading.Lock()
        self.metadata = ModelMetadata(
            id=model_id,
            name=name,
            task="classify",
            path=str(model_path),
            labels={},
            requires_person_detector=True,
        )

    def load(self) -> None:
        self.person_detector.load()
        with self._lock:
            if self._model is None:
                self._model = YOLO(str(self.model_path))
                self.metadata.labels = dict(self._model.names)

    def infer(self, frame: np.ndarray) -> InferenceResult:
        self.load()
        assert self._model is not None
        people = self.person_detector.infer(frame)
        output = InferenceResult()
        for person in people.boxes:
            x1, y1, x2, y2 = self._clip_box(person.xyxy, frame)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            with self._lock:
                predictions = self._model.predict(crop, conf=self.conf, verbose=False)
            label, score = self._top_class(predictions)
            color = self._label_color(label)
            text = f"{person.label} {label} {score:.2f}"
            output.boxes.append(InferenceBox((x1, y1, x2, y2), text, score, color, {"class_label": label}))
        return output

    def _top_class(self, predictions) -> tuple[str, float]:
        if not predictions or predictions[0].probs is None:
            return "unknown", 0.0
        probs = predictions[0].probs
        cls = int(probs.top1)
        return self.metadata.labels.get(cls, str(cls)), float(probs.top1conf.item())

    def _label_color(self, label: str):
        if label in self.pass_labels:
            return BLUE
        if label in self.fail_labels:
            return RED
        if label in self.neutral_labels:
            return GRAY
        return RED

    @staticmethod
    def _clip_box(xyxy, frame: np.ndarray):
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = xyxy
        return max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2)
