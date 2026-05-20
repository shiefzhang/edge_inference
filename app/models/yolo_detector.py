from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

_ultralytics_dir = Path(__file__).resolve().parents[2] / "data" / "ultralytics"
_ultralytics_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_ultralytics_dir))

from ultralytics import YOLO

from app.models.base import BLUE, InferenceBox, InferenceResult, ModelMetadata, BaseInferenceModule


class YoloDetectorModule(BaseInferenceModule):
    def __init__(
        self,
        model_id: str,
        name: str,
        model_path: Path,
        conf: float = 0.35,
        allowed_labels: Optional[Iterable[str]] = None,
    ) -> None:
        self.model_path = model_path
        self.conf = conf
        self.allowed_labels = set(allowed_labels or [])
        self._model: Optional[YOLO] = None
        self._lock = threading.Lock()
        self.metadata = ModelMetadata(
            id=model_id,
            name=name,
            task="detect",
            path=str(model_path),
            labels={},
        )

    def load(self) -> None:
        with self._lock:
            if self._model is None:
                self._model = YOLO(str(self.model_path))
                self.metadata.labels = dict(self._model.names)

    def infer(self, frame: np.ndarray) -> InferenceResult:
        self.load()
        assert self._model is not None
        with self._lock:
            predictions = self._model.predict(frame, conf=self.conf, verbose=False)
        result = InferenceResult()
        for prediction in predictions:
            if prediction.boxes is None:
                continue
            for box in prediction.boxes:
                cls = int(box.cls[0].item())
                label = self.metadata.labels.get(cls, str(cls))
                if self.allowed_labels and label not in self.allowed_labels:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                score = float(box.conf[0].item())
                result.boxes.append(InferenceBox((x1, y1, x2, y2), label, score, BLUE))
        return result
