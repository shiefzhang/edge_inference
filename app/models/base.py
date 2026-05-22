from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


Color = Tuple[int, int, int]
BLUE: Color = (255, 0, 0)
RED: Color = (0, 0, 255)
GRAY: Color = (160, 160, 160)


@dataclass
class InferenceBox:
    xyxy: Tuple[int, int, int, int]
    label: str
    score: float
    color: Color = BLUE
    metadata: Dict = field(default_factory=dict)


@dataclass
class InferenceResult:
    boxes: List[InferenceBox] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    annotated_frame: Optional[np.ndarray] = None


@dataclass
class ModelMetadata:
    id: str
    name: str
    task: str
    path: str
    labels: Dict[int, str]
    requires_person_detector: bool = False
    function_entrypoint: Optional[str] = None
    description: Optional[str] = None


class BaseInferenceModule:
    metadata: ModelMetadata

    def load(self) -> None:
        raise NotImplementedError

    def infer(self, frame: np.ndarray) -> InferenceResult:
        raise NotImplementedError
