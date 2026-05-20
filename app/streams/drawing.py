from __future__ import annotations

import cv2
import numpy as np

from app.models.base import InferenceResult


def draw_result(frame: np.ndarray, result: InferenceResult) -> np.ndarray:
    output = frame.copy()
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy
        cv2.rectangle(output, (x1, y1), (x2, y2), box.color, 2)
        label = f"{box.label} {box.score:.2f}" if box.score else box.label
        _draw_label(output, label, x1, max(0, y1 - 8), box.color)
    for index, label in enumerate(result.labels):
        cv2.putText(output, label, (16, 28 + index * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return output


def encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return b""
    return encoded.tobytes()


def _draw_label(frame: np.ndarray, text: str, x: int, y: int, color) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    top = max(0, y - height - baseline)
    cv2.rectangle(frame, (x, top), (x + width + 8, top + height + baseline + 6), color, -1)
    cv2.putText(frame, text, (x + 4, top + height + 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
