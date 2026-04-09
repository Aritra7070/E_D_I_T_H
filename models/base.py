from __future__ import annotations

import base64
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


def calibrate_score(value: float) -> float:
    return clamp(value, 0.05, 0.95)


def status_from_score(value: float) -> str:
    if value >= 0.68:
        return "Manipulated"
    if value >= 0.34:
        return "Suspicious"
    return "Authentic Signals"


def normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype("float32")
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if max_value - min_value < 1e-6:
        return np.zeros_like(values, dtype="float32")
    return (values - min_value) / (max_value - min_value)


def pil_image_to_data_url(image: Image.Image, image_format: str = "PNG") -> str:
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime = "image/png" if image_format.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def array_to_data_url(array: np.ndarray, image_format: str = "PNG") -> str:
    clipped = np.clip(array, 0, 255).astype("uint8")
    image = Image.fromarray(clipped)
    return pil_image_to_data_url(image, image_format=image_format)


@dataclass
class DetectorResult:
    fake_score: float
    confidence: float
    explanation: str
    details: dict[str, Any] = field(default_factory=dict)
    component_scores: dict[str, float] = field(default_factory=dict)
