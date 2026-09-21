"""Local PaddleOCR backend for wine labels.

PP-OCRv5 has one multilingual recognition model for Russian and Latin label text. The mobile
detector keeps end-to-end API latency practical on CPU. On the Windows CPU build PaddlePaddle
currently fails in oneDNN for the detector, therefore this adapter deliberately runs the regular
Paddle backend with oneDNN disabled. The models are loaded lazily and cached by PaddleX outside
the repository.
"""

import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .recognize import TextLine


def _box(value, width: int, height: int) -> tuple[float, float, float, float] | None:
    """Convert a Paddle polygon or rectangle to a fractional enclosing rectangle."""
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.size < 4:
        return None
    if array.ndim == 1 and array.size == 4:
        x0, y0, x1, y1 = array.tolist()
    else:
        points = array.reshape(-1, 2)
        x0, y0 = points.min(axis=0)
        x1, y1 = points.max(axis=0)
    return (
        max(0.0, min(1.0, float(x0) / max(width, 1))),
        max(0.0, min(1.0, float(y0) / max(height, 1))),
        max(0.0, min(1.0, float(x1) / max(width, 1))),
        max(0.0, min(1.0, float(y1) / max(height, 1))),
    )


@dataclass
class PaddleOCR:
    """Small stable adapter over the PaddleOCR 3.x result format."""

    min_confidence: float = 0.3
    _engine: object | None = field(default=None, init=False, repr=False)
    name: str = "paddle"

    def _get_engine(self):
        if self._engine is None:
            # Avoid a slow source reachability probe on every service startup.  PaddleX still
            # downloads a missing official model normally; installed models are used from cache.
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            from paddleocr import PaddleOCR as Engine

            self._engine = Engine(
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="eslav_PP-OCRv5_mobile_rec",
                device="cpu",
                enable_mkldnn=False,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        return self._engine

    def read(self, image: Image.Image) -> list[TextLine]:
        result = self._get_engine().predict(np.asarray(image.convert("RGB")))
        if not result:
            return []
        payload = getattr(result[0], "json", result[0])
        payload = payload.get("res", payload)
        texts = payload.get("rec_texts", [])
        scores = payload.get("rec_scores", [])
        polygons = payload.get("rec_polys") or payload.get("dt_polys") or []
        lines: list[TextLine] = []
        for index, text in enumerate(texts):
            confidence = float(scores[index]) if index < len(scores) else 0.0
            clean = str(text).strip()
            if clean and confidence >= self.min_confidence:
                polygon = polygons[index] if index < len(polygons) else None
                lines.append(TextLine(clean, confidence, self.name, _box(polygon, *image.size)))
        return lines
