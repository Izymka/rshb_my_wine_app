"""Yandex Vision OCR как облачный читатель этикеток.

Зачем облако, если есть PaddleOCR. На каталоге платформы решает текст: близнецы одной линейки
различаются словами «красный / белый / крымский», и одна неверная буква («МУСКАТЕАЬ») стоит
верной карточки. PaddleOCR может слабо прочитать мелкий или бликующий текст, поэтому Vision
служит резервом для сложных кадров и читает 1024-пиксельный
кадр за секунду на процессоре. Yandex Vision читает русский заметно чище, отвечает за доли
секунды и работает из России без обходных путей. ТЗ хакатона внешние сервисы разрешает прямо.

Облако — не единственная точка отказа: любая ошибка (сеть, таймаут, лимит, кривой ответ)
поднимает `OCRBackendError`, и `LabelOCR` молча дочитывает кадр PaddleOCR, считая откаты.
На живом прогоне у организаторов это важнее качества: `null` по таймауту — потерянный кадр.

Ключи — из окружения: `YANDEX_OCR_API_KEY` (сервисный аккаунт с ролью `ai.vision.user`)
и `YANDEX_FOLDER_ID`. Логирование данных на стороне облака выключено заголовком.
"""

import base64
import io
import os
from dataclasses import dataclass

import httpx
from PIL import Image

from .recognize import TextLine

URL = "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText"
DEFAULT_TIMEOUT = 2.5
# Уверенность строк v1 не отдаёт. Константа честно говорит решающему слою «не знаю»: на этом
# бэкенде признак ocr_conf почти постоянный, и обучаться он должен на признаках того же бэкенда.
DEFAULT_CONFIDENCE = 0.9


class OCRBackendError(RuntimeError):
    """Облачный читатель не ответил или ответил не тем. Сигнал для отката на PaddleOCR."""


@dataclass
class YandexOCR:
    api_key: str | None = None
    folder_id: str | None = None
    timeout: float = DEFAULT_TIMEOUT
    languages: tuple[str, ...] = ("ru", "en")
    client: httpx.Client | None = None
    name: str = "yandex"

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("YANDEX_OCR_API_KEY")
        self.folder_id = self.folder_id or os.environ.get("YANDEX_FOLDER_ID")
        if self.client is None:
            self.client = httpx.Client(timeout=httpx.Timeout(self.timeout, connect=1.0))

    def read(self, image: Image.Image) -> list[TextLine]:
        if not self.api_key:
            raise OCRBackendError("нет YANDEX_OCR_API_KEY")
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=90)
        payload = {
            "mimeType": "JPEG",
            "languageCodes": list(self.languages),
            "model": "page",
            "content": base64.b64encode(buffer.getvalue()).decode("ascii"),
        }
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "x-data-logging-enabled": "false",
        }
        if self.folder_id:
            headers["x-folder-id"] = self.folder_id
        try:
            response = self.client.post(URL, json=payload, headers=headers)
        except httpx.HTTPError as error:
            raise OCRBackendError(f"Yandex OCR: {error.__class__.__name__}: {error}") from error
        if response.status_code != 200:
            raise OCRBackendError(f"Yandex OCR: HTTP {response.status_code}: {response.text[:200]}")
        try:
            data = response.json()
        except ValueError as error:
            raise OCRBackendError("Yandex OCR: ответ не JSON") from error
        return parse_response(data, *image.size)


def parse_response(data: dict, width: int, height: int) -> list[TextLine]:
    """Строки из ответа v1: текст, уверенность (если есть) и прямоугольник в долях кадра.

    Координаты в ответе — пиксели отправленной картинки; её размер приходит в
    `textAnnotation.width/height` строками, при отсутствии берём размер того, что послали.
    Прямоугольник — охватывающий четырёхугольник строки в нормированных координатах.
    """
    try:
        annotation = data["result"]["textAnnotation"]
    except (KeyError, TypeError) as error:
        raise OCRBackendError("Yandex OCR: в ответе нет textAnnotation") from error
    frame_w = float(annotation.get("width") or width) or 1.0
    frame_h = float(annotation.get("height") or height) or 1.0

    lines: list[TextLine] = []
    for block in annotation.get("blocks", []):
        for line in block.get("lines", []):
            text = str(line.get("text", "")).strip()
            if not text:
                continue
            words = line.get("words", [])
            confidences = [float(w["confidence"]) for w in words if "confidence" in w]
            confidence = sum(confidences) / len(confidences) if confidences else DEFAULT_CONFIDENCE
            lines.append(
                TextLine(
                    text, confidence, "yandex", _rect(line.get("boundingBox"), frame_w, frame_h)
                )
            )
    return lines


def _rect(bounding_box, width: float, height: float):
    vertices = (bounding_box or {}).get("vertices") or []
    if len(vertices) < 2:
        return None
    xs = [float(v.get("x", 0)) / width for v in vertices]
    ys = [float(v.get("y", 0)) / height for v in vertices]
    return (max(0.0, min(xs)), max(0.0, min(ys)), min(1.0, max(xs)), min(1.0, max(ys)))
