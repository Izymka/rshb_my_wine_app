"""Local PaddleOCR backend for wine labels.

PP-OCRv5 has one multilingual recognition model for Russian and Latin label text. The mobile
detector keeps end-to-end API latency practical on CPU. On the Windows CPU build PaddlePaddle
currently fails in oneDNN for the detector, therefore this adapter deliberately runs the regular
Paddle backend with oneDNN disabled. The models are loaded lazily and cached by PaddleX outside
the repository.
"""

import os
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .recognize import TextLine
from .worker import new_engine, payload, receive, send


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
    device: str = "cpu"
    _engine: object | None = field(default=None, init=False, repr=False)
    name: str = "paddle"

    def _get_engine(self):
        if self._engine is None:
            self._engine = new_engine(self.device)
        return self._engine

    def _predict(self, array: np.ndarray) -> dict:
        return payload(self._get_engine().predict(array))

    def read(self, image: Image.Image) -> list[TextLine]:
        result = self._predict(np.asarray(image.convert("RGB")))
        if "error" in result:
            raise RuntimeError(f"PaddleOCR: {result['error']}")
        texts, scores, polygons = (result.get(k, []) for k in ("texts", "scores", "polys"))
        lines: list[TextLine] = []
        for index, text in enumerate(texts):
            confidence = float(scores[index]) if index < len(scores) else 0.0
            clean = str(text).strip()
            if clean and confidence >= self.min_confidence:
                polygon = polygons[index] if index < len(polygons) else None
                lines.append(TextLine(clean, confidence, self.name, _box(polygon, *image.size)))
        return lines


# Движок внутри процесса-воркера: один на процесс, создаётся при его старте.
_WORKER_ENGINE = None


def _worker_init(device: str) -> None:
    global _WORKER_ENGINE
    _WORKER_ENGINE = new_engine(device)


def _worker_predict(array: np.ndarray) -> dict:
    return payload(_WORKER_ENGINE.predict(array))


@dataclass
class ProcessPaddleOCR(PaddleOCR):
    """PaddleOCR в отдельном процессе того же окружения.

    Текстовая ветка идёт параллельно с ре-ранкингом, но в потоке того же процесса она
    параллельной только выглядела: Paddle держит GIL, и описание кадра XFeat (7 мс в одиночку)
    ждало его ~1 с. Замер 28.09 на 40 живых кадрах train: последовательный и «параллельный»
    режимы давали одинаковые 2.4 с, в своём процессе — 2.1 с (p95 3.6 → 3.1 с).
    Процесс запускается через spawn: fork процесса с CUDA-контекстом небезопасен.
    """

    _pool: object | None = field(default=None, init=False, repr=False)

    def _executor(self):
        if self._pool is None:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor

            self._pool = ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_worker_init,
                initargs=(self.device,),
            )
        return self._pool

    def warm(self):
        """Поднять воркер и модели заранее, чтобы первый запрос не ждал их загрузки."""
        return self._executor().submit(_worker_predict, np.zeros((64, 64, 3), dtype=np.uint8))

    def _predict(self, array: np.ndarray) -> dict:
        from concurrent.futures.process import BrokenProcessPool

        try:
            return self._executor().submit(_worker_predict, array).result()
        except BrokenProcessPool:
            # Воркер упал (нехватка памяти, сбой Paddle) — поднимаем новый и пробуем один раз.
            self._pool = None
            return self._executor().submit(_worker_predict, array).result()


@dataclass
class SubprocessPaddleOCR(PaddleOCR):
    """PaddleOCR в отдельном интерпретаторе — обычно окружение с `paddlepaddle-gpu`.

    `paddlepaddle-gpu` фиксирует CUDA-библиотеки под 12.6, torch проекта собран под 12.8: в одно
    окружение они не ставятся (см. worker.py). Воркер — самостоятельный скрипт, общается по
    stdin/stdout. Замер 28.09 на 30 вырезках train: GPU 27 мс на кадр против 1.4 с на CPU,
    текст совпал на 25 из 30, отличия — одиночные символы с уверенностью < 0.65.
    """

    python: str = ""
    _process: object | None = field(default=None, init=False, repr=False)
    _lock: object = field(default_factory=threading.Lock, init=False, repr=False)

    def _start(self):
        script = Path(__file__).with_name("worker.py")
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        # Путь к пакету воркеру не нужен и вреден: в его окружении нет torch.
        env.pop("PYTHONPATH", None)
        process = subprocess.Popen(
            [self.python, str(script), "--device", self.device],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=env,
        )
        ready = receive(process.stdout)
        if not ready or not ready.get("ready"):
            process.kill()
            raise RuntimeError(f"OCR-воркер {self.python} не поднялся: {ready!r}")
        self._process = process
        return process

    def warm(self):
        """Поднять воркер в фоне: загрузка моделей и CUDA занимает секунды."""
        thread = threading.Thread(target=self._ensure, daemon=True, name="ocr-warm")
        thread.start()
        return thread

    def _ensure(self):
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                self._start()
            return self._process

    def _predict(self, array: np.ndarray) -> dict:
        for attempt in range(2):
            try:
                with self._lock:
                    if self._process is None or self._process.poll() is not None:
                        self._start()
                    send(self._process.stdin, np.ascontiguousarray(array))
                    result = receive(self._process.stdout)
                if result is None:
                    raise BrokenPipeError("OCR-воркер закрыл поток")
                return result
            except (BrokenPipeError, OSError):
                # Воркер упал — поднимаем новый и пробуем ещё раз.
                self._process = None
                if attempt:
                    raise
        return {}
