"""Распознавание текста на этикетке.

PP-OCRv5 читает кириллицу и латиницу одной локальной моделью, поэтому текстовая ветка не
склеивает два несовместимых читателя и не получает латинский «похожий мусор» вместо русского.
Yandex Vision остаётся точечным облачным резервом для слабого локального чтения.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

OCR_CACHE = Path("models/ocr_cache")

# Сторона, до которой ужимается кадр перед распознаванием.
#
# Величина выбрана замером. Время распознавания линейно по площади: на кадре с телефона
# (сторона около 2000 после обрезки) OCR занимает 15.3 с против 3.4 при 1024 и 1.3 при 640.
# Прогон бенчмарка показал, что качество при этом не страдает: top-1 0.691 / 0.713 / 0.723,
# R@50 0.979 / 0.979 / 0.989. Разница в пределах шума набора, то есть выигрыш во времени
# достаётся бесплатно.
#
# Умеренный размер ограничивает задержку локального распознавания и сохраняет мелкий текст.
DEFAULT_MAX_SIDE = 640

# Версия формата и конфигурации кэша. Меняется при новом поле TextLine или модели OCR: иначе
# старые строки молча попадут в признаки нового CatBoost и смешают два разных читателя.
CACHE_VERSION = 3

# `paddle` — локальная ветка по умолчанию. `yandex` выбирает Vision целиком с локальным
# откатом. `hybrid` сначала читает PaddleOCR и посылает кадр в Vision только если локальный
# текст слишком слабый: так облако помогает сложным случаям, а не становится точкой отказа.
BACKENDS = ("paddle", "yandex", "hybrid")
DEFAULT_BACKEND = os.environ.get("WINE_OCR", "paddle")
# Облаку платим за запрос, а не за пиксели, и читает оно мелкий текст лучше на большем кадре.
CLOUD_MAX_SIDE = 1024


@dataclass
class TextLine:
    text: str
    confidence: float
    source: str  # какой читатель дал строку
    # Прямоугольник строки в долях кадра: (x0, y0, x1, y1), начало координат — левый верхний
    # угол. Доли, а не пиксели, потому что кадр по пути ужимается: до OCR — до max_side,
    # до XFeat — до своего предела. Абсолютные координаты пришлось бы пересчитывать при каждой
    # передаче между блоками, и однажды кто-нибудь забыл бы.
    box: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        # Из json бокс приходит списком — приводим к кортежу, чтобы строка из кэша и строка,
        # посчитанная только что, вели себя одинаково.
        if self.box is not None:
            self.box = tuple(float(v) for v in self.box)  # type: ignore[assignment]


class LabelOCR:
    def __init__(
        self,
        min_confidence: float = 0.3,
        cache_dir: Path = OCR_CACHE,
        max_side: int | None = DEFAULT_MAX_SIDE,
        gpu: bool | None = None,
        backend: str | None = None,
        cloud=None,
    ):
        self.min_confidence = min_confidence
        self.max_side = max_side
        self.backend = backend or DEFAULT_BACKEND
        if self.backend not in BACKENDS:
            raise ValueError(f"неизвестный OCR-бэкенд {self.backend!r}, знаю {BACKENDS}")
        # Облачный клиент можно подменить (тесты, другой провайдер): нужен только .read(image).
        self.cloud = cloud
        if self.cloud is None and self.backend in {"yandex", "hybrid"}:
            from .yandex import YandexOCR

            self.cloud = YandexOCR()
        # Сколько раз облако не ответило и кадр дочитал PaddleOCR. Уходит в /health.
        self.fallbacks = 0
        self.cloud_calls = 0
        # Оставлен для обратной совместимости вызывающего кода. PaddleOCR в текущей Windows
        # сборке работает на CPU; GPU потребует отдельный PaddlePaddle wheel.
        self.gpu = gpu
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._local = None

    def _prepare(self, image: Image.Image) -> Image.Image:
        """Ужать кадр до max_side по длинной стороне.

        Распознавание не выигрывает от избыточного разрешения, зато время растёт с площадью.
        Поэтому ограничение стороны остаётся главным ограничителем задержки OCR.
        """
        side = self.effective_max_side()
        if side is None or max(image.size) <= side:
            return image
        scaled = image.copy()
        scaled.thumbnail((side, side), Image.BICUBIC)
        return scaled

    def effective_max_side(self) -> int | None:
        """До какой стороны ужимается кадр: облако получает более детальный crop."""
        if self.max_side is None:
            return None
        if self.backend in {"yandex", "hybrid"}:
            return max(self.max_side, CLOUD_MAX_SIDE)
        return self.max_side

    def _cache_path(self, image: Image.Image) -> Path:
        """Ключ кэша считается по самим пикселям, а не по имени файла.

        Раньше ключом был `hash(путь)`, и это не работало вовсе: хэш строки в Python
        рандомизируется при каждом запуске процесса, поэтому кэш ни разу не попадал —
        файлы копились, а OCR каждый прогон считался заново. Путь как ключ плох и по существу:
        одна и та же картинка приходит сюда обрезанной по-разному в зависимости от детектора,
        и при смене конфигурации кэш молча отдавал бы чужой результат.

        Адресация по содержимому снимает обе проблемы разом. Хэш 12 мегабайт занимает
        миллисекунды против секунд распознавания.
        """
        digest = hashlib.sha1(image.tobytes())
        digest.update(
            f"{image.size}:{self.min_confidence}:{self.effective_max_side()}:v{CACHE_VERSION}".encode()
        )
        digest.update(f":{self.backend}".encode())
        return self.cache_dir / f"{digest.hexdigest()}.json"

    def _paddle(self):
        if self._local is None:
            from .paddle import PaddleOCR

            self._local = PaddleOCR(min_confidence=self.min_confidence)
        return self._local

    def read(self, image: Image.Image, use_cache: bool = False) -> list[TextLine]:
        """Распознать текст. Кэш включается явно — он нужен бенчмаркам, а не сервису.

        В сервисе кэшировать нечего: каждый снимок пользователя уникален, а файлы копились бы
        без ограничений.
        """
        image = self._prepare(image)

        cached = self._cache_path(image) if use_cache else None
        if cached is not None and cached.exists():
            data = json.loads(cached.read_text(encoding="utf-8"))
            return [TextLine(**line) for line in data]

        lines = self._read_backend(image)

        if cached is not None:
            cached.write_text(
                json.dumps([line.__dict__ for line in lines], ensure_ascii=False),
                encoding="utf-8",
            )
        return lines

    def _read_backend(self, image: Image.Image) -> list[TextLine]:
        if self.backend == "paddle" or self.cloud is None:
            return self._read_paddle(image)
        from .yandex import OCRBackendError

        local = self._read_paddle(image) if self.backend == "hybrid" else None
        if local is not None and not self._needs_cloud(local):
            return local
        try:
            self.cloud_calls += 1
            cloud = self.cloud.read(image)
            return cloud or (local or [])
        except OCRBackendError:
            self.fallbacks += 1
            return local if local is not None else self._read_paddle(image)

    @staticmethod
    def _needs_cloud(lines: list[TextLine]) -> bool:
        """Only weak local evidence pays for a Vision request in hybrid mode."""
        return len(lines) < 3 or max((line.confidence for line in lines), default=0.0) < 0.6

    def _read_paddle(self, image: Image.Image) -> list[TextLine]:
        return self._paddle().read(image)

    def read_digits(self, image: Image.Image, min_confidence: float = 0.1) -> str:
        """Прочитать на маленьком участке только цифры.

        Отличий от `read` три, и все ради Э8. Читатель один: цифры одинаковы во всех
        алфавитах, второй проход был бы платой ни за что. Распознавателю запрещено всё, кроме
        цифр, — иначе в «2021» он норовит увидеть слово, а нам нужно число. И порог уверенности
        ниже: на четырёх знаках без словарного контекста распознаватель уверен в себе слабее,
        а от мусора нас всё равно защищает проверка правдоподобия года.

        Кэша здесь нет: участок вырезан по геометрии конкретной пары и второй раз не повторится.
        Читает локальный PaddleOCR: облаку не объяснить «только цифры», а год не должен
        делать внешний запрос.
        """
        return " ".join(
            "".join(char for char in line.text if char.isdigit())
            for line in self._read_paddle(image)
            if line.confidence >= min_confidence
        ).strip()

    @staticmethod
    def joined(lines: list[TextLine]) -> str:
        return " ".join(line.text for line in lines)
