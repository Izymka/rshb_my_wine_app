"""Распознавание текста на этикетке.

EasyOCR не умеет держать кириллицу и французский в одном читателе — проверено, отвечает
«Cyrillic is only compatible with English». Российское вино может быть подписано и так, и так,
поэтому держим два читателя и объединяем их выводы.

Объединение безопаснее выбора: если читать латинским читателем кириллическую этикетку,
получится осмысленно выглядящий мусор, который трудно отличить от правды по уверенности.
А лишние нераспознанные слова в запросе поиску почти не мешают — BM25 считает совпавшие
термины, несовпавшие просто не дают вклада.
"""

import hashlib
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

OCR_CACHE = Path("models/ocr_cache")

CYRILLIC_LANGS = ["ru", "en"]
LATIN_LANGS = ["fr", "en"]

# Сторона, до которой ужимается кадр перед распознаванием.
#
# Величина выбрана замером. Время распознавания линейно по площади: на кадре с телефона
# (сторона около 2000 после обрезки) OCR занимает 15.3 с против 3.4 при 1024 и 1.3 при 640.
# Прогон бенчмарка показал, что качество при этом не страдает: top-1 0.691 / 0.713 / 0.723,
# R@50 0.979 / 0.979 / 0.989. Разница в пределах шума набора, то есть выигрыш во времени
# достаётся бесплатно.
#
# Почему лишние пиксели не помогают: EasyOCR внутри приводит найденные строки к своей рабочей
# высоте, и подача текста в четыре раза крупнее не добавляет ему информации. А вот детектору
# текста мелкая сетка даже мешает — на 640 группа blur читается лучше, чем на полном размере.
DEFAULT_MAX_SIDE = 640

# Версия формата кэша. Меняется, когда в TextLine появляется поле: старые записи его не
# содержат, и без версии они бы молча подсовывали строки без координат, а блок винтажа
# просто никогда бы не срабатывал — при полностью зелёных тестах.
CACHE_VERSION = 2

# Читатели. `easyocr` — локальный, работает всегда; `yandex` — облачный Yandex Vision
# (ocr/yandex.py), при любой ошибке откатывается на EasyOCR. Выбор — переменной окружения
# WINE_OCR, чтобы сервис и бенчмарк переключались без правки кода.
BACKENDS = ("easyocr", "yandex")
DEFAULT_BACKEND = os.environ.get("WINE_OCR", "easyocr")
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
        if self.cloud is None and self.backend == "yandex":
            from .yandex import YandexOCR

            self.cloud = YandexOCR()
        # Сколько раз облако не ответило и кадр дочитал EasyOCR. Уходит в /health.
        self.fallbacks = 0
        # EasyOCR умеет только CUDA: внутри он проверяет torch.cuda и ничего не знает про MPS.
        # На ноутбуке это значит процессор и 1.2 с на кадр, на машине с картой — порядок
        # выигрыша, потому что распознавание здесь самый дорогой блок после ре-ранкинга.
        self.gpu = torch.cuda.is_available() if gpu is None else gpu
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._readers: dict[str, object] = {}

    def _prepare(self, image: Image.Image) -> Image.Image:
        """Ужать кадр до max_side по длинной стороне.

        Распознавание не выигрывает от лишних пикселей: EasyOCR внутри всё равно приводит
        строки к своей высоте. Зато время растёт линейно по площади, а это самая дорогая
        часть всего пайплайна.
        """
        side = self.effective_max_side()
        if side is None or max(image.size) <= side:
            return image
        scaled = image.copy()
        scaled.thumbnail((side, side), Image.BICUBIC)
        return scaled

    def effective_max_side(self) -> int | None:
        """До какой стороны ужимается кадр: у облака порог выше, чем у EasyOCR."""
        if self.max_side is None:
            return None
        return max(self.max_side, CLOUD_MAX_SIDE) if self.backend != "easyocr" else self.max_side

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
        # Имя бэкенда попадает в ключ только у не-EasyOCR: иначе обесценился бы кэш на сотни
        # запросов, накопленный до появления облака, — а он и есть то, чем живёт сборка признаков.
        if self.backend != "easyocr":
            digest.update(f":{self.backend}".encode())
        return self.cache_dir / f"{digest.hexdigest()}.json"

    def _reader(self, name: str):
        """Читатели создаются лениво: каждый тянет свою модель распознавания."""
        if name not in self._readers:
            import easyocr

            langs = CYRILLIC_LANGS if name == "cyrillic" else LATIN_LANGS
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._readers[name] = easyocr.Reader(langs, gpu=self.gpu, verbose=False)
        return self._readers[name]

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
        if self.backend == "easyocr" or self.cloud is None:
            return self._read_easyocr(image)
        from .yandex import OCRBackendError

        try:
            return self.cloud.read(image)
        except OCRBackendError:
            self.fallbacks += 1
            return self._read_easyocr(image)

    def _read_easyocr(self, image: Image.Image) -> list[TextLine]:
        array = np.asarray(image)
        width, height = image.size
        lines: list[TextLine] = []
        for name in ("cyrillic", "latin"):
            for box, text, confidence in self._reader(name).readtext(array):
                if confidence >= self.min_confidence and text.strip():
                    lines.append(
                        TextLine(text.strip(), float(confidence), name, _rect(box, width, height))
                    )
        return lines

    def read_digits(self, image: Image.Image, min_confidence: float = 0.1) -> str:
        """Прочитать на маленьком участке только цифры.

        Отличий от `read` три, и все ради Э8. Читатель один: цифры одинаковы во всех
        алфавитах, второй проход был бы платой ни за что. Распознавателю запрещено всё, кроме
        цифр, — иначе в «2021» он норовит увидеть слово, а нам нужно число. И порог уверенности
        ниже: на четырёх знаках без словарного контекста распознаватель уверен в себе слабее,
        а от мусора нас всё равно защищает проверка правдоподобия года.

        Кэша здесь нет: участок вырезан по геометрии конкретной пары и второй раз не повторится.
        Читает всегда EasyOCR, независимо от бэкенда: облаку не объяснить «только цифры».
        """
        result = self._reader("latin").readtext(
            np.asarray(image), allowlist="0123456789", detail=1
        )
        return " ".join(
            text.strip() for _, text, confidence in result if confidence >= min_confidence
        )

    @staticmethod
    def joined(lines: list[TextLine]) -> str:
        return " ".join(line.text for line in lines)


def _rect(box, width: int, height: int) -> tuple[float, float, float, float]:
    """Четырёхугольник EasyOCR -> охватывающий прямоугольник в долях кадра.

    EasyOCR отдаёт четыре угла, потому что строка может идти под наклоном. Наклон нам не
    нужен: участок всё равно вырезается с запасом, а прямоугольник переносится через
    гомографию четырьмя углами так же, как любой другой.
    """
    xs = [float(point[0]) / width for point in box]
    ys = [float(point[1]) / height for point in box]
    return (
        max(0.0, min(xs)),
        max(0.0, min(ys)),
        min(1.0, max(xs)),
        min(1.0, max(ys)),
    )
