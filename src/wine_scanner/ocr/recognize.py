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


@dataclass
class TextLine:
    text: str
    confidence: float
    source: str  # какой читатель дал строку


class LabelOCR:
    def __init__(
        self,
        min_confidence: float = 0.3,
        cache_dir: Path = OCR_CACHE,
        max_side: int | None = DEFAULT_MAX_SIDE,
        gpu: bool | None = None,
    ):
        self.min_confidence = min_confidence
        self.max_side = max_side
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
        if self.max_side is None or max(image.size) <= self.max_side:
            return image
        scaled = image.copy()
        scaled.thumbnail((self.max_side, self.max_side), Image.BICUBIC)
        return scaled

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
        digest.update(f"{image.size}:{self.min_confidence}:{self.max_side}".encode())
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

        array = np.asarray(image)
        lines: list[TextLine] = []
        for name in ("cyrillic", "latin"):
            for _, text, confidence in self._reader(name).readtext(array):
                if confidence >= self.min_confidence and text.strip():
                    lines.append(TextLine(text.strip(), float(confidence), name))

        if cached is not None:
            cached.write_text(
                json.dumps([line.__dict__ for line in lines], ensure_ascii=False),
                encoding="utf-8",
            )
        return lines

    @staticmethod
    def joined(lines: list[TextLine]) -> str:
        return " ".join(line.text for line in lines)
