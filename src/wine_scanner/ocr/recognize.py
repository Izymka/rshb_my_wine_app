"""Распознавание текста на этикетке.

EasyOCR не умеет держать кириллицу и французский в одном читателе — проверено, отвечает
«Cyrillic is only compatible with English». Российское вино может быть подписано и так, и так,
поэтому держим два читателя и объединяем их выводы.

Объединение безопаснее выбора: если читать латинским читателем кириллическую этикетку,
получится осмысленно выглядящий мусор, который трудно отличить от правды по уверенности.
А лишние нераспознанные слова в запросе поиску почти не мешают — BM25 считает совпавшие
термины, несовпавшие просто не дают вклада.
"""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

OCR_CACHE = Path("models/ocr_cache")

CYRILLIC_LANGS = ["ru", "en"]
LATIN_LANGS = ["fr", "en"]


@dataclass
class TextLine:
    text: str
    confidence: float
    source: str  # какой читатель дал строку


class LabelOCR:
    def __init__(self, min_confidence: float = 0.3, cache_dir: Path = OCR_CACHE):
        self.min_confidence = min_confidence
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._readers: dict[str, object] = {}

    def _reader(self, name: str):
        """Читатели создаются лениво: каждый тянет свою модель распознавания."""
        if name not in self._readers:
            import easyocr

            langs = CYRILLIC_LANGS if name == "cyrillic" else LATIN_LANGS
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._readers[name] = easyocr.Reader(langs, gpu=False, verbose=False)
        return self._readers[name]

    def read(self, image: Image.Image, cache_key: str | None = None) -> list[TextLine]:
        if cache_key is not None:
            cached = self.cache_dir / f"{abs(hash(cache_key)):x}.json"
            if cached.exists():
                data = json.loads(cached.read_text(encoding="utf-8"))
                return [TextLine(**line) for line in data]

        array = np.asarray(image)
        lines: list[TextLine] = []
        for name in ("cyrillic", "latin"):
            for _, text, confidence in self._reader(name).readtext(array):
                if confidence >= self.min_confidence and text.strip():
                    lines.append(TextLine(text.strip(), float(confidence), name))

        if cache_key is not None:
            cached.write_text(
                json.dumps([line.__dict__ for line in lines], ensure_ascii=False),
                encoding="utf-8",
            )
        return lines

    @staticmethod
    def joined(lines: list[TextLine]) -> str:
        return " ".join(line.text for line in lines)
