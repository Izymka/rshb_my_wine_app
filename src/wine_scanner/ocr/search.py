"""Текстовый поиск по каталогу: BM25 плюс нечёткое сравнение.

Документ каталога собирается из полей карточки — название, линейка, производитель, регион,
страна, сорта. Распознанный с этикетки текст (`label_text` в meta.json) в документ НЕ входит
намеренно: он записан вручную с той же самой этикетки, и если искать по нему, метрики окажутся
завышенными, а на реальном каталоге платформы такого поля не будет вовсе.

Два механизма дополняют друг друга. BM25 хорошо отбирает кандидатов по совпадающим словам,
но не прощает опечаток. RapidFuzz прощает опечатки, но перебирать им весь каталог дорого.
Поэтому BM25 отбирает, RapidFuzz переоценивает отобранное.
"""

from dataclasses import dataclass

from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz

from .normalize import normalize, tokens, variants

CATALOG_FIELDS = ("name", "brand", "producer", "winery", "region", "country", "grapes")


def catalog_document(payload: dict) -> str:
    """Собрать текст карточки из тех полей, которые реально есть у каталога."""
    parts = []
    for field in CATALOG_FIELDS:
        value = payload.get(field)
        if not value:
            continue
        parts.append(" ".join(map(str, value)) if isinstance(value, list) else str(value))
    return " ".join(parts)


@dataclass
class TextHit:
    item_id: str
    score: float


class TextIndex:
    def __init__(self, item_ids: list[str], documents: list[str], fuzzy_pool: int = 50):
        self.item_ids = item_ids
        self.documents = [normalize(doc) for doc in documents]
        self.fuzzy_pool = fuzzy_pool
        self.bm25 = BM25Okapi([tokens(doc) for doc in documents])

    def search(self, query_text: str, top_k: int = 50) -> list[TextHit]:
        query_tokens = tokens(query_text)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        pool = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.fuzzy_pool]

        query_variants = variants(query_text)
        rescored = []
        for i in pool:
            # token_set_ratio не зависит от порядка слов и лишних слов в запросе — а с этикетки
            # прилетает много лишнего: объём, крепость, «mis en bouteille au château».
            fuzzy = max(fuzz.token_set_ratio(v, self.documents[i]) for v in query_variants) / 100
            # BM25 и fuzzy живут в разных шкалах, поэтому BM25 приводим к своему максимуму.
            bm25_norm = scores[i] / max(scores.max(), 1e-6)
            rescored.append(TextHit(self.item_ids[i], 0.5 * bm25_norm + 0.5 * fuzzy))

        rescored.sort(key=lambda hit: hit.score, reverse=True)
        return rescored[:top_k]
