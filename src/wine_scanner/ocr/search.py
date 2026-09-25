"""Текстовый поиск по каталогу и текстовые сигналы для решающего слоя.

Документ карточки собирается из полей каталога — название, винодельня, регион, сорта,
категория, год. Распознанный с этикетки текст (`label_text` в meta.json своего набора) в
документ НЕ входит намеренно: он записан вручную с той же самой этикетки, и если искать по
нему, метрики окажутся завышенными, а на каталоге платформы такого поля нет вовсе.

Как ищем. Основной скорер — TF-IDF по символьным n-граммам (3–5 букв) над свёрнутыми
документами (`normalize.fold`: общий алфавит, фонетика, псевдонимы). Он переживает то, на чём
ломался пословный поиск: одну неверную букву OCR («МУСКАТЕАЬ» вместо «МУСКАТЕЛЬ») и смену
алфавита («Chateau Tamagne» против «Шато Тамань»). Проверено на каталоге платформы: ранг
верной карточки на реальных строках OCR 12 -> 2 и 15 -> 0. Второй ярус — пословное нечёткое
покрытие с IDF-весами на верхушке списка: n-граммы любят длинные документы с общими
кусками, покрытие возвращает вес редким словам. Прежний путь BM25 + token_set_ratio оставлен
как `scorer="bm25"` для сравнения.

Кроме поиска индекс знает про каждую карточку то, что нужно решающему слою для близнецов:
`disc` — слова названия, редкие внутри её винодельни (у «Портвейн красный Алушта» это
«красный» и «алушта», а «портвейн» общий для шести карточек), и атрибуты — цвет, сладость,
игристость. Метод `signals` сравнивает их со словами этикетки.
"""

import math
import os
from dataclasses import dataclass, field, replace

import numpy as np
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz, process

from . import attributes
from .grapes import GrapeVocabulary
from .grapes import compare as compare_grapes
from .normalize import fold, fold_tokens, folded_variants, normalize, tokens, variants

CATALOG_FIELDS = (
    "name",
    "brand",
    "producer",
    "winery",
    "region",
    "country",
    "grapes",
    "category",
    "vintage",
)
# Слова, которые есть у половины виноделен и ничего не различают.
GENERIC_TOKENS = frozenset(
    fold(w)
    for w in ("вино", "винодельня", "усадьба", "винодельческое", "хозяйство", "дом", "имение")
)
DISC_MAX_SHARE = float(os.environ.get("WINE_DISC_MAX_SHARE", "0.5"))
DEFAULT_SCORER = os.environ.get("WINE_TEXT_SCORER", "ngram")
NGRAM_RANGE = (3, 5)


def catalog_document(payload: dict, fields: tuple[str, ...] = CATALOG_FIELDS) -> str:
    """Собрать текст карточки из тех полей, которые реально есть у каталога."""
    parts = []
    for field in fields:
        value = payload.get(field)
        if value is None or value == "" or (isinstance(value, float) and math.isnan(value)):
            continue
        parts.append(" ".join(map(str, value)) if isinstance(value, list) else str(value))
    return " ".join(parts)


@dataclass
class TextHit:
    item_id: str
    score: float


@dataclass(frozen=True)
class CardText:
    """Слова карточки в свёрнутой форме — то, с чем сравнивается этикетка."""

    name: frozenset[str]
    winery: frozenset[str]
    disc: frozenset[str]
    attrs: attributes.Attributes
    family: str
    # Вес различающего слова — редкость внутри винодельни: «алушта» (2 карточки из 44) весит
    # больше, чем «портвейн» (6 из 44). Иначе карточка с коротким названием выигрывала бы
    # долю за счёт одного общего слова.
    disc_weights: dict[str, float] = field(default_factory=dict)


def token_hits(
    ocr_tokens: list[str],
    tokens_: frozenset[str] | set[str],
    min_ratio: float = 0.8,
    weights: dict[str, float] | None = None,
) -> float:
    """Какая доля слов `tokens_` найдена среди слов этикетки, с весами.

    Длинные слова ищутся нечётко (одна ошибка OCR на слово — норма), короткие и числа —
    только точно: «2023» с одной другой цифрой — это другой год, а не опечатка. Нечёткое
    совпадение допускается только между словами почти одной длины: иначе «мускат» находится
    в «мускатель», «порт» в «портвейн», и короткие названия выигрывают у длинных даром.
    """
    if not tokens_:
        return 0.0
    if not ocr_tokens:
        return 0.0
    total = 0.0
    found = 0.0
    for token in tokens_:
        weight = weights.get(token, 1.0) if weights else 1.0
        total += weight
        if len(token) <= 3 or token.isdigit():
            hit = token in ocr_tokens
        else:
            pool = [t for t in ocr_tokens if abs(len(t) - len(token)) <= 1]
            hit = bool(
                pool
                and process.extractOne(token, pool, scorer=fuzz.ratio, score_cutoff=min_ratio * 100)
            )
        if hit:
            found += weight
    return found / total if total else 0.0


class TextIndex:
    def __init__(
        self,
        item_ids: list[str],
        documents: list[str],
        payloads: list[dict] | None = None,
        fuzzy_pool: int = 100,
        scorer: str | None = None,
        disc_max_share: float = DISC_MAX_SHARE,
        blend: float = 0.5,
    ):
        self.item_ids = list(item_ids)
        self.position = {item_id: i for i, item_id in enumerate(self.item_ids)}
        self.fuzzy_pool = fuzzy_pool
        self.scorer = scorer or DEFAULT_SCORER
        if self.scorer not in ("ngram", "bm25"):
            raise ValueError(f"неизвестный текстовый скорер: {self.scorer}")
        self.blend = blend

        # Прежний путь: BM25 по словам плюс нечёткая переоценка. Держим всегда — он дешёвый.
        self.documents = [normalize(doc) for doc in documents]
        self.bm25 = BM25Okapi([tokens(doc) for doc in documents])

        # Новый путь: символьные n-граммы над свёрнутыми документами.
        self.folded = [fold(doc) for doc in documents]
        self.vectorizer = None
        self.matrix = None
        if self.scorer == "ngram":
            from sklearn.feature_extraction.text import TfidfVectorizer

            self.vectorizer = TfidfVectorizer(
                analyzer="char_wb", ngram_range=NGRAM_RANGE, sublinear_tf=True, dtype=np.float32
            )
            self.matrix = self.vectorizer.fit_transform(self.folded)

        # IDF по словам — веса для покрытия: «массандра» есть у 44 карточек, «алушта» у двух.
        n = len(self.folded)
        df: dict[str, int] = {}
        for doc in self.folded:
            for token in set(doc.split()):
                df[token] = df.get(token, 0) + 1
        self.idf = {t: math.log((n + 1) / (d + 1)) + 1.0 for t, d in df.items()}

        self.cards: dict[str, CardText] = {}
        self.grapes = GrapeVocabulary()
        self.family_of: dict[str, str] = {}
        self.members: dict[str, list[str]] = {}
        if payloads is not None:
            self._build_cards(payloads, disc_max_share)

    @classmethod
    def from_payloads(
        cls,
        item_ids: list[str],
        payloads: list[dict],
        fields: tuple[str, ...] = CATALOG_FIELDS,
        **kwargs,
    ) -> "TextIndex":
        documents = [catalog_document(p, fields) for p in payloads]
        return cls(item_ids, documents, payloads=payloads, **kwargs)

    # --- карточки ------------------------------------------------------------------------

    def _build_cards(self, payloads: list[dict], disc_max_share: float) -> None:
        self.grapes = GrapeVocabulary.from_payloads(payloads)
        names = [frozenset(fold_tokens(str(p.get("name", "")))) for p in payloads]
        families = [
            str(p.get("winery") or p.get("producer") or p.get("brand") or "") for p in payloads
        ]
        for item_id, family in zip(self.item_ids, families, strict=True):
            self.family_of[item_id] = family
            self.members.setdefault(family, []).append(item_id)

        # Сколько карточек винодельни содержат слово: общее для линейки слово различать
        # не может, редкое — может.
        family_df: dict[str, dict[str, int]] = {}
        for name, family in zip(names, families, strict=True):
            counts = family_df.setdefault(family, {})
            for token in name:
                counts[token] = counts.get(token, 0) + 1

        for item_id, payload, name, family in zip(
            self.item_ids, payloads, names, families, strict=True
        ):
            size = len(self.members[family])
            counts = family_df[family]
            # Слова цвета и сладости в различающие не идут: их сравнивают атрибуты
            # (color_match / style_match), а здесь они бы дали карточке «Мускатель розовый»
            # подтверждение от одного слова «розовый» на этикетке чужого вина.
            disc = frozenset(
                t
                for t in name
                if t not in GENERIC_TOKENS
                and size > 1
                and counts[t] <= disc_max_share * size
                and not attributes.is_attribute_word(t)
            )
            winery = frozenset(t for t in fold_tokens(family) if t not in GENERIC_TOKENS)
            self.cards[item_id] = CardText(
                name=name,
                winery=winery,
                disc=disc,
                attrs=attributes.from_card(payload, sorted(name)),
                family=family,
                disc_weights={t: math.log(size / counts[t]) + 0.1 for t in disc},
            )

    @property
    def disc_tokens(self) -> dict[str, frozenset[str]]:
        return {item_id: card.disc for item_id, card in self.cards.items()}

    # --- запрос ---------------------------------------------------------------------------

    @staticmethod
    def query_tokens(query_text: str) -> list[str]:
        """Свёрнутые слова этикетки по обоим вариантам чтения (прямой и через гомоглифы)."""
        seen: list[str] = []
        for variant in variants(query_text):
            for token in fold_tokens(variant):
                if token not in seen:
                    seen.append(token)
        return seen

    def query_attributes(
        self, query_tokens: list[str], query_text: str | None = None
    ) -> attributes.Attributes:
        """Цвет, сладость, игристость по словам; сорта — по тексту, если он дан: сорт ищется
        фразой, а в `query_tokens` слова уже без повторов и могут стоять не рядом."""
        found = attributes.from_tokens(query_tokens)
        text = query_text if query_text is not None else " ".join(query_tokens)
        return replace(found, grapes=self.grapes.find(text))

    def score_all(self, query_text: str) -> np.ndarray:
        """Оценка каждой карточки каталога, в порядке `item_ids`."""
        if self.scorer == "bm25":
            scores = np.asarray(self.bm25.get_scores(tokens(query_text)), dtype=np.float32)
            top = float(scores.max()) if len(scores) else 0.0
            return scores / top if top > 0 else scores
        query = self.vectorizer.transform(folded_variants(query_text))
        sims = (self.matrix @ query.T).toarray()
        return sims.max(axis=1).astype(np.float32)

    def search(
        self, query_text: str, top_k: int = 50, scores: np.ndarray | None = None
    ) -> list[TextHit]:
        query_tokens = self.query_tokens(query_text)
        if not query_tokens:
            return []
        if self.scorer == "bm25":
            return self._search_bm25(query_text, top_k)

        if scores is None:
            scores = self.score_all(query_text)
        pool = np.argsort(-scores)[: self.fuzzy_pool]
        top = float(scores[pool[0]]) if len(pool) else 0.0
        rescored = []
        for i in pool:
            item_id = self.item_ids[i]
            card = self.cards.get(item_id)
            if card is not None:
                cover = token_hits(query_tokens, card.name | card.winery, weights=self.idf)
            else:
                cover = token_hits(
                    query_tokens, frozenset(self.folded[i].split()), weights=self.idf
                )
            ngram = float(scores[i]) / top if top > 0 else 0.0
            rescored.append(TextHit(item_id, (1 - self.blend) * ngram + self.blend * cover))
        rescored.sort(key=lambda hit: hit.score, reverse=True)
        return rescored[:top_k]

    def _search_bm25(self, query_text: str, top_k: int) -> list[TextHit]:
        """Прежний скорер: BM25 отбирает, RapidFuzz переоценивает отобранное."""
        query_tokens = tokens(query_text)
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        pool = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            : min(self.fuzzy_pool, 50)
        ]
        query_variants = variants(query_text)
        rescored = []
        for i in pool:
            fuzzy = max(fuzz.token_set_ratio(v, self.documents[i]) for v in query_variants) / 100
            bm25_norm = scores[i] / max(scores.max(), 1e-6)
            rescored.append(TextHit(self.item_ids[i], 0.5 * bm25_norm + 0.5 * fuzzy))
        rescored.sort(key=lambda hit: hit.score, reverse=True)
        return rescored[:top_k]

    # --- сигналы для решающего слоя ---------------------------------------------------------

    def signals(
        self, ocr_tokens: list[str], ocr_attrs: attributes.Attributes, item_id: str
    ) -> dict[str, float]:
        """Сколько слов карточки подтверждает этикетка и не противоречит ли ей цвет."""
        card = self.cards.get(item_id)
        if card is None:
            return {
                "disc_hit": 0.0,
                "disc_n": 0,
                "name_cover": 0.0,
                "winery_hit": 0,
                "color_match": 0,
                "style_match": 0,
                "grape_match": 0,
            }
        color, style = attributes.compare(ocr_attrs, card.attrs)
        return {
            "disc_hit": token_hits(ocr_tokens, card.disc, weights=card.disc_weights),
            "disc_n": len(card.disc),
            "name_cover": token_hits(ocr_tokens, card.name, weights=self.idf),
            "winery_hit": int(token_hits(ocr_tokens, card.winery) > 0),
            "color_match": color,
            "style_match": style,
            "grape_match": compare_grapes(ocr_attrs.grapes, card.attrs.grapes),
        }

    def family_rank(
        self, family: str, scores: np.ndarray | None, exclude: set[str], limit: int
    ) -> list[str]:
        """Карточки винодельни, которых ещё нет в списке, по убыванию текстовой оценки."""
        candidates = [i for i in self.members.get(family, []) if i not in exclude]
        if scores is not None:
            candidates.sort(key=lambda i: -float(scores[self.position[i]]))
        return candidates[:limit]
