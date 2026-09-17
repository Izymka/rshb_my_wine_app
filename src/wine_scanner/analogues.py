"""Аналоги из других виноделен: чем заменить вино, если его нет или хочется похожее.

ТЗ (п. 5) особенно ценит, когда на незнакомое вино сервис предлагает аналоги, а не только
«не найдено». Но аналог полезен и при удачном поиске: человек у полки видит, что ещё
попробовать в том же духе. Поэтому аналоги считаются всегда — от лучшего кандидата, — а
«похожие» при отказе остаются близнецами из окна ре-ранкинга: это разные списки с разной
целью, и смешивать их не надо.

Считается без нейросетей и без сети, по полям каталога: одинаковая категория (цвет) —
обязательное условие, дальше сорта винограда, регион, стиль (сладость, игристость) и текст
описания. Винодельня исходного вина исключается — иначе аналогами окажутся его же близнецы.
"""

from dataclasses import dataclass

import numpy as np

from .ocr import attributes
from .ocr.normalize import fold_tokens

WEIGHTS = {"grapes": 0.40, "region": 0.15, "style": 0.20, "description": 0.25}


# «Красные сорта винограда», «купаж» — не сорт, а отсутствие информации о сорте. Совпадение
# двух таких строк ничего не говорит о похожести вин.
GENERIC_GRAPES = ("сорта", "купаж", "смесь", "бленд", "ассамбляж")


def _grapes(payload: dict) -> frozenset[str]:
    raw = payload.get("grapes") or ""
    if isinstance(raw, list):
        raw = ", ".join(map(str, raw))
    grapes = (g.strip().lower() for g in str(raw).split(",") if g.strip())
    return frozenset(g for g in grapes if not any(word in g for word in GENERIC_GRAPES))


@dataclass(frozen=True)
class CardProfile:
    category: str
    region: str
    grapes: frozenset[str]
    style: attributes.Attributes


class Analogues:
    def __init__(self, item_ids: list[str], payloads: list[dict]):
        self.item_ids = list(item_ids)
        self.payloads = list(payloads)
        self.position = {item_id: i for i, item_id in enumerate(self.item_ids)}
        self.profiles = [
            CardProfile(
                category=str(p.get("category") or ""),
                region=str(p.get("region") or ""),
                grapes=_grapes(p),
                style=attributes.from_tokens(fold_tokens(str(p.get("name", "")))),
            )
            for p in payloads
        ]
        self.winery = [str(p.get("winery") or "") for p in payloads]
        self.category = np.array([pr.category for pr in self.profiles])

        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = [str(p.get("description") or "") for p in payloads]
        self.vectorizer = TfidfVectorizer(min_df=2, max_df=0.5, sublinear_tf=True, dtype=np.float32)
        try:
            self.matrix = self.vectorizer.fit_transform(texts)
        except ValueError:
            # Каталог без описаний (тесты, свой набор): текстовая ось просто не участвует.
            self.matrix = None

    def for_item(self, item_id: str, k: int = 5) -> list[dict]:
        """Аналоги карточки из других виноделен, по убыванию похожести."""
        i = self.position.get(item_id)
        if i is None:
            return []
        me = self.profiles[i]
        if self.matrix is not None:
            description = (self.matrix @ self.matrix[i].T).toarray().ravel()
        else:
            description = np.zeros(len(self.item_ids), dtype=np.float32)

        scores = np.zeros(len(self.item_ids), dtype=np.float32)
        for j, other in enumerate(self.profiles):
            if j == i or self.winery[j] == self.winery[i] or other.category != me.category:
                continue
            grapes = (
                len(me.grapes & other.grapes) / len(me.grapes | other.grapes)
                if me.grapes and other.grapes
                else 0.0
            )
            region = 1.0 if me.region and me.region == other.region else 0.0
            style = _style_similarity(me.style, other.style)
            scores[j] = (
                WEIGHTS["grapes"] * grapes
                + WEIGHTS["region"] * region
                + WEIGHTS["style"] * style
                + WEIGHTS["description"] * float(description[j])
            )
        order = np.argsort(-scores)[:k]
        return [self._card(j, float(scores[j])) for j in order if scores[j] > 0]

    def _card(self, j: int, score: float) -> dict:
        p = self.payloads[j]
        return {
            "slug": self.item_ids[j],
            "name": p.get("name"),
            "winery": p.get("winery"),
            "category": p.get("category"),
            "region": p.get("region"),
            "grapes": p.get("grapes"),
            "score": round(score, 3),
        }


def _style_similarity(a: attributes.Attributes, b: attributes.Attributes) -> float:
    """Сладость, игристость, креплёность: совпали — 1, противоречат — 0, неизвестно — 0.5."""
    axes = []
    for x, y in (
        (a.sweetness, b.sweetness),
        (a.sparkling, b.sparkling),
        (a.fortified, b.fortified),
    ):
        if x is None and y is None:
            continue
        axes.append(1.0 if x == y else 0.0 if (x is not None and y is not None) else 0.5)
    return sum(axes) / len(axes) if axes else 0.5
