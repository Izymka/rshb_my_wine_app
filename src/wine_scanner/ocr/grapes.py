"""Сорта винограда: какие сорта названы на этикетке и сходятся ли они с карточкой.

Зачем. Близнецы одной винодельни часто различаются только сортом: «Quintessence. Зинфандель»
и «Quintessence. Каберне Совиньон Ripasso», «Азюр» Рислинг и «Азюр» Вионье. Слово сорта на
этикетке крупное, а у карточки сорта лежат в отдельном поле `grapes`. Сравнение по этой оси
даёт то же, что цвет: +1 сошлось, −1 противоречит, 0 — сравнивать не с чем.

Словарь — не список из головы, а поле `grapes` самого каталога: сравнивать этикетку имеет
смысл только с тем, что может оказаться в карточке. Фразы сворачиваются той же `fold`, что и
текст этикетки, поэтому «RIESLING» и «Рислинг» — одно и то же.

Осторожности:
- ищется самая длинная фраза: «Совиньон Блан» на этикетке — не «Совиньон» из «Каберне
  Совиньон»;
- однословные сорта, которые пишутся как обычные слова или совпадают со словом из названия
  винодельни («Восторг», «Олег», «Молдова»), в словарь не идут — лучше промолчать, чем
  найти сорт в девизе;
- «Белые сорта винограда» — не сорт.
"""

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from .normalize import fold, fold_tokens

# Одно и то же под разными именами; ключ и значение — как пишут в каталоге.
_SYNONYMS = {
    "Пино черный": "Пино Нуар",
    "Пино чёрный": "Пино Нуар",
    "Шираз": "Сира",
    "Пино Гриджио": "Пино Гри",
    "Бьянка": "Бианка",
    # Латинское написание сворачивается не так, как русское: «Blanc» в «beloe», «Petit»
    # не в «пти».
    "Chenin Blanc": "Шенен Блан",
    "Petit Verdot": "Пти Вердо",
}
# Сорта, которых нет ни в одной карточке. Совпасть с карточкой им не с чем, но на этикетке
# они говорят «это другое вино»: «КАРИНЬЯН» у незнакомого вина Солнечной Долины не должен
# молча подтверждать «Чёрного Доктора» из Эким-Кара (разбор ошибок теста 18.10).
_EXTRA = (
    "Кариньян",
    "Альбариньо",
    "Гаме",
    "Барбера",
    "Дольчетто",
    "Пинотаж",
    "Фурминт",
    "Треббьяно",
    "Гарганега",
    "Фиано",
    "Ассиртико",
    "Аглианико",
    "Таннат",
)
# Первое слово сорта, которое на этикетке часто пишут одно: «CABERNET» без «Sauvignon» или
# «Franc». Такое слово совпадает с любым сортом каталога, который с него начинается
# (compare: фраза этикетки вложена в фразу карточки), и противоречит остальным.
_HEADS = ("Каберне",)
# Однословные «сорта», которые слишком легко встретить на этикетке в другом смысле.
_AMBIGUOUS = {
    "Достойный",
    "Восторг",
    "Олег",
    "Голубок",
    "Цветочный",
    "Молдова",
    "Станичный",
    "Декабрьский",
    "Шабаш",
    "Августин",
    "Сенсо",
}
_GENERIC_MARKER = fold("сорт")
_SPLIT = re.compile(r"[,;/\n()]| и ")
# Нечёткое совпадение слова сорта — как в token_hits: одна ошибка OCR на длинное слово.
FUZZY_MIN_LEN = 6
FUZZY_RATIO = 85


def _canonical(phrase: str) -> str:
    folded = " ".join(fold_tokens(phrase))
    return _CANON.get(folded, folded)


_CANON = {" ".join(fold_tokens(k)): " ".join(fold_tokens(v)) for k, v in _SYNONYMS.items()}
_AMBIGUOUS_FOLDED = {" ".join(fold_tokens(w)) for w in _AMBIGUOUS}


def parse(grapes: str) -> frozenset[str]:
    """Сорта из поля карточки: свёрнутые канонические фразы."""
    phrases = set()
    for part in _SPLIT.split(str(grapes or "")):
        phrase = _canonical(part)
        if phrase and not phrase.startswith(_GENERIC_MARKER) and _GENERIC_MARKER not in phrase:
            phrases.add(phrase)
    return frozenset(phrases)


def _same(a: str, b: str) -> bool:
    if a == b:
        return True
    if len(a) < FUZZY_MIN_LEN or abs(len(a) - len(b)) > 1:
        return False
    return fuzz.ratio(a, b) >= FUZZY_RATIO


@dataclass
class GrapeVocabulary:
    """Словарь сортов каталога и поиск их в тексте."""

    phrases: dict[str, tuple[str, ...]] = field(default_factory=dict)  # фраза -> слова

    @classmethod
    def from_payloads(cls, payloads: list[dict]) -> "GrapeVocabulary":
        winery_words = {
            token for p in payloads for token in fold_tokens(str(p.get("winery") or ""))
        }
        phrases: dict[str, tuple[str, ...]] = {}
        sources = [parse(payload.get("grapes", "")) for payload in payloads]
        sources.append(frozenset(_canonical(name) for name in _EXTRA))
        for found in sources:
            for phrase in found:
                words = tuple(phrase.split())
                single = len(words) == 1
                if single and (phrase in _AMBIGUOUS_FOLDED or phrase in winery_words):
                    continue
                if single and len(phrase) < 4:
                    continue
                phrases[phrase] = words
        for head in (_canonical(name) for name in _HEADS):
            if head not in winery_words and any(
                len(words) > 1 and words[0] == head for words in phrases.values()
            ):
                phrases.setdefault(head, (head,))
        # Синонимы ищутся на этикетке своим написанием («SHIRAZ»), а находятся каноном.
        for synonym, canonical in _CANON.items():
            if canonical in phrases:
                phrases[synonym] = tuple(synonym.split())
        return cls(phrases)

    def in_name(self, name_tokens: list[str]) -> frozenset[str]:
        """Сорт по словам названия — для карточек, у которых поле `grapes` общее («Белые сорта
        винограда»): «Мускатель белый» — мускат. Иначе этикетка «МУСКАТ» давала +1 соседу
        «Мускат Белый Южнобережный» и ничего верной карточке (разбор ошибок v8). Слово
        названия совпадает с однословным сортом целиком или начинается с него."""
        found = set()
        for token in name_tokens:
            for phrase, parts in self.phrases.items():
                if len(parts) == 1 and len(phrase) >= 5 and token.startswith(phrase):
                    found.add(_CANON.get(phrase, phrase))
        return frozenset(found)

    def find(self, text: str) -> frozenset[str]:
        """Сорта, названные в тексте: жадно, самая длинная фраза с каждой позиции."""
        found: set[str] = set()
        by_length = sorted(self.phrases.items(), key=lambda kv: -len(kv[1]))
        seen_variants = {fold(text), fold(text.replace("-", " "))}
        for variant in seen_variants:
            words = [w for w in variant.split() if len(w) > 1]
            i = 0
            while i < len(words):
                for phrase, parts in by_length:
                    span = words[i : i + len(parts)]
                    if len(span) == len(parts) and all(
                        _same(w, p) for w, p in zip(span, parts, strict=True)
                    ):
                        found.add(_CANON.get(phrase, phrase))
                        i += len(parts)
                        break
                else:
                    i += 1
        return frozenset(found)


def compare(label: frozenset[str] | None, card: frozenset[str] | None) -> int:
    """+1 — хотя бы один сорт этикетки есть у карточки, −1 — ни одного, 0 — не с чем сравнить.

    «Рислинг» на этикетке и «Рислинг Рейнский» в карточке — совпадение: одна фраза вложена
    в другую. Купаж на этикетке, названный одним сортом, тоже совпадает с карточкой купажа.
    """
    if not label or not card:
        return 0
    for a in label:
        words_a = set(a.split())
        for b in card:
            words_b = set(b.split())
            if words_a <= words_b or words_b <= words_a:
                return 1
    return -1
