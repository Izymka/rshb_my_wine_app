"""Приведение текста к общему виду для сравнения между алфавитами.

Центральная проблема ветки. Российское вино может быть подписано тремя способами:
кириллицей («Фанагория»), латиницей («Fanagoria») или по-французски («Château Tamagne»).
Причём на этикетке один вариант, а в карточке каталога вполне может оказаться другой.
Сравнивать такие строки посимвольно бессмысленно.

Решение: всё переводим в одну форму — латиница без диакритики, нижний регистр, без пунктуации.
После этого «Фанагория» и «Fanagoria» превращаются в fanagoriya и fanagoria, а это уже
близкие строки, которые нечёткое сравнение уверенно сводит вместе.
"""

import re
import unicodedata

# Упрощённая транслитерация по ГОСТ. Возвращает то, как название обычно пишут латиницей
# на экспортных этикетках, а не строгий стандарт с диакритикой.
CYRILLIC_TO_LATIN = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "i", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}

# Буквы, которые в кириллице и латинице выглядят одинаково. OCR их путает постоянно:
# «МОСКВА» легко прочитывается как латинское «MOCKBA». Складывание в кириллицу с последующей
# транслитерацией чинит такие случаи.
LATIN_TO_CYRILLIC_HOMOGLYPHS = {
    "a": "а", "b": "в", "c": "с", "e": "е", "h": "н", "k": "к", "m": "м",
    "o": "о", "p": "р", "t": "т", "x": "х", "y": "у",
}

_NON_WORD = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACES = re.compile(r"\s+")


def strip_accents(text: str) -> str:
    """Убрать диакритику: château -> chateau, rosé -> rose.

    Нужно и для французских этикеток, и потому что OCR диакритику часто теряет сам —
    так обе стороны сравнения приходят к одному виду.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def transliterate(text: str) -> str:
    return "".join(CYRILLIC_TO_LATIN.get(ch, ch) for ch in text)


def normalize(text: str) -> str:
    """Общая форма: латиница, нижний регистр, без диакритики и пунктуации."""
    text = strip_accents(text.lower())
    text = transliterate(text)
    text = _NON_WORD.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def fold_homoglyphs(text: str) -> str:
    """Считать латинские буквы кириллическими там, где они неотличимы на вид."""
    lowered = text.lower()
    return "".join(LATIN_TO_CYRILLIC_HOMOGLYPHS.get(ch, ch) for ch in lowered)


def variants(text: str) -> list[str]:
    """Нормализованные варианты строки, по которым имеет смысл искать.

    Второй вариант нужен на случай, когда OCR прочитал кириллицу как латиницу. Сравниваем
    по обоим и берём лучшее совпадение: лишний вариант ничего не портит, а пропущенный
    стоит промаха.
    """
    direct = normalize(text)
    folded = normalize(fold_homoglyphs(text))
    return [direct] if folded == direct else [direct, folded]


def tokens(text: str) -> list[str]:
    """Токены для BM25. Однобуквенные выбрасываем — это чаще всего мусор распознавания."""
    return [t for t in normalize(text).split() if len(t) > 1]


# --- Фонетическое сворачивание -------------------------------------------------------------
#
# `normalize` сводит алфавиты, но не написания: «Château Tamagne» и «Шато Тамань» после неё —
# `chateau tamagne` и `shato taman`, а это для нечёткого сравнения далёкие строки. Ниже — правила,
# которые сводят французскую и английскую орфографию к тому, как то же слово звучит по-русски
# и, значит, транслитерируется из каталога. Правила не обязаны быть лингвистически точными,
# они обязаны быть одинаковыми для обеих сторон и редко склеивать разные слова. Известная
# потеря: «ч» и «ш» сливаются (`ch` — французское «ш»); на названиях вин это не мешало.
#
# Порядок важен: многобуквенные сочетания раньше однобуквенных, `c` перед гласной раньше
# общего `c`, сдвоенные буквы — в самом конце.
_PHONETIC_RULES = [
    (re.compile(r"shch"), "sh"),
    (re.compile(r"sch"), "sh"),
    (re.compile(r"tch"), "ch"),
    (re.compile(r"ch"), "sh"),
    (re.compile(r"ck"), "k"),
    (re.compile(r"qu"), "k"),
    (re.compile(r"ph"), "f"),
    (re.compile(r"th"), "t"),
    (re.compile(r"kh"), "h"),
    (re.compile(r"c(?=[eiy])"), "s"),
    (re.compile(r"c"), "k"),
    (re.compile(r"eau"), "o"),
    (re.compile(r"au"), "o"),
    (re.compile(r"ou"), "u"),
    (re.compile(r"gne\b"), "n"),
    (re.compile(r"gn"), "n"),
    (re.compile(r"w"), "v"),
    (re.compile(r"y"), "i"),
    (re.compile(r"j"), "i"),
    (re.compile(r"x"), "ks"),
    (re.compile(r"iu"), "u"),
    (re.compile(r"(.)\1+"), r"\1"),
]


def fold_phonetic(latin: str) -> str:
    """Свернуть уже нормализованную латиницу к фонетической записи."""
    for pattern, replacement in _PHONETIC_RULES:
        latin = pattern.sub(replacement, latin)
    return latin


def fold(text: str) -> str:
    """Самая сильная общая форма: normalize -> фонетика -> псевдонимы.

    Этим сравниваются документы каталога и текст с этикетки в текстовом индексе. `normalize`
    оставлена как есть: на неё завязаны тесты и кэши, а здесь нужна форма грубее.
    """
    from .aliases import (
        apply_aliases,  # цикл импортов: aliases сворачивает свои ключи этой же функцией
    )

    return apply_aliases(fold_phonetic(normalize(text)))


def fold_tokens(text: str) -> list[str]:
    """Свёрнутые слова без однобуквенного мусора."""
    return [t for t in fold(text).split() if len(t) > 1]


def folded_variants(text: str) -> list[str]:
    """Свёрнутые варианты запроса: прямой и с гомоглифами, если они различаются."""
    seen: list[str] = []
    for v in variants(text):
        folded = fold(v)
        if folded not in seen:
            seen.append(folded)
    return seen
