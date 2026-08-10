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
