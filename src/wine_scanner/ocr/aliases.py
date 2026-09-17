"""Псевдонимы: слова и фразы, которые на этикетке и в каталоге пишутся по-разному.

Фонетическое сворачивание (`normalize.fold_phonetic`) сводит регулярные пары — «Fanagoria» и
«Фанагория», «Chateau» и «Шато». Но часть различий правилами не берётся: «Cabernet Sauvignon»
против «Каберне Совиньон» отличается не написанием, а тем, что русское — уже фонетическая
запись французского, с потерянными буквами; «Rouge» и «Красное» — просто разные слова для
одного признака. Такие пары перечислены здесь в человеческом виде, а сворачиваются при
импорте той же функцией, что и всё остальное, — так таблицу можно читать и проверять глазами,
а в сравнении участвуют гарантированно одинаково свёрнутые формы.

Правило пополнения: каждая новая строка — со своей парой в tests/test_normalize.py.
"""

from .normalize import fold_phonetic, normalize

# Фразы — применяются первыми, пока слова ещё стоят рядом. Слева — как пишут на этикетке,
# справа — как в каталоге платформы.
_PHRASES = {
    "Pinot Noir": "Пино Нуар",
    "Pinot Grigio": "Пино Гриджио",
    "Pinot Gris": "Пино Гри",
    "Pinot Blanc": "Пино Блан",
    "Pinot Franc": "Пино Фран",
    "Cabernet Sauvignon": "Каберне Совиньон",
    "Cabernet Franc": "Каберне Фран",
    "Sauvignon Blanc": "Совиньон Блан",
    "Chateau Tamagne": "Шато Тамань",
    "Golubitskoe Estate": "Поместье Голубицкое",
    "Derbent Wine": "Дербент Вино",
    "Fanagoria Estate": "Фанагория",
    "Semi Sweet": "Полусладкое",
    "Semi Dry": "Полусухое",
    "Demi Sec": "Полусухое",
    "Medium Sweet": "Полусладкое",
    "Medium Dry": "Полусухое",
}

# Отдельные слова.
_WORDS = {
    # тип вина и хозяйство
    "wine": "вино",
    "vin": "вино",
    "winery": "винодельня",
    "estate": "усадьба",
    "domaine": "дом",
    "sparkling": "игристое",
    "sekt": "игристое",
    "port": "портвейн",
    "porto": "портвейн",
    "portwine": "портвейн",
    "sherry": "херес",
    "madeira": "мадера",
    # цвет и сладость
    "rouge": "красное",
    "red": "красное",
    "rosso": "красное",
    "tinto": "красное",
    "blanc": "белое",
    "white": "белое",
    "bianco": "белое",
    "blanco": "белое",
    "rose": "розовое",
    "rosato": "розовое",
    "sec": "сухое",
    "dry": "сухое",
    "seco": "сухое",
    # «sweet» здесь нет намеренно: после схлопывания сдвоенных букв оно совпадает
    # со «svet» из «Новый Свет». Полусладкое ловится фразой «semi sweet» выше.
    "doux": "сладкое",
    "dolce": "сладкое",
    # сорта: французская запись -> русская
    "cabernet": "каберне",
    "merlot": "мерло",
    "pinot": "пино",
    "noir": "нуар",
    "chardonnay": "шардоне",
    "riesling": "рислинг",
    "sauvignon": "совиньон",
    "muscat": "мускат",
    "moscato": "мускат",
    "syrah": "сира",
    "shiraz": "шираз",
    "viognier": "вионье",
    "tempranillo": "темпранильо",
    "sangiovese": "санджовезе",
    "gewurztraminer": "гевюрцтраминер",
    "traminer": "траминер",
    "zweigelt": "цвайгельт",
}


def _fold(text: str) -> str:
    return fold_phonetic(normalize(text))


PHRASE_ALIASES = {_fold(k): _fold(v) for k, v in _PHRASES.items()}
WORD_ALIASES = {_fold(k): _fold(v) for k, v in _WORDS.items()}


def apply_aliases(folded: str) -> str:
    """Заменить известные написания на каталожные. Вход и выход — свёрнутая латиница."""
    for phrase, canon in PHRASE_ALIASES.items():
        if phrase != canon and phrase in folded:
            folded = folded.replace(phrase, canon)
    return " ".join(WORD_ALIASES.get(word, word) for word in folded.split())
