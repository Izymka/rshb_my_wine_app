"""Атрибуты вина из слов: цвет, сладость, игристость.

Зачем это отдельным блоком. Половина близнецов каталога различается ровно одним словом
такого рода: «Русское Игристое полусладкое» и «полусухое», «Par Amour» белое, красное и
розовое при одинаковом имени. Слово цвета на этикетке почти всегда крупное и читается даже
плохим OCR, а у карточки цвет лежит в поле `category` — самом надёжном поле каталога.
Сравнение по этим осям даёт признак, который не зависит от того, насколько похожи картинки.

Работает на свёрнутых словах (`normalize.fold_tokens`): русские и иностранные формы к этому
моменту уже сведены псевдонимами («rouge» -> «krasnoe»), остаётся снять окончания —
«красный», «красное», «красная» — это один признак.
"""

from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein

from .grapes import parse
from .normalize import fold_phonetic, normalize


def _fold(word: str) -> str:
    return fold_phonetic(normalize(word))


# Оси и значения. Порядок значений внутри оси не важен: сравнение — на равенство.
COLOR = "color"
SWEETNESS = "sweetness"
SPARKLING = "sparkling"

_SWEETNESS = {
    _fold(word): value
    for word, value in {
        "брют": "brut",
        "сухое": "dry",
        "сухой": "dry",
        "сухая": "dry",
        "полусухое": "semidry",
        "полусладкое": "semisweet",
        "сладкое": "sweet",
        "десертное": "sweet",
        "ликёрное": "sweet",
    }.items()
}
# Нечёткое чтение слова сладости: «Полусладкок», «посусладкоев» — мелкий шрифт с одной-двумя
# ошибками. Только длинные слова: у «сухое» и «сладкое» одна буква превращает их друг в друга
# или в чужое слово, а «полусухое» от «полусладкого» отстоит на четыре правки.
_FUZZY_MIN_LEN = 8
_FUZZY_SWEETNESS = {form: value for form, value in _SWEETNESS.items() if len(form) >= 8}


def _sweetness(token: str) -> str | None:
    if token in _SWEETNESS:
        return _SWEETNESS[token]
    if len(token) < _FUZZY_MIN_LEN:
        return None
    limit = 1 if len(token) < 10 else 2
    for form, value in _FUZZY_SWEETNESS.items():
        if Levenshtein.distance(token, form, score_cutoff=limit) <= limit:
            return value
    return None


_SPARKLING = {_fold(w) for w in ("игристое", "шампанское", "frizzante", "spumante", "petnat")}
# Креплёные: для поиска аналогов портвейну нужен портвейн, а не сухое красное того же сорта.
_FORTIFIED = {
    _fold(w) for w in ("портвейн", "мадера", "херес", "кагор", "креплёное", "ликёрное", "марсала")
}
# «белых сортов винограда» — не цвет вина. Слово цвета перед «sort…» не считается.
_GRAPE_MARKER = _fold("сорт")
_EXTRA = _fold("экстра")
# «Красная Горка», «Белая Поляна» — место, а не цвет: вино среднего рода, и прилагательное
# женского рода перед таким словом говорит о названии урочища (фото организаторов 013,
# «Шардоне» Галицкого белое, а «КРАСНАЯ ГОРКА» спорило с ним). Сравнение по началу слова.
_PLACE_NOUNS = tuple(
    _fold(w)
    for w in ("горк", "гор", "полян", "балк", "долин", "скал", "звезд", "слобод", "рек", "крепост")
)
_FEMININE_COLORS = {_fold(w) for w in ("красная", "белая", "розовая")}

# Формы перечислены явно, а не по основе: иначе «belmas» и «krasnostop» превратятся в цвет.
# Записаны по-русски и сворачиваются при импорте той же функцией, что и текст этикетки.
# Родительного падежа единственного числа здесь нет: «Мускат белый Красного Камня» — белое,
# а «КРАСНОГО» на его этикетке спорило с верной карточкой (разбор 20.10).
_COLOR_FORMS = {
    "red": {_fold(w) for w in ("красное", "красный", "красная", "красных", "красные")},
    "rose": {_fold(w) for w in ("розовое", "розовый", "розовая", "розовых", "розовые", "розе")},
    "orange": {_fold(w) for w in ("оранжевое", "оранжевый", "оранж", "оранжевых")},
    "white": {_fold(w) for w in ("белое", "белый", "белая", "белых", "белые")},
}
_NOMINATIVE = tuple(form for forms in _COLOR_FORMS.values() for form in forms)
_GENITIVE = tuple(_fold(w) for w in ("красного", "розового", "белого", "оранжевого"))


# Основы для обрезанных слов: OCR на краю этикетки отдаёт «РОЗОВ», «КРАСН». Основа должна
# быть длинной, чтобы не зацепить «Красностоп» (сорт) и «Бельбек» (винодельня).
_COLOR_STEMS = {
    "rose": (_fold("розов"),),
    "red": (_fold("красн"),),
    "orange": (_fold("оранж"),),
}
_STEM_EXCLUDE = (_fold("красностоп"),)


def _color(token: str) -> str | None:
    for value, forms in _COLOR_FORMS.items():
        if token in forms:
            return value
    if token.startswith(_STEM_EXCLUDE):
        return None
    # «КРАСНОГ» — обрезанное «красного»: начало родительного падежа, с которого не начинается
    # ни одна форма цвета.
    if any(g.startswith(token) and len(token) > len(g) - 3 for g in _GENITIVE) and not any(
        form.startswith(token) for form in _NOMINATIVE
    ):
        return None
    for value, stems in _COLOR_STEMS.items():
        if any(token.startswith(stem) and len(token) <= len(stem) + 3 for stem in stems):
            return value
    return None


@dataclass(frozen=True)
class Attributes:
    color: str | None = None
    sweetness: str | None = None
    sparkling: bool | None = None
    fortified: bool | None = None
    # Сорта (ocr/grapes.py): у этикетки — найденные в тексте, у карточки — из поля `grapes`.
    grapes: frozenset[str] = frozenset()
    # Только у карточки: все цвета, которые она о себе говорит, — категория и слово цвета в
    # названии. Они расходятся у 27 карточек: «Мускат Оранж» записан белым, «Фанагория.
    # Полусладкое розовое» — красным, «Красная Шапочка Серый Волк» — белым. Название
    # напечатано на этикетке, поэтому его цвет не должен противоречить этой же карточке.
    colors: frozenset[str] = frozenset()

    def as_dict(self) -> dict:
        return {
            "color": self.color,
            "sweetness": self.sweetness,
            "sparkling": self.sparkling,
            "fortified": self.fortified,
            "grapes": sorted(self.grapes),
        }


def from_tokens(tokens: list[str]) -> Attributes:
    """Атрибуты из свёрнутых слов этикетки или названия.

    Если на одной оси встретились разные значения (этикетка перечисляет линейку целиком),
    ось остаётся пустой: противоречивое свидетельство хуже отсутствующего.
    """
    colors: set[str] = set()
    sweetness: set[str] = set()
    sparkling = False
    fortified = False
    for i, token in enumerate(tokens):
        if token in _FORTIFIED:
            fortified = True
        color = _color(token)
        if color:
            following = tokens[i + 1] if i + 1 < len(tokens) else ""
            place = token in _FEMININE_COLORS and following.startswith(_PLACE_NOUNS)
            if not following.startswith(_GRAPE_MARKER) and not place:
                colors.add(color)
        value = _sweetness(token)
        if value:
            if value == "brut" and i > 0 and tokens[i - 1] == _EXTRA:
                value = "extra_brut"
            sweetness.add(value)
        if token in _SPARKLING:
            sparkling = True
    return Attributes(
        color=colors.pop() if len(colors) == 1 else None,
        sweetness=sweetness.pop() if len(sweetness) == 1 else None,
        # Креплёное не бывает игристым: здесь «нет» — знание, а не отсутствие слова.
        sparkling=True if sparkling else (False if fortified else None),
        fortified=fortified or None,
    )


_CATEGORY_COLOR = {"Белое": "white", "Красное": "red", "Розовое": "rose", "Оранжевое": "orange"}


_SLUG_SWEETNESS = {
    "suhoe": "dry",
    "polusuhoe": "semidry",
    "polusladkoe": "semisweet",
    "sladkoe": "sweet",
    "desertnoe": "sweet",
    "bryut": "brut",
}
_SLUG_COLORS = {"beloe", "krasnoe", "rozovoe", "oranzhevoe"}
# Игристое по адресу и описанию карточки. «Брют» бывает только у игристых.
_SLUG_SPARKLING = {"igristoe", "shampanskoe", "bryut", "prosekko", "kava", "asti", "kremant"}
_DESCRIPTION_SPARKLING = ("игрист", "шампанск", "перлаж", "пузырьк")


def sweetness_from_slug(slug: str) -> str | None:
    """Сладость из адреса карточки на сайте: «…-risling-reynskiy-beloe-polusladkoe-125».

    Сайт пишет в конце адреса цвет и сладость. Слово сладости берётся, только если оно стоит
    сразу после слова цвета или последним (без чисел объёма): иначе «Сухое» из названия
    вина стало бы сладостью. Разные значения в одном адресе — неизвестно.
    """
    words = [w for w in str(slug or "").split("-") if w and not w.isdigit()]
    found = set()
    for i, word in enumerate(words):
        if word in _SLUG_SWEETNESS and (
            i == len(words) - 1 or (i > 0 and words[i - 1] in _SLUG_COLORS)
        ):
            value = _SLUG_SWEETNESS[word]
            if value == "brut" and i > 0 and words[i - 1] == "ekstra":
                value = "extra_brut"
            found.add(value)
    return found.pop() if len(found) == 1 else None


def sparkling_from_card(payload: dict) -> bool:
    """Игристое ли вино по адресу и описанию: у «Цимлянского белого полусладкого» слова
    «игристое» нет в названии, но описание начинается с «Вкус игристого вина»."""
    words = set(str(payload.get("slug") or "").split("-"))
    if words & _SLUG_SPARKLING:
        return True
    description = str(payload.get("description") or "").lower()
    return any(stem in description for stem in _DESCRIPTION_SPARKLING)


def from_card(payload: dict, name_tokens: list[str], grapes_from_name=None) -> Attributes:
    """Атрибуты карточки: цвет — из `category`, остальное — из слов названия, а чего нет в
    названии — из адреса и описания.

    Неполная карточка проигрывает соседу просто тем, что о ней меньше известно: у красного
    «Цимлянского» игристость названа и даёт +1, у белого — нет (разбор ошибок v8). Поэтому:
    - сладость, которой нет в названии, берётся из адреса («…-beloe-polusladkoe-125»);
      название важнее адреса: в адресах сайта бывают ошибки;
    - игристость — из названия, адреса или описания; если нигде не сказано — неизвестно, а
      не «тихое»: карточка игристого без описания иначе спорила бы с этикеткой «ИГРИСТОЕ»;
    - сорт, если поле `grapes` общее («Белые сорта винограда»), — из слов названия
      (`grapes_from_name`: «Мускатель» — мускат).
    """
    guessed = from_tokens(name_tokens)
    category = _CATEGORY_COLOR.get(str(payload.get("category", "")).strip())
    color = category or guessed.color
    sweetness = guessed.sweetness or sweetness_from_slug(payload.get("slug", ""))
    if guessed.fortified:
        sparkling = False
    else:
        stated = guessed.sparkling or sweetness in ("brut", "extra_brut")
        sparkling = True if stated or sparkling_from_card(payload) else None
    # Сорта из названия добавляются и к заполненному полю: у «Пино Нуар - Саперави» в поле
    # записан только Пино Нуар, и «САПЕРАВИ» на этикетке спорил с верной карточкой.
    grapes = parse(payload.get("grapes", ""))
    if grapes_from_name is not None:
        grapes = grapes | grapes_from_name(name_tokens)
    return Attributes(
        color=color,
        sweetness=sweetness,
        sparkling=sparkling,
        fortified=guessed.fortified,
        grapes=grapes,
        colors=frozenset(c for c in (category, guessed.color) if c),
    )


def compare_axis(a, b) -> int:
    """+1 совпало, −1 противоречит, 0 — хотя бы с одной стороны неизвестно."""
    if a is None or b is None:
        return 0
    return 1 if a == b else -1


# Этикетка назвала часть слова карточки: OCR потерял «ПОЛУ» или «ЭКСТРА». Это не
# противоречие, а неполное чтение. Обратное («полусухое» на этикетке, «сухое» у карточки)
# остаётся противоречием: лишних букв OCR не придумывает.
_TRUNCATED = {("dry", "semidry"), ("sweet", "semisweet"), ("brut", "extra_brut")}


def compare_sweetness(label: str | None, card: str | None) -> int:
    if (label, card) in _TRUNCATED:
        return 0
    return compare_axis(label, card)


def compare_color(label: str | None, card: Attributes) -> int:
    """Цвет этикетки против всех цветов карточки (`colors`), а если их нет — против `color`."""
    known = card.colors or ({card.color} if card.color else set())
    if label is None or not known:
        return 0
    return 1 if label in known else -1


def compare(query: Attributes, card: Attributes) -> tuple[int, int]:
    """(цвет, стиль). Стиль — сладость и игристость вместе: −1 при любом противоречии,
    +1 если хоть одна ось совпала и ни одна не противоречит, иначе 0."""
    color = compare_color(query.color, card)
    axes = [compare_sweetness(query.sweetness, card.sweetness)]
    # Игристость сравнивается только когда этикетка сама сказала «игристое»: отсутствие
    # слова на этикетке тихого вина — норма, а не свидетельство.
    if query.sparkling:
        axes.append(compare_axis(query.sparkling, card.sparkling))
    if -1 in axes:
        style = -1
    elif 1 in axes:
        style = 1
    else:
        style = 0
    return color, style


def is_attribute_word(token: str) -> bool:
    """Слово цвета, сладости или игристости — то, что сравнивают атрибуты, а не название."""
    return (
        _color(token) is not None
        or _sweetness(token) is not None
        or token in _SPARKLING
        or token == _EXTRA
    )
