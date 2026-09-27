"""Год урожая: извлечение из текста и сравнение с карточкой каталога.

Год — это сотня пикселей на всей этикетке, и глобальный эмбеддинг его не видит: разница
между 2019 и 2021 в векторе тонет полностью. Зато для пользователя это половина ответа —
«Мутон Каде 2022» и «Мутон Каде 2023» стоят на полке рядом и стоят разных денег.

Здесь только текстовая часть работы: вытащить из строки правдоподобные годы и сопоставить
их с карточкой. Геометрия (где именно на кадре искать) живёт в locate.py.

Главная опасность блока — не промах, а уверенная ошибка. Промах мы показываем честно
(«год не распознан, уточните»), а неверный год выглядит как знание и вводит в заблуждение.
Поэтому все правила ниже сдвинуты в сторону отказа: сомнительное отбрасываем.
"""

import ast
import re
from dataclasses import dataclass
from datetime import date

# Год ищем как отдельное четырёхзначное число. Ограничители по краям обязательны: без них
# в «750ML 12%vol 20220» нашёлся бы «2022», а в артикуле — что угодно.
YEAR_RE = re.compile(r"(?<!\d)(1\d{3}|20\d{2})(?!\d)")

# Нижняя граница правдоподобия. На этикетках сплошь и рядом стоит год основания хозяйства
# («Depuis 1852», «Est. 1911»), и он читается ровно так же, как винтаж. Отсекать его по
# положению на этикетке ненадёжно, по величине — надёжно: вино 1980-х в рознице не стоит,
# а «основано в 1975» встречается.
#
# Цена решения: настоящий старый винтаж мы не прочитаем. Для сканера полки магазина это
# правильный размен, для аукционного каталога границу пришлось бы опустить.
OLDEST_VINTAGE = 1990

# Верхняя граница считается от сегодняшнего дня, а не прибита константой: иначе через год
# сканер перестанет узнавать свежий урожай, и виноват будет не он.
FUTURE_MARGIN = 1


@dataclass(frozen=True)
class Reading:
    """Что мы поняли про год.

    `source` важен не меньше самого года: он говорит, откуда взято значение — из общего
    распознавания этикетки или из увеличенного участка по гомографии. Без него нельзя
    ни отладить блок, ни честно посчитать, что именно дало прирост.
    """

    year: int | None = None
    source: str = ""  # "" | "text" | "zoom"
    candidates: tuple[int, ...] = ()

    def __bool__(self) -> bool:
        return self.year is not None


def newest_vintage(today: date | None = None) -> int:
    return (today or date.today()).year + FUTURE_MARGIN


def years(text: str, oldest: int = OLDEST_VINTAGE, newest: int | None = None) -> list[int]:
    """Все правдоподобные годы в строке, в порядке появления и с повторами.

    Повторы не схлопываем намеренно: у нас два читателя OCR, и год, увиденный обоими,
    заслуживает больше доверия, чем год, увиденный одним. Это различие использует `pick`.
    """
    newest = newest_vintage() if newest is None else newest
    return [year for year in map(int, YEAR_RE.findall(text)) if oldest <= year <= newest]


def pick(found: list[int]) -> Reading:
    """Выбрать год из найденных.

    Правило: побеждает самый частый. При ничьей между разными годами отказываемся — на
    этикетке действительно может стоять два числа (год розлива и год основания в допустимом
    диапазоне), и угадывать между ними не на чем. Отказ здесь дешевле ошибки: дальше есть
    увеличение по гомографии, а за ним вопрос пользователю.
    """
    if not found:
        return Reading(candidates=())

    counts: dict[int, int] = {}
    for year in found:
        counts[year] = counts.get(year, 0) + 1

    best = max(counts.values())
    winners = [year for year, count in counts.items() if count == best]
    candidates = tuple(sorted(counts))
    if len(winners) > 1:
        return Reading(candidates=candidates)
    return Reading(year=winners[0], candidates=candidates)


def from_text(text: str, source: str = "text") -> Reading:
    reading = pick(years(text))
    return Reading(reading.year, source if reading.year else "", reading.candidates)


def combine(text: Reading, zoom: Reading) -> Reading:
    """Свести оба пути в один ответ.

    Общее распознавание главнее увеличенного участка, и это не произвол. Текст этикетки
    читается по всему кадру, где год окружён контекстом; участок вырезается по чужой
    геометрии — если кандидат оказался не тем вином или гомография неточна, вырезать могло
    и соседнюю строку. Поэтому увеличение только заполняет пробел, а не спорит.
    """
    if text:
        return text
    return zoom


def catalog_years(payload: dict) -> set[int]:
    """Годы, которые допускает карточка каталога.

    Формата два. Свой набор хранит один год в поле `vintage` — это конкретная бутылка.
    X-Wines хранит список `Vintages` — все годы, в которые выпускалось это вино; там карточка
    описывает линейку целиком. Разница существенная: во втором случае «год совпал» означает
    гораздо меньше, и решающий слой это увидит сам по силе признака.
    """
    found: set[int] = set()

    single = payload.get("vintage")
    if isinstance(single, int | float | str) and str(single).strip():
        try:
            found.add(int(single))
        except ValueError:
            pass

    many = payload.get("vintages")
    if isinstance(many, str):
        # В CSV X-Wines список лежит строкой вида "[2018, 2016, 2015]".
        try:
            many = ast.literal_eval(many)
        except (ValueError, SyntaxError):
            many = []
    if isinstance(many, list | tuple):
        for value in many:
            try:
                found.add(int(value))
            except (TypeError, ValueError):
                continue

    return found


@dataclass(frozen=True)
class Answer:
    """Что показать пользователю про год и надо ли его спросить."""

    year: int | None
    source: str  # "catalog" | "text" | "zoom" | ""
    ask: bool  # показать вопрос «какой год на бутылке?»


def resolve(reading: Reading, payload: dict) -> Answer:
    """Свести прочитанное с этикетки и записанное в каталоге в один ответ.

    Каталог и фотография знают о годе разное, и правила ниже — про то, кому верить.

    Карточка на конкретную бутылку (ровно один год) отвечает сама: год есть в базе, читать
    его с этикетки незачем. Но если мы прочитали другой год, спорить не будем и спросим —
    расхождение означает либо чужую карточку, либо ошибку распознавания, и в обоих случаях
    один тап пользователя стоит дешевле неверного числа на экране.

    Карточка на линейку (список годов) сама не отвечает: выбрать из десяти лет может только
    этикетка. Прочитали — берём, не прочитали — спрашиваем.

    Карточка без года — это либо NV, либо неполные данные. Здесь молчание уместнее вопроса:
    у вина без винтажа пользователю просто нечего ответить.
    """
    known = catalog_years(payload)

    if len(known) == 1:
        year = next(iter(known))
        if reading.year is not None and reading.year != year:
            return Answer(reading.year, reading.source, ask=True)
        return Answer(year, "catalog", ask=False)

    if known:
        if reading.year in known:
            return Answer(reading.year, reading.source, ask=False)
        return Answer(None, "", ask=True)

    if reading.year is not None:
        return Answer(reading.year, reading.source, ask=False)
    return Answer(None, "", ask=False)


def compare(year: int | None, payload: dict, group_years: frozenset[int] = frozenset()) -> int:
    """Признак решающего слоя: +1 год совпал, -1 год у другой карточки того же вина, 0 иначе.

    Правило продукта (27.09.2026): вино, которое отличается от карточки только годом, — то же
    вино. Если год с этикетки в каталоге не заведён ни у одной карточки этой этикетки
    (`group_years` — годы соседей по группе года, см. `vintage_groups`), карточка остаётся
    верным ответом, и год против неё не свидетельствует. Минус — только когда есть карточка
    ровно с прочитанным годом: тогда эта карточка — не та бутылка.
    """
    if year is None:
        return 0
    if year in catalog_years(payload):
        return 1
    return -1 if year in group_years else 0


_YEAR_IN_NAME = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")


def _key_text(value: object) -> str:
    return " ".join(re.findall(r"\w+", str(value or "").lower().replace("ё", "е")))


def vintage_groups(payloads: dict[str, dict]) -> dict[str, frozenset[str]]:
    """Карточки одного вина, которые отличаются только годом: slug -> вся группа.

    Одно вино — та же винодельня, категория и сорта и то же название, если убрать из него год
    («LETO Рислинг 2020 сухое белое» и «... 2024 ...»). Группа нужна, только если год хоть у
    кого-то в ней записан: иначе различать членов по году нечем.
    """
    keys: dict[tuple, list[str]] = {}
    dated: set[str] = set()
    for slug, payload in payloads.items():
        name = str(payload.get("name", ""))
        if _YEAR_IN_NAME.search(name) or catalog_years(payload):
            dated.add(slug)
        key = (
            _key_text(payload.get("winery")),
            _key_text(_YEAR_IN_NAME.sub(" ", name)),
            _key_text(payload.get("category")),
            _key_text(payload.get("grapes")),
        )
        keys.setdefault(key, []).append(slug)
    groups: dict[str, frozenset[str]] = {}
    for members in keys.values():
        if len(members) > 1 and dated.intersection(members):
            group = frozenset(members)
            for slug in members:
                groups[slug] = group
    return groups


def pick_vintage(
    year: int | None, slug: str, groups: dict[str, frozenset[str]], payloads: dict[str, dict]
) -> str:
    """Карточка того же вина с прочитанным годом, если такая есть; иначе исходная."""
    if year is None or year in catalog_years(payloads.get(slug, {})):
        return slug
    for member in sorted(groups.get(slug, ())):
        if year in catalog_years(payloads.get(member, {})):
            return member
    return slug
