"""Защита от близнеца: правило, которое не даёт ответить соседом по линейке.

Зачем правило рядом с обученной моделью. Решающий слой учится на том, что есть, а живых
кадров близнецов-незнакомцев у нас единицы: «Портвейн Розовый Алушта» снят восемь раз, и
это единственный такой случай в наборе. Модель на восьми кадрах ничего не выучит, а
контрольная выборка организаторов строится ровно на этом. Правило закрывает дыру явно и
проверяемо, и его действие видно в ответе (`guard` в ScanResult).

Проверка на каталоге показала, что напрашивающееся правило «ни одно различающее слово не
подтверждено» не срабатывает: у розового портвейна слова «портвейн» и «алушта» — те же, что
у красного сиблинга, `disc_hit` у красного 2/3. Отличает его только противоречие по цвету:
на этикетке «розовый», в карточке — Красное. Поэтому режим `twin` срабатывает
на любом из двух свидетельств:

- слово цвета с этикетки противоречит карточке (`color_match == -1`);
- слово сладости или игристости противоречит карточке (`style_match == -1`): Цимлянское
  «полусухое» против карточки «полусладкое» — та же линейка, та же этикетка, разница в одном
  слове, и цвет тут не помогает (17.09: три кадра из трёх принимались за полусладкое);
- ни одно различающее слово карточки не найдено, при этом этикетка подтверждает саму
  винодельню или соседа по ней (`disc_hit == 0` и (`winery_hit` или `disc_contra > 0`)).

Все требуют читаемой этикетки (`ocr_lines >= 3`) и карточки, у которой есть что различать
(`disc_n > 0`). Режим `warn` (по умолчанию) оставляет этот сигнал в `guard`, но не
отменяет ответ. Режим `strict` — исходное, более узкое правило (обе улики сразу),
оставлен для сравнения в бенчмарке. `off` — выключено.
"""

import os

GUARD_MODES = ("off", "strict", "twin", "warn")
# ``warn`` exposes twin evidence to the client but keeps the best catalogue answer.
# The closed-world product flow prefers a useful nearest card to a silent refusal.
DEFAULT_MODE = os.environ.get("WINE_GUARD", "warn")
MIN_OCR_LINES = 3


def twin_guard(features: dict, mode: str = DEFAULT_MODE) -> str | None:
    """Вернуть имя сработавшей защиты или None. `features` — строка лучшего кандидата."""
    if mode == "off":
        return None
    if mode not in GUARD_MODES:
        raise ValueError(f"неизвестный режим защиты: {mode}")
    if features.get("ocr_lines", 0) < MIN_OCR_LINES:
        return None

    color_contradicts = features.get("color_match", 0) == -1
    style_contradicts = features.get("style_match", 0) == -1
    has_disc = features.get("disc_n", 0) > 0
    no_own_words = has_disc and features.get("disc_hit", 0.0) == 0.0
    family_confirmed = features.get("winery_hit", 0) == 1 or features.get("disc_contra", 0.0) > 0

    if mode == "strict":
        return "twin" if color_contradicts and no_own_words else None
    # Противоречие по цвету или сладости судит и карточки без различающих слов («Par Amour»
    # белое против розового на этикетке): это единственное, чем они отличаются.
    if color_contradicts or style_contradicts or (no_own_words and family_confirmed):
        return "twin"
    # Этикетка читается, у карточки есть сиблинги, а текст не подтверждает ни одного её
    # слова, ни винодельню: ответ держится на одной геометрии, которая близнецов не различает.
    no_name_or_winery = (
        features.get("name_cover", 0.0) == 0.0 and features.get("winery_hit", 0) == 0
    )
    if no_own_words and no_name_or_winery:
        return "twin"
    return None


# --- Выбор внутри семьи по тексту ------------------------------------------------------------

SIBLING_ENABLED = os.environ.get("WINE_SIBLING", "1") == "1"
# Сорт этикетки против поля `grapes` (grape_match) как ещё одна ось правила — рядом с цветом.
GRAPE_ENABLED = os.environ.get("WINE_GRAPE", "0") == "1"


def text_evidence(features: dict, grape: bool = False) -> tuple:
    """Насколько этикетка подтверждает именно эту карточку, а не соседку по линейке.

    Порядок сравнения — лексикографический: сначала согласие по цвету (самое надёжное слово
    этикетки, и противоречие по нему не перебивается ничем), потом по сладости и игристости,
    потом доля различающих слов с весами редкости внутри винодельни, потом место в текстовой
    ветке, потом покрытие названия. Инлаеры и косинус сюда не
    входят намеренно: внутри одной линейки они измеряют не «то ли это вино», а «насколько
    крупный у карточки эталон» (проверено: лидер по геометрии не совпал с верной картой ни на
    одном из 24 живых кадров).

    `grape` — сорт идёт сразу после цвета: у близнецов по сорту цвет обычно один и тот же.
    """
    txt_rank = features.get("txt_rank", 999)
    grape_axis = (int(features.get("grape_match", 0)),) if grape else ()
    return (
        int(features.get("color_match", 0)),
        *grape_axis,
        int(features.get("style_match", 0)),
        round(float(features.get("disc_hit", 0.0)), 2),
        -min(int(txt_rank), 999),
        round(float(features.get("name_cover", 0.0)), 2),
    )


def sibling_swap(scored: list, family_of: dict, grape: bool | None = None) -> int | None:
    """Индекс кандидата той же семьи, которого текст подтверждает вместо лидера, или None.

    Правило — вето, а не перевыбор: модель выбирает семью надёжно и внутри семьи чаще права,
    чем нет, поэтому переставлять карточки по любому перевесу текстовых улик нельзя (проверено:
    такая версия меняла верные ответы на короткие названия). Перестановка только когда текст
    *против* лидера и *за* соседку: у лидера нет ни одного своего слова на этикетке, а у
    соседки есть, — или цвет (сладость) на этикетке противоречит лидеру и не противоречит
    соседке. С `grape` (по умолчанию WINE_GRAPE) то же самое для сорта.
    """
    if grape is None:
        grape = GRAPE_ENABLED
    if not scored:
        return None
    top = scored[0]
    features = top.features
    if features.get("ocr_lines", 0) < MIN_OCR_LINES:
        return None
    family = family_of.get(top.item_id)
    if not family:
        return None
    top_color = int(features.get("color_match", 0))
    top_style = int(features.get("style_match", 0))
    top_grape = int(features.get("grape_match", 0)) if grape else 0
    top_has_disc = features.get("disc_n", 0) > 0
    top_unsupported = top_has_disc and features.get("disc_hit", 0.0) == 0.0

    best_index, best_evidence = None, None
    for index, candidate in enumerate(scored[1:], start=1):
        if family_of.get(candidate.item_id) != family:
            continue
        f = candidate.features
        color = int(f.get("color_match", 0))
        style = int(f.get("style_match", 0))
        grape_match = int(f.get("grape_match", 0)) if grape else 0
        supported = f.get("disc_hit", 0.0) > 0.0
        if color == -1 or style == -1 or grape_match == -1:
            continue  # соседка сама противоречит этикетке — не кандидат на замену
        against_leader = (
            (top_color == -1 and (supported or color == 1))
            or (top_style == -1 and (supported or style == 1))
            or (top_grape == -1 and (supported or grape_match == 1))
            or (top_unsupported and supported)
        )
        if not against_leader:
            continue
        evidence = text_evidence(f, grape)
        if best_evidence is None or evidence > best_evidence:
            best_index, best_evidence = index, evidence
    return best_index


def label_confirmed(features: dict, rule: dict | None) -> bool:
    """Этикетка подтверждает карточку сама — отвечаем, даже если вероятность ниже порога.

    Правило продукта (27.09.2026): вино той же этикетки другого года или без года в каталоге —
    то же вино, и ответ по нему не должен теряться из-за порога. Решающий слой на таких кадрах
    бывает неуверен (картинка карточки старого года, год мешает сравнению), а текст однозначен:
    текстовая ветка ставит карточку первой, винодельня прочитана, различающие слова или
    название совпадают, и ни цвет, ни сладость, ни сорт, ни год карточке не противоречат.

    Границы (`rule`) подбираются на train и лежат в meta решающего слоя (`label_rule`); без них
    правило выключено.
    """
    if not rule:
        return False
    if any(features.get(name, 0) == -1 for name in ("color_match", "style_match",
                                                     "grape_match", "vintage_match")):
        return False
    return (
        features.get("txt_rank", 999) <= rule.get("max_txt_rank", 0)
        and features.get("winery_hit", 0) == 1
        and (
            features.get("disc_hit", 0.0) >= rule["min_disc_hit"]
            or features.get("name_cover", 0.0) >= rule["min_name_cover"]
        )
    )
