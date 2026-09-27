"""Сорта с этикетки: словарь из каталога, поиск в тексте, сравнение с карточкой."""

from types import SimpleNamespace

import pytest

from wine_scanner.decide.guard import sibling_swap
from wine_scanner.decide.judge import consistent_pick
from wine_scanner.ocr import TextIndex
from wine_scanner.ocr.grapes import GrapeVocabulary, compare, parse

PAYLOADS = [
    {"winery": "Мысхако", "name": "Quintessence. Зинфандель", "grapes": "Зинфандель",
     "category": "Розовое"},
    {"winery": "Мысхако", "name": "Quintessence. Semi-Dry Rose", "grapes": "Пино Нуар",
     "category": "Розовое"},
    {"winery": "Мысхако", "name": "Quintessence. Каберне Совиньон Ripasso",
     "grapes": "Каберне Совиньон", "category": "Красное"},
    {"winery": "Криница", "name": "Азюр", "grapes": "Рислинг Рейнский", "category": "Белое"},
    {"winery": "Криница", "name": "Азюр", "grapes": "Вионье", "category": "Белое"},
    {"winery": "Восторг", "name": "Купаж", "grapes": "Восторг, Белые сорта винограда",
     "category": "Белое"},
    {"winery": "Юг", "name": "Совиньон Блан", "grapes": "Совиньон Блан", "category": "Белое"},
    {"winery": "Юг", "name": "Сира", "grapes": "Сира (Шираз)", "category": "Красное"},
    {"winery": "Юг", "name": "Pinot", "grapes": "Пино чёрный (Пино нуар)", "category": "Красное"},
    {"winery": "Юг", "name": "Рислинг", "grapes": "Рислинг", "category": "Белое"},
]
VOCAB = GrapeVocabulary.from_payloads(PAYLOADS)


def test_parse_splits_synonyms_and_drops_generic():
    assert parse("Сира (Шираз)") == {"sira"}
    assert parse("Пино чёрный (Пино нуар) и Мерло") == {"pino nuar", "merlo"}
    assert parse("Белые сорта винограда") == frozenset()


@pytest.mark.parametrize(
    ("text", "found"),
    [
        ("QUINTESSENCE ZINFANDEL SEMI-DRY ROSE", {"zinfandel"}),
        ("RIESLING", {"risling"}),
        ("Sauvignon Blanc 2022", {"sovinon blan"}),  # не «Совиньон» из Каберне Совиньон
        ("Cabernet Sauvignon", {"kaberne sovinon"}),
        ("SHIRAZ", {"sira"}),
        ("ЗИНФАНДЕЛЬ", {"zinfandel"}),
        ("ЗИНФАНДЕАЬ", {"zinfandel"}),  # одна ошибка OCR в длинном слове
        ("Восторг", set()),  # однословный сорт, совпадающий с винодельней
        ("Азюр 2023", set()),
    ],
)
def test_find(text, found):
    assert VOCAB.find(text) == found


def test_compare():
    assert compare(VOCAB.find("RIESLING"), parse("Рислинг Рейнский")) == 1
    assert compare(VOCAB.find("ZINFANDEL"), parse("Пино Нуар")) == -1
    assert compare(VOCAB.find("Cabernet Sauvignon"), parse("Каберне Совиньон, Мерло")) == 1
    assert compare(frozenset(), parse("Мерло")) == 0
    assert compare(VOCAB.find("Syrah"), frozenset()) == 0


@pytest.fixture(scope="module")
def index():
    ids = [f"c{i}" for i in range(len(PAYLOADS))]
    return TextIndex.from_payloads(ids, PAYLOADS)


def test_signals_grape_match(index):
    text = "QUINTESSENCE ZINFANDEL SEMI-DRY ROSE"
    tokens = index.query_tokens(text)
    attrs = index.query_attributes(tokens, text)
    assert index.signals(tokens, attrs, "c0")["grape_match"] == 1
    assert index.signals(tokens, attrs, "c1")["grape_match"] == -1


def cand(item_id, index, text, **extra):
    tokens = index.query_tokens(text)
    signals = index.signals(tokens, index.query_attributes(tokens, text), item_id)
    features = {"ocr_lines": 5, "txt_rank": 1, **signals, **extra}
    return SimpleNamespace(item_id=item_id, features=features, probability=0.5)


def test_sibling_swap_uses_grape_only_when_enabled(index):
    text = "Криница винодельня AZUR RIESLING 2023"
    top, twin = cand("c4", index, text), cand("c3", index, text)
    family = index.family_of
    assert sibling_swap([top, twin], family, grape=False) is None
    assert sibling_swap([top, twin], family, grape=True) == 1


def test_consistent_pick_follows_judge_text(index):
    shown = [SimpleNamespace(item_id=i) for i in ("c1", "c0", "c2")]
    read = "QUINTESSENCE Collection Light ZINFANDEL SEMI-DRY ROSE"
    assert consistent_pick(read, shown, shown[0], index).item_id == "c0"
    # Выбор, с которым текст согласен, не трогаем.
    assert consistent_pick(read, shown, shown[1], index) is None
    # Текст ничего не говорит — ничего не меняем.
    assert consistent_pick("QUINTESSENCE", shown, shown[0], index) is None


def test_consistent_pick_accepts_colour_and_sweetness_when_choice_refuted():
    """Линейка без различающих слов: подтверждение — цвет и сладость."""
    payloads = [
        {"winery": "Цимлянские вина", "name": "Цимлянское белое полусладкое", "grapes": "",
         "category": "Белое"},
        {"winery": "Цимлянские вина", "name": "Игристое красное сладкое Цимлянское", "grapes": "",
         "category": "Красное"},
    ]
    index = TextIndex.from_payloads(["white", "red"], payloads)
    white, red = SimpleNamespace(item_id="white"), SimpleNamespace(item_id="red")
    text = "ИГРИСТОЕ ВИНО ЦИМЛЯНСКОЕ Полусладкое Белое"
    assert consistent_pick(text, [white, red], red, index) is white
    # Выбор не опровергнут, только цвет у соседки совпал — не трогаем.
    assert consistent_pick("ЦИМЛЯНСКОЕ", [white, red], red, index) is None
