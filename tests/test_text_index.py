"""Текстовый индекс на мини-каталоге: n-граммы, различающие слова, сигналы для близнецов.

Каталог платформы в тесты не берём — он большой и меняется. Шесть карточек ниже повторяют
ровно ту ситуацию, ради которой индекс переписан: линейка Массандры с одной этикеткой на всех,
латинское название против кириллического, карточка-одиночка без сиблингов.
"""

import pytest

from wine_scanner.ocr import TextIndex, token_hits
from wine_scanner.ocr.attributes import Attributes

PAYLOADS = [
    (
        "krasnyy-alushta",
        {"name": "Портвейн красный Алушта", "winery": "Массандра", "category": "Красное"},
    ),
    (
        "belyy-alushta",
        {"name": "Портвейн белый Алушта", "winery": "Массандра", "category": "Белое"},
    ),
    (
        "belyy-krymskiy",
        {"name": "Портвейн белый крымский", "winery": "Массандра", "category": "Белое"},
    ),
    ("muskatel-belyy", {"name": "Мускатель белый", "winery": "Массандра", "category": "Белое"}),
    (
        "shato-taman",
        {"name": "Шато Тамань Руж 2023", "winery": "Кубань-Вино", "category": "Красное"},
    ),
    ("lone", {"name": "Гравити", "winery": "Alma Valley", "category": "Белое"}),
]


@pytest.fixture(scope="module")
def index() -> TextIndex:
    ids = [item_id for item_id, _ in PAYLOADS]
    return TextIndex.from_payloads(ids, [p for _, p in PAYLOADS])


def ranking(index: TextIndex, text: str) -> list[str]:
    return [hit.item_id for hit in index.search(text, top_k=10)]


def test_one_letter_ocr_error_still_ranks_first(index):
    """«МУСКАТЕАЬ» — реальная ошибка OCR на кадре организаторов."""
    assert ranking(index, "МАССАНДРА МУСКАТЕАЬ БЕЛЫЙ")[0] == "muskatel-belyy"


def test_latin_label_finds_cyrillic_card(index):
    assert ranking(index, "Chateau Tamagne Rouge 2023")[0] == "shato-taman"


def test_twins_ordered_by_their_own_words(index):
    order = ranking(index, "ПОРТВЕЙН КРАСНЫЙ АЛУШТА")
    assert order[0] == "krasnyy-alushta"
    assert order[1] == "belyy-alushta"


def test_disc_tokens_are_rare_within_winery(index):
    disc = index.disc_tokens
    # «портвейн» есть у трёх из четырёх Массандр — не различает; «алушта» — да. Слово
    # цвета «красный» в различающие не идёт: цвет сравнивают атрибуты.
    assert "portvein" not in disc["krasnyy-alushta"]
    assert disc["krasnyy-alushta"] == frozenset({"alushta"})
    assert "beli" not in disc["belyy-krymskiy"]
    # У одиночки различать нечего.
    assert disc["lone"] == frozenset()


def test_token_hits_exact_for_short_and_digits():
    assert token_hits(["2023", "shato"], frozenset({"2023"})) == 1.0
    assert token_hits(["2024"], frozenset({"2023"})) == 0.0
    # Длинное слово прощает одну ошибку, но не приставку: «мускат» — не «мускатель».
    assert token_hits(["muskatea"], frozenset({"muskatel"})) == 1.0
    assert token_hits(["muskat"], frozenset({"muskatel"})) == 0.0
    assert token_hits([], frozenset({"x"})) == 0.0


def test_signals_contradict_colour_for_rose_twin(index):
    """Розовый Алушта: слова красного сиблинга подтверждены, цвет — нет."""
    tokens = index.query_tokens("МАССАНДРА ПОРТВЕЙН РОЗОВЫЙ АЛУШТА 2023")
    attrs = index.query_attributes(tokens)
    assert attrs.color == "rose"
    signal = index.signals(tokens, attrs, "krasnyy-alushta")
    assert signal["color_match"] == -1
    assert signal["winery_hit"] == 1
    # «алушта» на этикетке есть — своё слово карточки подтверждено; различает только цвет.
    assert signal["disc_hit"] == 1.0
    assert signal["disc_n"] == len(index.disc_tokens["krasnyy-alushta"]) == 1


def test_signals_neutral_without_colour_word(index):
    tokens = index.query_tokens("МАССАНДРА ПОРТВЕЙН АЛУШТА")
    signal = index.signals(tokens, Attributes(), "krasnyy-alushta")
    assert signal["color_match"] == 0


def test_family_rank_excludes_seen(index):
    scores = index.score_all("Портвейн белый")
    family = index.family_rank("Массандра", scores, exclude={"belyy-alushta"}, limit=2)
    assert "belyy-alushta" not in family
    assert len(family) == 2
    assert family[0] == "belyy-krymskiy"


def test_bm25_scorer_still_works():
    ids = [item_id for item_id, _ in PAYLOADS]
    index = TextIndex.from_payloads(ids, [p for _, p in PAYLOADS], scorer="bm25")
    assert ranking(index, "Портвейн красный Алушта")[0] == "krasnyy-alushta"


def test_index_without_payloads_degrades_gracefully():
    index = TextIndex(["a", "b"], ["Chateau Alpha", "Chateau Beta"])
    assert ranking(index, "Alpha")[0] == "a"
    signal = index.signals(["alfa"], Attributes(), "a")
    assert signal == {
        "disc_hit": 0.0,
        "disc_n": 0,
        "name_cover": 0.0,
        "winery_hit": 0,
        "color_match": 0,
        "style_match": 0,
        "grape_match": 0,
    }
