"""Дозаполнение атрибутов карточки и разрывы по картинке против неопровергнутых соперников."""

from wine_scanner.decide.features import PairFeatures, derive, text_refutes
from wine_scanner.ocr import TextIndex
from wine_scanner.ocr.attributes import from_card, sparkling_from_card, sweetness_from_slug
from wine_scanner.ocr.grapes import GrapeVocabulary


def test_sweetness_from_slug_only_after_colour_or_last():
    assert sweetness_from_slug("valeriy-zaharin-risling-reynskiy-beloe-polusladkoe-125") == (
        "semisweet"
    )
    assert sweetness_from_slug("agora-vintage-blanc-de-blancs-extra-brut") is None
    assert sweetness_from_slug("fanagoriya-ekstra-bryut-beloe-rislinng") is None
    assert sweetness_from_slug("kokur-beloe-ekstra-bryut") == "extra_brut"


def test_sparkling_from_slug_and_description():
    assert sparkling_from_card({"slug": "agora-rosa-viva-beloe-bryut"})
    assert sparkling_from_card({"slug": "czimlyanskoe-beloe-polusladkoe",
                                "description": "Вкус игристого вина приятный"})
    assert not sparkling_from_card({"slug": "kokur-beloe-suhoe", "description": "Вкус свежий"})


def test_unstated_sparkling_stays_unknown():
    payload = {"name": "Цимлянское белое полусладкое", "category": "Белое", "grapes": ""}
    assert from_card(payload, ["tsimlianskoe"]).sparkling is None


def test_generic_grapes_taken_from_name():
    payloads = [
        {"name": "Мускатель белый", "grapes": "Белые сорта винограда", "winery": "Массандра"},
        {"name": "Мускат Белый Южнобережный", "grapes": "Мускат", "winery": "Массандра"},
    ]
    index = TextIndex.from_payloads(["muskatel", "muskat"], payloads)
    assert index.cards["muskatel"].attrs.grapes == {"muskat"}
    vocab = GrapeVocabulary.from_payloads(payloads)
    assert vocab.find("МАССАНДРА МУСКАТЕЛЬ ЧЕРНЫЙ") == frozenset()


def row(item_id, inliers, **kw):
    base = dict(item_id=item_id, vis_score=0.5, vis_rank=0, txt_score=0.5, txt_rank=0,
                rrf_rank=0, matches=inliers, inliers=inliers, inlier_ratio=0.5,
                reproj_error=1.0, homography_ok=1, ocr_lines=10, ocr_conf=0.9)
    return PairFeatures(**{**base, **kw})


def test_gap_ignores_rival_refuted_by_text():
    true = row("red", 390, color_match=1, disc_hit=1.0, disc_n=1)
    white = row("white", 520, color_match=-1, disc_hit=0.0, disc_n=1)
    assert text_refutes(white, true)
    assert not text_refutes(true, white)
    old = {r["item_id"]: r for r in derive([true, white])}
    new = {r["item_id"]: r for r in derive([true, white], rivals="text")}
    assert old["red"]["inliers_gap"] == 130 and old["red"]["inliers_margin"] == -130
    assert new["red"]["inliers_gap"] == 0 and new["red"]["inliers_margin"] == 390


def test_name_colour_counts_alongside_category():
    from wine_scanner.ocr.attributes import Attributes, compare_color

    card = from_card({"name": "Мускат Оранж", "category": "Белое"}, ["muskat", "oranzh"])
    assert card.color == "white" and card.colors == {"white", "orange"}
    assert compare_color("orange", card) == 1
    assert compare_color("red", card) == -1
    assert compare_color("red", Attributes()) == 0


def test_genitive_colour_is_not_wine_colour():
    from wine_scanner.ocr.attributes import from_tokens
    from wine_scanner.ocr.normalize import fold_tokens

    assert from_tokens(fold_tokens("МУСКАТ БЕЛЫЙ КРАСНОГО КАМНЯ")).color == "white"
    assert from_tokens(fold_tokens("МАССАНДРА КРАСНОГ")).color is None
    assert from_tokens(fold_tokens("КРАСНО")).color == "red"  # обрезанное «красное»


def test_truncated_sweetness_is_not_contradiction():
    from wine_scanner.ocr.attributes import compare_sweetness

    assert compare_sweetness("dry", "semidry") == 0  # OCR потерял «ПОЛУ»
    assert compare_sweetness("brut", "extra_brut") == 0
    assert compare_sweetness("semidry", "dry") == -1  # лишних букв OCR не придумывает
    assert compare_sweetness("dry", "semisweet") == -1


def test_fuzzy_sweetness_only_for_long_words():
    from wine_scanner.ocr.attributes import from_tokens
    from wine_scanner.ocr.normalize import fold_tokens

    assert from_tokens(fold_tokens("Полусладкок")).sweetness == "semisweet"
    assert from_tokens(fold_tokens("посусладкоев")).sweetness == "semisweet"
    assert from_tokens(fold_tokens("сухой")).sweetness == "dry"
    assert from_tokens(fold_tokens("сукое")).sweetness is None


def test_extra_brut_in_card_name_keeps_word_order():
    payloads = [
        {"name": "Victor Dravigny. Extra Brut", "category": "Белое", "winery": "Абрау"},
        {"name": "Пино Нуар - Саперави", "grapes": "Пино Нуар", "winery": "Абрау"},
        {"name": "Саперави", "grapes": "Саперави", "winery": "Абрау"},  # сорт в словаре
    ]
    index = TextIndex.from_payloads(["vd", "pns", "s"], payloads)
    assert index.cards["vd"].attrs.sweetness == "extra_brut"
    assert "saperavi" in index.cards["pns"].attrs.grapes


def test_label_twins_only_when_text_cannot_tell():
    from wine_scanner.decide.guard import label_twins
    from wine_scanner.pipeline import Candidate

    def cand(item_id, p, **f):
        base = {"color_match": 0, "style_match": 0, "grape_match": 0, "disc_hit": 0.0}
        return Candidate(item_id, p, {}, {**base, **f})

    family = {"riesling": "Криница", "viognier": "Криница", "other": "Чужая"}
    leader = cand("viognier", 0.62)
    assert label_twins([leader, cand("riesling", 0.4), cand("other", 0.5)], family) == ["riesling"]
    assert label_twins([leader, cand("riesling", 0.2)], family) == []  # модель уверена в лидере
    assert label_twins([leader, cand("riesling", 0.4, grape_match=1)], family) == []
    assert label_twins([leader, cand("riesling", 0.4, disc_hit=0.5)], family) == []


def test_contra_guard_refuses_only_on_contradiction():
    from wine_scanner.decide.guard import twin_guard

    base = {"ocr_lines": 5, "disc_n": 2, "disc_hit": 0.5, "winery_hit": 1}
    assert twin_guard({**base, "style_match": -1}, "contra") == "contra"
    assert twin_guard({**base, "grape_match": -1}, "contra") == "contra"
    # прочие улики близнеца — предупреждение (сканер не отменяет ответ)
    assert twin_guard({**base, "disc_hit": 0.0}, "contra") == "twin"
    assert twin_guard({**base, "style_match": -1, "ocr_lines": 2}, "contra") is None
