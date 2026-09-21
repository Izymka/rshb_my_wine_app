"""Защита от близнеца: табличная проверка трёх режимов."""

import pytest

from wine_scanner.decide.guard import twin_guard

BASE = {
    "ocr_lines": 5,
    "disc_n": 3,
    "disc_hit": 0.67,
    "winery_hit": 1,
    "color_match": 0,
    "style_match": 0,
    "disc_contra": 0.0,
}


def features(**overrides) -> dict:
    return {**BASE, **overrides}


@pytest.mark.parametrize(
    ("case", "mode", "expected"),
    [
        # Розовый Алушта против красной карточки: свои слова частично есть, цвет противоречит.
        (features(color_match=-1), "twin", "twin"),
        (features(color_match=-1), "warn", "twin"),
        (features(color_match=-1), "strict", None),
        (features(color_match=-1), "off", None),
        # Ни одного своего слова, но винодельня подтверждена.
        (features(disc_hit=0.0), "twin", "twin"),
        (features(disc_hit=0.0, winery_hit=0, disc_contra=0.5), "twin", "twin"),
        # Своих слов нет, но семья не подтверждена, а название частично прочитано — модели виднее.
        (features(disc_hit=0.0, winery_hit=0, disc_contra=0.0, name_cover=0.3), "twin", None),
        # Текст не подтверждает вообще ничего — ответ держится на одной геометрии.
        (features(disc_hit=0.0, winery_hit=0, disc_contra=0.0, name_cover=0.0), "twin", "twin"),
        # Обе улики сразу — срабатывает и строгий режим.
        (features(disc_hit=0.0, color_match=-1), "strict", "twin"),
        # Нормальный уверенный ответ.
        (features(color_match=1), "twin", None),
        (features(color_match=1), "warn", None),
        # Этикетка не читается — правилу нечем судить.
        (features(color_match=-1, ocr_lines=1), "twin", None),
        # Карточке нечего различать словами — но цвет противоречит, и этого достаточно
        # («Par Amour» белое против розового на этикетке).
        (features(color_match=-1, disc_n=0, disc_hit=0.0), "twin", "twin"),
        # Нет различающих слов и нет противоречия — модели виднее.
        (features(color_match=0, disc_n=0, disc_hit=0.0), "twin", None),
        # Цимлянское полусухое против карточки «полусладкое»: цвет молчит (белое = белое),
        # свои слова частично есть, различие — только в сладости.
        (features(style_match=-1, color_match=1), "twin", "twin"),
        (features(style_match=-1), "strict", None),
        # Сладость совпала — это подтверждение, а не улика.
        (features(style_match=1), "twin", None),
    ],
)
def test_twin_guard(case, mode, expected):
    assert twin_guard(case, mode) == expected


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        twin_guard(features(), "loose")


# --- Выбор внутри семьи по тексту ---------------------------------------------------------

from types import SimpleNamespace  # noqa: E402

from wine_scanner.decide.guard import sibling_swap, text_evidence  # noqa: E402

FAMILY = {"livadia": "Массандра", "alushta": "Массандра", "gurzuf": "Массандра", "taman": "Кубань"}


def scored(item_id, **f):
    return SimpleNamespace(item_id=item_id, probability=0.5, features={**BASE, **f})


def test_sibling_swap_is_a_veto_not_a_revote():
    """Реальный случай IMG_0261: лидер по геометрии — Ливадия, текст чуть лучше за Алушту.
    Оба подтверждены своими словами — правило молчит, решает модель."""
    top = scored("livadia", disc_hit=0.67, color_match=1, txt_rank=1, name_cover=0.62)
    true = scored("alushta", disc_hit=0.67, color_match=1, txt_rank=0, name_cover=0.63)
    assert sibling_swap([top, true], FAMILY) is None
    assert text_evidence(true.features) > text_evidence(top.features)


def test_sibling_swap_when_text_is_against_leader_and_for_sibling():
    # Лидер без единого своего слова, соседка подтверждена.
    top = scored("gurzuf", disc_hit=0.0, txt_rank=4, name_cover=0.3)
    loud = scored("alushta", disc_hit=0.67, txt_rank=0)
    assert sibling_swap([top, loud], FAMILY) == 1
    # Цвет против лидера, за соседку.
    red_top = scored("alushta", disc_hit=0.67, color_match=-1)
    rose = scored("livadia", disc_hit=0.0, color_match=1)
    assert sibling_swap([red_top, rose], FAMILY) == 1


def test_sibling_swap_needs_readable_label_and_family():
    top = scored("gurzuf", disc_hit=0.0, ocr_lines=1)
    loud = scored("alushta", disc_hit=0.67, txt_rank=0)
    assert sibling_swap([top, loud], FAMILY) is None
    foreign_top = scored("taman", disc_hit=0.0)
    assert sibling_swap([foreign_top, loud], FAMILY) is None


def test_sibling_swap_on_sweetness_contradiction():
    """Лидер — полусладкое, на этикетке «брют»; соседка-брют подтверждена сладостью."""
    top = scored("alushta", disc_hit=0.5, style_match=-1)
    brut = scored("livadia", disc_hit=0.0, style_match=1)
    assert sibling_swap([top, brut], FAMILY) == 1
    # Соседка сама противоречит этикетке по цвету — не замена, даже если лидер не подтверждён.
    wrong_color = scored("livadia", disc_hit=0.67, color_match=-1)
    top_unsupported = scored("alushta", disc_hit=0.0)
    assert sibling_swap([top_unsupported, wrong_color], FAMILY) is None
