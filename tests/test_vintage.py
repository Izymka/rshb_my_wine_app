"""Проверка блока винтажа: правила чтения года и правила молчания.

Само распознавание здесь не проверяется — для этого есть eval/vintage_benchmark.py на живых
кадрах. Проверяется то, что решает, покажем мы год или промолчим, и перенос участка с эталона
на наш кадр. Обе вещи легко сломать незаметно: ошибка не роняет прогон, а просто начинает
показывать пользователю чужой год.
"""

import numpy as np
import pytest
from PIL import Image

from wine_scanner.ocr import TextLine
from wine_scanner.vintage import (
    catalog_years,
    compare,
    from_text,
    patch,
    pick,
    project,
    reference_region,
    resolve,
    years,
)
from wine_scanner.vintage.locate import MIN_INLIERS


class FakeMatch:
    """Результат сопоставления с заданной геометрией."""

    def __init__(self, homography, inliers=100, ok=True, sizes=((640, 480), (640, 480))):
        self.homography = np.asarray(homography, dtype=np.float64)
        self.inliers = inliers
        self.homography_ok = ok
        self.query_size, self.candidate_size = sizes


IDENTITY = np.eye(3)


def test_year_extracted_from_noisy_text():
    assert years("MOUTON CADET 2022 BORDEAUX") == [2022]
    # Оба читателя OCR видят год — повтор сохраняется, на нём строится уверенность.
    assert years("2021 CHATEAU 2021") == [2021, 2021]


def test_founding_year_is_not_a_vintage():
    """«Основано в 1852» читается ровно как винтаж и обязано отсекаться по величине."""
    assert years("CHATEAU DEPUIS 1852") == []
    assert years("MAISON FONDEE EN 1930 MILLESIME 2019") == [2019]


def test_numbers_around_the_year_do_not_create_one():
    assert years("750ML 12.5%VOL 20220 12022") == []


def test_ambiguous_years_are_refused():
    """Два разных года с равным весом — отказ, а не выбор наугад."""
    assert pick([2019, 2021]).year is None
    assert pick([2021, 2021, 2019]).year == 2021
    assert from_text("BORDEAUX 2019 SUPERIEUR 2021").year is None


def test_catalog_years_read_both_formats():
    assert catalog_years({"vintage": 2022}) == {2022}
    # X-Wines хранит годы линейки строкой.
    assert catalog_years({"vintages": "[2018, 2016]"}) == {2016, 2018}
    assert catalog_years({"vintage": None}) == set()


def test_compare_separates_unknown_from_conflict():
    assert compare(2022, {"vintage": 2022}) == 1
    assert compare(2022, {"vintage": 2023}) == -1
    # Ноль означает «сравнивать не с чем», и это не то же самое, что несовпадение.
    assert compare(2022, {"vintage": None}) == 0
    assert compare(None, {"vintage": 2022}) == 0


def test_reference_region_uses_catalog_to_disambiguate():
    """На эталоне OCR видит два числа; выбрать помогает год из карточки."""
    lines = [
        TextLine("2025", 0.9, "latin", (0.1, 0.1, 0.2, 0.13)),
        TextLine("2023", 0.9, "latin", (0.4, 0.6, 0.6, 0.7)),
    ]
    region = reference_region(lines, {2023})
    assert region is not None
    assert region.year == 2023
    assert region.box == (0.4, 0.6, 0.6, 0.7)


def test_reference_region_refuses_when_catalog_is_silent():
    lines = [
        TextLine("2025", 0.9, "latin", (0.1, 0.1, 0.2, 0.13)),
        TextLine("2023", 0.9, "latin", (0.4, 0.6, 0.6, 0.7)),
    ]
    assert reference_region(lines, set()) is None
    # Единственный найденный год каталогу не противоречит — его берём и без подсказки.
    assert reference_region(lines[1:], set()).year == 2023


def test_projection_moves_the_box():
    """Сдвиг на 64 пикселя вправо в кадре 640 — это 0.1 в долях."""
    shift = [[1, 0, 64], [0, 1, 0], [0, 0, 1]]
    moved = project((0.2, 0.5, 0.3, 0.55), FakeMatch(shift))
    assert moved == pytest.approx((0.3, 0.5, 0.4, 0.55))


def test_projection_refuses_weak_geometry():
    box = (0.2, 0.5, 0.3, 0.55)
    assert project(box, FakeMatch(IDENTITY, inliers=MIN_INLIERS - 1)) is None
    assert project(box, FakeMatch(IDENTITY, ok=False)) is None
    assert project(box, None) is None


def test_projection_refuses_when_box_leaves_the_frame():
    """Уехавший за край участок означает, что гомография врёт, а не что год за кадром."""
    shift = [[1, 0, 600], [0, 1, 0], [0, 0, 1]]
    assert project((0.2, 0.5, 0.3, 0.55), FakeMatch(shift)) is None


def test_projection_refuses_absurdly_large_box():
    """Год занимает сотые доли этикетки; четверть кадра — это уже не год."""
    stretch = [[6, 0, 0], [0, 6, 0], [0, 0, 1]]
    assert project((0.0, 0.0, 0.15, 0.15), FakeMatch(stretch)) is None


def test_patch_is_enlarged_and_greyscale():
    image = Image.new("RGB", (800, 600), "black")
    cut = patch(image, (0.4, 0.5, 0.5, 0.53))
    assert cut is not None
    # Участок ушёл на увеличение: 18 пикселей высоты сами по себе не читаются.
    assert cut.height > 100
    array = np.asarray(cut)
    assert (array[..., 0] == array[..., 1]).all()


def test_answer_prefers_catalog_for_a_single_vintage():
    answer = resolve(from_text("MOUTON CADET 2022"), {"vintage": 2022})
    assert (answer.year, answer.source, answer.ask) == (2022, "catalog", False)
    # Год с этикетки не нужен вовсе, если карточка описывает конкретную бутылку.
    answer = resolve(from_text("MOUTON CADET"), {"vintage": 2022})
    assert (answer.year, answer.source, answer.ask) == (2022, "catalog", False)


def test_answer_asks_when_evidence_disagrees():
    answer = resolve(from_text("MOUTON CADET 2023"), {"vintage": 2022})
    assert answer.ask is True


def test_answer_asks_for_a_line_card_without_a_read_year():
    """Карточка на линейку сама год не назовёт — выбрать может только этикетка."""
    line_card = {"vintages": "[2020, 2019, 2018]"}
    assert resolve(from_text("CHATEAU"), line_card).ask is True
    assert resolve(from_text("CHATEAU 2019"), line_card).year == 2019
    assert resolve(from_text("CHATEAU 2019"), line_card).ask is False


def test_answer_stays_silent_for_wine_without_vintage():
    """У NV-вина спрашивать год бессмысленно: пользователю нечего ответить."""
    answer = resolve(from_text("VANG DALAT CLASSIC"), {"vintage": None})
    assert (answer.year, answer.ask) == (None, False)
