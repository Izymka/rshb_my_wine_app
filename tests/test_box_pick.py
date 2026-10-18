"""Выбор рамки бутылки: расширение частичной рамки («плечи и горлышко») до бутылки целиком."""

from wine_scanner.detect import Box, BoxCropper


class Boxes(BoxCropper):
    """Ступень бутылки без модели — как её настраивает `bottle_detector`."""

    min_area_share = 0.05
    require_center = True
    score_threshold = 0.3

    def __init__(self, boxes, partial_min_score=0.15):
        self.boxes = boxes
        self.partial_min_score = partial_min_score

    def detect(self, image):
        return self.boxes


def test_partial_box_expands_to_whole_bottle():
    # Endemy Рислинг брют, эталон 1200×1200: рамка верха 0.75, рамка всей бутылки 0.20.
    top = Box(435, 3, 761, 851, 0.75)
    whole = Box(431, -1, 764, 1195, 0.20)
    neck = Box(544, 77, 651, 166, 0.07)
    assert Boxes([top, whole, neck]).pick([top, whole, neck], (1200, 1200)) is whole


def test_tallest_is_not_needed_most_confident_wins():
    # Alma Valley Мерло: две полные рамки кандидатами — берётся более уверенная.
    top = Box(96, 58, 394, 651, 0.50)
    whole = Box(96, 56, 396, 1274, 0.26)
    whole_weak = Box(95, 50, 397, 1280, 0.16)
    picked = Boxes([]).pick([top, whole_weak, whole], (506, 1280))
    assert picked is whole


def test_weak_whole_box_is_ignored():
    top = Box(435, 3, 761, 851, 0.75)
    whole = Box(431, -1, 764, 1195, 0.10)
    assert Boxes([]).pick([top, whole], (1200, 1200)) is top


def test_rule_off_keeps_old_choice():
    top = Box(435, 3, 761, 851, 0.75)
    whole = Box(431, -1, 764, 1195, 0.20)
    assert Boxes([], partial_min_score=0.0).pick([top, whole], (1200, 1200)) is top


def test_box_over_two_bottles_is_not_an_expansion():
    # Полка: рамка на две соседние бутылки содержит снятую, но по ширине с ней не совпадает.
    target = Box(400, 100, 600, 900, 0.6)
    pair = Box(200, 50, 620, 1150, 0.25)
    assert Boxes([]).pick([target, pair], (1000, 1200)) is target


def test_weak_boxes_alone_do_not_win():
    # Ниже основного порога рамка годится только как расширение, но не как победитель.
    assert Boxes([]).pick([Box(100, 100, 900, 1100, 0.2)], (1000, 1200)) is None
