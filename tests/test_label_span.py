"""Текст только с выбранной бутылки (WINE_OCR_LABEL_ONLY): строки за краями этикетки — прочь."""

from PIL import Image

from wine_scanner.detect.bottle import Box, CachedCropper, CascadeCropper
from wine_scanner.ocr.recognize import TextLine
from wine_scanner.pipeline import WineScanner


def line(text, x0, x1):
    return TextLine(text=text, confidence=1.0, source="test", box=(x0, 0.4, x1, 0.5))


def test_lines_outside_label_are_dropped():
    lines = [
        line("МУСКАТЕЛЬ", 0.32, 0.66),
        line("МУСКАТ", 0.78, 1.0),  # соседняя бутылка справа
        line("ДРА", 0.0, 0.16),  # соседняя слева
        line("БЕЛЫЙ", 0.37, 0.61),
    ]
    kept = WineScanner._label_lines(lines, (0.255, 0.745))
    assert [x.text for x in kept] == ["МУСКАТЕЛЬ", "БЕЛЫЙ"]


def test_line_near_edge_is_kept_and_no_span_keeps_all():
    edge = line("0,75л", 0.70, 0.80)  # центр 0.75 — у самого края, но в запасе
    no_box = TextLine(text="2023", confidence=1.0, source="test", box=None)
    assert WineScanner._label_lines([edge, no_box], (0.25, 0.73)) == [edge, no_box]
    assert WineScanner._label_lines([edge], None) == [edge]


class FixedBox:
    """Детектор с заранее известной рамкой."""

    margin = 0.08
    square_label = True
    mode = "label"
    cache_tag = "fixed"

    def __init__(self, box):
        self.box = box

    def locate(self, image):
        return self.box

    def crop_rect(self, box, size):
        from wine_scanner.image_preprocess import square_rect

        side = max(box.x2 - box.x1, box.y2 - box.y1) * (1 + 2 * self.margin)
        return square_rect(((box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2), side, size)


def test_span_of_tall_label_in_square(tmp_path):
    image = Image.new("RGB", (1000, 1000), "white")
    label = FixedBox(Box(400, 300, 600, 700, 0.9))  # этикетка 200×400: по бокам квадрата — полка
    cascade = CascadeCropper(bottle=FixedBox(None), label=label)
    crop, span = cascade.crop_with_span(image)
    assert crop.size[0] == crop.size[1]
    side = 400 * 1.16
    assert abs(span[0] - (400 - (500 - side / 2)) / side) < 1e-3
    assert abs(span[1] - (600 - (500 - side / 2)) / side) < 1e-3

    path = tmp_path / "frame.png"
    image.save(path)
    cached = CachedCropper(cascade, tmp_path / "cache")
    first = cached.with_span(path, image)[1]
    second = cached.with_span(path, image)[1]  # из кэша, рядом с картинкой
    assert first == second == span
