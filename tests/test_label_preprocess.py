import numpy as np
from PIL import Image

from wine_scanner.detect.bottle import Box, BoxCropper, CascadeCropper
from wine_scanner.image_preprocess import (
    LABEL_SIZE,
    crop_square,
    pad_to_square,
    prepare_label_image,
    square_rect,
)


def test_square_label_crop_keeps_box_and_moves_inside_frame():
    cropper = BoxCropper()
    cropper.square_label = True
    cropper.margin = 0.10
    box = Box(2, 10, 32, 60, 0.9)
    x1, y1, x2, y2 = cropper.crop_rect(box, (80, 100))
    assert x2 - x1 == y2 - y1
    assert 0 <= x1 <= x2 <= 80 and 0 <= y1 <= y2 <= 100
    assert x1 <= box.x1 <= box.x2 <= x2
    assert y1 <= box.y1 <= box.y2 <= y2


def test_preparation_keeps_colours_and_fixed_size():
    # A colour cast must survive: SiglipImageProcessor normalizes, the preparation does not.
    image = Image.new("RGB", (64, 64), (15, 80, 220))
    prepared = prepare_label_image(image)
    assert prepared.mode == "RGB"
    assert prepared.size == (LABEL_SIZE, LABEL_SIZE)
    assert np.all(np.asarray(prepared) == (15, 80, 220))


class StubCropper(BoxCropper):
    """Detector without a model: returns preset boxes in the coordinates of its input."""

    def __init__(self, boxes, square_label=False, margin=0.08):
        self.boxes, self.square_label, self.margin = boxes, square_label, margin
        self.seen: list[tuple[int, int]] = []

    def detect(self, image):
        self.seen.append(image.size)
        return self.boxes


def test_square_rect_pads_symmetrically_when_frame_is_too_small():
    assert square_rect((20, 60), 60, (40, 120)) == (-10, 30, 50, 90)
    assert square_rect((5, 5), 30, (40, 120)) == (0, 0, 30, 30)


def test_prepare_label_image_pads_instead_of_squashing():
    tall = Image.new("RGB", (40, 120), (200, 30, 30))
    padded = pad_to_square(tall)
    assert padded.size == (120, 120)
    pixels = np.asarray(padded)
    # The original occupies the middle columns unchanged; the sides are its border colour.
    assert np.array_equal(pixels[:, 40:80], np.asarray(tall))
    assert tuple(pixels[0, 0]) == (200, 30, 30)
    prepared = prepare_label_image(tall)
    assert prepared.size == (LABEL_SIZE, LABEL_SIZE)


def test_cascade_cuts_label_square_from_frame_not_from_bottle_crop():
    frame = Image.new("RGB", (1000, 1000), (10, 120, 10))
    bottle = StubCropper([Box(400, 100, 600, 900, 0.9)])
    # A label as wide as the bottle and much wider than tall; boxes are in bottle-crop pixels.
    label = StubCropper([Box(16, 400, 216, 500, 0.9)], square_label=True, margin=0.10)
    crop, bottle_rect, label_rect = CascadeCropper(bottle, label).crop_with_metadata(frame)
    assert bottle_rect == (384, 36, 616, 964)
    assert label.seen == [(232, 928)]  # the label is still searched inside the bottle crop
    x1, y1, x2, y2 = label_rect
    side = x2 - x1
    assert side == y2 - y1 == 240 and crop.size == (side, side)
    # Wider than the bottle crop: the whole label with a border, background on both sides.
    assert side > bottle_rect[2] - bottle_rect[0]
    assert x1 <= 400 and x2 >= 600 and y1 <= 436 and y2 >= 536


def test_cascade_fallbacks_are_square():
    frame = Image.new("RGB", (800, 600), "white")
    bottle = StubCropper([Box(350, 100, 450, 500, 0.9)])
    label = StubCropper([], square_label=True)
    crop, _, rect = CascadeCropper(bottle, label).crop_with_metadata(frame)
    assert crop.size[0] == crop.size[1] == rect[2] - rect[0] == 116
    assert rect[1] <= 100 + 400 * 0.58 <= rect[3]  # around the expected label height

    crop, bottle_rect, rect = CascadeCropper(StubCropper([]), label).crop_with_metadata(frame)
    assert bottle_rect is None and rect == (100, 0, 700, 600) and crop.size == (600, 600)


def test_label_only_detector_without_box_returns_central_square():
    frame = Image.new("RGB", (300, 200), "white")
    crop, rect = StubCropper([], square_label=True).crop_with_box(frame)
    assert rect == (50, 0, 250, 200) and crop.size == (200, 200)


def test_padding_uses_frame_background_not_the_bottle():
    # A tight catalogue shot: white background, a dark bottle filling most of the width.
    frame = Image.new("RGB", (100, 400), "white")
    frame.paste((20, 10, 10), (5, 50, 95, 400))
    crop = crop_square(frame, square_rect((50, 300), 160, frame.size))
    assert crop.size == (160, 160)
    assert tuple(np.asarray(crop)[80, 0]) == (255, 255, 255)
