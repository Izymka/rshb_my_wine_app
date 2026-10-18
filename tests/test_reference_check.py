"""Проверка вырезки эталона: флаги каскада, починка по маске переднего плана, поправки."""

import numpy as np
from PIL import Image, ImageDraw

from wine_scanner.detect import Box, BoxCropper, CascadeCropper
from wine_scanner.detect.reference_check import (
    OverrideCropper,
    check_reference,
    foreground_box,
    label_square,
    load_overrides,
    repair_bottle,
    save_overrides,
)


class FixedBottle(BoxCropper):
    """Первая ступень без модели: всегда одна заданная рамка бутылки."""

    def __init__(self, box):
        self.box = box
        self.cache_tag = "fixed"

    def detect(self, image):
        return [Box(*self.box, score=0.9)]


class RedLabel(BoxCropper):
    """Вторая ступень без модели: этикетка — красные пиксели в том, что ей дали."""

    square_label = True
    margin = 0.08

    def detect(self, image):
        pixels = np.asarray(image.convert("RGB")).astype(int)
        mask = (pixels[..., 0] > 200) & (pixels[..., 1] < 60) & (pixels[..., 2] < 60)
        rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
        if rows.size == 0:
            return []
        return [Box(cols[0], rows[0], cols[-1] + 1, rows[-1] + 1, score=0.95)]


def reference() -> Image.Image:
    """Эталон-вырезка: тёмная бутылка 100..300 × 50..950 на белом, этикетка 600..850."""
    image = Image.new("RGB", (400, 1000), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((100, 50, 299, 949), fill=(20, 30, 20))
    draw.rectangle((110, 600, 289, 849), fill=(230, 10, 10))
    return image


def test_foreground_box_on_white_and_none_on_busy_frame():
    assert foreground_box(reference()) == (100, 50, 300, 950)
    rng = np.random.default_rng(0)
    noise = Image.fromarray(rng.integers(0, 255, (100, 100, 3), dtype=np.uint8))
    assert foreground_box(noise) is None


def test_good_reference_has_no_flags():
    cascade = CascadeCropper(FixedBottle((100, 50, 300, 950)), RedLabel())
    check = check_reference(cascade, reference())
    assert check.ok and check.flags == []
    assert 0.65 < check.center_level < 0.85


def test_truncated_bottle_is_flagged_and_repaired_by_foreground():
    # Детектор бутылки накрыл только верх до середины этикетки: этикетка обрезана рамкой.
    cascade = CascadeCropper(FixedBottle((100, 50, 300, 700)), RedLabel())
    image = reference()
    check = check_reference(cascade, image)
    assert "bottle_truncated" in check.flags and "label_clipped" in check.flags
    bottle = repair_bottle(check)
    assert bottle == (100, 50, 300, 950)
    retry = check_reference(cascade, image, bottle_rect=bottle)
    assert retry.ok
    x1, y1, x2, y2 = label_square(cascade, retry)
    # Квадрат накрывает этикетку целиком: 600..850 по высоте.
    assert y1 <= 600 and y2 >= 850 and x2 - x1 == y2 - y1


def test_neck_label_is_flagged():
    image = reference()
    ImageDraw.Draw(image).rectangle((110, 600, 289, 849), fill=(20, 30, 20))
    ImageDraw.Draw(image).rectangle((170, 120, 230, 200), fill=(230, 10, 10))
    cascade = CascadeCropper(FixedBottle((100, 50, 300, 950)), RedLabel())
    check = check_reference(cascade, image)
    assert "label_on_neck" in check.flags and "small_label" in check.flags
    assert not check.ok


def test_overrides_roundtrip_and_cropper(tmp_path):
    path = tmp_path / "label_overrides.csv"
    save_overrides({"wine-a": {"rect": (100, 500, 300, 700), "source": "manual"}}, path)
    rows = load_overrides(path)
    assert rows["wine-a"]["rect"] == (100, 500, 300, 700)
    assert rows["wine-a"]["source"] == "manual"

    calls = []
    inner = lambda p, image: calls.append(p) or image  # noqa: E731
    cropper = OverrideCropper(inner, rows)
    image = reference()
    fixed = cropper(tmp_path / "wine-a.png", image)
    assert fixed.size == (512, 512) and not calls
    cropper(tmp_path / "wine-b.png", image)
    assert calls == [tmp_path / "wine-b.png"]
    assert load_overrides(tmp_path / "missing.csv") == {}


def test_accepted_decision_keeps_cascade_crop(tmp_path):
    rows = {
        "wine-a": {"rect": (0, 0, 10, 10), "source": "accepted"},
        "wine-b": {"rect": (100, 500, 300, 700), "source": "manual"},
    }
    calls = []
    cropper = OverrideCropper(lambda p, image: calls.append(p) or image, rows)
    image = reference()
    assert cropper(tmp_path / "wine-a.png", image) is image
    assert cropper(tmp_path / "wine-b.png", image).size == (512, 512)
    assert calls == [tmp_path / "wine-a.png"]


def test_review_reference_statuses():
    from wine_scanner.detect.reference_check import review_reference

    image = reference()
    good = CascadeCropper(FixedBottle((100, 50, 300, 950)), RedLabel())
    assert review_reference(good, image).status == "ok"

    # Рамка бутылки до середины этикетки: повтор по маске переднего плана чинит вырезку.
    short = CascadeCropper(FixedBottle((100, 50, 300, 700)), RedLabel())
    repaired = review_reference(short, image)
    assert repaired.status == "repaired" and repaired.passed
    assert repaired.square == repaired.rect and repaired.rect != repaired.crop

    # Решение человека важнее проверки.
    manual = review_reference(short, image, {"rect": (1, 2, 3, 4), "source": "manual"})
    assert manual.status == "manual" and manual.square == (1, 2, 3, 4)
    accepted = review_reference(short, image, {"rect": (0, 0, 1, 1), "source": "accepted"})
    assert accepted.status == "accepted" and accepted.square == accepted.crop

    # Этикетка на горлышке у целой бутылки: маска не поможет — нужен человек.
    neck = reference()
    ImageDraw.Draw(neck).rectangle((110, 600, 289, 849), fill=(20, 30, 20))
    ImageDraw.Draw(neck).rectangle((170, 120, 230, 200), fill=(230, 10, 10))
    review = review_reference(good, neck)
    assert review.status == "review" and not review.passed
