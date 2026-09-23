"""Фабрика обрезки: архитектура по весам, каскад из двух ступеней, общее правило выбора рамки."""

import pytest
from PIL import Image

import wine_scanner.detect as detect
from wine_scanner.detect import Box, BoxCropper, CascadeCropper, build_cropper, detector_kind
from wine_scanner.image_preprocess import PreparedLabelCropper


class FakeDetector(BoxCropper):
    """Детектор без модели: отдаёт заранее заданные рамки, помнит, как его создали."""

    made: list = []

    def __init__(self, *args, boxes=None, **kwargs):
        self.args, self.kwargs = args, kwargs
        self.boxes = boxes or []
        self.mode = kwargs.get("mode", "bottle")
        self.cache_tag = "fake"
        FakeDetector.made.append(self)

    def detect(self, image):
        return self.boxes


def test_detector_kind_by_weights(tmp_path):
    (tmp_path / "rtdetr_label").mkdir()
    (tmp_path / "rtdetr_label" / "config.json").write_text("{}")
    (tmp_path / "label_detector.pt").write_bytes(b"")
    assert detector_kind(tmp_path / "rtdetr_label") == "rtdetr"
    with pytest.raises(ValueError, match="retired"):
        detector_kind(tmp_path / "label_detector.pt")
    with pytest.raises(FileNotFoundError):
        detector_kind(tmp_path / "missing")


def test_build_cropper_picks_architecture(monkeypatch, tmp_path):
    monkeypatch.setattr(detect, "RTDetrDetector", FakeDetector)
    weights_dir = tmp_path / "rtdetr_label"
    weights_dir.mkdir()
    (weights_dir / "config.json").write_text("{}")
    weights_pt = tmp_path / "label_detector.pt"
    weights_pt.write_bytes(b"")

    FakeDetector.made.clear()
    cascade = build_cropper("cascade", weights_dir, device="cpu")
    # Каталог, API и бенчмарк получают одну и ту же подготовленную этикетку поверх каскада.
    assert isinstance(cascade, PreparedLabelCropper)
    assert isinstance(cascade.detector, CascadeCropper)
    label, bottle = FakeDetector.made
    # RT-DETR: первая ступень — COCO-модель с целью bottle, вторая — свои веса с целью label.
    assert bottle.args[0] == detect.COCO_BOTTLE_MODEL and bottle.kwargs["target"] == "bottle"
    assert bottle.mode == "bottle"
    assert label.args[0] == weights_dir and label.kwargs["target"] == "label"

    FakeDetector.made.clear()
    with pytest.raises(ValueError, match="retired"):
        build_cropper("cascade", weights_pt, device="cpu")
    assert not FakeDetector.made

    assert build_cropper(None, weights_pt, device="cpu") is None
    FakeDetector.made.clear()
    only_label = build_cropper("trained", weights_dir, device="cpu")
    assert isinstance(only_label, PreparedLabelCropper)
    assert only_label.detector is FakeDetector.made[0] and len(FakeDetector.made) == 1


def test_box_cropper_picks_central_large_bottle_and_crops_with_margin():
    frame = Image.new("RGB", (1000, 1000), "white")
    # Крупная у края против чуть меньшей в центре: побеждает центральная.
    edge = Box(0, 0, 400, 900, score=0.9)
    center = Box(350, 200, 650, 800, score=0.9)
    detector = FakeDetector(boxes=[edge, center])
    assert detector.pick([edge, center], frame.size) is center
    crop = detector.crop(frame)
    # Поля 8 % от размера рамки с каждой стороны.
    assert crop.size == (300 + 2 * 24, 600 + 2 * 48)
    # Без рамок кадр возвращается как есть.
    assert FakeDetector(boxes=[]).crop(frame) is frame
