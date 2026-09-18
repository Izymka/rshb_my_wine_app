"""Датасет детекции: COCO JSON и CSV makesense.ai дают одинаковые рамки."""

import json

import pytest
from PIL import Image

from wine_scanner.detect.dataset import CocoDetectionDataset


def make_images(root, names):
    for name in names:
        Image.new("RGB", (40, 60), "white").save(root / name)


def test_makesense_csv_matches_coco(tmp_path):
    coco_dir, csv_dir = tmp_path / "coco", tmp_path / "csv"
    coco_dir.mkdir(), csv_dir.mkdir()
    make_images(coco_dir, ["a.jpg", "b.jpg"])
    make_images(csv_dir, ["a.jpg", "b.jpg"])
    (coco_dir / "_annotations.coco.json").write_text(
        json.dumps(
            {
                "images": [
                    {"id": 0, "file_name": "a.jpg", "width": 40, "height": 60},
                    {"id": 1, "file_name": "b.jpg", "width": 40, "height": 60},
                ],
                "annotations": [
                    {"image_id": 0, "bbox": [5, 10, 20, 30]},
                    {"image_id": 1, "bbox": [1, 1, 0.5, 0.5]},  # вырожденная — кадр выпадает
                ],
            }
        )
    )
    (csv_dir / "_annotations.csv").write_text(
        "label,5,10,20,30,a.jpg,40,60\nlabel,1,1,0.5,0.5,b.jpg,40,60\n"
    )
    coco, csv_ = CocoDetectionDataset(coco_dir), CocoDetectionDataset(csv_dir)
    assert len(coco) == len(csv_) == 1
    _, coco_target = coco[0]
    _, csv_target = csv_[0]
    assert coco_target["boxes"].tolist() == csv_target["boxes"].tolist() == [[5, 10, 25, 40]]
    assert csv_target["labels"].tolist() == [1]


def test_csv_with_header_and_two_boxes_per_image(tmp_path):
    make_images(tmp_path, ["a.jpg"])
    (tmp_path / "_annotations.csv").write_text(
        "label_name,bbox_x,bbox_y,bbox_width,bbox_height,image_name,image_width,image_height\n"
        "label,0,0,10,10,a.jpg,40,60\nlabel,20,20,10,10,a.jpg,40,60\n"
    )
    dataset = CocoDetectionDataset(tmp_path)
    assert len(dataset) == 1
    assert len(dataset[0][1]["boxes"]) == 2


def test_missing_annotations_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        CocoDetectionDataset(tmp_path)


def test_crop_box_returns_to_frame_coordinates():
    """Рамка с ужатой вырезки → кадр: обратный масштаб вырезки, сдвиг, масштаб кадра."""
    from scripts.prepare_frame_annotation import crop_to_frame

    # Вырезка 1000×2000 из кадра с углом (100, 200), сохранена ужатой до 800×1600 (k = 1.25);
    # кадр сохранён в половину (sx = sy = 0.5).
    x, y, w, h = crop_to_frame((80, 160, 400, 240), (100, 200, 1100, 2200), (800, 1600), (0.5, 0.5))
    assert (x, y, w, h) == ((100 + 100) * 0.5, (200 + 200) * 0.5, 500 * 0.5, 300 * 0.5)


def test_crop_rect_matches_crop():
    """`crop` режет ровно по `crop_rect` — иначе предразметка разъедется с вырезками."""
    from wine_scanner.detect.bottle import Box, BottleDetector

    detector = BottleDetector.__new__(BottleDetector)
    detector.mode, detector.margin = "bottle", 0.08
    rect = detector.crop_rect(Box(100, 200, 300, 600, score=0.9), (400, 800))
    assert rect == (84, 168, 316, 632)
