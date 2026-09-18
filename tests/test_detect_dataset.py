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
