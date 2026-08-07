"""Чтение датасета детекции в формате COCO.

Под экспорт Roboflow: каждая часть выборки лежит своей папкой с картинками и файлом
`_annotations.coco.json` рядом.

    data/third-party datasets/wine-labels/
        train/_annotations.coco.json + *.jpg
        valid/...
        test/...
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ..embed.preprocess import load_image


class CocoDetectionDataset(Dataset):
    """Отдаёт пары (тензор изображения, разметка) в том виде, какой ждёт torchvision.

    Все объекты сводим к одному классу «этикетка»: в исходном датасете класс и так один,
    а фиксированная нумерация избавляет от возни с переносом id категорий COCO — там первый
    id обычно занят служебной супер-категорией, и про это легко забыть.
    """

    LABEL_CLASS = 1  # 0 в torchvision зарезервирован под фон

    def __init__(self, root: Path, transform=None):
        self.root = Path(root)
        self.transform = transform

        data = json.loads((self.root / "_annotations.coco.json").read_text(encoding="utf-8"))
        self.images = {img["id"]: img for img in data["images"]}

        boxes_by_image: dict[int, list[list[float]]] = {}
        for ann in data["annotations"]:
            x, y, w, h = ann["bbox"]
            if w <= 1 or h <= 1:  # вырожденные рамки ломают обучение
                continue
            boxes_by_image.setdefault(ann["image_id"], []).append([x, y, x + w, y + h])

        # Кадры без объектов пропускаем: для одноклассовой задачи они не нужны,
        # а часть детекторов на пустой разметке падает.
        self.ids = [i for i in self.images if boxes_by_image.get(i)]
        self.boxes = boxes_by_image

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        image_id = self.ids[i]
        image = load_image(self.root / self.images[image_id]["file_name"])
        boxes = torch.tensor(self.boxes[image_id], dtype=torch.float32)

        target = {
            "boxes": boxes,
            "labels": torch.full((len(boxes),), self.LABEL_CLASS, dtype=torch.int64),
            "image_id": torch.tensor([image_id]),
        }
        tensor = self.transform(image) if self.transform else _to_tensor(image)
        return tensor, target


def _to_tensor(image):
    from torchvision.transforms import functional as TF

    # Faster R-CNN нормализует вход сам, поэтому здесь только перевод в тензор.
    return TF.to_tensor(image)


def collate(batch):
    """Кадры разного размера в один тензор не складываются — детектор принимает список."""
    return tuple(zip(*batch, strict=True))
