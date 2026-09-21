"""Чтение датасета детекции: COCO JSON (Roboflow) или CSV makesense.ai.

Под экспорт Roboflow: каждая часть выборки лежит своей папкой с картинками и файлом
`_annotations.coco.json` рядом.

    data/third-party datasets/wine-labels/
        train/_annotations.coco.json + *.jpg
        valid/...
        test/...

Своя разметка из makesense.ai («Export → Single CSV file») — та же папка с картинками и
`_annotations.csv` рядом: строка на рамку, без заголовка,
`label_name,bbox_x,bbox_y,bbox_width,bbox_height,image_name,image_width,image_height`.
Координаты в пикселях исходной картинки. Если рядом лежат оба файла, берётся COCO.
"""

import csv
import json
import unicodedata
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

    COCO_FILE = "_annotations.coco.json"
    CSV_FILE = "_annotations.csv"

    def __init__(self, root: Path, transform=None):
        self.root = Path(root)
        self.transform = transform

        if (self.root / self.COCO_FILE).exists():
            self.images, annotations = self._read_coco(self.root / self.COCO_FILE)
        elif (self.root / self.CSV_FILE).exists():
            self.images, annotations = self._read_makesense_csv(self.root / self.CSV_FILE)
        else:
            raise FileNotFoundError(
                f"в {self.root} нет ни {self.COCO_FILE}, ни {self.CSV_FILE} (экспорт makesense.ai)"
            )

        self._resolve_file_names()

        boxes_by_image: dict[int, list[list[float]]] = {}
        for image_id, (x, y, w, h) in annotations:
            if w <= 1 or h <= 1:  # вырожденные рамки ломают обучение
                continue
            boxes_by_image.setdefault(image_id, []).append([x, y, x + w, y + h])

        # Кадры без объектов пропускаем: для одноклассовой задачи они не нужны,
        # а часть детекторов на пустой разметке падает.
        self.ids = [i for i in self.images if boxes_by_image.get(i)]
        self.boxes = boxes_by_image

    def _resolve_file_names(self) -> None:
        """Сопоставить имена из разметки с файлами на диске, терпя разную нормализацию Unicode.

        macOS хранит кириллицу в именах «разложенной» (NFD: «й» — это «и» + бреве), Windows и
        Linux — «собранной» (NFC), а архиватор на Windows может и вовсе прочитать UTF-8 как
        CP866 — тогда «Лабра» на диске зовётся «╪Ы╪░╪▒» (`_name_keys`). Разметка, сделанная
        на одной системе, на другой перестаёт находить файлы, хотя они на месте. Сравниваем
        по всем формам и подставляем то имя, которое реально лежит в папке. Чего нет совсем
        — ошибка со списком сразу, а не FileNotFoundError посреди первой эпохи.
        """
        on_disk: dict[str, str] = {}
        for path in self.root.iterdir():
            for key in _name_keys(path.name):
                on_disk.setdefault(key, path.name)
        missing = []
        for image in self.images.values():
            name = image["file_name"]
            actual = on_disk.get(unicodedata.normalize("NFC", name))
            if actual is None:
                missing.append(name)
            else:
                image["file_name"] = actual
        if missing:
            raise FileNotFoundError(
                f"в {self.root} нет {len(missing)} файлов из разметки: "
                + ", ".join(missing[:5])
                + (" …" if len(missing) > 5 else "")
            )

    @staticmethod
    def _read_coco(path: Path):
        data = json.loads(path.read_text(encoding="utf-8"))
        images = {img["id"]: img for img in data["images"]}
        categories = {c["id"]: c["name"].lower() for c in data.get("categories", [])}
        annotations = [
            (ann["image_id"], tuple(ann["bbox"]))
            for ann in data["annotations"]
            if not categories or categories.get(ann["category_id"]) in {"label", "wine-labels"}
        ]
        return images, annotations

    @staticmethod
    def _read_makesense_csv(path: Path):
        """CSV makesense: id картинки — порядковый номер её первого появления в файле."""
        images: dict[int, dict] = {}
        ids_by_name: dict[str, int] = {}
        annotations = []
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.reader(f):
                if len(row) < 8 or row[0] == "label_name":  # пустые строки и заголовок
                    continue
                category, x, y, w, h, name, width, height = row[:8]
                if category.lower() not in {"label", "wine-labels"}:
                    continue
                if name not in ids_by_name:
                    ids_by_name[name] = len(ids_by_name)
                    images[ids_by_name[name]] = {
                        "id": ids_by_name[name],
                        "file_name": name,
                        "width": int(float(width)),
                        "height": int(float(height)),
                    }
                annotations.append((ids_by_name[name], (float(x), float(y), float(w), float(h))))
        return images, annotations

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


# Кодировки, через которые Windows-архиваторы портят UTF-8 в именах: «Лабра» превращается в
# «╪Ы╪░╪▒» (CP866 — консольная русская), «Р›Р°Р±СЂР°» (CP1251) или «Ð›Ð°Ð±Ñ€Ð°» (CP1252).
_MOJIBAKE_CODEPAGES = ("cp866", "cp1251", "cp1252", "cp437", "latin-1")


def _name_keys(name: str) -> set[str]:
    """Все формы, под которыми файл с диска может встретиться в разметке.

    Само имя в NFC, плюс «расшифровки» на случай, если архиватор на Windows прочитал
    UTF-8 как однобайтовую кодировку: кодируем видимый мусор обратно в байты каждой из
    кодировок и читаем как UTF-8; что не читается — не имя, пропускаем.
    """
    keys = {unicodedata.normalize("NFC", name)}
    for codepage in _MOJIBAKE_CODEPAGES:
        try:
            keys.add(unicodedata.normalize("NFC", name.encode(codepage).decode("utf-8")))
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return keys


def split_by_wine(root: Path, valid_share: float, seed: int = 0):
    """Своя разметка одной папкой -> (train, valid) с непересекающимися **винами**.

    Делить по кадрам нельзя: четыре кадра одной бутылки почти одинаковы, и валидация с
    тремя из них в обучении измеряет запоминание, а не обобщение. Вино — префикс имени
    файла до `__` (так их называет `scripts/prepare_label_annotation.py`); файл без
    префикса — сам себе вино.
    """
    import random

    full = CocoDetectionDataset(root)
    wine_of = {i: full.images[i]["file_name"].split("__")[0] for i in full.ids}
    wines = sorted(set(wine_of.values()))
    random.Random(seed).shuffle(wines)
    n_valid = max(1, int(len(wines) * valid_share)) if len(wines) > 4 else 0
    valid_wines = set(wines[:n_valid])
    valid, train = CocoDetectionDataset(root), CocoDetectionDataset(root)
    valid.ids = [i for i in full.ids if wine_of[i] in valid_wines]
    train.ids = [i for i in full.ids if wine_of[i] not in valid_wines]
    return train, valid


def _to_tensor(image):
    from torchvision.transforms import functional as TF

    # Нормализацию и resize выполняет RTDetrImageProcessor в collate_for.
    return TF.to_tensor(image)


def collate(batch):
    """Кадры разного размера в один тензор не складываются — детектор принимает список."""
    return tuple(zip(*batch, strict=True))
