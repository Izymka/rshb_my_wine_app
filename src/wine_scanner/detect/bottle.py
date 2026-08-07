"""Детекция бутылки готовым детектором COCO.

Первый шаг Э3, и он не требует ни разметки, ни обучения: в COCO есть класс bottle, а веса
Faster R-CNN идут вместе с torchvision. Смысл в том, чтобы сначала дёшево измерить, сколько
даёт сама обрезка кадра, и только потом решать, стоит ли вечер разметки ради собственного
детектора этикетки. Если прироста не будет — размечать нечего.

Дальше (Э3б) сюда же встанет дообученная голова на один класс «этикетка»: она обрежет плотнее,
без горлышка и дна, а интерфейс останется тот же.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


class BottleDetector:
    """Находит бутылку на кадре и отдаёт обрезанное изображение."""

    def __init__(
        self,
        device: torch.device | None = None,
        score_threshold: float = 0.5,
        margin: float = 0.08,
        mode: str = "bottle",
    ):
        self.device = device or torch.device("cpu")
        self.score_threshold = score_threshold
        self.margin = margin
        # "bottle" — рамка бутылки целиком, "label" — оценка области этикетки по её геометрии.
        self.mode = mode

        weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights).to(self.device).eval()
        self.preprocess = weights.transforms()
        # Номер класса берём из метаданных весов, а не константой: в torchvision своя нумерация
        # на 91 категорию с пропусками, и угаданное число молча отрежет не то.
        self.bottle_label = weights.meta["categories"].index("bottle")

    @torch.inference_mode()
    def detect(self, image: Image.Image) -> list[Box]:
        tensor = self.preprocess(image).to(self.device)
        output = self.model([tensor])[0]
        keep = (output["labels"] == self.bottle_label) & (output["scores"] >= self.score_threshold)
        return [
            Box(*box.tolist(), score=float(score))
            for box, score in zip(output["boxes"][keep], output["scores"][keep], strict=True)
        ]

    def pick(self, boxes: list[Box], size: tuple[int, int]) -> Box | None:
        """Выбрать целевую бутылку среди найденных.

        На кадрах с полки бутылок много, нужна одна — та, которую снимали. Ранжируем по
        произведению трёх величин: уверенность детектора, доля площади кадра и близость центра
        рамки к центру кадра. Снимающий почти всегда наводит камеру на нужную бутылку и
        подходит ближе, поэтому целевая обычно и крупнее, и центральнее соседей.
        """
        if not boxes:
            return None
        width, height = size
        frame_area = width * height

        def rank(box: Box) -> float:
            cx = (box.x1 + box.x2) / 2 / width
            cy = (box.y1 + box.y2) / 2 / height
            # 1.0 в центре кадра, 0.0 по углам
            centrality = 1.0 - (abs(cx - 0.5) + abs(cy - 0.5))
            return box.score * (box.area / frame_area) ** 0.5 * centrality

        return max(boxes, key=rank)

    def label_window(self, box: Box, size: tuple[int, int]) -> tuple[int, int, int, int]:
        """Оценить прямоугольник этикетки по рамке бутылки.

        Пока своего детектора этикетки нет, пользуемся геометрией бутылки: этикетка занимает
        почти всю ширину и расположена в нижней половине, её центр примерно на 58% высоты
        сверху. Берём квадрат со стороной чуть шире бутылки — так в кадр попадает этикетка
        целиком и почти ничего лишнего.

        Это заглушка до Э3б: обученный детектор даст рамку точнее, а интерфейс не изменится.
        """
        width, height = size
        box_width = box.x2 - box.x1
        box_height = box.y2 - box.y1

        side = box_width * (1 + 2 * self.margin)
        cx = (box.x1 + box.x2) / 2
        cy = box.y1 + box_height * 0.58

        return (
            max(0, int(cx - side / 2)),
            max(0, int(cy - side / 2)),
            min(width, int(cx + side / 2)),
            min(height, int(cy + side / 2)),
        )

    def crop(self, image: Image.Image) -> Image.Image:
        """Обрезать кадр по найденной бутылке. Если бутылки нет — вернуть кадр как есть."""
        box = self.pick(self.detect(image), image.size)
        if box is None:
            return image

        if self.mode == "label":
            return image.crop(self.label_window(box, image.size))

        width, height = image.size
        dx = (box.x2 - box.x1) * self.margin
        dy = (box.y2 - box.y1) * self.margin
        return image.crop(
            (
                max(0, int(box.x1 - dx)),
                max(0, int(box.y1 - dy)),
                min(width, int(box.x2 + dx)),
                min(height, int(box.y2 + dy)),
            )
        )


class CachedCropper:
    """Обрезка с кэшем на диске.

    Детекция стоит доли секунды на кадр, и на каждом прогоне бенчмарка пересчитывать её заново
    незачем: сами кадры не меняются. Ключ кэша учитывает время правки и размер файла, поэтому
    пересъёмка кадра под тем же именем кэш не переиспользует.
    """

    def __init__(self, detector: BottleDetector, cache_dir: Path):
        self.detector = detector
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, path: Path) -> Path:
        stat = path.stat()
        digest = hashlib.sha1(
            f"{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}:{self.detector.mode}".encode()
        ).hexdigest()
        return self.cache_dir / f"{digest}.jpg"

    def __call__(self, path: Path, image: Image.Image) -> Image.Image:
        cached = self._key(path)
        if cached.exists():
            return Image.open(cached).convert("RGB")
        cropped = self.detector.crop(image)
        cropped.save(cached, quality=95)
        return cropped
