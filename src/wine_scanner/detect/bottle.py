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
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

BACKBONES = {
    "resnet50": (fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1),
    "mobilenet": (
        fasterrcnn_mobilenet_v3_large_fpn,
        FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1,
    ),
    "mobilenet320": (
        fasterrcnn_mobilenet_v3_large_320_fpn,
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1,
    ),
}


def build_label_detector(
    backbone: str = "mobilenet320",
    pretrained: bool = True,
    trainable_backbone_layers: int | None = None,
    min_size: int | None = None,
):
    """Faster R-CNN с головой на два класса: фон и этикетка.

    Веса COCO оставляем во всём, кроме последнего слоя-классификатора — его меняем под свою
    задачу. Это и есть стандартное дообучение детектора: бэкбон уже умеет видеть объекты
    вообще, доучиваем только «что считать целью».

    `trainable_backbone_layers` задаёт, сколько верхних блоков бэкбона размораживается.
    Чем меньше, тем быстрее обратный проход и тем меньше риск переобучения на небольшой выборке.
    `min_size` — сторона, к которой детектор ужимает вход; главный рычаг скорости.
    """
    factory, weights = BACKBONES[backbone]
    kwargs = {}
    if trainable_backbone_layers is not None:
        kwargs["trainable_backbone_layers"] = trainable_backbone_layers

    model = factory(weights=weights if pretrained else None, **kwargs)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)

    if min_size is not None:
        # Разрешение обязано совпадать между обучением и инференсом, иначе рамки поедут.
        # Поэтому min_size сохраняется в чекпоинт и применяется при загрузке.
        model.transform.min_size = (min_size,)
        model.transform.max_size = int(min_size * 5 / 3)
    return model


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
        weights_path: Path | None = None,
        infer_min_size: int | None = None,
        backbone: str = "resnet50",
    ):
        self.device = device or torch.device("cpu")
        self.score_threshold = score_threshold
        self.margin = margin
        # "bottle" — рамка целиком, "label" — оценка области этикетки по геометрии бутылки.
        # Для дообученного детектора этикетки режим всегда "bottle": рамка уже и есть этикетка.
        self.mode = mode

        if weights_path is not None:
            checkpoint = torch.load(weights_path, map_location="cpu")
            model = build_label_detector(
                checkpoint["backbone"], pretrained=False, min_size=checkpoint.get("min_size")
            )
            model.load_state_dict(checkpoint["state_dict"])
            self.model = model.to(self.device).eval()
            self.preprocess = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.transforms()
            self.bottle_label = 1
            self.mode = "bottle"
            # Метка для ключа кэша: у дообученного детектора рамки другие, чем у COCO,
            # и кропы не должны переиспользоваться между ними.
            self.cache_tag = f"trained:{weights_path.name}"
            if infer_min_size:
                # Разрешение инференса выше обучающего. Обычно так делать не следует, но FPN
                # к смене масштаба устойчив, а на дальних кадрах этикетка при 320 пикселях
                # просто исчезает. Прирост проверяем замером, а не предположением.
                self.model.transform.min_size = (infer_min_size,)
                self.model.transform.max_size = int(infer_min_size * 5 / 3)
                self.cache_tag += f":min{infer_min_size}"
            return

        factory, weights = BACKBONES[backbone]
        self.model = factory(weights=weights).to(self.device).eval()
        self.preprocess = weights.transforms()
        # Номер класса берём из метаданных весов, а не константой: в torchvision своя нумерация
        # на 91 категорию с пропусками, и угаданное число молча отрежет не то.
        self.bottle_label = weights.meta["categories"].index("bottle")
        self.cache_tag = f"coco:{backbone}:{mode}"

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

    def __init__(self, detector, cache_dir: Path):
        self.detector = detector
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, path: Path) -> Path:
        stat = path.stat()
        digest = hashlib.sha1(
            f"{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}:{self.detector.cache_tag}".encode()
        ).hexdigest()
        return self.cache_dir / f"{digest}.jpg"

    def __call__(self, path: Path, image: Image.Image) -> Image.Image:
        cached = self._key(path)
        if cached.exists():
            return Image.open(cached).convert("RGB")
        cropped = self.detector.crop(image)
        cropped.save(cached, quality=95)
        return cropped


class CascadeCropper:
    """Двухступенчатая обрезка: сначала бутылка, потом этикетка внутри неё.

    Детектор этикетки обучен на крупных планах, и на кадре, снятом издалека, ему нечего
    разглядывать. Детектор бутылки из COCO с дальними кадрами справляется. Соединяем:
    первый приводит кадр к тому виду, на котором обучался второй.

    Это же и есть правильная архитектура для продакшена — каскад из грубого и точного шага,
    а не одна модель, которой приходится уметь всё сразу.
    """

    def __init__(self, bottle: BottleDetector, label: BottleDetector):
        self.bottle = bottle
        self.label = label
        self.cache_tag = f"cascade:{bottle.cache_tag}->{label.cache_tag}"
        self.mode = "cascade"

    def crop(self, image: Image.Image) -> Image.Image:
        return self.label.crop(self.bottle.crop(image))
