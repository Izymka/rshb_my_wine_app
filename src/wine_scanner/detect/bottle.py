"""Shared box selection, rectangular crops and cascade for RT-DETR."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from ..image_preprocess import centered_square, crop_square, square_rect


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


class BoxCropper:
    """Общая часть детекторов: из списка рамок выбрать целевую и вырезать кадр.

    Обе ступени используют RT-DETR; правило выбора рамки и поля вырезки общие
    для индекса, сервиса и измерений качества.
    """

    margin: float = 0.08
    mode: str = "bottle"
    cache_tag: str = "?"
    # Рамки мельче этой доли кадра не считаются: снятая бутылка всегда крупная, а крошечная
    # уверенная рамка — это горлышко соседа или блик. 0 — без ограничения (Faster R-CNN).
    min_area_share: float = 0.0
    # Рамка обязана накрывать центр кадра: снимающий наводит камеру на нужную бутылку, и
    # рамка сбоку — сосед по полке. Если такой нет — кадр целиком, дальше решает детектор
    # этикетки (так каскад вёл себя и раньше, когда бутылка не находилась). Faster R-CNN: выкл.
    require_center: bool = False
    square_label: bool = False

    def detect(self, image: Image.Image) -> list[Box]:
        raise NotImplementedError

    def pick(self, boxes: list[Box], size: tuple[int, int]) -> Box | None:
        """Выбрать целевую бутылку среди найденных.

        На кадрах с полки бутылок много, нужна одна — та, которую снимали. Ранжируем по
        произведению трёх величин: уверенность детектора, доля площади кадра и близость центра
        рамки к центру кадра. Снимающий почти всегда наводит камеру на нужную бутылку и
        подходит ближе, поэтому целевая обычно и крупнее, и центральнее соседей.
        """
        width, height = size
        frame_area = width * height
        boxes = [b for b in boxes if b.area >= self.min_area_share * frame_area]
        if self.require_center:
            boxes = [b for b in boxes if b.x1 <= width / 2 <= b.x2 and b.y1 <= height / 2 <= b.y2]
        if not boxes:
            return None

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

    def crop_rect(self, box: Box, size: tuple[int, int]) -> tuple[int, int, int, int]:
        """Прямоугольник вырезки по рамке: бутылка с полями `margin` или окно этикетки.

        Вынесен отдельно, чтобы разметку, сделанную на вырезке, можно было вернуть в
        координаты исходного кадра (`scripts/prepare_frame_annotation.py`).
        """
        if self.mode == "label":
            return self.label_window(box, size)

        if self.square_label:
            # Square by the label's long side plus a proportional border, translated inside the
            # frame so the extra room is real background. The side is never clamped to the
            # frame: a label cut by the frame edge keeps its full extent, padded if needed.
            side = max(box.x2 - box.x1, box.y2 - box.y1) * (1 + 2 * self.margin)
            center = ((box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2)
            return square_rect(center, side, size)

        width, height = size
        dx = (box.x2 - box.x1) * self.margin
        dy = (box.y2 - box.y1) * self.margin
        return (
            max(0, int(box.x1 - dx)),
            max(0, int(box.y1 - dy)),
            min(width, int(box.x2 + dx)),
            min(height, int(box.y2 + dy)),
        )

    def crop(self, image: Image.Image) -> Image.Image:
        """Обрезать кадр по найденной бутылке. Если бутылки нет — вернуть кадр как есть."""
        cropped, _ = self.crop_with_box(image)
        return cropped

    def crop_with_box(
        self, image: Image.Image
    ) -> tuple[Image.Image, tuple[int, int, int, int] | None]:
        """Crop plus its source coordinates for auditable derivative provenance.

        A square-label detector always returns a square: without a box it falls back to the
        central square of the frame (the coordinates are then still reported).
        """
        box = self.locate(image)
        if box is None:
            if not self.square_label:
                return image, None
            rect = centered_square(image.size)
            return crop_square(image, rect), rect
        rect = self.crop_rect(box, image.size)
        if self.square_label:
            return crop_square(image, rect), rect
        return image.crop(rect), rect

    def locate(self, image: Image.Image) -> Box | None:
        """The target box in `image` coordinates, or None."""
        return self.pick(self.detect(image), image.size)


class CachedCropper:
    """Обрезка с кэшем на диске.

    Детекция стоит доли секунды на кадр, и на каждом прогоне бенчмарка пересчитывать её заново
    незачем: сами кадры не меняются. Ключ кэша учитывает время правки и размер файла, поэтому
    пересъёмка кадра под тем же именем кэш не переиспользует.
    """

    def __init__(self, detector, cache_dir: Path):
        self.detector = detector
        self._fingerprint = self._model_fingerprint(detector)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _model_fingerprint(detector) -> str:
        """Invalidate crops after retraining even when the checkpoint folder is unchanged."""
        if hasattr(detector, "detector"):
            fingerprint = CachedCropper._model_fingerprint(detector.detector)
            return f"{getattr(detector, 'cache_tag', '')}:{fingerprint}"
        if hasattr(detector, "bottle") and hasattr(detector, "label"):
            return ":".join(
                CachedCropper._model_fingerprint(d) for d in (detector.bottle, detector.label)
            )
        parts = [str(getattr(detector, "margin", ""))]
        source = Path(getattr(detector, "source", "__remote__"))
        for name in ("model.safetensors", "config.json", "preprocessor_config.json"):
            path = source / name
            if path.is_file():
                with path.open("rb") as stream:
                    parts.append(hashlib.file_digest(stream, "sha256").hexdigest())
        return ":".join(parts)

    def _key(self, path: Path) -> Path:
        stat = path.stat()
        digest = hashlib.sha1(
            (
                f"{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}:"
                f"{self.detector.cache_tag}:{self._fingerprint}"
            ).encode()
        ).hexdigest()
        return self.cache_dir / f"{digest}.png"

    def __call__(self, path: Path, image: Image.Image) -> Image.Image:
        cached = self._key(path)
        if cached.exists():
            return Image.open(cached).convert("RGB")
        cropped = self.detector.crop(image)
        # Lossless cache: a cache hit must not change pixels relative to the first call.
        cropped.save(cached, compress_level=1)
        return cropped


class CascadeCropper:
    """Двухступенчатая обрезка: сначала бутылка, потом этикетка внутри неё.

    Детектор этикетки обучен на крупных планах, и на кадре, снятом издалека, ему нечего
    разглядывать. Детектор бутылки из COCO с дальними кадрами справляется. Соединяем:
    первый приводит кадр к тому виду, на котором обучался второй.

    Это же и есть правильная архитектура для продакшена — каскад из грубого и точного шага,
    а не одна модель, которой приходится уметь всё сразу.
    """

    def __init__(self, bottle: BoxCropper, label: BoxCropper):
        self.bottle = bottle
        self.label = label
        self.cache_tag = f"cascade:{bottle.cache_tag}->{label.cache_tag}"
        self.mode = "cascade"

    def crop(self, image: Image.Image) -> Image.Image:
        return self.crop_with_metadata(image)[0]

    def crop_with_metadata(
        self, image: Image.Image
    ) -> tuple[Image.Image, tuple[int, int, int, int] | None, tuple[int, int, int, int] | None]:
        """Return the final crop, the bottle rectangle and the label square, all in frame pixels.

        The label is searched inside the bottle crop (that is what the detector was trained on),
        but a square-label stage cuts its square from the original frame: the bottle crop is too
        narrow for a square around a wide label and would cut its edges or leave no background.
        """
        bottle_box = self.bottle.locate(image)
        bottle_rect = self.bottle.crop_rect(bottle_box, image.size) if bottle_box else None
        source = image.crop(bottle_rect) if bottle_rect else image
        box = self.label.locate(source)
        if not self.label.square_label:
            if box is None:
                return source, bottle_rect, None
            rect = self.label.crop_rect(box, source.size)
            return source.crop(rect), bottle_rect, rect
        if box is not None:
            dx, dy = bottle_rect[:2] if bottle_rect else (0, 0)
            frame_box = Box(box.x1 + dx, box.y1 + dy, box.x2 + dx, box.y2 + dy, box.score)
            label_rect = self.label.crop_rect(frame_box, image.size)
        elif bottle_box is not None:
            label_rect = self.estimated_label_square(bottle_box, image.size)
        else:
            label_rect = centered_square(image.size)
        return crop_square(image, label_rect), bottle_rect, label_rect

    def estimated_label_square(
        self, bottle: Box, size: tuple[int, int]
    ) -> tuple[int, int, int, int]:
        """Bottle found, label not: the label usually spans the width around 58% of the height."""
        side = (bottle.x2 - bottle.x1) * (1 + 2 * self.label.margin)
        center = ((bottle.x1 + bottle.x2) / 2, bottle.y1 + (bottle.y2 - bottle.y1) * 0.58)
        return square_rect(center, side, size)
