from pathlib import Path

from .bottle import Box, BottleDetector, BoxCropper, CachedCropper, CascadeCropper, build_label_detector
from .dataset import CocoDetectionDataset, collate, split_by_wine
from .rtdetr import COCO_BOTTLE_MODEL, RTDetrDetector

__all__ = [
    "COCO_BOTTLE_MODEL",
    "BottleDetector",
    "Box",
    "BoxCropper",
    "CachedCropper",
    "CascadeCropper",
    "CocoDetectionDataset",
    "RTDetrDetector",
    "bottle_detector",
    "build_cropper",
    "build_label_detector",
    "collate",
    "detector_kind",
    "split_by_wine",
]


def detector_kind(weights: str | Path) -> str:
    """`rtdetr` для папки с моделью transformers, `frcnn` для чекпойнта `.pt` torchvision."""
    return "rtdetr" if Path(weights).is_dir() else "frcnn"


def bottle_detector(kind: str, device, mode: str = "bottle", bottle_model: str = COCO_BOTTLE_MODEL):
    """Первая ступень каскада — бутылка по весам COCO. Настройки RT-DETR подобраны 18.09 на 32
    живых кадрах: порог 0.3 (у DETR без NMS уверенность на тесном плане ниже), рамка не мельче
    5 % кадра и обязана накрывать его центр — иначе кадр целиком."""
    if kind == "rtdetr":
        return RTDetrDetector(
            bottle_model,
            target="bottle",
            device=device,
            mode=mode,
            min_area_share=0.05,
            require_center=True,
        )
    return BottleDetector(device=device, mode=mode)


def build_cropper(detect: str | None, weights: str | Path, device, bottle_model: str = COCO_BOTTLE_MODEL):
    """Единственное место, где собирается обрезка — одна и та же для индекса, сервиса и скриптов.

    `detect`: `cascade` — бутылка, потом этикетка внутри неё (рабочая конфигурация);
    `trained` — только детектор этикетки; `bottle` / `label` — только детектор бутылки COCO,
    рамка целиком или окно этикетки по геометрии; None — без обрезки. Архитектура обеих
    ступеней — по весам этикетки: папка RT-DETR или файл `.pt` Faster R-CNN (`detector_kind`).
    """
    if detect is None:
        return None
    weights = Path(weights)
    kind = detector_kind(weights)
    if kind == "rtdetr":
        label = lambda: RTDetrDetector(weights, target="label", device=device)  # noqa: E731
    else:
        label = lambda: BottleDetector(device=device, weights_path=weights)  # noqa: E731
    if detect == "cascade":
        return CascadeCropper(bottle_detector(kind, device, "bottle", bottle_model), label())
    if detect == "trained":
        return label()
    return bottle_detector(kind, device, detect, bottle_model)
