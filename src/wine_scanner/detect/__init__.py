from pathlib import Path

from .bottle import Box, BoxCropper, CachedCropper, CascadeCropper
from .dataset import CocoDetectionDataset, collate, split_by_wine
from .rtdetr import COCO_BOTTLE_MODEL, RTDetrDetector

__all__ = [
    "COCO_BOTTLE_MODEL",
    "Box",
    "BoxCropper",
    "CachedCropper",
    "CascadeCropper",
    "CocoDetectionDataset",
    "RTDetrDetector",
    "bottle_detector",
    "build_cropper",
    "collate",
    "detector_kind",
    "split_by_wine",
]


def detector_kind(weights: str | Path) -> str:
    """Only RT-DETR is supported; legacy indexes must be rebuilt explicitly."""
    weights = Path(weights)
    if weights.suffix in {".pt", ".pth"}:
        raise ValueError("Faster R-CNN retired; rebuild index using models/rtdetr_label")
    if not (weights / "config.json").is_file():
        raise FileNotFoundError(f"RT-DETR model directory missing config.json: {weights}")
    return "rtdetr"


def bottle_detector(kind: str, device, mode: str = "bottle", bottle_model: str = COCO_BOTTLE_MODEL):
    if kind != "rtdetr":
        raise ValueError("Only RT-DETR is supported")
    return RTDetrDetector(
        bottle_model,
        target="bottle",
        device=device,
        mode=mode,
        min_area_share=0.05,
        require_center=True,
    )


def build_cropper(
    detect: str | None, weights: str | Path, device, bottle_model: str = COCO_BOTTLE_MODEL
):
    """Shared crop path for catalog, API and benchmarks."""
    if detect is None:
        return None
    if detect in {"bottle", "label"}:
        return bottle_detector("rtdetr", device, detect, bottle_model)
    if detect not in {"cascade", "trained"}:
        raise ValueError(f"Unknown crop mode: {detect}")
    detector_kind(weights)
    label = RTDetrDetector(Path(weights), target="label", device=device)
    if detect == "trained":
        return label
    return CascadeCropper(bottle_detector("rtdetr", device, "bottle", bottle_model), label)
