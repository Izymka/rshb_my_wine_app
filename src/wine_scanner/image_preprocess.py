"""Versioned, deterministic label-image preparation shared by indexing and inference."""

from dataclasses import dataclass

import numpy as np
from PIL import Image

LABEL_PREPROCESS_VERSION = "label-rgb-v1"
LABEL_SIZE = 512


@dataclass(frozen=True)
class PhotometricParameters:
    """Parameters persisted with a derivative so its pixels are reproducible."""

    white_balance: tuple[float, float, float]
    luminance_low: float
    luminance_high: float


def normalize_label_rgb(image: Image.Image) -> tuple[Image.Image, PhotometricParameters]:
    """Robust gray-world white balance plus percentile brightness/contrast normalization.

    Percentiles deliberately ignore the darkest shadows and specular highlights.  The transform
    is global and deterministic: it does not depend on a model, GPU, or surrounding images.
    """
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    flat = pixels.reshape(-1, 3)
    low, high = np.quantile(flat, (0.05, 0.95), axis=0)
    central = np.clip(flat, low, high)
    means = central.mean(axis=0)
    target = float(means.mean())
    gains = np.clip(target / np.maximum(means, 1e-4), 0.70, 1.40)
    balanced = np.clip(pixels * gains, 0.0, 1.0)

    luminance = 0.2126 * balanced[..., 0] + 0.7152 * balanced[..., 1] + 0.0722 * balanced[..., 2]
    lum_low, lum_high = (float(v) for v in np.quantile(luminance, (0.02, 0.98)))
    span = max(lum_high - lum_low, 1e-4)
    target_luminance = np.clip((luminance - lum_low) / span * 0.84 + 0.08, 0.0, 1.0)
    ratio = target_luminance / np.maximum(luminance, 1e-4)
    result = np.clip(balanced * ratio[..., None], 0.0, 1.0)
    return (
        Image.fromarray(np.rint(result * 255).astype(np.uint8), mode="RGB"),
        PhotometricParameters(tuple(float(v) for v in gains), lum_low, lum_high),
    )


def prepare_label_image(
    image: Image.Image, size: int = LABEL_SIZE
) -> tuple[Image.Image, PhotometricParameters]:
    """Normalize a square label crop and make the persisted 512px RGB PNG."""
    normalized, parameters = normalize_label_rgb(image)
    if normalized.size != (size, size):
        normalized = normalized.resize((size, size), Image.Resampling.LANCZOS)
    return normalized, parameters


class PreparedLabelCropper:
    """Wrap a detector so every caller receives the canonical prepared label image."""

    def __init__(self, detector):
        self.detector = detector
        self.cache_tag = f"{getattr(detector, 'cache_tag', '?')}:{LABEL_PREPROCESS_VERSION}"
        self.mode = "prepared-label"

    def crop(self, image: Image.Image) -> Image.Image:
        prepared, _ = prepare_label_image(self.detector.crop(image))
        return prepared

    def __call__(self, path, image: Image.Image) -> Image.Image:
        return self.crop(image)
