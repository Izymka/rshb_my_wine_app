"""Photometric-only preprocessing for OCR/local features; preserves pixel coordinates."""

import cv2
import numpy as np
from PIL import Image

BRANCH_MODES = ("rgb", "gray", "clahe2", "clahe4")


def branch_image(image: Image.Image, mode: str = "rgb") -> Image.Image:
    if mode == "rgb":
        return image
    if mode not in BRANCH_MODES:
        raise ValueError(f"Unknown branch preprocessing: {mode}")
    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    if mode.startswith("clahe"):
        gray = cv2.createCLAHE(clipLimit=float(mode[-1]), tileGridSize=(8, 8)).apply(gray)
    return Image.fromarray(gray).convert("RGB")
