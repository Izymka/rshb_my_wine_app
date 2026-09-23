"""Versioned, deterministic label-image preparation shared by indexing and inference."""

import numpy as np
from PIL import Image

# v2: the square is cut from the source frame (not from the bottle crop), fallbacks stay square,
# nothing is resized with a changed aspect ratio, and there is no colour correction.
# v3: out-of-frame padding takes the frame background (median of its border), not the crop's.
LABEL_PREPROCESS_VERSION = "label-square-v3"
LABEL_SIZE = 512


def square_rect(
    center: tuple[float, float], side: float, size: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Integer square of `side` around `center`, shifted to stay inside a `size` frame.

    The square is translated rather than shrunk, so the label keeps its proportional border and
    the extra room is filled with real background.  Only when the frame itself is smaller than
    the square along an axis does the rectangle leave the frame, symmetrically around it.
    """
    width, height = size
    side_px = max(1, int(round(side)))

    def start(middle: float, dim: int) -> int:
        low, high = min(0, dim - side_px), max(0, dim - side_px)
        return int(round(min(max(middle - side_px / 2, low), high)))

    x1, y1 = start(center[0], width), start(center[1], height)
    return x1, y1, x1 + side_px, y1 + side_px


def centered_square(size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Largest square in the middle of the frame: the photographer aims at the target."""
    width, height = size
    return square_rect((width / 2, height / 2), min(width, height), size)


def border_color(image: Image.Image) -> tuple[int, int, int]:
    """Median colour of the frame's outer one-pixel ring: a deterministic stand-in for background.

    The median, not the mean: a bottle touching the frame edge must not tint the fill.
    """
    pixels = np.asarray(image.convert("RGB"))
    ring = np.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]])
    return tuple(int(v) for v in np.median(ring, axis=0))


def crop_square(image: Image.Image, rect: tuple[int, int, int, int]) -> Image.Image:
    """Crop `rect`; parts outside the frame get the whole frame's background colour.

    Catalogue shots are often cropped tight to the bottle, so a label square is wider than the
    frame; the fill must match the photo's background, not the bottle inside the crop.
    """
    x1, y1, x2, y2 = rect
    width, height = image.size
    if x1 >= 0 and y1 >= 0 and x2 <= width and y2 <= height:
        return image.crop(rect)
    inside = image.crop((max(x1, 0), max(y1, 0), min(x2, width), min(y2, height)))
    canvas = Image.new("RGB", (x2 - x1, y2 - y1), border_color(image))
    canvas.paste(inside.convert("RGB"), (max(-x1, 0), max(-y1, 0)))
    return canvas


def pad_to_square(image: Image.Image) -> Image.Image:
    """Safety net for non-square inputs: extend the short side, never distort the label."""
    width, height = image.size
    if width == height:
        return image
    rect = square_rect((width / 2, height / 2), max(width, height), (width, height))
    return crop_square(image, rect)


def prepare_label_image(image: Image.Image, size: int = LABEL_SIZE) -> Image.Image:
    """Make the persisted 512px RGB label: pad to a square, then scale uniformly.

    Colours are left as captured; SiglipImageProcessor applies the model's own normalization.
    A non-square input is padded, not squashed: uniform scaling is the only geometric change.
    """
    prepared = pad_to_square(image.convert("RGB"))
    if prepared.size != (size, size):
        prepared = prepared.resize((size, size), Image.Resampling.LANCZOS)
    return prepared


class PreparedLabelCropper:
    """Wrap a detector so every caller receives the canonical prepared label image."""

    def __init__(self, detector):
        self.detector = detector
        self.cache_tag = f"{getattr(detector, 'cache_tag', '?')}:{LABEL_PREPROCESS_VERSION}"
        self.mode = "prepared-label"

    def crop(self, image: Image.Image) -> Image.Image:
        return prepare_label_image(self.detector.crop(image))

    def __call__(self, path, image: Image.Image) -> Image.Image:
        return self.crop(image)
