import numpy as np
from PIL import Image

from wine_scanner.detect.bottle import Box, BoxCropper
from wine_scanner.image_preprocess import LABEL_SIZE, normalize_label_rgb, prepare_label_image


def test_square_label_crop_keeps_box_and_moves_inside_frame():
    cropper = BoxCropper()
    cropper.square_label = True
    cropper.margin = 0.10
    box = Box(2, 10, 32, 60, 0.9)
    x1, y1, x2, y2 = cropper.crop_rect(box, (80, 100))
    assert x2 - x1 == y2 - y1
    assert 0 <= x1 <= x2 <= 80 and 0 <= y1 <= y2 <= 100
    assert x1 <= box.x1 <= box.x2 <= x2
    assert y1 <= box.y1 <= box.y2 <= y2


def test_photometric_preparation_is_deterministic_rgb_and_fixed_size():
    pixels = np.zeros((41, 61, 3), dtype=np.uint8)
    pixels[..., 0], pixels[..., 1], pixels[..., 2] = 80, 110, 160
    pixels[0, 0] = (255, 255, 255)  # a specular highlight must not break the transform
    image = Image.fromarray(pixels)
    first, first_parameters = prepare_label_image(image)
    second, second_parameters = prepare_label_image(image)
    assert first.mode == "RGB"
    assert first.size == (LABEL_SIZE, LABEL_SIZE)
    assert np.array_equal(np.asarray(first), np.asarray(second))
    assert first_parameters == second_parameters
    assert np.asarray(first).min() >= 0 and np.asarray(first).max() <= 255


def test_white_balance_limits_channel_gains():
    image = Image.new("RGB", (24, 24), (15, 80, 220))
    _, parameters = normalize_label_rgb(image)
    assert all(0.70 <= gain <= 1.40 for gain in parameters.white_balance)
