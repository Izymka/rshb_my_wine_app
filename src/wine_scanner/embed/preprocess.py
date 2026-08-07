"""Подготовка изображения перед подачей в модель.

Единственное место в проекте, где картинка превращается в тензор. Каталог и запрос обязаны
проходить ровно через этот код: расхождение в один шаг препроцессинга (другой resize, забытая
нормализация, неучтённый поворот из EXIF) ломает поиск полностью, а отлаживается тяжело —
модель не падает, просто выдаёт мусор.
"""

from pathlib import Path

import torch
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from torchvision import transforms

# iPhone снимает в HEIC, Pillow сам по себе его не открывает.
register_heif_opener()

# Константы нормализации ImageNet — на них обучался DINOv2, менять нельзя.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ViT/14 режет картинку на патчи 14x14, поэтому сторона входа должна делиться на 14.
DEFAULT_SIZE = 224


def build_transform(size: int = DEFAULT_SIZE) -> transforms.Compose:
    """Resize по короткой стороне с запасом, центральный кроп, нормализация.

    Центральный кроп — временное решение на время baseline: он предполагает, что этикетка
    примерно в середине кадра. На этапе Э3 его заменит обрезка по рамке детектора, и вот тогда
    метрики на живых фото должны заметно подрасти.
    """
    return transforms.Compose(
        [
            transforms.Resize(int(size * 256 / 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_image(path: str | Path) -> Image.Image:
    """Открыть файл и привести к RGB в правильной ориентации.

    exif_transpose обязателен. Телефон почти всегда пишет пиксели в ориентации матрицы,
    а как их повернуть — отдельным тегом в EXIF. Просмотрщики этот тег читают, PIL по умолчанию
    нет. Без этой строки половина снятых с рук кадров попадёт в модель лежащими на боку,
    и найти причину провала метрик будет непросто: глазами-то в Finder всё стоит ровно.
    """
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def prepare(path: str | Path, transform: transforms.Compose | None = None) -> torch.Tensor:
    """Файл -> тензор [3, H, W], готовый к подаче в модель."""
    transform = transform or build_transform()
    return transform(load_image(path))
