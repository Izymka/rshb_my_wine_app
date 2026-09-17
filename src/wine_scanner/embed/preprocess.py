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


class PadToSquare:
    """Дополнить изображение серым до квадрата, не меняя пропорций.

    Альтернатива режиму squash. Squash растягивает область под квадрат, и степень растяжения
    зависит от формы рамки: у бутылки в лоб она узкая и высокая, у наклонённой — шире и ниже.
    Значит одна и та же этикетка искажается по-разному в зависимости от ракурса, и эмбеддинги
    расходятся. Дополнение полями сохраняет геометрию ценой части полезной площади кадра.
    """

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        side = max(width, height)
        if width == height:
            return image
        canvas = Image.new("RGB", (side, side), (124, 116, 104))  # средний цвет ImageNet
        canvas.paste(image, ((side - width) // 2, (side - height) // 2))
        return canvas


def build_transform(
    size: int = DEFAULT_SIZE,
    fit: str = "center_crop",
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> transforms.Compose:
    """Привести изображение к квадрату size x size и нормализовать.

    Режим `fit` подбирается под то, что подаётся на вход, и это не мелочь — на нём мы уже
    один раз потеряли восемь пунктов top-1:

    * `center_crop` — для целого кадра с телефона (примерно 3:4). Ужимаем по короткой стороне
      и берём центральный квадрат. Предполагает, что снимаемое находится в середине кадра.

    * `squash` — для кадра, уже обрезанного детектором. Центральный кроп здесь применять нельзя:
      обрезанная бутылка имеет пропорции вроде 1:4, и центральный квадрат вырежет из неё узкую
      горизонтальную полоску, а этикетка окажется срезана. Поэтому масштабируем область целиком,
      не сохраняя пропорции. Искажение одинаково применяется и к каталогу, и к запросу,
      поэтому сравнению не мешает.
    """
    if fit == "center_crop":
        head = [
            transforms.Resize(
                int(size * 256 / 224), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(size),
        ]
    elif fit == "squash":
        head = [
            transforms.Resize(
                (size, size), interpolation=transforms.InterpolationMode.BICUBIC
            )
        ]
    elif fit == "pad":
        head = [
            PadToSquare(),
            transforms.Resize(
                (size, size), interpolation=transforms.InterpolationMode.BICUBIC
            ),
        ]
    else:
        raise ValueError(f"неизвестный fit: {fit}")

    return transforms.Compose(
        [
            *head,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
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
