"""Выбор эмбеддера по имени модели — одно место вместо семи прямых вызовов Dinov2Embedder.

Имя модели хранится в config.json индекса; запрос обязан считаться той же моделью и тем же
препроцессингом, что каталог, иначе векторы окажутся из разных пространств. Поэтому и сборка
индекса, и пайплайн, и бенчмарки создают эмбеддер только здесь.
"""

from .dinov2 import DEFAULT_MODEL, Dinov2Embedder


def build_embedder(
    model_name: str = DEFAULT_MODEL,
    device=None,
    cropper=None,
    fit: str | None = None,
    precision: str = "fp32",
    size: int | None = None,
    descriptor: str | None = None,
    pad_color: tuple[int, int, int] = (124, 116, 104),
):
    if "siglip" in model_name.lower():
        from .siglip import Siglip2Embedder

        return Siglip2Embedder(
            model_name=model_name,
            device=device,
            size=size,
            cropper=cropper,
            fit=fit,
            precision=precision,
            pad_color=pad_color,
        )
    kwargs = {}
    if size is not None:
        kwargs["size"] = size
    if descriptor is not None:
        kwargs["descriptor"] = descriptor
    return Dinov2Embedder(
        model_name=model_name,
        device=device,
        cropper=cropper,
        fit=fit,
        precision=precision,
        **kwargs,
    )
