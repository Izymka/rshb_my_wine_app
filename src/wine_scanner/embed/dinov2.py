"""Глобальный эмбеддинг изображения на DINOv2.

Модель по умолчанию — facebook/dinov2-with-registers-large. Register-токены добавлены авторами,
чтобы убрать артефакты внимания на фоне; для нас важно, что из-за них меняется раскладка выхода,
см. extract_descriptor.
"""

from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel

from .preprocess import DEFAULT_SIZE, build_transform, load_image

DEFAULT_MODEL = "facebook/dinov2-with-registers-large"


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class _ImageDataset(Dataset):
    """Нужен только чтобы DataLoader читал и декодировал файлы в несколько процессов."""

    def __init__(self, paths: list[Path], size: int, cropper=None, fit: str = "center_crop"):
        self.paths = paths
        self.transform = build_transform(size, fit)
        self.cropper = cropper

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        path = self.paths[i]
        image = load_image(path)
        if self.cropper is not None:
            image = self.cropper(path, image)
        return self.transform(image)


class Dinov2Embedder:
    """Превращает изображения в L2-нормированные векторы."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: torch.device | None = None,
        size: int = DEFAULT_SIZE,
        descriptor: str = "cls_patchmean",
        cropper=None,
        fit: str | None = None,
        precision: str = "fp32",
    ):
        self.device = device or pick_device()
        self.precision = precision
        self.size = size
        self.descriptor = descriptor
        # cropper — вызываемый объект (path, PIL.Image) -> PIL.Image, обычно detect.CachedCropper.
        # Если он задан, кадр обрезается по бутылке до подачи в модель.
        self.cropper = cropper
        # После обрезки детектором центральный кроп срезает этикетку — см. build_transform.
        self.fit = fit or ("squash" if cropper is not None else "center_crop")
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        # Сколько register-токенов вставлено между CLS и патчами. У версии без регистров — 0.
        self.num_registers = getattr(self.model.config, "num_register_tokens", 0)

    @property
    def dim(self) -> int:
        hidden = self.model.config.hidden_size
        return hidden * 2 if self.descriptor == "cls_patchmean" else hidden

    def extract_descriptor(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Собрать вектор картинки из выхода трансформера.

        Раскладка last_hidden_state по оси токенов:
            [0]                       CLS — общее представление всей картинки
            [1 : 1+num_registers]     register-токены
            [1+num_registers : ]      патчи, то есть участки изображения 14x14

        Здесь два места, где легко ошибиться.

        Первое: register-токены не являются участками изображения. Это служебная память,
        куда модель складывает глобальные вычисления, чтобы не портить патчи. Если усреднить
        всё после CLS (`hidden_state[:, 1:].mean(1)` — ровно то, что напрашивается по аналогии
        с обычным ViT), в среднее попадут четыре вектора, не относящиеся ни к какому участку
        картинки. Ошибка тихая: код работает, размерность правильная, качество ниже.

        Второе: pooler_output, который возвращает та же модель, — это просто CLS. Брать его
        удобно, но CLS кодирует картинку целиком и довольно грубо. Для instance-retrieval
        полезнее добавить усреднённые патчи: они несут локальную фактуру — шрифт, орнамент,
        раскладку блоков на этикетке, то есть ровно то, чем одно вино отличается от другого.
        Конкатенация CLS и среднего по патчам — descriptor, который используют сами авторы
        DINOv2 в линейных пробах.
        """
        cls_token = hidden_state[:, 0]
        patch_tokens = hidden_state[:, 1 + self.num_registers :]

        if self.descriptor == "cls":
            return cls_token
        if self.descriptor == "patchmean":
            return patch_tokens.mean(dim=1)
        if self.descriptor == "cls_patchmean":
            return torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)
        raise ValueError(f"неизвестный descriptor: {self.descriptor}")

    @torch.inference_mode()
    def encode_batch(self, batch: torch.Tensor) -> torch.Tensor:
        with self._autocast():
            outputs = self.model(pixel_values=batch.to(self.device))
            vectors = self.extract_descriptor(outputs.last_hidden_state)
        # L2-нормализация здесь, а не в индексе: так гарантированно нормированы и каталог,
        # и запрос, потому что оба идут через этот метод.
        return F.normalize(vectors.float(), p=2, dim=1).cpu()

    def _autocast(self):
        """Половинная точность — по явному запросу, а не по наличию карты.

        Соблазн включить fp16 автоматически на CUDA велик, и это была бы ошибка того же
        рода, что и наши прежние замеры с прогретыми кэшами: числа изменятся, а проверять
        их никто не станет. Эмбеддинги каталога и запроса обязаны считаться одинаково, и
        если индекс собран в fp32, а запрос считается в fp16, расхождение будет маленьким,
        но никем не измеренным.

        Поэтому переключатель есть, умолчание — fp32, а порядок такой: на новой машине
        собрать индекс и прогнать бенчмарк в обоих режимах, сравнить, и только потом решать.
        """
        if self.precision == "fp32" or self.device.type != "cuda":
            return nullcontext()
        return torch.autocast("cuda", dtype=torch.float16)

    def encode_paths(
        self, paths: list[Path], batch_size: int = 16, num_workers: int = 4, progress: bool = True
    ) -> torch.Tensor:
        if self.cropper is not None:
            # Детектор — это модель на GPU, в дочерние процессы DataLoader её не отдать.
            # После первого прогона кроп берётся из кэша, и медленно уже не будет.
            num_workers = 0
        loader = DataLoader(
            _ImageDataset(paths, self.size, self.cropper, self.fit),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,  # порядок обязан совпадать с порядком paths
        )
        chunks = []
        for batch in tqdm(loader, disable=not progress, desc="эмбеддинги"):
            chunks.append(self.encode_batch(batch))
        return torch.cat(chunks) if chunks else torch.empty(0, self.dim)

    def encode_one(self, path: Path) -> torch.Tensor:
        image = load_image(path)
        if self.cropper is not None:
            image = self.cropper(path, image)
        return self.encode_image(image)

    def encode_image(self, image) -> torch.Tensor:
        """Готовое изображение -> вектор. Обрезка сюда не входит: она уже сделана вызывающим.

        Нужен пайплайну, где кадр приходит не из файла, а из запроса, и обрезанная картинка
        дальше используется ещё дважды — для OCR и для локальных признаков. Читать и резать
        её трижды незачем.
        """
        tensor = build_transform(self.size, self.fit)(image)
        return self.encode_batch(tensor.unsqueeze(0))[0]
