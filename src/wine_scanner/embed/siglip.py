"""SigLIP 2 как второй глобальный эмбеддер — кандидат из рекомендаций ТЗ.

Чем он может быть лучше DINOv2 именно здесь. DINOv2 обучен без текста и видит этикетку как
картинку; SigLIP 2 учился сопоставлять картинки с подписями и в какой-то мере «читает»
крупный текст на изображении. На каталоге, где близнецы одной линейки различаются словами,
это может сдвинуть визуальный ранг верной карточки. Может и не сдвинуть — потому это
отдельный эмбеддер за флагом, а не замена: побеждает тот, у кого выше R@50/R@100 на живых
кадрах (`eval/platform_benchmark.py`), проигравший в пайплайн не идёт.

Интерфейс тот же, что у Dinov2Embedder (наследуемся, чтобы не дублировать чтение файлов и
батчи): другой препроцессинг (mean = std = 0.5, сторона 384 или 512), другой выход —
пулинговый вектор башни изображений. Индекс, собранный этим эмбеддером, помечен в config.json
своей моделью, и пайплайн поднимет тот же класс через `build_embedder`.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModel

from .dinov2 import Dinov2Embedder, pick_device
from .preprocess import PadToSquare, build_transform

SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)
DEFAULT_SIGLIP_MODEL = "google/siglip2-so400m-patch16-384"


def siglip_size(model_name: str, fallback: int = 384) -> int:
    """Сторона входа зашита в имя модели: ...-patch16-384, ...-patch16-512."""
    tail = model_name.rsplit("-", 1)[-1]
    return int(tail) if tail.isdigit() else fallback


class Siglip2Embedder(Dinov2Embedder):
    def __init__(
        self,
        model_name: str = DEFAULT_SIGLIP_MODEL,
        device: torch.device | None = None,
        size: int | None = None,
        cropper=None,
        fit: str | None = None,
        precision: str = "fp32",
        pad_color: tuple[int, int, int] = (124, 116, 104),
    ):
        # Не зовём родительский __init__: он грузит модель как DINOv2 и ищет register-токены.
        self.device = device or pick_device()
        self.precision = precision
        self.pad_color = tuple(pad_color)
        self.size = size or siglip_size(model_name)
        self.descriptor = "pooled"
        self.cropper = cropper
        self.fit = fit or ("squash" if cropper is not None else "center_crop")
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision, torch.float32)
        full_model = AutoModel.from_pretrained(model_name, dtype=dtype)
        # Only the vision tower is needed for image retrieval. Keep the exact pooled output
        # of get_image_features while avoiding the unused text tower in GPU memory.
        self.model = full_model.vision_model.to(self.device).eval()
        # Fixed-resolution SigLIP 2 checkpoints reuse the `siglip` architecture/config.
        # Generation is identified by the official weight source, not this class name.
        if "siglip2" not in str(model_name).lower():
            raise ValueError("Use the verified SigLIP 2 checkpoint/source")
        self.num_registers = 0

    @property
    def dim(self) -> int:
        config = self.model.config
        vision = getattr(config, "vision_config", config)
        return int(getattr(config, "projection_dim", None) or vision.hidden_size)

    def transform(self):
        transform = build_transform(self.size, self.fit, mean=SIGLIP_MEAN, std=SIGLIP_STD)
        if self.fit == "pad":
            transform.transforms[0] = PadToSquare(self.pad_color)
        return transform

    @torch.inference_mode()
    def encode_batch(self, batch: torch.Tensor) -> torch.Tensor:
        with self._autocast():
            pixel_values = batch.to(self.device, dtype=next(self.model.parameters()).dtype)
            if hasattr(self.model, "get_image_features"):
                vectors = self.model.get_image_features(pixel_values=pixel_values)
            else:
                vectors = self.model(pixel_values=pixel_values).pooler_output
            if not isinstance(vectors, torch.Tensor):
                vectors = vectors.pooler_output
        return F.normalize(vectors.float(), p=2, dim=1).cpu()
