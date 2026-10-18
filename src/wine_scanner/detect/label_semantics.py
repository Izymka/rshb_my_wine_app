"""Что на вырезке эталона: этикетка или горлышко, капсула, пустое стекло. Нулевой классификатор.

Геометрическая проверка (`reference_check`) судит по рамкам: где бутылка, где выбранная рамка
этикетки. Она не видит содержимого вырезки, а вектор индекса считается именно по содержимому.
Этот модуль задаёт вырезке прямой вопрос через тот же SigLIP 2, которым считается индекс: на что
она больше похожа — на этикетку или на горлышко с кольереткой, фольгу капсулы, тёмное плечо
бутылки без этикетки.

Замер 18.10.2026 на каталоге 2103 эталона: у 7 из 8 грубых ошибок вырезки (кольеретка
«Дербент Вино» у трёх «Эндемы», капсула Мальбека Гай-Кодзор, плечо бутылки у трёх Argonne и
«Плохой девочки») оценка ниже нуля; у верных вырезок ниже нуля — 20 из 2102 (1 %: этикетки,
напечатанные по стеклу, тетрапаки). Поэтому флаг `not_label` отправляет карточку человеку, а не
бракует её сам.

Нужна текстовая башня SigLIP 2 и её токенизатор: веса — из папки модели индекса, токенизатор —
из Hugging Face той же ревизии (в папке модели его нет), один раз в кэш.
"""

import json
from pathlib import Path

import torch
from PIL import Image

LABEL_PROMPTS = (
    "the front label of a wine bottle",
    "a wine label with the name of the wine",
    "a printed paper label with text and a picture",
)
NOT_LABEL_PROMPTS = (
    "the neck of a wine bottle with a foil capsule",
    "a small collar label on the bottle neck",
    "the dark glass shoulder of a bottle without a label",
    "a cork with a wire cage on a sparkling wine bottle",
)
# Ниже — вырезка больше похожа на горлышко или стекло, чем на этикетку (см. замер выше).
NOT_LABEL_BELOW = 0.0


def _tensor(output) -> torch.Tensor:
    """transformers 5 отдаёт из get_*_features объект с pooler_output, 4.x — тензор."""
    return output if isinstance(output, torch.Tensor) else output.pooler_output


class LabelSemantics:
    """Оценка «этикетка минус не-этикетка» по косинусам SigLIP 2 с подсказками."""

    def __init__(self, model_dir: str | Path = "models/siglip2", device=None):
        from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

        model_dir = Path(model_dir)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(model_dir, dtype=dtype).to(self.device).eval()
        self.dtype = dtype
        self.processor = AutoImageProcessor.from_pretrained(model_dir)
        provenance = model_dir / "provenance.json"
        source = json.loads(provenance.read_text(encoding="utf-8")) if provenance.exists() else {}
        tokenizer = AutoTokenizer.from_pretrained(
            source.get("repository", "google/siglip2-so400m-patch16-384"),
            revision=source.get("revision"),
        )
        prompts = list(LABEL_PROMPTS + NOT_LABEL_PROMPTS)
        with torch.inference_mode():
            tokens = tokenizer(prompts, padding="max_length", max_length=64, return_tensors="pt")
            text = _tensor(self.model.get_text_features(**tokens.to(self.device))).float()
        self.text = torch.nn.functional.normalize(text, dim=-1)

    @torch.inference_mode()
    def score(self, crop: Image.Image) -> float:
        pixels = self.processor(images=crop.convert("RGB"), return_tensors="pt")["pixel_values"]
        pixels = pixels.to(self.device, self.dtype)
        image = _tensor(self.model.get_image_features(pixel_values=pixels))
        sims = (torch.nn.functional.normalize(image.float(), dim=-1) @ self.text.T)[0]
        n = len(LABEL_PROMPTS)
        return float(sims[:n].max() - sims[n:].max())

    def is_label(self, crop: Image.Image) -> bool:
        return self.score(crop) >= NOT_LABEL_BELOW
