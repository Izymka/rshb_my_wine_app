"""RT-DETR как детектор бутылки и этикетки — обе ступени каскада одной архитектурой.

Решение 18.09.2026: каскад остаётся (замер `honest-v5-detector-v2-label-only`: без ступени
«бутылка» top-1 падает 0.657 → 0.463), но обе ступени переезжают с Faster R-CNN на RT-DETR:

- **бутылка** — веса COCO `PekingU/rtdetr_r18vd` как есть, класс `bottle`, без обучения —
  ровно та же роль, что у torchvision-COCO раньше;
- **этикетка** — та же модель, дообученная на один класс (`scripts/train_rtdetr.py`,
  папка `models/rtdetr_label`).

RT-DETR — детектор-трансформер без якорей и NMS, реализация из HuggingFace transformers,
Apache 2.0 (LICENSES.md). Интерфейс тот же, что у `BottleDetector`: `detect` отдаёт рамки,
выбор целевой бутылки и поля вырезки — из общего `BoxCropper`, поэтому смена детектора не
меняет правило «какую бутылку считать снятой».

Какая ветка используется, определяется весами: папка (`save_pretrained`) — RT-DETR, файл
`.pt` — Faster R-CNN (`detect.build_cropper`). Индекс хранит путь к весам в `config.json`,
и запрос режется тем же детектором, что и каталог.
"""

from pathlib import Path

import torch
from PIL import Image

from .bottle import Box, BoxCropper

COCO_BOTTLE_MODEL = "PekingU/rtdetr_r18vd"


class RTDetrDetector(BoxCropper):
    """RT-DETR: бутылка по весам COCO (`target="bottle"`) или этикетка по своим (`target="label"`)."""

    def __init__(
        self,
        source: str | Path = COCO_BOTTLE_MODEL,
        target: str = "bottle",
        device: torch.device | None = None,
        score_threshold: float = 0.3,
        min_area_share: float = 0.0,
        require_center: bool = False,
        margin: float = 0.08,
        mode: str = "bottle",
    ):
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

        self.device = device or torch.device("cpu")
        if self.device.type == "mps":
            # Позиционные эмбеддинги RT-DETR считаются в float64, которого на MPS нет; на
            # ноутбуке детектор живёт на процессоре, на целевой машине — на CUDA.
            self.device = torch.device("cpu")
        # Порог ниже, чем у Faster R-CNN: у DETR-моделей нет NMS, и на тесном крупном плане
        # уверенность в нужной бутылке 0.3–0.45, а 0.6 достаётся горлышку соседа. От горлышка
        # защищает минимальная площадь рамки, а не порог (замер 18.09 на кадрах eval и полки).
        self.score_threshold = score_threshold
        self.min_area_share = min_area_share
        self.require_center = require_center
        self.margin = margin
        self.mode = mode
        self.source = str(source)
        self.processor = RTDetrImageProcessor.from_pretrained(self.source)
        self.model = RTDetrForObjectDetection.from_pretrained(self.source).to(self.device).eval()
        # Номер класса — из конфига модели, а не константой: у дообученной модели класс один
        # и называется «label», у COCO «bottle» стоит на своём месте среди восьмидесяти.
        id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        matches = [i for i, name in id2label.items() if name == target]
        if not matches:
            raise ValueError(f"{self.source}: нет класса {target!r} среди {sorted(id2label.values())[:5]}…")
        self.target_label = matches[0]
        name = Path(self.source).name
        self.cache_tag = f"rtdetr:{name}:{target}:{mode}:t{score_threshold}:a{min_area_share}:c{int(require_center)}"

    @torch.inference_mode()
    def detect(self, image: Image.Image) -> list[Box]:
        rgb = image.convert("RGB")
        inputs = self.processor(images=rgb, return_tensors="pt")
        outputs = self.model(pixel_values=inputs["pixel_values"].to(self.device))
        result = self.processor.post_process_object_detection(
            outputs,
            threshold=self.score_threshold,
            target_sizes=torch.tensor([[rgb.height, rgb.width]]),
        )[0]
        keep = result["labels"] == self.target_label
        return [
            Box(*box.tolist(), score=float(score))
            for box, score in zip(result["boxes"][keep], result["scores"][keep], strict=True)
        ]
