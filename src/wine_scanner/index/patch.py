"""Пересчитать или дописать карточки собранного индекса, не пересобирая каталог целиком.

Нужен в двух местах: `scripts/patch_index_crops.py` (у карточки сменилась вырезка этикетки) и
`scripts/add_site_wines.py` (в каталог пришло новое вино). В обоих случаях от вырезки зависят
три вещи — вектор эмбеддера, локальные признаки XFeat и бокс года, — и считаться они обязаны
ровно так, как их считала сборка индекса, иначе в индексе окажутся векторы двух «версий».
Поэтому `IndexPatch.verify` сначала пересчитывает несколько нетронутых карточек и сравнивает с
сохранёнными векторами.

Индекс никогда не правится на месте: копия пишется в новую папку (сервис читает `models/`).
Дескрипторы нетронутых карточек копируются жёсткими ссылками. Whitening не переподбирается:
он обучен на тысячах векторов, и десяток замен или добавлений его не сдвигает.
"""

import json
import os
import shutil
from pathlib import Path

import faiss
import numpy as np
from PIL import Image

from ..detect import build_cropper
from ..embed import build_embedder, load_image, pick_device
from ..embed.branches import branch_image
from ..rerank import DescriptorStore, XFeatMatcher
from ..vintage import catalog_years, reference_region

MIN_COSINE = 0.995


def copy_index(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=False)
    for name in ("config.json", "meta.json", "vectors.faiss", "whitening.npz"):
        if (source / name).exists():
            shutil.copy2(source / name, target / name)
    if (source / "descriptors").exists():
        shutil.copytree(source / "descriptors", target / "descriptors", copy_function=os.link)


class IndexPatch:
    """Собранный индекс в памяти: векторы, payload и всё, чем считается новая карточка."""

    def __init__(
        self,
        index_dir: Path,
        device=None,
        rerank_max_side: int = 640,
        rerank_points: int = 2048,
    ):
        self.source = Path(index_dir)
        self.config = json.loads((self.source / "config.json").read_text(encoding="utf-8"))
        self.meta = json.loads((self.source / "meta.json").read_text(encoding="utf-8"))
        self.position = {item_id: n for n, item_id in enumerate(self.meta["item_ids"])}
        index = faiss.read_index(str(self.source / "vectors.faiss"))
        self.vectors = index.reconstruct_n(0, index.ntotal)
        self.device = device or pick_device()
        self.weights = Path(str(self.config["weights"]).replace("\\", "/"))
        self.embedder = build_embedder(
            model_name=self.config["model"],
            device=self.device,
            cropper=None,
            fit=self.config.get("fit", "pad"),
            precision=self.config.get("precision", "fp32"),
            size=self.config.get("size"),
            descriptor=self.config.get("descriptor"),
            pad_color=tuple(self.config.get("pad_color", (255, 255, 255))),
        )
        self.rerank_max_side = rerank_max_side
        self.rerank_points = rerank_points
        self._cascade = None
        self._matcher = None
        self._ocr = None
        self.target: Path | None = None
        self.store: DescriptorStore | None = None

    # --- то, чем индекс режет эталон --------------------------------------------------------

    @property
    def cascade(self):
        """Каскад с настройками индекса (`PreparedLabelCropper` поверх `CascadeCropper`)."""
        if self._cascade is None:
            self._cascade = build_cropper(
                self.config.get("detect", "cascade"),
                self.weights,
                self.device,
                partial_min_score=self.config.get("bottle_partial_min_score", 0.0),
            )
        return self._cascade

    def image_of(self, item_id: str) -> Image.Image:
        payload = self.meta["payloads"][self.position[item_id]]
        path = Path(str(payload.get("image_path", "")).replace("\\", "/"))
        return load_image(path if path.exists() else Path(f"data/catalog/originals/{item_id}.png"))

    def encode(self, crop: Image.Image) -> np.ndarray:
        return self.embedder.encode_image(crop).float().cpu().numpy()

    def verify(self, n: int = 3, skip: set[str] = frozenset()) -> float:
        """Пересчитать `n` нетронутых карточек штатным каскадом; ниже MIN_COSINE — исключение."""
        probe = [i for i in self.meta["item_ids"] if i not in skip]
        probe = probe[:: max(1, len(probe) // n)][:n]
        cosines = [
            float(self.encode(self.cascade.crop(self.image_of(i))) @ self.vectors[self.position[i]])
            for i in probe
        ]
        worst = min(cosines)
        if worst < MIN_COSINE:
            raise RuntimeError(
                f"пересчёт не воспроизводит индекс (min cos {worst:.4f} < {MIN_COSINE}): "
                "другая точность, устройство или веса"
            )
        return worst

    # --- запись -------------------------------------------------------------------------------

    def open_target(self, target: Path) -> None:
        copy_index(self.source, target)
        self.target = Path(target)
        self.store = DescriptorStore(self.target / "descriptors")

    def _describe(self, item_id: str, crop: Image.Image) -> None:
        if self._matcher is None:
            self._matcher = XFeatMatcher(max_side=self.rerank_max_side, top_k=self.rerank_points)
        local = self.config.get("local_preprocess", "rgb")
        # Жёсткая ссылка на файл исходного индекса: сначала разорвать, потом писать.
        path = self.store.path_for(item_id)
        if path.exists():
            path.unlink()
        self.store.save(item_id, self._matcher.describe(branch_image(crop, local)))

    def _vintage(self, payload: dict, crop: Image.Image) -> None:
        if self._ocr is None:
            from ..ocr import LabelOCR

            # Как в build_index.year_boxes: каталог читает всегда PaddleOCR, не WINE_OCR.
            self._ocr = LabelOCR(backend="paddle")
        region = reference_region(self._ocr.read(crop, use_cache=False), catalog_years(payload))
        payload.pop("vintage_box", None)
        if region is not None:
            payload["vintage_box"] = list(region.box)

    def replace(self, item_id: str, crop: Image.Image) -> float:
        """Пересчитать карточку по новой вырезке. Возвращает cos(старый, новый)."""
        n = self.position[item_id]
        old = self.vectors[n].copy()
        self.vectors[n] = self.encode(crop)
        self._describe(item_id, crop)
        self._vintage(self.meta["payloads"][n], crop)
        return float(old @ self.vectors[n])

    def add(self, item_id: str, payload: dict, crop: Image.Image) -> np.ndarray:
        """Дописать новую карточку в конец индекса. Возвращает её вектор."""
        if item_id in self.position:
            raise ValueError(f"{item_id} уже в индексе")
        vector = self.encode(crop)
        payload = dict(payload)
        self._describe(item_id, crop)
        self._vintage(payload, crop)
        self.vectors = np.vstack([self.vectors, vector[None, :]])
        self.position[item_id] = len(self.meta["item_ids"])
        self.meta["item_ids"].append(item_id)
        self.meta["payloads"].append(payload)
        return vector

    def save(self, provenance_key: str, provenance: dict) -> None:
        """Записать векторы, meta и config в `target`; `config[provenance_key] = provenance`."""
        index = faiss.IndexFlatIP(self.vectors.shape[1])
        index.add(np.ascontiguousarray(self.vectors, dtype=np.float32))
        faiss.write_index(index, str(self.target / "vectors.faiss"))
        (self.target / "meta.json").write_text(
            json.dumps(self.meta, ensure_ascii=False), encoding="utf-8"
        )
        config = {**self.config, "items": len(self.meta["item_ids"])}
        config[provenance_key] = provenance
        (self.target / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
