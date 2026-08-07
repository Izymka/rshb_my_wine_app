"""Векторный индекс на FAISS.

На baseline берём IndexFlatIP — честный полный перебор без приближений. Для тысячи векторов
это доли миллисекунды, зато поиск точный, и когда метрики окажутся низкими, на индекс думать
не придётся. HNSW подключим на Э11, когда каталог вырастет и появится смысл мерить скорость.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np


@dataclass
class SearchHit:
    item_id: str
    score: float
    payload: dict


class VectorIndex:
    """FAISS-индекс плюс метаданные: какой вектор какому вину соответствует.

    Про выбор IndexFlatIP, а не IndexFlatL2.

    Косинусная близость — это скалярное произведение векторов единичной длины. Отдельного
    «косинусного» индекса в FAISS нет, поэтому схема такая: нормируем векторы сами (это делает
    Dinov2Embedder), а индекс считает скалярное произведение — IP, inner product.

    Тонкость, которую стоит понимать точно. Если векторы уже нормированы, то квадрат
    евклидова расстояния равен 2 - 2*cos, то есть монотонно связан с косинусом, и IndexFlatL2
    отранжирует кандидатов ровно так же. Ошибкой был бы не выбор L2 как таковой, а отказ
    от нормализации: длина вектора DINOv2 зависит от контраста и насыщенности деталями,
    и без нормализации евклидово расстояние начинает сравнивать яркость картинок, а не их
    содержание.

    IP при этом удобнее по двум причинам. Он сразу отдаёт число в диапазоне [-1, 1] с понятным
    смыслом «насколько похоже» — а нам это число дальше нужно как признак для решающего слоя
    и для порога отказа. И его не надо переводить в похожесть, меняя знак, из-за чего в коде
    не появляется путаницы «больше значит лучше или хуже».
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.item_ids: list[str] = []
        self.payloads: list[dict] = []

    def add(self, vectors: np.ndarray, item_ids: list[str], payloads: list[dict]) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError(
                "в индекс пришли ненормированные векторы — скалярное произведение перестанет "
                "быть косинусом. Нормализация должна происходить в Dinov2Embedder."
            )
        self.index.add(vectors)
        self.item_ids.extend(item_ids)
        self.payloads.extend(payloads)

    def search(self, query: np.ndarray, top_k: int = 5) -> list[SearchHit]:
        query = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        scores, positions = self.index.search(query, top_k)
        return [
            SearchHit(self.item_ids[pos], float(score), self.payloads[pos])
            for score, pos in zip(scores[0], positions[0], strict=True)
            if pos != -1
        ]

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "vectors.faiss"))
        meta = {"dim": self.dim, "item_ids": self.item_ids, "payloads": self.payloads}
        (directory / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "VectorIndex":
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        obj = cls(meta["dim"])
        obj.index = faiss.read_index(str(directory / "vectors.faiss"))
        obj.item_ids = meta["item_ids"]
        obj.payloads = meta["payloads"]
        return obj
