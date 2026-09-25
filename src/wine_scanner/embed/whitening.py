"""PCA-whitening векторов каталога (Э5).

Зачем. Компоненты сырого вектора DINOv2 коррелированы и разного масштаба, и косинус между
двумя картинками определяется несколькими «громкими» направлениями — на каталоге платформы
это буквально «бутылка на белом фоне». Замер Э5 на DINOv2 (ноутбук 01 с тех пор показывает
соседей SigLIP 2 на действующем индексе): медиана
косинуса карточки к ближайшему соседу 0.945, у 47 % карточек сосед ближе 0.95, и есть карточка-хаб,
к которой близко всё. Whitening центрирует векторы, проецирует на главные компоненты и выравнивает
их масштаб, после чего косинус снова начинает означать «похожая этикетка», а не «похожая бутылка».

Обучается на векторах каталога за секунды; хранится рядом с индексом и применяется на лету — сырые
векторы в индексе не меняются, так что переобучить его можно без пересборки.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Whitening:
    mean: np.ndarray  # (d,)
    projection: np.ndarray  # (d, k): главные компоненты, уже поделённые на масштаб
    power: float
    dim: int

    @classmethod
    def fit(cls, vectors: np.ndarray, dim: int = 512, power: float = 0.5) -> "Whitening":
        """Подобрать преобразование по векторам каталога.

        `dim` — сколько главных компонент оставить: при двух тысячах векторов на 2048 измерений
        хвост спектра — шум, и делить на него нельзя. `power` — степень выравнивания: 0.5 — полное
        whitening (деление на корень из дисперсии), меньше — мягче, 0 — просто PCA без масштаба.
        """
        x = np.asarray(vectors, dtype=np.float64)
        if len(x) <= dim:
            raise ValueError(f"векторов {len(x)}, а компонент просят {dim}: нечего оценивать")
        mean = x.mean(axis=0)
        _, singular, components = np.linalg.svd(x - mean, full_matrices=False)
        scale = (singular[:dim] / np.sqrt(len(x))) ** power
        projection = components[:dim].T / scale
        return cls(mean=mean, projection=projection, power=power, dim=dim)

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        """Сырые векторы -> отбелённые, L2-нормированные, float32 (то, чего ждёт индекс)."""
        x = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
        y = (x - self.mean) @ self.projection
        y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
        return y.astype(np.float32)

    def save(self, path: Path) -> None:
        np.savez(path, mean=self.mean, projection=self.projection, power=self.power, dim=self.dim)

    @classmethod
    def load(cls, path: Path) -> "Whitening":
        data = np.load(path)
        return cls(
            mean=data["mean"],
            projection=data["projection"],
            power=float(data["power"]),
            dim=int(data["dim"]),
        )
