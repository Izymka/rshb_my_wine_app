"""Хранилище локальных признаков каталога.

Зачем оно нужно, видно из замера сквозного запроса. Ре-ранкинг занимал 10–15 секунд, и почти
всё это время уходило не на сопоставление точек, а на подготовку кандидатов: для каждой из
25 карточек заново открывалась картинка, прогонялся каскад детекторов и считались признаки
XFeat. Всё это — свойства каталога, а не запроса: они не меняются от того, кто и что снял.
Значит, считаться должны один раз, при сборке индекса.

В бенчмарках эта стоимость была невидима. Там дескрипторы оседали в памяти процесса и
размазывались по 94 запросам, поэтому в среднем выходило дёшево. В сервисе первый же запрос
платит за всё сам, и платит по полной.

Формат — по файлу `.npz` на карточку рядом с индексом. Один общий файл был бы компактнее,
но потребовал бы держать в памяти весь каталог: 1024 карточки это около 550 МБ. Отдельные
файлы читаются за миллисекунды и только те, что реально понадобились.
"""

import re
from pathlib import Path

import numpy as np
import torch

# Поля, которые XFeat возвращает из detectAndCompute и которых ждёт LighterGlue.
TENSOR_FIELDS = ("keypoints", "scores", "descriptors")

_UNSAFE = re.compile(r"[^\w.\-]", re.UNICODE)


def safe_name(item_id: str) -> str:
    """Идентификатор карточки -> имя файла. Идентификаторы приходят из имён папок и из чужого
    каталога, поэтому полагаться на их безопасность нельзя."""
    return _UNSAFE.sub("_", item_id)[:200]


class DescriptorStore:
    """Признаки карточек каталога: запись при сборке индекса, ленивое чтение при поиске."""

    def __init__(self, directory: str | Path, device: torch.device | None = None):
        self.directory = Path(directory)
        self.device = device or torch.device("cpu")
        self._memory: dict[str, dict] = {}

    def path_for(self, item_id: str) -> Path:
        return self.directory / f"{safe_name(item_id)}.npz"

    def exists(self, item_id: str) -> bool:
        return self.path_for(item_id).exists()

    def save(self, item_id: str, features: dict) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        arrays = {name: features[name].cpu().numpy() for name in TENSOR_FIELDS}
        arrays["image_size"] = np.asarray(features["image_size"], dtype=np.int32)
        np.savez(self.path_for(item_id), **arrays)

    def load(self, item_id: str) -> dict | None:
        if item_id in self._memory:
            return self._memory[item_id]

        path = self.path_for(item_id)
        if not path.exists():
            return None

        with np.load(path) as data:
            features = {
                name: torch.from_numpy(data[name]).to(self.device) for name in TENSOR_FIELDS
            }
            features["image_size"] = tuple(int(x) for x in data["image_size"])

        self._memory[item_id] = features
        return features

    def __len__(self) -> int:
        return len(list(self.directory.glob("*.npz"))) if self.directory.exists() else 0
