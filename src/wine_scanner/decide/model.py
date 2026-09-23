"""Обученный решающий слой: хранение на диске и применение на инференсе.

До этого модуля решающий слой существовал только внутри eval/decide_benchmark.py — обучался,
печатал метрики и умирал вместе с процессом. Для сервиса нужна ровно обратная вещь: обучить
один раз, положить на диск, поднимать за миллисекунды.

Формат хранения без pickle:

* `model.cbm` — родной бинарный формат CatBoost;
* `meta.json` — коэффициенты калибровки, порог, порядок признаков и метрики прогона.

Pickle-файл sklearn перестаёт открываться после обновления библиотеки, причём молча и в самый
неудобный момент. Калибровка Платта — это два числа, хранить ради них слепок объекта незачем.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES, FEATURE_VERSION, PairFeatures, derive

DEFAULT_DIR = Path("models/decider")


@dataclass
class Scored:
    """Кандидат после решающего слоя."""

    item_id: str
    probability: float  # калиброванная вероятность того, что кандидат верен
    raw: float  # score ranker'а; при старой модели совпадает с сырым score classifier
    features: dict  # всё, что участвовало в решении, — для отладки и объяснения ответа


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Вероятность -> логарифм отношения шансов.

    Обрезка нужна, чтобы 0 и 1 не превратились в бесконечность.
    """
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


class Decider:
    """Ranker для порядка кандидатов и classifier с калибровкой для решения об отказе.

    Калибровка обучается на отложенных предсказаниях лучшего кандидата (см. scripts/train_decider),
    поэтому строго говоря она верна именно для победителя. Остальным кандидатам она тоже
    применяется — порядок от этого не меняется, преобразование монотонное, — но читать их числа
    как вероятности не стоит.
    """

    def __init__(
        self,
        booster,
        calib_weight: float,
        calib_bias: float,
        threshold: float,
        feature_names: tuple[str, ...] = FEATURE_NAMES,
        meta: dict | None = None,
        ranker=None,
    ):
        self.booster = booster
        self.calib_weight = calib_weight
        self.calib_bias = calib_bias
        self.threshold = threshold
        self.feature_names = tuple(feature_names)
        self.meta = meta or {}
        self.ranker = ranker

        # New derived features can be appended without changing the meaning of old columns.
        # Keep old validated classifier artefacts usable until a ranker passes evaluation;
        # a reordered or altered prefix is still unsafe and is rejected.
        if self.feature_names != FEATURE_NAMES[: len(self.feature_names)]:
            raise ValueError(
                "порядок признаков в сохранённой модели не совпадает с кодом: "
                f"{self.feature_names} не является началом {FEATURE_NAMES}. "
                "Модель надо переобучить "
                "(scripts/build_platform_features.py, затем scripts/train_decider.py) — "
                "иначе бустер получит колонки не на своих местах и молча начнёт врать."
            )
        version = self.meta.setdefault("feature_version", FEATURE_VERSION)
        if version not in {3, FEATURE_VERSION}:
            raise ValueError(
                f"решающий слой обучен на неподдерживаемой версии признаков {version}. "
                "Переобучить: scripts/build_platform_features.py, "
                "затем scripts/train_decider.py."
            )

    def calibrate(self, raw: np.ndarray) -> np.ndarray:
        """Сырой выход бустера -> вероятность правильности.

        Преобразование идёт через логит, а не по самой вероятности, и это не украшение.
        Логистическая регрессия, обученная прямо на числе из [0, 1], растягивает его в лучшем
        случае линейно, и весь верх шкалы схлопывается: на нашем наборе все уверенные ответы
        оказались между 0.909 и 0.910, то есть порог отказа стало невозможно выставить в
        принципе — сетка с шагом 0.01 их просто не различала. По логиту шкала распрямляется,
        и те же ответы расходятся по диапазону от 0.12 до 0.99.
        """
        return sigmoid(self.calib_weight * logit(raw) + self.calib_bias)

    def score(self, rows: list[PairFeatures]) -> list[Scored]:
        """Оценить кандидатов и вернуть порядок ranker'а с confidence для каждого."""
        if not rows:
            return []
        derived = derive(rows)
        values = np.asarray([[float(row[name]) for name in self.feature_names] for row in derived])
        if hasattr(self.booster, "predict_proba"):
            confidence_raw = np.asarray(self.booster.predict_proba(values)[:, 1], dtype=float)
        else:
            confidence_raw = np.asarray(self.booster.predict(values), dtype=float).ravel()
        # CatBoostRanker возвращает не вероятность, а относительную оценку внутри запроса.
        # Его нельзя подставлять в порог отказа: для этого остаётся отдельно обученный
        # classifier, калиброванный по OOF-победителям.
        ranking_raw = (
            np.asarray(self.ranker.predict(values), dtype=float).ravel()
            if self.ranker is not None
            else confidence_raw
        )
        probability = self.calibrate(confidence_raw)
        scored = [
            Scored(row.item_id, float(p), float(r), features)
            for row, features, p, r in zip(rows, derived, probability, ranking_raw, strict=True)
        ]
        scored.sort(key=lambda s: s.raw, reverse=True)
        return scored

    def save(self, directory: str | Path = DEFAULT_DIR) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(directory / "model.cbm"))
        if self.ranker is not None:
            self.ranker.save_model(str(directory / "ranker.cbm"))
        (directory / "meta.json").write_text(
            json.dumps(
                {
                    "backend": "catboost",
                    "ranking_backend": "catboost_ranker" if self.ranker is not None else None,
                    "calib_weight": self.calib_weight,
                    "calib_bias": self.calib_bias,
                    "threshold": self.threshold,
                    "feature_names": list(self.feature_names),
                    "feature_version": FEATURE_VERSION,
                    **{k: v for k, v in self.meta.items() if k != "feature_version"},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path = DEFAULT_DIR) -> "Decider":
        from catboost import CatBoostClassifier, CatBoostRanker

        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        if meta.get("backend") != "catboost":
            raise ValueError("Rebuild decider with CatBoost and current feature schema")
        booster = CatBoostClassifier()
        booster.load_model(str(directory / "model.cbm"))
        ranker = None
        if meta.get("ranking_backend") == "catboost_ranker":
            ranker_path = directory / "ranker.cbm"
            if not ranker_path.exists():
                raise ValueError("В metadata указан CatBoostRanker, но ranker.cbm отсутствует")
            ranker = CatBoostRanker()
            ranker.load_model(str(ranker_path))
        return cls(
            booster=booster,
            calib_weight=meta["calib_weight"],
            calib_bias=meta["calib_bias"],
            threshold=meta["threshold"],
            feature_names=tuple(meta["feature_names"]),
            meta={k: v for k, v in meta.items() if k not in {"calib_weight", "calib_bias"}},
            ranker=ranker,
        )
