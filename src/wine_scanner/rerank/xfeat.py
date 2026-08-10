"""Ре-ранкинг локальными признаками: XFeat + RANSAC.

Глобальный эмбеддинг отвечает на вопрос «похоже ли». Здесь отвечаем на более строгий:
«это буквально та же самая этикетка». Ищем на обоих изображениях характерные точки — углы
букв, элементы орнамента — сопоставляем их и проверяем, укладывается ли соответствие
в одно геометрическое преобразование. Случайные совпадения в него не укладываются.

XFeat (CVPR 2024, Apache 2.0) выбран вместо напрашивающегося SuperPoint: у того
некоммерческая лицензия без возможности купить коммерческую. Разбор в LICENSES.md.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

MIN_POINTS_FOR_HOMOGRAPHY = 4


@dataclass
class MatchFeatures:
    """Признаки одной пары «запрос — кандидат».

    Все они пойдут на Э10 в решающий слой, поэтому возвращаем набор, а не одно число:
    LightGBM сам разберётся, как их взвесить, а мы на этом этапе не угадываем.
    """

    matches: int  # сколько точек сопоставилось по дескрипторам
    inliers: int  # сколько из них согласовано единой гомографией
    inlier_ratio: float
    reproj_error: float  # средняя ошибка репроекции по инлаерам, пиксели
    homography_ok: bool  # не выродилась ли найденная геометрия

    @property
    def score(self) -> float:
        """Насколько уверенно пара подтверждена геометрией."""
        return 0.0 if not self.homography_ok else float(self.inliers)


EMPTY = MatchFeatures(0, 0, 0.0, 0.0, False)


class XFeatMatcher:
    def __init__(
        self,
        device: torch.device | None = None,
        top_k: int = 2048,
        max_side: int = 640,
        matcher: str = "lighterglue",
        min_cossim: float = 0.82,
    ):
        self.device = device or torch.device("cpu")
        self.top_k = top_k
        self.max_side = max_side
        self.matcher = matcher
        self.min_cossim = min_cossim
        self.model = torch.hub.load(
            "verlab/accelerated_features", "XFeat", pretrained=True, top_k=top_k, trust_repo=True
        )
        self._cache: dict[str, dict] = {}

    def _prepare(self, image: Image.Image) -> np.ndarray:
        """Ужать до max_side по длинной стороне.

        Полное разрешение здесь не нужно и вредно: точек станет на порядок больше, время
        вырастет так же, а различать этикетки помогает не размер, а взаимное расположение
        деталей.
        """
        width, height = image.size
        scale = self.max_side / max(width, height)
        if scale < 1:
            image = image.resize((int(width * scale), int(height * scale)), Image.BICUBIC)
        return np.asarray(image)

    def cached(self, cache_key: str) -> dict | None:
        """Готовые признаки, если они уже считались. Позволяет не открывать и не резать картинку
        кандидата ради того, чтобы затем выбросить результат."""
        return self._cache.get(cache_key)

    def describe(self, image: Image.Image, cache_key: str | None = None) -> dict:
        if cache_key is not None and cache_key in self._cache:
            return self._cache[cache_key]

        array = self._prepare(image)
        features = self.model.detectAndCompute(array, top_k=self.top_k)[0]
        # LighterGlue нормирует координаты на размер кадра, без этого поля он не работает.
        features["image_size"] = (array.shape[1], array.shape[0])
        if cache_key is not None:
            self._cache[cache_key] = features
        return features

    def _correspondences(self, query: dict, candidate: dict):
        """Сопоставить точки: обученным матчером LighterGlue или перебором по дескрипторам.

        LighterGlue смотрит на весь набор точек сразу и учитывает их взаимное расположение,
        поэтому отсеивает ложные пары там, где простое сравнение дескрипторов их пропускает.
        Он из того же репозитория, что XFeat, и под той же лицензией Apache 2.0.
        """
        if self.matcher == "lighterglue":
            idx = self.model.match_lighterglue(query, candidate)
            return idx[0].astype(np.float32), idx[1].astype(np.float32)

        idx_q, idx_c = self.model.match(
            query["descriptors"], candidate["descriptors"], min_cossim=self.min_cossim
        )
        return (
            query["keypoints"][idx_q].cpu().numpy().astype(np.float32),
            candidate["keypoints"][idx_c].cpu().numpy().astype(np.float32),
        )

    def match(self, query: dict, candidate: dict) -> MatchFeatures:
        """Сопоставить два набора точек и проверить их геометрией."""
        points_q, points_c = self._correspondences(query, candidate)
        n_matches = len(points_q)
        if n_matches < MIN_POINTS_FOR_HOMOGRAPHY:
            return MatchFeatures(n_matches, 0, 0.0, 0.0, False)

        # RANSAC перебирает случайные четвёрки точек, строит по ним гомографию и смотрит,
        # сколько остальных точек в неё укладываются. Правильная пара даёт одну согласованную
        # модель, случайная — не даёт никакой.
        homography, mask = cv2.findHomography(
            points_c, points_q, cv2.USAC_MAGSAC, ransacReprojThreshold=4.0, maxIters=2000
        )
        if homography is None or mask is None:
            return MatchFeatures(n_matches, 0, 0.0, 0.0, False)

        mask = mask.ravel().astype(bool)
        inliers = int(mask.sum())
        if inliers < MIN_POINTS_FOR_HOMOGRAPHY:
            return MatchFeatures(n_matches, inliers, inliers / n_matches, 0.0, False)

        projected = cv2.perspectiveTransform(points_c[mask].reshape(-1, 1, 2), homography)
        error = float(np.linalg.norm(projected.reshape(-1, 2) - points_q[mask], axis=1).mean())

        return MatchFeatures(
            matches=n_matches,
            inliers=inliers,
            inlier_ratio=inliers / n_matches,
            reproj_error=error,
            homography_ok=_homography_sane(homography),
        )


def _homography_sane(homography: np.ndarray) -> bool:
    """Отсеять вырожденную геометрию.

    RANSAC иногда находит «согласованную» модель, которая схлопывает картинку в полоску или
    выворачивает её наизнанку. Формально инлаеры есть, физически такого преобразования между
    двумя фото одной этикетки быть не может. Проверяем определитель верхней части матрицы:
    он равен изменению площади, и разумные значения лежат в пределах примерно от 1/25 до 25.
    """
    determinant = float(np.linalg.det(homography[:2, :2]))
    return 0.04 < abs(determinant) < 25.0


def features_to_dict(features: MatchFeatures) -> dict:
    return asdict(features)


def image_key(path: Path) -> str:
    return str(path.resolve())
