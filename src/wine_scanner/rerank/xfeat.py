"""Ре-ранкинг локальными признаками: XFeat + RANSAC.

Глобальный эмбеддинг отвечает на вопрос «похоже ли». Здесь отвечаем на более строгий:
«это буквально та же самая этикетка». Ищем на обоих изображениях характерные точки — углы
букв, элементы орнамента — сопоставляем их и проверяем, укладывается ли соответствие
в одно геометрическое преобразование. Случайные совпадения в него не укладываются.

XFeat (CVPR 2024, Apache 2.0) выбран вместо напрашивающегося SuperPoint: у того
некоммерческая лицензия без возможности купить коммерческую.
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
    CatBoost обучается взвешивать эти признаки по размеченным парам.
    """

    matches: int  # сколько точек сопоставилось по дескрипторам
    inliers: int  # сколько из них согласовано единой гомографией
    inlier_ratio: float
    reproj_error: float  # средняя ошибка репроекции по инлаерам, пиксели
    homography_ok: bool  # не выродилась ли найденная геометрия

    # Сама матрица перехода от кадра кандидата к кадру запроса, в координатах уменьшенных
    # изображений (см. _prepare). В признаки решающего слоя не идёт — она нужна Э8: зная,
    # где год напечатан на эталоне, по ней находим тот же участок на нашей фотографии.
    homography: np.ndarray | None = None
    # Размеры кадров, в которых посчитана матрица. Без них она бесполезна: координаты в неё
    # входят не в долях, а в пикселях уменьшенных копий.
    query_size: tuple[int, int] | None = None
    candidate_size: tuple[int, int] | None = None
    query_coverage: float = 0.0
    candidate_coverage: float = 0.0
    normalized_reproj_error: float = 0.0

    @property
    def score(self) -> float:
        """Насколько уверенно пара подтверждена геометрией."""
        return 0.0 if not self.homography_ok else float(self.inliers)


EMPTY = MatchFeatures(0, 0, 0.0, 0.0, False)

# Поля, которые не являются признаками: матрица геометрии для проекции области года.
GEOMETRY_FIELDS = ("homography", "query_size", "candidate_size")


# Минимальная сторона входа XFeat: его сеть работает на сетке 32 px.
MIN_SIDE = 32


class XFeatMatcher:
    def __init__(
        self,
        device: torch.device | None = None,
        top_k: int = 2048,
        max_side: int = 640,
        matcher: str = "lighterglue",
        min_cossim: float = 0.82,
    ):
        self.top_k = top_k
        self.max_side = max_side
        self.matcher = matcher
        self.min_cossim = min_cossim
        self.model = torch.hub.load(
            "verlab/accelerated_features", "XFeat", pretrained=True, top_k=top_k, trust_repo=True
        )

        # XFeat выбирает устройство сам, в конструкторе: cuda, если она есть, иначе cpu.
        # Параметра для этого у него нет, а знать его выбор обязательно — дескрипторы каталога
        # мы читаем с диска и кладём на `self.device`. Разойдись эти два устройства, и на
        # машине с GPU всё падало бы при первом же сопоставлении, причём далеко от причины.
        self.device = torch.device(device) if device is not None else self.model.dev
        if self.matcher == "lighterglue" and self.model.lighterglue is None:
            from importlib import import_module

            package = type(self.model).__module__.rsplit(".", 1)[0]
            self.model.lighterglue = import_module(f"{package}.lighterglue").LighterGlue()
        if self.device != self.model.dev:
            self.model.dev = self.device
            self.model.net = self.model.net.to(self.device)
            if getattr(self.model, "lighterglue", None) is not None:
                self.model.lighterglue = self.model.lighterglue.to(self.device)

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
        # XFeat округляет стороны вниз до кратных 32 и делит на результат: кроп уже 32 px
        # роняет его нулём. Такой кроп — почти всегда ошибка детектора (на каталоге платформы
        # он однажды выбрал полоску 28×107 на бутылке 294×1000), но падать сборке из-за него
        # нельзя; растягиваем до минимума, точек на нём всё равно не будет.
        width, height = image.size
        if min(width, height) < MIN_SIDE:
            scale = MIN_SIDE / min(width, height)
            image = image.resize(
                (max(MIN_SIDE, round(width * scale)), max(MIN_SIDE, round(height * scale))),
                Image.BICUBIC,
            )
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
            try:
                idx = self.model.match_lighterglue(query, candidate)
            except IndexError:
                # LighterGlue по пути отбрасывает неуверенные точки и на редких парах приходит
                # к пустой матрице оценок — kornia падает редукцией по пустой оси. Для нас это
                # просто пара без совпадений.
                empty = np.zeros((0, 2), dtype=np.float32)
                return empty, empty
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
        # Пустой набор точек (крошечный или однотонный кроп) роняет LighterGlue внутри kornia
        # ошибкой редукции по пустой оси. Сопоставлять здесь нечего — пара без совпадений.
        if len(query.get("keypoints", ())) < 2 or len(candidate.get("keypoints", ())) < 2:
            return MatchFeatures(0, 0, 0.0, 0.0, False)
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

        def coverage(points, size):
            if not size or min(size) <= 0:
                return 0.0
            return min(1.0, float(cv2.contourArea(cv2.convexHull(points))) / (size[0] * size[1]))

        qsize = query.get("image_size", ())
        csize = candidate.get("image_size", ())

        return MatchFeatures(
            matches=n_matches,
            inliers=inliers,
            inlier_ratio=inliers / n_matches,
            reproj_error=error,
            homography_ok=_homography_sane(homography),
            homography=homography,
            # Размеры отдаются только вместе с матрицей: порознь они бессмысленны, а из
            # ранних выходов выше возвращать нечего — гомографии там нет.
            query_size=tuple(query.get("image_size", ())) or None,
            candidate_size=tuple(candidate.get("image_size", ())) or None,
            query_coverage=coverage(points_q[mask], qsize),
            candidate_coverage=coverage(points_c[mask], csize),
            normalized_reproj_error=error / max(float(np.linalg.norm(qsize)), 1.0),
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
    """Только числовые признаки. Геометрия отбрасывается: матрица не сериализуется в json
    и в решающем слое ей делать нечего."""
    return {k: v for k, v in asdict(features).items() if k not in GEOMETRY_FIELDS}


def image_key(path: Path) -> str:
    return str(path.resolve())
