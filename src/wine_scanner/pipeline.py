"""Сквозной путь: фотография на входе, карточка вина и вероятность на выходе.

Единственная точка входа в систему. До этого модуля весь пайплайн существовал только внутри
eval/: бенчмарки собирали те же блоки заново, каждый по-своему, с предпосчитанными кэшами.
Так можно доводить качество, но нельзя ничего показать и не на чем мерить настоящую задержку.

Порядок блоков ровно тот, что закреплён замерами (см. CLAUDE.md и PLAN.md):

    кадр -> каскадная обрезка -> DINOv2 -> FAISS (50 кандидатов)
                              \\-> OCR -> BM25 + fuzzy (50 кандидатов)
         слияние RRF -> 25 кандидатов -> XFeat + RANSAC -> LightGBM -> вероятность -> ответ

Три вещи, о которых стоит помнить, читая код.

Первое: тяжёлые блоки поднимаются лениво и живут в объекте, а не создаются на запрос. Загрузка
DINOv2, детектора, EasyOCR и XFeat занимает десятки секунд — в сервисе это делается один раз
на старте, иначе первый же запрос упрётся в таймаут.

Второе: у каждого запроса собирается разбивка времени по блокам. Не для красоты — Э11 требует
p95 < 2 с, а оптимизировать без разбивки означает угадывать.

Третье: конфигурация обрезки и приведения к квадрату берётся из config.json рядом с индексом.
Запрос обязан пройти ровно тот же путь, что и карточки каталога, иначе векторы окажутся из
разных пространств, а выглядеть это будет просто как «плохо ищет».
"""

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .burst import fuse_rrf, ranked, sharpness
from .decide import Decider, PairFeatures
from .detect import BottleDetector, CachedCropper, CascadeCropper
from .embed import DEFAULT_MODEL, Dinov2Embedder, load_image, pick_device
from .index import VectorIndex
from .ocr import LabelOCR, TextIndex, catalog_document
from .rerank import DescriptorStore, XFeatMatcher

INDEX_DIR = Path("models/index")
DECIDER_DIR = Path("models/decider")
LABEL_WEIGHTS = Path("models/label_detector.pt")

VISUAL_CANDIDATES = 50
TEXT_CANDIDATES = 50
RERANK_CANDIDATES = 25


@dataclass
class Candidate:
    """Кандидат после решающего слоя."""

    item_id: str
    probability: float
    payload: dict
    features: dict = field(default_factory=dict)

    @property
    def card(self) -> dict:
        """Карточка для показа: всё, кроме служебных полей."""
        return {k: v for k, v in self.payload.items() if k != "image_path"}


@dataclass
class ScanResult:
    """Ответ системы на один снимок.

    `answered` отделено от `best` намеренно. Лучший кандидат есть всегда — индекс что-нибудь
    да вернёт, — но показывать его пользователю можно, только если вероятность выше порога.
    Иначе честнее сказать «не узнал», чем уверенно назвать чужое вино.
    """

    answered: bool
    best: Candidate | None
    candidates: list[Candidate]
    text: str
    timings: dict[str, float]
    threshold: float
    frames: int = 1

    def to_dict(self) -> dict:
        # При отказе поля ответа пустые, а кандидаты остаются: клиенту есть что показать
        # («возможно, одно из этих»), но выдать догадку за ответ он уже не сможет.
        answer = self.best if self.answered else None
        return {
            "answered": self.answered,
            "probability": self.best.probability if self.best else 0.0,
            "item_id": answer.item_id if answer else None,
            "card": answer.card if answer else None,
            "candidates": [
                {"item_id": c.item_id, "probability": c.probability, "card": c.card}
                for c in self.candidates
            ],
            "recognized_text": self.text,
            "threshold": self.threshold,
            "frames": self.frames,
            "timings_ms": {k: round(v * 1000, 1) for k, v in self.timings.items()},
        }


class WineScanner:
    """Собранный пайплайн. Создаётся один раз, дальше отвечает на запросы.

    Компоненты можно передать готовыми — этим пользуются тесты, чтобы проверить логику
    сборки признаков и порога отказа, не поднимая четыре нейросети.
    """

    def __init__(
        self,
        index_dir: Path = INDEX_DIR,
        decider_dir: Path = DECIDER_DIR,
        weights: Path = LABEL_WEIGHTS,
        candidates: int = RERANK_CANDIDATES,
        threshold: float | None = None,
        device=None,
        crop_cache: Path | None = None,
        ocr_cache: bool = False,
        embedder=None,
        index=None,
        text_index=None,
        ocr=None,
        matcher=None,
        decider=None,
    ):
        self.candidates = candidates
        # Кэш OCR по умолчанию выключен. Он полезен при повторных прогонах по одним и тем же
        # файлам, но замер задержки с ним показывает не работу системы, а скорость чтения
        # json-а — именно так и родилась цифра 2 секунды, в которую мы верили.
        self.ocr_cache = ocr_cache
        self.index = index if index is not None else VectorIndex.load(index_dir)
        self.decider = decider if decider is not None else Decider.load(decider_dir)
        self.threshold = self.decider.threshold if threshold is None else threshold

        config_path = Path(index_dir) / "config.json"
        self.config = (
            json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        )

        if embedder is None:
            device = device or pick_device()
            embedder = Dinov2Embedder(
                model_name=self.config.get("model", DEFAULT_MODEL),
                device=device,
                cropper=self._build_cropper(weights, device, crop_cache),
                fit=self.config.get("fit", "pad"),
            )
        self.embedder = embedder
        self.cropper = getattr(embedder, "cropper", None)

        self.ocr = ocr if ocr is not None else LabelOCR()
        self.matcher = matcher if matcher is not None else XFeatMatcher()
        self.text_index = (
            text_index
            if text_index is not None
            else TextIndex(
                self.index.item_ids, [catalog_document(p) for p in self.index.payloads]
            )
        )
        descriptor_dir = Path(index_dir) / "descriptors"
        self.descriptors = (
            DescriptorStore(descriptor_dir, device=self.matcher.device)
            if descriptor_dir.exists()
            else None
        )
        # Счётчик кандидатов, признаки которых пришлось считать на лету. В норме он остаётся
        # нулём; выросший означает, что индекс и дескрипторы разошлись.
        self.recomputed_descriptors = 0

        self.path_by_id = {
            item_id: payload.get("image_path")
            for item_id, payload in zip(self.index.item_ids, self.index.payloads, strict=True)
        }
        self.payload_by_id = dict(
            zip(self.index.item_ids, self.index.payloads, strict=True)
        )

    def _build_cropper(self, weights: Path, device, crop_cache: Path | None):
        """Обрезка ровно та же, которой строился индекс."""
        detect = self.config.get("detect", "cascade")
        if detect == "cascade":
            detector = CascadeCropper(
                BottleDetector(device=device, mode="bottle"),
                BottleDetector(device=device, weights_path=weights),
            )
        elif detect == "trained":
            detector = BottleDetector(device=device, weights_path=weights)
        else:
            detector = BottleDetector(device=device, mode=detect)
        # Кэш кропов ключуется путём файла и его временем правки — для загруженного по сети
        # снимка это бессмысленно, поэтому в сервисе он выключен.
        return CachedCropper(detector, crop_cache) if crop_cache else detector

    def _crop(self, image: Image.Image, key: str | None) -> Image.Image:
        if isinstance(self.cropper, CachedCropper) and key:
            return self.cropper(Path(key), image)
        if self.cropper is None:
            return image
        return self.cropper.crop(image)

    def _candidate_descriptor(self, item_id: str) -> dict | None:
        """Локальные признаки карточки каталога.

        Штатный путь — чтение готового файла, положенного при сборке индекса. Пересчёт на лету
        оставлен запасным вариантом: он работает, но стоит открытия картинки и прогона каскада
        детекторов на каждого кандидата, то есть тех самых секунд, ради которых всё и затевалось.
        """
        if self.descriptors is not None:
            stored = self.descriptors.load(item_id)
            if stored is not None:
                return stored

        path = self.path_by_id.get(item_id)
        if not path:
            return None
        key = str(path)
        cached = self.matcher.cached(key)
        if cached is not None:
            return cached
        self.recomputed_descriptors += 1
        return self.matcher.describe(self._crop(load_image(path), key), cache_key=key)

    def identify(self, image: Image.Image, image_key: str | None = None) -> ScanResult:
        """Опознать вино по одному кадру."""
        timings: dict[str, float] = {}

        @contextmanager
        def stage(name: str):
            start = time.perf_counter()
            yield
            timings[name] = time.perf_counter() - start

        with stage("crop"):
            crop = self._crop(image, image_key)

        with stage("embed"):
            vector = self.embedder.encode_image(crop).numpy()

        with stage("search"):
            visual = self.index.search(vector, top_k=VISUAL_CANDIDATES)

        with stage("ocr"):
            lines = self.ocr.read(crop, use_cache=self.ocr_cache and image_key is not None)
            text = LabelOCR.joined(lines)
            confidence = sum(line.confidence for line in lines) / len(lines) if lines else 0.0

        with stage("text_search"):
            textual = self.text_index.search(text, top_k=TEXT_CANDIDATES) if text else []

        # Слияние веток. RRF складывает позиции, а не оценки, поэтому шкалы приводить не нужно:
        # у визуальной ветки это косинус, у текстовой — смесь BM25 и fuzzy.
        fused = fuse_rrf([[h.item_id for h in visual], [h.item_id for h in textual]])
        order = ranked(fused)[: self.candidates]

        vis_rank = {h.item_id: i for i, h in enumerate(visual)}
        vis_score = {h.item_id: h.score for h in visual}
        txt_rank = {h.item_id: i for i, h in enumerate(textual)}
        txt_score = {h.item_id: h.score for h in textual}

        with stage("rerank"):
            query_features = self.matcher.describe(crop)
            rows = []
            for position, item_id in enumerate(order):
                candidate = self._candidate_descriptor(item_id)
                match = self.matcher.match(query_features, candidate) if candidate else None
                rows.append(
                    PairFeatures(
                        item_id=item_id,
                        vis_score=vis_score.get(item_id, 0.0),
                        vis_rank=vis_rank.get(item_id, 999),
                        txt_score=txt_score.get(item_id, -1.0),
                        txt_rank=txt_rank.get(item_id, 999),
                        rrf_rank=position,
                        matches=match.matches if match else 0,
                        inliers=match.inliers if match else 0,
                        inlier_ratio=match.inlier_ratio if match else 0.0,
                        reproj_error=match.reproj_error if match else 0.0,
                        homography_ok=int(match.homography_ok) if match else 0,
                        ocr_lines=len(lines),
                        ocr_conf=confidence,
                    )
                )

        with stage("decide"):
            scored = self.decider.score(rows)

        candidates = [
            Candidate(
                item_id=s.item_id,
                probability=s.probability,
                payload=self.payload_by_id.get(s.item_id, {}),
                features=s.features,
            )
            for s in scored
        ]
        best = candidates[0] if candidates else None
        timings["total"] = sum(v for k, v in timings.items())

        return ScanResult(
            answered=bool(best and best.probability >= self.threshold),
            best=best,
            candidates=candidates,
            text=text,
            timings=timings,
            threshold=self.threshold,
        )

    def identify_path(self, path: str | Path) -> ScanResult:
        path = Path(path)
        return self.identify(load_image(path), image_key=str(path))

    def identify_burst(self, images: list[Image.Image]) -> ScanResult:
        """Опознать вино по серии кадров.

        Из серии берётся самый резкий кадр, а не объединяются результаты по всем. Так решено
        по замерам Э12: объединение оценок (max, mean, RRF) работает хуже отбора и стоит в
        разы дороже — каждый лишний кадр это ещё один прогон всего пайплайна.
        """
        if not images:
            raise ValueError("серия кадров пуста")
        start = time.perf_counter()
        best_frame = max(images, key=sharpness)
        picked = time.perf_counter() - start

        result = self.identify(best_frame)
        result.frames = len(images)
        result.timings["pick_frame"] = picked
        result.timings["total"] += picked
        return result


def load_scanner(**kwargs) -> WineScanner:
    """Поднять пайплайн со всеми моделями. Занимает десятки секунд — вызывать один раз."""
    return WineScanner(**kwargs)


__all__ = ["Candidate", "ScanResult", "WineScanner", "load_scanner"]
