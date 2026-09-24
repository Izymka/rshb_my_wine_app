"""Сквозной путь: фотография на входе, карточка вина и вероятность на выходе.

Единственная точка входа в систему. До этого модуля весь пайплайн существовал только внутри
eval/: бенчмарки собирали те же блоки заново, каждый по-своему, с предпосчитанными кэшами.
Так можно доводить качество, но нельзя ничего показать и не на чем мерить настоящую задержку.

Порядок блоков ровно тот, что закреплён замерами (см. CLAUDE.md и PLAN.md):

    кадр -> RT-DETR бутылка + этикетка -> SigLIP 2 (+whitening) -> FAISS
                              \\-> OCR -> n-граммы + покрытие слов (50 кандидатов)
         слияние RRF + родня по винодельне -> длинный список (текстовые сигналы всем)
         -> окно 25 + подтверждённые текстом сиблинги -> XFeat + RANSAC
         -> CatBoost -> защита от близнеца -> вероятность -> ответ

Длинный список и окно — два яруса кандидатов (16.09.2026, каталог платформы). Близнецы
внутри линейки различимы словами, а не картинкой, и слова считаются дёшево: поэтому
текстовые сигналы получают все кандидаты длинного списка, включая родню по винодельне
верхних визуальных и текстовых совпадений. Дорогое сопоставление локальными признаками
достаётся только окну — верхушке слитого порядка плюс тем, кого текст подтвердил отдельно.

Три вещи, о которых стоит помнить, читая код.

Первое: тяжёлые блоки поднимаются лениво и живут в объекте, а не создаются на запрос. Загрузка
SigLIP 2, детектора, PaddleOCR и XFeat занимает десятки секунд — в сервисе это делается один раз
на старте, иначе первый же запрос упрётся в таймаут.

Второе: у каждого запроса собирается разбивка времени по блокам. Не для красоты — Э11 требует
p95 < 2 с, а оптимизировать без разбивки означает угадывать.

Третье: конфигурация обрезки и приведения к квадрату берётся из config.json рядом с индексом.
Запрос обязан пройти ровно тот же путь, что и карточки каталога, иначе векторы окажутся из
разных пространств, а выглядеть это будет просто как «плохо ищет».
"""

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .analogues import Analogues
from .burst import fuse_rrf, ranked, sharpness
from .decide import Decider, PairFeatures, Scored, derive
from .decide.features import text_pair_features
from .decide.guard import DEFAULT_MODE as DEFAULT_GUARD
from .decide.guard import SIBLING_ENABLED, sibling_swap, twin_guard
from .decide.judge import VlmJudge
from .detect import COCO_BOTTLE_MODEL, CachedCropper, build_cropper
from .embed import Whitening, build_embedder, load_image, pick_device
from .embed.siglip import DEFAULT_SIGLIP_MODEL as DEFAULT_MODEL
from .index import VectorIndex
from .ocr import LabelOCR, TextIndex
from .rerank import DescriptorStore, XFeatMatcher
from .vintage import Answer, VintageReader, compare, project, resolve

INDEX_DIR = Path("models/index")
DECIDER_DIR = Path("models/decider")
LABEL_WEIGHTS = Path("models/rtdetr_label")


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


VISUAL_CANDIDATES = _env_int("WINE_VISUAL_CANDIDATES", 100)
TEXT_CANDIDATES = _env_int("WINE_TEXT_CANDIDATES", 50)
RERANK_CANDIDATES = _env_int("WINE_RERANK_CANDIDATES", 25)
# Расширение по винодельне: у скольких лидеров каждой ветки берём семью и сколько карточек
# семьи добавляем. 15 — больше, чем у типичной линейки, и меньше, чем у Фанагории целиком.
FAMILY_SEEDS = 3
FAMILY_CAP = _env_int("WINE_FAMILY_CAP", 15)
# Кандидат вне окна, у которого этикетка подтвердила хотя бы половину различающих слов,
# попадает в окно: текстом подтверждённый сиблинг обязан получить геометрию.
DISC_WINDOW_MIN = 0.5
GUARD_EPS = 1e-3

# Сколько кандидатов уходит клиенту. Решающий слой оценивает все 25, но интерфейсу нужны
# первые несколько — на экране «возможно, одно из этих» больше и не поместится. Разница не
# косметическая: карточка весит сотни байт, и полные 25 штук это десятки килобайт на каждый
# ответ, которые едут по мобильной сети ради строк, недоступных пользователю.
REPORTED_CANDIDATES = 5

# Поля payload, которые нужны пайплайну, но не клиенту: путь к картинке каталога для
# ре-ранкинга и координаты года на карточке для Э8. В карточке вина им делать нечего.
SERVICE_FIELDS = frozenset({"image_path", "vintage_box"})


def digest(paths: list[Path]) -> str | None:
    """Короткий отпечаток набора файлов — им версионируются артефакты.

    Считается по содержимому, а не по времени правки: пересборка индекса из тех же данных
    обязана дать ту же версию, иначе поле бесполезно ровно там, где нужно, — при разборе
    «почему на прошлой неделе отвечало иначе».

    Читается всё целиком, но происходит это один раз на старте сервиса, рядом с загрузкой
    моделей, которая занимает десятки секунд. `None` означает, что файлов нет: так бывает
    в тестах, где блоки подменены заглушками и артефактов на диске не существует вовсе.
    """
    hasher = hashlib.sha1()
    found = False
    for path in paths:
        if not path.exists():
            continue
        hasher.update(path.name.encode())
        hasher.update(path.read_bytes())
        found = True
    return hasher.hexdigest()[:12] if found else None


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
        return {k: v for k, v in self.payload.items() if k not in SERVICE_FIELDS}


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
    vintage: Answer | None = None
    # Сработавшая защита («twin») или None. Отдельным полем, а не в вероятности: клиент и
    # бенчмарк должны видеть, что отказ пришёл от правила, а не от модели.
    guard: str | None = None
    # Что сказал VLM-судья, если его звали.
    judge: dict | None = None
    # Аналоги лучшего кандидата из других виноделен (analogues.py) — функция после поиска.
    analogues: list[dict] | None = None
    # Уверенность для API: калиброванная вероятность лучшего (`top1`), доля массы первых пяти
    # среди всех кандидатов (`top5`) и отрыв от второго (`margin`). ТЗ просит метрику для
    # топ-1 и топ-5 — это она.
    confidence: dict | None = None
    # Внутренности запроса для бенчмарка: вектор, полные списки обеих веток, окно, инлаеры.
    # Заполняется только по просьбе (`identify(..., trace=True)`) и наружу через to_dict
    # не уходит: сервису это лишние килобайты, а измерителю — единственный способ узнать,
    # на каком месте стояла верная карточка, не пересобирая пайплайн заново.
    trace: dict | None = None

    def to_dict(self) -> dict:
        # При отказе поля ответа пустые, а кандидаты остаются: клиенту есть что показать
        # («возможно, одно из этих»), но выдать догадку за ответ он уже не сможет.
        answer = self.best if self.answered else None
        return {
            "answered": self.answered,
            "probability": self.best.probability if self.best else 0.0,
            "item_id": answer.item_id if answer else None,
            "card": answer.card if answer else None,
            # Наружу уходит верхушка списка, а не всё окно ре-ранкинга: см. REPORTED_CANDIDATES.
            # В самом ScanResult кандидаты остаются все — на них считаются метрики.
            "candidates": [
                {"item_id": c.item_id, "probability": c.probability, "card": c.card}
                for c in self.candidates[:REPORTED_CANDIDATES]
            ],
            "recognized_text": self.text,
            # Год отдаётся только вместе с карточкой: при отказе показывать нечего, и год
            # неизвестно от какого вина будет не сведением, а поводом для путаницы.
            # `ask` — просьба к клиенту показать вопрос о годе одним тапом.
            "vintage": (
                {
                    "year": self.vintage.year,
                    "source": self.vintage.source,
                    "ask": self.vintage.ask,
                }
                if answer and self.vintage
                else None
            ),
            "threshold": self.threshold,
            "confidence": self.confidence,
            "guard": self.guard,
            "judge": self.judge,
            "analogues": self.analogues,
            "frames": self.frames,
            "timings_ms": {k: round(v * 1000, 1) for k, v in self.timings.items()},
        }


class RetrievalOnlyDecider:
    """Заглушка решающего слоя: порядок слияния веток и ничего больше.

    Нужна в двух местах. Сборка признаков: пока список признаков меняется, обученной модели
    под него ещё нет, а пайплайн должен собраться. Бенчмарк: сколько даёт сам решающий слой
    поверх простого слияния. Вероятность — 1/(1 + ранг), порог — половина, то есть «отвечаем
    всегда первым».
    """

    threshold = 0.5
    meta: dict = {"stub": "retrieval-only"}

    def score(self, rows: list[PairFeatures]) -> list[Scored]:
        derived = derive(rows)
        scored = [
            Scored(row.item_id, 1.0 / (1.0 + row.rrf_rank), -float(row.rrf_rank), features)
            for row, features in zip(rows, derived, strict=True)
        ]
        scored.sort(key=lambda s: s.raw, reverse=True)
        return scored


def confidence_of(candidates: list[Candidate]) -> dict | None:
    """Уверенность для API из калиброванных вероятностей кандидатов.

    `top1` — вероятность лучшего как есть. `top5` — доля массы первых пяти среди всех: если
    у лидера 0.9, а у остальных по 0.01, пятёрка забирает почти всё; если вероятности размазаны
    по линейке, пятёрка получает меньше, и это честный сигнал, что выбирать надо из списка.
    Калибровка точна только для победителя (см. decide/model.py), поэтому `top5` — оценка,
    а не вероятность в строгом смысле. `margin` — отрыв от второго.
    """
    if not candidates:
        return None
    probabilities = [c.probability for c in candidates]
    total = sum(probabilities) or 1.0
    return {
        "top1": probabilities[0],
        "top5": sum(probabilities[:5]) / total,
        "margin": probabilities[0] - (probabilities[1] if len(probabilities) > 1 else 0.0),
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
        weights: Path | None = None,
        candidates: int = RERANK_CANDIDATES,
        threshold: float | None = None,
        device=None,
        crop_cache: Path | None = None,
        ocr_cache: bool = False,
        parallel: bool = True,
        precision: str | None = None,
        embedder=None,
        index=None,
        text_index=None,
        ocr=None,
        matcher=None,
        decider=None,
        visual_candidates: int = VISUAL_CANDIDATES,
        text_candidates: int = TEXT_CANDIDATES,
        family_expansion: bool | None = None,
        window_extra: int | None = None,
        guard: str = DEFAULT_GUARD,
        sibling: bool = SIBLING_ENABLED,
        judge=None,
        local_preprocess: str | None = None,
        ocr_preprocess: str | None = None,
    ):
        self.candidates = candidates
        self.visual_candidates = visual_candidates
        self.text_candidates = text_candidates
        self.family_expansion = (
            bool(int(os.environ.get("WINE_FAMILY", "1")))
            if family_expansion is None
            else family_expansion
        )
        # Сколько текстом подтверждённых кандидатов можно добавить в окно сверх основного:
        # ограничение нужно, чтобы длинная линейка не удвоила стоимость ре-ранкинга.
        self.window_extra = candidates if window_extra is None else window_extra
        self.guard = guard
        # Выбор внутри семьи по тексту (decide/guard.py, sibling_swap).
        self.sibling = sibling
        # Судья создаётся из окружения только по явному WINE_VLM=1; переданный объект — как есть.
        self.judge = judge if judge is not None else VlmJudge.from_env()
        # Кэш OCR по умолчанию выключен. Он полезен при повторных прогонах по одним и тем же
        # файлам, но замер задержки с ним показывает не работу системы, а скорость чтения
        # json-а — именно так и родилась цифра 2 секунды, в которую мы верили.
        self.ocr_cache = ocr_cache
        # Текстовая ветка в отдельном потоке. Оставлено выключаемым, чтобы замерять выигрыш
        # и чтобы было куда отступить, если чужая библиотека окажется не потокобезопасной.
        self.parallel = parallel
        self.index = index if index is not None else VectorIndex.load(index_dir)
        self.decider = decider if decider is not None else Decider.load(decider_dir)
        self.threshold = self.decider.threshold if threshold is None else threshold

        config_path = Path(index_dir) / "config.json"
        self.config = (
            json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        )
        self.local_preprocess = local_preprocess or self.config.get("local_preprocess", "rgb")
        precision = precision or self.config.get("precision", "fp32")
        self.ocr_preprocess = ocr_preprocess or self.config.get("ocr_preprocess", "rgb")
        if local_preprocess and local_preprocess != self.config.get("local_preprocess", "rgb"):
            if (Path(index_dir) / "descriptors").exists():
                raise ValueError(
                    "Rebuild catalog descriptors for the requested local preprocessing"
                )

        # Whitening (Э5) лежит рядом с индексом и применяется на лету: индекс хранит сырые
        # векторы, а здесь они отбеливаются вместе с каждым запросом. Так преобразование
        # переподбирается за секунды, а не за полтора часа пересборки каталога.
        whitening_path = Path(index_dir) / "whitening.npz"
        self.whitening = Whitening.load(whitening_path) if whitening_path.exists() else None
        if self.whitening is not None:
            self.index = self._whitened(self.index, self.whitening)

        # Версия артефактов уходит в каждый ответ. Индекс и решающий слой пересобираются, и без
        # этого поля ответы, полученные на разных версиях, в логах клиента неразличимы: жалоба
        # «оно показало не то» приходит через неделю, когда на диске давно лежит другой индекс.
        index_dir, decider_dir = Path(index_dir), Path(decider_dir)
        self.version = {
            "index": digest(
                [index_dir / "vectors.faiss", index_dir / "meta.json", config_path, whitening_path]
            ),
            "decider": digest(
                [decider_dir / "model.cbm", decider_dir / "ranker.cbm", decider_dir / "meta.json"]
            ),
        }

        # Веса детектора этикетки — те же, что резали каталог при сборке индекса: иначе запрос и
        # эталон кропаются по-разному, и сравнение детекторов на бенчмарке ничего не значит.
        if weights is None:
            # Index artifacts may have been built on Windows and deployed on Linux.
            weights = Path(str(self.config.get("weights", LABEL_WEIGHTS)).replace("\\", "/"))
        self.weights = weights
        if embedder is None:
            expected_hash = self.config.get("label_sha256")
            if expected_hash:
                with (Path(weights) / "model.safetensors").open("rb") as stream:
                    actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual_hash != expected_hash:
                    raise ValueError("Label weights changed: rebuild the index and decider")
            device = device or pick_device()
            embedder = build_embedder(
                model_name=self.config.get("model", DEFAULT_MODEL),
                device=device,
                cropper=self._build_cropper(weights, device, crop_cache),
                fit=self.config.get("fit", "pad"),
                precision=precision,
                size=self.config.get("size"),
                descriptor=self.config.get("descriptor"),
                pad_color=tuple(self.config.get("pad_color", (124, 116, 104))),
            )
        self.embedder = embedder
        self.cropper = getattr(embedder, "cropper", None)

        self.ocr = ocr if ocr is not None else LabelOCR()
        self.matcher = matcher if matcher is not None else XFeatMatcher()
        self.vintage = VintageReader(self.ocr)
        self.text_index = (
            text_index
            if text_index is not None
            else TextIndex.from_payloads(self.index.item_ids, self.index.payloads)
        )
        self.analogues = Analogues(self.index.item_ids, self.index.payloads)
        descriptor_dir = Path(index_dir) / "descriptors"
        self.descriptors = (
            DescriptorStore(descriptor_dir, device=self.matcher.device)
            if descriptor_dir.exists()
            else None
        )
        # Счётчик кандидатов, признаки которых пришлось считать на лету. В норме он остаётся
        # нулём; выросший означает, что индекс и дескрипторы разошлись.
        self.recomputed_descriptors = 0
        # Сколько пар «запрос — кандидат» реально сопоставлено. В параллельном режиме их больше
        # заявленного окна: часть кандидатов сопоставляется заранее, до того как известен
        # итоговый порядок, и часть этой работы уходит впустую.
        self.matched_pairs = 0

        self.path_by_id = {
            item_id: payload["image_path"].replace("\\", "/") if payload.get("image_path") else None
            for item_id, payload in zip(self.index.item_ids, self.index.payloads, strict=True)
        }
        self.payload_by_id = dict(zip(self.index.item_ids, self.index.payloads, strict=True))
        # Где на карточке напечатан год. Считается при сборке индекса тем же распознавателем,
        # что и всё остальное, — на запросе это лишний вызов OCR по каждому кандидату.
        # Индекс, собранный до Э8, поля не содержит: тогда год читается только из общего
        # текста этикетки, а увеличение по гомографии просто не включается.
        self.year_boxes = {
            item_id: tuple(payload["vintage_box"])
            for item_id, payload in self.payload_by_id.items()
            if payload.get("vintage_box")
        }

    @staticmethod
    def _whitened(index: VectorIndex, whitening: Whitening) -> VectorIndex:
        """Тот же индекс с отбелёнными векторами; ключи и карточки не меняются."""
        raw = index.index.reconstruct_n(0, index.index.ntotal)
        result = VectorIndex(whitening.dim)
        result.add(whitening.apply(raw), item_ids=index.item_ids, payloads=index.payloads)
        return result

    def devices(self) -> dict[str, str]:
        """На чём реально считается каждый блок.

        Нужно не для красоты: XFeat и PaddleOCR выбирают устройство сами, независимо от того,
        что мы передали эмбеддеру, и молча остаться на процессоре здесь легче лёгкого.
        Поле уходит в /health, чтобы после переезда это проверялось одним запросом.
        """
        return {
            "embed": str(getattr(self.embedder, "device", "?")),
            "precision": str(getattr(self.embedder, "precision", "?")),
            "rerank": str(getattr(self.matcher, "device", "?")),
            "ocr": (
                getattr(self.ocr, "backend", "paddle")
                if getattr(self.ocr, "backend", "paddle") != "paddle"
                else ("cuda" if getattr(self.ocr, "gpu", False) else "cpu")
            ),
            "ocr_fallbacks": str(getattr(self.ocr, "fallbacks", 0)),
        }

    def _build_cropper(self, weights: Path, device, crop_cache: Path | None):
        """Обрезка ровно та же, которой строился индекс."""
        detect = self.config.get("detect", "cascade")
        detector = build_cropper(
            detect, weights, device, bottle_model=self.config.get("bottle_model", COCO_BOTTLE_MODEL)
        )
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
        from .embed.branches import branch_image

        return self.matcher.describe(
            branch_image(self._crop(load_image(path), key), self.local_preprocess), cache_key=key
        )

    def _text_branch(self, crop: Image.Image, use_cache: bool) -> dict:
        """Текстовая ветка целиком: распознать этикетку и найти по тексту кандидатов.

        Вынесена в отдельный метод, чтобы её можно было запустить в потоке параллельно
        с ре-ранкингом. Никакого общего состояния с визуальной веткой у неё нет.
        """
        timings: dict[str, float] = {}
        started = time.perf_counter()
        from .embed.branches import branch_image

        image = branch_image(crop, self.ocr_preprocess)
        # Подменные OCR в тестах умеют только read(); маршрута у них нет.
        if hasattr(self.ocr, "read_with_route"):
            lines, route = self.ocr.read_with_route(image, use_cache=use_cache)
        else:
            lines, route = self.ocr.read(image, use_cache=use_cache), None
        text = LabelOCR.joined(lines)
        timings["ocr"] = time.perf_counter() - started

        started = time.perf_counter()
        scores = None
        hits = []
        query_tokens: list[str] = []
        if text:
            query_tokens = self.text_index.query_tokens(text)
            if query_tokens:
                # Оценки всего каталога считаются один раз: по ним и список кандидатов,
                # и порядок родни при расширении по винодельне.
                scores = self.text_index.score_all(text)
                hits = self.text_index.search(text, top_k=self.text_candidates, scores=scores)
        timings["text_search"] = time.perf_counter() - started

        confidence = sum(line.confidence for line in lines) / len(lines) if lines else 0.0
        return {
            "lines": lines,
            "route": route,
            "text": text,
            "confidence": confidence,
            "hits": hits,
            "scores": scores,
            "tokens": query_tokens,
            "attrs": self.text_index.query_attributes(query_tokens),
            "timings": timings,
        }

    def _long_list(self, visual, textual, scores) -> tuple[list[str], list[str]]:
        """Слияние веток плюс родня по винодельне. Возвращает (список, добавленные роднёй).

        Родня добавляется в хвост: у неё нет позиции ни в одной ветке, и ставить её выше
        честно найденных кандидатов не за что. Своё место она получит по текстовым сигналам.
        """
        visual_ids = [h.item_id for h in visual]
        textual_ids = [h.item_id for h in textual]
        base = ranked(fuse_rrf([visual_ids, textual_ids]))
        added: list[str] = []
        if self.family_expansion:
            seen = set(base)
            families: list[str] = []
            for item_id in visual_ids[:FAMILY_SEEDS] + textual_ids[:FAMILY_SEEDS]:
                family = self.text_index.family_of.get(item_id, "")
                if family and family not in families:
                    families.append(family)
            for family in families:
                for item_id in self.text_index.family_rank(family, scores, seen, FAMILY_CAP):
                    added.append(item_id)
                    seen.add(item_id)
        return base + added, added

    def _window(self, long_list: list[str], signals: dict[str, dict]) -> list[str]:
        """Кому достаётся геометрия: верхушка слитого порядка и подтверждённые текстом."""
        window = long_list[: self.candidates]
        extra = [
            item_id
            for item_id in long_list[self.candidates :]
            if signals[item_id]["disc_hit"] >= DISC_WINDOW_MIN
        ]
        extra.sort(key=lambda item_id: -signals[item_id]["disc_hit"])
        return window + extra[: self.window_extra]

    def _match(self, query_features: dict, item_ids, found: dict) -> None:
        """Сопоставить запрос с кандидатами, пропуская уже сопоставленных."""
        for item_id in item_ids:
            if item_id in found:
                continue
            candidate = self._candidate_descriptor(item_id)
            found[item_id] = self.matcher.match(query_features, candidate) if candidate else None
            self.matched_pairs += 1

    def _read_vintage(self, crop: Image.Image, lines, order, matches):
        """Год урожая на кадре: сначала даром, потом за деньги.

        Общий текст этикетки уже прочитан текстовой веткой, поэтому первая попытка бесплатна
        и покрывает две трети кадров. Второй заход — вырезать участок по геометрии лидера
        и перечитать его крупно — стоит ещё одного вызова распознавания, поэтому делается
        только там, где иначе года не будет вовсе.

        Геометрия берётся у лидера по инлаерам, а не у первого после слияния веток: гомографию
        мы применяем к пикселям, и здесь важнее всего, чтобы она была точной, а не чтобы
        кандидат нравился остальным признакам.
        """
        reading = self.vintage.from_lines(lines)
        if reading:
            return reading

        ranked_by_geometry = [item_id for item_id in order if matches.get(item_id)]
        if not ranked_by_geometry:
            return reading

        leader = max(ranked_by_geometry, key=lambda item_id: matches[item_id].inliers)
        box = self.year_boxes.get(leader)
        if box is None:
            return reading

        projected = project(box, matches[leader])
        return self.vintage.zoom(crop, projected) if projected else reading

    def identify(
        self, image: Image.Image, image_key: str | None = None, trace: bool = False
    ) -> ScanResult:
        """Опознать вино по одному кадру. `trace` — сохранить внутренности для измерителя."""
        timings: dict[str, float] = {}
        wall_started = time.perf_counter()

        @contextmanager
        def stage(name: str):
            start = time.perf_counter()
            yield
            timings[name] = time.perf_counter() - start

        with stage("crop"):
            crop = self._crop(image, image_key)
        from .embed.branches import branch_image

        local_crop = branch_image(crop, self.local_preprocess)

        with stage("embed"):
            vector = self.embedder.encode_image(crop).numpy()
            if self.whitening is not None:
                vector = self.whitening.apply(vector)[0]

        with stage("search"):
            visual = self.index.search(vector, top_k=self.visual_candidates)

        use_cache = self.ocr_cache and image_key is not None
        matches: dict[str, object] = {}

        # Текстовая ветка и ре-ранкинг связаны только через итоговый порядок кандидатов, а
        # считаются оба долго. Поэтому пока читается этикетка, в главном потоке уже идёт
        # сопоставление точек с визуальными кандидатами — их список известен сразу после
        # поиска в индексе и от текста не зависит.
        if self.parallel:
            # Заранее берём не всё окно, а его верхнюю половину. Кандидат, стоящий у визуальной
            # ветки высоко, из итогового порядка почти никогда не выпадает, а вот нижняя часть
            # окна после слияния с текстом перетасовывается сильно — и сопоставлять её заранее
            # значит просто выбрасывать работу.
            head = [h.item_id for h in visual[: max(1, self.candidates // 2)]]
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="text") as pool:
                future = pool.submit(self._text_branch, crop, use_cache)
                with stage("rerank"):
                    query_features = self.matcher.describe(local_crop)
                    self._match(query_features, head, matches)
                text_result = future.result()
        else:
            text_result = self._text_branch(crop, use_cache)
            with stage("rerank"):
                query_features = self.matcher.describe(local_crop)

        timings.update(text_result["timings"])
        lines = text_result["lines"]
        textual = text_result["hits"]

        # Слияние веток. RRF складывает позиции, а не оценки, поэтому шкалы приводить не нужно:
        # у визуальной ветки это косинус, у текстовой — n-граммы с покрытием слов.
        with stage("signals"):
            long_list, family_added = self._long_list(visual, textual, text_result["scores"])
            ocr_tokens, ocr_attrs = text_result["tokens"], text_result["attrs"]
            signals = {
                item_id: self.text_index.signals(ocr_tokens, ocr_attrs, item_id)
                for item_id in long_list
            }
            window = self._window(long_list, signals)
            window = list(dict.fromkeys([*window, *(hit.item_id for hit in textual[:5])]))
            window_set = set(window)

        vis_rank = {h.item_id: i for i, h in enumerate(visual)}
        vis_score = {h.item_id: h.score for h in visual}
        txt_rank = {h.item_id: i for i, h in enumerate(textual)}
        txt_score = {h.item_id: h.score for h in textual}

        started = time.perf_counter()
        self._match(query_features, window, matches)
        timings["rerank"] = timings.get("rerank", 0.0) + (time.perf_counter() - started)

        with stage("vintage"):
            reading = self._read_vintage(crop, lines, window, matches)

        rows = []
        for position, item_id in enumerate(long_list):
            # Геометрия учитывается только у членов окна. Предварительное сопоставление могло
            # посчитать инлаеры и тому, кто в окно не попал, — но тогда параллельный режим
            # отвечал бы иначе, чем последовательный, а обучение видело бы не то, что сервис.
            match = matches.get(item_id) if item_id in window_set else None
            signal = signals[item_id]
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
                    geometry_query_coverage=match.query_coverage if match else 0.0,
                    geometry_candidate_coverage=match.candidate_coverage if match else 0.0,
                    geometry_normalized_error=match.normalized_reproj_error if match else 0.0,
                    ocr_lines=len(lines),
                    ocr_conf=text_result["confidence"],
                    vintage_known=int(bool(reading)),
                    vintage_match=compare(reading.year, self.payload_by_id.get(item_id, {})),
                    in_window=int(item_id in window_set),
                    disc_hit=signal["disc_hit"],
                    disc_n=signal["disc_n"],
                    name_cover=signal["name_cover"],
                    winery_hit=signal["winery_hit"],
                    color_match=signal["color_match"],
                    style_match=signal["style_match"],
                    family=self.text_index.family_of.get(item_id, ""),
                    **text_pair_features(
                        " ".join(line.text for line in lines),
                        str(self.payload_by_id.get(item_id, {}).get("name", "")),
                        [line.text for line in lines],
                        winery=str(self.payload_by_id.get(item_id, {}).get("winery", "")),
                        grapes=str(self.payload_by_id.get(item_id, {}).get("grapes", "")),
                    ),
                )
            )

        with stage("decide"):
            scored = self.decider.score(rows)

        # Защита от близнеца: правило поверх модели, см. decide/guard.py. ``twin`` режет
        # вероятность ниже порога; ``warn`` сохраняет тот же сигнал, но оставляет лучший
        # ответ. В закрытом каталоге это даёт пользователю полезную близкую карточку.
        applied: list[str] = []
        # Внутри семьи решает текст: если соседку по линейке этикетка подтверждает лучше,
        # чем лидера модели, наверх идёт она, с уверенностью лидера — семья та же.
        if scored and self.sibling:
            swap = sibling_swap(scored, self.text_index.family_of)
            if swap:
                chosen = scored.pop(swap)
                chosen.probability = max(chosen.probability, scored[0].probability)
                scored.insert(0, chosen)
                applied.append("sibling")
        if scored and self.guard != "off":
            twin = twin_guard(scored[0].features, self.guard)
            if twin:
                if self.guard == "warn":
                    applied.append(f"{twin}_warning")
                else:
                    scored[0].probability = min(scored[0].probability, self.threshold - GUARD_EPS)
                    applied.append(twin)
        guard = "+".join(applied) or None

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
        answered = bool(best and best.probability >= self.threshold)

        judge_report = None
        if self.judge is not None and candidates:
            with stage("judge"):
                candidates, answered, judge_report = self.judge.consult(
                    crop, candidates, answered, self.threshold, self.text_index.family_of
                )
                best = candidates[0]

        # Не сумма этапов, а настоящее время запроса: этапы теперь идут внахлёст, и их сумма
        # больше того, что ждёт пользователь. Ровно эту величину и требует Э11.
        timings["total"] = time.perf_counter() - wall_started

        return ScanResult(
            answered=answered,
            best=best,
            candidates=candidates,
            text=text_result["text"],
            timings=timings,
            threshold=self.threshold,
            vintage=resolve(reading, best.payload) if best else None,
            guard=guard,
            judge=judge_report,
            analogues=self.analogues.for_item(best.item_id, k=REPORTED_CANDIDATES)
            if best
            else None,
            confidence=confidence_of(candidates),
            trace=(
                {
                    "vector": vector,
                    "visual": [(h.item_id, h.score) for h in visual],
                    "textual": [(h.item_id, h.score) for h in textual],
                    "long_list": list(long_list),
                    "family_added": list(family_added),
                    "window": list(window),
                    "signals": signals,
                    "inliers": {
                        item_id: (m.inliers if m else 0)
                        for item_id, m in matches.items()
                        if item_id in window_set
                    },
                    "ocr_lines": [line.text for line in lines],
                    # Кто прочитал строку (paddle / yandex) и насколько уверенно — по этому
                    # бенчмарк видит, на каких кадрах hybrid ходил в облако.
                    "ocr_sources": [line.source for line in lines],
                    "ocr_confidences": [line.confidence for line in lines],
                    "ocr_route": text_result.get("route"),
                    "ocr_tokens": list(ocr_tokens),
                }
                if trace
                else None
            ),
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
