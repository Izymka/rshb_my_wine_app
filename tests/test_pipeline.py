"""Проверка логики сборки пайплайна — без единой нейросети.

Тяжёлые блоки (детектор, DINOv2, EasyOCR, XFeat) подменяются заглушками. Смысл в том, чтобы
проверяемым оказалось ровно то, что ломается тихо: порядок кандидатов, состав признаков,
поведение порога отказа. Ошибка в любом из этих мест не роняет процесс — она просто делает
ответы хуже, и заметить её на глаз в выдаче почти невозможно.

Что проверить так нельзя — качество распознавания. Оно меряется на своём наборе в eval/.
"""

import numpy as np
import pytest

from wine_scanner.decide import FEATURE_NAMES, Decider, PairFeatures
from wine_scanner.index import VectorIndex
from wine_scanner.ocr import TextIndex, TextLine, catalog_document
from wine_scanner.pipeline import REPORTED_CANDIDATES, Candidate, ScanResult, WineScanner
from wine_scanner.rerank import MatchFeatures

DIM = 8


class FakeEmbedder:
    """Отдаёт заранее заданный вектор. Обрезки нет — картинка идёт как есть."""

    size = 224
    fit = "pad"
    cropper = None

    def __init__(self, vector):
        self.vector = vector

    def encode_image(self, image):
        import torch

        return torch.tensor(self.vector, dtype=torch.float32)


class FakeOCR:
    def __init__(self, lines):
        self._lines = lines

    def read(self, image, use_cache=False):
        return self._lines

    @staticmethod
    def joined(lines):
        return " ".join(line.text for line in lines)


class FakeMatcher:
    """Инлаеры задаются по item_id, картинки кандидатов не читаются."""

    def __init__(self, inliers_by_id):
        import torch

        self.inliers_by_id = inliers_by_id
        self.device = torch.device("cpu")
        self._cache = {}

    def cached(self, key):
        return {"id": key}

    def describe(self, image, cache_key=None):
        return {"id": cache_key}

    def match(self, query, candidate):
        inliers = self.inliers_by_id.get(candidate["id"], 0)
        return MatchFeatures(
            matches=max(inliers, 4),
            inliers=inliers,
            inlier_ratio=1.0 if inliers else 0.0,
            reproj_error=1.0,
            homography_ok=inliers > 4,
        )


class FakeBooster:
    """Сырая оценка = доля инлаеров кандидата. Модель здесь не проверяется, проверяется обвязка."""

    def predict(self, matrix):
        matrix = np.asarray(matrix)
        return np.clip(matrix[:, FEATURE_INDEX_INLIERS_SHARE], 1e-3, 1 - 1e-3)


# Номер колонки спрашиваем у самого списка признаков. Раньше здесь стояло число, и стоило
# добавить признак в середину, как фиктивный бустер начинал читать чужую колонку — три теста
# падали с сообщением про не то вино, ни словом не намекая на причину.
FEATURE_INDEX_INLIERS_SHARE = FEATURE_NAMES.index("inliers_share")


def build_index(extra: dict | None = None) -> VectorIndex:
    vectors = np.eye(3, DIM, dtype="float32")
    index = VectorIndex(DIM)
    index.add(
        vectors,
        item_ids=["wine_a", "wine_b", "wine_c"],
        payloads=[
            {
                "name": "Chateau Alpha",
                "winery": "Alpha",
                "image_path": "/нет/такого/a.jpg",
                **(extra or {}),
            },
            {"name": "Chateau Beta", "winery": "Beta", "image_path": "/нет/такого/b.jpg"},
            {"name": "Gamma Reserve", "winery": "Gamma", "image_path": "/нет/такого/c.jpg"},
        ],
    )
    return index


def build_scanner(
    vector, inliers, lines=(), threshold=0.5, parallel=True, extra=None
) -> WineScanner:
    index = build_index(extra)
    return WineScanner(
        index=index,
        parallel=parallel,
        embedder=FakeEmbedder(vector),
        ocr=FakeOCR(list(lines)),
        matcher=FakeMatcher(inliers),
        text_index=TextIndex(index.item_ids, [catalog_document(p) for p in index.payloads]),
        decider=Decider(FakeBooster(), calib_weight=1.0, calib_bias=0.0, threshold=threshold),
        candidates=3,
    )


def test_best_candidate_wins_by_inliers():
    """Визуально ближе wine_a, но геометрия подтверждает wine_b — верить надо геометрии."""
    scanner = build_scanner(vector=[1, 0, 0, 0, 0, 0, 0, 0], inliers={"/нет/такого/b.jpg": 120})
    result = scanner.identify(image=None)

    assert result.best.item_id == "wine_b"
    assert [c.item_id for c in result.candidates][0] == "wine_b"


def test_refuses_when_probability_below_threshold():
    """Ни один кандидат не подтверждён геометрией — отвечать нельзя, хотя лучший всё равно есть."""
    scanner = build_scanner(vector=[1, 0, 0, 0, 0, 0, 0, 0], inliers={}, threshold=0.9)
    result = scanner.identify(image=None)

    assert result.answered is False
    assert result.best is not None
    assert result.to_dict()["card"] is None


def test_answer_carries_card_without_service_fields():
    scanner = build_scanner(
        vector=[1, 0, 0, 0, 0, 0, 0, 0],
        inliers={"/нет/такого/a.jpg": 200},
        extra={"vintage_box": [0.1, 0.2, 0.3, 0.4]},
    )
    payload = scanner.identify(image=None).to_dict()

    assert payload["answered"] is True
    assert payload["item_id"] == "wine_a"
    assert payload["card"]["name"] == "Chateau Alpha"
    # Служебные поля: путь к картинке нужен ре-ранкингу, координаты года — блоку винтажа.
    # Наружу не уходит ни то, ни другое, и в списке кандидатов тоже.
    assert "image_path" not in payload["card"]
    assert "vintage_box" not in payload["card"]
    assert all("vintage_box" not in c["card"] for c in payload["candidates"])


def test_only_the_top_of_the_list_goes_out():
    """Решающий слой оценивает всё окно, клиенту уходит верхушка."""
    candidates = [
        Candidate(item_id=f"wine_{i}", probability=1.0 - i / 100, payload={"name": f"Вино {i}"})
        for i in range(REPORTED_CANDIDATES + 3)
    ]
    result = ScanResult(
        answered=True,
        best=candidates[0],
        candidates=candidates,
        text="",
        timings={"total": 0.1},
        threshold=0.5,
    )
    payload = result.to_dict()

    assert len(payload["candidates"]) == REPORTED_CANDIDATES
    # Обрезается только выдача: метрики считаются по полному списку.
    assert len(result.candidates) == REPORTED_CANDIDATES + 3
    assert payload["candidates"][0]["item_id"] == "wine_0"


def test_vintage_of_a_single_bottle_comes_from_the_card():
    """Карточка описывает конкретную бутылку — год берём из неё, а не с фотографии."""
    scanner = build_scanner(
        vector=[1, 0, 0, 0, 0, 0, 0, 0],
        inliers={"/нет/такого/a.jpg": 200},
        extra={"vintage": 2022},
    )
    payload = scanner.identify(image=None).to_dict()

    assert payload["vintage"] == {"year": 2022, "source": "catalog", "ask": False}


def test_vintage_of_a_line_card_is_read_from_the_label():
    """Карточка на линейку год не называет: выбрать из списка может только этикетка."""
    line = TextLine("CHATEAU ALPHA 2019 BORDEAUX", 0.9, "latin")
    scanner = build_scanner(
        vector=[1, 0, 0, 0, 0, 0, 0, 0],
        inliers={"/нет/такого/a.jpg": 200},
        lines=[line],
        extra={"vintages": "[2020, 2019, 2018]"},
    )
    payload = scanner.identify(image=None).to_dict()

    assert payload["vintage"] == {"year": 2019, "source": "text", "ask": False}


def test_vintage_is_not_reported_when_the_answer_is_refused():
    """При отказе карточки нет, и год показывать не от чего: он относился бы к чужому вину."""
    scanner = build_scanner(
        vector=[1, 0, 0, 0, 0, 0, 0, 0], inliers={}, threshold=0.9, extra={"vintage": 2022}
    )
    payload = scanner.identify(image=None).to_dict()

    assert payload["answered"] is False
    assert payload["vintage"] is None


def test_timings_cover_every_stage():
    """`total` — время запроса целиком, а не сумма этапов: они идут внахлёст."""
    scanner = build_scanner(vector=[1, 0, 0, 0, 0, 0, 0, 0], inliers={})
    timings = scanner.identify(image=None).timings
    stages = {k: v for k, v in timings.items() if k != "total"}

    assert {"crop", "embed", "search", "ocr", "text_search", "rerank", "decide"} <= set(stages)
    assert max(stages.values()) <= timings["total"]
    assert timings["total"] <= sum(stages.values()) + 0.5


def test_parallel_and_sequential_agree():
    """Перенос текстовой ветки в поток обязан не менять ответ, только время."""
    kwargs = dict(vector=[1, 0, 0, 0, 0, 0, 0, 0], inliers={"/нет/такого/b.jpg": 120})
    parallel = build_scanner(**kwargs).identify(image=None)
    sequential = build_scanner(**kwargs, parallel=False).identify(image=None)

    assert [c.item_id for c in parallel.candidates] == [
        c.item_id for c in sequential.candidates
    ]
    assert parallel.best.probability == pytest.approx(sequential.best.probability)


def test_decider_rejects_reordered_features():
    """Перепутанный порядок колонок — самая тихая из возможных поломок, ловим её на загрузке."""
    with pytest.raises(ValueError, match="порядок признаков"):
        Decider(FakeBooster(), 1.0, 0.0, 0.5, feature_names=("inliers", "vis_score"))


def test_calibration_spreads_the_scale():
    """Калибровка по логиту обязана растягивать шкалу, а не сплющивать её у единицы."""
    decider = Decider(FakeBooster(), calib_weight=1.0, calib_bias=0.0, threshold=0.5)
    values = decider.calibrate(np.array([0.05, 0.5, 0.95]))

    assert values[0] == pytest.approx(0.05, abs=1e-6)
    assert values[2] == pytest.approx(0.95, abs=1e-6)
    assert values[2] - values[0] > 0.8


def test_pair_features_need_no_labels():
    """На инференсе правильного ответа нет, и признаки обязаны собираться без него."""
    row = PairFeatures(
        item_id="wine_a",
        vis_score=0.8,
        vis_rank=0,
        txt_score=-1.0,
        txt_rank=999,
        rrf_rank=0,
        matches=10,
        inliers=8,
        inlier_ratio=0.8,
        reproj_error=1.0,
        homography_ok=1,
        ocr_lines=3,
        ocr_conf=0.7,
    )
    assert row.label == 0
    assert row.true_id == ""


# --- Длинный список, окно, защита от близнеца, уверенность (каталог платформы, 16.09) ------


def build_family_index() -> VectorIndex:
    """Шесть карточек: четыре Массандры на одном шаблоне, одна чужая, одна одиночка."""
    payloads = [
        ("krasnyy", "Портвейн красный Алушта", "Массандра", "Красное"),
        ("belyy", "Портвейн белый Алушта", "Массандра", "Белое"),
        ("krymskiy", "Портвейн белый крымский", "Массандра", "Белое"),
        ("muskatel", "Мускатель белый", "Массандра", "Белое"),
        ("taman", "Шато Тамань Руж", "Кубань-Вино", "Красное"),
        ("lone", "Гравити", "Alma Valley", "Белое"),
    ]
    vectors = np.eye(len(payloads), DIM, dtype="float32")
    index = VectorIndex(DIM)
    index.add(
        vectors,
        item_ids=[p[0] for p in payloads],
        payloads=[
            {"name": n, "winery": w, "category": c, "image_path": f"/нет/такого/{i}.jpg"}
            for i, n, w, c in payloads
        ],
    )
    return index


def build_family_scanner(vector, inliers, lines=(), threshold=0.5, **kwargs) -> WineScanner:
    index = build_family_index()
    return WineScanner(
        index=index,
        embedder=FakeEmbedder(vector),
        ocr=FakeOCR(list(lines)),
        matcher=FakeMatcher(inliers),
        text_index=TextIndex.from_payloads(index.item_ids, index.payloads),
        decider=Decider(FakeBooster(), calib_weight=1.0, calib_bias=0.0, threshold=threshold),
        **kwargs,
    )


def label(*texts: str) -> list[TextLine]:
    return [TextLine(t, 0.9, "cyrillic") for t in texts]


def test_long_list_is_wider_than_window_and_outsiders_get_no_geometry():
    """Кандидаты за окном получают текстовые сигналы, но не инлаеры — даже если фейковый
    матчер их бы дал."""
    scanner = build_family_scanner(
        vector=[0.5, 0.4, 0.3, 0.2, 1.0, 0.1, 0, 0],  # taman первый, lone последний
        inliers={"/нет/такого/lone.jpg": 500},
        candidates=2,
        visual_candidates=6,
        family_expansion=False,
    )
    result = scanner.identify(image=None, trace=True)

    assert len(result.trace["long_list"]) > 2
    assert len(result.trace["window"]) == 2
    lone = next(c for c in result.candidates if c.item_id == "lone")
    assert lone.features["in_window"] == 0
    assert lone.features["inliers"] == 0


def test_text_confirmed_sibling_enters_the_window():
    """Визуально последний, но этикетка называет его различающие слова — геометрию получает."""
    scanner = build_family_scanner(
        vector=[0, 0, 0, 0, 0, 1, 0, 0],  # ближе всего lone, Массандры в хвосте
        inliers={"/нет/такого/krymskiy.jpg": 90},
        lines=label("МАССАНДРА", "ПОРТВЕЙН БЕЛЫЙ КРЫМСКИЙ", "КРЫМ"),
        candidates=1,
        visual_candidates=6,
        text_candidates=6,
    )
    result = scanner.identify(image=None, trace=True)

    assert "krymskiy" in result.trace["window"]
    krymskiy = next(c for c in result.candidates if c.item_id == "krymskiy")
    assert krymskiy.features["in_window"] == 1
    assert krymskiy.features["disc_hit"] == 1.0
    assert result.best.item_id == "krymskiy"


def test_family_expansion_adds_siblings_of_visual_leader():
    scanner = build_family_scanner(
        vector=[0, 0, 0, 1, 0, 0, 0, 0],  # ближе всего muskatel
        inliers={},
        candidates=1,
        visual_candidates=1,
        text_candidates=1,
    )
    with_family = scanner.identify(image=None, trace=True)
    assert set(with_family.trace["family_added"]) == {"krasnyy", "belyy", "krymskiy"}

    scanner.family_expansion = False
    without = scanner.identify(image=None, trace=True)
    assert without.trace["family_added"] == []
    assert len(without.trace["long_list"]) == 1


def test_twin_guard_refuses_rose_label_on_red_card():
    """Розовый Алушта: геометрия и слова за красный сиблинг, цвет — против."""
    lines = label("МАССАНДРА", "ПОРТВЕЙН РОЗОВЫЙ АЛУШТА", "ГОД УРОЖАЯ 2023")
    kwargs = dict(
        vector=[1, 0, 0, 0, 0, 0, 0, 0],
        inliers={"/нет/такого/krasnyy.jpg": 150},
        lines=lines,
        visual_candidates=6,
    )
    guarded = build_family_scanner(**kwargs).identify(image=None)
    # Текст может переставить сиблинга наверх («белый Алушта» подтверждён не хуже), но
    # цвет противоречит любому из них — итог в любом случае отказ с пометкой twin.
    assert guarded.best.item_id in {"krasnyy", "belyy"}
    assert guarded.answered is False
    assert guarded.guard and "twin" in guarded.guard
    assert "twin" in guarded.to_dict()["guard"]

    unguarded = build_family_scanner(**kwargs, guard="off", sibling=False).identify(image=None)
    assert unguarded.answered is True
    assert unguarded.guard is None


def test_twin_guard_stays_quiet_when_label_confirms_the_card():
    lines = label("МАССАНДРА", "ПОРТВЕЙН КРАСНЫЙ АЛУШТА", "ГОД УРОЖАЯ 2024")
    result = build_family_scanner(
        vector=[1, 0, 0, 0, 0, 0, 0, 0],
        inliers={"/нет/такого/krasnyy.jpg": 150},
        lines=lines,
        visual_candidates=6,
    ).identify(image=None)
    assert result.answered is True
    assert result.guard is None
    assert result.best.features["color_match"] == 1


def test_twin_guard_needs_readable_label():
    """Одна строка OCR — правилу нечем судить, ответ остаётся за моделью."""
    result = build_family_scanner(
        vector=[1, 0, 0, 0, 0, 0, 0, 0],
        inliers={"/нет/такого/krasnyy.jpg": 150},
        lines=label("РОЗОВЫЙ"),
        visual_candidates=6,
    ).identify(image=None)
    assert result.answered is True
    assert result.guard is None


def test_confidence_block_is_consistent():
    result = build_family_scanner(
        vector=[1, 0, 0, 0, 0, 0, 0, 0],
        inliers={"/нет/такого/krasnyy.jpg": 150, "/нет/такого/belyy.jpg": 30},
        visual_candidates=6,
    ).identify(image=None)
    confidence = result.confidence
    probabilities = [c.probability for c in result.candidates]

    assert confidence["top1"] == pytest.approx(probabilities[0])
    assert 0 < confidence["top5"] <= 1
    assert confidence["margin"] == pytest.approx(probabilities[0] - probabilities[1])
    assert result.to_dict()["confidence"] == confidence


def test_family_key_travels_with_features():
    result = build_family_scanner(
        vector=[1, 0, 0, 0, 0, 0, 0, 0], inliers={}, visual_candidates=6
    ).identify(image=None)
    rows = {c.item_id: c.features for c in result.candidates}
    assert rows["krasnyy"]["family"] == "Массандра"
    assert rows["taman"]["family"] == "Кубань-Вино"
    # disc_contra у красного — лучший disc_hit среди других Массандр; без OCR он нулевой.
    assert rows["krasnyy"]["disc_contra"] == 0.0
