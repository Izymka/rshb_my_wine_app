"""Признаки пары «запрос — кандидат» для решающего слоя.

До этого этапа три ветки складывались с фиксированными весами: визуальная и текстовая через
RRF, ре-ранкинг сортировкой по инлаерам. Замеры показали, что фиксированные веса неоптимальны —
на смазанных кадрах текстовая ветка мешает, но учитывается наравне с остальными.

Здесь мы перестаём угадывать веса и отдаём все сигналы модели. Заодно получаем то, чего
до сих пор не было вовсе: калиброванную вероятность правильности ответа, а значит возможность
честно отказаться, когда вина нет в каталоге.
"""

from dataclasses import asdict, dataclass, fields

# Порядок важен: он же задаёт порядок колонок в матрице признаков.
FEATURE_NAMES = (
    "vis_score",
    "vis_rank",
    "txt_score",
    "txt_rank",
    "rrf_rank",
    "matches",
    "inliers",
    "inlier_ratio",
    "reproj_error",
    "homography_ok",
    "ocr_lines",
    "ocr_conf",
    "vintage_known",
    "vintage_match",
    "vis_gap",
    "inliers_gap",
    "inliers_share",
    "vis_margin",
    "inliers_margin",
    # Версия 2 (16.09.2026, каталог платформы): текстовые сигналы для близнецов. Дописаны
    # в конец, чтобы старые колонки не сдвигались; модель, обученная на версии 1, с этим
    # списком не загрузится — и это правильно, её надо переобучить.
    "in_window",
    "disc_hit",
    "disc_n",
    "name_cover",
    "winery_hit",
    "color_match",
    "style_match",
    "disc_contra",
    "txt_margin",
    "vis_distance",
    "vis_available",
    "vis_top12_gap",
    "vis_mean",
    "vis_std",
    "vis_zscore",
    "txt_top1_score",
    "txt_top2_score",
    "txt_top3_score",
    "txt_top4_score",
    "txt_top5_score",
    "txt_in_top5",
    "name_levenshtein_distance",
    "name_jaro_winkler",
    "name_best_line_jaro",
    "name_token_set_ratio",
    "geometry_query_coverage",
    "geometry_candidate_coverage",
    "geometry_normalized_error",
)

# Меняется вместе с FEATURE_NAMES. Пишется в meta решающего слоя и проверяется при загрузке:
# несовпадение версии — сигнал переобучить, а не молча считать по чужим колонкам.
FEATURE_VERSION = 3


@dataclass
class PairFeatures:
    """Сырые признаки одной пары. Производные считаются позже, по всему набору кандидатов.

    Поля разметки (`query`, `true_id`, `group`) стоят последними и необязательны: на инференсе
    правильного ответа не существует, а признаки собираются ровно те же. Без значений по
    умолчанию пайплайну пришлось бы подставлять фиктивную разметку — а это верный способ
    однажды посчитать метрику по выдуманным меткам.
    """

    item_id: str

    vis_score: float  # косинус из индекса
    vis_rank: int
    txt_score: float  # счёт текстовой ветки, -1 если кандидат ею не найден
    txt_rank: int
    rrf_rank: int  # позиция после слияния веток

    matches: int
    inliers: int
    inlier_ratio: float
    reproj_error: float
    homography_ok: int

    ocr_lines: int  # сколько строк распознано на запросе
    ocr_conf: float  # средняя уверенность распознавания

    # Год урожая. `vintage_known` — прочитан ли он у запроса вообще, `vintage_match` — сошёлся
    # ли с карточкой: +1 совпал, -1 противоречит, 0 сравнивать не с чем. Два признака вместо
    # одного, потому что «года не видно» и «год чужой» — разные новости, и вторая гораздо
    # весомее: именно она позволяет усомниться в вине, которое сходится по всему, кроме года.
    vintage_known: int = 0
    vintage_match: int = 0

    # Текстовые сигналы (версия 2). Считаются текстовым индексом по свёрнутым словам этикетки.
    # `in_window` — сопоставлялся ли кандидат локальными признаками вообще: ноль инлаеров у
    # того, кого не сопоставляли, и у того, кого сопоставили впустую, — разные вещи.
    # `disc_hit` — доля различающих слов карточки (редких внутри её винодельни), найденных
    # на этикетке; `disc_n` — сколько их у карточки вообще: без него ноль ничего не значит.
    # `name_cover` — доля всех слов названия с весами IDF; `winery_hit` — нашлась ли винодельня.
    # `color_match` / `style_match` — цвет и сладость: +1 сошлись, −1 противоречат, 0 неизвестно.
    in_window: int = 1
    disc_hit: float = 0.0
    disc_n: int = 0
    name_cover: float = 0.0
    winery_hit: int = 0
    color_match: int = 0
    style_match: int = 0

    query: str = ""
    true_id: str = ""
    group: str = ""
    # Винодельня кандидата. Не признак, а ключ для производных: `disc_contra` считается по
    # соседям той же винодельни, а `derive()` в обучении вызывается на подмножествах строк,
    # поэтому винодельня должна ехать вместе со строкой, а не искаться в каталоге.
    family: str = ""

    # Разметка блока винтажа: не признаки, а материал для его собственного бенчмарка.
    # Держим рядом с признаками, потому что считаются они в одном дорогом прогоне, и
    # заводить ради них второй файл значило бы гонять пайплайн дважды.
    vintage_year: int = 0  # что в итоге прочитали, 0 — не прочитали
    vintage_source: str = ""  # "text" — общий OCR, "zoom" — увеличенный участок
    vintage_text: int = 0  # результат одного лишь общего OCR
    vintage_zoom: int = 0  # результат одного лишь увеличения по гомографии
    vintage_ref: str = ""  # по чьей геометрии вырезался участок
    name_levenshtein_distance: float = 1.0
    name_jaro_winkler: float = 0.0
    name_best_line_jaro: float = 0.0
    name_token_set_ratio: float = 0.0
    feature_version: int = FEATURE_VERSION
    geometry_query_coverage: float = 0.0
    geometry_candidate_coverage: float = 0.0
    geometry_normalized_error: float = 0.0

    @property
    def label(self) -> int:
        """Верен ли кандидат. Осмысленно только при заполненном true_id, то есть на обучении."""
        return int(bool(self.true_id) and self.item_id == self.true_id)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PairFeatures":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def derive(rows: list[PairFeatures]) -> list[dict]:
    """Добавить признаки, которые имеют смысл только в сравнении с остальными кандидатами.

    Одинокий косинус 0.83 не говорит ни о чём: у одних вин так выглядит верный ответ, у других
    случайный сосед. А вот отрыв от лучшего кандидата информативен всегда — он показывает,
    насколько модель колеблется. Это ровно та величина, которой не хватало для порога отказа.
    """
    import numpy as np

    visual = sorted((r.vis_score for r in rows if r.vis_rank < 999), reverse=True)
    text_top = sorted((r for r in rows if r.txt_rank < 999), key=lambda r: r.txt_rank)[:5]
    vis_mean = float(np.mean(visual)) if visual else 0.0
    vis_std = float(np.std(visual)) if visual else 0.0
    best_vis = max((r.vis_score for r in rows), default=0.0)
    best_inliers = max((r.inliers for r in rows), default=0)
    total_inliers = sum(r.inliers for r in rows) or 1
    by_family: dict[str, list[PairFeatures]] = {}
    for r in rows:
        if r.family:
            by_family.setdefault(r.family, []).append(r)

    out = []
    for row in rows:
        data = row.to_dict()
        data["vis_available"] = int(row.vis_rank < 999)
        # IndexFlatIP operates on L2-normalized vectors: score is cosine similarity.
        # No vector coordinate is included in the feature matrix.
        data["vis_distance"] = 1.0 - row.vis_score if row.vis_rank < 999 else 2.0
        data["vis_top12_gap"] = visual[0] - visual[1] if len(visual) > 1 else 0.0
        data["vis_mean"], data["vis_std"] = vis_mean, vis_std
        data["vis_zscore"] = (
            (row.vis_score - vis_mean) / vis_std if vis_std > 1e-8 and row.vis_rank < 999 else 0.0
        )
        for rank in range(5):
            data[f"txt_top{rank + 1}_score"] = (
                text_top[rank].txt_score if rank < len(text_top) else -1.0
            )
        data["txt_in_top5"] = int(row.txt_rank < 5)
        data["vis_gap"] = best_vis - row.vis_score
        data["inliers_gap"] = best_inliers - row.inliers
        # Улика за сиблинга: лучшее подтверждение различающих слов у другой карточки той же
        # винодельни. Если этикетка подтверждает соседа, а не этого кандидата, — перед нами
        # близнец, и визуальная похожесть здесь ничего не стоит.
        siblings = [r for r in by_family.get(row.family, []) if r is not row]
        data["disc_contra"] = max((r.disc_hit for r in siblings), default=0.0)
        data["txt_margin"] = row.txt_score - max(
            (r.txt_score for r in rows if r is not row), default=-1.0
        )
        # Отрыв от сильнейшего из остальных. У лучшего кандидата величина положительна и
        # показывает, насколько уверенно он выиграл; у прочих отрицательна.
        #
        # Без этого признака отказ работать не может в принципе: vis_gap и inliers_gap
        # считаются относительно лучшего кандидата, поэтому у самого лучшего они всегда равны
        # нулю. Убери из списка правильный ответ — и занявший его место кандидат получит
        # ровно те же нули, то есть станет для модели неотличим от верного.
        others = [r for r in rows if r is not row]
        data["vis_margin"] = row.vis_score - max((r.vis_score for r in others), default=0.0)
        data["inliers_margin"] = row.inliers - max((r.inliers for r in others), default=0)
        # Доля инлаеров кандидата среди всех: если один кандидат забрал их почти все,
        # это гораздо убедительнее, чем просто большое абсолютное число.
        data["inliers_share"] = row.inliers / total_inliers
        data["label"] = row.label
        out.append(data)
    return out


def matrix(rows: list[dict]) -> list[list[float]]:
    return [[float(row[name]) for name in FEATURE_NAMES] for row in rows]


def text_pair_features(text: str, name: str, lines: list[str]) -> dict:
    """Normalized edit distance and string similarities; no ground-truth text is used."""
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler, Levenshtein

    from ..ocr.normalize import fold

    query, candidate = fold(text), fold(name)
    valid = bool(query and candidate)
    return {
        "name_levenshtein_distance": Levenshtein.normalized_distance(query, candidate)
        if valid
        else 1.0,
        "name_jaro_winkler": JaroWinkler.normalized_similarity(query, candidate) if valid else 0.0,
        "name_best_line_jaro": max(
            (
                JaroWinkler.normalized_similarity(fold(line), candidate)
                for line in lines
                if fold(line) and candidate
            ),
            default=0.0,
        ),
        "name_token_set_ratio": fuzz.token_set_ratio(query, candidate) / 100 if valid else 0.0,
    }
