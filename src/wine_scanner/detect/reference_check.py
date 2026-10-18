"""Проверка вырезки этикетки на эталоне каталога: при заведении вина и при аудите индекса.

Эталон каталога режется тем же каскадом, что и запрос («бутылка → этикетка»), и ошибка
детектора на эталоне не видна нигде, кроме ответов: запрос с верной вырезкой ищет то, чего в
индексе нет. Для запроса она случайна, для эталона — постоянна, поэтому эталон проверяется
отдельно, один раз, когда карточка попадает в каталог.

У эталона есть то, чего нет у запроса: это вырезанная бутылка на однородном фоне. Поэтому
бутылку можно найти без детектора — по маске переднего плана, — и сравнить с тем, что нашёл
детектор. Расхождение и есть главный найденный режим ошибки: рамка бутылки накрывает только
верх, и этикетка ищется в горлышке или режется краем рамки (замер 16.10.2026, каталог 2103).

Правила намеренно простые и объяснимые: результат — список флагов, а решение (принять,
починить автоматически, вырезать руками) принимает человек или `repair_rect`.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..image_preprocess import border_color, centered_square, crop_square, prepare_label_image
from .audit import iou
from .bottle import Box, CascadeCropper

# Поправки вырезки эталонов: slug → квадрат этикетки в пикселях исходного эталона. Пишет
# `scripts/check_catalog_crops.py` (source=auto) или человек (source=manual); читают сборка
# индекса и `scripts/patch_index_crops.py`.
OVERRIDES = Path("data/catalog/label_overrides.csv")
# source=accepted — вырезка каскада просмотрена и оставлена: квадрат записан для справки,
# сборка индекса его не применяет.
ACCEPTED = "accepted"
OVERRIDE_FIELDS = ["slug", "x1", "y1", "x2", "y2", "source", "flags", "note"]

# Порог отличия пикселя от фона, по максимуму из каналов. Фон эталона — белый после
# композитинга (или однотонный у части студийных снимков); JPEG-шум у краёв ≤ 15.
FOREGROUND_DELTA = 25
# Рамка бутылки короче переднего плана больше чем на столько — бутылка найдена не целиком.
BOTTLE_COVERAGE = 0.85
# Рамка этикетки ближе к краю рамки бутылки, чем доля высоты бутылки, — она обрезана рамкой.
EDGE_MARGIN = 0.02
# Центр этикетки выше этой доли высоты бутылки — найдена кольеретка или горлышко.
NECK_LEVEL = 0.45
# Уверенность детектора этикетки ниже — рамка сомнительна.
LOW_SCORE = 0.6
# Короткая сторона этикетки на эталоне меньше — сравнивать будет нечего (не ошибка детектора).
SMALL_LABEL_PX = 120


# Предупреждения, а не ошибки вырезки. `no_bottle` на вырезке-эталоне безвреден: этикетка тогда
# ищется по всему кадру, а кадр и есть бутылка (так у 637 эталонов из 2103, вырезки верные).
# `small_label` — эталон слишком мелкий, сравнивать будет нечего, но режется он правильно.
# `low_score` сам по себе — по разбору 16.10 это верные вырезки (Бельбек, Бурлюк, Grape Dance).
WARNINGS = frozenset({"no_bottle", "small_label", "low_score"})


@dataclass
class ReferenceCheck:
    """Что нашёл каскад на эталоне и что в этом подозрительно."""

    size: tuple[int, int]
    foreground: tuple[int, int, int, int] | None
    bottle: tuple[int, int, int, int] | None
    label: tuple[float, float, float, float, float] | None
    alternatives: list[tuple[float, float, float, float, float]] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    # Доли рамки бутылки: ширина и высота этикетки, высота её центра.
    width_share: float = 0.0
    height_share: float = 0.0
    center_level: float = 0.0
    bottle_coverage: float = 1.0

    @property
    def ok(self) -> bool:
        return not [flag for flag in self.flags if flag not in WARNINGS]

    def to_dict(self) -> dict:
        return {
            "size": list(self.size),
            "foreground": list(self.foreground) if self.foreground else None,
            "bottle": list(self.bottle) if self.bottle else None,
            "label": [round(v, 2) for v in self.label] if self.label else None,
            "alternatives": [[round(v, 2) for v in box] for box in self.alternatives],
            "flags": self.flags,
            "width_share": round(self.width_share, 3),
            "height_share": round(self.height_share, 3),
            "center_level": round(self.center_level, 3),
            "bottle_coverage": round(self.bottle_coverage, 3),
        }


def foreground_box(image: Image.Image, delta: int = FOREGROUND_DELTA):
    """Рамка всего, что отличается от фона кадра (медиана внешнего кольца пикселей).

    None — если переднего плана нет или фон неоднороден (кадр не вырезка): тогда сравнивать
    детектор не с чем, и проверка опирается только на геометрию этикетки.
    """
    pixels = np.asarray(image.convert("RGB")).astype(np.int16)
    background = np.array(border_color(image), dtype=np.int16)
    ring = np.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]])
    if np.mean(np.abs(ring - background).max(axis=1) > delta) > 0.5:
        return None
    mask = np.abs(pixels - background).max(axis=2) > delta
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if rows.size == 0:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def locate_label(cascade: CascadeCropper, image: Image.Image, bottle_rect=None):
    """Рамка бутылки, выбранная рамка этикетки и все остальные — в координатах кадра.

    `bottle_rect` задан — первая ступень не вызывается (починка по маске переднего плана).
    """
    if bottle_rect is None:
        bottle_box = cascade.bottle.locate(image)
        bottle_rect = cascade.bottle.crop_rect(bottle_box, image.size) if bottle_box else None
    source = image.crop(bottle_rect) if bottle_rect else image
    boxes = cascade.label.detect(source)
    chosen = cascade.label.pick(boxes, source.size)
    dx, dy = bottle_rect[:2] if bottle_rect else (0, 0)

    def shift(box: Box):
        return (box.x1 + dx, box.y1 + dy, box.x2 + dx, box.y2 + dy, box.score)

    label = shift(chosen) if chosen else None
    others = [shift(b) for b in boxes if b is not chosen]
    return bottle_rect, label, others


def check_reference(
    cascade: CascadeCropper, image: Image.Image, bottle_rect=None
) -> ReferenceCheck:
    """Прогнать каскад на эталоне и разметить подозрительное."""
    foreground = foreground_box(image)
    bottle, label, others = locate_label(cascade, image, bottle_rect)
    check = ReferenceCheck(image.size, foreground, bottle, label, others)
    # Без рамки бутылки её роль играет передний план эталона, а у снимка с фоном — весь кадр.
    frame = bottle or foreground or (0, 0, *image.size)
    bw, bh = frame[2] - frame[0], frame[3] - frame[1]

    if bottle is None:
        check.flags.append("no_bottle")
    if foreground is not None and bottle is not None:
        fg_h = foreground[3] - foreground[1]
        # Высота перекрытия рамки бутылки с передним планом: поля `margin` за край не считаются.
        covered = min(bottle[3], foreground[3]) - max(bottle[1], foreground[1])
        check.bottle_coverage = covered / fg_h if fg_h else 1.0
        if check.bottle_coverage < BOTTLE_COVERAGE:
            check.flags.append("bottle_truncated")
    if label is None:
        check.flags.append("no_label")
        return check

    x1, y1, x2, y2, score = label
    check.width_share = (x2 - x1) / bw
    check.height_share = (y2 - y1) / bh
    check.center_level = ((y1 + y2) / 2 - frame[1]) / bh
    # Этикетка упирается в верх или низ рамки бутылки, которая сама не доходит до края
    # переднего плана, — значит, этикетку обрезала рамка, а не кадр.
    if bottle is not None:
        fg = foreground or (0, 0, *image.size)
        margin = EDGE_MARGIN * bh
        cut_bottom = y2 >= bottle[3] - margin and bottle[3] < fg[3] - margin
        cut_top = y1 <= bottle[1] + margin and bottle[1] > fg[1] + margin
        if cut_bottom or cut_top:
            check.flags.append("label_clipped")
    if check.center_level < NECK_LEVEL:
        check.flags.append("label_on_neck")
    if score < LOW_SCORE:
        check.flags.append("low_score")
    if min(x2 - x1, y2 - y1) < SMALL_LABEL_PX:
        check.flags.append("small_label")
    # Другая уверенная рамка не меньше выбранной и с ней не пересекается: выбор неоднозначен.
    area = (x2 - x1) * (y2 - y1)
    for other in others:
        if (
            other[4] >= 0.3
            and iou(other, label) < 0.3
            and (other[2] - other[0]) * (other[3] - other[1]) >= area
        ):
            check.flags.append("ambiguous")
            break
    return check


def repair_bottle(check: ReferenceCheck) -> tuple[int, int, int, int] | None:
    """Рамка бутылки для повторной попытки: передний план эталона, если детектор его обрезал."""
    if check.foreground is None:
        return None
    if {"bottle_truncated", "label_clipped"} & set(check.flags) or (
        "no_bottle" in check.flags and not check.ok
    ):
        return check.foreground
    return None


def label_square(cascade: CascadeCropper, check: ReferenceCheck) -> tuple[int, int, int, int]:
    """Квадрат вырезки по результату проверки — тот же, что строит `CascadeCropper`."""
    if check.label is not None:
        return cascade.label.crop_rect(Box(*check.label), check.size)
    if check.bottle is not None:
        x1, y1, x2, y2 = check.bottle
        return cascade.estimated_label_square(Box(x1, y1, x2, y2, 1.0), check.size)
    return centered_square(check.size)


# Починка дала почти тот же квадрат — вырезка каскада и была верной, индекс не трогаем.
SAME_CROP_IOU = 0.8


@dataclass
class ReferenceReview:
    """Итог проверки эталона: что делать с его вырезкой.

    ``ok`` — замечаний нет; ``repaired`` — повтор с бутылкой по маске снял все флаги, квадрат
    `rect` идёт в поправки; ``manual`` — квадрат задан человеком; ``accepted`` — человек оставил
    вырезку каскада; ``review`` — нужен человек.
    """

    status: str
    check: ReferenceCheck
    crop: tuple[int, int, int, int]
    rect: tuple[int, int, int, int] | None = None
    retry: ReferenceCheck | None = None

    @property
    def passed(self) -> bool:
        return self.status != "review"

    @property
    def square(self) -> tuple[int, int, int, int]:
        """Квадрат, которым эталон режется в индексе."""
        return self.rect or self.crop


def review_reference(
    cascade: CascadeCropper, image: Image.Image, decision: dict | None = None
) -> ReferenceReview:
    """Проверить эталон и, если можно, починить вырезку. `decision` — строка поправок."""
    check = check_reference(cascade, image)
    crop = label_square(cascade, check)
    if decision and decision.get("source") == ACCEPTED:
        return ReferenceReview("accepted", check, crop)
    if decision:
        return ReferenceReview("manual", check, crop, rect=tuple(decision["rect"]))
    if check.ok:
        return ReferenceReview("ok", check, crop)
    bottle = repair_bottle(check)
    if bottle is None:
        return ReferenceReview("review", check, crop)
    retry = check_reference(cascade, image, bottle_rect=bottle)
    if not retry.ok:
        return ReferenceReview("review", check, crop, retry=retry)
    rect = label_square(cascade, retry)
    if iou(rect, crop) > SAME_CROP_IOU:
        return ReferenceReview("ok", check, crop, retry=retry)
    return ReferenceReview("repaired", check, crop, rect=rect, retry=retry)


def load_overrides(path: Path = OVERRIDES) -> dict[str, dict]:
    """slug → строка поправки с целочисленным `rect`. Нет файла — поправок нет."""
    if not Path(path).exists():
        return {}
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {
        row["slug"]: {**row, "rect": tuple(int(row[k]) for k in ("x1", "y1", "x2", "y2"))}
        for row in rows
        if row.get("slug")
    }


def save_overrides(rows: dict[str, dict], path: Path = OVERRIDES) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OVERRIDE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for slug in sorted(rows):
            x1, y1, x2, y2 = rows[slug]["rect"]
            writer.writerow({**rows[slug], "slug": slug, "x1": x1, "y1": y1, "x2": x2, "y2": y2})


class OverrideCropper:
    """Обрезка эталонов каталога с поправками: для slug из файла — сохранённый квадрат.

    Действует только на вызов `(path, image)`, которым режется каталог при сборке индекса;
    запрос (`crop`) режется каскадом как есть — поправка есть свойство эталона, а не модели.
    """

    def __init__(self, inner, overrides: dict[str, dict]):
        self.inner = inner
        self.overrides = {k: v for k, v in overrides.items() if v.get("source") != ACCEPTED}
        self.cache_tag = getattr(inner, "cache_tag", "?")

    def __call__(self, path, image: Image.Image) -> Image.Image:
        row = self.overrides.get(Path(path).stem)
        if row is None:
            return self.inner(path, image)
        return prepare_label_image(crop_square(image, row["rect"]))

    def crop(self, image: Image.Image) -> Image.Image:
        return self.inner.crop(image)


# --- Листы на ручной разбор ------------------------------------------------------------------

TILE = 260


def tile(image: Image.Image, check, old_rect, new_rect, title: str) -> Image.Image:
    """Эталон с рамками (бутылка синим, этикетка красным, починка зелёным) и две вырезки."""
    shown = image.convert("RGB")
    draw = ImageDraw.Draw(shown)
    width = max(2, shown.width // 150)
    if check.bottle:
        draw.rectangle(check.bottle, outline=(0, 0, 255), width=width)
    for other in check.alternatives:
        draw.rectangle(other[:4], outline=(160, 160, 160), width=width)
    if check.label:
        draw.rectangle(check.label[:4], outline=(255, 0, 0), width=width * 2)
    if new_rect:
        draw.rectangle(new_rect, outline=(0, 180, 0), width=width * 2)
    shown = shown.resize((max(1, shown.width * TILE // shown.height), TILE))
    old = prepare_label_image(crop_square(image, old_rect), TILE)
    new = prepare_label_image(crop_square(image, new_rect), TILE) if new_rect else None
    canvas = Image.new("RGB", (shown.width + 2 * TILE + 20, TILE + 18), "white")
    canvas.paste(shown, (0, 18))
    canvas.paste(old, (shown.width + 10, 18))
    if new is not None:
        canvas.paste(new, (shown.width + TILE + 20, 18))
    ImageDraw.Draw(canvas).text((2, 2), title[:90], fill="black")
    return canvas


def sheet(tiles: list[Image.Image], path: Path, columns: int = 2, per_page: int = 40) -> None:
    """Листы по `per_page` плиток: `review.jpg`, `review_2.jpg`, …"""
    for page, start in enumerate(range(0, len(tiles), per_page), start=1):
        chunk = tiles[start : start + per_page]
        width = max(t.width for t in chunk)
        height = max(t.height for t in chunk)
        rows = (len(chunk) + columns - 1) // columns
        canvas = Image.new("RGB", (width * columns, height * rows), "white")
        for n, t in enumerate(chunk):
            canvas.paste(t, ((n % columns) * width, (n // columns) * height))
        target = path if page == 1 else path.with_name(f"{path.stem}_{page}{path.suffix}")
        canvas.save(target, quality=85)
