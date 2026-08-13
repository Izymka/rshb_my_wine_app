"""Где на нашем кадре напечатан год — и как прочитать его крупно.

Идея этапа целиком в одном наблюдении. Общее распознавание работает по кадру, ужатому до 640
пикселей (так решено на Э11: время OCR линейно по площади). Название на такой картинке
читается прекрасно, а год — четыре мелких цифры, часто вдавленных или выбитых золотом, —
разваливается: на нашем наборе он не прочитался в трети кадров, и почти все промахи это
`blur` и `far`.

Дочитывать весь кадр в полном разрешении — 15 секунд, весь бюджет запроса. Но и не нужно:
после ре-ранкинга у нас есть гомография, то есть точное соответствие между нашей фотографией
и эталоном каталога. На эталоне мы знаем, где напечатан год. Значит, знаем и где он на нашем
кадре — и вырезаем оттуда участок в исходном разрешении. Это сотня килопикселей вместо
четырёх мегапикселей.

Отдельно про доверие к геометрии. Гомографию нам даёт лучший кандидат, а он может оказаться
не тем вином. Поэтому участок вырезается с запасом, прочитанное проверяется тем же правилом
правдоподобия, а при любом сомнении блок молчит: неверный год хуже отсутствующего.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from ..ocr import LabelOCR, TextLine
from .read import Reading, from_text, pick, years

Box = tuple[float, float, float, float]

# Запас вокруг участка, в долях его собственного размера. Полкоробки с каждой стороны —
# это компромисс: меньше, и ошибка гомографии срежет цифру; больше, и в кадр войдёт соседняя
# строка, из которой OCR вытащит чужое число.
DEFAULT_MARGIN = 0.6

# До какой высоты растягивается участок и куда упирается увеличение. Смысл не в том, чтобы
# дать распознавателю больше пикселей — строку он всё равно приводит к своей высоте, — а в том,
# чтобы детектор текста вообще заметил четыре цифры на мелком участке.
ZOOM_TARGET_HEIGHT = 128
ZOOM_MAX_SIDE = 1024

# Ниже этого числа инлаеров геометрия считается ненадёжной и участок не вырезается вовсе.
# Порог взят с запасом к минимуму RANSAC (четыре точки): четыре случайные точки складываются
# в «согласованную» гомографию слишком легко.
MIN_INLIERS = 12


@dataclass(frozen=True)
class YearRegion:
    """Участок эталона, на котором напечатан год."""

    box: Box
    year: int


def reference_region(lines: list[TextLine], known: set[int]) -> YearRegion | None:
    """Найти на эталоне строку с годом.

    Когда каталог знает год карточки (а он его обычно знает — это поле карточки, а не наша
    догадка), выбор однозначен: берём строку с этим самым годом. Именно так снимается
    неоднозначность, на которой спотыкается чистое распознавание: на эталоне Mouton Cadet
    2023 OCR видит два числа, 2025 и 2023, и без подсказки каталога выбрать не из чего.

    Если год карточке неизвестен, соглашаемся только на единственный найденный — иначе
    рискуем принять за винтаж год основания хозяйства.
    """
    found: list[tuple[TextLine, int]] = []
    for line in lines:
        if line.box is None:
            continue
        for year in years(line.text):
            found.append((line, year))

    if not found:
        return None

    if known:
        matching = [(line, year) for line, year in found if year in known]
        if not matching:
            return None
        # Самая крупная строка: год на этикетке печатают заметно, а мелкое совпадение
        # чаще всего часть артикула или адреса разливщика.
        line, year = max(matching, key=lambda pair: _area(pair[0].box))
        return YearRegion(line.box, year)

    distinct = {year for _, year in found}
    if len(distinct) != 1:
        return None
    line, year = max(found, key=lambda pair: _area(pair[0].box))
    return YearRegion(line.box, year)


def project(box: Box, match) -> Box | None:
    """Перенести участок с эталона на кадр запроса по гомографии.

    Координаты по всему блоку нормированные, а матрица посчитана в пикселях уменьшенных копий,
    поэтому по краям идут два пересчёта. Возня выглядит лишней ровно до первого раза, когда
    кто-нибудь передаст доли туда, где ждут пиксели: участок уедет в угол, блок начнёт молча
    читать пустоту, а выглядеть это будет как «увеличение не помогает».
    """
    if match is None or getattr(match, "homography", None) is None:
        return None
    if not match.homography_ok or match.inliers < MIN_INLIERS:
        return None
    if not match.candidate_size or not match.query_size:
        return None

    width, height = match.candidate_size
    x0, y0, x1, y1 = box
    corners = np.array(
        [
            [x0 * width, y0 * height],
            [x1 * width, y0 * height],
            [x1 * width, y1 * height],
            [x0 * width, y1 * height],
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    moved = cv2.perspectiveTransform(corners, match.homography).reshape(-1, 2)
    if not np.isfinite(moved).all():
        return None

    query_width, query_height = match.query_size
    xs = moved[:, 0] / query_width
    ys = moved[:, 1] / query_height
    projected = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))

    # Участок, уехавший за пределы кадра или растянувшийся на пол-этикетки, означает, что
    # гомография врёт. Формально она прошла проверку определителем, но переносить по ней
    # координаты уже нельзя.
    if not _inside(projected) or _area(projected) > 0.25:
        return None
    return projected


def patch(image: Image.Image, box: Box, margin: float = DEFAULT_MARGIN) -> Image.Image | None:
    """Вырезать участок с запасом, увеличить и вытянуть контраст.

    Контраст здесь не украшение. Год печатают тиснением или тёмной краской по тёмному фону
    (на Mouton Cadet это бордовые цифры по чёрному), и на таком участке распознаватель молчит,
    хотя цифры глазом различимы. CLAHE выравнивает освещённость по клеткам и вытягивает
    именно локальный перепад — из четырёх опробованных вариантов подготовки он единственный
    добавил прочитанных кадров, не добавив ошибочных.
    """
    x0, y0, x1, y1 = _expand(box, margin)
    width, height = image.size
    left, top = int(x0 * width), int(y0 * height)
    right, bottom = int(np.ceil(x1 * width)), int(np.ceil(y1 * height))
    if right - left < 8 or bottom - top < 8:
        return None

    cut = image.crop((left, top, right, bottom))
    scale = min(ZOOM_MAX_SIDE / max(cut.size), max(1.0, ZOOM_TARGET_HEIGHT / cut.height))
    if scale > 1.0:
        cut = cut.resize((int(cut.width * scale), int(cut.height * scale)), Image.LANCZOS)

    grey = cv2.cvtColor(np.asarray(cut.convert("RGB")), cv2.COLOR_RGB2GRAY)
    equalized = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(grey)
    return Image.fromarray(equalized).convert("RGB")


class VintageReader:
    """Год урожая по кадру: сначала из общего текста, затем — если не вышло — по геометрии.

    Порядок не случаен и стоит денег ровно тогда, когда помогает. Общий текст у нас уже есть,
    он бесплатен; увеличение участка это ещё один вызов распознавания, и запускается он
    только на тех кадрах, где год иначе потерян.
    """

    def __init__(
        self,
        ocr: LabelOCR,
        margin: float = DEFAULT_MARGIN,
        min_inliers: int = MIN_INLIERS,
    ):
        self.ocr = ocr
        self.margin = margin
        self.min_inliers = min_inliers

    def from_lines(self, lines: list[TextLine]) -> Reading:
        return from_text(LabelOCR.joined(lines))

    def zoom(self, crop: Image.Image, box: Box) -> Reading:
        """Прочитать год на увеличенном участке.

        Читаем распознавателем, ограниченным цифрами: в участке нет ничего, кроме числа,
        а без ограничения OCR норовит увидеть в «2021» слово. Проверка правдоподобия та же,
        что и везде, — участок мог быть вырезан не там.
        """
        cut = patch(crop, box, self.margin)
        if cut is None:
            return Reading()
        reading = pick(years(self.ocr.read_digits(cut)))
        return Reading(reading.year, "zoom" if reading.year else "", reading.candidates)


def _area(box: Box) -> float:
    x0, y0, x1, y1 = box
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _inside(box: Box) -> bool:
    x0, y0, x1, y1 = box
    return 0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0


def _expand(box: Box, margin: float) -> Box:
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * margin, (y1 - y0) * margin
    return (max(0.0, x0 - dx), max(0.0, y0 - dy), min(1.0, x1 + dx), min(1.0, y1 + dy))
