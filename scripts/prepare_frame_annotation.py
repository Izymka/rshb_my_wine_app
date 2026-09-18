"""Подготовить исходные кадры (до кропа) к разметке бутылок и этикеток — для RT-DETR.

    uv run python scripts/prepare_frame_annotation.py                                   # обучение: data/own -> data/own_frames
    uv run python scripts/prepare_frame_annotation.py --sources live,eval --out data/live_frames  # тест
    uv run python scripts/prepare_frame_annotation.py --extra data/own/shelf_raw --out data/own_frames  # + полки

Чем это отличается от `prepare_label_annotation.py`. Тот скрипт режет кадр детектором бутылки
и отдаёт на разметку вырезку — так учится второй детектор каскада. Здесь на разметку идёт
**весь кадр**, и классов два: `bottle` и `label`. Один детектор по целому кадру заменяет каскад
(COCO-бутылка → этикетка на вырезке): не режет края этикетки из-за плохой рамки бутылки и
не тратит два прогона на кадр.

Две папки, и они не смешиваются — те же, что у вырезок: `data/own_frames` — свой набор
(импорт, снят во Вьетнаме) плюс, по желанию, кадры полок `data/own/shelf_raw`, на них детектор
**учится**; `data/live_frames` — кадры российских вин и публичные кадры организаторов,
изолированный **тест** (`catalog.is_holdout`). Кадры полок из манифеста (`group == multi`)
попадают в тест вместе с остальными `data/live`.

Кадр ужимается до `--max-side` (1600 px): RT-DETR всё равно смотрит на 640, а разметке
хватает, зато папка весит десятки мегабайт, а не гигабайты, и makesense не тормозит.
Ориентация EXIF применена (`load_image`), поэтому кадр в makesense, на диске и в обучении —
один и тот же.

**Предразметка.** Рамки этикеток на вырезках уже есть (`data/own_labels`, `data/live_labels`,
285 штук), рисовать их заново незачем. Скрипт прогоняет детектор бутылки заново, восстанавливает
прямоугольник вырезки и переносит рамку этикетки в координаты кадра; рамки всех бутылок,
которые нашёл COCO-детектор, кладёт классом `bottle`. Итог — `_prefill.coco.json`: в makesense
после загрузки картинок Actions → Import Annotations → COCO. Дальше руками: поправить
рамки бутылок (COCO-детектор их часто режет по горлышку), дорисовать этикетки соседних
бутылок, где они видны целиком, удалить лишнее. Если размер вырезки в старой разметке не
сошёлся с пересчитанным (детектор на другом устройстве дал другую рамку), рамка этикетки
переносится по масштабу и в отчёт попадает предупреждение — такие кадры проверить в первую
очередь.

Правила разметки:

1. `bottle` — вся бутылка от пробки до дна, включая то, что закрыто рукой или соседом.
   Обводить все бутылки, у которых видно больше половины; на полках — каждую.
2. `label` — вся лицевая этикетка от края до края бумаги, включая поля; кольеретку и
   контрэтикетку не обводить. Этикетка обрезана краем кадра — обвести видимое.
3. Export → Single CSV file → `<папка>/_annotations.csv` (или COCO JSON —
   `_annotations.coco.json`, загрузчик `detect/dataset.py` понимает оба).

Дальше на видеокарте — `scripts/train_rtdetr.py`, ему нужен режим на два класса.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from wine_scanner.catalog import LIVE_MANIFEST, Query, load_live
from wine_scanner.detect import BottleDetector, CocoDetectionDataset
from wine_scanner.embed import load_image, pick_device

OUT = Path("data/own_frames")
CATEGORIES = [{"id": 1, "name": "bottle"}, {"id": 2, "name": "label"}]
BOTTLE, LABEL = 1, 2


def crop_to_frame(
    box: tuple[float, float, float, float],
    crop_rect: tuple[int, int, int, int],
    crop_saved: tuple[int, int],
    frame_scale: tuple[float, float],
) -> tuple[float, float, float, float]:
    """Рамку `(x, y, w, h)` с ужатой вырезки вернуть в координаты ужатого кадра.

    Три преобразования по порядку: обратно из масштаба, в котором вырезка была сохранена на
    разметку; сдвиг на левый верхний угол вырезки внутри кадра; масштаб, в котором сохранён
    сам кадр.
    """
    x, y, w, h = box
    cx1, cy1, cx2, cy2 = crop_rect
    kx = (cx2 - cx1) / crop_saved[0]
    ky = (cy2 - cy1) / crop_saved[1]
    sx, sy = frame_scale
    return (
        (cx1 + x * kx) * sx,
        (cy1 + y * ky) * sy,
        w * kx * sx,
        h * ky * sy,
    )


def load_crop_labels(root: Path) -> dict[str, tuple[tuple[int, int], list[tuple]]]:
    """Разметка вырезок: имя файла -> (размер вырезки, рамки этикетки x,y,w,h)."""
    if (root / CocoDetectionDataset.COCO_FILE).exists():
        images, annotations = CocoDetectionDataset._read_coco(root / CocoDetectionDataset.COCO_FILE)
    elif (root / CocoDetectionDataset.CSV_FILE).exists():
        images, annotations = CocoDetectionDataset._read_makesense_csv(root / CocoDetectionDataset.CSV_FILE)
    else:
        return {}
    boxes = defaultdict(list)
    for image_id, bbox in annotations:
        boxes[image_id].append(bbox)
    return {
        img["file_name"]: ((img["width"], img["height"]), boxes.get(img_id, []))
        for img_id, img in images.items()
    }


def extra_queries(folder: Path) -> list[Query]:
    """Кадры без манифеста (полки `data/own/shelf_raw`): вино — имя папки."""
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic"})
    return [Query(path=p, true_id="", group="shelf", wine_id=folder.name) for p in paths]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=LIVE_MANIFEST)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--max-side", type=int, default=1600, help="ужать кадр для разметки; 0 — оригинал")
    parser.add_argument(
        "--sources", default="own", help="какие кадры брать: own (обучение), live, eval (тест)"
    )
    parser.add_argument(
        "--labels",
        type=Path,
        help="папка с разметкой вырезок для предразметки (по умолчанию own_labels / live_labels)",
    )
    parser.add_argument("--extra", type=Path, action="append", default=[], help="папка кадров без манифеста")
    parser.add_argument("--no-prefill", action="store_true", help="только кадры, без детектора")
    args = parser.parse_args()

    sources = set(args.sources.split(","))
    queries = [q for q in load_live(args.manifest, include_multi=True) if q.source in sources]
    for folder in args.extra:
        queries += extra_queries(folder)
    if args.labels is None and not args.no_prefill:
        args.labels = Path("data/live_labels" if sources & {"live", "eval"} else "data/own_labels")
    crop_labels = load_crop_labels(args.labels) if args.labels else {}
    detector = None if args.no_prefill else BottleDetector(device=pick_device(), mode="bottle")
    args.out.mkdir(parents=True, exist_ok=True)

    names, images, annotations, warnings = [], [], [], []
    transferred = 0
    for q in tqdm(queries, desc="кадры"):
        name = f"{q.wine_id}__{q.path.stem}.jpg"
        target = args.out / name
        image = load_image(q.path)
        thumb = image.copy()
        if args.max_side > 0:
            thumb.thumbnail((args.max_side, args.max_side))
        if not target.exists():
            thumb.save(target, quality=90)
        names.append(name)
        image_id = len(images)
        images.append({"id": image_id, "file_name": name, "width": thumb.width, "height": thumb.height})
        if detector is None:
            continue

        scale = (thumb.width / image.width, thumb.height / image.height)
        boxes = detector.detect(image)
        for box in boxes:
            annotations.append(
                _annotation(image_id, BOTTLE, (box.x1 * scale[0], box.y1 * scale[1],
                                               (box.x2 - box.x1) * scale[0], (box.y2 - box.y1) * scale[1]),
                            score=box.score)
            )
        if name not in crop_labels:
            continue
        picked = detector.pick(boxes, image.size)
        # Бутылка не найдена — `crop` отдал кадр целиком, и разметка сделана на нём.
        rect = (0, 0, *image.size) if picked is None else detector.crop_rect(picked, image.size)
        saved, label_boxes = crop_labels[name]
        # Как вырезка была бы сохранена prepare_label_annotation.py: thumbnail до 1600.
        crop_size = (rect[2] - rect[0], rect[3] - rect[1])
        k = min(1.0, 1600 / max(crop_size))
        expected = (round(crop_size[0] * k), round(crop_size[1] * k))
        if max(abs(expected[0] - saved[0]), abs(expected[1] - saved[1])) > 2:
            warnings.append(f"{name}: вырезка в разметке {saved}, пересчитанная {expected}")
        for box in label_boxes:
            annotations.append(_annotation(image_id, LABEL, crop_to_frame(box, rect, saved, scale)))
            transferred += 1

    (args.out / "_images.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    if detector is not None:
        coco = {"images": images, "categories": CATEGORIES, "annotations": annotations}
        (args.out / "_prefill.coco.json").write_text(
            json.dumps(coco, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        bottles = sum(1 for a in annotations if a["category_id"] == BOTTLE)
        print(f"предразметка: бутылок {bottles}, этикеток перенесено {transferred} "
              f"из {sum(len(v[1]) for v in crop_labels.values())} в {args.out}/_prefill.coco.json")
        if warnings:
            (args.out / "_prefill_warnings.txt").write_text("\n".join(warnings) + "\n", encoding="utf-8")
            print(f"проверить в первую очередь ({len(warnings)}): {args.out}/_prefill_warnings.txt")
    print(f"кадров для разметки: {len(names)} в {args.out}/ — дальше makesense.ai, см. docstring")


def _annotation(image_id: int, category: int, bbox: tuple, score: float | None = None) -> dict:
    x, y, w, h = (round(float(v), 1) for v in bbox)
    ann = {
        "id": 0,  # проставляется ниже
        "image_id": image_id,
        "category_id": category,
        "bbox": [x, y, w, h],
        "area": round(w * h, 1),
        "iscrowd": 0,
    }
    if score is not None:
        ann["score"] = round(score, 3)
    ann["id"] = _annotation.counter = getattr(_annotation, "counter", 0) + 1
    return ann


if __name__ == "__main__":
    main()
