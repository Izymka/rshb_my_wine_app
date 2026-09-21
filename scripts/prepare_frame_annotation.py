"""Подготовить исходные кадры (до кропа) к разметке бутылок и этикеток — для RT-DETR.

    uv run python scripts/prepare_frame_annotation.py                                    # data/test: новое → frames_unannotated
    uv run python scripts/prepare_frame_annotation.py --manifest data/train/manifest.csv  # data/train
    uv run python scripts/prepare_frame_annotation.py --manifest data/train/manifest.csv --extra data/own/shelf_raw  # + полки

Чем это отличается от `prepare_label_annotation.py`. Тот скрипт режет кадр детектором бутылки
и отдаёт на разметку вырезку — так учится второй детектор каскада. Здесь на разметку идёт
**весь кадр**, и классов два: `bottle` и `label`. Один детектор по целому кадру заменяет каскад
(COCO-бутылка → этикетка на вырезке): не режет края этикетки из-за плохой рамки бутылки и
не тратит два прогона на кадр.

Раскладка с 18.09.2026: у каждого манифеста (`data/test`, `data/train`) рядом лежат две папки
кадров — `frames_annotated/` (кадры + `_annotations.csv` из makesense) и `frames_unannotated/`
(то, что ещё предстоит разметить). Скрипт берёт кадры манифеста, пропускает те, что уже есть
в любой из двух папок, и кладёт новые в `frames_unannotated/` с предразметкой. После разметки
кадры вместе со строками `_annotations.csv` переносятся в `frames_annotated/` — руками или
`--merge`. `data/test` — изолированный тест (`catalog.is_holdout`), `data/train` — обучение;
они не смешиваются.

Кадр ужимается до `--max-side` (1600 px): RT-DETR всё равно смотрит на 640, а разметке
хватает, зато папка весит десятки мегабайт, а не гигабайты, и makesense не тормозит.
Ориентация EXIF применена (`load_image`), поэтому кадр в makesense, на диске и в обучении —
один и тот же.

**Предразметка.** Рамки этикеток на вырезках, если они есть (`<корень>/labels/`, 285 штук
на 18.09), рисовать заново незачем. Скрипт прогоняет детектор бутылки заново, восстанавливает
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
import csv
import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from wine_scanner.catalog import TEST_MANIFEST, Query, load_manifest
from wine_scanner.detect import CocoDetectionDataset, bottle_detector
from wine_scanner.embed import load_image, pick_device

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
        images, annotations = CocoDetectionDataset._read_makesense_csv(
            root / CocoDetectionDataset.CSV_FILE
        )
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
    paths = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic"}
    )
    return [Query(path=p, true_id="", group="shelf", wine_id=folder.name) for p in paths]


def merge_annotated(root: Path) -> None:
    """Перенести размеченные кадры из frames_unannotated в frames_annotated вместе со строками CSV."""
    src, dst = root / "frames_unannotated", root / "frames_annotated"
    csv_src, csv_dst = src / "_annotations.csv", dst / "_annotations.csv"
    if not csv_src.exists():
        raise SystemExit(f"нет {csv_src}: экспортируйте разметку из makesense (Single CSV file)")
    rows = list(csv.DictReader(csv_src.open(encoding="utf-8", newline="")))
    if not rows:
        raise SystemExit(f"файл {csv_src} пуст")
    done = {r["image_name"] for r in rows}
    dst.mkdir(exist_ok=True)
    existing = (
        list(csv.DictReader(csv_dst.open(encoding="utf-8", newline=""))) if csv_dst.exists() else []
    )
    with csv_dst.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(existing + rows)
    for name in sorted(done):
        if (src / name).exists():
            (src / name).rename(dst / name)
    csv_src.rename(src / "_annotations.merged.csv")
    left = [p.name for p in src.iterdir() if p.suffix.lower() == ".jpg"]
    print(f"перенесено кадров {len(done)}, рамок {len(rows)}; осталось без разметки {len(left)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=TEST_MANIFEST)
    parser.add_argument(
        "--out", type=Path, help="по умолчанию <корень манифеста>/frames_unannotated"
    )
    parser.add_argument(
        "--max-side", type=int, default=1600, help="ужать кадр для разметки; 0 — оригинал"
    )
    parser.add_argument(
        "--sources", default="live,own,eval", help="какие кадры манифеста брать: live, own, eval"
    )
    parser.add_argument(
        "--labels",
        type=Path,
        help="папка с разметкой вырезок для предразметки (по умолчанию <корень>/labels)",
    )
    parser.add_argument(
        "--extra", type=Path, action="append", default=[], help="папка кадров без манифеста"
    )
    parser.add_argument("--no-prefill", action="store_true", help="только кадры, без детектора")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="вместо подготовки: перенести размеченное из frames_unannotated в frames_annotated",
    )
    args = parser.parse_args()

    root = args.manifest.parent
    if args.merge:
        merge_annotated(root)
        return
    args.out = args.out or root / "frames_unannotated"
    annotated = root / "frames_annotated"
    already = {
        p.name for folder in (annotated, args.out) if folder.exists() for p in folder.iterdir()
    }

    sources = set(args.sources.split(","))
    queries = [q for q in load_manifest(args.manifest, include_multi=True) if q.source in sources]
    for folder in args.extra:
        queries += extra_queries(folder)
    queries = [q for q in queries if f"{q.wine_id}__{q.path.stem}.jpg" not in already]
    print(f"кадров в манифесте и --extra: новых {len(queries)}, уже подготовлено {len(already)}")
    if not queries:
        return
    if args.labels is None and not args.no_prefill:
        args.labels = root / "labels"
    crop_labels = load_crop_labels(args.labels) if args.labels and args.labels.exists() else {}
    if crop_labels and not args.no_prefill:
        raise SystemExit(
            "Legacy crops have no saved crop transform; cannot reconstruct frame annotations "
            "with a new detector. Use --no-prefill or annotate original frames."
        )
    detector = None if args.no_prefill else bottle_detector("rtdetr", pick_device())
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
        images.append(
            {"id": image_id, "file_name": name, "width": thumb.width, "height": thumb.height}
        )
        if detector is None:
            continue

        scale = (thumb.width / image.width, thumb.height / image.height)
        boxes = detector.detect(image)
        for box in boxes:
            annotations.append(
                _annotation(
                    image_id,
                    BOTTLE,
                    (
                        box.x1 * scale[0],
                        box.y1 * scale[1],
                        (box.x2 - box.x1) * scale[0],
                        (box.y2 - box.y1) * scale[1],
                    ),
                    score=box.score,
                )
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

    with (args.out / "_images.txt").open("a", encoding="utf-8") as f:
        f.write("\n".join(names) + "\n")
    if detector is not None:
        prefill_path = args.out / "_prefill.coco.json"
        if prefill_path.exists():
            old = json.loads(prefill_path.read_text(encoding="utf-8"))
            shift = max((i["id"] for i in old["images"]), default=-1) + 1
            for img in images:
                img["id"] += shift
            ann_shift = max((a["id"] for a in old["annotations"]), default=0)
            for ann in annotations:
                ann["image_id"] += shift
                ann["id"] += ann_shift
            images, annotations = old["images"] + images, old["annotations"] + annotations
        coco = {"images": images, "categories": CATEGORIES, "annotations": annotations}
        prefill_path.write_text(json.dumps(coco, ensure_ascii=False, indent=1), encoding="utf-8")
        bottles = sum(1 for a in annotations if a["category_id"] == BOTTLE)
        print(
            f"предразметка: бутылок {bottles}, этикеток перенесено {transferred} "
            f"из {sum(len(v[1]) for v in crop_labels.values())} в {args.out}/_prefill.coco.json"
        )
        if warnings:
            (args.out / "_prefill_warnings.txt").write_text(
                "\n".join(warnings) + "\n", encoding="utf-8"
            )
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
