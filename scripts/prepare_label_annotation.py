"""Подготовить кадры к разметке этикеток — для дообучения детектора.

    uv run python scripts/prepare_label_annotation.py                       # обучение: data/own -> data/own_labels
    uv run python scripts/prepare_label_annotation.py --sources live,eval --out data/live_labels   # тест

Детектор этикетки в каскаде работает не по целому кадру, а по вырезке бутылки, которую даёт
первый детектор (COCO). Значит, размечать надо ровно такие вырезки — иначе обучение увидит не
то, что увидит инференс. Скрипт прогоняет каждый кадр через детектор бутылки, сохраняет
вырезку в `--out` под именем `<вино>__<кадр>.jpg` и рядом кладёт `_images.txt` со списком.

Две папки, и они не смешиваются: `data/own_labels` — свой набор (импорт, снят во Вьетнаме),
на нём детектор **учится**; `data/live_labels` — кадры российских вин и публичные кадры
организаторов, изолированный **тест** (`catalog.is_holdout`), размечен 17.09 (169 рамок).

Дальше руками, 10–15 минут на сотню кадров:

1. Открыть https://www.makesense.ai → Get Started → перетащить папку `data/own_labels/`
   (все jpg) → Object Detection → создать один класс `label` → Start project.
2. На каждом кадре обвести **всю лицевую этикетку целиком, от края до края бумаги**, включая
   поля; кольеретку на горлышке и контрэтикетку не обводить. Если этикетка обрезана краем
   кадра — обвести то, что видно. Если этикетки нет или она неразличима — пропустить кадр.
3. Actions → Export Annotations → **«Single CSV file»** → сохранить как
   `data/own_labels/_annotations.csv` (COCO JSON, если он предложен, тоже подходит —
   `_annotations.coco.json`; загрузчик `detect/dataset.py` понимает оба).

Потом на машине с видеокартой:

    uv run python scripts/train_label_detector.py --data "data/third-party datasets/wine-labels" \\
        --test data/live_labels --init models/label_detector.pt --eval-only                # до
    uv run python scripts/train_label_detector.py --data "data/third-party datasets/wine-labels" \\
        --own data/own_labels --test data/live_labels --init models/label_detector.pt \\
        --own-repeat 20 --epochs 4 --device cuda --out models/label_detector_v2.pt       # после

Новый детектор меняет кропы, значит индекс и дескрипторы пересобираются (`build_index.py`
с `--weights models/label_detector_v2.pt`), и только потом — бенчмарк до/после.
"""

import argparse
from pathlib import Path

from tqdm import tqdm

from wine_scanner.catalog import LIVE_MANIFEST, load_live
from wine_scanner.detect import bottle_detector
from wine_scanner.embed import load_image, pick_device

OUT = Path("data/own_labels")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=LIVE_MANIFEST)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--max-side", type=int, default=1600, help="ужать вырезку для разметки")
    parser.add_argument(
        "--sources", default="own", help="какие кадры брать: own (обучение), live, eval (тест)"
    )
    args = parser.parse_args()

    sources = set(args.sources.split(","))
    queries = [q for q in load_live(args.manifest, include_multi=False) if q.source in sources]
    # Та же первая ступень, что в каскаде: вырезка бутылки RT-DETR по весам COCO.
    detector = bottle_detector("rtdetr", pick_device())
    args.out.mkdir(parents=True, exist_ok=True)

    names = []
    for q in tqdm(queries, desc="вырезки бутылок"):
        name = f"{q.wine_id}__{q.path.stem}.jpg"
        target = args.out / name
        if not target.exists():
            crop = detector.crop(load_image(q.path))
            crop.thumbnail((args.max_side, args.max_side))
            crop.save(target, quality=90)
        names.append(name)
    (args.out / "_images.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"кадров для разметки: {len(names)} в {args.out}/ — дальше makesense.ai, см. docstring")


if __name__ == "__main__":
    main()
