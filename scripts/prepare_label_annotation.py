"""Подготовить кадры к разметке этикеток — для дообучения детектора.

    uv run python scripts/prepare_label_annotation.py
    uv run python scripts/prepare_label_annotation.py --out data/live_labels --max-side 1600

Детектор этикетки в каскаде работает не по целому кадру, а по вырезке бутылки, которую даёт
первый детектор (COCO). Значит, размечать надо ровно такие вырезки — иначе обучение увидит не
то, что увидит инференс. Скрипт прогоняет каждый живой кадр через детектор бутылки, сохраняет
вырезку в `--out` под именем `<вино>__<кадр>.jpg` и рядом кладёт `_images.txt` со списком.

Дальше руками, 10–15 минут на сотню кадров:

1. Открыть https://www.makesense.ai → Get Started → перетащить папку `data/live_labels/`
   (все jpg) → Object Detection → создать один класс `label` → Start project.
2. На каждом кадре обвести **всю лицевую этикетку целиком, от края до края бумаги**, включая
   поля; кольеретку на горлышке и контрэтикетку не обводить. Если этикетка обрезана краем
   кадра — обвести то, что видно. Если этикетки нет или она неразличима — пропустить кадр.
3. Actions → Export Annotations → «Single file in COCO JSON format» → сохранить как
   `data/live_labels/_annotations.coco.json`.

Потом на машине с видеокартой:

    uv run python scripts/train_label_detector.py --data "data/third-party datasets/wine-labels" \\
        --own data/live_labels --init models/label_detector.pt --eval-only        # до
    uv run python scripts/train_label_detector.py --data "data/third-party datasets/wine-labels" \\
        --own data/live_labels --init models/label_detector.pt --own-repeat 20 --epochs 4 \\
        --device cuda --out models/label_detector_v2.pt                               # после

Новый детектор меняет кропы, значит индекс и дескрипторы пересобираются (`build_index.py`
с `--weights models/label_detector_v2.pt`), и только потом — бенчмарк до/после.
"""

import argparse
from pathlib import Path

from tqdm import tqdm

from wine_scanner.catalog import LIVE_MANIFEST, load_live
from wine_scanner.detect import BottleDetector
from wine_scanner.embed import load_image, pick_device

OUT = Path("data/live_labels")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=LIVE_MANIFEST)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--max-side", type=int, default=1600, help="ужать вырезку для разметки")
    parser.add_argument(
        "--sources", default="live,own,eval", help="какие кадры брать: live, own, eval"
    )
    args = parser.parse_args()

    sources = set(args.sources.split(","))
    queries = [q for q in load_live(args.manifest, include_multi=False) if q.source in sources]
    detector = BottleDetector(device=pick_device(), mode="bottle")
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
