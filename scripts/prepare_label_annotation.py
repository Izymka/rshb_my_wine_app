"""Подготовить кадры к разметке этикеток — для дообучения детектора.

    uv run python scripts/prepare_label_annotation.py --manifest data/train/manifest.csv   # обучение → data/train/labels
    uv run python scripts/prepare_label_annotation.py                                       # тест → data/test/labels

Детектор этикетки в каскаде работает не по целому кадру, а по вырезке бутылки, которую даёт
первый детектор (COCO). Значит, размечать надо ровно такие вырезки — иначе обучение увидит не
то, что увидит инференс. Скрипт прогоняет каждый кадр через детектор бутылки, сохраняет
вырезку в `--out` под именем `<вино>__<кадр>.jpg` и рядом кладёт `_images.txt` со списком.

Две папки, и они не смешиваются: `data/train/labels` — свой набор (импорт, снят во Вьетнаме)
и незнакомцы вне теста, на них детектор **учится**; `data/test/labels` — тестовые кадры российских вин и публичные кадры
организаторов, изолированный **тест** (`catalog.is_holdout`), размечен 17.09 (169 рамок).

Дальше руками, 10–15 минут на сотню кадров:

1. Открыть https://www.makesense.ai → Get Started → перетащить папку `data/train/labels/`
   (все jpg) → Object Detection → создать один класс `label` → Start project.
2. На каждом кадре обвести **всю лицевую этикетку целиком, от края до края бумаги**, включая
   поля; кольеретку на горлышке и контрэтикетку не обводить. Если этикетка обрезана краем
   кадра — обвести то, что видно. Если этикетки нет или она неразличима — пропустить кадр.
3. Actions → Export Annotations → **«Single CSV file»** → сохранить как
   `data/train/labels/_annotations.csv` (COCO JSON, если он предложен, тоже подходит —
   `_annotations.coco.json`; загрузчик `detect/dataset.py` понимает оба).

Для переноса уже существующей разметки исходных кадров используйте
scripts/audit_rtdetr_crops.py: он сохраняет точное преобразование и COCO-разметку кропов.
Для обучения RT-DETR используйте scripts/train_rtdetr.py на проверенных train/valid;
тест передаётся только с --eval-only после выбора чекпойнта.
Новый детектор требует пересборки индекса и дескрипторов, затем сквозного бенчмарка.
"""

import argparse
from pathlib import Path

from tqdm import tqdm

from wine_scanner.catalog import TEST_MANIFEST, load_manifest
from wine_scanner.detect import bottle_detector
from wine_scanner.embed import load_image, pick_device



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=TEST_MANIFEST)
    parser.add_argument("--out", type=Path, help="по умолчанию <корень манифеста>/labels")
    parser.add_argument("--max-side", type=int, default=1600, help="ужать вырезку для разметки")
    parser.add_argument(
        "--sources", default="live,own,eval", help="какие кадры манифеста брать: live, own, eval"
    )
    args = parser.parse_args()

    sources = set(args.sources.split(","))
    args.out = args.out or args.manifest.parent / "labels"
    queries = [q for q in load_manifest(args.manifest, include_multi=False) if q.source in sources]
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
