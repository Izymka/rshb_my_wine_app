"""Пересчитать в индексе карточки с поправленной вырезкой этикетки — без пересборки каталога.

    uv run python scripts/patch_index_crops.py --index models/index_platform_sq_v3 \\
        --out models/index_platform_sq_v3_fix
    uv run python scripts/patch_index_crops.py --index models/index_platform_sq_v3_fix \\
        --out models/index_platform_sq_v3_partial --recrop 0.15

Два источника новой вырезки:

- поправки `data/catalog/label_overrides.csv` (их пишет `check_catalog_crops.py` или человек) —
  всегда;
- `--recrop P` — каскад с новым порогом расширения частичной рамки бутылки
  (`BoxCropper.partial_min_score`). Детектор бутылки проходит по каждому эталону один раз, рамка
  выбирается по старому и новому правилу, и только у карточек, где она сменилась, заново
  ищется этикетка и пересчитывается карточка. Порог пишется в `config.json` нового индекса —
  запрос в сервисе режется так же.

Для каждой такой карточки заново считаются вектор SigLIP 2, локальные признаки XFeat и бокс
года (`wine_scanner.index.patch`); остальные копируются как есть. `--verify N` (по умолчанию 3)
сначала пересчитывает N карточек без поправок штатным каскадом и сравнивает с сохранёнными
векторами; косинус ниже 0.995 — остановка (другая точность, другое устройство, другие веса).
"""

import argparse
import hashlib
from pathlib import Path

from tqdm import tqdm

from wine_scanner.detect import build_cropper
from wine_scanner.detect.audit import iou
from wine_scanner.detect.reference_check import ACCEPTED, OVERRIDES, load_overrides
from wine_scanner.image_preprocess import crop_square, prepare_label_image
from wine_scanner.index.patch import IndexPatch

# Квадрат этикетки сдвинулся меньше — карточку не трогаем: вектор почти тот же.
SAME_SQUARE_IOU = 0.98


def recrop(patch: IndexPatch, partial: float, skip: set[str]) -> dict:
    """slug → новая вырезка для карточек, у которых новое правило меняет рамку бутылки."""
    old_partial = patch.config.get("bottle_partial_min_score", 0.0)
    new = build_cropper(
        patch.config.get("detect", "cascade"),
        patch.weights,
        patch.device,
        partial_min_score=partial,
    )
    cascade = new.detector
    bottle = cascade.bottle
    changed = {}
    for item_id in tqdm(patch.meta["item_ids"], desc="рамки бутылки"):
        if item_id in skip:
            continue
        image = patch.image_of(item_id)
        boxes = bottle.detect(image)
        picked = {}
        for value in (old_partial, partial):
            bottle.partial_min_score = value
            box = bottle.pick(boxes, image.size)
            picked[value] = (box.x1, box.y1, box.x2, box.y2) if box else None
        bottle.partial_min_score = partial
        if picked[old_partial] == picked[partial]:
            continue
        bottle.partial_min_score = old_partial
        old_square = cascade.crop_with_metadata(image)[2]
        bottle.partial_min_score = partial
        raw, _, new_square = cascade.crop_with_metadata(image)
        if old_square and new_square and iou(old_square, new_square) > SAME_SQUARE_IOU:
            continue
        changed[item_id] = prepare_label_image(raw)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overrides", type=Path, default=OVERRIDES)
    parser.add_argument("--slugs", nargs="*", help="только эти карточки из поправок")
    parser.add_argument("--recrop", type=float, help="новый порог расширения рамки бутылки")
    parser.add_argument("--verify", type=int, default=3)
    parser.add_argument("--rerank-max-side", type=int, default=640)
    parser.add_argument("--rerank-points", type=int, default=2048)
    args = parser.parse_args()

    patch = IndexPatch(
        args.index, rerank_max_side=args.rerank_max_side, rerank_points=args.rerank_points
    )
    overrides = {
        slug: row
        for slug, row in load_overrides(args.overrides).items()
        if row.get("source") != ACCEPTED and slug in patch.position
    }
    if args.slugs:
        overrides = {slug: overrides[slug] for slug in args.slugs}

    if args.verify:
        worst = patch.verify(args.verify, skip=set(overrides))
        print(f"проверка воспроизводимости на {args.verify} карточках: min cos {worst:.4f}")

    crops = {
        slug: prepare_label_image(crop_square(patch.image_of(slug), row["rect"]))
        for slug, row in overrides.items()
    }
    recropped = []
    if args.recrop is not None:
        extra = recrop(patch, args.recrop, skip=set(overrides))
        recropped = sorted(extra)
        crops.update(extra)
        print(f"новое правило меняет вырезку у {len(extra)} карточек")
    if not crops:
        raise SystemExit("пересчитывать нечего")

    patch.open_target(args.out)
    for item_id, crop in tqdm(crops.items(), desc="пересчёт"):
        print(f"  {item_id}: cos(старый, новый) {patch.replace(item_id, crop):.3f}")
    if args.recrop is not None:
        patch.config["bottle_partial_min_score"] = args.recrop
    with args.overrides.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    patch.save(
        "label_overrides",
        {
            "file": str(args.overrides),
            "sha256": digest,
            "slugs": sorted(overrides),
            "recrop": {"bottle_partial_min_score": args.recrop, "slugs": recropped}
            if args.recrop is not None
            else None,
            "patched_from": str(args.index),
        },
    )
    print(f"индекс: {args.out}, пересчитано карточек {len(crops)}")


if __name__ == "__main__":
    main()
