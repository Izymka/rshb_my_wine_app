"""Проверить вырезку этикетки на эталонах каталога — весь каталог или только новые вина.

    uv run python scripts/check_catalog_crops.py                       # весь каталог
    uv run python scripts/check_catalog_crops.py --slugs slug-1 slug-2 # заведение новых вин
    uv run python scripts/check_catalog_crops.py --write-overrides     # записать автопочинку

Каскад тот же, что режет каталог при сборке индекса (веса — из config.json индекса). На каждом
эталоне `detect.reference_check` ставит флаги; при обрезанной рамке бутылки каскад повторяется
с бутылкой по маске переднего плана. Итог по карточке:

- ``ok`` — замечаний нет;
- ``repaired`` — повтор снял все флаги, квадрат пишется в поправки (`--write-overrides`);
- ``manual`` — квадрат уже задан человеком в `data/catalog/label_overrides.csv`;
- ``accepted`` — человек посмотрел и оставил вырезку каскада (``source=accepted`` в том же файле:
  бутылка без бумажной этикетки, тетрапак, — запрос режется так же, и править один эталон вредно);
- ``review`` — починить автоматически не удалось, нужен квадрат руками (лист `review.jpg`).

Код выхода 1, если остались ``review``: так проверку можно ставить в сценарий заведения вин.
Флаги ``no_bottle``, ``small_label``, ``low_score`` — предупреждения, а не ошибки вырезки.
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

from wine_scanner.catalog import load_platform
from wine_scanner.detect import build_cropper
from wine_scanner.detect.reference_check import (
    OVERRIDES,
    load_overrides,
    review_reference,
    save_overrides,
    sheet,
    tile,
)
from wine_scanner.embed import load_image, pick_device


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--index", type=Path, default=Path("models/index_platform_sq_v3"))
    parser.add_argument("--weights", type=Path, help="веса этикетки; по умолчанию из индекса")
    parser.add_argument("--slugs", nargs="*", help="проверить только эти карточки")
    parser.add_argument("--overrides", type=Path, default=OVERRIDES)
    parser.add_argument("--out", type=Path, default=Path("data/derived/catalog_crop_check"))
    parser.add_argument("--write-overrides", action="store_true")
    parser.add_argument(
        "--bottle-partial",
        type=float,
        help="порог расширения частичной рамки бутылки; по умолчанию из индекса",
    )
    args = parser.parse_args()

    config = json.loads((args.index / "config.json").read_text(encoding="utf-8"))
    weights = args.weights or Path(str(config["weights"]).replace("\\", "/"))
    partial = (
        args.bottle_partial
        if args.bottle_partial is not None
        else config.get("bottle_partial_min_score", 0.0)
    )
    cascade = build_cropper("cascade", weights, pick_device(), partial_min_score=partial).detector
    items = load_platform()
    if args.slugs:
        wanted = set(args.slugs)
        items = [item for item in items if item.item_id in wanted]
        missing = wanted - {item.item_id for item in items}
        if missing:
            raise SystemExit(f"нет в каталоге: {sorted(missing)}")
    overrides = load_overrides(args.overrides)
    args.out.mkdir(parents=True, exist_ok=True)

    report, tiles = [], {"repaired": [], "review": [], "manual": [], "accepted": []}
    for item in tqdm(items, desc="эталоны"):
        image = load_image(item.image_path)
        review = review_reference(cascade, image, overrides.get(item.item_id))
        check, old_rect, new_rect, status = review.check, review.crop, review.rect, review.status
        row = {"slug": item.item_id, "name": item.payload.get("name"), **check.to_dict()}
        if review.retry is not None:
            row["retry"] = review.retry.to_dict()
        row.update(status=status, crop=list(old_rect), fixed=list(new_rect) if new_rect else None)
        report.append(row)
        if status != "ok":
            flags = ",".join(check.flags)
            tiles[status].append(
                tile(image, check, old_rect, new_rect, f"{item.item_id} [{flags}]")
            )
        if status == "repaired":
            overrides.setdefault(
                item.item_id,
                {"rect": new_rect, "source": "auto", "flags": ",".join(check.flags), "note": ""},
            )

    with (args.out / "report.jsonl").open("w", encoding="utf-8") as stream:
        for row in report:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    for status, items_tiles in tiles.items():
        sheet(items_tiles, args.out / f"{status}.jpg")
    counts = {
        s: sum(r["status"] == s for r in report)
        for s in ("ok", "repaired", "manual", "accepted", "review")
    }
    small = sum("small_label" in r["flags"] for r in report)
    print(json.dumps({**counts, "small_label": small, "out": str(args.out)}, ensure_ascii=False))
    for row in report:
        if row["status"] in {"review", "repaired"}:
            print(f"  {row['status']:9} {row['slug']}  [{','.join(row['flags'])}]")
    if args.write_overrides:
        save_overrides(overrides, args.overrides)
        print(f"поправки: {args.overrides} ({len(overrides)})")
    sys.exit(1 if counts["review"] else 0)


if __name__ == "__main__":
    main()
