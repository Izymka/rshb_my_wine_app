"""Проверка отснятых кадров до того, как они попадут в метрики.

    uv run python scripts/check_frames.py
    uv run python scripts/check_frames.py --wine Chateau_Tamagne_красное_2025 --sheet проверка.png

Проверяются физические свойства кадра — то, что можно исправить пересъёмкой: достаточно ли
крупно снята этикетка, не смазан ли кадр, не обрезалось ли по чужой бутылке.

Чего этот скрипт намеренно НЕ делает: не выбраковывает кадры по тому, узнаёт их модель или нет.
Соблазн велик — косинус с каталожной карточкой считается тут же, — но отбор по нему означает
оставить в тесте только то, что система и так умеет. Метрика после такого отбора показывает
не качество системы, а качество отбора, и растёт она сама собой. Косинус используется здесь
ровно для одного: поймать грубую ошибку, когда кадр резкий и качественный, но снято на нём
явно другое вино.

Пороги взяты из уже отснятого набора (109 запросов, 20 вин), а не придуманы:

* короткая сторона кропа: медиана 1064 у каталожных кадров, 995–1083 у angle/glare/blur,
  838 у partial и всего 241 у far;
* резкость (дисперсия лапласиана): 270–405 у нормальных групп против 14 у смазанных;
* косинус со своей карточкой: 0.923 у angle, 0.847 у glare, 0.795 у blur.
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from wine_scanner.burst import sharpness
from wine_scanner.catalog import load_own
from wine_scanner.detect import CachedCropper, build_cropper
from wine_scanner.embed import Dinov2Embedder, load_image, pick_device

CROP_CACHE = Path("models/crop_cache")

# Ниже этого размера этикетке уже не хватает пикселей: OCR перестаёт читать год, а XFeat —
# находить точки. В группе far медиана 241, и это осознанно трудная группа; для easy столько
# быть не должно.
MIN_SIDE_EASY = 600
MIN_SIDE_ANY = 250

# Резкость. У смазанных кадров медиана 14, у остальных 270–405.
MIN_SHARPNESS_EASY = 250
MIN_SHARPNESS_ANY = 40
# Для группы blur смаз — это задание, а не брак, поэтому там проверяется только нижний предел
# осмысленности: кадр, на котором и человек ничего не разберёт, ничего не доказывает.
MIN_SHARPNESS_BLUR = 5

# Доля кадра, которую занимает этикетка после обрезки. У каталожных кадров 0.18, у far 0.015.
MIN_FILL_EASY = 0.10

# Признак грубой ошибки: кадр резкий, крупный — и при этом не похож на свою карточку.
# Смазанный кадр тоже непохож, но там причина честная, поэтому смотрим только на резкие.
WRONG_BOTTLE_COS = 0.65
WRONG_BOTTLE_SHARPNESS = 200


def cached_cascade(device, weights: Path):
    return CachedCropper(build_cropper("cascade", weights, device), CROP_CACHE)


def check(group: str, side: int, sharp: float, fill: float, cos: float | None) -> list[str]:
    """Замечания по кадру. Пустой список — кадр годен."""
    notes = []
    easy = group in {"easy", "catalog"}

    if side < (MIN_SIDE_EASY if easy else MIN_SIDE_ANY):
        notes.append(f"этикетка мелкая: {side} px по короткой стороне")

    if group == "blur":
        if sharp < MIN_SHARPNESS_BLUR:
            notes.append(f"смазано до неразличимости: резкость {sharp:.0f}")
    elif sharp < (MIN_SHARPNESS_EASY if easy else MIN_SHARPNESS_ANY):
        notes.append(f"смаз: резкость {sharp:.0f}")
    if easy and fill < MIN_FILL_EASY:
        notes.append(f"этикетка занимает {fill:.1%} кадра — подойти ближе")
    if cos is not None and sharp > WRONG_BOTTLE_SHARPNESS and cos < WRONG_BOTTLE_COS:
        notes.append(
            f"резкий кадр не похож на свою карточку (косинус {cos:.2f}) — "
            "проверьте глазами, не обрезалось ли по соседней бутылке"
        )
    return notes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wine", default=None, help="проверить одну папку, иначе весь набор")
    parser.add_argument("--weights", type=Path, default=Path("models/label_detector.pt"))
    parser.add_argument("--sheet", type=Path, default=None, help="куда сохранить лист кропов")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--live",
        type=Path,
        default=None,
        help="манифест живых кадров (data/live/manifest.csv) вместо своего набора; "
        "эталон берётся из каталога платформы по true_slug",
    )
    parser.add_argument("--catalog-images", type=Path, default=Path("data/catalog/images"))
    args = parser.parse_args()

    device = pick_device()
    cropper = cached_cascade(device, args.weights)
    embedder = Dinov2Embedder(device=device, cropper=cropper, fit="pad")

    if args.live:
        # Живые кадры платформы: эталон — вырезка каталога; у незнакомцев эталона нет, для них
        # проверяются только физические свойства кадра.
        from wine_scanner.catalog import CatalogItem, load_live

        queries = [q for q in load_live(args.live, include_multi=True) if q.source == "live"]
        slugs = sorted({q.true_id for q in queries if q.known})
        catalog = [
            CatalogItem(slug, args.catalog_images / f"{slug}.png", {})
            for slug in slugs
            if (args.catalog_images / f"{slug}.png").exists()
        ]
    else:
        catalog, queries = load_own()
    if args.wine:
        catalog = [it for it in catalog if it.item_id == args.wine]
        queries = [q for q in queries if q.true_id == args.wine]
        if not catalog:
            raise SystemExit(f"нет такого вина: {args.wine}")

    reference = {
        it.item_id: vector
        for it, vector in zip(
            catalog,
            embedder.encode_paths([it.image_path for it in catalog], batch_size=args.batch_size),
            strict=True,
        )
    }
    query_vectors = embedder.encode_paths([q.path for q in queries], batch_size=args.batch_size)

    rows = [(it.image_path, it.item_id, "catalog", None) for it in catalog]
    rows += [
        (
            q.path,
            q.wine_id or q.true_id,
            q.group,
            float(vector @ reference[q.true_id]) if q.true_id in reference else None,
        )
        for q, vector in zip(queries, query_vectors, strict=True)
    ]

    tiles, problems = [], 0
    current = None
    for path, item_id, group, cos in sorted(rows, key=lambda r: (r[1], r[2])):
        crop = cropper(path, load_image(path))
        full = load_image(path)
        side = min(crop.size)
        sharp = sharpness(crop)
        fill = (crop.width * crop.height) / (full.width * full.height)
        # Косинус к студийной вырезке платформы лежит около 0.4 и у верных кадров — порог
        # «чужая бутылка» снят с телефонных эталонов и здесь не судит, только печатается.
        notes = check(group, side, sharp, fill, None if args.live else cos)

        if item_id != current:
            print(f"\n{item_id}")
            current = item_id
        mark = "  ok " if not notes else "  !! "
        similarity = "     —" if cos is None else f"{cos:>6.2f}"
        print(f"{mark}{group:<9}{path.name:<18}{side:>6} px{sharp:>8.0f}{similarity}")
        for note in notes:
            print(f"       └ {note}")
        problems += bool(notes)
        tiles.append((f"{group} {path.stem}", crop, bool(notes)))

    print(f"\nкадров: {len(rows)}, с замечаниями: {problems}")

    if args.sheet:
        _sheet(tiles, args.sheet)
        print(f"лист кропов: {args.sheet}")


def _sheet(tiles, path: Path, size: int = 190, columns: int = 6) -> None:
    """Лист кропов: смотреть глазами на то, что увидела модель, полезнее любых порогов."""
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (size * columns, (size + 18) * rows), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (caption, image, flagged) in enumerate(tiles):
        x, y = (i % columns) * size, (i // columns) * (size + 18)
        thumb = image.copy()
        thumb.thumbnail((size - 8, size - 8))
        sheet.paste(thumb, (x + (size - thumb.width) // 2, y + 18 + (size - thumb.height) // 2))
        draw.text((x + 4, y + 4), caption[:28], fill="red" if flagged else "black")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


if __name__ == "__main__":
    main()
