"""Завести в каталог и индекс вина, появившиеся на сайте после дампа, — с проверкой эталона.

    uv run python scripts/add_site_wines.py --index models/index_platform_sq_v3_fix \\
        --out models/index_platform_sq_v3_site                  # все новые вина из sitemap
    uv run python scripts/add_site_wines.py --index ... --slugs gevyurcztraminer-oranzh --dry-run
    uv run python scripts/add_site_wines.py --index ... --out ... --update-images

Что новое — решает свежий sitemap сайта (`wines-sitemap.xml`) против `data/catalog/catalog.csv`.
Второй список — **сменившиеся эталоны**: slug в каталоге есть, а файл картинки в sitemap другой
(винодельня сменила этикетку). По умолчанию они только печатаются, `--update-images` — заменить.

Каждое вино проходит четыре шага, и в индекс попадает только прошедшее все:

1. **Карточка** со страницы вина (`__NUXT_DATA__`, `wine_scanner.site.site_wine`) и эталон —
   оригинал с сайта, композитный на белом, как у `build_catalog.py`.
2. **Вырезка этикетки** — `review_reference`, та же проверка, что `check_catalog_crops.py`:
   каскад индекса против маски переднего плана. `ok` и `repaired` (квадрат по маске, пишется в
   `label_overrides.csv`) проходят; `review` — лист `review.jpg`, человек ставит квадрат руками
   (`source=manual`) или оставляет каскад (`source=accepted`) и запускает скрипт заново.
3. **Вектор**: воспроизводимость индекса (пересчёт трёх старых карточек, cos ≥ 0.995), затем
   ближайший сосед нового вектора после whitening. Почти совпадение (cos ≥ `--dup-cos`) с
   карточкой, у которой другая картинка, — подозрение на дубль или хаб, на ручной разбор.
4. **Самопоиск**: `--probes` псевдофото из эталона (бутылка крупно на размытом до пятен фоне,
   наклон, перспектива, блик, смаз, JPEG — `wine_scanner.synthetic`, `quiet=True`) через весь
   `WineScanner` на временном индексе с новыми карточками. Вино должно быть top-1 хотя бы в
   `--min-hits` из них. Это проверка всей цепочки: вырезки запроса, вектора, дескрипторов XFeat
   и текстовой ветки по новой карточке. Фон размыт намеренно: на обычной полке каскад иногда
   режет фоновую бутылку, и проба проверяла бы выбор бутылки, а не карточку.

Итоговый индекс собирается из исходного заново и только с прошедшими винами, в `--out`
(исходный не трогается: сервис читает `models/`). Строки `catalog.csv`, эталоны в
`data/catalog/originals/` и поправки вырезки пишутся последним шагом и только для прошедших.
`--dry-run` проходит все проверки и ничего не пишет. Код выхода 1, если что-то не прошло.
Отчёт — `data/derived/site_add/report.jsonl`.
"""

import argparse
import json
import os
import random
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from wine_scanner.catalog import PLATFORM_ROOT, vintage_from_name
from wine_scanner.detect.reference_check import (
    OVERRIDES,
    load_overrides,
    review_reference,
    save_overrides,
    sheet,
    tile,
)
from wine_scanner.embed import Whitening, load_image
from wine_scanner.image_preprocess import crop_square, prepare_label_image
from wine_scanner.index.patch import IndexPatch
from wine_scanner.site import (
    SITE_ORIGINAL,
    SITE_WINE_URL,
    digest,
    fetch,
    flatten,
    load_sitemap,
    site_wine,
)
from wine_scanner.synthetic import cutout, shelf_paths, synthesize

# Колонки catalog.csv, которые несёт payload карточки в индексе (как `load_platform`).
PAYLOAD_FIELDS = ["name", "category", "color", "region", "grapes", "description", "winery"]


def candidates(catalog: pd.DataFrame, sitemap: pd.DataFrame, slugs: list[str] | None):
    """(новые вина, вина со сменившимся файлом эталона) — slug → имя файла на сайте."""
    on_site = dict(zip(sitemap["slug"], sitemap["file"], strict=True))
    known = dict(zip(catalog["slug"], catalog["source_file"].fillna(""), strict=True))
    new = {s: f for s, f in on_site.items() if s not in known}
    # В каталоге source_file — путь в uploads или downloaded; имя файла сайта совпадает с его
    # последней частью, если эталон не менялся.
    changed = {s: f for s, f in on_site.items() if s in known and Path(known[s]).name != f}
    if slugs:
        wanted = set(slugs)
        missing = wanted - set(on_site)
        if missing:
            raise SystemExit(f"нет в sitemap: {sorted(missing)}")
        new = {s: f for s, f in new.items() if s in wanted}
        changed = {s: f for s, f in changed.items() if s in wanted}
    return new, changed


def fetch_reference(slug: str, file: str, downloads: Path, staging: Path) -> tuple[Path, Path]:
    """Оригинал эталона с сайта (кэш в downloads) и его композит на белом в staging."""
    source = downloads / file
    if not source.exists():
        payload = fetch(SITE_ORIGINAL.format(file=file))
        if not payload:
            raise RuntimeError(f"эталон не скачался: {file}")
        downloads.mkdir(parents=True, exist_ok=True)
        source.write_bytes(payload)
    staging.mkdir(parents=True, exist_ok=True)
    flat = staging / f"{slug}.png"
    flatten(source, flat)
    return source, flat


def nearest(patch: IndexPatch, whitening: Whitening | None, vector: np.ndarray):
    """Ближайшая карточка индекса к вектору — в том пространстве, где ищет сервис."""
    base = patch.vectors
    query = vector[None, :]
    if whitening is not None:
        base, query = whitening.apply(base), whitening.apply(query)
    scores = base @ query[0]
    n = int(np.argmax(scores))
    return patch.meta["item_ids"][n], float(scores[n])


def self_search(scanner, original: Path, slug: str, probes: int, seed: int) -> list[int]:
    """Ранг вина среди кандидатов на каждом псевдофото (0 — top-1, -1 — не найдено)."""
    rng = random.Random(f"{seed}:{slug}")
    shelves = shelf_paths()
    bottle = cutout(original)
    ranks = []
    for _ in range(probes):
        image = synthesize(bottle, load_image(rng.choice(shelves)), rng, quiet=True)
        result = scanner.identify(image, use_judge=False)
        ids = [c.item_id for c in result.candidates]
        ranks.append(ids.index(slug) if slug in ids else -1)
    return ranks


def catalog_row(card: dict, source: Path, flat: Path, md5: str, shared: bool) -> dict:
    with Image.open(source) as original:
        width, height, mode = *original.size, original.mode
    return {
        **{k: card[k] for k in PAYLOAD_FIELDS},
        "slug": card["slug"],
        "photo_name": card["photo_name"],
        "match": "site",
        "source_file": str(source),
        "on_site": True,
        "vintage": vintage_from_name(card["name"], card["slug"]),
        "shared_photo": False,
        "image_md5": md5,
        "shared_image": shared,
        "width": width,
        "height": height,
        "mode": mode,
    }


def payload_of(row: dict, image_path: Path) -> dict:
    return {
        **{k: row[k] for k in PAYLOAD_FIELDS},
        "slug": row["slug"],
        "vintage": int(row["vintage"]) if pd.notna(row["vintage"]) else None,
        "image_path": str(image_path),
    }


def apply(patch: IndexPatch, wines: dict[str, dict], originals_root: Path | None) -> None:
    """Дописать новые и заменить обновлённые карточки в открытом индексе."""
    for slug, wine in wines.items():
        if wine["update"]:
            patch.replace(slug, wine["crop"])
        else:
            image_path = (originals_root / f"{slug}.png") if originals_root else wine["flat"]
            patch.add(slug, payload_of(wine["row"], image_path), wine["crop"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--index", type=Path, required=True, help="исходный индекс")
    parser.add_argument("--out", type=Path, help="новый индекс (не нужен с --dry-run)")
    parser.add_argument(
        "--decider",
        type=Path,
        default=os.environ.get("WINE_DECIDER"),
        help="решающий слой для самопоиска; по умолчанию WINE_DECIDER, как у сервиса",
    )
    parser.add_argument("--slugs", nargs="*", help="только эти вина")
    parser.add_argument("--update-images", action="store_true", help="заменить сменившиеся эталоны")
    parser.add_argument("--catalog", type=Path, default=PLATFORM_ROOT)
    parser.add_argument("--overrides", type=Path, default=OVERRIDES)
    parser.add_argument("--no-refresh", action="store_true", help="взять sitemap из кэша")
    parser.add_argument("--report", type=Path, default=Path("data/derived/site_add"))
    parser.add_argument("--dup-cos", type=float, default=0.98)
    parser.add_argument("--probes", type=int, default=3)
    parser.add_argument("--min-hits", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and args.out is None:
        parser.error("--out обязателен без --dry-run")
    if args.probes and args.decider is None:
        parser.error("нужен --decider (или WINE_DECIDER) — тот же, что у сервиса")
    if args.out is not None and args.out.exists():
        parser.error(f"{args.out} уже существует")

    catalog_csv = args.catalog / "catalog.csv"
    catalog = pd.read_csv(catalog_csv, dtype={"vintage": "Int64"})
    sitemap = load_sitemap(args.catalog / "wines-sitemap.xml", refresh=not args.no_refresh)
    if sitemap.empty:
        raise SystemExit("sitemap недоступен")
    new, changed = candidates(catalog, sitemap, args.slugs)
    print(f"в sitemap {len(sitemap)}, новых {len(new)}, со сменившимся эталоном {len(changed)}")
    for slug in sorted(changed):
        print(f"  эталон сменился: {slug}" + ("" if args.update_images else " (--update-images)"))
    todo = {**new, **(changed if args.update_images else {})}
    if not todo:
        print("добавлять нечего")
        return

    args.report.mkdir(parents=True, exist_ok=True)
    staging = args.report / "staging"
    patch = IndexPatch(args.index)
    print(f"воспроизводимость индекса: min cos {patch.verify(3):.4f}")
    whitening_path = args.index / "whitening.npz"
    whitening = Whitening.load(whitening_path) if whitening_path.exists() else None
    md5_to_slugs = catalog.groupby("image_md5")["slug"].apply(list).to_dict()
    overrides = load_overrides(args.overrides)
    cascade = patch.cascade.detector

    report: dict[str, dict] = {}
    wines: dict[str, dict] = {}
    review_tiles = []
    for slug, file in sorted(todo.items()):
        entry = report.setdefault(slug, {"slug": slug, "update": slug in changed, "file": file})
        html = fetch(SITE_WINE_URL.format(slug=slug))
        card = site_wine(html.decode("utf-8", "replace"), slug) if html else None
        if card is None or not card["name"]:
            entry["status"] = "no_card"
            continue
        entry["name"], entry["winery"] = card["name"], card["winery"]
        try:
            source, flat = fetch_reference(slug, file, args.catalog / "downloaded", staging)
        except RuntimeError as error:
            entry["status"] = "no_image"
            entry["error"] = str(error)
            continue
        md5 = digest(source)
        twins = [s for s in md5_to_slugs.get(md5, []) if s != slug]
        entry["shared_image_with"] = twins

        # Шаг 2: вырезка этикетки.
        image = load_image(flat)
        # Поправка вырезки относится к картинке: у сменившегося эталона старая не годится.
        decision = None if slug in changed else overrides.get(slug)
        review = review_reference(cascade, image, decision)
        entry["crop"] = {"status": review.status, "flags": review.check.flags}
        if not review.passed:
            review_tiles.append(
                tile(
                    image,
                    review.check,
                    review.crop,
                    None,
                    f"{slug} [{','.join(review.check.flags)}]",
                )
            )
            entry["status"] = "review_crop"
            continue
        crop = (
            prepare_label_image(crop_square(image, review.rect))
            if review.rect
            else patch.cascade.crop(image)
        )

        # Шаг 3: вектор и его сосед. Соседи с той же картинкой — честный дубль карточки, не хаб.
        vector = patch.encode(crop)
        neighbour, cosine = nearest(patch, whitening, vector)
        entry["nearest"] = {"slug": neighbour, "cos": round(cosine, 4)}
        if cosine >= args.dup_cos and neighbour not in twins and neighbour != slug:
            entry["status"] = "near_duplicate"
            continue

        row = catalog_row(card, source, flat, md5, shared=bool(twins))
        wines[slug] = {
            "row": row,
            "crop": crop,
            "flat": flat,
            "source": source,
            "update": slug in changed,
            "override": {
                "rect": review.rect,
                "source": "auto",
                "flags": ",".join(review.check.flags),
                "note": "add_site_wines",
            }
            if review.status == "repaired"
            else None,
        }

    # Шаг 4: самопоиск на временном индексе со всеми кандидатами.
    if wines and args.probes:
        from wine_scanner.pipeline import WineScanner  # noqa: PLC0415 — тяжёлый

        temp = args.report / "index_staging"
        if temp.exists():
            shutil.rmtree(temp)
        trial = IndexPatch(args.index, device=patch.device)
        trial.open_target(temp)
        apply(trial, wines, originals_root=None)
        trial.save("added_from_site", {"staging": True})
        scanner = WineScanner(index_dir=temp, decider_dir=args.decider)
        for slug, wine in list(wines.items()):
            ranks = self_search(scanner, wine["source"], slug, args.probes, args.seed)
            report[slug]["self_search_ranks"] = ranks
            if sum(r == 0 for r in ranks) < args.min_hits:
                report[slug]["status"] = "self_search"
                del wines[slug]
        shutil.rmtree(temp)

    for slug in wines:
        report[slug]["status"] = "added" if not args.dry_run else "would_add"

    with (args.report / "report.jsonl").open("w", encoding="utf-8") as stream:
        for entry in report.values():
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
    if review_tiles:
        sheet(review_tiles, args.report / "review.jpg")
    for entry in report.values():
        extra = entry.get("nearest", {})
        print(
            f"  {entry['status']:15} {entry['slug']}  {entry.get('name', '')}"
            f"  сосед {extra.get('slug', '-')} {extra.get('cos', '')}"
            f"  самопоиск {entry.get('self_search_ranks', '-')}"
        )

    if wines and not args.dry_run:
        originals = args.catalog / "originals"
        final = IndexPatch(args.index, device=patch.device)
        final.open_target(args.out)
        apply(final, wines, originals_root=originals)
        final.save(
            "added_from_site",
            {
                "date": date.today().isoformat(),
                "added": sorted(s for s, w in wines.items() if not w["update"]),
                "updated": sorted(s for s, w in wines.items() if w["update"]),
                "patched_from": str(args.index),
            },
        )
        for slug, wine in wines.items():
            shutil.copy2(wine["flat"], originals / f"{slug}.png")
            if wine["override"]:
                overrides[slug] = wine["override"]
            elif wine["update"]:
                overrides.pop(slug, None)
        rows = pd.DataFrame([w["row"] for w in wines.values()])
        updated = catalog[~catalog["slug"].isin(rows["slug"])]
        merged = pd.concat([updated, rows[catalog.columns]], ignore_index=True)
        # Новая картинка может совпасть байт в байт с уже имеющейся — флаг у обеих сторон.
        merged["shared_image"] = (
            merged["image_md5"].duplicated(keep=False) & merged["image_md5"].notna()
        )
        merged.to_csv(catalog_csv, index=False)
        if any(w["override"] or w["update"] for w in wines.values()):
            save_overrides(overrides, args.overrides)
        print(f"индекс: {args.out}; в каталоге {len(merged)} строк")
    shutil.rmtree(staging, ignore_errors=True)
    sys.exit(0 if len(wines) == len(todo) else 1)


if __name__ == "__main__":
    main()
