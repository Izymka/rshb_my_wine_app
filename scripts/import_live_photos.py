"""Новые съёмки — из `data/incoming/` в `data/test` или `data/train` с дописыванием манифеста.

    uv run python scripts/import_live_photos.py                 # всё из data/incoming
    uv run python scripts/import_live_photos.py --dest test     # незнакомцев тоже в тест

Раскладка с 18.09.2026 (до этого все кадры лежали в одном `data/live`, он ушёл в `data/archive`):

    data/test/            изолированный тест — только замер (catalog.is_holdout)
      manifest.csv        path, wine_id, true_slug, group, source, note
      <slug>/*.jpg        вино из каталога, все его кадры (кадры полок с несколькими бутылками —
                          те же папки, group=multi)
      unknown-*/*.jpg     незнакомец с российской полки
      frames_annotated/   кадры 1600 px под детектор + _annotations.csv (makesense)
      frames_unannotated/ кадры 1600 px, которые ещё предстоит разметить
      labels/             вырезки бутылок с разметкой этикетки (старый каскад, Faster R-CNN)
    data/train/           обучающий материал: свой набор data/own (по манифесту, файлы на месте),
      ...                 незнакомцы, не вошедшие в тест; те же frames_*/labels

Единица разметки — кадр, правда лежит в манифесте. `true_slug` пустой означает «этого вина в
каталоге нет» — на таких кадрах проверяется отказ. `group` для незнакомых — `unknown-hard`
(российская полка: кириллица, в каталоге есть близнец или та же винодельня) или `unknown-easy`
(импорт, латиница); `multi` — несколько бутылок серии в кадре, в метрики по умолчанию не идёт.

**Новые съёмки** кладутся папками в `data/incoming/` — по папке на вино, без таблицы в коде:

    data/incoming/<slug из каталога>/IMG_0310.HEIC ...   известное вино → data/test/<slug>, группа live
    data/incoming/unknown-<что это>/IMG_0340.HEIC ...    незнакомец с российской полки (unknown-hard)
    data/incoming/import-<что это>/IMG_0350.HEIC ...     импорт, латиница (unknown-easy)

Имя папки известного вина — ровно slug из `data/catalog/catalog.csv`; опечатка в slug — ошибка
при импорте, а не молчаливый незнакомец. **Известное вино всегда идёт в тест** (тест — все вина,
которые есть в каталоге), незнакомец — в `--dest` (по умолчанию `train`: тест зафиксирован
50 незнакомцами, и раздувать его без нужды не стоит). Кадр с несколькими бутылками — в
подпапку `multi/`. HEIC переписывается в JPEG полного разрешения: скрипт оценки шлёт
JPEG/WebP, и мерить надо ровно то, что уйдёт по сети; MP4 от Live Photo не трогаются.
Папка после импорта переезжает в `data/archive/incoming/`, чтобы не импортироваться дважды;
манифест дописывается, уже известные пути пропускаются.

Кадры под детектор для новых съёмок делает `scripts/prepare_frame_annotation.py --manifest
data/<test|train>/manifest.csv` — он кладёт в `frames_unannotated` только то, чего ещё нет в
`frames_annotated`.
"""

import argparse
import csv
import shutil
from pathlib import Path

from wine_scanner.catalog import PLATFORM_ROOT, TEST_ROOT, TRAIN_ROOT
from wine_scanner.embed import load_image

INCOMING = Path("data/incoming")
ARCHIVE = Path("data/archive/incoming")
CATALOG_CSV = PLATFORM_ROOT / "catalog.csv"
RAW_SUFFIXES = {".heic", ".heif", ".jpg", ".jpeg", ".png", ".webp"}
FIELDS = ("path", "wine_id", "true_slug", "group", "source", "note")
ROOTS = {"test": TEST_ROOT, "train": TRAIN_ROOT}


def read_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def import_folder(folder: Path, slugs: set[str], unknown_dest: str, force: bool) -> tuple[Path, list[dict]]:
    """Папка одного вина из data/incoming → (корень назначения, строки манифеста)."""
    wine_id = folder.name
    if wine_id.startswith("unknown-"):
        slug, group, root = "", "unknown-hard", ROOTS[unknown_dest]
    elif wine_id.startswith("import-"):
        slug, group, root = "", "unknown-easy", ROOTS[unknown_dest]
    elif wine_id in slugs:
        slug, group, root = wine_id, "live", TEST_ROOT
    else:
        raise SystemExit(
            f"папка {folder}: это не slug каталога и не unknown-/import- — проверьте имя "
            f"по data/catalog/catalog.csv"
        )
    files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in RAW_SUFFIXES)
    if not files:
        print(f"пусто: {folder}")
    rows = []
    for path in files:
        multi = path.parent.name == "multi"
        out = root / wine_id / f"{path.stem.replace(' ', '_')}.jpg"
        if force or not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            load_image(path).save(out, quality=92)
        rows.append(
            {
                "path": str(out),
                "wine_id": wine_id,
                "true_slug": slug,
                "group": "multi" if multi else group,
                "source": "live",
                "note": "несколько бутылок" if multi else "",
            }
        )
    return root, rows


def report(root: Path) -> None:
    rows = read_manifest(root / "manifest.csv")
    known = [r for r in rows if r["true_slug"] and r["group"] != "multi"]
    print(f"{root}/manifest.csv: кадров {len(rows)}")
    print(f"  известных: {len(known)} ({len({r['true_slug'] for r in known})} вин)")
    for group in ("unknown-hard", "unknown-easy", "multi"):
        part = [r for r in rows if r["group"] == group]
        print(f"  {group}: {len(part)} ({len({r['wine_id'] for r in part})} вин)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incoming", type=Path, default=INCOMING, help="новые съёмки папками по винам")
    parser.add_argument("--catalog", type=Path, default=CATALOG_CSV)
    parser.add_argument(
        "--dest", choices=("test", "train"), default="train",
        help="куда класть незнакомцев (известные вина всегда в test)",
    )
    parser.add_argument("--force", action="store_true", help="перезаписать уже сконвертированные JPEG")
    parser.add_argument("--keep", action="store_true", help="не переносить папки в data/archive/incoming")
    args = parser.parse_args()

    with args.catalog.open(encoding="utf-8", newline="") as f:
        slugs = {row["slug"] for row in csv.DictReader(f)}
    folders = sorted(p for p in args.incoming.iterdir() if p.is_dir() and not p.name.startswith("."))
    if not folders:
        print(f"в {args.incoming} нет папок — нечего импортировать")
        for root in ROOTS.values():
            report(root)
        return

    added: dict[Path, list[dict]] = {root: [] for root in ROOTS.values()}
    for folder in folders:
        root, rows = import_folder(folder, slugs, args.dest, args.force)
        added[root] += rows
        if not args.keep:
            ARCHIVE.mkdir(parents=True, exist_ok=True)
            shutil.move(str(folder), str(ARCHIVE / folder.name))

    for root, rows in added.items():
        manifest = root / "manifest.csv"
        existing = read_manifest(manifest)
        seen = {r["path"] for r in existing}
        new = [r for r in rows if r["path"] not in seen]
        write_manifest(manifest, existing + new)
        print(f"{manifest}: +{len(new)} кадров ({len(rows) - len(new)} уже были)")
        report(root)


if __name__ == "__main__":
    main()
